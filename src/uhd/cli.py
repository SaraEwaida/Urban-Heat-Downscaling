"""Command line: uhd fetch | align | preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="uhd", description="Urban heat downscaling pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="find a Landsat/MODIS pair and write the scene manifest")
    f.add_argument("config")
    f.add_argument("--date", help="pick the pair on this date (YYYY-MM-DD)")
    f.add_argument("--list", action="store_true", help="only list candidate pairs")

    a = sub.add_parser("align", help="align all layers to the 30 m grid and save Zarr")
    a.add_argument("config")

    v = sub.add_parser("preview", help="save the input vs reference figure")
    v.add_argument("config")

    aa = sub.add_parser("align-all", help="align every configured date onto one grid -> stack Zarr")
    aa.add_argument("config")

    b = sub.add_parser("baseline", help="bilinear baseline on the fixed splits -> results CSV")
    b.add_argument("config")

    lg = sub.add_parser("lightgbm", help="train LightGBM on the train split, score on every split")
    lg.add_argument("config")

    un = sub.add_parser("unet", help="train the U-Net on training-area patches, score on every split")
    un.add_argument("config")

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "fetch":
        from .fetch import fetch

        fetch(cfg, date=args.date, list_only=args.list)

    elif args.cmd == "align":
        from .align import align, save

        manifest = json.loads(Path(cfg["paths"]["scene"]).read_text())
        ds = align(cfg, manifest)
        path = save(ds, cfg["paths"]["zarr"])
        print(f"Grid {ds.sizes['y']}x{ds.sizes['x']} @30 m, coarse {ds.sizes['yc']}x{ds.sizes['xc']} "
              f"@{30 * ds.attrs['coarse_factor']} m ({ds.attrs['crs']})")
        print(f"Valid: reference {ds.attrs['valid_fraction_ref']:.1%}, coarse {ds.attrs['valid_fraction_coarse']:.1%}")
        print(f"Saved {path}")

    elif args.cmd == "align-all":
        from .align import save
        from .stack import align_all

        ds = align_all(cfg)
        path = save(ds, cfg["paths"]["stack"])
        for d, sc in ds.attrs["scenes"].items():
            print(f"{d}: valid reference {sc['valid_fraction_ref']:.1%}, coarse {sc['valid_fraction_coarse']:.1%}")
        print(f"Saved {ds.sizes['time']} dates, grid {ds.sizes['y']}x{ds.sizes['x']} -> {path}")

    elif args.cmd == "baseline":
        import xarray as xr

        from .baseline import predict_bilinear
        from .metrics import evaluate, format_table, write_results
        from .splits import describe, save_description, split_masks

        ds = xr.open_zarr(cfg["paths"]["stack"]).load()
        masks = split_masks(ds, cfg)
        save_description(describe(ds, cfg, masks), cfg["paths"]["splits"])
        pred = predict_bilinear(ds.lst_coarse.values, int(ds.attrs["coarse_factor"]))
        rows = evaluate("bilinear", pred, ds.lst_ref.values, ds.landcover.values, masks)
        out = write_results(rows, cfg["paths"]["results"])
        print(format_table(rows))
        print(f"Saved {out} and {cfg['paths']['splits']}")

    elif args.cmd == "lightgbm":
        import csv

        import xarray as xr

        from .features import FeatureBuilder
        from .lgbm import importance, predict, predict_domain, save_predictions, train
        from .metrics import evaluate, format_table, write_results
        from .splits import eval_mask, split_masks

        ds = xr.open_zarr(cfg["paths"]["stack"]).load()
        masks = split_masks(ds, cfg)
        fb = FeatureBuilder(ds)
        booster, info = train(ds, cfg, masks, fb)
        print(f"Fitted on {info['fit_rows']:,} pixels, early-stopped on {info['val_rows']:,} "
              f"validation pixels at {info['best_iteration']} trees")

        print("Predicting every date...", flush=True)
        pred = predict(ds, booster, predict_domain(ds) | eval_mask(ds), fb, verbose=True)
        rows = evaluate("lightgbm", pred, ds.lst_ref.values, ds.landcover.values, masks)
        out = write_results(rows, cfg["paths"]["results"])

        model_path = Path(cfg["paths"]["model_lightgbm"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_path))
        save_predictions(ds, pred, "lightgbm", Path(cfg["paths"]["predictions_dir"]) / "lightgbm.zarr")
        imp = importance(booster)
        imp_path = Path(cfg["paths"]["importance_lightgbm"])
        with imp_path.open("w", newline="") as f:
            csv.writer(f).writerows([("feature", "gain_share")] + [(k, f"{v:.4f}") for k, v in imp])

        _print_comparison(out)
        print("Top features (gain share): " + ", ".join(f"{k} {v:.0%}" for k, v in imp[:5]))
        print(f"Saved {out}, {model_path}, {imp_path}")

    elif args.cmd == "unet":
        import csv

        import xarray as xr

        from . import unet
        from .lgbm import save_predictions
        from .metrics import evaluate, write_results
        from .splits import split_masks

        ds = xr.open_zarr(cfg["paths"]["stack"]).load()
        masks = split_masks(ds, cfg)
        print("Preparing inputs (normalisation fitted on training pixels)...", flush=True)
        model, prep, info = unet.train(ds, cfg, masks)
        print(f"Trained on {info['fit_patches']} patches ({info['device']}), best epoch {info['best_epoch']} "
              f"of {info['epochs_run']}, validated on {info['val_patches']} patches")

        pred = unet.predict(model, prep)
        rows = evaluate("unet", pred, ds.lst_ref.values, ds.landcover.values, masks)
        out = write_results(rows, cfg["paths"]["results"])
        model_path = unet.save(model, prep, info, cfg, cfg["paths"]["model_unet"])
        save_predictions(ds, pred, "unet", Path(cfg["paths"]["predictions_dir"]) / "unet.zarr")
        hist_path = Path(cfg["paths"]["history_unet"])
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with hist_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_rmse_c"])
            w.writeheader()
            w.writerows(info["history"])
        _print_comparison(out)
        print(f"Saved {out}, {model_path}, {hist_path}")

    elif args.cmd == "preview":
        import xarray as xr

        from .preview import plot_input_vs_reference

        ds = xr.open_zarr(cfg["paths"]["zarr"]).load()
        out = plot_input_vs_reference(ds, cfg["paths"]["figure"])
        print(f"Saved {out}")


def _print_comparison(results_path) -> None:
    import csv

    from .metrics import format_table

    with open(results_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["landcover"] == "all"]
    for r in rows:
        r["n"] = int(r["n"])
        for k in ("mae_c", "rmse_c", "bias_c"):
            r[k] = float(r[k])
    order = ["bilinear", "lightgbm", "unet"]
    rows.sort(key=lambda r: (["train", "test_area", "test_dates", "test_both"].index(r["split"]),
                             order.index(r["method"]) if r["method"] in order else 99))
    print(format_table(rows))


if __name__ == "__main__":
    main()
