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

    elif args.cmd == "preview":
        import xarray as xr

        from .preview import plot_input_vs_reference

        ds = xr.open_zarr(cfg["paths"]["zarr"]).load()
        out = plot_input_vs_reference(ds, cfg["paths"]["figure"])
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
