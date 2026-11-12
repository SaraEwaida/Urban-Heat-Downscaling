"""LightGBM pixel model.

Target: the Landsat temperature minus the bilinear surface (a residual). Trees cannot
extrapolate, so learning a correction to the bilinear field is more robust on warmer or
cooler dates than learning absolute temperatures.

Data use:
- fit on the `train` split without the validation date,
- early stopping on the validation date (training area only),
- the test splits are only predicted, never used for fitting or stopping.
LightGBM needs no scaling, so there is no other preprocessing to fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .features import CATEGORICAL, FEATURES, FeatureBuilder


def _date_flags(ds: xr.Dataset, dates: list[str]) -> np.ndarray:
    names = [str(t)[:10] for t in ds.time.values.astype("datetime64[D]")]
    missing = set(dates) - set(names)
    if missing:
        raise ValueError(f"dates not in the stack: {sorted(missing)}")
    return np.array([n in dates for n in names])


def fit_masks(ds: xr.Dataset, cfg: dict[str, Any], masks: dict[str, np.ndarray]):
    """(fit, val) masks, both inside the train split."""
    val_dates = list(cfg["lightgbm"]["val_dates"])
    test_dates = set(cfg["splits"]["test_dates"])
    if test_dates & set(val_dates):
        raise ValueError("val_dates must not overlap test_dates")
    v = _date_flags(ds, val_dates)[:, None, None]
    return masks["train"] & ~v, masks["train"] & v


def train(ds: xr.Dataset, cfg: dict[str, Any], masks: dict[str, np.ndarray], fb: FeatureBuilder | None = None):
    import lightgbm as lgb

    p = cfg["lightgbm"]
    fb = fb or FeatureBuilder(ds)
    fit_m, val_m = fit_masks(ds, cfg, masks)
    obs = ds.lst_ref.values

    X, base, idx = fb.matrix(fit_m)
    y = obs[idx[:, 0], idx[:, 1], idx[:, 2]] - base
    rng = np.random.default_rng(p.get("seed", 0))
    if len(y) > p["max_train_rows"]:
        keep = rng.choice(len(y), p["max_train_rows"], replace=False)
        X, y = X[keep], y[keep]
    Xv, bv, iv = fb.matrix(val_m)
    yv = obs[iv[:, 0], iv[:, 1], iv[:, 2]] - bv

    params = {
        "objective": "regression_l2",
        "learning_rate": p["learning_rate"],
        "num_leaves": p["num_leaves"],
        "min_child_samples": p["min_child_samples"],
        "feature_fraction": p["feature_fraction"],
        "bagging_fraction": p["bagging_fraction"],
        "bagging_freq": 1,
        "seed": p.get("seed", 0),
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X, y, feature_name=FEATURES, categorical_feature=CATEGORICAL, free_raw_data=False)
    dval = lgb.Dataset(Xv, yv, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=p["num_boost_round"], valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False)],
    )
    info = {"fit_rows": int(len(y)), "val_rows": int(len(yv)), "best_iteration": int(booster.best_iteration)}
    return booster, info


def predict(ds: xr.Dataset, booster, mask: np.ndarray, fb: FeatureBuilder | None = None,
            verbose: bool = False) -> np.ndarray:
    """(time, y, x) predictions in degC, NaN outside mask."""
    fb = fb or FeatureBuilder(ds)
    dates = [str(t)[:10] for t in ds.time.values.astype("datetime64[D]")]
    out = np.full(mask.shape, np.nan, dtype=np.float32)
    for t in range(mask.shape[0]):
        m = np.zeros_like(mask)
        m[t] = mask[t]
        X, base, idx = fb.matrix(m)
        if len(X):
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = base + booster.predict(X, num_iteration=booster.best_iteration)
        if verbose:
            print(f"  predicted {dates[t]} ({t + 1}/{mask.shape[0]}, {len(X):,} pixels)", flush=True)
    return out


def predict_domain(ds: xr.Dataset) -> np.ndarray:
    """(time, y, x) pixels where the model can be applied: land, valid coarse cell, NDVI available."""
    from .splits import WATER

    f = int(ds.attrs["coarse_factor"])
    coarse_ok = np.repeat(np.repeat(ds.valid_coarse.values, f, axis=1), f, axis=2)
    lc = ds.landcover.values
    land = (lc != WATER) & (lc != 0)
    return coarse_ok & land[None] & np.isfinite(ds.ndvi.values)


def importance(booster) -> list[tuple[str, float]]:
    gain = booster.feature_importance(importance_type="gain")
    total = gain.sum() or 1.0
    return sorted(zip(FEATURES, (gain / total).tolist()), key=lambda r: -r[1])


def save_predictions(ds: xr.Dataset, pred: np.ndarray, name: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = xr.Dataset({name: (("time", "y", "x"), pred, {"units": "degC"})},
                     coords={"time": ds.time, "y": ds.y, "x": ds.x}, attrs={"crs": ds.attrs["crs"]})
    out.to_zarr(path, mode="w")
    return path
