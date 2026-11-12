"""Milestone 3: LightGBM on a synthetic stack with known fine-scale structure."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from uhd.baseline import bilinear_upsample, predict_bilinear
from uhd.features import FEATURES, FeatureBuilder, _block_mean
from uhd.lgbm import fit_masks, predict, train
from uhd.metrics import evaluate
from uhd.splits import eval_mask, split_masks

F = 31
DATES = ["2023-05-03", "2023-05-11", "2023-06-20", "2023-08-23"]
CFG = {
    "splits": {"test_dates": ["2023-08-23"], "test_area_east_frac": 0.3},
    "lightgbm": {"val_dates": ["2023-06-20"], "max_train_rows": 200_000, "learning_rate": 0.1,
                 "num_leaves": 31, "min_child_samples": 50, "feature_fraction": 1.0, "bagging_fraction": 1.0,
                 "num_boost_round": 300, "early_stopping_rounds": 30, "seed": 0},
}


def make_stack(seed: int = 0) -> xr.Dataset:
    """Fine LST = smooth coarse field + land-cover and NDVI effects that cancel inside each cell
    on average, so the coarse input alone cannot recover them."""
    rng = np.random.default_rng(seed)
    yc, xc = 6, 12
    h, w = yc * F, xc * F
    lc = rng.choice([50, 60, 10, 80], size=(h // 3, w // 3), p=[0.35, 0.35, 0.2, 0.1])
    lc = np.repeat(np.repeat(lc, 3, axis=0), 3, axis=1).astype(np.uint8)
    elev = rng.normal(10, 3, (h, w)).astype(np.float32)

    lst_ref, ndvi, coarse = [], [], []
    for t in range(len(DATES)):
        base_c = 40 + t + 2 * np.sin(np.arange(xc) / 3)[None, :] + np.zeros((yc, 1))
        nd = np.where(lc == 10, 0.6, np.where(lc == 80, -0.1, 0.1)) + rng.normal(0, 0.03, (h, w))
        effect = 6 * (lc == 50) + 4 * (lc == 60) - 8 * (nd - 0.1) - 10 * (lc == 80)
        fine = bilinear_upsample(base_c, F) + effect - _block_mean(effect, F) + rng.normal(0, 0.3, (h, w))
        lst_ref.append(fine.astype(np.float32))
        ndvi.append(nd.astype(np.float32))
        coarse.append(base_c.astype(np.float32) - 5)          # MODIS-like cold offset

    times = np.array([np.datetime64(d) for d in DATES], dtype="datetime64[ns]")
    x = 230000 + 30 * (np.arange(w) + 0.5)
    y = 2718000 - 30 * (np.arange(h) + 0.5)
    return xr.Dataset(
        {
            "lst_ref": (("time", "y", "x"), np.stack(lst_ref)),
            "valid_ref": (("time", "y", "x"), np.ones((len(DATES), h, w), bool)),
            "ndvi": (("time", "y", "x"), np.stack(ndvi)),
            "lst_coarse": (("time", "yc", "xc"), np.stack(coarse)),
            "valid_coarse": (("time", "yc", "xc"), np.ones((len(DATES), yc, xc), bool)),
            "landcover": (("y", "x"), lc),
            "elevation": (("y", "x"), elev),
        },
        coords={"time": times, "x": x, "y": y, "xc": x[F // 2::F], "yc": y[F // 2::F]},
        attrs={"coarse_factor": F, "crs": "EPSG:32640", "transform": [30, 0, 230000, 0, -30, 2718000]},
    )


@pytest.fixture(scope="module")
def fitted():
    ds = make_stack()
    masks = split_masks(ds, CFG)
    fb = FeatureBuilder(ds)
    booster, info = train(ds, CFG, masks, fb)
    return ds, masks, fb, booster, info


def test_features_use_no_target_and_no_coordinates():
    assert not any(k in FEATURES for k in ("lst_ref", "x", "y", "row", "col"))
    ds = make_stack()
    fb = FeatureBuilder(ds)
    m = fb.maps(0)
    ds2 = ds.copy(deep=True)
    ds2["lst_ref"] = ds2.lst_ref * 0 + 99                     # changing the target ...
    m2 = FeatureBuilder(ds2).maps(0)
    assert all(np.array_equal(m[k], m2[k], equal_nan=True) for k in FEATURES)   # ... changes no feature


def test_fit_and_val_stay_inside_train_split(fitted):
    ds, masks, _, _, info = fitted
    fit_m, val_m = fit_masks(ds, CFG, masks)
    assert not (fit_m & val_m).any()
    for name in ("test_area", "test_dates", "test_both"):
        assert not ((fit_m | val_m) & masks[name]).any()
    assert info["fit_rows"] == fit_m.sum() and info["val_rows"] == val_m.sum()


def test_val_dates_cannot_be_test_dates():
    ds = make_stack()
    bad = {**CFG, "lightgbm": {**CFG["lightgbm"], "val_dates": ["2023-08-23"]}}
    with pytest.raises(ValueError, match="overlap"):
        fit_masks(ds, bad, split_masks(ds, bad))


def test_lightgbm_beats_bilinear_on_held_out_splits(fitted):
    ds, masks, fb, booster, _ = fitted
    obs = ds.lst_ref.values
    p_lgb = predict(ds, booster, eval_mask(ds), fb)
    p_bil = predict_bilinear(ds.lst_coarse.values, F)
    lgb = {r["split"]: r for r in evaluate("lightgbm", p_lgb, obs, ds.landcover.values, masks) if r["landcover"] == "all"}
    bil = {r["split"]: r for r in evaluate("bilinear", p_bil, obs, ds.landcover.values, masks) if r["landcover"] == "all"}
    for split in ("test_area", "test_dates", "test_both"):
        assert lgb[split]["mae_c"] < 0.5 * bil[split]["mae_c"], split
        assert abs(lgb[split]["bias_c"]) < 1.0, split
