"""Milestone 4: U-Net patches, normalisation and training on the synthetic stack."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_lightgbm import CFG as LGB_CFG, make_stack  # noqa: E402
from uhd import unet  # noqa: E402
from uhd.baseline import predict_bilinear  # noqa: E402
from uhd.metrics import evaluate  # noqa: E402
from uhd.splits import split_masks  # noqa: E402

CFG = {
    "splits": LGB_CFG["splits"],
    "unet": {"val_dates": ["2023-06-20"], "patch_cells": 2, "train_stride_cells": 1, "base_channels": 8,
             "batch_size": 8, "learning_rate": 0.003, "epochs": 25, "patience": 6, "seed": 0},
}


@pytest.fixture(scope="module")
def fitted():
    ds = make_stack()
    masks = split_masks(ds, CFG)
    model, prep, info = unet.train(ds, CFG, masks, verbose=False)
    return ds, masks, model, prep, info


def test_tile_starts_cover_and_stay_inside():
    s = unet.tile_starts(0, 100, 31, 31)
    assert s[0] == 0 and s[-1] + 31 == 100 and all(0 <= a and a + 31 <= 100 for a in s)
    with pytest.raises(ValueError):
        unet.tile_starts(0, 20, 31, 31)


def test_training_patches_stay_in_training_area_and_dates(fitted):
    ds, masks, _, prep, _ = fitted
    test_t = {list(ds.time.values.astype("datetime64[D]").astype(str)).index(d) for d in CFG["splits"]["test_dates"]}
    for dates in (prep.fit_t, prep.val_t):
        for t, y0, x0 in unet.train_patches(prep, dates, 1):
            assert x0 + prep.patch <= prep.boundary           # never crosses into the held-out area
            assert t not in test_t
            assert x0 % prep.f == 0 and y0 % prep.f == 0 or y0 + prep.patch == ds.sizes["y"]
    assert not set(prep.fit_t) & set(prep.val_t)


def test_normalisation_ignores_held_out_data():
    ds = make_stack()
    masks = split_masks(ds, CFG)
    a = unet.Prepared(ds, CFG, masks)
    ds2 = ds.copy(deep=True)
    ds2["ndvi"].values[:, :, a.boundary:] = 5.0                # change the held-out area everywhere
    t_test = list(ds.time.values.astype("datetime64[D]").astype(str)).index("2023-08-23")
    ds2["ndvi"].values[t_test] = 9.0                           # change a held-out date everywhere
    ds2["lst_ref"].values[t_test] += 50
    b = unet.Prepared(ds2, CFG, split_masks(ds2, CFG))
    assert a.mean == b.mean and a.std == b.std and a.res_mean == b.res_mean and a.res_std == b.res_std


def test_unet_beats_bilinear_on_held_out_splits(fitted):
    ds, masks, model, prep, _ = fitted
    obs = ds.lst_ref.values
    p_unet = unet.predict(model, prep)
    p_bil = predict_bilinear(ds.lst_coarse.values, prep.f)
    u = {r["split"]: r for r in evaluate("unet", p_unet, obs, ds.landcover.values, masks) if r["landcover"] == "all"}
    b = {r["split"]: r for r in evaluate("bilinear", p_bil, obs, ds.landcover.values, masks) if r["landcover"] == "all"}
    for split in ("test_area", "test_dates", "test_both"):
        assert u[split]["mae_c"] < 0.5 * b[split]["mae_c"], (split, u[split], b[split])


def test_val_dates_cannot_be_test_dates():
    ds = make_stack()
    bad = copy.deepcopy(CFG)
    bad["unet"]["val_dates"] = ["2023-08-23"]
    with pytest.raises(ValueError, match="overlap"):
        unet.Prepared(ds, bad, split_masks(ds, bad))
