"""Milestone 2: multi-date stack, splits, bilinear baseline, metrics."""

from __future__ import annotations

import copy
import csv

import numpy as np
import pytest
from affine import Affine

from uhd.align import align, grid_of
from uhd.baseline import bilinear_upsample, predict_bilinear
from uhd.metrics import evaluate, scores, write_results
from uhd import splits as S
from uhd.splits import eval_mask, split_masks
from uhd.stack import stack_scenes


def _two_dates(scene):
    cfg, manifest, _ = scene
    m2 = copy.deepcopy(manifest)
    m2["landsat"]["datetime_utc"] = "2023-08-23T06:46:00+00:00"
    m2["landsat"]["id"] = "LC09_TEST_2"
    cfg = {**cfg, "splits": {"test_dates": ["2023-08-23"], "test_area_east_frac": 0.3}}
    return cfg, [m2, manifest]          # deliberately out of order


@pytest.fixture(scope="module")
def stack(scene):
    cfg, manifests = _two_dates(scene)
    return cfg, stack_scenes(cfg, manifests)


def test_bilinear_reproduces_linear_field():
    f = 31
    yc, xc = np.meshgrid(np.arange(6), np.arange(8), indexing="ij")
    coarse = (30 + 0.5 * yc + 0.25 * xc).astype(np.float64)
    fine = bilinear_upsample(coarse, f)
    u = (np.arange(8 * f) + 0.5) / f - 0.5
    v = (np.arange(6 * f) + 0.5) / f - 0.5
    truth = 30 + 0.5 * v[:, None] + 0.25 * u[None, :]
    inner = (slice(f, -f), slice(f, -f))   # away from the clamped outer half-cells
    assert np.allclose(fine[inner], truth[inner], atol=1e-4)


def test_bilinear_skips_missing_neighbours():
    f = 31
    coarse = np.full((3, 3), 40.0)
    coarse[1, 2] = np.nan                  # a water cell
    fine = bilinear_upsample(coarse, f)
    assert np.isfinite(fine[f:2 * f, f:2 * f]).all()        # its valid neighbour is fully covered
    assert np.allclose(fine[f:2 * f, f:2 * f], 40.0)        # and not pulled towards 0 by the gap


def test_stack_shares_one_grid(stack):
    _, ds = stack
    assert ds.sizes["time"] == 2
    assert list(ds.time.values.astype("datetime64[D]").astype(str)) == ["2023-07-15", "2023-08-23"]
    assert ds.landcover.dims == ("y", "x") and ds.elevation.dims == ("y", "x")
    assert ds.lst_ref.dims == ("time", "y", "x") and ds.lst_coarse.dims == ("time", "yc", "xc")
    assert list(ds.attrs["scenes"]) == ["2023-07-15", "2023-08-23"]


def test_misaligned_grid_is_rejected(scene):
    cfg, manifest, _ = scene
    t, shape, crs = grid_of(align(cfg, manifest))
    shifted = Affine(t.a, t.b, t.c + 15, t.d, t.e, t.f)     # half a pixel off
    with pytest.raises(ValueError, match="not aligned"):
        align(cfg, manifest, grid=(shifted, shape, crs))


def test_splits_are_disjoint_and_follow_coarse_cells(stack):
    cfg, ds = stack
    f = ds.attrs["coarse_factor"]
    area = S.test_area_mask(ds, cfg)
    first = int(np.argmax(area[0]))
    assert first % f == 0 and area[:, first:].all() and not area[:, :first].any()

    masks = split_masks(ds, cfg)
    total = sum(m.astype(int) for m in masks.values())
    assert total.max() == 1                                 # no pixel in two splits
    assert (total.astype(bool) == eval_mask(ds)).all()      # together they cover every scored pixel
    assert not masks["train"][1].any()                      # held-out date never in train
    assert not masks["train"][:, :, first:].any()           # held-out area never in train
    assert not eval_mask(ds)[:, ds.landcover.values == 80].any()   # water never scored


def test_bilinear_baseline_end_to_end(stack, tmp_path):
    cfg, ds = stack
    masks = split_masks(ds, cfg)
    pred = predict_bilinear(ds.lst_coarse.values, ds.attrs["coarse_factor"])
    rows = evaluate("bilinear", pred, ds.lst_ref.values, ds.landcover.values, masks, min_pixels=1)
    overall = {r["split"]: r for r in rows if r["landcover"] == "all"}
    assert set(overall) == {"train", "test_area", "test_dates", "test_both"}
    for r in overall.values():
        assert r["n"] > 0 and r["rmse_c"] >= r["mae_c"] >= abs(r["bias_c"])

    path = write_results(rows, tmp_path / "results.csv")
    write_results([{**rows[0], "method": "lightgbm"}], path)
    write_results(rows, path)                               # rerun replaces, does not duplicate
    with path.open() as f:
        methods = [r["method"] for r in csv.DictReader(f)]
    assert methods.count("bilinear") == len(rows) and methods.count("lightgbm") == 1


def test_scores_sign_convention():
    obs = np.array([40.0, 42.0])
    s = scores(obs - 2, obs)
    assert s["bias_c"] == pytest.approx(-2) and s["mae_c"] == pytest.approx(2) and s["rmse_c"] == pytest.approx(2)
