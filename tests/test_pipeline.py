"""End-to-end test of milestone 1 on the synthetic scene from conftest.py."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import _temp_c
from uhd import products as P
from uhd.align import align, save
from uhd.fetch import bbox_within_geometry
from uhd.preview import plot_input_vs_reference


def test_qa_decoding():
    qa = np.array([21824, 21824 | (1 << P.QA_CLOUD), 1, 21824 | (1 << P.QA_SHADOW)], dtype=np.uint16)
    assert P.landsat_clear(qa).tolist() == [True, False, False, False]
    qc = np.array([0b00000000, 0b00000001, 0b01000001, 0b10000001, 0b00000010], dtype=np.uint8)
    assert P.modis_qc_ok(qc, 2).tolist() == [True, True, True, False, False]


def test_bbox_within_geometry():
    geom = {"type": "Polygon", "coordinates": [[[54, 24], [55, 24], [55, 25], [54, 25], [54, 24]]]}
    assert bbox_within_geometry([54.3, 24.3, 54.7, 24.5], geom)
    assert not bbox_within_geometry([53.9, 24.3, 54.7, 24.5], geom)


def test_align_and_preview(scene):
    cfg, manifest, d = scene
    ds = align(cfg, manifest)
    f = cfg["coarse"]["factor"]

    assert ds.attrs["crs"] == "EPSG:32640"
    assert ds.sizes["y"] % f == 0 and ds.sizes["x"] % f == 0
    assert ds.sizes["yc"] * f == ds.sizes["y"] and ds.sizes["xc"] * f == ds.sizes["x"]

    # Landsat values decode back to the synthetic field (within DN rounding)
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    truth = _temp_c(xx, yy)
    ok = ds.valid_ref.values
    assert np.nanmax(np.abs(ds.lst_ref.values[ok] - truth[ok])) < 0.01
    assert np.isnan(ds.lst_ref.values[~ok]).all()
    assert 0 < ds.attrs["valid_fraction_ref"] < 1      # cloud and high-uncertainty patches removed

    # Coarse input: 45 C where QC is good, NaN where not produced
    vc = ds.valid_coarse.values
    assert vc.mean() > 0.5                              # QC=0 ("good") must survive nodata=0 in the file
    assert np.allclose(ds.lst_coarse.values[vc], 45, atol=0.02)
    assert np.isnan(ds.lst_coarse.values[~vc]).all()
    assert ds.lst_coarse_on_fine.shape == ds.lst_ref.shape

    # Land cover stays categorical (nearest) and is mosaicked across tiles
    lc_vals = set(np.unique(ds.landcover.values).tolist())
    assert lc_vals - {0} == {10, 60}                    # one class from each tile

    assert np.isfinite(ds.elevation.values).mean() > 0.99
    assert np.allclose(np.nanmean(ds.ndvi.values), 0.4, atol=0.01)
    assert ds.attrs["modis"]["mean_view_time_local_solar_h"] == pytest.approx(10.3)

    # Zarr round trip and figure
    path = save(ds, d / "aligned.zarr")
    back = xr.open_zarr(path).load()
    np.testing.assert_array_equal(back.landcover.values, ds.landcover.values)
    assert back.attrs["landsat"]["id"] == "LC09_TEST"
    png = plot_input_vs_reference(back, d / "preview.png")
    assert png.exists() and png.stat().st_size > 10_000
