"""End-to-end test on synthetic rasters that mimic the real products' CRS, grids and encodings."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from uhd import products as P
from uhd.align import align, save
from uhd.fetch import bbox_within_geometry
from uhd.preview import plot_input_vs_reference

BBOX = [54.40, 24.40, 54.50, 24.48]
SINU = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
R = 6371007.181


def _write(path, arr, crs, transform, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr, 1)
    return str(path)


def _temp_c(xx, yy):
    return 40 + 8 * np.sin(xx / 4000.0) + 5 * np.cos(yy / 3000.0)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    # Landsat-like: UTM 40N, 30 m
    x0, y0, w, h = 230000.0, 2718000.0, 1000, 900
    t = from_origin(x0, y0, 30, 30)
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xx, yy = x0 + 30 * (cols + 0.5), y0 - 30 * (rows + 0.5)
    temp = _temp_c(xx, yy)
    st = np.round((temp + P.KELVIN_TO_C - P.LANDSAT_ST_OFFSET) / P.LANDSAT_ST_SCALE).astype(np.uint16)
    qa = np.full((h, w), 21824, dtype=np.uint16)          # clear land
    qa[400:450, 400:450] |= 1 << P.QA_CLOUD                # a cloud
    st_qa = np.full((h, w), 150, dtype=np.int16)           # 1.5 K
    st_qa[100:120, 100:120] = 500                          # 5 K, should be masked
    red = np.full((h, w), int((0.15 + 0.2) / P.LANDSAT_SR_SCALE), dtype=np.uint16)
    nir = np.full((h, w), int((0.35 + 0.2) / P.LANDSAT_SR_SCALE), dtype=np.uint16)
    crs = "EPSG:32640"
    ls = {
        "st": _write(d / "st.tif", st, crs, t, 0),
        "qa_pixel": _write(d / "qa.tif", qa, crs, t, 1),
        "st_qa": _write(d / "stqa.tif", st_qa, crs, t, -9999),
        "red": _write(d / "red.tif", red, crs, t, 0),
        "nir": _write(d / "nir.tif", nir, crs, t, 0),
    }

    # MODIS-like: sinusoidal, 926.625 m
    px = P.MODIS_NATIVE_M
    lat0, lon0 = np.deg2rad(24.7), np.deg2rad(54.2)
    sx0 = R * lon0 * np.cos(np.deg2rad(24.4))
    sy0 = R * lat0
    mt = from_origin(sx0, sy0, px, px)
    mw, mh = 80, 60
    lst = np.full((mh, mw), int((45 + P.KELVIN_TO_C) / P.MODIS_LST_SCALE), dtype=np.uint16)
    qc = np.zeros((mh, mw), dtype=np.uint8)
    qc[:, :5] = 0b10                                       # not produced (cloud)
    vt = np.full((mh, mw), 103, dtype=np.uint8)            # 10.3 h local solar
    md = {
        "lst": _write(d / "lst.tif", lst, SINU, mt, 0),
        "qc": _write(d / "qc.tif", qc, SINU, mt, None),
        "view_time": _write(d / "vt.tif", vt, SINU, mt, 255),
    }

    # WorldCover-like (EPSG:4326, ~10 m) split into two tiles; DEM-like (EPSG:4326, ~30 m)
    res = 1 / 12000
    lc_hrefs = []
    for i, (lon_a, cls) in enumerate([(54.3, [50, 60]), (54.45, [10, 80])]):
        lc = np.full((2400, 1800), cls[0], dtype=np.uint8)
        lc[:, 900:] = cls[1]
        lc_hrefs.append(_write(d / f"lc{i}.tif", lc, "EPSG:4326", from_origin(lon_a, 24.55, res, res), 0))
    dem = np.linspace(0, 40, 900 * 900, dtype=np.float32).reshape(900, 900)
    dem_href = _write(d / "dem.tif", dem, "EPSG:4326", from_origin(54.3, 24.6, 1 / 3600, 1 / 3600), -32767.0)

    manifest = {
        "city": "test_city",
        "bbox": BBOX,
        "landsat": {"id": "LC09_TEST", "platform": "landsat-9", "datetime_utc": "2023-07-15T06:40:00+00:00",
                    "assets": ls},
        "modis": {"id": "MOD11A1_TEST", "date": "2023-07-15", "assets": md},
        "landcover": {"ids": ["wc0", "wc1"], "hrefs": lc_hrefs},
        "dem": {"ids": ["dem0"], "hrefs": [dem_href]},
    }
    cfg = {"coarse": {"factor": 31, "max_lst_error_k": 2}, "fine": {"max_st_uncertainty_k": 3}}
    return cfg, manifest, d


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
