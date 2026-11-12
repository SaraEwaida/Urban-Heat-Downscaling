"""Synthetic rasters that mimic the real products' CRS, grids and encodings."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from uhd import products as P

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
        "qc": _write(d / "qc.tif", qc, SINU, mt, 0),         # real QC_Day COGs declare nodata=0
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
