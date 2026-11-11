"""Align coarse LST, Landsat LST, NDVI, land cover and elevation to one projected grid
and save to Zarr.

Grids
- Fine grid: the Landsat scene's own 30 m UTM grid, cropped to the bbox. Landsat is read
  without resampling. Height and width are trimmed to multiples of `coarse.factor`.
- Coarse grid: the fine grid aggregated by `factor` (31 x 30 m = 930 m), so every coarse
  cell contains exactly factor x factor fine pixels. MODIS is warped onto it with nearest
  neighbour so QC flags stay attached to the value they describe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from . import products as P


def _open(href: str):
    """Open a local path or a (to-be-signed) Planetary Computer URL."""
    if href.startswith("http"):
        import planetary_computer

        href = planetary_computer.sign(href)
    return rasterio.open(href)


def fine_grid(st_href: str, bbox, factor: int):
    """Window of the Landsat scene covering bbox, trimmed to a multiple of factor."""
    with _open(st_href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        w = from_bounds(left, bottom, right, top, src.transform)
        col0, row0 = int(np.ceil(w.col_off)), int(np.ceil(w.row_off))
        col0, row0 = max(col0, 0), max(row0, 0)
        width = min(int(np.floor(w.width)), src.width - col0) // factor * factor
        height = min(int(np.floor(w.height)), src.height - row0) // factor * factor
        if width <= 0 or height <= 0:
            raise ValueError("bbox does not overlap the Landsat scene enough for one coarse cell")
        window = Window(col0, row0, width, height)
        return window, src.window_transform(window), src.crs


def _read_window(href: str, window: Window) -> np.ndarray:
    with _open(href) as src:
        return src.read(1, window=window)


def _warp(hrefs: list[str], crs, transform, shape, resampling: Resampling, nodata, dtype) -> np.ndarray:
    """Warp one or more tiles onto the target grid; first valid value wins."""
    out = np.full(shape, nodata, dtype=dtype)
    for href in hrefs:
        with _open(href) as src:
            src_nodata = src.nodata if src.nodata is not None else nodata
            with WarpedVRT(src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                           resampling=resampling, nodata=src_nodata) as vrt:
                arr = vrt.read(1).astype(dtype)
        fill = (out == nodata) & (arr != src_nodata)
        out[fill] = arr[fill]
    return out


def _coords(transform, shape):
    h, w = shape
    x = transform.c + transform.a * (np.arange(w) + 0.5)
    y = transform.f + transform.e * (np.arange(h) + 0.5)
    return x, y


def align(cfg: dict[str, Any], manifest: dict[str, Any]) -> xr.Dataset:
    factor = int(cfg["coarse"]["factor"])
    ls = manifest["landsat"]["assets"]
    md = manifest["modis"]["assets"]

    window, transform, crs = fine_grid(ls["st"], manifest["bbox"], factor)
    shape = (int(window.height), int(window.width))
    cshape = (shape[0] // factor, shape[1] // factor)
    ctransform = transform @ Affine.scale(factor)

    # Landsat on its native 30 m grid (no resampling)
    st_dn = _read_window(ls["st"], window)
    qa = _read_window(ls["qa_pixel"], window)
    st_unc = P.landsat_st_uncertainty_k(_read_window(ls["st_qa"], window))
    lst_ref = P.landsat_st_celsius(st_dn)
    valid_ref = (
        P.landsat_clear(qa)
        & np.isfinite(lst_ref)
        & (np.nan_to_num(st_unc, nan=np.inf) <= cfg["fine"]["max_st_uncertainty_k"])
    )
    lst_ref = np.where(valid_ref, lst_ref, np.nan).astype(np.float32)

    red = P.landsat_sr(_read_window(ls["red"], window))
    nir = P.landsat_sr(_read_window(ls["nir"], window))
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    ndvi = np.where(P.landsat_clear(qa) & np.isfinite(ndvi), np.clip(ndvi, -1, 1), np.nan).astype(np.float32)

    # MODIS onto the coarse grid (nearest, keeps QC aligned with values)
    lst_dn = _warp([md["lst"]], crs, ctransform, cshape, Resampling.nearest, P.MODIS_LST_FILL, np.uint16)
    qc = _warp([md["qc"]], crs, ctransform, cshape, Resampling.nearest, 255, np.uint8)
    vt = _warp([md["view_time"]], crs, ctransform, cshape, Resampling.nearest, P.MODIS_VIEW_TIME_FILL, np.uint8)
    lst_coarse = P.modis_lst_celsius(lst_dn)
    valid_coarse = np.isfinite(lst_coarse) & P.modis_qc_ok(qc, cfg["coarse"]["max_lst_error_k"])
    lst_coarse = np.where(valid_coarse, lst_coarse, np.nan).astype(np.float32)
    lst_coarse_on_fine = np.repeat(np.repeat(lst_coarse, factor, axis=0), factor, axis=1)

    # Land cover: categorical -> nearest. DEM: continuous -> bilinear.
    landcover = _warp(manifest["landcover"]["hrefs"], crs, transform, shape, Resampling.nearest,
                      P.WORLDCOVER_NODATA, np.uint8)
    elevation = _warp(manifest["dem"]["hrefs"], crs, transform, shape, Resampling.bilinear,
                      -32767.0, np.float32)
    elevation[elevation == -32767.0] = np.nan

    # MODIS view time (local solar hours) -> approximate UTC
    vt_valid = (vt != P.MODIS_VIEW_TIME_FILL) & valid_coarse
    lon_c = (manifest["bbox"][0] + manifest["bbox"][2]) / 2
    solar_h = float(vt[vt_valid].mean() * P.MODIS_VIEW_TIME_SCALE) if vt_valid.any() else None
    utc_h = None if solar_h is None else (solar_h - lon_c / 15.0) % 24

    x, y = _coords(transform, shape)
    xc, yc = _coords(ctransform, cshape)
    epsg = crs.to_epsg()

    def var(dims, data, **attrs):
        return (dims, data, attrs)

    ds = xr.Dataset(
        {
            "lst_ref": var(("y", "x"), lst_ref, units="degC", long_name="Landsat surface temperature (reference)",
                           grid_m=P.LANDSAT_GRID_M, effective_thermal_resolution_m=P.LANDSAT_TIRS_NATIVE_M,
                           resampling="none (native Landsat grid)"),
            "valid_ref": var(("y", "x"), valid_ref, long_name="Landsat clear and low-uncertainty mask"),
            "ndvi": var(("y", "x"), ndvi, units="1", long_name="NDVI from Landsat SR (red, nir08)",
                        grid_m=P.LANDSAT_GRID_M, resampling="none"),
            "landcover": var(("y", "x"), landcover, long_name="ESA WorldCover class", nodata=P.WORLDCOVER_NODATA,
                             source_resolution_m=10, resampling="nearest",
                             classes=json.dumps(P.WORLDCOVER_CLASSES)),
            "elevation": var(("y", "x"), elevation, units="m", long_name="Copernicus DEM GLO-30",
                             source_resolution_m=30, resampling="bilinear"),
            "lst_coarse_on_fine": var(("y", "x"), lst_coarse_on_fine, units="degC",
                                      long_name="Coarse LST repeated onto the 30 m grid (blocky, no interpolation)"),
            "lst_coarse": var(("yc", "xc"), lst_coarse, units="degC", long_name="MODIS LST (coarse input)",
                              source_resolution_m=P.MODIS_NATIVE_M, grid_m=P.LANDSAT_GRID_M * factor,
                              resampling="nearest"),
            "valid_coarse": var(("yc", "xc"), valid_coarse, long_name="MODIS QC mask"),
        },
        coords={"x": x, "y": y, "xc": xc, "yc": yc},
    )
    ds.attrs = {
        "city": manifest["city"],
        "bbox_wgs84": list(manifest["bbox"]),
        "crs": f"EPSG:{epsg}" if epsg else crs.to_wkt(),
        "transform": list(transform)[:6],
        "coarse_transform": list(ctransform)[:6],
        "coarse_factor": factor,
        "landsat": {k: v for k, v in manifest["landsat"].items() if k != "assets"},
        "modis": {**{k: v for k, v in manifest["modis"].items() if k != "assets"},
                  "mean_view_time_local_solar_h": solar_h, "approx_view_time_utc_h": utc_h},
        "landcover_ids": manifest["landcover"]["ids"],
        "dem_ids": manifest["dem"]["ids"],
        "valid_fraction_ref": float(valid_ref.mean()),
        "valid_fraction_coarse": float(valid_coarse.mean()),
    }
    return ds


def save(ds: xr.Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w")
    return path
