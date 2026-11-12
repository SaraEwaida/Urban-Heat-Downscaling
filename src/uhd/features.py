"""Per-pixel features for the pixel-based model.

Only inputs are used: coarse LST, NDVI, land cover and elevation. The Landsat temperature
(the target) never enters a feature. No pixel coordinates are used, so the model cannot
memorise locations and has to generalise to the held-out area.

Cell-level context (mean NDVI, land-cover fractions inside the 930 m cell) tells the model
what the coarse value is an average of, which is the core idea of thermal downscaling:
a pixel greener than its cell should be cooler than the cell value, and vice versa.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .baseline import bilinear_upsample
from .products import WORLDCOVER_NODATA

LC_GROUPS = {
    "frac_water": [80],
    "frac_built": [50],
    "frac_bare": [60],
    "frac_veg": [10, 20, 30, 40, 90, 95],
}

FEATURES = [
    "lst_cell",          # raw coarse value of the pixel's cell
    "lst_bilinear",      # bilinear surface (also the residual base)
    "lst_date_mean",     # mean valid coarse LST of the scene (date-level warmth)
    "ndvi",
    "ndvi_cell_mean",
    "ndvi_anom",         # pixel NDVI minus its cell's mean
    "elevation",
    "elev_anom",
    "landcover",         # categorical
    *LC_GROUPS,
]
CATEGORICAL = ["landcover"]


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean of finite values in each f x f block, repeated back to full resolution."""
    h, w = a.shape
    blocks = a.reshape(h // f, f, w // f, f)
    ok = np.isfinite(blocks)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, blocks, 0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / n, np.nan)
    return np.repeat(np.repeat(m, f, axis=0), f, axis=1).astype(np.float32)


class FeatureBuilder:
    """Builds feature maps date by date; static layers are computed once."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self.f = int(ds.attrs["coarse_factor"])
        lc = ds.landcover.values
        elev = ds.elevation.values.astype(np.float32)
        self.static = {
            "landcover": np.where(lc == WORLDCOVER_NODATA, np.nan, lc).astype(np.float32),
            "elevation": elev,
            "elev_anom": elev - _block_mean(elev, self.f),
        }
        for name, codes in LC_GROUPS.items():
            self.static[name] = _block_mean(np.isin(lc, codes).astype(np.float32), self.f)

    def maps(self, t: int) -> dict[str, np.ndarray]:
        ds, f = self.ds, self.f
        coarse = ds.lst_coarse.values[t]
        ndvi = ds.ndvi.values[t].astype(np.float32)
        ndvi_cell = _block_mean(ndvi, f)
        out = dict(self.static)
        out.update({
            "lst_cell": np.repeat(np.repeat(coarse, f, axis=0), f, axis=1),
            "lst_bilinear": bilinear_upsample(coarse, f),
            "lst_date_mean": np.full(ndvi.shape, np.nanmean(coarse), dtype=np.float32),
            "ndvi": ndvi,
            "ndvi_cell_mean": ndvi_cell,
            "ndvi_anom": ndvi - ndvi_cell,
        })
        return out

    def matrix(self, mask: np.ndarray):
        """Rows for every True pixel of a (time, y, x) mask.

        Returns X (n, len(FEATURES)), the bilinear base (n,), and the (t, y, x) index.
        """
        xs, bases, idx = [], [], []
        for t in range(mask.shape[0]):
            iy, ix = np.nonzero(mask[t])
            if iy.size == 0:
                continue
            m = self.maps(t)
            xs.append(np.column_stack([m[k][iy, ix] for k in FEATURES]).astype(np.float32))
            bases.append(m["lst_bilinear"][iy, ix])
            idx.append(np.column_stack([np.full(iy.size, t), iy, ix]))
        if not xs:
            return np.empty((0, len(FEATURES)), np.float32), np.empty(0, np.float32), np.empty((0, 3), int)
        return np.concatenate(xs), np.concatenate(bases), np.concatenate(idx)
