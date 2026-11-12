"""Train / test splits, defined in the config before any model is fitted.

Two independent axes:
- area: the eastern `test_area_east_frac` of coarse columns is the held-out area. The cut
  follows coarse-cell edges, so no 930 m input cell is shared between areas.
- dates: `test_dates` are held out.

Combined splits:
    train       training area, training dates   (models fit here only)
    test_area   held-out area, training dates   (new place)
    test_dates  training area, held-out dates   (new day)
    test_both   held-out area, held-out dates   (new place and new day)

Later steps that cut image patches must take patches from one split only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .products import WORLDCOVER_NODATA

WATER = 80
SPLITS = {"train": (False, False), "test_area": (True, False), "test_dates": (False, True), "test_both": (True, True)}


def test_area_mask(ds: xr.Dataset, cfg: dict[str, Any]) -> np.ndarray:
    """(y, x) True in the held-out area."""
    f = int(ds.attrs["coarse_factor"])
    col_start_c = int(round(ds.sizes["xc"] * (1 - cfg["splits"]["test_area_east_frac"])))
    m = np.zeros((ds.sizes["y"], ds.sizes["x"]), dtype=bool)
    m[:, col_start_c * f:] = True
    return m


def test_date_flags(ds: xr.Dataset, cfg: dict[str, Any]) -> np.ndarray:
    """(time,) True for held-out dates."""
    dates = [str(t)[:10] for t in ds.time.values.astype("datetime64[D]")]
    unknown = set(cfg["splits"]["test_dates"]) - set(dates)
    if unknown:
        raise ValueError(f"test_dates not in the stack: {sorted(unknown)}")
    return np.array([d in cfg["splits"]["test_dates"] for d in dates])


def eval_mask(ds: xr.Dataset) -> np.ndarray:
    """(time, y, x) pixels every method is scored on: clear Landsat, valid MODIS cell, land."""
    f = int(ds.attrs["coarse_factor"])
    coarse_ok = np.repeat(np.repeat(ds.valid_coarse.values, f, axis=1), f, axis=2)
    lc = ds.landcover.values
    land = (lc != WATER) & (lc != WORLDCOVER_NODATA)
    return ds.valid_ref.values & coarse_ok & land[None]


def split_masks(ds: xr.Dataset, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """name -> (time, y, x) boolean mask, already intersected with eval_mask."""
    area = test_area_mask(ds, cfg)
    dates = test_date_flags(ds, cfg)
    base = eval_mask(ds)
    out = {}
    for name, (in_test_area, in_test_dates) in SPLITS.items():
        a = area if in_test_area else ~area
        t = dates if in_test_dates else ~dates
        out[name] = base & a[None] & t[:, None, None]
    return out


def describe(ds: xr.Dataset, cfg: dict[str, Any], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    f = int(ds.attrs["coarse_factor"])
    col_start = int(np.argmax(test_area_mask(ds, cfg)[0]))
    dates = [str(t)[:10] for t in ds.time.values.astype("datetime64[D]")]
    flags = test_date_flags(ds, cfg)
    return {
        "test_dates": [d for d, k in zip(dates, flags) if k],
        "train_dates": [d for d, k in zip(dates, flags) if not k],
        "test_area": {"rule": f"east of x = {float(ds.x.values[col_start] - 15):.0f} m ({ds.attrs['crs']})",
                      "first_fine_col": col_start, "first_coarse_col": col_start // f},
        "pixels": {k: int(v.sum()) for k, v in masks.items()},
    }


def save_description(desc: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(desc, indent=2))
    return path
