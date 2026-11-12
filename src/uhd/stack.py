"""Align several dates onto one grid and stack them along a time axis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .align import align, grid_of

STATIC = ["landcover", "elevation"]


def stack_scenes(cfg: dict[str, Any], manifests: list[dict[str, Any]]) -> xr.Dataset:
    """Align each manifest on the grid of the first one and concatenate along `time`.

    Land cover and elevation do not change between dates and are kept once.
    """
    manifests = sorted(manifests, key=lambda m: m["landsat"]["datetime_utc"])
    first = align(cfg, manifests[0])
    grid = grid_of(first)
    per_date = [first] + [align(cfg, m, grid=grid) for m in manifests[1:]]

    scenes = {}
    dynamic = []
    for ds in per_date:
        date = ds.attrs["landsat"]["datetime_utc"][:10]
        scenes[date] = {
            "landsat": ds.attrs["landsat"],
            "modis": ds.attrs["modis"],
            "valid_fraction_ref": ds.attrs["valid_fraction_ref"],
            "valid_fraction_coarse": ds.attrs["valid_fraction_coarse"],
        }
        dynamic.append(ds.drop_vars(STATIC))

    times = np.array([np.datetime64(d["landsat"]["datetime_utc"][:19]) for d in scenes.values()], dtype="datetime64[ns]")
    out = xr.concat(dynamic, dim="time", combine_attrs="drop").assign_coords(time=times)
    for v in STATIC:
        out[v] = first[v]
    out.attrs = {k: v for k, v in first.attrs.items()
                 if k not in ("landsat", "modis", "valid_fraction_ref", "valid_fraction_coarse")}
    out.attrs["scenes"] = scenes
    return out


def align_all(cfg: dict[str, Any]) -> xr.Dataset:
    """Fetch a manifest for every configured date, align them all, return the stack."""
    from .fetch import build_manifest, find_candidates

    wanted = list(cfg["dates"])
    by_date = {c["landsat"].datetime.date().isoformat(): c for c in find_candidates(cfg)}
    missing = [d for d in wanted if d not in by_date]
    if missing:
        raise RuntimeError(f"No Landsat/Terra pair for {missing}; run `uhd fetch --list` and edit `dates`.")

    scenes_dir = Path(cfg["paths"]["scenes_dir"])
    scenes_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for d in wanted:
        m = build_manifest(cfg, by_date[d])
        (scenes_dir / f"{d}.json").write_text(json.dumps(m, indent=2))
        manifests.append(m)
        print(f"{d}: {m['landsat']['id']} + {m['modis']['id']}")
    return stack_scenes(cfg, manifests)
