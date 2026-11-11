"""Load a city config (YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    required = ["city", "bbox", "search", "coarse", "fine", "paths"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if len(cfg["bbox"]) != 4:
        raise ValueError(f"{path}: bbox must be [lon_min, lat_min, lon_max, lat_max]")
    return cfg
