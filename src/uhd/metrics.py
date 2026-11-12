"""MAE, RMSE and mean bias (prediction minus observation), in degC."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .products import WORLDCOVER_CLASSES

FIELDS = ["method", "split", "landcover", "n", "mae_c", "rmse_c", "bias_c"]


def scores(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    d = (pred - obs).astype(np.float64)
    if d.size == 0:
        return {"n": 0, "mae_c": np.nan, "rmse_c": np.nan, "bias_c": np.nan}
    return {"n": int(d.size), "mae_c": float(np.abs(d).mean()),
            "rmse_c": float(np.sqrt((d ** 2).mean())), "bias_c": float(d.mean())}


def evaluate(method: str, pred: np.ndarray, obs: np.ndarray, landcover: np.ndarray,
             masks: dict[str, np.ndarray], min_pixels: int = 1000) -> list[dict[str, Any]]:
    """Scores per split, overall and per land-cover class.

    pred, obs: (time, y, x). landcover: (y, x). masks: split -> (time, y, x).
    Every mask must already be the shared evaluation mask, so all methods use the same pixels.
    """
    lc = np.broadcast_to(landcover, pred.shape)
    rows = []
    for split, m in masks.items():
        m = m & np.isfinite(obs)
        if (m & ~np.isfinite(pred)).any():
            raise ValueError(f"{method}: missing predictions inside the {split} mask")
        rows.append({"method": method, "split": split, "landcover": "all", **scores(pred[m], obs[m])})
        for code, name in WORLDCOVER_CLASSES.items():
            mc = m & (lc == code)
            if mc.sum() >= min_pixels:
                rows.append({"method": method, "split": split, "landcover": name, **scores(pred[mc], obs[mc])})
    return rows


def write_results(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write rows, replacing earlier rows for the same method(s) and keeping other methods."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = {r["method"] for r in rows}
    kept = []
    if path.exists():
        with path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["method"] not in methods]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in kept + rows:
            w.writerow({k: (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k]) for k in FIELDS})
    return path


def format_table(rows: list[dict[str, Any]], landcover: str = "all") -> str:
    sel = [r for r in rows if r["landcover"] == landcover]
    lines = [f"{'method':<10} {'split':<11} {'n':>10} {'MAE':>7} {'RMSE':>7} {'bias':>7}"]
    for r in sel:
        lines.append(f"{r['method']:<10} {r['split']:<11} {r['n']:>10,} "
                     f"{r['mae_c']:>7.2f} {r['rmse_c']:>7.2f} {r['bias_c']:>+7.2f}")
    return "\n".join(lines)
