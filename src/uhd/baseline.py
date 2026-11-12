"""Bilinear interpolation baseline: coarse LST -> 30 m, no fine-scale information used."""

from __future__ import annotations

import numpy as np


def _axis_weights(n_coarse: int, factor: int):
    # Fine pixel centres in coarse index units (coarse centres sit at 0, 1, 2, ...)
    u = (np.arange(n_coarse * factor) + 0.5) / factor - 0.5
    i0 = np.floor(u).astype(int)
    w1 = u - i0
    return np.clip(i0, 0, n_coarse - 1), np.clip(i0 + 1, 0, n_coarse - 1), 1 - w1, w1


def bilinear_upsample(coarse: np.ndarray, factor: int) -> np.ndarray:
    """Bilinear upsampling of a (yc, xc) grid with NaN gaps to (yc*factor, xc*factor).

    Normalised weights: a missing neighbour (e.g. a water cell) is left out and the
    remaining weights are rescaled, so valid land cells next to water still get values.
    Beyond the outermost cell centres the edge value is held constant.
    """
    iy0, iy1, wy0, wy1 = _axis_weights(coarse.shape[0], factor)
    ix0, ix1, wx0, wx1 = _axis_weights(coarse.shape[1], factor)
    valid = np.isfinite(coarse)
    vals = np.where(valid, coarse, 0.0).astype(np.float64)

    def interp(a):
        rows = a[iy0] * wy0[:, None] + a[iy1] * wy1[:, None]
        return rows[:, ix0] * wx0 + rows[:, ix1] * wx1

    num = interp(vals * valid)
    den = interp(valid.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-9, num / den, np.nan)
    return out.astype(np.float32)


def predict_bilinear(lst_coarse: np.ndarray, factor: int) -> np.ndarray:
    """(time, yc, xc) -> (time, y, x)."""
    return np.stack([bilinear_upsample(c, factor) for c in lst_coarse])
