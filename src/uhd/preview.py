"""Side-by-side view: coarse input vs 30 m reference, one shared degC scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

TEMP_CMAP = "inferno"       # perceptually uniform, monotonic lightness
MASK_COLOR = "#bdbdbd"      # masked / invalid pixels


def shared_limits(*arrays: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if vals.size == 0:
        raise ValueError("No valid temperature pixels to plot")
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def _extent_km(x: np.ndarray, y: np.ndarray, x0: float, y0: float):
    dx, dy = abs(x[1] - x[0]) if x.size > 1 else 0, abs(y[1] - y[0]) if y.size > 1 else 0
    return ((x.min() - dx / 2 - x0) / 1000, (x.max() + dx / 2 - x0) / 1000,
            (y.min() - dy / 2 - y0) / 1000, (y.max() + dy / 2 - y0) / 1000)


def plot_input_vs_reference(ds: xr.Dataset, out: str | Path) -> Path:
    coarse = ds["lst_coarse"].values
    ref = ds["lst_ref"].values
    vmin, vmax = shared_limits(coarse, ref)

    cmap = plt.get_cmap(TEMP_CMAP).copy()
    cmap.set_bad(MASK_COLOR)

    x0, y0 = float(ds.x.min()), float(ds.y.min())
    ext_c = _extent_km(ds.xc.values, ds.yc.values, x0, y0)
    ext_f = _extent_km(ds.x.values, ds.y.values, x0, y0)

    ls, md = ds.attrs["landsat"], ds.attrs["modis"]
    utc = md.get("approx_view_time_utc_h")
    md_time = f"{int(utc):02d}:{int(round((utc % 1) * 60)) % 60:02d}Z" if utc is not None else "time n/a"
    ls_time = ls["datetime_utc"][11:16] + "Z"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.1, top=0.82, wspace=0.08)
    titles = [
        f"Input: MODIS LST, {ds['lst_coarse'].attrs['source_resolution_m']:.0f} m\n{md['date']} ~{md_time}",
        f"Reference: Landsat surface temperature, 30 m grid (100 m thermal)\n{ls['datetime_utc'][:10]} {ls_time}",
    ]
    for ax, arr, ext, title in zip(axes, [coarse, ref], [ext_c, ext_f], titles):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Easting (km)")
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_color("#999999")
    axes[0].set_ylabel("Northing (km)")
    cax = fig.add_axes([0.9, 0.18, 0.015, 0.56])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Land surface temperature (°C)")
    city = ds.attrs["city"].replace("_", " ").title()
    fig.suptitle(f"{city}: coarse input vs 30 m reference  (grey = masked)", fontsize=12, x=0.06, y=0.98, ha="left")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
