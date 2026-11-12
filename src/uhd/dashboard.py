"""Data and figures for the Streamlit dashboard (app/dashboard.py is the thin UI layer).

Maps are linked: all four panels share x and y axes, so zooming or panning one moves
all of them. Temperature panels share one colour scale; the error panel has its own
diverging scale centred on zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .baseline import predict_bilinear
from .metrics import scores
from .splits import eval_mask, test_area_mask

METHOD_LABELS = {"bilinear": "Bilinear", "lightgbm": "LightGBM", "unet": "U-Net"}
TEMP_SCALE = "Inferno"     # sequential, perceptually uniform, shared by every temperature map
ERROR_SCALE = "RdBu_r"     # diverging: blue = too cool, white = 0, red = too warm
MASK_BG = "#bdbdbd"


def load(cfg: dict[str, Any]) -> tuple[xr.Dataset, dict[str, np.ndarray]]:
    """The stack and every available prediction, (time, y, x) in degC."""
    ds = xr.open_zarr(cfg["paths"]["stack"]).load()
    preds = {"bilinear": predict_bilinear(ds.lst_coarse.values, int(ds.attrs["coarse_factor"]))}
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    for name in ("lightgbm", "unet"):
        path = pred_dir / f"{name}.zarr"
        if path.exists():
            p = xr.open_zarr(path)[name].values
            if p.shape != preds["bilinear"].shape:
                raise ValueError(f"{path} does not match the stack grid; rerun `uhd {name}`")
            preds[name] = p
    return ds, preds


def date_roles(ds: xr.Dataset, cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """(date, role) with role one of: training, validation, held-out."""
    val = set(cfg.get("lightgbm", {}).get("val_dates", [])) | set(cfg.get("unet", {}).get("val_dates", []))
    out = []
    for d in ds.time.values.astype("datetime64[D]").astype(str):
        role = "held-out" if d in cfg["splits"]["test_dates"] else "validation" if d in val else "training"
        out.append((str(d), role))
    return out


def scene_metrics(ds: xr.Dataset, cfg: dict[str, Any], pred_t: np.ndarray, t: int) -> list[dict[str, Any]]:
    """Scores on one date, separately for the training area and the held-out area."""
    m = eval_mask(ds)[t]
    area = test_area_mask(ds, cfg)
    obs = ds.lst_ref.values[t]
    rows = []
    for name, a in (("Training area", ~area), ("Held-out area", area)):
        sel = m & a & np.isfinite(pred_t)
        rows.append({"area": name, **scores(pred_t[sel], obs[sel])})
    return rows


def _km(ds: xr.Dataset):
    x0, y0 = float(ds.x.min()) - 15, float(ds.y.min()) - 15
    k = lambda v, o: (np.asarray(v) - o) / 1000.0  # noqa: E731
    return k(ds.x.values, x0), k(ds.y.values, y0), k(ds.xc.values, x0), k(ds.yc.values, y0)


def temperature_range(*arrays: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def error_range(err: np.ndarray, hi: float = 98) -> float:
    e = np.abs(err[np.isfinite(err)])
    return float(max(np.percentile(e, hi), 0.5)) if e.size else 1.0


def build_figure(ds: xr.Dataset, cfg: dict[str, Any], pred_t: np.ndarray, t: int, method: str,
                 stride: int = 2):
    """2 x 2 linked maps: coarse input, prediction, observed, signed error."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x, y, xc, yc = _km(ds)
    coarse = ds.lst_coarse.values[t]
    obs = np.where(ds.valid_ref.values[t], ds.lst_ref.values[t], np.nan)
    err = np.where(eval_mask(ds)[t], pred_t - obs, np.nan)
    tmin, tmax = temperature_range(coarse, pred_t, obs)
    emax = error_range(err)

    s = slice(None, None, stride)
    label = METHOD_LABELS.get(method, method)
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes="all", shared_yaxes="all", horizontal_spacing=0.04, vertical_spacing=0.08,
        subplot_titles=("Coarse input: MODIS, 930 m", f"Prediction: {label}, 30 m",
                        "Observed: Landsat, 30 m", f"Error: {label} minus observed"),
    )
    hover_t = "x %{x:.1f} km, y %{y:.1f} km<br>%{z:.1f} °C<extra></extra>"
    hover_e = "x %{x:.1f} km, y %{y:.1f} km<br>%{z:+.1f} °C<extra></extra>"
    fig.add_trace(go.Heatmap(x=xc, y=yc, z=coarse, coloraxis="coloraxis", hovertemplate=hover_t), 1, 1)
    fig.add_trace(go.Heatmap(x=x[s], y=y[s], z=pred_t[s, s], coloraxis="coloraxis", hovertemplate=hover_t), 1, 2)
    fig.add_trace(go.Heatmap(x=x[s], y=y[s], z=obs[s, s], coloraxis="coloraxis", hovertemplate=hover_t), 2, 1)
    fig.add_trace(go.Heatmap(x=x[s], y=y[s], z=err[s, s], coloraxis="coloraxis2", hovertemplate=hover_e), 2, 2)

    # Held-out area boundary on every panel
    col = int(np.argmax(test_area_mask(ds, cfg)[0]))
    if col > 0:
        xb = float(x[col] - 0.015)
        for i in ("", "2", "3", "4"):
            for line in (dict(color="black", width=4), dict(color="white", width=2, dash="dash")):
                fig.add_shape(type="line", x0=xb, x1=xb, y0=0, y1=1, xref=f"x{i}", yref=f"y{i} domain",
                              line=line, layer="above")
        fig.add_annotation(x=xb, y=float(y.max()), text="held-out area →", xanchor="left", yanchor="top",
                           showarrow=False, font=dict(color="white", size=11), row=1, col=1)

    fig.update_layout(
        coloraxis=dict(colorscale=TEMP_SCALE, cmin=tmin, cmax=tmax,
                       colorbar=dict(title="°C", x=1.0, y=0.78, len=0.45)),
        coloraxis2=dict(colorscale=ERROR_SCALE, cmin=-emax, cmax=emax, cmid=0,
                        colorbar=dict(title="Error °C", x=1.0, y=0.22, len=0.45)),
        height=760, margin=dict(l=40, r=60, t=40, b=30), plot_bgcolor=MASK_BG,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, constrain="domain")
    fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_xaxes(title_text="Easting (km)", row=2)
    fig.update_yaxes(title_text="Northing (km)", col=1)
    return fig
