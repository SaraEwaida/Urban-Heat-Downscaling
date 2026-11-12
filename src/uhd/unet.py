"""U-Net on image patches.

Same inputs as LightGBM, as image channels; same residual target (Landsat minus bilinear);
same scored pixels and splits.

Leakage rules:
- Patches are cut after the splits are fixed and lie entirely inside one area: training
  patches inside the training area, prediction tiles inside the area being predicted.
  The area boundary follows coarse-cell edges and patches are cut in whole coarse cells.
- Channel normalisation (mean, std) and the target scale are fitted on training pixels
  of the fitting dates only, and stored with the model.
- Epoch selection uses the validation dates in the training area; test splits are only
  predicted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .features import FeatureBuilder

LC_ONEHOT = {"lc_water": [80], "lc_built": [50], "lc_bare": [60], "lc_veg": [10, 20, 30, 40, 90, 95]}
CHANNELS = [
    "lst_bilinear", "lst_cell", "lst_date_mean",
    "ndvi", "ndvi_cell_mean", "ndvi_anom",
    "elevation", "elev_anom",
    "frac_water", "frac_built", "frac_bare", "frac_veg",
    *LC_ONEHOT, "coarse_valid",
]
_NO_NORMALISE = {*LC_ONEHOT, "coarse_valid"}


def _date_index(ds: xr.Dataset, dates: list[str]) -> list[int]:
    names = [str(t)[:10] for t in ds.time.values.astype("datetime64[D]")]
    missing = set(dates) - set(names)
    if missing:
        raise ValueError(f"dates not in the stack: {sorted(missing)}")
    return [names.index(d) for d in dates]


def _raw_channels(fb: FeatureBuilder, t: int) -> dict[str, np.ndarray]:
    m = fb.maps(t)
    lc = fb.ds.landcover.values
    out = {k: m[k] for k in CHANNELS if k in m}
    for k, codes in LC_ONEHOT.items():
        out[k] = np.isin(lc, codes).astype(np.float32)
    out["coarse_valid"] = np.isfinite(m["lst_cell"]).astype(np.float32)
    return out


# ---------------------------------------------------------------- data preparation

class Prepared:
    """Normalised input cubes, residual targets and masks for every date."""

    def __init__(self, ds: xr.Dataset, cfg: dict[str, Any], masks: dict[str, np.ndarray]):
        from .splits import test_area_mask

        p = cfg["unet"]
        self.f = int(ds.attrs["coarse_factor"])
        self.patch = int(p["patch_cells"]) * self.f
        self.boundary = int(np.argmax(test_area_mask(ds, cfg)[0])) or ds.sizes["x"]
        val_t = _date_index(ds, list(p["val_dates"]))
        if set(cfg["splits"]["test_dates"]) & set(p["val_dates"]):
            raise ValueError("val_dates must not overlap test_dates")
        test_t = _date_index(ds, list(cfg["splits"]["test_dates"]))
        self.fit_t = [t for t in range(ds.sizes["time"]) if t not in val_t + test_t]
        self.val_t = val_t

        fb = FeatureBuilder(ds)
        obs = ds.lst_ref.values
        self.train_mask = masks["train"]          # (time, y, x), training area, non-test dates

        # Pass 1: statistics on fitting pixels only
        sums = {k: 0.0 for k in CHANNELS}
        sq = {k: 0.0 for k in CHANNELS}
        n = {k: 0 for k in CHANNELS}
        res_vals = []
        for t in self.fit_t:
            ch = _raw_channels(fb, t)
            m = self.train_mask[t]
            for k in CHANNELS:
                v = ch[k][m]
                v = v[np.isfinite(v)].astype(np.float64)
                sums[k] += v.sum()
                sq[k] += (v ** 2).sum()
                n[k] += v.size
            res_vals.append((obs[t] - ch["lst_bilinear"])[m])
        self.mean = {k: (0.0 if k in _NO_NORMALISE else sums[k] / max(n[k], 1)) for k in CHANNELS}
        self.std = {}
        for k in CHANNELS:
            if k in _NO_NORMALISE:
                self.std[k] = 1.0
            else:
                var = sq[k] / max(n[k], 1) - self.mean[k] ** 2
                self.std[k] = float(np.sqrt(var)) if var > 1e-12 else 1.0
        res = np.concatenate(res_vals) if res_vals else np.zeros(1)
        res = res[np.isfinite(res)]
        self.res_mean, self.res_std = float(res.mean()), float(res.std() or 1.0)

        # Pass 2: normalised cubes (float16 keeps memory low), residual targets
        T, H, W = obs.shape
        self.x = np.zeros((T, len(CHANNELS), H, W), dtype=np.float16)
        self.y = np.full((T, H, W), np.nan, dtype=np.float32)
        self.base = np.full((T, H, W), np.nan, dtype=np.float32)
        for t in range(T):
            ch = _raw_channels(fb, t)
            for i, k in enumerate(CHANNELS):
                v = (ch[k] - self.mean[k]) / self.std[k]
                self.x[t, i] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            self.base[t] = ch["lst_bilinear"]
            self.y[t] = (obs[t] - ch["lst_bilinear"] - self.res_mean) / self.res_std

    def norm_state(self) -> dict[str, Any]:
        return {"channels": CHANNELS, "mean": self.mean, "std": self.std,
                "res_mean": self.res_mean, "res_std": self.res_std}


def tile_starts(lo: int, hi: int, size: int, step: int) -> list[int]:
    """Start offsets of windows of `size` covering [lo, hi), all inside it."""
    if hi - lo < size:
        raise ValueError(f"area of width {hi - lo} is smaller than the patch size {size}")
    starts = list(range(lo, hi - size + 1, step))
    if starts[-1] + size < hi:
        starts.append(hi - size)
    return starts


def train_patches(prep: Prepared, dates: list[int], stride_cells: int) -> list[tuple[int, int, int]]:
    """(t, y0, x0) for patches fully inside the training area that contain training pixels."""
    H, W = prep.y.shape[1:]
    s = prep.patch
    out = []
    for t in dates:
        for y0 in tile_starts(0, H, s, stride_cells * prep.f):
            for x0 in tile_starts(0, prep.boundary, s, stride_cells * prep.f):
                if prep.train_mask[t, y0:y0 + s, x0:x0 + s].any():
                    out.append((t, y0, x0))
    return out


# ---------------------------------------------------------------- model

def build_model(in_ch: int, base: int):
    import torch
    from torch import nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        """3-level U-Net. Input is reflect-padded to a multiple of 8 and cropped back."""

        def __init__(self):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.e1, self.e2, self.e3 = block(in_ch, c[0]), block(c[0], c[1]), block(c[1], c[2])
            self.mid = block(c[2], c[3])
            self.pool = nn.MaxPool2d(2)
            self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), block(c[3], c[2])
            self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), block(c[2], c[1])
            self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), block(c[1], c[0])
            self.out = nn.Conv2d(c[0], 1, 1)

        def forward(self, x):
            h, w = x.shape[-2:]
            ph, pw = (-h) % 8, (-w) % 8
            x = nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="reflect")
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(m), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            y = self.out(d1)[:, 0]
            return y[:, ph // 2: ph // 2 + h, pw // 2: pw // 2 + w]

    return UNet()


def _batch(prep: Prepared, idx, rng=None):
    import torch

    s = prep.patch
    xs, ys, ms = [], [], []
    for t, y0, x0 in idx:
        x = prep.x[t, :, y0:y0 + s, x0:x0 + s].astype(np.float32)
        y = prep.y[t, y0:y0 + s, x0:x0 + s]
        m = prep.train_mask[t, y0:y0 + s, x0:x0 + s] & np.isfinite(y)
        if rng is not None:                      # flips and 90-degree rotations
            k = int(rng.integers(4))
            x, y, m = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(m, k)
            if rng.random() < 0.5:
                x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.nan_to_num(np.ascontiguousarray(y)))
        ms.append(np.ascontiguousarray(m))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ms))


def _masked_mse(pred, y, m):
    m = m.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def train(ds: xr.Dataset, cfg: dict[str, Any], masks: dict[str, np.ndarray], verbose: bool = True):
    import torch

    p = cfg["unet"]
    torch.manual_seed(p.get("seed", 0))
    rng = np.random.default_rng(p.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prep = Prepared(ds, cfg, masks)
    fit_idx = train_patches(prep, prep.fit_t, p["train_stride_cells"])
    val_idx = train_patches(prep, prep.val_t, p["patch_cells"])
    if not fit_idx or not val_idx:
        raise RuntimeError("No training or validation patches; check splits and patch_cells")

    model = build_model(len(CHANNELS), p["base_channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=1e-4)
    bs = p["batch_size"]
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    history = []
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        order = rng.permutation(len(fit_idx))
        tr_loss = []
        for i in range(0, len(order), bs):
            x, y, m = _batch(prep, [fit_idx[j] for j in order[i:i + bs]], rng)
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = _masked_mse(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss.append(loss.item())

        model.eval()
        num = den = 0.0
        with torch.no_grad():
            for i in range(0, len(val_idx), bs):
                x, y, m = _batch(prep, val_idx[i:i + bs])
                x, y, m = x.to(device), y.to(device), m.to(device)
                num += (((model(x) - y) ** 2) * m.float()).sum().item()
                den += m.sum().item()
        val = num / max(den, 1.0)
        rmse_c = float(np.sqrt(val)) * prep.res_std
        history.append({"epoch": epoch, "train_loss": float(np.mean(tr_loss)), "val_rmse_c": rmse_c})
        if verbose:
            print(f"  epoch {epoch:3d}  train loss {np.mean(tr_loss):.3f}  val RMSE {rmse_c:.2f} C", flush=True)
        if val < best - 1e-4:
            best, best_epoch, wait = val, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= p["patience"]:
                break

    model.load_state_dict(best_state)
    info = {"device": device, "fit_patches": len(fit_idx), "val_patches": len(val_idx),
            "best_epoch": best_epoch, "epochs_run": len(history), "history": history}
    return model.cpu(), prep, info


def predict(model, prep: Prepared, batch_size: int = 16) -> np.ndarray:
    """(time, y, x) predictions in degC. Each area is tiled separately, so no tile mixes areas.

    Overlapping tiles are averaged. Pixels without a bilinear value stay NaN.
    """
    import torch

    T, H, W = prep.y.shape
    s = prep.patch
    out = np.zeros((T, H, W), dtype=np.float64)
    cnt = np.zeros((T, H, W), dtype=np.float64)
    areas = [(0, prep.boundary)] + ([(prep.boundary, W)] if prep.boundary < W else [])
    tiles = [(t, y0, x0) for t in range(T) for lo, hi in areas
             for y0 in tile_starts(0, H, s, s) for x0 in tile_starts(lo, hi, s, s)]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            idx = tiles[i:i + batch_size]
            x = torch.from_numpy(np.stack([prep.x[t, :, y0:y0 + s, x0:x0 + s].astype(np.float32)
                                           for t, y0, x0 in idx]))
            r = model(x).numpy()
            for (t, y0, x0), ri in zip(idx, r):
                out[t, y0:y0 + s, x0:x0 + s] += ri
                cnt[t, y0:y0 + s, x0:x0 + s] += 1
    res = out / np.maximum(cnt, 1) * prep.res_std + prep.res_mean
    return (prep.base + res).astype(np.float32)


def save(model, prep: Prepared, info: dict[str, Any], cfg: dict[str, Any], path: str | Path) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "norm": prep.norm_state(), "config": cfg["unet"],
                "info": {k: v for k, v in info.items() if k != "history"}}, path)
    return path
