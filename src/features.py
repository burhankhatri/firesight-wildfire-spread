"""Spatial and physics-informed feature engineering.

Every input is a batch of 64x64 tiles with shape (N, 64, 64). Features are
computed per tile and returned with the same shape, so each pixel becomes one
row of a tabular dataset.
"""
import numpy as np
from scipy import ndimage

from .config import RAW_VARS, CLIP

ENGINEERED = [
    "prev_fire", "fire_density_3", "fire_density_7", "fire_density_15",
    "dist_to_fire", "upwind_fire_2", "upwind_fire_5", "wind_u", "wind_v",
    "temp_range", "slope", "dryness", "fire_front_wind",
]
FEATURES = RAW_VARS + ENGINEERED
BASELINE_FEATURES = RAW_VARS + ["prev_fire"]  # what the original paper feeds its models


def clean(batch):
    """Clip each raw layer to a physically plausible range and fill NaNs with the median."""
    out = {}
    for v in RAW_VARS:
        a = batch[v].astype(np.float32)
        if np.isnan(a).any():
            a = np.where(np.isnan(a), np.nanmedian(a), a)
        lo, hi = CLIP[v]
        out[v] = np.clip(a, lo, hi)
    return out


def _distance_to_fire(prev, cap=32.0):
    d = np.full(prev.shape, cap, np.float32)
    for i in range(prev.shape[0]):
        if prev[i].any():
            d[i] = np.minimum(ndimage.distance_transform_edt(~prev[i]), cap)
    return d


def _sample_upwind(field, th_deg, k):
    """Value of `field` k pixels upwind of each pixel.

    `th` is the meteorological wind direction: the direction the wind comes
    FROM, 0 = north, clockwise. Row 0 is the north edge of the tile.
    """
    n, h, w = field.shape
    rad = np.deg2rad(th_deg)
    rows = np.arange(h)[None, :, None] - k * np.cos(rad)
    cols = np.arange(w)[None, None, :] + k * np.sin(rad)
    r = np.rint(rows).astype(int)
    c = np.rint(cols).astype(int)
    inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    r, c = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
    idx = np.arange(n)[:, None, None]
    return np.where(inside, field[idx, r, c], 0.0).astype(np.float32)


def engineer(batch):
    """Return a dict of all FEATURES (cleaned raw layers + engineered ones)."""
    f = clean(batch)
    prev = batch["PrevFireMask"] == 1
    f["prev_fire"] = prev.astype(np.float32)
    for k in (3, 7, 15):
        f[f"fire_density_{k}"] = ndimage.uniform_filter(
            f["prev_fire"], size=(1, k, k), mode="constant").astype(np.float32)
    f["dist_to_fire"] = _distance_to_fire(prev)
    for k in (2, 5):
        f[f"upwind_fire_{k}"] = _sample_upwind(f["fire_density_7"], f["th"], k)
    rad = np.deg2rad(f["th"])
    # Components of the vector the wind blows TOWARD (u: east +, v: north +).
    f["wind_u"] = (-f["vs"] * np.sin(rad)).astype(np.float32)
    f["wind_v"] = (-f["vs"] * np.cos(rad)).astype(np.float32)
    f["temp_range"] = f["tmmx"] - f["tmmn"]
    gy, gx = np.gradient(f["elevation"], axis=(1, 2))
    f["slope"] = np.degrees(np.arctan(np.hypot(gx, gy) / 1000.0)).astype(np.float32)
    f["dryness"] = (np.clip(f["NDVI"], 0, None) / 10000.0) * (f["erc"] / 100.0)
    f["fire_front_wind"] = f["vs"] * f["upwind_fire_2"]
    return f
