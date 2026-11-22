"""Load the Next Day Wildfire Spread tiles and turn them into pixel tables.

The dataset (Huot et al., 2022) was built in Google Earth Engine. We read the
zarr version published on Hugging Face directly from its zip file, so nothing
has to be unpacked to disk.
"""
import urllib.request

import numpy as np
import pandas as pd
import zarr

from .config import DATA_DIR, RAW_VARS, SEED
from .features import engineer, FEATURES

HF_URL = ("https://huggingface.co/datasets/TheRootOf3/next-day-wildfire-spread/"
          "resolve/main/data/{split}.zarr.zip")
LAYERS = RAW_VARS + ["PrevFireMask", "FireMask"]


def download(splits=("train", "eval", "test")):
    for s in splits:
        path = DATA_DIR / f"{s}.zarr.zip"
        if not path.exists():
            print(f"Downloading {s} split ...")
            urllib.request.urlretrieve(HF_URL.format(split=s), path)


def open_split(split):
    store = zarr.ZipStore(str(DATA_DIR / f"{split}.zarr.zip"), mode="r")
    return zarr.open_group(store, mode="r", path=f"{split}.zarr")


def n_samples(split):
    return open_split(split)["FireMask"].shape[0]


def load_tiles(split, start=0, stop=None):
    """Return {layer: array (N, 64, 64)} for tiles [start, stop)."""
    g = open_split(split)
    return {v: g[v][start:stop].astype(np.float32) for v in LAYERS}


def iter_batches(split, batch=500, limit=None):
    n = n_samples(split) if limit is None else min(limit, n_samples(split))
    for s in range(0, n, batch):
        yield s, load_tiles(split, s, min(s + batch, n))


def _to_rows(feats, label, tile_offset, mask):
    idx = np.nonzero(mask)
    df = pd.DataFrame({k: feats[k][idx] for k in FEATURES})
    df["tile"] = (idx[0] + tile_offset).astype(np.int32)
    df["row"], df["col"] = idx[1].astype(np.int16), idx[2].astype(np.int16)
    df["y"] = (label[idx] == 1).astype(np.int8)
    return df


def build_table(split, mode="all", neg_ratio=3, max_neg_frac=None, limit=None, seed=SEED):
    """Pixel table for a split.

    mode="all": every pixel with a valid label (FireMask != -1); this is the
        realistic, heavily imbalanced setting used for evaluation.
    mode="balanced": all fire pixels plus `neg_ratio` non-fire pixels per fire
        pixel, half of them sampled near the fire (hard negatives) and half
        at random. Used for training.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for off, b in iter_batches(split, limit=limit):
        f = engineer(b)
        y = b["FireMask"]
        valid = y != -1
        if mode == "all":
            mask = valid
            if max_neg_frac is not None:
                keep = rng.random(y.shape) < max_neg_frac
                mask = valid & ((y == 1) | keep)
        else:
            pos = valid & (y == 1)
            neg = valid & (y == 0)
            n_pos = int(pos.sum())
            near = neg & (f["dist_to_fire"] <= 8)
            far = neg & ~near
            mask = pos.copy()
            for pool, share in ((near, 0.5), (far, 0.5)):
                cand = np.flatnonzero(pool)
                k = min(len(cand), int(n_pos * neg_ratio * share))
                if k:
                    mask.flat[rng.choice(cand, k, replace=False)] = True
        parts.append(_to_rows(f, y, off, mask))
    return pd.concat(parts, ignore_index=True)
