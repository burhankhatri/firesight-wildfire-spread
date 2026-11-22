"""Wildfire digital twin: re-run the trained model under what-if scenarios.

For a few real test fires we change the wind direction, wind speed and a
climate-warming scenario, recompute every feature from the modified inputs
and predict tomorrow's fire again. Output:
  outputs/twin.json          compact scenario grid used by the website
  outputs/figures/twin_*.png figures for the paper and slides

Run:  python -m src.twin
"""
import base64
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIG_DIR, MODEL_DIR, OUT_DIR
from .data import load_tiles
from .features import FEATURES, engineer
from .style import apply_style

apply_style()
ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]   # degrees added to wind direction
SPEEDS = [0.5, 1.0, 1.5, 2.0]                      # wind speed multipliers
# Climate scenarios: (+deg C on temperatures, +ERC units, humidity multiplier, PDSI shift).
# Changes are additive and moderate so inputs stay inside the training range.
CLIMATE = {"Today": (0, 0.0, 1.0, 0.0), "+2 C warming": (2, 4.0, 0.94, -0.5),
           "+4 C warming": (4, 8.0, 0.88, -1.0)}
# Logistic Regression drives the twin: it is smooth and monotone, so it extrapolates
# sensibly to warmer-than-observed conditions, where tree ensembles stay flat or jump
# (see paper, Sec. Digital twin). Its test AUC-PR (0.227) is on par with XGBoost (0.228).
TWIN_MODEL = "Logistic Regression"


def scenario(tiles, i, rot=0, speed=1.0, climate="Today"):
    dT, erc_d, sph_m, pdsi_d = CLIMATE[climate]
    b = {k: v[i:i + 1].copy() for k, v in tiles.items()}
    b["th"] = (b["th"] + rot) % 360
    b["vs"] = b["vs"] * speed
    b["tmmx"] = b["tmmx"] + dT
    b["tmmn"] = b["tmmn"] + dT
    b["erc"] = b["erc"] + erc_d
    b["sph"] = b["sph"] * sph_m
    b["pdsi"] = b["pdsi"] + pdsi_d
    return b


def predict_map(model, b):
    f = engineer(b)
    X = np.stack([f[k].ravel() for k in FEATURES], 1)
    import pandas as pd
    return model.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1].reshape(64, 64)


def pick_tiles(n=4):
    """Growing fires, skipping overlapping crops of the same fire (the dataset
    contains several shifted 64 km tiles of large fires)."""
    t = load_tiles("test")
    prev = (t["PrevFireMask"] == 1).sum((1, 2))
    nxt = (t["FireMask"] == 1).sum((1, 2))
    ok = np.flatnonzero((prev >= 15) & (prev <= 150) & (nxt > 1.3 * prev))
    picks, sigs = [], []
    for k in ok[np.argsort(-(nxt[ok] - prev[ok]))]:
        sig = (float(t["tmmx"][k].mean()), float(t["elevation"][k].mean()))
        if all(abs(sig[0] - a) > 1 or abs(sig[1] - b) > 150 for a, b in sigs):
            picks.append(int(k)); sigs.append(sig)
        if len(picks) == n:
            break
    return t, picks


def main():
    res = json.load(open(OUT_DIR / "results.json"))
    name = TWIN_MODEL
    model = joblib.load(MODEL_DIR / "logistic.joblib")
    thr = res["models"][name]["threshold"]
    t, picks = pick_tiles()
    enc = lambda a: base64.b64encode(np.clip(a * 255, 0, 255).astype(np.uint8).tobytes()).decode()
    out = {"model": name, "threshold": thr, "rotations": ROTATIONS, "speeds": SPEEDS,
           "climates": list(CLIMATE), "tiles": []}
    for k in picks:
        entry = {"id": k, "prev": enc((t["PrevFireMask"][k] == 1).astype(float)),
                 "actual": enc((t["FireMask"][k] == 1).astype(float)),
                 "elev": enc((t["elevation"][k] - t["elevation"][k].min()) /
                             max(np.ptp(t["elevation"][k]), 1)),
                 "wind_dir": float(np.mean(t["th"][k])), "wind_speed": float(np.mean(t["vs"][k])),
                 "grid": {}, "area": {}}
        for c in CLIMATE:
            for s in SPEEDS:
                for r in ROTATIONS:
                    p = predict_map(model, scenario(t, k, r, s, c))
                    key = f"{c}|{s}|{r}"
                    entry["grid"][key] = enc(p)
                    entry["area"][key] = int((p >= thr).sum())
        out["tiles"].append(entry)
        print(f"tile {k}: done")
    json.dump(out, open(OUT_DIR / "twin.json", "w"))

    # Figures: wind rotation and climate scenarios for the first tile
    k = picks[0]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for ax, r in zip(axes.flat, ROTATIONS):
        p = predict_map(model, scenario(t, k, r))
        ax.imshow(p, cmap="inferno", vmin=0, vmax=1)
        ax.contour(t["PrevFireMask"][k] == 1, levels=[0.5], colors="cyan", linewidths=0.8)
        d = (np.mean(t["th"][k]) + r) % 360
        ax.set_title(f"Wind from {d:.0f} deg | {int((p >= thr).sum())} km2", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"Digital twin: same fire, wind rotated ({name}; cyan = fire today)")
    fig.savefig(FIG_DIR / "twin_wind.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, c in zip(axes, CLIMATE):
        p = predict_map(model, scenario(t, k, 0, 1.0, c))
        ax.imshow(p, cmap="inferno", vmin=0, vmax=1)
        ax.contour(t["PrevFireMask"][k] == 1, levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(f"{c}: {int((p >= thr).sum())} km2 at risk", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Digital twin: climate-warming scenarios on the same fire")
    fig.savefig(FIG_DIR / "twin_climate.png"); plt.close(fig)

    # Average climate effect over all picked tiles
    summary = {c: float(np.mean([e["area"][f"{c}|1.0|0"] for e in out["tiles"]])) for c in CLIMATE}
    json.dump(summary, open(OUT_DIR / "twin_summary.json", "w"), indent=2)
    print("Twin summary (mean predicted burn area, km2):", summary)


if __name__ == "__main__":
    main()
