"""Exploratory data analysis: summary statistics and figures.

Run:  python -m src.eda   (writes outputs/figures/eda_*.png and outputs/eda_*.csv)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIG_DIR, OUT_DIR, PRETTY, RAW_VARS
from .data import build_table, download, load_tiles, n_samples, open_split
from .features import FEATURES
from .style import apply_style, FIRE, CALM, ACCENT

apply_style()


def missing_and_balance():
    rows = []
    for s in ("train", "eval", "test"):
        g = open_split(s)
        fm, pm = g["FireMask"][:], g["PrevFireMask"][:]
        nan_any = sum(int(np.isnan(g[v][:2000]).sum()) for v in RAW_VARS)
        rows.append({"split": s, "tiles": fm.shape[0], "pixels": fm.size,
                     "next_day_fire_%": 100 * (fm == 1).mean(),
                     "label_missing_%": 100 * (fm == -1).mean(),
                     "prev_fire_%": 100 * (pm == 1).mean(),
                     "prev_missing_%": 100 * (pm == -1).mean(),
                     "nan_values_in_inputs": nan_any})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "eda_splits.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(3)
    ax.bar(x - 0.2, df["next_day_fire_%"], 0.4, color=FIRE, label="Fire next day")
    ax.bar(x + 0.2, df["label_missing_%"], 0.4, color=CALM, label="Missing label (cloud)")
    ax.set_xticks(x, df.split.str.title())
    ax.set_ylabel("% of pixels")
    ax.set_title("Severe class imbalance: about 1 in 75 pixels burns")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "eda_class_balance.png")
    plt.close(fig)
    return df


def sample_tiles(split="test", picks=(5, 77, 301)):
    layers = ["elevation", "vs", "tmmx", "sph", "pr", "pdsi", "NDVI", "erc",
              "population", "th", "PrevFireMask", "FireMask"]
    titles = ["Elevation", "Wind speed", "Max temp", "Humidity", "Rain", "Drought (PDSI)",
              "NDVI", "Energy release", "Population", "Wind dir.", "Fire today", "Fire tomorrow"]
    t = load_tiles(split, 0, max(picks) + 1)
    fig, axes = plt.subplots(len(picks), len(layers), figsize=(17, 1.6 * len(picks) + 0.6))
    fire_cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#1f1f1f", FIRE])
    for r, i in enumerate(picks):
        for c, (v, title) in enumerate(zip(layers, titles)):
            ax = axes[r, c]
            if "Fire" in v:
                ax.imshow(t[v][i], cmap=fire_cmap, vmin=-1, vmax=1)
            else:
                ax.imshow(t[v][i], cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
    fig.suptitle("Example 64 km x 64 km tiles: 11 Earth Engine inputs, today's fire, tomorrow's fire",
                 y=1.0)
    fig.savefig(FIG_DIR / "eda_sample_tiles.png")
    plt.close(fig)


def feature_views(df):
    show = ["vs", "tmmx", "sph", "pdsi", "NDVI", "erc", "elevation", "pr"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    for ax, v in zip(axes.flat, show):
        lo, hi = df[v].quantile([0.01, 0.99])
        bins = np.linspace(lo, hi, 40)
        for lab, col, name in ((0, CALM, "No fire"), (1, FIRE, "Fire")):
            ax.hist(df.loc[df.y == lab, v].clip(lo, hi), bins=bins, density=True, alpha=0.55,
                    color=col, label=name)
        ax.set_title(PRETTY[v]); ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Weather and fuel conditions: fire pixels vs non-fire pixels")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_distributions.png")
    plt.close(fig)

    corr = df[FEATURES + ["y"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    labels = [PRETTY.get(c, "Fire tomorrow") for c in corr.columns]
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True,
                xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.7})
    ax.set_title("Spearman correlation of all 24 features and the target")
    fig.savefig(FIG_DIR / "eda_correlation.png")
    plt.close(fig)
    corr["y"].drop("y").sort_values().to_csv(OUT_DIR / "eda_target_correlation.csv")


def spread_drivers(df):
    """How the chance of burning tomorrow depends on distance, wind and dryness."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    d = df.copy()
    d["dist_bin"] = d.dist_to_fire.clip(0, 20).round()
    g = d.groupby("dist_bin").y.mean()
    axes[0].plot(g.index, 100 * g.values, "o-", color=FIRE)
    axes[0].set_xlabel("Distance to today's fire (km)"); axes[0].set_ylabel("% burning tomorrow")
    axes[0].set_title("Fire spreads to nearby pixels")

    near = d[d.dist_to_fire.between(1, 4)]
    near = near.assign(wind=pd.qcut(near.vs, 3, labels=["Calm", "Moderate", "Windy"]),
                       up=np.where(near.upwind_fire_2 > 0, "Fire upwind", "Fire not upwind"))
    s = near.groupby(["wind", "up"], observed=True).y.mean().unstack()
    x = np.arange(len(s))
    axes[1].bar(x - 0.2, 100 * s["Fire upwind"], 0.4, color=FIRE, label="Fire upwind")
    axes[1].bar(x + 0.2, 100 * s["Fire not upwind"], 0.4, color=CALM, label="Fire not upwind")
    axes[1].set_xticks(x, s.index)
    axes[1].set_title("Being downwind of a fire raises risk\n(pixels 1-4 km from fire)")
    axes[1].legend(frameon=False, loc="lower center", ncol=2, fontsize=8)

    near = near.assign(dry=pd.qcut(near.erc, 8, duplicates="drop"))
    s = near.groupby("dry", observed=True).y.mean()
    axes[2].plot([i.mid for i in s.index], 100 * s.values, "o-", color=ACCENT)
    axes[2].set_xlabel("Energy release component (fuel dryness)")
    axes[2].set_title("Drier fuel burns more\n(pixels 1-4 km from fire)")
    fig.suptitle("What drives next-day spread (balanced training sample, 4,000 tiles; base rate 25%)", y=1.04)
    fig.savefig(FIG_DIR / "eda_spread_drivers.png")
    plt.close(fig)


def main():
    download()
    splits = missing_and_balance()
    print(splits.round(3).to_string(index=False))
    sample_tiles()
    df = build_table("train", mode="balanced", limit=4000)
    desc = df[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.index = [PRETTY.get(i, i) for i in desc.index]
    desc.round(3).to_csv(OUT_DIR / "eda_summary_stats.csv")
    feature_views(df)
    spread_drivers(df)
    with open(OUT_DIR / "eda_meta.json", "w") as fh:
        json.dump({"eda_rows": len(df), "eda_tiles": 4000,
                   "train_tiles": n_samples("train")}, fh)
    print("EDA done")


if __name__ == "__main__":
    main()
