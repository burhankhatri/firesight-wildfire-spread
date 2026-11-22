"""Turn training outputs into tables and figures.

Run:  python -m src.report   (after python -m src.train)
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import FIG_DIR, MODEL_DIR, OUT_DIR, PRETTY, SEED
from .data import build_table, load_tiles
from .features import FEATURES, engineer
from .style import apply_style, MODEL_COLORS, FIRE, CALM

apply_style()
ORDER = ["Persistence baseline", "Logistic Regression", "Random Forest", "XGBoost",
         "LightGBM", "MLP (Neural Net)"]


def load():
    res = json.load(open(OUT_DIR / "results.json"))
    cur = np.load(OUT_DIR / "curves.npz")
    curves = {}
    for k in cur.files:
        m, c = k.split("|")
        curves.setdefault(m, {})[c] = cur[k]
    return res, curves


def table(res):
    rows = []
    for m in ORDER:
        r = res["models"][m]
        rows.append({"Model": m, "Accuracy": r["accuracy"], "Precision": r["precision"],
                     "Recall": r["recall"], "F1": r["f1"], "AUC-PR": r["auc_pr"],
                     "ROC-AUC": r["roc_auc"],
                     "CV AUC-PR": r["cv_auc_pr"] if r["cv_auc_pr"] is not None else np.nan,
                     "Threshold": r["threshold"], "Time (s)": r["fit_seconds"]})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT_DIR / "results_table.csv", index=False)
    return df


def fig_metrics(df):
    mets = ["Precision", "Recall", "F1", "AUC-PR"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.13
    for i, m in enumerate(ORDER):
        vals = df.set_index("Model").loc[m, mets].values.astype(float)
        ax.bar(np.arange(4) + (i - 2.5) * w, vals, w, color=MODEL_COLORS[m], label=m)
    ax.set_xticks(np.arange(4), mets)
    ax.set_ylabel("Score on test set")
    ax.set_title("Model comparison on 6.7 million unseen test pixels")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.set_ylim(0, max(0.6, df[mets].values.max() + 0.12))
    fig.savefig(FIG_DIR / "results_metrics.png")
    plt.close(fig)


def fig_curves(curves, res):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4))
    for m in ORDER:
        c = curves[m]
        r = res["models"][m]
        a.plot(c["rec"], c["prec"], color=MODEL_COLORS[m], lw=2,
               label=f"{m} (AP={r['auc_pr']:.3f})")
        b.plot(c["fpr"], c["tpr"], color=MODEL_COLORS[m], lw=2,
               label=f"{m} (AUC={r['roc_auc']:.3f})")
    base = res["data"]["test_fire_rate"]
    a.axhline(base, ls=":", color="k", lw=1)
    a.text(0.98, base + 0.01, f"random = {base:.3f}", ha="right", fontsize=8)
    a.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves", ylim=(0, 1))
    b.plot([0, 1], [0, 1], ":", color="k", lw=1)
    b.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    a.legend(frameon=False, fontsize=7.5); b.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIG_DIR / "results_curves.png")
    plt.close(fig)


def fig_confusion(res):
    best = res["best_model"]
    tn, fp, fn, tp = res["models"][best]["confusion"]
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Oranges", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center",
                    va="center", fontsize=9)
    ax.set_xticks([0, 1], ["No fire", "Fire"]); ax.set_yticks([0, 1], ["No fire", "Fire"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
    ax.set_title(f"{best}: confusion matrix")
    fig.savefig(FIG_DIR / "results_confusion.png")
    plt.close(fig)


def fig_ablation(res):
    ab = res["ablation"]
    mets = ["precision", "recall", "f1", "auc_pr"]
    o, e = ab["original_12_features"], ab["engineered_24_features"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(4)
    ax.bar(x - 0.2, [o[m] for m in mets], 0.4, color=CALM, label="Original 12 inputs")
    ax.bar(x + 0.2, [e[m] for m in mets], 0.4, color=FIRE, label="+ 12 engineered features")
    for i, m in enumerate(mets):
        ax.text(i + 0.2, e[m] + 0.01, f"+{100 * (e[m] - o[m]) / max(o[m], 1e-9):.0f}%",
                ha="center", fontsize=8)
    ax.set_xticks(x, ["Precision", "Recall", "F1", "AUC-PR"])
    ax.set_title(f"Feature ablation ({ab['model']})")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "results_ablation.png")
    plt.close(fig)


def explain_model(res):
    """SHAP on the best saved gradient-boosting model (fast TreeExplainer)."""
    cands = [m for m in ("LightGBM", "XGBoost") if (MODEL_DIR / f"{m.lower()}.joblib").exists()]
    name = max(cands, key=lambda m: res["models"][m]["auc_pr"])
    model = joblib.load(MODEL_DIR / f"{name.lower()}.joblib")
    df = build_table("test", mode="balanced", limit=600, neg_ratio=3)
    X = df[FEATURES].sample(4000, random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    Xp = X.rename(columns=PRETTY)
    shap.summary_plot(sv, Xp, show=False, max_display=15, plot_size=(8, 6))
    plt.title(f"SHAP: what drives {name}'s prediction")
    plt.savefig(FIG_DIR / "results_shap.png"); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=Xp.columns).sort_values()
    imp.to_csv(OUT_DIR / "shap_importance.csv")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    top = imp.tail(15)
    ax.barh(top.index, top.values, color=[FIRE if c in [PRETTY[f] for f in FEATURES[11:]]
                                          else CALM for c in top.index])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature importance (orange = engineered)")
    fig.savefig(FIG_DIR / "results_shap_bar.png"); plt.close(fig)
    return name, imp


def fig_maps(res, n=5):
    tp = np.load(OUT_DIR / "test_probs.npz")
    best = res["best_model"]
    thr = res["models"][best]["threshold"]
    tile, row, col = tp["tile"], tp["row"], tp["col"]
    fire_per_tile = np.bincount(tile, weights=tp["y"])
    t = load_tiles("test")
    picks, sigs = [], []  # skip overlapping crops of the same fire
    for k in np.argsort(fire_per_tile)[::-1][3:]:
        sig = (float(t["tmmx"][k].mean()), float(t["elevation"][k].mean()))
        if all(abs(sig[0] - a) > 1 or abs(sig[1] - b) > 150 for a, b in sigs):
            picks.append(int(k)); sigs.append(sig)
        if len(picks) == n:
            break
    cols = ["Fire today", "Actual fire tomorrow", f"{best} probability", f"{best} prediction",
            "Logistic Reg. probability"]
    fig, axes = plt.subplots(n, 5, figsize=(12, 2.5 * n))
    fire_cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#1f1f1f", FIRE])
    for r, k in enumerate(picks):
        sel = tile == k
        prob = np.full((64, 64), np.nan); lr = np.full((64, 64), np.nan)
        prob[row[sel], col[sel]] = tp[best][sel]
        lr[row[sel], col[sel]] = tp["Logistic Regression"][sel]
        imgs = [(t["PrevFireMask"][k], fire_cmap, -1, 1), (t["FireMask"][k], fire_cmap, -1, 1),
                (prob, "inferno", 0, 1), ((prob >= thr).astype(float), "Oranges", 0, 1),
                (lr, "inferno", 0, 1)]
        for c, (im, cm, lo, hi) in enumerate(imgs):
            axes[r, c].imshow(im, cmap=cm, vmin=lo, vmax=hi)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9)
    fig.suptitle("Next-day fire maps on unseen test fires (grey = cloud / missing)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "results_maps.png")
    plt.close(fig)


def main():
    res, curves = load()
    df = table(res)
    print(df.round(3).to_string(index=False))
    fig_metrics(df); fig_curves(curves, res); fig_confusion(res); fig_ablation(res)
    name, imp = explain_model(res)
    print("SHAP model:", name); print(imp.tail(10).round(4))
    fig_maps(res)
    print("Report figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
