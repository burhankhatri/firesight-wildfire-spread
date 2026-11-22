"""Tune, train and evaluate five models plus two baselines.

Run:  python -m src.train
Writes outputs/results.json, outputs/curves.npz and outputs/models/*.
"""
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import OUT_DIR, MODEL_DIR, SEED
from .data import build_table, download
from .features import FEATURES, BASELINE_FEATURES

TUNE_ROWS = 200_000   # rows used for the randomized hyper-parameter search
FIT_ROWS = 1_500_000  # rows used to refit each tuned model


def model_specs():
    return {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            {"logisticregression__C": loguniform(1e-3, 1e2),
             "logisticregression__class_weight": [None, "balanced"]}, 8),
        "Random Forest": (
            RandomForestClassifier(n_jobs=-1, random_state=SEED),
            {"n_estimators": randint(100, 300), "max_depth": randint(8, 24),
             "min_samples_leaf": randint(5, 80), "max_features": ["sqrt", 0.3, 0.5]}, 8),
        "XGBoost": (
            XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="aucpr"),
            {"n_estimators": randint(200, 700), "max_depth": randint(4, 11),
             "learning_rate": loguniform(0.02, 0.25), "subsample": uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": loguniform(1, 50)}, 12),
        "LightGBM": (
            lgb.LGBMClassifier(n_jobs=-1, random_state=SEED, verbose=-1),
            {"n_estimators": randint(200, 800), "num_leaves": randint(15, 256),
             "learning_rate": loguniform(0.02, 0.25), "subsample": uniform(0.6, 0.4),
             "subsample_freq": [1], "colsample_bytree": uniform(0.5, 0.5),
             "min_child_samples": randint(10, 200), "reg_lambda": loguniform(1e-3, 10)}, 12),
        "MLP (Neural Net)": (
            make_pipeline(StandardScaler(), MLPClassifier(early_stopping=True, max_iter=60,
                                                          random_state=SEED)),
            {"mlpclassifier__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
             "mlpclassifier__alpha": loguniform(1e-5, 1e-2),
             "mlpclassifier__learning_rate_init": loguniform(3e-4, 3e-3),
             "mlpclassifier__batch_size": [512, 1024]}, 6),
    }


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": thr, "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
        "auc_pr": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
        "confusion": [int(tn), int(fp), int(fn), int(tp)],
    }


def curves(y, p, n=300):
    prec, rec, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    pick = lambda a, m: a[np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)]
    return {"prec": pick(prec, len(prec)), "rec": pick(rec, len(rec)),
            "fpr": pick(fpr, len(fpr)), "tpr": pick(tpr, len(tpr))}


def predict(model, df, cols, chunk=1_000_000):
    out = np.empty(len(df), np.float32)
    for s in range(0, len(df), chunk):
        out[s:s + chunk] = model.predict_proba(df[cols].iloc[s:s + chunk])[:, 1]
    return out


def main():
    download()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    print("Building tables ...")
    train = build_table("train", mode="balanced")
    evald = build_table("eval", mode="all")
    test = build_table("test", mode="all")
    print(f"train {train.shape}  eval {evald.shape}  test {test.shape}  ({time.time()-t0:.0f}s)")

    tune = train.iloc[rng.choice(len(train), min(TUNE_ROWS, len(train)), replace=False)]
    fit = train.iloc[rng.choice(len(train), min(FIT_ROWS, len(train)), replace=False)]
    cv = GroupKFold(n_splits=3)

    results = {"data": {
        "train_rows": len(train), "train_fire_rate": float(train.y.mean()),
        "eval_rows": len(evald), "eval_fire_rate": float(evald.y.mean()),
        "test_rows": len(test), "test_fire_rate": float(test.y.mean()),
        "n_features": len(FEATURES), "features": FEATURES}, "models": {}}
    curve_store, test_probs = {}, {}

    # Baseline 1: persistence (tomorrow's fire = today's fire)
    for name, p in [("Persistence baseline", test.prev_fire.values)]:
        m = metrics(test.y.values, p, 0.5)
        m.update(cv_auc_pr=None, best_params=None, fit_seconds=0.0)
        results["models"][name] = m
        curve_store[name] = curves(test.y.values, p)

    for name, (est, space, n_iter) in model_specs().items():
        t = time.time()
        print(f"Tuning {name} ...")
        search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="average_precision",
                                    cv=cv, random_state=SEED, n_jobs=1, refit=False)
        search.fit(tune[FEATURES], tune.y, groups=tune.tile)
        model = est.set_params(**search.best_params_).fit(fit[FEATURES], fit.y)
        thr = best_threshold(evald.y.values, predict(model, evald, FEATURES))
        p = predict(model, test, FEATURES)
        m = metrics(test.y.values, p, thr)
        m.update(cv_auc_pr=float(search.best_score_),
                 best_params={k: (v if isinstance(v, (int, float, str, list, tuple, type(None)))
                                  else str(v)) for k, v in search.best_params_.items()},
                 fit_seconds=round(time.time() - t, 1))
        results["models"][name] = m
        curve_store[name] = curves(test.y.values, p)
        test_probs[name] = p.astype(np.float16)
        print(f"  {name}: F1={m['f1']:.3f} AUC-PR={m['auc_pr']:.3f} ({m['fit_seconds']}s)")
        if name in ("LightGBM", "XGBoost", "Logistic Regression"):
            joblib.dump(model, MODEL_DIR / f"{name.split()[0].lower()}.joblib")

    # Ablation: best boosting model on the paper's original 12 inputs only
    best = max((k for k in results["models"] if k != "Persistence baseline"),
               key=lambda k: results["models"][k]["auc_pr"])
    print(f"Ablation with {best} on baseline features ...")
    est = model_specs()[best][0]
    params = results["models"][best]["best_params"]
    abl = est.set_params(**params).fit(fit[BASELINE_FEATURES], fit.y)
    thr = best_threshold(evald.y.values, predict(abl, evald, BASELINE_FEATURES))
    pa = predict(abl, test, BASELINE_FEATURES)
    results["ablation"] = {"model": best, "original_12_features": metrics(test.y.values, pa, thr),
                           "engineered_24_features": results["models"][best]}
    results["best_model"] = best
    results["total_seconds"] = round(time.time() - t0, 1)

    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    np.savez_compressed(OUT_DIR / "curves.npz",
                        **{f"{k}|{c}": v for k, d in curve_store.items() for c, v in d.items()})
    np.savez_compressed(OUT_DIR / "test_probs.npz", tile=test.tile.values, row=test.row.values,
                        col=test.col.values, y=test.y.values,
                        **{k: v for k, v in test_probs.items()})
    print(f"Done in {results['total_seconds']}s. Best model: {best}")


if __name__ == "__main__":
    main()
