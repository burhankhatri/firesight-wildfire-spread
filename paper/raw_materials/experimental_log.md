# Experimental log (FireSight)

Machine: Apple Silicon laptop, 10 CPU cores, 16 GB RAM, no GPU. Total training run: 2,245 s.
Code: `src/train.py`; all numbers below are copied from `outputs/results.json`.

## Data
| Split | Tiles | Pixels | Next-day fire % | Missing label % | Rows used |
|---|---|---|---|---|---|
| train | 14,979 | 61.35 M | 1.07 | 2.35 | 2,633,062 (balanced, 25% fire) |
| eval | 1,877 | 7.69 M | 1.33 | 2.44 | 7,500,385 (all valid pixels) |
| test | 1,689 | 6.92 M | 1.22 | 2.75 | 6,727,699 (all valid, 1.25% fire) |
Input layers contain no NaN values; outliers clipped (e.g. ERC to [0,110], NDVI to [-2000,10000]).
Found: the dataset contains overlapping 64 km crops of the same large fire (test: 1,689 tiles,
1,648 exact-unique; more near-duplicates differ only by a shift).

## Hyper-parameter search (RandomizedSearchCV, 3-fold GroupKFold by tile, 200k rows, AP scoring)
- LR: C=0.985, class_weight=None (8 iters)
- RF: n_estimators=157, max_depth=23, min_samples_leaf=37, max_features=sqrt (8 iters)
- XGBoost: n_estimators=391, max_depth=9, lr=0.124, subsample=0.997, colsample=0.528, min_child_weight=2.29 (12)
- LightGBM: n_estimators=585, num_leaves=206, lr=0.164, subsample=0.847, colsample=0.985, min_child_samples=47, lambda=9.31 (12)
- MLP: (128,64,32), alpha=1.33e-4, lr_init=1.62e-3, batch=512, early stopping, max_iter=60 (6); ConvergenceWarnings seen.
Refit on 1.5 M balanced rows. Threshold = argmax F1 on eval split.

## Test results (6,727,699 pixels)
| Model | Acc | Prec | Rec | F1 | AUC-PR | ROC-AUC | CV AUC-PR | Thr | Time s |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | 0.985 | 0.357 | 0.273 | 0.309 | 0.107 | 0.633 | - | 0.5 | 0 |
| Logistic Regression | 0.978 | 0.272 | 0.453 | 0.340 | 0.227 | 0.901 | 0.640 | 0.443 | 14.9 |
| Random Forest | 0.976 | 0.264 | 0.491 | 0.343 | 0.258 | 0.875 | 0.745 | 0.459 | 618.3 |
| XGBoost | 0.977 | 0.258 | 0.445 | 0.326 | 0.228 | 0.866 | 0.831 | 0.416 | 156.7 |
| LightGBM | 0.975 | 0.237 | 0.465 | 0.314 | 0.215 | 0.845 | 0.848 | 0.327 | 959.0 |
| MLP | 0.974 | 0.198 | 0.362 | 0.256 | 0.116 | 0.790 | 0.742 | 0.568 | 259.4 |
Confusion (TN, FP, FN, TP) RF: 6,527,682 / 115,686 / 42,895 / 41,436.

Original paper (Huot et al., Table II, AUC-PR / precision / recall, %): Neural network 28.4/33.6/43.1;
Random forest 22.5/26.3/46.9; Logistic regression 19.8/32.5/35.3; Persistence 11.5/35.7/27.3.
(Their evaluation uses random 32x32 km crops, ours full 64x64 tiles; persistence precision/recall match exactly.)

## Ablation (Random Forest, same hyper-parameters)
| Features | Prec | Rec | F1 | AUC-PR |
|---|---|---|---|---|
| Original 12 | 0.368 | 0.270 | 0.311 | 0.163 |
| 24 (with engineered) | 0.264 | 0.491 | 0.343 | 0.258 (+59%) |

## SHAP (XGBoost, 4,000 test pixels), mean |SHAP|
distance to fire 0.870; fire density 7 km 0.206; fire density 3 km 0.184; elevation 0.170; ERC 0.169;
NDVI 0.147; population 0.147; humidity 0.146; min temp 0.142; max temp 0.141.
Spearman correlation with target (balanced sample): fire_density_7 0.51, fire_density_3 0.49,
upwind_fire_2 0.48, fire_front_wind 0.47, fire_density_15 0.43, prev_fire 0.38.

## Digital twin (Logistic Regression, 4 distinct growing test fires; predicted km2 above threshold)
Distinct = next-day masks not shifted copies (FFT cross-correlation overlap < 0.6 of the larger fire).
| Fire | Real | Wind x2 | Wind rotated min-max | +2 C | +4 C |
|---|---|---|---|---|---|
| A (438) | 248 | 243 | 242-283 | 261 | 290 |
| B (399) | 301 | 1082 | 253-355 | 335 | 380 |
| C (529) | 110 | 167 | 95-133 | 121 | 134 |
| D (369) | 45 | 109 | 43-57 | 54 | 67 |
Mean: today 176, +2 C 193 (+9.5%), +4 C 218 (+23.7%); wind x2 mean 400.
Warming = +dT on tmin/tmax, +2 ERC per C, humidity x(1-0.03 per C), PDSI -0.25 per C.
Negative result: first version multiplied ERC by 1.25; tree models (XGBoost/LightGBM) then
predicted LESS fire (312 -> 208 km2, LightGBM 213 -> 49) because inputs left the training range;
fixed by additive perturbations and a monotone (linear) model for the twin.
