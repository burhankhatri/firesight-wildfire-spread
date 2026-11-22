# FireSight: physics-informed features and a digital twin for next-day wildfire spread

## Problem
Wildfires are growing in size and frequency as the climate warms. Responders need to
know where a fire will be tomorrow, not only where it is today (which MODIS/VIIRS
already show). The task: given today's active-fire mask and 11 environmental layers
for a 64 km x 64 km area at 1 km resolution, predict which pixels will be burning the
next day.

## Data
Next Day Wildfire Spread (NDWS) dataset (Huot et al., IEEE TGRS 2022), built in Google
Earth Engine from MODIS fire masks (MOD14A1), VIIRS NDVI, SRTM elevation, GRIDMET
weather (wind direction/speed, min/max temperature, specific humidity, precipitation),
GRIDMET drought (PDSI, energy release component) and GPWv4 population. 18,545 fire
events across the contiguous US, 2012-2020, split into train/eval/test (14,979 / 1,877 / 1,689 tiles).

## Key idea
The original paper treated the problem as image segmentation and found deep models
only modestly beat simple ones. We argue the useful information is spatial context
and physics, which per-pixel models never see. We turn each pixel into a row and add
13 engineered features: fire density at 3/7/15 km, distance to the nearest active
fire, *upwind fire* (fire density sampled 2 and 5 km upwind using the per-pixel wind
direction), wind u/v components, daily temperature range, terrain slope, a fuel
dryness index (NDVI x ERC) and wind-toward-fire-front. This lets fast tabular models
reason about fire spread like a simple physical fire model.

## Method
- Preprocessing: drop pixels with missing labels (cloud, FireMask = -1); clip each layer
  to physically plausible ranges; median-impute NaNs; standardize for LR and MLP.
- Class imbalance (about 1.1% of pixels burn): balanced training sample (all fire pixels +
  3 non-fire per fire pixel, half hard negatives within 8 km of fire, half random).
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM, MLP; persistence baseline.
- Tuning: RandomizedSearchCV, 3-fold GroupKFold grouped by fire tile (prevents leakage
  between neighbouring pixels of the same fire), scored by average precision.
- Decision threshold chosen to maximise F1 on the eval split; all metrics reported on
  the full, imbalanced test split (6.7 M pixels).
- Ablation: best model with the original 12 inputs vs the 24 engineered features.
- Explainability: SHAP TreeExplainer.
- Digital twin: re-run the model on real test fires with wind rotated / scaled and under
  +2 C and +4 C warming scenarios (temperature, ERC, humidity, PDSI perturbed).
- GEE script to build the same inputs for any location, with ERA5-Land/TerraClimate
  substitutes outside the US (example: Margalla Hills, Islamabad, Pakistan).

## Contributions
1. Spatial and physics-informed features that let tabular ML beat the persistence baseline
   and the original 12-input setup by a large margin.
2. A fair 5-model comparison under grouped CV, evaluated on the true class balance.
3. An explainable digital twin that shows how wind and warming change next-day risk.
4. A reproducible GEE pipeline to apply the model to new regions such as Pakistan.
