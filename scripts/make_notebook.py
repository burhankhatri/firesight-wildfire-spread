"""Generate FireSight.ipynb, a walkthrough notebook that runs the whole project."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# FireSight: Predicting Next-Day Wildfire Spread from Google Earth Engine Data\n\n"
       "**Problem.** Satellites show where a wildfire is burning today. Responders need to know where "
       "it will burn *tomorrow*. Given today's active-fire mask and 11 environmental layers (64 km x 64 km "
       "at 1 km resolution), we predict which pixels will be burning the next day.\n\n"
       "**Objectives.**\n"
       "1. Build a clean, leakage-free pixel dataset from the Earth Engine-derived Next Day Wildfire Spread data.\n"
       "2. Engineer spatial and physics-informed features (upwind fire, distance to fire, wind vectors, slope, dryness).\n"
       "3. Compare 5 tuned ML models against a persistence baseline on the real, imbalanced test set.\n"
       "4. Explain the drivers with SHAP and build a digital twin for wind and climate-warming what-if scenarios.\n\n"
       "Run order: `python -m src.eda`, `python -m src.train`, `python -m src.report`, `python -m src.twin` "
       "(each step is reproduced below)."),
    code("import json, pandas as pd\nfrom IPython.display import Image, display\n"
         "from src.data import download, build_table, n_samples\nfrom src.features import FEATURES, BASELINE_FEATURES, engineer\n"
         "from src.config import OUT_DIR, FIG_DIR\npd.set_option('display.width', 160)"),
    md("## 1. Dataset\nThe **Next Day Wildfire Spread** dataset (Huot et al., 2022) was built in **Google Earth Engine** "
       "from MODIS active fire (MOD14A1), VIIRS NDVI, SRTM elevation, GRIDMET weather and drought, and GPWv4 population. "
       "We download the public zarr copy (no Kaggle login needed). `gee/export_fire_features.js` rebuilds the same "
       "inputs in Earth Engine for any region (example: Margalla Hills, Pakistan)."),
    code("download()\n{s: n_samples(s) for s in ['train', 'eval', 'test']}"),
    md("## 2. Missing values and class balance\n`FireMask = -1` marks cloud or missing observations. These pixels are "
       "**removed** from training and evaluation. Input layers have no NaNs, but outliers are clipped to physically "
       "plausible ranges and any NaN would be median-imputed (`src/features.py: clean`)."),
    code("pd.read_csv(OUT_DIR / 'eda_splits.csv').round(3)"),
    code("display(Image(FIG_DIR / 'eda_class_balance.png'))"),
    md("## 3. Feature engineering\n12 original inputs (11 layers + today's fire) plus 12 engineered features:\n\n"
       "| Feature | Idea |\n|---|---|\n"
       "| fire_density_3/7/15 | share of burning pixels within 3, 7, 15 km |\n"
       "| dist_to_fire | km to nearest active fire (Euclidean distance transform) |\n"
       "| upwind_fire_2/5 | fire density sampled 2 and 5 km **upwind** using each pixel's wind direction |\n"
       "| wind_u, wind_v | wind vector components (direction as numbers, no 0/360 jump) |\n"
       "| temp_range | daily max - min temperature |\n| slope | terrain slope from the elevation gradient |\n"
       "| dryness | NDVI x energy release component (fuel amount x dryness) |\n"
       "| fire_front_wind | wind speed x upwind fire |"),
    code("sample = build_table('eval', mode='balanced', limit=200)\nsample[FEATURES].describe().T.round(3)"),
    md("## 4. Exploratory data analysis"),
    code("display(Image(FIG_DIR / 'eda_sample_tiles.png'))\ndisplay(Image(FIG_DIR / 'eda_distributions.png'))"),
    code("display(Image(FIG_DIR / 'eda_spread_drivers.png'))\ndisplay(Image(FIG_DIR / 'eda_correlation.png'))"),
    code("pd.read_csv(OUT_DIR / 'eda_summary_stats.csv', index_col=0)"),
    md("## 5. Models and hyper-parameter tuning\n"
       "* **Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, MLP neural network, and a persistence baseline.\n"
       "* **Standardization:** `StandardScaler` inside the LR and MLP pipelines (trees do not need it).\n"
       "* **Imbalance:** balanced training sample (all fire pixels + 3 non-fire per fire pixel, half of them near the fire).\n"
       "* **Tuning:** `RandomizedSearchCV`, 3-fold **GroupKFold by fire tile** (stops pixels of the same fire leaking "
       "between folds), scored by average precision.\n"
       "* **Threshold:** chosen to maximise F1 on the separate eval split; all metrics on the full test split.\n\n"
       "Training takes about 30-60 minutes on a laptop. Set `RETRAIN = True` to run it here; otherwise the saved results load."),
    code("RETRAIN = False\nif RETRAIN:\n    from src.train import main as train_main\n    train_main()\n"
         "R = json.load(open(OUT_DIR / 'results.json'))\nR['data']"),
    code("pd.DataFrame({m: r['best_params'] for m, r in R['models'].items() if r['best_params']}).T"),
    md("## 6. Evaluation and comparative analysis"),
    code("pd.read_csv(OUT_DIR / 'results_table.csv').round(3)"),
    code("for f in ['results_metrics', 'results_curves', 'results_confusion', 'results_ablation']:\n"
         "    display(Image(FIG_DIR / f'{f}.png'))"),
    md("## 7. Explainability (SHAP)"),
    code("display(Image(FIG_DIR / 'results_shap.png'))\npd.read_csv(OUT_DIR / 'shap_importance.csv', index_col=0).tail(10)"),
    md("## 8. Prediction maps"),
    code("display(Image(FIG_DIR / 'results_maps.png'))"),
    md("## 9. Digital twin: wind and climate what-if scenarios\nFor real test fires we rotate or scale the wind, or apply "
       "+2 C / +4 C warming (higher temperature and ERC, lower humidity, more negative PDSI), recompute all features and "
       "predict again. The interactive version is on the project website."),
    code("display(Image(FIG_DIR / 'twin_wind.png'))\ndisplay(Image(FIG_DIR / 'twin_climate.png'))\n"
         "json.load(open(OUT_DIR / 'twin_summary.json'))"),
    md("## 10. Conclusion\nSee the research paper (`paper/FireSight_Paper.pdf`) for the full discussion."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "FireSight.ipynb")
print("wrote FireSight.ipynb")
