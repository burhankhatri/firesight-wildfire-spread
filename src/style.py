"""One consistent look for every figure in the paper, slides and site."""
import matplotlib.pyplot as plt

FIRE = "#e4572e"
CALM = "#4c78a8"
ACCENT = "#f2a541"
MODEL_COLORS = {
    "Persistence baseline": "#9e9e9e",
    "Logistic Regression": "#4c78a8",
    "Random Forest": "#54a24b",
    "XGBoost": "#b279a2",
    "LightGBM": "#e4572e",
    "MLP (Neural Net)": "#f2a541",
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans",
    })
