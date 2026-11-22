"""Shared paths and constants for the FireSight pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG = 64  # each sample is a 64 x 64 km tile at 1 km resolution

# The 11 environmental input layers of the Next Day Wildfire Spread dataset
# (exported from Google Earth Engine by Huot et al.).
RAW_VARS = ["elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr",
            "pdsi", "NDVI", "population", "erc"]

# Physically plausible clipping ranges (removes sensor / interpolation outliers).
CLIP = {
    "elevation": (0, 4500), "th": (0, 360), "vs": (0, 12), "tmmn": (250, 310),
    "tmmx": (250, 320), "sph": (0, 0.025), "pr": (0, 50), "pdsi": (-10, 10),
    "NDVI": (-2000, 10000), "population": (0, 3000), "erc": (0, 110),
}

PRETTY = {
    "elevation": "Elevation", "th": "Wind direction", "vs": "Wind speed",
    "tmmn": "Min temperature", "tmmx": "Max temperature", "sph": "Humidity",
    "pr": "Precipitation", "pdsi": "Drought index (PDSI)", "NDVI": "Vegetation (NDVI)",
    "population": "Population", "erc": "Energy release (ERC)", "prev_fire": "Fire today",
    "fire_density_3": "Fire nearby (3 km)", "fire_density_7": "Fire nearby (7 km)",
    "fire_density_15": "Fire nearby (15 km)", "dist_to_fire": "Distance to fire",
    "upwind_fire_2": "Upwind fire (2 km)", "upwind_fire_5": "Upwind fire (5 km)",
    "wind_u": "Wind east-west", "wind_v": "Wind north-south", "temp_range": "Daily temp range",
    "slope": "Terrain slope", "dryness": "Fuel dryness (NDVI x ERC)",
    "fire_front_wind": "Wind toward fire front",
}
