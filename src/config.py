"""Shared constants, paths, and configuration."""

from pathlib import Path

# paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# season splits (time-aware, no leakage)
TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS = [2024]
TEST_SEASONS = [2025]
ALL_SEASONS = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS

# fantasy constraints
BUDGET_CAP = 100.0              # total budget (at the start of the season)
DRIVER_ROSTER_SIZE = 5          # no. drivers to pick
CONSTRUCTOR_ROSTER_SIZE = 2     # no. constructors to pick

# feature columns (populated by features.py, listed here for reference)
TARGET_COL = "fantasy_points"
