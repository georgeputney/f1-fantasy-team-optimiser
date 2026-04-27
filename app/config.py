"""Shared constants, paths, and configuration."""

from pathlib import Path


# paths
ROOT_DIR                = Path(__file__).resolve().parent.parent
DATA_DIR                = ROOT_DIR / "data"
RAW_DIR                 = DATA_DIR / "raw"          # fastf1 downloads, fantasy_prices.csv
FASTF1_CACHE_DIR        = RAW_DIR / "fastf1_cache"
FANTASY_PRICES_DIR      = RAW_DIR / "fantasy_prices"
INTERIM_DIR             = DATA_DIR / "interim"      # cleaned tables, before feature engineering
PROCESSED_DIR           = DATA_DIR / "processed"    # feature store, model-ready datasets
ARTIFACTS_DIR           = DATA_DIR / "artifacts"    # trained model files


# season splits (time-aware, no leakage)
TRAIN_SEASONS           = [2018, 2019, 2020, 2021, 2022, 2023]  # earliest data with stable telemetry
VAL_SEASONS             = [2024]
TEST_SEASONS            = [2025]
ALL_SEASONS             = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS


# fantasy constraints
BUDGET_CAP              = 100.0     # total budget (at the start of the season)
DRIVER_ROSTER_SIZE      = 5         # no. drivers to pick
CONSTRUCTOR_ROSTER_SIZE = 2         # no. constructors to pick


# targets
TARGET_COL              = "fantasy_points"
SPRINT_TARGET           = "sprint_position"       
QUALI_TARGET            = "quali_position"        
RACE_TARGET             = "race_position"        
COMPONENT_TARGETS       = ["quali_points", "race_points", "sprint_points"]  # per-session points