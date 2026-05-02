"""Shared constants, paths, and configuration."""

from pathlib import Path


# paths
ROOT_DIR                            = Path(__file__).resolve().parent.parent
DATA_DIR                            = ROOT_DIR / "data"

RAW_DIR                             = DATA_DIR / "raw"          # fastf1 downloads, fantasy_prices.csv
RAW_EVENTS_DIR                      = RAW_DIR / "events"
RAW_RACES_DIR                       = RAW_DIR / "races"
RAW_QUALI_DIR                       = RAW_DIR / "quali"
RAW_FP3_DIR                         = RAW_DIR / "fp3"
RAW_FP2_DIR                         = RAW_DIR / "fp2"
RAW_FP1_DIR                         = RAW_DIR / "fp1"
FASTF1_CACHE_DIR                    = RAW_DIR / "fastf1_cache"
FANTASY_PRICES_DIR                  = RAW_DIR / "fantasy_prices"

INTERIM_DIR                         = DATA_DIR / "interim"      # cleaned tables, before feature engineering
INTERIM_EVENTS_DIR                  = INTERIM_DIR / "events"
INTERIM_RACES_DIR                   = INTERIM_DIR / "races"
INTERIM_QUALI_DIR                   = INTERIM_DIR / "quali"

PROCESSED_DIR                       = DATA_DIR / "processed"    # feature store, model-ready datasets
PROCESSED_TARGETS_DIR               = PROCESSED_DIR / "targets"
PROCESSED_HISTORIC_FEATURES_DIR     = PROCESSED_DIR / "historic_features"
PROCESSED_PRACTICE_FEATURES_DIR     = PROCESSED_DIR / "practice_features"

ARTIFACTS_DIR                       = DATA_DIR / "artifacts"    # trained model files
REPORTS_DIR                         = ROOT_DIR / "reports"      # backtest plots and output tables

# season splits (time-aware, no leakage)
TRAIN_SEASONS                       = [2018, 2019, 2020, 2021, 2022, 2023]  # earliest data with stable telemetry
VAL_SEASONS                         = [2024]
TEST_SEASONS                        = [2025]
ALL_SEASONS                         = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS


# fantasy constraints
BUDGET_CAP                          = 100.0     # total budget (at the start of the season)
DRIVER_ROSTER_SIZE                  = 5         # no. drivers to pick
CONSTRUCTOR_ROSTER_SIZE             = 2         # no. constructors to pick


# targets
TARGET_COL                          = "fantasy_points"
SPRINT_TARGET                       = "sprint_position"       
QUALI_TARGET                        = "quali_position"        
RACE_TARGET                         = "finish_position"        
COMPONENT_TARGETS                   = ["quali_points", "race_points", "sprint_points"]  # per-session points