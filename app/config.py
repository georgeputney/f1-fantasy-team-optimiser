"""Shared constants, paths, and configuration."""

from pathlib import Path


# paths
ROOT_DIR                            = Path(__file__).resolve().parent.parent
DATA_DIR                            = ROOT_DIR / "data"

RAW_DIR                             = DATA_DIR / "raw"          # fastf1 API downloads
RAW_EVENTS_DIR                      = RAW_DIR / "events"
RAW_RACES_DIR                       = RAW_DIR / "race"
RAW_RACE_LAPS_DIR                   = RAW_DIR / "race_laps"
RAW_QUALI_DIR                       = RAW_DIR / "qualifying"
RAW_SPRINT_DIR                      = RAW_DIR / "sprint"
RAW_SPRINT_LAPS_DIR                 = RAW_DIR / "sprint_laps"
RAW_SPRINT_QUALIFYING_DIR           = RAW_DIR / "sprint_qualifying"
RAW_FP3_DIR                         = RAW_DIR / "fp3"
RAW_FP2_DIR                         = RAW_DIR / "fp2"
RAW_FP1_DIR                         = RAW_DIR / "fp1"
RAW_PITSTOPS_DIR                    = RAW_DIR / "race_pitstops"
RAW_RACE_OVERTAKES_DIR              = RAW_DIR / "race_overtakes"
RAW_SPRINT_OVERTAKES_DIR            = RAW_DIR / "sprint_overtakes"
RAW_DOTD_DIR                        = RAW_DIR / "race_dotd"
FASTF1_CACHE_DIR                    = RAW_DIR / "fastf1_cache"

MANUAL_DIR                          = DATA_DIR / "manual"       # manually maintained inputs
FANTASY_PRICES_DIR                  = MANUAL_DIR / "fantasy_prices"
FANTASY_POINTS_DIR                  = MANUAL_DIR / "fantasy_points"
RACE_OVERTAKES_DIR                  = MANUAL_DIR / "race_overtakes"
SPRINT_OVERTAKES_DIR                = MANUAL_DIR / "sprint_overtakes"

INTERIM_DIR                         = DATA_DIR / "interim"      # cleaned tables, before feature engineering
INTERIM_EVENTS_DIR                  = INTERIM_DIR / "events"
INTERIM_RACES_DIR                   = INTERIM_DIR / "race"
INTERIM_RACE_LAPS_DIR               = INTERIM_DIR / "race_laps"
INTERIM_QUALI_DIR                   = INTERIM_DIR / "qualifying"
INTERIM_SPRINT_DIR                  = INTERIM_DIR / "sprint"
INTERIM_SPRINT_LAPS_DIR             = INTERIM_DIR / "sprint_laps"
INTERIM_SPRINT_QUALIFYING_DIR       = INTERIM_DIR / "sprint_qualifying"
INTERIM_FP3_DIR                     = INTERIM_DIR / "fp3"
INTERIM_FP2_DIR                     = INTERIM_DIR / "fp2"
INTERIM_FP1_DIR                     = INTERIM_DIR / "fp1"
INTERIM_PITSTOPS_DIR                = INTERIM_DIR / "race_pitstops"
INTERIM_RACE_OVERTAKES_DIR          = INTERIM_DIR / "race_overtakes"
INTERIM_SPRINT_OVERTAKES_DIR        = INTERIM_DIR / "sprint_overtakes"
INTERIM_DOTD_DIR                    = INTERIM_DIR / "race_dotd"

PROCESSED_DIR                       = DATA_DIR / "processed"    # feature store, model-ready datasets
PROCESSED_TARGETS_DIR               = PROCESSED_DIR / "targets"
PROCESSED_PRICES_DIR                = PROCESSED_DIR / "prices"
PROCESSED_HISTORIC_FEATURES_DIR     = PROCESSED_DIR / "historic_features"
PROCESSED_PRACTICE_FEATURES_DIR     = PROCESSED_DIR / "practice_features"
PROCESSED_CIRCUIT_FEATURES_DIR      = PROCESSED_DIR / "circuit_features"

ARTIFACTS_DIR                       = DATA_DIR / "artifacts"    # trained model files
REPORTS_DIR                         = ROOT_DIR / "reports"      # backtest plots and output tables
PREDICTIONS_DIR                     = REPORTS_DIR / "predictions"  # per-race prediction snapshots

TEAM_STATE_FILE                     = DATA_DIR / "team_state_file.json"


# season splits (time-aware, no leakage)
TRAIN_SEASONS                       = [2018, 2019, 2020, 2021, 2022, 2023]  # earliest data with stable telemetry
VAL_SEASONS                         = [2024]
TEST_SEASONS                        = [2025]
LIVE_SEASONS                         = [2026]
ALL_SEASONS                         = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS + LIVE_SEASONS


# fantasy constraints
BUDGET_CAP                          = 100.0     # total budget (at the start of the season)
DRIVER_ROSTER_SIZE                  = 5         # no. drivers to pick
CONSTRUCTOR_ROSTER_SIZE             = 2         # no. constructors to pick
STARTING_PRICES_DIR                 = MANUAL_DIR / "starting_prices"  # round 1 prices per season
PRICE_FLOOR                         = {2025: 4.5, 2026: 3.5}         # minimum asset price by season


# targets
TARGET_COL                          = "fantasy_points"
SPRINT_TARGET                       = "sprint_position"       
QUALI_TARGET                        = "quali_position"        
RACE_TARGET                         = "finish_position"        
COMPONENT_TARGETS                   = ["quali_points", "race_points", "sprint_points"]  # per-session points