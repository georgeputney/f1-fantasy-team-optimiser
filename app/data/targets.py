"""Computes actual fantasy points from cleaned race and qualifying results, used as training labels for the models."""

import pandas as pd
import app.data.schemas as schemas
import app.data.scoring_rules as scoring_rules

from app.config import INTERIM_RACES_DIR, INTERIM_QUALI_DIR, PROCESSED_TARGETS_DIR


# returns fantasy points for all drivers and constructors for a single qualifying session
def compute_qualifying_targets(season, round_num):
    results = pd.read_parquet(INTERIM_QUALI_DIR / f"{season}_{round_num}.parquet")

    drivers_score = results.apply(lambda row: scoring_rules.score_driver_qualifying(row["quali_position"], row["q1_time"]), axis=1)
    driver_targets = pd.DataFrame({
        "race_id": results["race_id"],
        "asset_id": results["driver_id"],
        "asset_type": "driver",
        "actual_fantasy_points": drivers_score
    })

    constructor_groups = results.groupby(["race_id", "constructor_id"]).agg(
        quali_position=("quali_position", list),
        q1_time=("q1_time", list),
        q2_time=("q2_time", list),
        q3_time=("q3_time", list),
    ).reset_index() 

    constructors_score = constructor_groups.apply(lambda row: scoring_rules.score_constructor_qualifying(row["quali_position"], row["q1_time"], row["q2_time"], row["q3_time"]), axis=1)
    constructor_targets = pd.DataFrame({
        "race_id": constructor_groups["race_id"],
        "asset_id": constructor_groups["constructor_id"],
        "asset_type": "constructor",
        "actual_fantasy_points": constructors_score
    })

    return pd.concat([driver_targets, constructor_targets]).reset_index(drop=True) 

    
# returns fantasy points for all drivers and constructors for a single race session
def compute_race_targets(season, round_num):
    results = pd.read_parquet(INTERIM_RACES_DIR / f"{season}_{round_num}.parquet")

    drivers_score = results.apply(lambda row: scoring_rules.score_driver_race(row["finish_position"], row["positions_gained"], row["dnf_flag"], row["dsq_flag"], row["fastest_lap_flag"], row["dotd_flag"]), axis=1)
    driver_targets = pd.DataFrame({
        "race_id": results["race_id"],
        "asset_id": results["driver_id"],
        "asset_type": "driver",
        "actual_fantasy_points": drivers_score
    })

    constructor_groups = results.groupby(["race_id", "constructor_id"]).agg(
        finish_position=("finish_position", list),
        positions_gained=("positions_gained", list),
        dnf_flag=("dnf_flag", list),
        dsq_flag=("dsq_flag", list),
        fastest_lap_flag=("fastest_lap_flag", list),
    ).reset_index() 

    constructors_score = constructor_groups.apply(lambda row: scoring_rules.score_constructor_race(row["finish_position"], row["positions_gained"], row["dnf_flag"], row["dsq_flag"], row["fastest_lap_flag"]), axis=1)
    constructor_targets = pd.DataFrame({
        "race_id": constructor_groups["race_id"],
        "asset_id": constructor_groups["constructor_id"],
        "asset_type": "constructor",
        "actual_fantasy_points": constructors_score
    })

    return pd.concat([driver_targets, constructor_targets]).reset_index(drop=True) 


# computes total fantasy points per asset per race by summing qualifying and race scores, 
# validates against schema, 
# writes to data/processed/targets/
def compute_targets(season, round_num):
    quali_targets = compute_qualifying_targets(season, round_num)
    race_targets = compute_race_targets(season, round_num)

    targets = pd.concat([quali_targets, race_targets]).reset_index(drop=True) 
    targets = targets.groupby(["race_id", "asset_id", "asset_type"], as_index=False)["actual_fantasy_points"].sum()

    schemas.fantasy_targets.validate(targets)

    PROCESSED_TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(PROCESSED_TARGETS_DIR / f"{season}_{round_num}.parquet")

    return targets
