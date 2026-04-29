"""Historical rolling features for F1 constructors, joined into driver feature rows by build_driver_features."""

import numpy as np
import pandas as pd
import app.data.schemas as schemas

from app.features.utils import _get_prior_results
from app.config import INTERIM_RACES_DIR, INTERIM_QUALI_DIR, PROCESSED_DIR, PROCESSED_TARGETS_DIR, PROCESSED_CONSTRUCTOR_FEATURES_DIR


# average fantasy points scored over the last 3 and 5 races
def constructor_rolling_fantasy_points(fantasy_targets, asset_id, season, round_num):
    fantasy_targets = fantasy_targets[fantasy_targets["asset_type"] == "constructor"]

    prior_points = _get_prior_results(fantasy_targets, asset_id, season, round_num, "asset_id")
    prior_points = prior_points.sort_values(["season", "round"])

    last_3 = prior_points.tail(3)["actual_fantasy_points"].mean()
    last_5 = prior_points.tail(5)["actual_fantasy_points"].mean()

    return {"constructor_rolling_fantasy_points_last_3": last_3, "constructor_rolling_fantasy_points_last_5": last_5}


# fraction of races where at least one driver DNF'd, averaged over the last 5 races
def constructor_rolling_dnf_rate(race_results, constructor_id, season, round_num):
    prior_races = _get_prior_results(race_results, constructor_id, season, round_num, "constructor_id")
    prior_races = prior_races.sort_values(["season", "round"])

    last_5_races = prior_races.groupby(["season", "round", "race_id"])["dnf_flag"].mean().tail(5)

    return {"constructor_rolling_dnf_rate_last_5": last_5_races.mean()}


# average qualifying position across both drivers over the last 3 races
def constructor_rolling_quali_position(quali_results, constructor_id, season, round_num):
    prior_quali = _get_prior_results(quali_results, constructor_id, season, round_num, "constructor_id")
    prior_quali = prior_quali.sort_values(["season", "round"])

    last_3 = prior_quali.groupby(["season", "round", "race_id"])["quali_position"].mean().tail(3)

    return {"constructor_rolling_quali_pos_last_3": last_3.mean()}


# linear slope of fantasy points over the last 5 races - positive means improving form
def constructor_form_trend(fantasy_targets, asset_id, season, round_num):
    fantasy_targets = fantasy_targets[fantasy_targets["asset_type"] == "constructor"]

    prior_points = _get_prior_results(fantasy_targets, asset_id, season, round_num, "asset_id")
    prior_points = prior_points.sort_values(["season", "round"])

    last_5 = prior_points.tail(5)["actual_fantasy_points"].values

    if len(last_5) < 2:
        return {"constructor_form_trend_last_5": float("nan")}
    
    slope = np.polyfit(range(len(last_5)), last_5, 1)[0]

    return {"constructor_form_trend_last_5": slope}


# builds constructor feature rows for all constructors in a given race
# returns a DataFrame, does not write to parquet (yet)
def build_constructor_features(race_results, quali_results, fantasy_targets, events, season, round_num):
    race_id = f"{season}_{round_num}"
    constructors = race_results[race_results["race_id"] == race_id]["constructor_id"].unique()
    
    rows = []
    for constructor_id in constructors:
        features = {"race_id": race_id, "constructor_id": constructor_id}
        features.update(constructor_rolling_fantasy_points(fantasy_targets, constructor_id, season, round_num))
        features.update(constructor_rolling_dnf_rate(race_results, constructor_id, season, round_num))
        features.update(constructor_rolling_quali_position(quali_results, constructor_id, season, round_num))
        features.update(constructor_form_trend(fantasy_targets, constructor_id, season, round_num))
        rows.append(features)
    
    features_df = pd.DataFrame(rows)
    
    return features_df




