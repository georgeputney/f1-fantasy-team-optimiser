"""Historical rolling features for F1 drivers, used as inputs to the race finish prediction model."""

import pandas as pd
import app.data.schemas as schemas

from app.features.utils import _get_prior_results
from app.features.build_constructor_features import build_constructor_features
from app.config import INTERIM_RACES_DIR, INTERIM_QUALI_DIR, PROCESSED_DIR, PROCESSED_TARGETS_DIR, PROCESSED_DRIVER_FEATURES_DIR


# average qualifying position over the last 3 and 5 races
def rolling_quali_position(quali_results, driver_id, season, round_num):
    prior_quali = _get_prior_results(quali_results, driver_id, season, round_num)
    prior_quali = prior_quali.sort_values(["season", "round"])

    last_3 = prior_quali.tail(3)["quali_position"].mean()
    last_5 = prior_quali.tail(5)["quali_position"].mean()

    return {"rolling_quali_pos_last_3": last_3, "rolling_quali_pos_last_5": last_5}


# average finish position over the last 3 and 5 races
def rolling_finish_position(race_results, driver_id, season, round_num):
    prior_races = _get_prior_results(race_results, driver_id, season, round_num)
    prior_races = prior_races.sort_values(["season", "round"])

    last_3 = prior_races.tail(3)["finish_position"].mean()
    last_5 = prior_races.tail(5)["finish_position"].mean()

    return {"rolling_finish_pos_last_3": last_3, "rolling_finish_pos_last_5": last_5}


# average fantasy points scored over the last 3 and 5 races
def rolling_fantasy_points(fantasy_targets, asset_id, season, round_num):
    fantasy_targets = fantasy_targets[fantasy_targets["asset_type"] == "driver"]

    prior_points = _get_prior_results(fantasy_targets, asset_id, season, round_num, "asset_id")
    prior_points = prior_points.sort_values(["season", "round"])

    last_3 = prior_points.tail(3)["actual_fantasy_points"].mean()
    last_5 = prior_points.tail(5)["actual_fantasy_points"].mean()

    return {"rolling_fantasy_points_last_3": last_3, "rolling_fantasy_points_last_5": last_5}


# fraction of races ending in DNF over the last 5 races
def rolling_dnf_rate(race_results, driver_id, season, round_num):
    prior_races = _get_prior_results(race_results, driver_id, season, round_num)
    prior_races = prior_races.sort_values(["season", "round"])

    return {"rolling_dnf_rate_last_5": prior_races.tail(5)["dnf_flag"].mean()}


# average qualifying position at this circuit over the last 3 and 5 visits
def circuit_rolling_quali_pos(quali_results, events, driver_id, season, round_num):
    prior_quali = _get_prior_results(quali_results, driver_id, season, round_num)
    location = events[events["race_id"] == f"{season}_{round_num}"]["location"].iloc[0]

    prior_quali = prior_quali.merge(events[["race_id", "location"]], on="race_id")
    prior_quali = prior_quali[prior_quali["location"] == location]
    prior_quali = prior_quali.sort_values(["season", "round"])

    last_3 = prior_quali.tail(3)["quali_position"].mean()
    last_5 = prior_quali.tail(5)["quali_position"].mean()

    return {"circuit_rolling_quali_pos_last_3": last_3, "circuit_rolling_quali_pos_last_5": last_5}


# average finish position at this circuit over the last 3 and 5 visits
def circuit_rolling_finish_pos(race_results, events, driver_id, season, round_num):
    prior_races = _get_prior_results(race_results, driver_id, season, round_num)
    location = events[events["race_id"] == f"{season}_{round_num}"]["location"].iloc[0]

    prior_races = prior_races.merge(events[["race_id", "location"]], on="race_id")
    prior_races = prior_races[prior_races["location"] == location]
    prior_races = prior_races.sort_values(["season", "round"])

    last_3 = prior_races.tail(3)["finish_position"].mean()
    last_5 = prior_races.tail(5)["finish_position"].mean()

    return {"circuit_rolling_finish_pos_last_3": last_3, "circuit_rolling_finish_pos_last_5": last_5}


# total F1 championship points scored in the current season before this race
def season_points_to_date(race_results, driver_id, season, round_num):
    prior_races = _get_prior_results(race_results, driver_id, season, round_num)
    prior_season = prior_races[prior_races["season"] == season]

    return {"season_points_to_date": prior_season["points"].sum()}


# round number within the season
def season(season):
    return {"season": season}

# round number within the season
def round_number(round_num):
    return {"round_number": round_num}


# whether the current race is held on a street circuit
def is_street_circuit(events, season, round_num):
    return {"is_street_circuit": events[events["race_id"] == f"{season}_{round_num}"]["is_street_circuit"].iloc[0]}


# builds the full driver feature row for a single race, joins constructor context, validates, and writes to parquet
def build_driver_features(race_results, quali_results, fantasy_targets, events, season, round_num):
    race_id = f"{season}_{round_num}"
    drivers = race_results[race_results["race_id"] == race_id]["driver_id"].unique()
    
    rows = []
    for driver_id in drivers:
        features = {"race_id": race_id, "driver_id": driver_id}
        features.update(rolling_quali_position(quali_results, driver_id, season, round_num))
        features.update(rolling_finish_position(race_results, driver_id, season, round_num))
        features.update(rolling_fantasy_points(fantasy_targets, driver_id, season, round_num))
        features.update(rolling_dnf_rate(race_results, driver_id, season, round_num))
        features.update(circuit_rolling_quali_pos(quali_results, events, driver_id, season, round_num))
        features.update(circuit_rolling_finish_pos(race_results, events, driver_id, season, round_num))
        features.update(season_points_to_date(race_results, driver_id, season, round_num))
        features["season"] = season
        features.update(round_number(round_num))
        features.update(is_street_circuit(events, season, round_num))

        constructor_id = race_results[
            (race_results["race_id"] == race_id) & 
            (race_results["driver_id"] == driver_id)
        ]["constructor_id"].iloc[0]
        features["constructor_id"] = constructor_id

        rows.append(features)
    
    features_df = pd.DataFrame(rows)

    constructor_features = build_constructor_features(race_results, quali_results, fantasy_targets, events, season, round_num)
    features_df = features_df.merge(constructor_features, on=["race_id", "constructor_id"])
    features_df["prediction_stage"] = "training"

    schemas.driver_features.validate(features_df)

    PROCESSED_DRIVER_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(PROCESSED_DRIVER_FEATURES_DIR / f"{season}_{round_num}.parquet")
    
    return features_df

