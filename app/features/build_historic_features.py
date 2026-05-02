"""Historical rolling features for F1 drivers and constructors, used as inputs to the prediction models."""

import numpy as np
import pandas as pd
import app.data.schemas as schemas

from app.features.utils import _get_prior_results
from app.config import PROCESSED_HISTORIC_FEATURES_DIR


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
    location = events[events["race_id"] == f"{season}_{round_num:02d}"]["location"].iloc[0]

    prior_quali = prior_quali.merge(events[["race_id", "location"]], on="race_id")
    prior_quali = prior_quali[prior_quali["location"] == location]
    prior_quali = prior_quali.sort_values(["season", "round"])

    last_3 = prior_quali.tail(3)["quali_position"].mean()
    last_5 = prior_quali.tail(5)["quali_position"].mean()

    return {"circuit_rolling_quali_pos_last_3": last_3, "circuit_rolling_quali_pos_last_5": last_5}


# average finish position at this circuit over the last 3 and 5 visits
def circuit_rolling_finish_pos(race_results, events, driver_id, season, round_num):
    prior_races = _get_prior_results(race_results, driver_id, season, round_num)
    location = events[events["race_id"] == f"{season}_{round_num:02d}"]["location"].iloc[0]

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


# average constructor fantasy points scored over the last 3 and 5 races
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


# linear slope of constructor fantasy points over the last 5 races - positive means improving form
def constructor_form_trend(fantasy_targets, asset_id, season, round_num):
    fantasy_targets = fantasy_targets[fantasy_targets["asset_type"] == "constructor"]

    prior_points = _get_prior_results(fantasy_targets, asset_id, season, round_num, "asset_id")
    prior_points = prior_points.sort_values(["season", "round"])

    last_5 = prior_points.tail(5)["actual_fantasy_points"].values

    if len(last_5) < 2:
        return {"constructor_form_trend_last_5": float("nan")}
    
    slope = np.polyfit(range(len(last_5)), last_5, 1)[0]

    return {"constructor_form_trend_last_5": slope}


# season
def season(season):
    return {"season": season}

# round number within the season
def round_number(round_num):
    return {"round_number": round_num}


# whether the current race is held on a street circuit
def is_street_circuit(events, season, round_num):
    return {"is_street_circuit": events[events["race_id"] == f"{season}_{round_num:02d}"]["is_street_circuit"].iloc[0]}


# builds the full feature row for a single race for all drivers, joins constructor features, validates, and writes to parquet
def build_historic_features(race_results, quali_results, fantasy_targets, events, season, round_num):
    race_id = f"{season}_{round_num:02d}"
    drivers = race_results[race_results["race_id"] == race_id]["driver_id"].unique()

    quali_results_round = quali_results[quali_results["race_id"] == race_id]  # exclude drivers with no qualifying row (DNS)
    drivers = [d for d in drivers if d in quali_results_round["driver_id"].values]

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

    constructor_rows = []
    for constructor_id in features_df["constructor_id"].unique():
        c = {"race_id": race_id, "constructor_id": constructor_id}
        c.update(constructor_rolling_fantasy_points(fantasy_targets, constructor_id, season, round_num))
        c.update(constructor_rolling_dnf_rate(race_results, constructor_id, season, round_num))
        c.update(constructor_rolling_quali_position(quali_results, constructor_id, season, round_num))
        c.update(constructor_form_trend(fantasy_targets, constructor_id, season, round_num))
        constructor_rows.append(c)

    constructor_features = pd.DataFrame(constructor_rows)
    features_df = features_df.merge(constructor_features, on=["race_id", "constructor_id"])

    schemas.features.validate(features_df)

    PROCESSED_HISTORIC_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(PROCESSED_HISTORIC_FEATURES_DIR / f"{season}_{round_num:02d}.parquet")

    return features_df




