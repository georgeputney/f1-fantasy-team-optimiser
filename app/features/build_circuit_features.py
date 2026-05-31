"""Circuit-level historical features for the F1 fantasy pipeline - overtake index, pole conversion, DNF rate, SC rate, and FP3 predictiveness."""

import numpy as np
import pandas as pd

from app.config import PROCESSED_CIRCUIT_FEATURES_DIR


# number of prior seasons with data at this circuit - signals reliability of circuit stats for newer venues
def circuit_data_n_seasons(circuit_races):
    return {"circuit_data_n_seasons": int(circuit_races["season"].nunique())}


# mean ratio of F1 Fantasy overtakes at this circuit vs the season average, season-normalised
# uses manual race_overtakes data; circuit_race_ids already enforces the temporal cutoff
def circuit_overtake_index(overtake_history, circuit_race_ids):
    if overtake_history.empty:
        return {"circuit_overtake_index": float("nan")}

    ot = overtake_history.copy()
    ot["race_id"] = ot["season"].astype(str) + "_" + ot["round"].astype(str).str.zfill(2)

    race_totals = ot.groupby(["race_id", "season"])["race_overtakes"].sum().reset_index()
    season_avg = race_totals.groupby("season")["race_overtakes"].mean().rename("season_avg")
    race_totals = race_totals.join(season_avg, on="season")

    circuit_df = race_totals[race_totals["race_id"].isin(circuit_race_ids)]

    if circuit_df.empty or (circuit_df["season_avg"] == 0).all():
        return {"circuit_overtake_index": float("nan")}

    return {"circuit_overtake_index": (circuit_df["race_overtakes"] / circuit_df["season_avg"]).mean()}


# fraction of prior races at this circuit where the pole sitter won
def circuit_pole_to_win_rate(circuit_races):
    pole = circuit_races[circuit_races["grid_position"] == 1]
    if pole.empty:
        return {"circuit_pole_to_win_rate": float("nan")}
    return {"circuit_pole_to_win_rate": (pole["finish_position"] == 1).mean()}


# fraction of top-3 starters who finished on the podium across all prior races at this circuit
def circuit_top3_grid_to_podium_rate(circuit_races):
    top3 = circuit_races[circuit_races["grid_position"] <= 3]
    if top3.empty:
        return {"circuit_top3_grid_to_podium_rate": float("nan")}
    return {"circuit_top3_grid_to_podium_rate": (top3["finish_position"] <= 3).mean()}


# mean DNF rate per driver-race entry at this circuit across all prior seasons
def circuit_dnf_rate(circuit_races):
    if circuit_races.empty:
        return {"circuit_dnf_rate": float("nan")}
    return {"circuit_dnf_rate": circuit_races["dnf_flag"].mean()}


# fraction of prior races at this circuit that featured at least one safety car or VSC lap
# returns NaN for any race ingested before TrackStatus was added to the race laps ingest
def circuit_sc_vsc_rate(race_laps_all, circuit_race_ids):
    if race_laps_all is None or "track_status" not in race_laps_all.columns:
        return {"circuit_sc_vsc_rate": float("nan")}

    circuit_laps = race_laps_all[race_laps_all["race_id"].isin(circuit_race_ids)]
    if circuit_laps.empty:
        return {"circuit_sc_vsc_rate": float("nan")}

    sc_by_race = circuit_laps.groupby("race_id")["track_status"].apply(
        lambda s: s.astype(str).str.contains("4|6", regex=True).any()
    )
    return {"circuit_sc_vsc_rate": sc_by_race.mean()}


# fraction of FP3 top-3 drivers (by best lap time) who also qualified in the top 3
# averaged across all prior seasons at this circuit - sprint weekends are excluded as FP3 does not run
def circuit_fp3_top3_to_quali_top3_rate(fp3_all, prior_quali, circuit_race_ids):
    if fp3_all is None or fp3_all.empty:
        return {"circuit_fp3_top3_to_quali_top3_rate": float("nan")}

    circuit_fp3 = fp3_all[fp3_all["race_id"].isin(circuit_race_ids)]
    circuit_quali = prior_quali[prior_quali["race_id"].isin(circuit_race_ids)]

    rates = []
    for race_id in circuit_race_ids:
        fp3_race = circuit_fp3[circuit_fp3["race_id"] == race_id]
        quali_race = circuit_quali[circuit_quali["race_id"] == race_id]

        if fp3_race.empty or quali_race.empty:
            continue

        fp3_top3 = set(fp3_race.groupby("driver_id")["lap_time"].min().nsmallest(3).index)
        quali_top3 = set(quali_race[quali_race["quali_position"] <= 3]["driver_id"])

        if not fp3_top3 or not quali_top3:
            continue

        rates.append(len(fp3_top3 & quali_top3) / 3)

    if not rates:
        return {"circuit_fp3_top3_to_quali_top3_rate": float("nan")}
    return {"circuit_fp3_top3_to_quali_top3_rate": np.mean(rates)}


# builds circuit features for a single race and writes to data/processed/circuit_features/
# one row per race (same values apply to all drivers); merged on race_id in train.py and predict.py
# race_laps_all and fp3_all may be None if those interim directories are empty
def build_circuit_features(race_results, quali_results, events, race_laps_all, fp3_all, overtake_history, season, round_num):
    race_id = f"{season}_{round_num:02d}"
    location = events[events["race_id"] == race_id]["location"].iloc[0]

    # strict season cutoff - never use data from the current or future seasons
    prior_events = events[events["season"] < season]
    circuit_race_ids = prior_events[prior_events["location"] == location]["race_id"].tolist()

    prior_race_results = race_results[race_results["season"] < season]
    circuit_races = prior_race_results[prior_race_results["race_id"].isin(circuit_race_ids)]

    prior_quali = quali_results[quali_results["season"] < season]

    features = {"race_id": race_id}
    features.update(circuit_data_n_seasons(circuit_races))
    features.update(circuit_overtake_index(overtake_history, circuit_race_ids))
    features.update(circuit_pole_to_win_rate(circuit_races))
    features.update(circuit_top3_grid_to_podium_rate(circuit_races))
    features.update(circuit_dnf_rate(circuit_races))
    features.update(circuit_sc_vsc_rate(race_laps_all, circuit_race_ids))
    features.update(circuit_fp3_top3_to_quali_top3_rate(fp3_all, prior_quali, circuit_race_ids))

    features_df = pd.DataFrame([features])

    PROCESSED_CIRCUIT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(PROCESSED_CIRCUIT_FEATURES_DIR / f"{season}_{round_num:02d}.parquet")

    return features_df
