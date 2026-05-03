"""Reads raw parquet files from data/raw/, normalises and transforms them into clean internal tables, validates against pandera schemas, and writes to data/interim/."""

import pandas as pd
import app.data.schemas as schemas

from app.config import (
    RAW_RACES_DIR, RAW_QUALI_DIR, RAW_EVENTS_DIR, RAW_FP3_DIR, RAW_FP2_DIR,
    INTERIM_RACES_DIR, INTERIM_QUALI_DIR, INTERIM_EVENTS_DIR, INTERIM_FP3_DIR, INTERIM_FP2_DIR
)

RAW_PRACTICE_DIRS = {"FP2": RAW_FP2_DIR, "FP3": RAW_FP3_DIR}
INTERIM_PRACTICE_DIRS = {"FP2": INTERIM_FP2_DIR, "FP3": INTERIM_FP3_DIR}


STREET_CIRCUITS = {
    "Monaco", "Baku", "Singapore", "Jeddah", "Melbourne",
    "Las Vegas", "Miami", "Sochi",
}

DRIVER_ID_NORMALISATION = {
    "kimi_antonelli": "andrea_kimi_antonelli",
}

CONSTRUCTOR_ID_NORMALISATION = {
    "alfa": "alfa_romeo",       # Alfa Romeo (2023 and prior)
    "rb": "racing_bulls",       # RB rebranded to Racing Bulls (2025)
    "sauber": "kick_sauber",    # Sauber rebranded to Kick Sauber (2024-2025)
}


# read raw event metadata from data/raw/events/, rename columns, 
# derive is_sprint and is_street_circuit, validate against schema,  
# write to data/interim/events/
def clean_events(season, round_num):
    events = pd.read_parquet(RAW_EVENTS_DIR / f"{season}_{round_num:02d}.parquet")

    events = events.rename(columns={
    "RoundNumber": "round",
    "Country": "country",
    "Location": "location",
    "EventName": "event_name",
    "EventDate": "event_date",
    })

    events["season"] = season
    events["is_sprint"] = events["EventFormat"].isin(["sprint", "sprint_qualifying", "sprint_shootout"])
    events["is_street_circuit"] = events["location"].isin(STREET_CIRCUITS)

    events = events.drop(columns=["EventFormat"])
    
    schemas.events.validate(events)

    INTERIM_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(INTERIM_EVENTS_DIR / f"{season}_{round_num:02d}.parquet")

    return events


# read raw practice lap data, normalise driver IDs via driver_code lookup from cleaned quali results,
# convert lap and sector times to seconds, validate against schema, write to data/interim/fp{n}/
def clean_practice_results(season, round_num, session_name):
    laps = pd.read_parquet(RAW_PRACTICE_DIRS[session_name] / f"{season}_{round_num:02d}.parquet")

    # build driver_code -> driver_id mapping from cleaned quali results
    cleaned_quali = pd.read_parquet(INTERIM_QUALI_DIR / f"{season}_{round_num:02d}.parquet")
    driver_map = cleaned_quali.set_index("driver_code")["driver_id"]

    laps["driver_id"] = laps["Driver"].map(driver_map)
    laps = laps.drop(columns=["Driver"])

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col] = laps[col].dt.total_seconds()

    laps = laps.rename(columns={
        "LapTime": "lap_time",
        "Sector1Time": "sector1_time",
        "Sector2Time": "sector2_time",
        "Sector3Time": "sector3_time",
        "Compound": "compound",
        "LapNumber": "lap_number",
        "IsPersonalBest": "is_personal_best",
    })

    laps["race_id"] = f"{season}_{round_num:02d}"
    laps["season"] = season
    laps["round"] = round_num

    schemas.practice_results.validate(laps)

    out_dir = INTERIM_PRACTICE_DIRS[session_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    laps.to_parquet(out_dir / f"{season}_{round_num:02d}.parquet")

    return laps


# read raw qualifying results from data/raw/quali/, normalise driver and constructor IDs,
# convert Q1/Q2/Q3 lap times to seconds, validate against schema, 
# write to data/interim/quali/
def clean_qualifying_results(season, round_num):
    results = pd.read_parquet(RAW_QUALI_DIR / f"{season}_{round_num:02d}.parquet")

    for col in ["Q1", "Q2", "Q3"]:
        results[col] = results[col].dt.total_seconds()
    
    results = results.rename(columns={
        "Abbreviation": "driver_code",
        "TeamId": "constructor_id",
        "Position": "quali_position",
        "Q1": "q1_time",
        "Q2": "q2_time", 
        "Q3": "q3_time",
    })

    results["season"] = season
    results["round"] = round_num
    # replace spaces to handle multi-part names
    results["driver_id"] = results["FirstName"].str.lower().str.replace(" ", "_") + "_" + results["LastName"].str.lower().str.replace(" ", "_")
    results["driver_id"] = results["driver_id"].replace(DRIVER_ID_NORMALISATION)
    results["constructor_id"] = results["constructor_id"].replace(CONSTRUCTOR_ID_NORMALISATION)

    results = results.drop(columns={
        "DriverId",
        "FirstName",
        "LastName",
    })

    schemas.quali_results.validate(results)

    INTERIM_QUALI_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(INTERIM_QUALI_DIR / f"{season}_{round_num:02d}.parquet")

    return results


# read raw race results from data/raw/races/, normalise driver and constructor IDs,
# derive dnf_flag, positions_gained, and fastest_lap_flag, validate against schema,
# write to data/interim/races/
def clean_race_results(season, round_num):
    results = pd.read_parquet(RAW_RACES_DIR / f"{season}_{round_num:02d}.parquet")
    
    results = results.rename(columns={
        "TeamId": "constructor_id",
        "GridPosition": "grid_position",
        "Position": "finish_position",
        "Status": "status",
        "Points": "points",
    })

    results["season"] = season
    results["round"] = round_num
    # replace spaces to handle multi-part names
    results["driver_id"] = results["FirstName"].str.lower().str.replace(" ", "_") + "_" + results["LastName"].str.lower().str.replace(" ", "_")
    results["driver_id"] = results["driver_id"].replace(DRIVER_ID_NORMALISATION)
    results["constructor_id"] = results["constructor_id"].replace(CONSTRUCTOR_ID_NORMALISATION)

    results["status"] = results["status"].str.lower()
    results["dnf_flag"] = ~(results["status"].str.startswith("+") | results["status"].eq("finished") | results["status"].eq("disqualified"))
    results["dsq_flag"] = results["status"].eq("disqualified")    
    results["positions_gained"] = results["grid_position"] - results["finish_position"]
    results["fastest_lap_flag"] = False     # TODO: derive from lap data once laps are ingested
    results["dotd_flag"] = False            # TODO: derive probability from historic data

    results = results.drop(columns={
        "DriverId",
        "FirstName",
        "LastName",
    })

    schemas.race_results.validate(results)

    INTERIM_RACES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(INTERIM_RACES_DIR / f"{season}_{round_num:02d}.parquet")

    return results