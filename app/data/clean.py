"""Reads raw parquet files from data/raw/, normalises and transforms them into clean internal tables, validates against pandera schemas, and writes to data/interim/."""

import pandas as pd
import app.data.schemas as schemas

from app.config import (
    RAW_RACES_DIR, RAW_QUALI_DIR, RAW_EVENTS_DIR,
    INTERIM_RACES_DIR, INTERIM_QUALI_DIR, INTERIM_EVENTS_DIR,
)

STREET_CIRCUITS = {
    "Monaco", "Baku", "Singapore", "Jeddah", "Melbourne",
    "Las Vegas", "Miami", "Sochi",
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


# read raw qualifying results from data/raw/quali/, normalise driver and constructor IDs,
# convert Q1/Q2/Q3 lap times to seconds, validate against schema, 
# write to data/interim/quali/
def clean_qualifying_results(season, round_num):
    results = pd.read_parquet(RAW_QUALI_DIR / f"{season}_{round_num:02d}.parquet")

    for col in ["Q1", "Q2", "Q3"]:
        results[col] = results[col].dt.total_seconds()
    
    results = results.rename(columns={
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