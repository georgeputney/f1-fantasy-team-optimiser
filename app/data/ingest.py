"""
Fetches raw F1 session data from the FastF1 API and writes it to parquet files in data/raw/.

Enables the FastF1 cache on import. Each function fetches one session type for one race weekend.
"""

import fastf1
import pandas as pd

from app.config import FASTF1_CACHE_DIR, RAW_EVENTS_DIR, RAW_RACES_DIR, RAW_QUALI_DIR, RAW_FP3_DIR, RAW_FP2_DIR, RAW_FP1_DIR

fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

FP2_COLUMNS = ["Driver", "LapTime", "Compound", "LapNumber"]
FP3_COLUMNS = ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]

PRACTICE_DIRS = {"FP2": RAW_FP2_DIR, "FP3": RAW_FP3_DIR}
PRACTICE_COLUMNS = {"FP2": FP2_COLUMNS, "FP3": FP3_COLUMNS}
PRACTICE_BEST_LAP_ONLY = {"FP2": False, "FP3": True}


# fetch event metadata for a single round from FastF1 and write to data/raw/events/
# returns the raw DataFrame
# circuit name, is_sprint, and is_street_circuit are derived in clean.py
def get_event_metadata(season, round_num):
    event = fastf1.get_event(season, round_num)

    results = pd.DataFrame([event[["RoundNumber", "Country", "Location", "EventName", "EventDate", "EventFormat"]]])
    results["race_id"] = f"{season}_{round_num:02d}"

    RAW_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RAW_EVENTS_DIR / f"{season}_{round_num:02d}.parquet")

    return results


# fetch practice lap data for a single session (FP2/FP3) from FastF1 and write to data/raw/fp{n}/.
# raises if the session doesn't exist (e.g. sprint weekends) - callers should handle this.
def get_practice_results(season, round_num, session_name):
    session = fastf1.get_session(season, round_num, session_name)
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = session.laps[PRACTICE_COLUMNS[session_name]].copy()

    if PRACTICE_BEST_LAP_ONLY[session_name]:
        laps = laps[session.laps["IsPersonalBest"] == True]

    laps["race_id"] = f"{season}_{round_num:02d}"

    out_dir = PRACTICE_DIRS[session_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    laps.to_parquet(out_dir / f"{season}_{round_num:02d}.parquet")

    return laps


# fetch qualifying results for a single round from FastF1 and write to data/raw/quali/
# returns the raw DataFrame
def get_qualifying_results(season, round_num):
    session = fastf1.get_session(season, round_num, 'Q')
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results[["DriverId", "Abbreviation", "FirstName", "LastName", "TeamId", "Position", "Q1", "Q2", "Q3"]].copy()
    results["race_id"] = f"{season}_{round_num:02d}"

    RAW_QUALI_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RAW_QUALI_DIR / f"{season}_{round_num:02d}.parquet")

    return results


# fetch race results for a single round from FastF1 and write to data/raw/races/
# returns the raw DataFrame
def get_race_results(season, round_num):
    session = fastf1.get_session(season, round_num, 'R')
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results[["DriverId", "FirstName", "LastName", "TeamId", "GridPosition", "Position", "Status", "Points"]].copy()
    results["race_id"] = f"{season}_{round_num:02d}"

    RAW_RACES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RAW_RACES_DIR / f"{season}_{round_num:02d}.parquet")

    return results