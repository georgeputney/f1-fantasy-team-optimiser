"""
Fetches raw F1 session data from the FastF1 API and writes it to parquet files in data/raw/.

Enables the FastF1 cache on import. Each function fetches one session type for one race weekend.
"""

import fastf1
import pandas as pd

from app.config import FASTF1_CACHE_DIR, RAW_EVENTS_DIR, RAW_RACES_DIR, RAW_QUALI_DIR

fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


# fetch event metadata for a single round from FastF1 and write to data/raw/events/
# returns the raw DataFrame
# circuit name, is_sprint, and is_street_circuit are derived in clean.py
def get_event_metadata(season, round_num):
    event = fastf1.get_event(season, round_num)

    results = pd.DataFrame([event[["RoundNumber", "Country", "Location", "EventName", "EventDate", "EventFormat"]]])
    results["race_id"] = f"{season}_{round_num}"

    RAW_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RAW_EVENTS_DIR / f"{season}_{round_num}.parquet")

    return results


# fetch race results for a single round from FastF1 and write to data/raw/races/
# returns the raw DataFrame
def get_race_results(season, round_num):
    session = fastf1.get_session(season, round_num, 'R')
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results[["DriverId", "FirstName", "LastName", "TeamId", "GridPosition", "Position", "Status", "Points"]].copy()
    results["race_id"] = f"{season}_{round_num}"

    RAW_RACES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RAW_RACES_DIR / f"{season}_{round_num}.parquet")

    return results


# fetch qualifying results for a single round from FastF1 and write to data/raw/quali/
# returns the raw DataFrame
def get_qualifying_results(season, round_num):
    session = fastf1.get_session(season, round_num, 'Q')
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results[["DriverId", "FirstName", "LastName", "TeamId", "Position", "Q1", "Q2", "Q3"]].copy()
    results["race_id"] = f"{season}_{round_num}"

    RAW_QUALI_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RAW_QUALI_DIR / f"{season}_{round_num}.parquet")

    return results