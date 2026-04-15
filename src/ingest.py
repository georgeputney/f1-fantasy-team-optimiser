"""
Ingest historical F1 data via FastF1 and save as parquet.

Pulls race results, qualifying results, and free practice session lap data
for feature engineering (especially FP2 long-run pace).
"""

import argparse
import logging

import fastf1
import numpy as np
import pandas as pd

from src.config import ALL_SEASONS, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FastF1 will cache HTTP responses here to avoid re-downloading
CACHE_DIR = RAW_DIR / "fastf1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

PRACTICE_SESSIONS = ["FP1", "FP2", "FP3"]


def get_season_schedule(year: int) -> pd.DataFrame:
    """
    Return the event schedule for a given season, excluding testing events.

    Args:
    - year (int): The championship year to fetch.

    Returns:
    - pd.DataFrame: Filtered schedule containing only conventional and sprint-format events.
    """

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule[schedule["EventFormat"].isin(
        ["conventional", "sprint_shootout", "sprint_qualifying"]
    )]


def load_session_results(year: int, round_number: int, session_name: str) -> pd.DataFrame | None:
    """
    Load results for a single session. Returns None on failure.

    Args:
    - year (int): Championship year.
    - round_number (int): Round number within the season.
    - session_name (str): FastF1 session identifier, e.g. "R", "Q", "S", "SQ".

    Returns:
    - pd.DataFrame | None: Session results with Season, RoundNumber, EventName, and
      SessionType columns appended, or None if the session could not be loaded.
    """

    try:
        session = fastf1.get_session(year, round_number, session_name)
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results.copy()
        results["Season"] = year
        results["RoundNumber"] = round_number
        results["EventName"] = session.event["EventName"]
        results["SessionType"] = session_name
        return results

    except Exception as e:
        logger.warning(f"Failed to load results {year} R{round_number} {session_name}: {e}")
        return None


def load_practice_laps(year: int, round_number: int, session_name: str) -> pd.DataFrame | None:
    """
    Load lap-level data from a practice session.

    Returns a DataFrame with one row per driver-lap, including LapTime,
    Sector1/2/3Time, Compound, TyreLife, etc.

    Args:
    - year (int): Championship year.
    - round_number (int): Round number within the season.
    - session_name (str): Practice session identifier, e.g. "FP1", "FP2", "FP3".

    Returns:
    - pd.DataFrame | None: Lap data with Season, RoundNumber, EventName, and
      SessionType columns appended, or None if the session could not be loaded.
    """

    try:
        session = fastf1.get_session(year, round_number, session_name)
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()

        if laps.empty:
            logger.warning(f"No lap data for {year} R{round_number} {session_name}")
            return None

        laps["Season"] = year
        laps["RoundNumber"] = round_number
        laps["EventName"] = session.event["EventName"]
        laps["SessionType"] = session_name

        return laps

    except Exception as e:
        logger.warning(f"Failed to load laps {year} R{round_number} {session_name}: {e}")
        return None


def filter_representative_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Filter practice laps to keep only representative running.

    Removes in/out laps, laps with no time, and outlier laps
    (traffic, mistakes, cooldown — anything > 107% of driver median).

    Args:
    - laps (pd.DataFrame): Raw lap data from load_practice_laps.

    Returns:
    - pd.DataFrame: Filtered lap data with a LapTime_s column added.
    """

    df = laps.copy()

    # drop laps with no time
    df = df.dropna(subset=["LapTime"])

    # convert LapTime to seconds
    df["LapTime_s"] = df["LapTime"].dt.total_seconds()

    # remove pit in/out laps
    if "PitInTime" in df.columns:
        df = df[df["PitInTime"].isna()]
    if "PitOutTime" in df.columns:
        df = df[df["PitOutTime"].isna()]

    # remove outlier laps per driver (> 107% of their median)
    driver_medians = df.groupby("Driver")["LapTime_s"].transform("median")
    df = df[df["LapTime_s"] <= driver_medians * 1.07]

    # remove installation laps (tyre life < 2)
    if "TyreLife" in df.columns:
        df = df[df["TyreLife"] >= 2]

    return df


def classify_lap_type(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Classify laps as 'short_run' or 'long_run'.

    Heuristic: within each driver, group consecutive laps on the same compound
    into stints. Stints of 5+ laps → long run, stints of 1-2 laps → short run.

    Args:
    - laps (pd.DataFrame): Filtered lap data from filter_representative_laps.

    Returns:
    - pd.DataFrame: Laps with a RunType column added ('long_run', 'short_run', or 'unknown').
    """

    df = laps.copy()
    df["RunType"] = "unknown"

    if "TyreLife" not in df.columns or "Compound" not in df.columns:
        return df

    for driver in df["Driver"].unique():
        mask = df["Driver"] == driver
        driver_laps = df.loc[mask].sort_values("LapNumber")

        compound_changes = driver_laps["Compound"] != driver_laps["Compound"].shift(1)
        stint_id = compound_changes.cumsum()

        for current_stint_id in stint_id.unique():
            stint_mask = mask & (stint_id == current_stint_id)
            stint_length = stint_mask.sum()

            if stint_length >= 5:
                df.loc[stint_mask, "RunType"] = "long_run"
            elif stint_length <= 2:
                df.loc[stint_mask, "RunType"] = "short_run"

    return df


def ingest_seasons(seasons: list[int]) -> dict[str, pd.DataFrame]:
    """
    Pull all data for the given seasons.

    Args:
    - seasons (list[int]): List of championship years to ingest.

    Returns:
    - dict[str, pd.DataFrame]: Mapping of dataset name to DataFrame, with keys:
        - 'race_results': race finish data
        - 'quali_results': qualifying positions
        - 'sprint_results': sprint race results (sprint weekends only)
        - 'sprint_quali_results': sprint qualifying results (sprint weekends only)
        - 'practice_laps': filtered lap data from FP1/FP2/FP3
    """

    race_results = []
    quali_results = []
    sprint_results = []
    sprint_quali_results = []
    practice_laps = []

    for year in seasons:
        schedule = get_season_schedule(year)
        logger.info(f"Season {year}: {len(schedule)} events")

        for _, event in schedule.iterrows():
            round_number = event["RoundNumber"]
            event_name = event["EventName"]
            event_format = event.get("EventFormat", "conventional")
            is_sprint = event_format != "conventional"
            logger.info(f"  {year} R{round_number} — {event_name} ({'sprint' if is_sprint else 'conventional'})")

            # race results
            race_df = load_session_results(year, round_number, "R")
            if race_df is not None:
                race_results.append(race_df)

            # qualifying results
            quali_df = load_session_results(year, round_number, "Q")
            if quali_df is not None:
                quali_results.append(quali_df)

            # sprint results and sprint qualifying (sprint weekends only)
            if is_sprint:
                sprint_df = load_session_results(year, round_number, "S")
                if sprint_df is not None:
                    sprint_results.append(sprint_df)

                # sprint qualifying format changed across years:
                #   2021-2022: no sprint qualifying session (normal Q set the sprint grid)
                #   2023: "Sprint Shootout" — FastF1 session identifier "SS"
                #   2024+: "Sprint Qualifying" — FastF1 session identifier "SQ"
                if year >= 2024:
                    sq_df = load_session_results(year, round_number, "SQ")
                elif year == 2023:
                    sq_df = load_session_results(year, round_number, "SS")
                else:
                    sq_df = None

                if sq_df is not None:
                    sprint_quali_results.append(sq_df)

            # free practice laps
            # sprint weekends typically only have FP1 (no FP2/FP3)
            sessions_to_load = ["FP1"] if is_sprint else PRACTICE_SESSIONS
            for fp in sessions_to_load:
                fp_laps = load_practice_laps(year, round_number, fp)
                if fp_laps is not None:
                    fp_laps = filter_representative_laps(fp_laps)
                    fp_laps = classify_lap_type(fp_laps)
                    practice_laps.append(fp_laps)

    data = {}

    if race_results:
        data["race_results"] = pd.concat(race_results, ignore_index=True)
    else:
        raise RuntimeError("No race results loaded")

    if quali_results:
        data["quali_results"] = pd.concat(quali_results, ignore_index=True)

    if sprint_results:
        data["sprint_results"] = pd.concat(sprint_results, ignore_index=True)
        logger.info(f"Loaded sprint results for {len(sprint_results)} events")

    if sprint_quali_results:
        data["sprint_quali_results"] = pd.concat(sprint_quali_results, ignore_index=True)
        logger.info(f"Loaded sprint qualifying results for {len(sprint_quali_results)} events")

    if practice_laps:
        data["practice_laps"] = pd.concat(practice_laps, ignore_index=True)

    return data


def main():
    """Parse CLI arguments and ingest the requested seasons to parquet files."""

    parser = argparse.ArgumentParser(description="Ingest F1 data via FastF1")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=ALL_SEASONS,
        help="Seasons to download (default: all configured seasons)",
    )
    args = parser.parse_args()

    logger.info(f"Ingesting seasons: {args.seasons}")
    data = ingest_seasons(args.seasons)

    for name, df in data.items():
        out_path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {name}: {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
