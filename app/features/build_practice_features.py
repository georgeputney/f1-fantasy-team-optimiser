"""Practice session features (FP2/FP3) for race weekend predictions - pace gaps, sector deltas, long run pace, and lap counts."""

import pandas as pd

from app.config import INTERIM_FP2_DIR, INTERIM_FP3_DIR, PROCESSED_PRACTICE_FEATURES_DIR


# percentage gap between driver's best lap and the session leader in FP2 and FP3
def gap_to_leader_pct(fp2_laps, fp3_laps, driver_id):
    fp2_best = fp2_laps[fp2_laps["driver_id"] == driver_id]["lap_time"].min()
    fp2_fastest = fp2_laps["lap_time"].min()
    fp2_gap = (fp2_best - fp2_fastest) / fp2_fastest * 100

    fp3_best = fp3_laps[fp3_laps["driver_id"] == driver_id]["lap_time"].min()
    fp3_fastest = fp3_laps["lap_time"].min()
    fp3_gap = (fp3_best - fp3_fastest) / fp3_fastest * 100

    return {"fp2_gap_to_leader_pct": fp2_gap, "fp3_gap_to_leader_pct": fp3_gap}


# percentage gap between driver's best sector times and the fastest sector times in FP3
def sector_gap_to_leader_pct(fp3_laps, driver_id):
    driver_laps = fp3_laps[fp3_laps["driver_id"] == driver_id]
    
    result = {}
    for i, col in enumerate(["sector1_time", "sector2_time", "sector3_time"], 1):
        driver_best = driver_laps[col].min()
        field_fastest = fp3_laps[col].min()
        result[f"fp3_sector{i}_gap_to_leader_pct"] = (driver_best - field_fastest) / field_fastest * 100

    return result


# percentage gap between driver's best FP3 lap and their teammate's best FP3 lap
def teammate_gap_pct(fp3_laps, driver_id):
    constructor_id = fp3_laps[fp3_laps["driver_id"] == driver_id]["constructor_id"].iloc[0]
    teammate_laps = fp3_laps[
        (fp3_laps["constructor_id"] == constructor_id) & 
        (fp3_laps["driver_id"] != driver_id)
    ]
    
    if teammate_laps.empty:
        return {"fp3_teammate_gap_pct": float("nan")}
    
    driver_best = fp3_laps[fp3_laps["driver_id"] == driver_id]["lap_time"].min()
    teammate_best = teammate_laps["lap_time"].min()
    
    return {"fp3_teammate_gap_pct": (driver_best - teammate_best) / teammate_best * 100}


# percentage gap between driver's long run average pace and the field long run average in FP2
def longrun_avg_gap_to_field_pct(fp2_laps, driver_id, field_longrun_avg):
    driver_laps = fp2_laps[fp2_laps["driver_id"] == driver_id].sort_values("lap_number")
    driver_laps = driver_laps.dropna(subset=["lap_time"])

    # identify stints as consecutive lap number sequences
    driver_laps["stint"] = (driver_laps["lap_number"].diff() > 1).cumsum()
    
    # find longest stint with 5+ laps
    long_runs = driver_laps.groupby("stint").filter(lambda x: len(x) >= 5)
    
    if long_runs.empty or pd.isna(field_longrun_avg):
        return {"fp2_longrun_avg_gap_to_field_pct": float("nan")}
    
    driver_longrun_avg = long_runs["lap_time"].mean()

    return {"fp2_longrun_avg_gap_to_field_pct": (driver_longrun_avg - field_longrun_avg) / field_longrun_avg * 100}


# number of laps completed by the driver in FP2
def laps_completed(fp2_laps, driver_id):
    fp2 = len(fp2_laps[fp2_laps["driver_id"] == driver_id])

    return {"fp2_laps_completed": fp2}  # TODO: add fp3_laps_completed if we ingest all FP3 laps


# builds practice features for all drivers in a given race and writes to data/processed/practice_features/
def build_practice_features(season, round_num):
    race_id = f"{season}_{round_num:02d}"
    
    fp2_laps = pd.read_parquet(INTERIM_FP2_DIR / f"{season}_{round_num:02d}.parquet")
    fp3_laps = pd.read_parquet(INTERIM_FP3_DIR / f"{season}_{round_num:02d}.parquet")
    
    drivers = fp3_laps["driver_id"].unique()  # FP3 is the reference session
    
    field_longrun_avg = _compute_field_longrun_avg(fp2_laps)

    rows = []
    for driver_id in drivers:
        features = {"race_id": race_id, "driver_id": driver_id}
        features.update(gap_to_leader_pct(fp2_laps, fp3_laps, driver_id))
        features.update(sector_gap_to_leader_pct(fp3_laps, driver_id))
        features.update(teammate_gap_pct(fp3_laps, driver_id))
        features.update(longrun_avg_gap_to_field_pct(fp2_laps, driver_id, field_longrun_avg))
        features.update(laps_completed(fp2_laps, driver_id))
        rows.append(features)
    
    features_df = pd.DataFrame(rows)
    
    PROCESSED_PRACTICE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(PROCESSED_PRACTICE_FEATURES_DIR / f"{season}_{round_num:02d}.parquet")
    
    return features_df


# computes the field average long run pace from FP2 - called once per race to avoid recomputation per driver
def _compute_field_longrun_avg(fp2_laps):
    avgs = []
    for did in fp2_laps["driver_id"].unique():
        d = fp2_laps[fp2_laps["driver_id"] == did].sort_values("lap_number").dropna(subset=["lap_time"])
        d["stint"] = (d["lap_number"].diff() > 1).cumsum()

        runs = d.groupby("stint").filter(lambda x: len(x) >= 5)

        if not runs.empty:
            avgs.append(runs["lap_time"].mean())

    return sum(avgs) / len(avgs) if avgs else float("nan")
