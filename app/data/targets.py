"""Computes actual fantasy points from cleaned race and qualifying results, used as training labels for the models."""

import pandas as pd
import app.data.schemas as schemas
import app.data.scoring_rules as scoring_rules

from app.config import INTERIM_RACES_DIR, INTERIM_QUALI_DIR, INTERIM_SPRINT_DIR, INTERIM_SPRINT_QUALIFYING_DIR, INTERIM_PITSTOPS_DIR, PROCESSED_TARGETS_DIR, INTERIM_RACE_OVERTAKES_DIR, INTERIM_SPRINT_OVERTAKES_DIR


# returns fantasy points for all drivers and constructors for a single sprint weekend
def compute_sprint_targets(season, round_num):
    results = pd.read_parquet(INTERIM_SPRINT_DIR / f"{season}_{round_num:02d}.parquet")

    # merge in sprint overtake counts (0 if file not yet available)
    sprint_overtakes_path = INTERIM_SPRINT_OVERTAKES_DIR / f"{season}_{round_num:02d}.parquet"
    if sprint_overtakes_path.exists():
        sprint_overtakes = pd.read_parquet(sprint_overtakes_path)[["driver_id", "sprint_overtakes"]]
        results = results.merge(sprint_overtakes, on="driver_id", how="left")
        results["sprint_overtakes"] = results["sprint_overtakes"].fillna(0).astype(int)
    else:
        results["sprint_overtakes"] = 0

    drivers_score = results.apply(lambda row: scoring_rules.score_driver_sprint(row["finish_position"], row["positions_gained"], row["dnf_flag"], row["dsq_flag"], row["fastest_lap_flag"], row["sprint_overtakes"]), axis=1)
    driver_targets = pd.DataFrame({
        "race_id": results["race_id"],
        "season": season,
        "round": round_num,
        "asset_id": results["driver_id"],
        "asset_type": "driver",
        "actual_fantasy_points": drivers_score
    })

    constructor_groups = results.groupby(["race_id", "constructor_id"]).agg(
        finish_position=("finish_position", list),
        positions_gained=("positions_gained", list),
        dnf_flag=("dnf_flag", list),
        dsq_flag=("dsq_flag", list),
        fastest_lap_flag=("fastest_lap_flag", list),
        sprint_overtakes=("sprint_overtakes", list),
    ).reset_index()

    constructors_score = constructor_groups.apply(lambda row: scoring_rules.score_constructor_sprint(row["finish_position"], row["positions_gained"], row["dnf_flag"], row["dsq_flag"], row["fastest_lap_flag"], sprint_overtakes=row["sprint_overtakes"]), axis=1)
    constructor_targets = pd.DataFrame({
        "race_id": constructor_groups["race_id"],
        "season": season,
        "round": round_num,
        "asset_id": constructor_groups["constructor_id"],
        "asset_type": "constructor",
        "actual_fantasy_points": constructors_score
    })

    return pd.concat([driver_targets, constructor_targets]).reset_index(drop=True)


# returns fantasy points for all drivers and constructors for a single qualifying session
# drivers missing from qualifying data (e.g. not classified) get the -5 DNS penalty
def compute_qualifying_targets(season, round_num):
    results = pd.read_parquet(INTERIM_QUALI_DIR / f"{season}_{round_num:02d}.parquet")

    # fill missing constructor_id from race entry list (FastF1 returns "nan" for DNS/NC drivers)
    # and add any drivers who raced but are entirely absent from qualifying data
    race_path = INTERIM_RACES_DIR / f"{season}_{round_num:02d}.parquet"
    if race_path.exists():
        race_entries = pd.read_parquet(race_path)[["driver_id", "constructor_id"]].drop_duplicates()
        race_cid = race_entries.set_index("driver_id")["constructor_id"]
        nan_mask = results["constructor_id"].eq("nan") | results["constructor_id"].isna()
        results.loc[nan_mask, "constructor_id"] = results.loc[nan_mask, "driver_id"].map(race_cid)

        missing = race_entries[~race_entries["driver_id"].isin(results["driver_id"])]
        if len(missing) > 0:
            missing = missing.assign(
                race_id=f"{season}_{round_num:02d}", season=season, round=round_num,
                quali_position=float("nan"), q1_time=float("nan"), q2_time=float("nan"), q3_time=float("nan"),
            )
            results = pd.concat([results, missing], ignore_index=True)

    drivers_score = results.apply(lambda row: scoring_rules.score_driver_qualifying(row["quali_position"], row["q1_time"]), axis=1)
    driver_targets = pd.DataFrame({
        "race_id": results["race_id"],
        "season": season,
        "round": round_num,
        "asset_id": results["driver_id"],
        "asset_type": "driver",
        "actual_fantasy_points": drivers_score
    })

    results_with_constructor = results[results["constructor_id"].notna() & results["constructor_id"].ne("nan")]
    constructor_groups = results_with_constructor.groupby(["race_id", "constructor_id"]).agg(
        quali_position=("quali_position", list),
        q1_time=("q1_time", list),
        q2_time=("q2_time", list),
        q3_time=("q3_time", list),
    ).reset_index()

    # Q1 and Q2 each eliminate 6 drivers: 22 grid -> Q2 cutoff P16, 20 grid -> P15
    grid_size = results["driver_id"].nunique()
    q2_cutoff = grid_size - 6

    constructors_score = constructor_groups.apply(lambda row: scoring_rules.score_constructor_qualifying(row["quali_position"], row["q1_time"], row["q2_time"], row["q3_time"], q2_cutoff=q2_cutoff), axis=1)
    constructor_targets = pd.DataFrame({
        "race_id": constructor_groups["race_id"],
        "season": season,
        "round": round_num,
        "asset_id": constructor_groups["constructor_id"],
        "asset_type": "constructor",
        "actual_fantasy_points": constructors_score
    })

    return pd.concat([driver_targets, constructor_targets]).reset_index(drop=True)

    
# returns fantasy points for all drivers and constructors for a single race session
def compute_race_targets(season, round_num):
    results = pd.read_parquet(INTERIM_RACES_DIR / f"{season}_{round_num:02d}.parquet")

    # merge in overtake counts (0 if file not yet available)
    overtakes_path = INTERIM_RACE_OVERTAKES_DIR / f"{season}_{round_num:02d}.parquet"
    if overtakes_path.exists():
        overtakes = pd.read_parquet(overtakes_path)[["driver_id", "race_overtakes"]]
        results = results.merge(overtakes, on="driver_id", how="left")
        results["race_overtakes"] = results["race_overtakes"].fillna(0).astype(int)
    else:
        results["race_overtakes"] = 0

    drivers_score = results.apply(lambda row: scoring_rules.score_driver_race(row["finish_position"], row["positions_gained"], row["dnf_flag"], row["dsq_flag"], row["fastest_lap_flag"], row["dotd_flag"], row["race_overtakes"]), axis=1)
    driver_targets = pd.DataFrame({
        "race_id": results["race_id"],
        "season": season,
        "round": round_num,
        "asset_id": results["driver_id"],
        "asset_type": "driver",
        "actual_fantasy_points": drivers_score
    })

    constructor_groups = results.groupby(["race_id", "constructor_id"]).agg(
        finish_position=("finish_position", list),
        positions_gained=("positions_gained", list),
        dnf_flag=("dnf_flag", list),
        dsq_flag=("dsq_flag", list),
        fastest_lap_flag=("fastest_lap_flag", list),
        race_overtakes=("race_overtakes", list),
    ).reset_index()

    # compute pitstop points per constructor from DHL stationary times
    pitstop_path = INTERIM_PITSTOPS_DIR / f"{season}_{round_num:02d}.parquet"
    pitstop_scores = {}
    if pitstop_path.exists():
        pitstops = pd.read_parquet(pitstop_path)
        best_per_constructor = pitstops.loc[pitstops.groupby("constructor_id")["stationary_s"].idxmin()]
        race_fastest = best_per_constructor["stationary_s"].min()
        for _, row in best_per_constructor.iterrows():
            is_race_fastest = row["stationary_s"] == race_fastest
            pitstop_scores[row["constructor_id"]] = scoring_rules.score_pitstop(row["stationary_s"], is_race_fastest)

    constructors_score = constructor_groups.apply(lambda row: scoring_rules.score_constructor_race(
        row["finish_position"], row["positions_gained"], row["dnf_flag"], row["dsq_flag"], row["fastest_lap_flag"],
        race_overtakes=row["race_overtakes"],
        pitstop_points=pitstop_scores.get(row["constructor_id"], 0),
    ), axis=1)
    constructor_targets = pd.DataFrame({
        "race_id": constructor_groups["race_id"],
        "season": season,
        "round": round_num,
        "asset_id": constructor_groups["constructor_id"],
        "asset_type": "constructor",
        "actual_fantasy_points": constructors_score
    })

    return pd.concat([driver_targets, constructor_targets]).reset_index(drop=True)


# computes total fantasy points per asset per race by summing sprint, qualifying, and race scores, 
# validates against schema, 
# writes to data/processed/targets/
def compute_targets(season, round_num):
    quali_targets = compute_qualifying_targets(season, round_num)
    race_targets = compute_race_targets(season, round_num)

    targets = pd.concat([quali_targets, race_targets]).reset_index(drop=True)

    # add sprint points if this is a sprint weekend
    sprint_path = INTERIM_SPRINT_DIR / f"{season}_{round_num:02d}.parquet"
    if sprint_path.exists():
        sprint_targets = compute_sprint_targets(season, round_num)
        targets = pd.concat([targets, sprint_targets]).reset_index(drop=True)

    targets = targets.groupby(["race_id", "season", "round", "asset_id", "asset_type"], as_index=False)["actual_fantasy_points"].sum()

    schemas.fantasy_targets.validate(targets)

    PROCESSED_TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(PROCESSED_TARGETS_DIR / f"{season}_{round_num:02d}.parquet")

    return targets