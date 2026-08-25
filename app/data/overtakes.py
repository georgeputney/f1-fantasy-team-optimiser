"""Overtake prediction model - expected overtake points per driver for a given race."""

import pandas as pd

from app.config import INTERIM_RACE_OVERTAKES_DIR, INTERIM_EVENTS_DIR, INTERIM_QUALI_DIR


# per-round files are named {season}_{round:02d}.parquet - true if a file's (season, round)
# is strictly before the cutoff, so a walk-forward caller (e.g. backtest) can exclude any race
# that hadn't happened yet as of the round being predicted
def _is_before(path, before):
    if before is None:
        return True
    season, round_num = (int(x) for x in path.stem.split("_"))
    return (season, round_num) < before


def _load_overtake_history(before=None) -> pd.DataFrame:
    frames = []
    for f in sorted(INTERIM_RACE_OVERTAKES_DIR.glob("*.parquet")):
        if _is_before(f, before):
            frames.append(pd.read_parquet(f))
    return pd.concat(frames).reset_index(drop=True)


# uses a multiplicative model: driver_index * circuit_index * grid_factor * season_mean.
# all indices are season-normalised (1.0 = season average), defaults to 1.0 for
# unknown drivers, circuits, or grid positions.
# `before` restricts the calibration data to races strictly before (season, round) - used by the
# walk-forward backtest so a later round's actual results can't leak into an earlier round's
# prediction; live callers (generate-reports, optimise-team) omit it and use everything available.
def build_overtake_predictor(before=None):
    ot = _load_overtake_history(before)

    # season baselines
    season_means = ot.groupby("season")["race_overtakes"].mean().to_dict()

    # season-normalised overtake index per driver-race
    ot["overtake_index"] = ot["race_overtakes"] / ot.groupby("season")["race_overtakes"].transform("mean")

    # per-driver index (min 10 races for a stable estimate)
    driver_index = (
        ot.groupby("driver_id")
        .agg(avg_index=("overtake_index", "mean"), races=("overtake_index", "count"))
        .query("races >= 10")["avg_index"]
    )

    # per-circuit index (min 2 races) -- requires events parquets for location names
    events = pd.concat([
        pd.read_parquet(f) for f in sorted(INTERIM_EVENTS_DIR.glob("*.parquet")) if _is_before(f, before)
    ])[["season", "round", "location"]].drop_duplicates()

    race_totals = (
        ot.groupby(["season", "round"])
        .agg(per_driver_avg=("race_overtakes", "mean"))
        .reset_index()
        .join(ot.groupby("season")["race_overtakes"].mean().rename("season_avg"), on="season")
    )
    race_totals["overtake_index"] = race_totals["per_driver_avg"] / race_totals["season_avg"]
    race_totals = race_totals.merge(events, on=["season", "round"], how="left")

    circuit_index = (
        race_totals.groupby("location")
        .agg(avg_index=("overtake_index", "mean"), races=("overtake_index", "count"))
        .query("races >= 2")["avg_index"]
    )

    # per-grid-position factor -- drivers starting further back overtake more
    # merge quali positions onto overtake history to compute grid factor
    quali_frames = []
    for f in sorted(INTERIM_QUALI_DIR.glob("*.parquet")):
        if _is_before(f, before):
            quali_frames.append(pd.read_parquet(f)[["season", "round", "driver_id", "quali_position"]])
    all_quali = pd.concat(quali_frames)
    ot_with_grid = ot.merge(all_quali, on=["season", "round", "driver_id"], how="left")
    ot_with_grid = ot_with_grid.dropna(subset=["quali_position"])
    ot_with_grid["quali_position"] = ot_with_grid["quali_position"].astype(int)

    grid_factor = ot_with_grid.groupby("quali_position")["overtake_index"].mean()

    def predict_overtakes(driver_id: str, location: str, season: int, quali_position: int = None) -> float:
        """Expected overtakes for a driver at a circuit in a given season."""
        d = driver_index.get(driver_id, 1.0)
        c = circuit_index.get(location, 1.0)
        g = grid_factor.get(quali_position, 1.0) if quali_position is not None else 1.0
        s = season_means.get(season, season_means[max(season_means)])
        return round(d * c * g * s, 2)

    return predict_overtakes
