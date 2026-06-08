"""Overtake prediction model - expected overtake points per driver for a given race."""

import pandas as pd

from app.config import INTERIM_RACE_OVERTAKES_DIR, INTERIM_EVENTS_DIR, INTERIM_QUALI_DIR


def _load_overtake_history() -> pd.DataFrame:
    frames = []
    for f in sorted(INTERIM_RACE_OVERTAKES_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    return pd.concat(frames).reset_index(drop=True)


# uses a multiplicative model: driver_index * circuit_index * grid_factor * season_mean.
# all indices are season-normalised (1.0 = season average), defaults to 1.0 for
# unknown drivers, circuits, or grid positions
def build_overtake_predictor():
    ot = _load_overtake_history()

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
        pd.read_parquet(f) for f in sorted(INTERIM_EVENTS_DIR.glob("*.parquet"))
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
