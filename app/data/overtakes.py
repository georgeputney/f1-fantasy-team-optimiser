"""Overtake prediction model — expected overtake points per driver for a given race."""

import pandas as pd

from app.config import RACE_OVERTAKES_DIR, INTERIM_EVENTS_DIR


def _load_overtake_history() -> pd.DataFrame:
    # swap this for a processed/overtakes/ parquet glob once overtakes are derived from telemetry
    frames = []
    for f in sorted(RACE_OVERTAKES_DIR.glob("*.csv")):
        season, round_ = f.stem.split("_")
        df = pd.read_csv(f)
        df["season"] = int(season)
        df["round"] = int(round_)
        frames.append(df)
    return pd.concat(frames).reset_index(drop=True)


def build_overtake_predictor():
    """
    Returns a predict_overtakes(driver_id, location, season) function built from
    historical overtake data.

    Uses a multiplicative model: driver_index × circuit_index × season_mean.
    Both indices are season-normalised (1.0 = season average). Defaults to 1.0 for
    unknown drivers or circuits.
    """
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

    # per-circuit index (min 2 races) — requires events parquets for location names
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

    def predict_overtakes(driver_id: str, location: str, season: int) -> float:
        """Expected overtakes for a driver at a circuit in a given season."""
        d = driver_index.get(driver_id, 1.0)
        c = circuit_index.get(location, 1.0)
        s = season_means.get(season, season_means[max(season_means)])
        return round(d * c * s, 2)

    return predict_overtakes
