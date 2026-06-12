"""Composes model predictions into expected fantasy points using the scoring rules formula."""

import numpy as np
import pandas as pd

from app.config import INTERIM_PITSTOPS_DIR
from app.data.scoring_rules import (
    CONSTRUCTOR_QUALI_BONUS, DRIVER_QUALI_POSITION_POINTS, DRIVER_RACE_POSITION_POINTS,
    FASTEST_LAP_POINTS, DOTD_POINTS, RACE_PENALTY, POSITION_GAINED_POINTS, OVERTAKE_MADE_POINTS,
    DRIVER_SPRINT_POSITION_POINTS, SPRINT_FASTEST_LAP_POINTS, SPRINT_OVERTAKE_MADE_POINTS,
    score_pitstop,
)

_positions = np.arange(1, 21)
_exp_weights = np.exp(-0.275 * (_positions - 1))
_exp_probs = _exp_weights / _exp_weights.sum()

FASTEST_LAP_PROB = dict(zip(_positions, _exp_probs))

_DOTD_FALLBACK = 0.05    # flat prior used when no dotd predictor is provided


# compute expected pitstop bracket points per constructor from historical DHL data.
# only uses 2022+ data (current wheel/tyre regulations made stops slower) and applies
# exponential recency weighting so recent form dominates. predicts a weighted time
# distribution per constructor and computes E[bracket_pts] from that, which handles
# the nonlinear step-function scoring better than averaging past scores directly.
_PITSTOP_MIN_SEASON = 2022
_PITSTOP_HALF_LIFE = 8  # races; weight halves every 8 races back


def expected_pitstop_points(season, round_num):
    files = sorted(INTERIM_PITSTOPS_DIR.glob("*.parquet"))
    prior = []
    for f in files:
        parts = f.stem.split("_")
        s, r = int(parts[0]), int(parts[1])
        if s >= _PITSTOP_MIN_SEASON and (s, r) < (season, round_num):
            prior.append(pd.read_parquet(f))

    if not prior:
        return {}

    pitstops = pd.concat(prior)
    best = pitstops.loc[pitstops.groupby(["season", "round", "constructor_id"])["stationary_s"].idxmin()].copy()

    # recency weight: exponential decay from the most recent race
    race_keys = best[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys["_race_idx"] = range(len(race_keys))
    max_idx = race_keys["_race_idx"].max()
    best = best.merge(race_keys, on=["season", "round"])
    best["_weight"] = np.exp(-np.log(2) / _PITSTOP_HALF_LIFE * (max_idx - best["_race_idx"]))

    # for each constructor, compute E[bracket_pts] from the weighted time distribution.
    # sample from the empirical distribution (weighted) and score each sample.
    rng = np.random.default_rng(42)
    n_samples = 5000
    result = {}
    for cid, group in best.groupby("constructor_id"):
        times = group["stationary_s"].values
        weights = group["_weight"].values
        probs = weights / weights.sum()
        samples = rng.choice(times, size=n_samples, p=probs)
        scores = np.array([score_pitstop(t, is_race_fastest=False) for t in samples])
        result[cid] = float(scores.mean())

    return result


# computes expected fantasy points per driver from predicted quali and finish positions.
# predict_overtakes is an optional callable from build_overtake_predictor() - if provided,
# expected overtake points are included per driver using their driver_id, location, and season.
# predict_dotd is an optional callable from build_dotd_predictor() - if provided, per-driver
# DOTD probabilities are used instead of a flat prior. probabilities sum to 1.0 across the field.
# optionally accepts a dnf_prob series for exploration - when provided, race points are weighted by P(finish)
# and a DNF penalty is applied. production code omits dnf_prob (no compose-level DNF adjustment).
def compose_drivers(predictions, location=None, season=None, predict_overtakes=None, predict_dotd=None, dnf_prob=None, fastest_lap_prob=None):
    quali_position = predictions["predicted_quali_position"].astype(int)
    finish_position = predictions["predicted_finish_position"].astype(int)

    quali_points = quali_position.map(lambda p: DRIVER_QUALI_POSITION_POINTS.get(p, 0))
    finish_points = finish_position.map(lambda p: DRIVER_RACE_POSITION_POINTS.get(p, 0))
    positions_gained = quali_position - finish_position
    positions_gained_points = positions_gained * POSITION_GAINED_POINTS

    if dnf_prob is not None:
        race_component = (
            (1 - dnf_prob) * (finish_points + positions_gained_points)
            + dnf_prob * RACE_PENALTY
        )
    else:
        race_component = finish_points + positions_gained_points

    fl_prob = (
        fastest_lap_prob
        if fastest_lap_prob is not None
        else quali_position.map(FASTEST_LAP_PROB).fillna(0)
    )
    if predict_overtakes is not None and location is not None and season is not None:
        expected_overtakes = predictions.apply(
            lambda r: predict_overtakes(r["driver_id"], location, season, int(r["predicted_quali_position"])),
            axis=1,
        )
    else:
        expected_overtakes = pd.Series(0.0, index=predictions.index)

    dotd_prob = (
        predict_dotd(predictions["driver_id"]).values
        if predict_dotd is not None
        else pd.Series(_DOTD_FALLBACK, index=predictions.index)
    )

    predictions["points_quali"] = quali_points
    predictions["points_finish"] = finish_points
    predictions["points_positions_gained"] = positions_gained_points
    predictions["expected_overtakes"] = expected_overtakes
    predictions["prob_fl"] = fl_prob           # probability 0-1, sums to ~1.0 across all drivers
    predictions["prob_dotd"] = dotd_prob       # probability 0-1, sums to 1.0 across all drivers
    predictions["points_sprint"] = 0.0

    predictions["expected_fantasy_points"] = (
        quali_points
        + race_component
        + expected_overtakes * OVERTAKE_MADE_POINTS
        + fl_prob * FASTEST_LAP_POINTS
        + dotd_prob * DOTD_POINTS
    )

    # add sprint points if sprint weekend
    if "sprint_quali_position" in predictions.columns and predictions["sprint_quali_position"].notna().any():
        sprint_position = predictions["sprint_quali_position"].fillna(20).astype(int)
        sprint_finish_points = sprint_position.map(lambda p: DRIVER_SPRINT_POSITION_POINTS.get(p, 0))
        sprint_fl_prob = sprint_position.map(FASTEST_LAP_PROB).fillna(0)
        # expected sprint overtakes: 1/3 of predicted race overtakes (sprint is ~1/3 race length)
        sprint_expected_overtakes = expected_overtakes / 3
        sprint_points = sprint_finish_points + sprint_fl_prob * SPRINT_FASTEST_LAP_POINTS + sprint_expected_overtakes * SPRINT_OVERTAKE_MADE_POINTS
        predictions["points_sprint"] = sprint_points
        predictions["expected_fantasy_points"] += sprint_points

    return predictions.sort_values("expected_fantasy_points", ascending=False).reset_index(drop=True)

     
# computes expected fantasy points per constructor by summing both drivers' expected points
# plus Q2/Q3 quali bonus and expected pitstop bracket points.
# pitstop_pts is an optional dict of constructor_id -> expected bracket points (from expected_pitstop_points()).
def compose_constructor(predictions, pitstop_pts=None):
    q2_cutoff = (len(predictions) + 10) // 2  # top half of non-Q3 drivers advanced to Q2

    predictions = predictions.copy()

    predictions["_q2"] = (predictions["predicted_quali_position"] <= q2_cutoff).astype(int)
    predictions["_q3"] = (predictions["predicted_quali_position"] <= 10).astype(int)

    constructor_points = predictions.groupby("constructor_id").agg(
        expected_fantasy_points=("expected_fantasy_points", "sum"),
        _q2=("_q2", "sum"),
        _q3=("_q3", "sum"),
    ).reset_index()

    constructor_points["expected_fantasy_points"] += constructor_points.apply(
        lambda row: CONSTRUCTOR_QUALI_BONUS.get((int(row["_q2"]), int(row["_q3"])), 0), axis=1
    )

    if pitstop_pts:
        constructor_points["points_pitstop"] = constructor_points["constructor_id"].map(pitstop_pts).fillna(0)
        constructor_points["expected_fantasy_points"] += constructor_points["points_pitstop"]

    return (
        constructor_points[["constructor_id", "expected_fantasy_points"]]
        .sort_values("expected_fantasy_points", ascending=False)
        .reset_index(drop=True)
    )
