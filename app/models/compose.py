"""Composes model predictions into expected fantasy points using the scoring rules formula."""

import numpy as np
import pandas as pd

from app.data.scoring_rules import (
    CONSTRUCTOR_QUALI_BONUS, DRIVER_QUALI_POSITION_POINTS, DRIVER_RACE_POSITION_POINTS,
    FASTEST_LAP_POINTS, DOTD_POINTS, RACE_PENALTY, POSITION_GAINED_POINTS, OVERTAKE_MADE_POINTS
)

_positions = np.arange(1, 21)
_exp_weights = np.exp(-0.275 * (_positions - 1))
_exp_probs = _exp_weights / _exp_weights.sum()

FASTEST_LAP_PROB = dict(zip(_positions, _exp_probs))

# MVP stubs - replace with model outputs in V2
OVERTAKE_PROB = 0  # TODO V2: replace with predicted overtakes once overtake data is available
DOTD_PRIOR = 0.05        # 1 in 20 drivers


# computes expected fantasy points per driver from predicted quali and finish positions.
# optionally accepts a dnf_prob series for exploration - when provided, race points are weighted by P(finish)
# and a DNF penalty is applied. production code omits dnf_prob (no compose-level DNF adjustment).
def compose_drivers(predictions, dnf_prob=None, fastest_lap_prob=None):
    quali_position = predictions["predicted_quali_position"].astype(int)
    finish_position = predictions["predicted_finish_position"].astype(int)

    quali_points = quali_position.map(lambda p: DRIVER_QUALI_POSITION_POINTS.get(p, 0))
    finish_points = finish_position.map(lambda p: DRIVER_RACE_POSITION_POINTS.get(p, 0))
    positions_gained = quali_position - finish_position

    if dnf_prob is not None:
        race_component = (
            (1 - dnf_prob) * (finish_points + positions_gained * POSITION_GAINED_POINTS)
            + dnf_prob * RACE_PENALTY
        )
    else:
        race_component = finish_points + positions_gained * POSITION_GAINED_POINTS

    fl_prob = (
        fastest_lap_prob
        if fastest_lap_prob is not None
        else quali_position.map(FASTEST_LAP_PROB).fillna(0)
    )

    predictions["expected_fantasy_points"] = (
        quali_points
        + race_component
        + OVERTAKE_PROB * OVERTAKE_MADE_POINTS
        + fl_prob * FASTEST_LAP_POINTS
        + DOTD_PRIOR * DOTD_POINTS
    )

    return predictions.sort_values("expected_fantasy_points", ascending=False).reset_index(drop=True)

     
# computes expected fantasy points per constructor by summing both drivers' expected points plus Q2/Q3 quali bonus
def compose_constructor(predictions):
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

    return (
        constructor_points[["constructor_id", "expected_fantasy_points"]]
        .sort_values("expected_fantasy_points", ascending=False)
        .reset_index(drop=True)
    )
