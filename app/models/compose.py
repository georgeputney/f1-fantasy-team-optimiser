"""Composes model predictions into expected fantasy points using the scoring rules formula."""

import pandas as pd

from app.data.scoring_rules import (
    CONSTRUCTOR_QUALI_BONUS, DRIVER_QUALI_POSITION_POINTS, DRIVER_RACE_POSITION_POINTS,
    FASTEST_LAP_POINTS, DOTD_POINTS, RACE_PENALTY, 
    POSITION_GAINED_POINTS, OVERTAKE_MADE_POINTS
)


# MVP stubs - replace with model outputs in V2
OVERTAKE_PROB = 0  # TODO V2: replace with predicted overtakes once overtake data is available
FASTEST_LAP_PROB = 0.05  # 1 in 20 drivers
DOTD_PRIOR = 0.05        # 1 in 20 drivers


# computes expected fantasy points per driver from predicted quali and finish positions
def compose_drivers(predictions):
    quali_position = predictions["predicted_quali_position"].astype(int)
    finish_position = predictions["predicted_finish_position"].astype(int)
    dnf_prob = predictions["dnf_prob"]

    quali_points = quali_position.map(lambda p: DRIVER_QUALI_POSITION_POINTS.get(p, 0))
    finish_points = finish_position.map(lambda p: DRIVER_RACE_POSITION_POINTS.get(p, 0))
    positions_gained = quali_position - finish_position

    predictions["expected_fantasy_points"] = (
        quali_points
        + (1 - dnf_prob) * (finish_points + positions_gained * POSITION_GAINED_POINTS) # weighted by P(finish)
        + OVERTAKE_PROB * OVERTAKE_MADE_POINTS
        + dnf_prob * RACE_PENALTY
        + FASTEST_LAP_PROB * FASTEST_LAP_POINTS
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
