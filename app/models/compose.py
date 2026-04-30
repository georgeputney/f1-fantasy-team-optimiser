"""Composes model predictions into expected fantasy points using the scoring rules formula."""

import pandas as pd

from app.data.scoring_rules import (
    DRIVER_QUALI_POSITION_POINTS, DRIVER_RACE_POSITION_POINTS,
    FASTEST_LAP_POINTS, DOTD_POINTS, RACE_PENALTY, 
    POSITION_GAINED_POINTS, OVERTAKE_MADE_POINTS
)


# MVP stubs - replace with model outputs in V2
OVERTAKE_PROB = 0  # TODO V2: replace with predicted overtakes once overtake data is available
DNF_PROB = 0.1           # flat prior, roughly historical average
FASTEST_LAP_PROB = 0.05  # 1 in 20 drivers
DOTD_PRIOR = 0.05        # 1 in 20 drivers


# computes expected fantasy points per driver from predicted finish positions
def compose_drivers(predictions):
    finish_position = predictions["predicted_finish_position"].astype(int)
    quali_position = finish_position  # TODO V2: replace with Model 1 predicted quali position

    quali_points = finish_position.map(lambda p: DRIVER_QUALI_POSITION_POINTS.get(p, 0))
    finish_points = finish_position.map(lambda p: DRIVER_RACE_POSITION_POINTS.get(p, 0))
    positions_gained = quali_position - finish_position  # 0 for MVP

    predictions["expected_fantasy_points"] = (
        quali_points
        + (1 - DNF_PROB) * (finish_points + positions_gained * POSITION_GAINED_POINTS) # weighted by P(finish)
        + OVERTAKE_PROB * OVERTAKE_MADE_POINTS
        + DNF_PROB * RACE_PENALTY
        + FASTEST_LAP_PROB * FASTEST_LAP_POINTS
        + DOTD_PRIOR * DOTD_POINTS
    )

    return predictions.sort_values("expected_fantasy_points", ascending=False).reset_index(drop=True)


# computes expected fantasy points per constructor by summing both drivers' expected points
def compose_constructor(predictions):
    # TODO V2: add constructor quali bonus based on Q2/Q3 advancement
    return ( 
        predictions.groupby("constructor_id")["expected_fantasy_points"]
        .sum()
        .reset_index()
        .sort_values("expected_fantasy_points", ascending=False)
        .reset_index(drop=True)
    )
     