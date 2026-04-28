"""Scoring rules and point lookup tables for F1 fantasy, converts race outcomes to fantasy points."""

import pandas as pd


# qualifying
DRIVER_QUALI_POSITION_POINTS = {
    1: 10,
    2: 9,
    3: 8,
    4: 7,
    5: 6,
    6: 5,
    7: 4,
    8: 3, 
    9: 2,
    10: 1,
}
CONSTRUCTOR_QUALI_BONUS = {
    (0, 0): -1,
    (1, 0): 1,
    (2, 0): 3,
    (1, 1): 5,
    (2, 1): 5,
    (2, 2): 10,
}
QUALI_PENALTY = -5


# race
DRIVER_RACE_POSITION_POINTS = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4, 
    9: 2,
    10: 1,
}
FASTEST_LAP_POINTS = 10
DOTD_POINTS = 10
POSITION_GAINED_POINTS = 1
OVERTAKE_MADE_POINTS = 1 # TODO: wire up once overtake data is available
RACE_PENALTY = -20


# calculate fantasy points for a driver's qualifying result
def score_driver_qualifying(position, q1_time):
    if pd.isna(q1_time):
        return QUALI_PENALTY
    
    return DRIVER_QUALI_POSITION_POINTS.get(position, 0)


# calculate fantasy points for a constructor's qualifying result, including Q2/Q3 bonus
def score_constructor_qualifying(positions, q1_times, q2_times, q3_times):
    score = sum(
        score_driver_qualifying(p, t) for p, t in zip(positions, q1_times)
    )

    q2_count = sum(1 for t in q2_times if not pd.isna(t))
    q3_count = sum(1 for t in q3_times if not pd.isna(t))

    score += CONSTRUCTOR_QUALI_BONUS.get((q2_count, q3_count), 0)

    return score


# calculate fantasy points for a driver's race result
def score_driver_race(position, positions_gained, dnf_flag, dsq_flag, fastest_lap_flag, dotd_flag):
    if dnf_flag or dsq_flag:
        return RACE_PENALTY
    
    score = DRIVER_RACE_POSITION_POINTS.get(position, 0)

    if pd.isna(positions_gained):
        positions_gained = 0

    # positions_gained is signed: gains add, losses subtract
    score += positions_gained * POSITION_GAINED_POINTS

    if fastest_lap_flag:
        score += FASTEST_LAP_POINTS
    if dotd_flag:
        score += DOTD_POINTS

    return score


# calculate fantasy points for a constructor's race result, pitstop scoring added in V2
def score_constructor_race(positions, positions_gained, dnf_flags, dsq_flags, fastest_lap_flags):
    score = sum(
        score_driver_race(p, pg, dnf, dsq, fl, dotd_flag=False)
        for p, pg, dnf, dsq, fl in zip(
            positions, positions_gained, dnf_flags, dsq_flags, fastest_lap_flags
        )
    )
    # TODO: add pitstop points
 
    return score