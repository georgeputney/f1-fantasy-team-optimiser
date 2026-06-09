"""Scoring rules and point lookup tables for F1 fantasy, converts race outcomes to fantasy points.

NOTE: these rules reflect the 2026 season onwards. Earlier seasons had different penalties,
pitstop scoring, and DSQ handling - targets computed for pre-2026 data will not match exactly.
Use manual fantasy point CSVs for historical seasons.
"""

import pandas as pd


# sprint
DRIVER_SPRINT_POSITION_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
SPRINT_FASTEST_LAP_POINTS = 5
SPRINT_DNF_PENALTY = -10
SPRINT_DSQ_PENALTY = -10
SPRINT_POSITION_GAINED_POINTS = 1
SPRINT_MAX_POSITION_LOSS_PENALTY = -10
SPRINT_OVERTAKE_MADE_POINTS = 1


# qualifying
DRIVER_QUALI_POSITION_POINTS = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
CONSTRUCTOR_QUALI_BONUS = {
    (0, 0): -1,
    (1, 0): 1,
    (2, 0): 3,
    (1, 1): 5,
    (2, 1): 5,
    (2, 2): 10,
}
QUALI_PENALTY = -5  # DNS / not classified in qualifying
CONSTRUCTOR_QUALI_DSQ_PENALTY = -5  # additional per-driver constructor penalty for qualifying DSQ


# race
DRIVER_RACE_POSITION_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
FASTEST_LAP_POINTS = 10
DOTD_POINTS = 10
POSITION_GAINED_POINTS = 1
OVERTAKE_MADE_POINTS = 1
RACE_DNF_PENALTY = -20
RACE_DSQ_DRIVER_PENALTY = -20  # driver gets flat penalty, no bonuses (FL/DOTD/overtakes stripped)
RACE_DSQ_CONSTRUCTOR_PENALTY = -20  # additional per-driver constructor penalty for race DSQ
RACE_PENALTY = RACE_DNF_PENALTY  # alias used by compose.py prediction pipeline

# constructor pitstop scoring - time-bracket-based
PITSTOP_WORLD_RECORD = 1.80  # McLaren, Qatar 2023
PITSTOP_WORLD_RECORD_BONUS = 15
PITSTOP_BRACKETS = [
    (2.00, 20),
    (2.20, 10),
    (2.50, 5),
    (3.005, 2),  # inclusive of 3.00 ("over 3.0s" = 0)
]
PITSTOP_RACE_FASTEST_BONUS = 5


# calculate fantasy points for a driver's sprint result
# DNF/DSQ replaces position and positions-gained with a flat penalty,
# but overtakes and fastest lap still count
def score_driver_sprint(position, positions_gained, dnf_flag, dsq_flag, fastest_lap_flag, sprint_overtakes=0):
    if dsq_flag:
        score = SPRINT_DSQ_PENALTY
    elif dnf_flag:
        score = SPRINT_DNF_PENALTY
    else:
        score = DRIVER_SPRINT_POSITION_POINTS.get(position, 0)
        if pd.isna(positions_gained):
            positions_gained = 0
        capped = max(positions_gained, SPRINT_MAX_POSITION_LOSS_PENALTY)
        score += capped * SPRINT_POSITION_GAINED_POINTS

    if fastest_lap_flag:
        score += SPRINT_FASTEST_LAP_POINTS

    score += sprint_overtakes * SPRINT_OVERTAKE_MADE_POINTS

    return score


# calculate fantasy points for a constructor's sprint result
# DSQ'd drivers incur an additional constructor penalty on top of their driver score
def score_constructor_sprint(positions, positions_gained, dnf_flags, dsq_flags, fastest_lap_flags, sprint_overtakes=None):
    if sprint_overtakes is None:
        sprint_overtakes = [0] * len(positions)
    score = sum(
        score_driver_sprint(p, pg, dnf, dsq, fl, so)
        for p, pg, dnf, dsq, fl, so in zip(
            positions, positions_gained, dnf_flags, dsq_flags, fastest_lap_flags, sprint_overtakes
        )
    )
    score += sum(SPRINT_DSQ_PENALTY for dsq in dsq_flags if dsq)
    return score


# calculate fantasy points for a driver's qualifying result
def score_driver_qualifying(position, q1_time):
    if pd.isna(q1_time):
        return QUALI_PENALTY

    return DRIVER_QUALI_POSITION_POINTS.get(position, 0)


# calculate fantasy points for a constructor's qualifying result, including Q2/Q3 bonus
# DSQ'd drivers (no Q1 but have Q2/Q3 times) incur an additional constructor penalty
# Q2/Q3 bonus based on reaching each session: Q2 cutoff depends on grid size, Q3 is always P1-10
def score_constructor_qualifying(positions, q1_times, q2_times, q3_times, q2_cutoff=15):
    score = sum(
        score_driver_qualifying(p, q1) for p, q1 in zip(positions, q1_times)
    )

    # DSQ = no Q1 time but has Q2 or Q3 (set times before disqualification)
    dsq_count = sum(
        1 for q1, q2, q3 in zip(q1_times, q2_times, q3_times)
        if pd.isna(q1) and (not pd.isna(q2) or not pd.isna(q3))
    )
    score += dsq_count * CONSTRUCTOR_QUALI_DSQ_PENALTY

    # Q2/Q3 bonus: classified drivers (have Q1 time) who reached each session
    # Q2 = had a Q2 time or position within cutoff, Q3 = position P1-10
    q2_count = sum(
        1 for p, q1, q2 in zip(positions, q1_times, q2_times)
        if not pd.isna(q1) and (not pd.isna(q2) or (not pd.isna(p) and p <= q2_cutoff))
    )
    q3_count = sum(1 for p, q1 in zip(positions, q1_times) if not pd.isna(q1) and not pd.isna(p) and p <= 10)

    score += CONSTRUCTOR_QUALI_BONUS.get((q2_count, q3_count), 0)

    return score


# calculate fantasy points for a driver's race result
# DSQ'd driver gets flat penalty, no bonuses
# DNF replaces position and positions-gained with a flat penalty,
# but overtakes, fastest lap, and DOTD still count
def score_driver_race(position, positions_gained, dnf_flag, dsq_flag, fastest_lap_flag, dotd_flag, race_overtakes=0):
    if dsq_flag:
        return RACE_DSQ_DRIVER_PENALTY

    if dnf_flag:
        score = RACE_DNF_PENALTY
    else:
        score = DRIVER_RACE_POSITION_POINTS.get(position, 0)
        if pd.isna(positions_gained):
            positions_gained = 0
        score += positions_gained * POSITION_GAINED_POINTS

    if fastest_lap_flag:
        score += FASTEST_LAP_POINTS
    if dotd_flag:
        score += DOTD_POINTS

    score += race_overtakes * OVERTAKE_MADE_POINTS

    return score


# score a constructor's fastest pit stop against the bracket thresholds
def score_pitstop(best_time, is_race_fastest=False):
    if pd.isna(best_time):
        return 0

    points = 0
    if best_time < PITSTOP_WORLD_RECORD:
        points = 20 + PITSTOP_WORLD_RECORD_BONUS
    else:
        for threshold, pts in PITSTOP_BRACKETS:
            if best_time < threshold:
                points = pts
                break

    if is_race_fastest:
        points += PITSTOP_RACE_FASTEST_BONUS

    return points


# calculate fantasy points for a constructor's race result
# DSQ'd drivers incur an additional constructor penalty on top of their driver score
def score_constructor_race(positions, positions_gained, dnf_flags, dsq_flags, fastest_lap_flags, race_overtakes=None, pitstop_points=0):
    if race_overtakes is None:
        race_overtakes = [0] * len(positions)
    score = sum(
        score_driver_race(p, pg, dnf, dsq, fl, dotd_flag=False, race_overtakes=ro)
        for p, pg, dnf, dsq, fl, ro in zip(
            positions, positions_gained, dnf_flags, dsq_flags, fastest_lap_flags, race_overtakes
        )
    )
    score += sum(RACE_DSQ_CONSTRUCTOR_PENALTY for dsq in dsq_flags if dsq)
    score += pitstop_points

    return score
