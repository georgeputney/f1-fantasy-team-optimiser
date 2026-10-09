"""Grid penalties - turning a predicted qualifying order plus known penalties into a race grid, and the
extra overtakes a driver starting out of position makes recovering.

Penalised drivers finish about where their pace says (walk-forward test over every 2020-2026 race with a
3+ place grid drop: the model's finish prediction for them was already unbiased, and pushing it back made
things worse), so the finish prediction is left alone for most drivers. The exception is a front-runner:
one qualifying near the front and dropped well back has to pass the other fast cars, and since 2019 those
have finished ~1.5 places further back than unpenalised drivers from the same qualifying slot, so their
finish is pushed back by that much (front_runner_finish_shift). What changes is everything scored off the grid:
positions gained, and overtakes, which rise with the places a driver has to recover.
"""

import numpy as np
import pandas as pd

from app.config import GRID_PENALTIES_DIR, INTERIM_RACES_DIR, INTERIM_QUALI_DIR, INTERIM_RACE_OVERTAKES_DIR

BACK_OF_GRID = "back"


# hand-entered penalties for one round, as {driver_id: places (int) or "back"}. the season file has one
# row per penalty - round, driver_id, penalty - where penalty is a number of grid places, or "back" for
# a back-of-grid start (a big engine penalty, or a pit-lane start)
def load_grid_penalties(season, round_num):
    path = GRID_PENALTIES_DIR / f"{season}.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str)
    df = df[df["round"].astype(int) == round_num]
    return {
        row["driver_id"].strip(): BACK_OF_GRID if row["penalty"].strip().lower() == BACK_OF_GRID else int(row["penalty"])
        for _, row in df.iterrows()
    }


# the grid drops a past race actually had, in the same shape load_grid_penalties returns - used by the
# backtest to replay penalties as if they'd been entered by hand. a drop that put the driver at the very
# back is "back", since the number of places then depends on where they qualified
def historical_grid_penalties(season, round_num, min_drop=3):
    race_path = INTERIM_RACES_DIR / f"{season}_{round_num:02d}.parquet"
    quali_path = INTERIM_QUALI_DIR / f"{season}_{round_num:02d}.parquet"
    if not race_path.exists() or not quali_path.exists():
        return {}

    race = pd.read_parquet(race_path, columns=["driver_id", "grid_position"])
    quali = pd.read_parquet(quali_path, columns=["driver_id", "quali_position"])
    df = race.merge(quali, on="driver_id").dropna()
    n = len(race)

    penalties = {}
    for _, row in df.iterrows():
        drop = int(row["grid_position"] - row["quali_position"])
        if drop >= min_drop:
            penalties[row["driver_id"]] = BACK_OF_GRID if row["grid_position"] >= n - 1 else drop
    return penalties


# race grid from a qualifying order and penalties. a penalised driver drops the given number of places
# from where they qualified (or to the back), and everyone they drop behind moves up one. returns grid
# slots aligned to quali_position's index
def apply_grid_penalties(driver_ids, quali_position, penalties):
    n = len(quali_position)
    keys = quali_position.astype(float).copy()
    for i, did in zip(quali_position.index, driver_ids):
        pen = penalties.get(did)
        if pen is None:
            continue
        # +0.5 sorts the penalised car behind whoever qualified in its target slot; back-of-grid
        # starters keep their qualifying order among themselves
        keys[i] = n + 1 + quali_position[i] / 100 if pen == BACK_OF_GRID else min(quali_position[i] + pen, n) + 0.5
    return keys.rank(method="first").astype(int)


# extra overtakes per place a driver has to recover (grid slot minus finish), fitted on finishers in races
# strictly before `before` - the slope of overtakes against positions gained, 0.14-0.29 across 2024-2026.
# one flat rate for every circuit: scaling it by the circuit's overtake index tested a touch more accurate
# on 47 penalised drivers but worse on the 2024-2026 backtest, with a much less stable fitted rate, so the
# evidence didn't justify it. the base overtake prediction already accounts for the circuit
def recovery_overtake_rate(before=None):
    rows = []
    for f in sorted(INTERIM_RACE_OVERTAKES_DIR.glob("*.parquet")):
        season, round_num = (int(x) for x in f.stem.split("_"))
        if before is not None and (season, round_num) >= before:
            continue
        race_path = INTERIM_RACES_DIR / f.name
        if not race_path.exists():
            continue
        race = pd.read_parquet(race_path, columns=["driver_id", "grid_position", "finish_position", "dnf_flag"])
        rows.append(race.merge(pd.read_parquet(f)[["driver_id", "race_overtakes"]], on="driver_id"))

    if not rows:
        return 0.0

    df = pd.concat(rows).dropna()
    df = df[~df["dnf_flag"]]
    gained = (df["grid_position"] - df["finish_position"]).clip(lower=0)
    if gained.nunique() < 2:
        return 0.0
    slope, _ = np.polyfit(gained, df["race_overtakes"], 1)
    return max(float(slope), 0.0)


# the penalised drivers the finish shift applies to: predicted to qualify near the front, and dropped far
# enough that they have to pass the other fast cars to get back. a midfield car starting at the back lands
# about where its pace says; a front-runner loses ground, since the places at the front are the hardest to take
FRONT_RUNNER_QUALI = 4
FRONT_RUNNER_MIN_DROP = 8


def front_runner_mask(driver_ids, quali_position, penalties):
    return np.array([
        did in penalties and q <= FRONT_RUNNER_QUALI
        and (penalties[did] == BACK_OF_GRID or penalties[did] >= FRONT_RUNNER_MIN_DROP)
        for did, q in zip(driver_ids, quali_position)
    ])


# finish places a penalised front-runner loses beyond what qualifying in that slot normally costs, fitted on
# races strictly before `before`: each such finisher's finish against the mean finish of unpenalised drivers
# who qualified in the same slot, averaged. ~1.4 places on 2019-2026 data, from only ~14 cases
def front_runner_finish_shift(before=None):
    frames = []
    for f in sorted(INTERIM_RACES_DIR.glob("*.parquet")):
        season, round_num = (int(x) for x in f.stem.split("_"))
        quali_path = INTERIM_QUALI_DIR / f.name
        if (before is not None and (season, round_num) >= before) or not quali_path.exists():
            continue
        race = pd.read_parquet(f, columns=["driver_id", "grid_position", "finish_position", "dnf_flag"])
        frames.append(race.merge(pd.read_parquet(quali_path, columns=["driver_id", "quali_position"]), on="driver_id"))

    if not frames:
        return 0.0

    df = pd.concat(frames).dropna()
    df = df[~df["dnf_flag"]]
    drop = df["grid_position"] - df["quali_position"]
    ref = df[drop.abs() <= 1].groupby("quali_position")["finish_position"].mean()
    cases = df[(df["quali_position"] <= FRONT_RUNNER_QUALI) & (drop >= FRONT_RUNNER_MIN_DROP)]
    if cases.empty:
        return 0.0

    return max(float((cases["finish_position"] - cases["quali_position"].map(ref)).mean()), 0.0)
