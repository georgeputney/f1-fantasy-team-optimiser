"""Backtest utilities - baseline strategies and actual points scoring for evaluating the optimiser against historical race results."""

import pandas as pd

from app.optimiser.optimiser import optimiser
from app.data.targets import load_fantasy_targets


# looks up actual fantasy points scored by a selected team from historical targets, applying x2 to the doubled driver
def get_actual_team_points(team, season, round_num, transfer_penalty=0):
    targets = load_fantasy_targets(season, round_num).set_index("asset_id")["actual_fantasy_points"]
    points = 0

    for driver in team["drivers"]:
        multiplier = 2 if driver == team["doubled_driver"] else 1
        points += targets[driver] * multiplier

    for constructor in team["constructors"]:
        points += targets[constructor]

    return points - transfer_penalty


# selects the optimal team using actual race points as the objective - the theoretical ceiling for any strategy
def oracle_baseline(season, round_num, prices, budget, state=None):
    targets = load_fantasy_targets(season, round_num)

    drivers = targets[targets["asset_type"] == "driver"][["asset_id", "actual_fantasy_points"]].dropna(subset=["asset_id"])
    driver_points = drivers.rename(columns={
        "asset_id": "driver_id",
        "actual_fantasy_points": "expected_fantasy_points"  # use actual points as the objective so the ILP picks the best possible team in hindsight
    })

    constructors = targets[targets["asset_type"] == "constructor"][["asset_id", "actual_fantasy_points"]].dropna(subset=["asset_id"])
    constructor_points = constructors.rename(columns={
        "asset_id": "constructor_id",
        "actual_fantasy_points": "expected_fantasy_points" # use actual points as the objective so the ILP picks the best possible team in hindsight
    })


    return optimiser(driver_points, constructor_points, prices, budget, state)


# selects the oracle-optimal team from the previous round to be played unchanged in the current round
# represents a "momentum" strategy: assume last week's best performers will repeat
# returns None for round 1 of a season (no prior round available)
def lagged_baseline(season, round_num, prices, budget):
    from app.config import FANTASY_POINTS_DIR

    prev_path = FANTASY_POINTS_DIR / f"{season}_{(round_num - 1):02d}.csv"
    if not prev_path.exists():
        return None

    targets = load_fantasy_targets(season, round_num - 1)

    # only keep assets available in this round's prices - handles team/driver changes between seasons
    available_assets = set(prices["asset_id"])

    drivers = targets[(targets["asset_type"] == "driver") & (targets["asset_id"].isin(available_assets))][["asset_id", "actual_fantasy_points"]].dropna(subset=["asset_id"])
    driver_points = drivers.rename(columns={
        "asset_id": "driver_id",
        "actual_fantasy_points": "expected_fantasy_points"
    })

    constructors = targets[(targets["asset_type"] == "constructor") & (targets["asset_id"].isin(available_assets))][["asset_id", "actual_fantasy_points"]].dropna(subset=["asset_id"])
    constructor_points = constructors.rename(columns={
        "asset_id": "constructor_id",
        "actual_fantasy_points": "expected_fantasy_points"
    })

    return optimiser(driver_points, constructor_points, prices, budget, state=None)


# uses expanding-window mean of each asset's actual fantasy points from prior rounds as the optimiser objective.
# represents a "pick whoever's been scoring well" strategy - no model, just recency.
# for round 1, falls back to previous season's averages.
def mean_prior_baseline(season, round_num, prices, budget, state=None):
    from app.config import FANTASY_POINTS_DIR

    # collect all prior rounds in the season
    prior_frames = []
    for r in range(1, round_num):
        path = FANTASY_POINTS_DIR / f"{season}_{r:02d}.csv"
        if path.exists():
            prior_frames.append(load_fantasy_targets(season, r))

    # fall back to previous season if no prior rounds yet
    if not prior_frames:
        prev_files = sorted(FANTASY_POINTS_DIR.glob(f"{season - 1}_*.csv"))
        for f in prev_files:
            _, rnd = f.stem.split("_")
            prior_frames.append(load_fantasy_targets(season - 1, int(rnd)))

    if not prior_frames:
        return None

    prior = pd.concat(prior_frames)
    means = prior.groupby(["asset_id", "asset_type"])["actual_fantasy_points"].mean().reset_index()

    available_assets = set(prices["asset_id"])

    drivers = means[(means["asset_type"] == "driver") & (means["asset_id"].isin(available_assets))]
    driver_points = drivers[["asset_id", "actual_fantasy_points"]].rename(columns={
        "asset_id": "driver_id",
        "actual_fantasy_points": "expected_fantasy_points",
    })

    constructors = means[(means["asset_type"] == "constructor") & (means["asset_id"].isin(available_assets))]
    constructor_points = constructors[["asset_id", "actual_fantasy_points"]].rename(columns={
        "asset_id": "constructor_id",
        "actual_fantasy_points": "expected_fantasy_points",
    })

    return optimiser(driver_points, constructor_points, prices, budget, state)


# estimates the expected fantasy points for a random valid team by averaging over N random selections under budget constraints
def random_baseline(season, round_num, prices, budget, n=1000):
    drivers = prices[prices["asset_type"] == "driver"]
    constructors = prices[prices["asset_type"] == "constructor"]

    total = 0
    valid = 0

    for _ in range(n):
        sampled_drivers = drivers.sample(5)
        sampled_constructors = constructors.sample(2)

        cost = sampled_drivers["price"].sum() + sampled_constructors["price"].sum()

        if cost > budget:
            continue

        doubled = sampled_drivers.sample(1)["asset_id"].iloc[0]

        team = {
            "drivers": sampled_drivers["asset_id"].tolist(),
            "constructors": sampled_constructors["asset_id"].tolist(),
            "doubled_driver": doubled,
        }

        total += get_actual_team_points(team, season, round_num)
        valid += 1

    return total / valid if valid > 0 else 0
