"""Backtest utilities - baseline strategies and actual points scoring for evaluating the optimiser against historical race results."""

import pandas as pd

from app.optimiser import optimiser
from app.config import PROCESSED_TARGETS_DIR


# looks up actual fantasy points scored by a selected team from historical targets, applying x2 to the doubled driver
def get_actual_team_points(team, season, round_num):
    targets = pd.read_parquet(PROCESSED_TARGETS_DIR / f"{season}_{round_num:02d}.parquet").set_index("asset_id")["actual_fantasy_points"]
    points = 0

    for driver in team["drivers"]:
        multiplier = 2 if driver == team["doubled_driver"] else 1
        points += targets[driver] * multiplier

    for constructor in team["constructors"]:
        points += targets[constructor]

    return points
    

# selects the optimal team using actual race points as the objective - the theoretical ceiling for any strategy
def oracle_baseline(season, round_num, prices, budget):
    targets = pd.read_parquet(PROCESSED_TARGETS_DIR / f"{season}_{round_num:02d}.parquet")
    
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
    

    return optimiser(driver_points, constructor_points, prices, budget)


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
