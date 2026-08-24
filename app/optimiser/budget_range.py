"""Computes the theoretical max/min budget achievable by a given round - the budget a manager would
have if they'd made the single best (or worst) price-appreciating legal transfer every round since
round 1. Pure price optimisation: transfer-count limits don't apply here, since in the real game extra
transfers cost POINTS, not budget, so they don't constrain what budget is reachable - only roster
legality (5 drivers + 2 constructors) and affordability at each round's prices do.
"""

import json

import pandas as pd
import pulp

from app.config import PROCESSED_PRICES_DIR, BUDGET_CAP, DRIVER_ROSTER_SIZE, CONSTRUCTOR_ROSTER_SIZE


def _load_round_prices(season, round_num):
    path = PROCESSED_PRICES_DIR / f"{season}_{round_num:02d}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path).set_index("asset_id")[["asset_type", "price"]]


# selects a legal 5+2 team affordable at this round's prices, maximising (or minimising) price
# movement into next round's prices. returns the new budget after moving into that team
def _solve_transition(prices_now, prices_next, budget, maximize):
    assets = [a for a in prices_now.index if a in prices_next.index]
    drivers = [a for a in assets if prices_now.loc[a, "asset_type"] == "driver"]
    constructors = [a for a in assets if prices_now.loc[a, "asset_type"] == "constructor"]
    delta = {a: float(prices_next.loc[a, "price"]) - float(prices_now.loc[a, "price"]) for a in assets}

    prob = pulp.LpProblem("budget_range", pulp.LpMaximize if maximize else pulp.LpMinimize)
    selected = pulp.LpVariable.dicts("sel", assets, cat="Binary")
    prob += pulp.lpSum(delta[a] * selected[a] for a in assets)
    prob += pulp.lpSum(selected[d] for d in drivers) == DRIVER_ROSTER_SIZE
    prob += pulp.lpSum(selected[c] for c in constructors) == CONSTRUCTOR_ROSTER_SIZE
    prob += pulp.lpSum(float(prices_now.loc[a, "price"]) * selected[a] for a in assets) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    chosen = [a for a in assets if pulp.value(selected[a]) == 1]
    return budget + sum(delta[a] for a in chosen)


# simulates round 1 through round_num - 1 twice (best-case and worst-case), returning the
# (min_budget, max_budget) any legal manager could plausibly be sitting on entering round_num
def compute_budget_range(season, round_num):
    min_budget = max_budget = BUDGET_CAP

    for r in range(1, round_num):
        prices_now = _load_round_prices(season, r)
        prices_next = _load_round_prices(season, r + 1)
        if prices_now is None or prices_next is None:
            break
        max_budget = _solve_transition(prices_now, prices_next, max_budget, maximize=True)
        min_budget = _solve_transition(prices_now, prices_next, min_budget, maximize=False)

    return round(min_budget, 1), round(max_budget, 1)


# computes and writes the (min, max) budget range cache to disk - shared by the CLI (which should
# pre-build this every round the pipeline runs) and the API's lazy on-demand build, so a round's cache
# always exists before the live site's first request rather than paying for a ~2s solve mid-request
def cache_budget_range(season, round_num, path):
    lo, hi = compute_budget_range(season, round_num)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"min": lo, "max": hi}))
    return lo, hi
