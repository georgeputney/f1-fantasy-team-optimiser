"""Shared helpers for the API layer - loading the latest prediction snapshot into the shapes every
section needs, and the cached Monte Carlo simulation both the ladder and team sections read from."""

import json

import pandas as pd

from app.config import PREDICTIONS_DIR, REPORTS_DIR, PRICE_LAMBDA, BUDGET_CAP
from app.models.monte_carlo import simulate_round
from app.optimiser.budget_range import compute_budget_range

try:
    from app.data.prices import expected_price_delta
except Exception:
    expected_price_delta = None

MC_CACHE_DIR = REPORTS_DIR / "predictions"


def surname(driver_id):
    return driver_id.split("_")[-1].title()


def fullname(constructor_id):
    return constructor_id.replace("_", " ").title()


def latest_predictions_path():
    available = sorted(PREDICTIONS_DIR.glob("predictions_????_??.json"))
    if not available:
        raise FileNotFoundError("No predictions available - run generate-reports first.")
    return available[-1]


# loads the latest prediction snapshot into dataframes/indexes every section builds from, plus the
# no-lookahead expected price move used to weight the optimiser toward assets likely to rise
def load_predictions():
    data = json.loads(latest_predictions_path().read_text())
    season, rnd, circuit = data["season"], data["round"], data["circuit"]

    driver_team = {d["driver_id"]: d.get("constructor_id", "") for d in data["drivers"]}
    driver_df = pd.DataFrame(data["drivers"]).rename(columns={"expected_points": "expected_fantasy_points"})
    constructor_df = pd.DataFrame(data["constructors"]).rename(columns={"expected_points": "expected_fantasy_points"})
    prices_df = pd.DataFrame(
        [{"asset_id": d["driver_id"], "price": d["price"]} for d in data["drivers"]]
        + [{"asset_id": c["constructor_id"], "price": c["price"]} for c in data["constructors"]]
    )
    prices_index = prices_df.set_index("asset_id")["price"]
    driver_pts = driver_df.set_index("driver_id")["expected_fantasy_points"]
    constructor_pts = constructor_df.set_index("constructor_id")["expected_fantasy_points"]

    price_delta, have_delta = {}, False
    if expected_price_delta is not None:
        try:
            predicted_points = pd.concat([driver_pts, constructor_pts])
            price_delta = expected_price_delta(season, rnd, prices_index, predicted_points)
            have_delta = True
        except Exception:
            price_delta, have_delta = {}, False
    lam = PRICE_LAMBDA if have_delta else 0.0

    return {
        "season": season, "round": rnd, "circuit": circuit,
        "trigger": data.get("trigger"), "generated_at": data["generated_at"],
        "driver_team": driver_team, "driver_df": driver_df, "constructor_df": constructor_df,
        "prices_df": prices_df, "prices_index": prices_index,
        "driver_pts": driver_pts, "constructor_pts": constructor_pts,
        "price_delta": price_delta, "price_lambda": lam, "have_delta": have_delta,
    }


# Monte Carlo sims take a few seconds; cache quantiles + the raw per-sim matrices (drivers AND
# constructors - constructor points aren't just their two drivers summed, they also carry a
# per-sim Q2/Q3 qualifying bonus and pitstop points) alongside the predictions snapshot so repeat
# requests for the same round are instant. Old-schema cache files are rebuilt once rather than
# erroring on a missing key.
def load_or_build_mc(season, round_num, circuit):
    cache_path = MC_CACHE_DIR / f"mc_{season}_{round_num:02d}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if "raw_constructor_totals" in cached:
            return cached

    result = simulate_round(season, round_num, circuit)
    cached = {
        "drivers": {aid: {k: round(v, 3) for k, v in q.items()} for aid, q in result["drivers"].items()},
        "constructors": {aid: {k: round(v, 3) for k, v in q.items()} for aid, q in result["constructors"].items()},
        "raw_totals": result["raw_totals"],
        "raw_driver_ids": result["raw_driver_ids"],
        "raw_constructor_totals": result["raw_constructor_totals"],
        "raw_constructor_ids": result["raw_constructor_ids"],
    }
    MC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cached))
    return cached


# the retrospective solve takes ~2s; cache it the same way as the MC sims
def load_or_build_budget_range(season, round_num):
    cache_path = MC_CACHE_DIR / f"budget_range_{season}_{round_num:02d}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return cached["min"], cached["max"]

    lo, hi = compute_budget_range(season, round_num)
    MC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"min": lo, "max": hi}))
    return lo, hi


def model_default_budget(model_state, prices_index):
    if not model_state:
        return BUDGET_CAP
    held = model_state["drivers"] + model_state["constructors"]
    value = sum(float(prices_index.get(i, model_state["prices"][i])) for i in held)
    return round(model_state["budget_remaining"] + value, 1)


# resolves the team state the optimiser transfers against. "model" mode always uses the committed
# state file. "custom" mode builds a state from whatever squad the user has picked so far in the UI -
# empty selections mean a from-scratch build (state=None, no transfer penalty). Shared by the ladder
# and team sections so a squad edit in the controls is reflected in both.
def resolve_state(squad_mode, drivers, constructors, free_transfers, budget, prices_index, model_state):
    if squad_mode != "custom":
        return model_state

    if not drivers and not constructors:
        return None

    held = drivers + constructors
    team_cost = sum(float(prices_index.get(i, 0)) for i in held)
    return {
        "drivers": drivers,
        "constructors": constructors,
        "prices": {i: float(prices_index.get(i, 0)) for i in held},
        "budget_remaining": budget - team_cost,
        "free_transfers_carried": free_transfers - 2,
    }
