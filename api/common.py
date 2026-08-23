"""Shared helpers for the API layer - loading the latest prediction snapshot into the shapes every
section needs, and the cached Monte Carlo simulation both the ladder and team sections read from."""

import json
import threading
import time

import pandas as pd

from app.config import PREDICTIONS_DIR, REPORTS_DIR, PRICE_LAMBDA, BUDGET_CAP
from app.models.monte_carlo import simulate_round
from app.optimiser.budget_range import compute_budget_range
from app.optimiser.optimiser import optimiser, enumerate_teams

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


# state files saved before driver_teams existed have no snapshot at all - rather than showing every
# held driver from one of those with no team/colour forever, this reconstructs it from that state's
# own season/round predictions file (still on disk for any recent round), the same source the
# snapshot would have been taken from had it existed at save time
def historical_driver_teams(season, round_num):
    path = PREDICTIONS_DIR / f"predictions_{season}_{round_num:02d}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {d["driver_id"]: d.get("constructor_id", "") for d in data["drivers"]}


# every one of the 5 endpoints a page load fires calls this - re-parsing the predictions JSON and
# recomputing the price delta (which itself reads every target parquet up to this round) 5 times
# over on every load costs more than the optimiser solve does, so it gets the same short-lived cache
_PREDICTIONS_CACHE = {}
_PREDICTIONS_CACHE_LOCK = threading.Lock()
_PREDICTIONS_CACHE_TTL = 30  # seconds - just needs to outlive one page load's burst of requests


# loads the latest prediction snapshot into dataframes/indexes every section builds from, plus the
# no-lookahead expected price move used to weight the optimiser toward assets likely to rise
def load_predictions():
    path = latest_predictions_path()
    key = str(path)
    now = time.monotonic()
    with _PREDICTIONS_CACHE_LOCK:
        cached = _PREDICTIONS_CACHE.get(key)
        if cached and now - cached[0] < _PREDICTIONS_CACHE_TTL:
            return cached[1]

    result = _load_predictions(path)
    with _PREDICTIONS_CACHE_LOCK:
        _PREDICTIONS_CACHE[key] = (now, result)
    return result


def _load_predictions(path):
    data = json.loads(path.read_text())
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
# erroring on a missing key. team/ladder/breakdown all call this on every page load - the file
# itself can be a couple MB, so an in-memory cache on top of the on-disk one avoids re-parsing that
# JSON three times over for what's otherwise an identical result within the same short window
_MC_MEMORY_CACHE = {}
_MC_MEMORY_CACHE_LOCK = threading.Lock()


def load_or_build_mc(season, round_num, circuit):
    mem_key = (season, round_num)
    now = time.monotonic()
    with _MC_MEMORY_CACHE_LOCK:
        cached = _MC_MEMORY_CACHE.get(mem_key)
        if cached and now - cached[0] < _PREDICTIONS_CACHE_TTL:
            return cached[1]

    cache_path = MC_CACHE_DIR / f"mc_{season}_{round_num:02d}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if "raw_constructor_totals" in cached:
            with _MC_MEMORY_CACHE_LOCK:
                _MC_MEMORY_CACHE[mem_key] = (now, cached)
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
    with _MC_MEMORY_CACHE_LOCK:
        _MC_MEMORY_CACHE[mem_key] = (now, cached)
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
def resolve_state(squad_mode, drivers, constructors, free_transfers, budget, prices_index, model_state, driver_team=None):
    if squad_mode != "custom":
        return model_state

    if not drivers and not constructors:
        return None

    held = drivers + constructors
    team_cost = sum(float(prices_index.get(i, 0)) for i in held)
    # a driver in a custom-mode slot is either currently active (live driver_team has them) or an
    # inactive holdover the UI kept selected in its own slot - for that case there's no live team to
    # look up, so it falls back to whatever was last snapshotted in the committed state file
    driver_team = driver_team or {}
    prev_driver_teams = model_state.get("driver_teams", {}) if model_state else {}
    driver_teams = {d: driver_team.get(d, prev_driver_teams.get(d, "")) for d in drivers}
    return {
        "drivers": drivers,
        "constructors": constructors,
        "prices": {i: float(prices_index.get(i, 0)) for i in held},
        "budget_remaining": budget - team_cost,
        "free_transfers_carried": free_transfers - 2,
        "driver_teams": driver_teams,
    }


# team/ladder/breakdown/value all solve the identical ILP for the same squad state on every page
# load (the frontend fires all 5 requests in parallel with the same params) - a short-lived cache
# means only the first of those does the actual solve and the rest reuse it, instead of paying for
# the same ~150ms-plus (much more under Render's shared/throttled CPU) solve four times over
_OPTIMISER_CACHE = {}
_OPTIMISER_CACHE_LOCK = threading.Lock()
_OPTIMISER_CACHE_TTL = 30  # seconds - just needs to outlive one page load's burst of requests


def cached_optimiser(season, round_num, driver_df, constructor_df, prices_df, budget, state, price_delta, price_lambda):
    state_key = (
        tuple(sorted(state["drivers"])), tuple(sorted(state["constructors"])), state["free_transfers_carried"],
    ) if state else None
    key = (season, round_num, round(budget, 2), state_key, price_lambda)

    now = time.monotonic()
    with _OPTIMISER_CACHE_LOCK:
        cached = _OPTIMISER_CACHE.get(key)
        if cached and now - cached[0] < _OPTIMISER_CACHE_TTL:
            return cached[1]

    result = optimiser(driver_df, constructor_df, prices_df, budget, state, price_delta=price_delta, price_lambda=price_lambda)
    with _OPTIMISER_CACHE_LOCK:
        _OPTIMISER_CACHE[key] = (now, result)
    return result


# the ladder's 5 alternative-team solves take ~600ms and are otherwise uncached - a plain reload
# within the TTL window (unchanged squad state) would pay for them again for no reason
_ENUMERATE_CACHE = {}
_ENUMERATE_CACHE_LOCK = threading.Lock()


def cached_enumerate_teams(season, round_num, driver_df, constructor_df, prices_df, budget, state, price_delta, price_lambda, n):
    state_key = (
        tuple(sorted(state["drivers"])), tuple(sorted(state["constructors"])), state["free_transfers_carried"],
    ) if state else None
    key = (season, round_num, round(budget, 2), state_key, price_lambda, n)

    now = time.monotonic()
    with _ENUMERATE_CACHE_LOCK:
        cached = _ENUMERATE_CACHE.get(key)
        if cached and now - cached[0] < _OPTIMISER_CACHE_TTL:
            return cached[1]

    result = enumerate_teams(driver_df, constructor_df, prices_df, budget, state, price_delta=price_delta, price_lambda=price_lambda, n=n)
    with _ENUMERATE_CACHE_LOCK:
        _ENUMERATE_CACHE[key] = (now, result)
    return result
