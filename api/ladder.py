"""Builds the ladder section payload - every driver and constructor ranked by expected points,
with the Monte Carlo display distribution, price, current-squad selection, and alternative teams."""

from app.config import TEAM_STATE_FILE
from app.data.driver_codes import fia_code
from app.data.constructor_codes import constructor_code
from app.data.team_colors import TEAM_COLORS
from app.optimiser.state import load_state

from api.common import surname, fullname, load_predictions, load_or_build_mc, model_default_budget, resolve_state, cached_optimiser, cached_enumerate_teams

N_ALTERNATIVES = 5


# accepts the same squad-controls params as build_team() so the ladder's selection highlighting and
# alternative teams stay in sync with whatever the user is exploring in the controls above, rather
# than always reflecting the model's own committed squad
def build_ladder(budget=None, squad_mode="model", drivers=None, constructors=None, free_transfers=2):
    drivers = drivers or []
    constructors = constructors or []

    pred = load_predictions()
    season, rnd, circuit = pred["season"], pred["round"], pred["circuit"]
    driver_team = pred["driver_team"]
    driver_df, constructor_df, prices_df = pred["driver_df"], pred["constructor_df"], pred["prices_df"]
    prices_index, driver_pts, constructor_pts = pred["prices_index"], pred["driver_pts"], pred["constructor_pts"]
    price_delta, lam = pred["price_delta"], pred["price_lambda"]

    model_state = load_state(TEAM_STATE_FILE)
    resolved_budget = budget if budget is not None else model_default_budget(model_state, prices_index)
    state = resolve_state(squad_mode, drivers, constructors, free_transfers, resolved_budget, prices_index, model_state)

    team = cached_optimiser(season, rnd, driver_df, constructor_df, prices_df, resolved_budget, state, price_delta, lam)
    selected_ids = set(team["drivers"] + team["constructors"])
    captain = team["doubled_driver"]

    # enumerate_teams solves in order of the TRUE objective (points + price_lambda * price_delta -
    # transfer penalty), so raw points alone can be non-monotonic across solves. The ladder displays
    # and labels rows by raw points ("ranked next, by expected points"), so re-sort for display -
    # the recommended team (index 0) is pinned first since it must match the highlighted selection
    # above, even in the rare case something later has marginally higher raw points
    alt_teams = cached_enumerate_teams(
        season, rnd, driver_df, constructor_df, prices_df, resolved_budget, state,
        price_delta, lam, N_ALTERNATIVES,
    )
    if alt_teams:
        alt_teams = [alt_teams[0]] + sorted(alt_teams[1:], key=lambda t: -t["total_points"])

    mc = load_or_build_mc(season, rnd, circuit)
    mc_drivers = mc["drivers"]
    mc_constructors = mc["constructors"]

    driver_rows = []
    for did in driver_pts.sort_values(ascending=False).index:
        driver_rows.append({
            "id": did,
            "name": surname(did),
            "fia_code": fia_code(did),
            "constructor_id": driver_team.get(did, ""),
            "color": TEAM_COLORS.get(driver_team.get(did, ""), "#888888"),
            "points": round(float(driver_pts[did]), 1),
            "price": float(prices_index[did]),
            "selected": did in selected_ids,
            "captain": did == captain,
            "distribution": mc_drivers.get(did),
        })

    constructor_rows = []
    for cid in constructor_pts.sort_values(ascending=False).index:
        constructor_rows.append({
            "id": cid,
            "name": fullname(cid),
            "fia_code": constructor_code(cid),
            "constructor_id": cid,
            "color": TEAM_COLORS.get(cid, "#888888"),
            "points": round(float(constructor_pts[cid]), 1),
            "price": float(prices_index[cid]),
            "selected": cid in selected_ids,
            "captain": False,
            "distribution": mc_constructors.get(cid),
        })

    best_points = alt_teams[0]["total_points"] if alt_teams else 0.0
    best_set = set(alt_teams[0]["drivers"] + alt_teams[0]["constructors"]) if alt_teams else set()
    alt_rows = []
    for rank, t in enumerate(alt_teams, start=1):
        spend = sum(float(prices_index[i]) for i in t["drivers"] + t["constructors"])
        alt_rows.append({
            "rank": rank,
            "total_points": round(t["total_points"], 1),
            "gap_to_best": round(t["total_points"] - best_points, 1),
            "spend": round(spend, 1),
            "captain_driver_id": t["doubled_driver"],
            "drivers": [
                {
                    "id": d, "fia_code": fia_code(d),
                    "color": TEAM_COLORS.get(driver_team.get(d, ""), "#888888"),
                    "differs": d not in best_set, "captain": d == t["doubled_driver"],
                }
                for d in sorted(t["drivers"], key=lambda d: -driver_pts[d])
            ],
            "constructors": [
                {
                    "id": c, "code": constructor_code(c),
                    "color": TEAM_COLORS.get(c, "#888888"),
                    "differs": c not in best_set,
                }
                for c in sorted(t["constructors"], key=lambda c: -constructor_pts[c])
            ],
        })

    return {
        "season": season,
        "round": rnd,
        "circuit": circuit,
        "drivers": driver_rows,
        "constructors": constructor_rows,
        "alternative_teams": alt_rows,
    }
