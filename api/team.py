"""Builds the squad-controls / projected-points hero / lineup / suggested-transfers payload.

Mirrors the logic in app/dashboard.py (the Streamlit predecessor) - same optimiser call, same
transfer-row derivation - but adds the team-level "likely range" from the Monte Carlo raw matrix,
which the Streamlit version never had (it showed a placeholder).
"""

from datetime import datetime

from app.config import TEAM_STATE_FILE
from app.data.team_colors import TEAM_COLORS
from app.optimiser.optimiser import optimiser
from app.optimiser.state import load_state
from app.models.monte_carlo import team_distribution

from api.common import (
    surname, fullname, load_predictions, load_or_build_mc, load_or_build_budget_range,
    model_default_budget, resolve_state,
)

# matches app/dashboard.py's _trigger_labels - which point in the race weekend this snapshot's
# predictions were generated at, read straight from the predictions file rather than hardcoded.
# post-sprint-quali covers sprint weekends, which skip FP2/FP3 entirely
TRIGGER_LABELS = {"post-fp2": "Post-FP2", "pre-race": "Post-FP3", "post-sprint-quali": "Post-Sprint Quali"}


# derives suggested transfers (dropped -> added, paired by asset type) from the diff between the
# prior state and the optimiser's picks - same pairing logic as dashboard.py's transfer_rows()
def _transfer_rows(state, team, driver_pts, constructor_pts, selected_ids):
    if not state:
        return [], 0.0

    prev = set(state["drivers"] + state["constructors"])
    added = [i for i in team["drivers"] + team["constructors"] if i not in prev]
    dropped = [i for i in prev if i not in selected_ids]

    def pts_of(i):
        if i in driver_pts.index:
            return float(driver_pts[i])
        if i in constructor_pts.index:
            return float(constructor_pts[i])
        return 0.0

    def is_driver(i):
        return i in driver_pts.index or i in state.get("drivers", [])

    add_d = [i for i in added if is_driver(i)]
    add_c = [i for i in added if not is_driver(i)]
    drop_d = [i for i in dropped if is_driver(i)]
    drop_c = [i for i in dropped if not is_driver(i)]

    rows = []
    for outs, ins, is_drv in ((drop_d, add_d, True), (drop_c, add_c, False)):
        for out_i, in_i in zip(outs, ins):
            label = surname if is_drv else fullname
            delta = pts_of(in_i) - pts_of(out_i)
            rows.append({
                "out_id": out_i, "out_name": label(out_i),
                "in_id": in_i, "in_name": label(in_i),
                "delta": round(delta, 1),
            })
    net = sum(r["delta"] for r in rows) - 10 * team["transfer_penalty"]
    return rows, round(net, 1)


def build_team(budget=None, squad_mode="model", drivers=None, constructors=None, free_transfers=2):
    drivers = drivers or []
    constructors = constructors or []

    pred = load_predictions()
    season, rnd, circuit = pred["season"], pred["round"], pred["circuit"]
    driver_team = pred["driver_team"]
    driver_df, constructor_df, prices_df = pred["driver_df"], pred["constructor_df"], pred["prices_df"]
    prices_index, driver_pts, constructor_pts = pred["prices_index"], pred["driver_pts"], pred["constructor_pts"]
    price_delta, lam = pred["price_delta"], pred["price_lambda"]
    trigger_label = TRIGGER_LABELS.get(pred["trigger"], "Latest")
    generated_at = datetime.fromisoformat(pred["generated_at"])
    status = f"{trigger_label}, {generated_at:%H:%M}"

    model_state = load_state(TEAM_STATE_FILE)
    default_budget = model_default_budget(model_state, prices_index)
    resolved_budget = budget if budget is not None else default_budget

    # the slider's bounds are the actual best/worst-case budget any manager could be sitting on
    # entering this round (a retrospective solve over the season's real price history), not an
    # arbitrary fixed range - widened slightly if needed so the current budget is never out of bounds
    budget_min, budget_max = load_or_build_budget_range(season, rnd)
    budget_min = min(budget_min, resolved_budget)
    budget_max = max(budget_max, resolved_budget)

    state = resolve_state(squad_mode, drivers, constructors, free_transfers, resolved_budget, prices_index, model_state)

    team = optimiser(driver_df, constructor_df, prices_df, resolved_budget, state, price_delta=price_delta, price_lambda=lam)
    selected_ids = set(team["drivers"] + team["constructors"])
    captain = team["doubled_driver"]

    proj_points = sum(float(driver_pts[d]) * (2 if d == captain else 1) for d in team["drivers"])
    proj_points += sum(float(constructor_pts[c]) for c in team["constructors"])
    spend = sum(float(prices_index[i]) for i in team["drivers"] + team["constructors"])
    cash = resolved_budget - spend

    mc = load_or_build_mc(season, rnd, circuit)
    likely_range = team_distribution(mc, team["drivers"], captain, set(team["constructors"]))

    transfer_rows, net = _transfer_rows(state, team, driver_pts, constructor_pts, selected_ids)

    added_set = set()
    if state:
        prev = set(state["drivers"] + state["constructors"])
        added_set = selected_ids - prev

    ordered = [captain] + sorted([d for d in team["drivers"] if d != captain], key=lambda d: -driver_pts[d])
    ordered += sorted(team["constructors"], key=lambda c: -constructor_pts[c])

    lineup = []
    for aid in ordered:
        is_drv = aid in driver_pts.index
        price = float(prices_index[aid])
        cid = driver_team.get(aid, "") if is_drv else aid
        if aid == captain:
            pts = float(driver_pts[aid])
            lineup.append({
                "id": aid, "name": surname(aid), "is_driver": True, "captain": True, "in": aid in added_set,
                "constructor_id": cid, "color": TEAM_COLORS.get(cid, "#888888"),
                "points": round(pts, 1), "doubled_points": round(pts * 2, 1), "price": price,
            })
        else:
            pts = float(driver_pts[aid]) if is_drv else float(constructor_pts[aid])
            lineup.append({
                "id": aid, "name": surname(aid) if is_drv else fullname(aid), "is_driver": is_drv,
                "captain": False, "in": aid in added_set,
                "constructor_id": cid, "color": TEAM_COLORS.get(cid, "#888888"),
                "points": round(pts, 1), "doubled_points": None, "price": price,
            })

    driver_options = [
        {"id": d, "name": surname(d), "price": float(prices_index[d])}
        for d in driver_df.sort_values("price", ascending=False)["driver_id"]
    ]
    constructor_options = [
        {"id": c, "name": fullname(c), "price": float(prices_index[c])}
        for c in constructor_df.sort_values("price", ascending=False)["constructor_id"]
    ]

    return {
        "season": season, "round": rnd, "circuit": circuit, "status": status,
        "controls": {
            # 0.1 not 0.5 - budget_min/max come from a retrospective solve and won't generally sit on
            # a 0.5 grid (e.g. min=57.5 but your actual budget might be 114.8), so a 0.5 step could
            # never land back on your real starting value once you'd moved the slider
            "budget": resolved_budget, "budget_min": budget_min, "budget_max": budget_max, "budget_step": 0.1,
            "default_budget": default_budget,
            "squad_mode": squad_mode,
            "free_transfers": free_transfers,
            "current_drivers": state["drivers"] if state else [],
            "current_constructors": state["constructors"] if state else [],
            "remaining_budget": round(cash, 1),
            "team_value": round(spend, 1),
        },
        "driver_options": driver_options,
        "constructor_options": constructor_options,
        "hero": {
            "projected_points": round(proj_points),
            "likely_range": likely_range,
            "spend": round(spend, 1),
            "net_after_hit": net if state else None,
            "transfers_made": team["transfers_made"] if state else 0,
            "captain_id": captain,
        },
        "lineup": lineup,
        "transfers": {
            "rows": transfer_rows,
            "net": net,
            "has_state": state is not None,
            "free": (2 + state["free_transfers_carried"]) if state else 0,
            "paid": team["transfer_penalty"] if state else 0,
        },
    }
