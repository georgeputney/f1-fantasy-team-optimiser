"""Builds the value & buying power payload - the budget waterfall into next round, the expected
price move of the current squad, and a pool of assets (held + top alternatives) for the value-map
scatter plot. Mirrors app/dashboard.py's value section; the scatter's axis scaling and label
placement live in the frontend (web/src/valueMap.ts), same split as the ladder's distribution bars.
"""

from app.config import TEAM_STATE_FILE
from app.data.team_colors import TEAM_COLORS
from app.optimiser.state import load_state

from api.common import surname, fullname, load_predictions, model_default_budget, resolve_state, cached_optimiser


def build_value(budget=None, squad_mode="model", drivers=None, constructors=None, free_transfers=2):
    drivers = drivers or []
    constructors = constructors or []

    pred = load_predictions()
    season, rnd, circuit = pred["season"], pred["round"], pred["circuit"]
    driver_team = pred["driver_team"]
    driver_df, constructor_df, prices_df = pred["driver_df"], pred["constructor_df"], pred["prices_df"]
    prices_index, driver_pts, constructor_pts = pred["prices_index"], pred["driver_pts"], pred["constructor_pts"]
    price_delta, lam, have_delta = pred["price_delta"], pred["price_lambda"], pred["have_delta"]

    model_state = load_state(TEAM_STATE_FILE)
    default_budget = model_default_budget(model_state, prices_index)
    resolved_budget = budget if budget is not None else default_budget
    state = resolve_state(squad_mode, drivers, constructors, free_transfers, resolved_budget, prices_index, model_state, driver_team)

    team = cached_optimiser(season, rnd, driver_df, constructor_df, prices_df, resolved_budget, state, price_delta, lam)
    selected_ids = set(team["drivers"] + team["constructors"])

    def name_of(i, is_drv):
        return surname(i) if is_drv else fullname(i)

    def color_of(i, is_drv):
        cid = driver_team.get(i, "") if is_drv else i
        return TEAM_COLORS.get(cid, "#888888")

    spend = sum(float(prices_index[i]) for i in selected_ids)
    cash = resolved_budget - spend
    rises = sum(price_delta.get(i, 0.0) for i in selected_ids) if have_delta else 0.0
    budget_next = resolved_budget + rises

    held_drivers = sorted([d for d in team["drivers"]], key=lambda i: -float(prices_index[i]))
    held_constructors = sorted([c for c in team["constructors"]], key=lambda i: -float(prices_index[i]))

    price_moves = []
    for i in held_drivers + held_constructors:
        is_drv = i in driver_pts.index
        price = float(prices_index[i])
        pts = float(driver_pts[i]) if is_drv else float(constructor_pts[i])
        price_moves.append({
            "id": i, "name": name_of(i, is_drv), "color": color_of(i, is_drv), "is_driver": is_drv,
            "price": round(price, 1), "ppm": round(pts / price, 2) if price else 0.0,
            "move": round(price_delta[i], 1) if have_delta and i in price_delta else None,
        })

    # value map pool - held assets plus the highest-scoring drivers not currently held, so the
    # chart shows the recommended picks alongside the strongest alternatives to weigh them against
    extra = [d for d in driver_pts.sort_values(ascending=False).index if d not in selected_ids][:6]
    pool = list(selected_ids) + list(extra)

    value_map = []
    for i in pool:
        is_drv = i in driver_pts.index
        price = float(prices_index[i])
        pts = float(driver_pts[i]) if is_drv else float(constructor_pts[i])
        value_map.append({
            "id": i, "name": name_of(i, is_drv), "color": color_of(i, is_drv),
            "ppm": round(pts / price, 2) if price else 0.0,
            "move": round(price_delta[i], 2) if have_delta and i in price_delta else None,
            "points": round(pts, 1),
            "selected": i in selected_ids,
        })

    return {
        "season": season, "round": rnd, "circuit": circuit,
        "next_round": rnd + 1,
        "price_lambda": lam,
        "have_price_data": have_delta,
        "waterfall": {
            "team_value": round(spend, 1),
            "cash": round(cash, 1),
            "expected_rises": round(rises, 1),
            "budget_next": round(budget_next, 1),
        },
        "price_moves": price_moves,
        "value_map": value_map,
    }
