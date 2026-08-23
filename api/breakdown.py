"""Builds the driver-breakdown payload - where each driver's expected-points total comes from,
browsable by round. Desktop only per the design brief (mobile drops this section)."""

import json

from app.config import PREDICTIONS_DIR, TEAM_STATE_FILE
from app.data.team_colors import TEAM_COLORS
from app.optimiser.state import load_state

from api.common import surname, load_predictions, load_or_build_mc, model_default_budget, resolve_state, cached_optimiser


def _available_rounds():
    rounds = {}
    for f in sorted(PREDICTIONS_DIR.glob("predictions_????_??.json")):
        season_str, round_str = f.stem.split("_")[1:3]
        rounds.setdefault(int(season_str), []).append(int(round_str))
    for season in rounds:
        rounds[season].sort()
    return rounds


# accepts the same squad-controls params as build_team()/build_ladder() so the highlighted row set
# matches whatever the user is currently exploring above, not just the model's committed squad
def build_breakdown(
    season=None, round_num=None,
    budget=None, squad_mode="model", drivers=None, constructors=None, free_transfers=2,
):
    drivers = drivers or []
    constructors = constructors or []

    available = _available_rounds()
    latest_season = max(available)
    latest_round = available[latest_season][-1]
    is_latest = season is None and round_num is None

    season = season or latest_season
    round_num = round_num or (available[season][-1] if season in available else latest_round)

    path = PREDICTIONS_DIR / f"predictions_{season}_{round_num:02d}.json"
    if not path.exists():
        season, round_num = latest_season, latest_round
        is_latest = True
        path = PREDICTIONS_DIR / f"predictions_{season}_{round_num:02d}.json"

    data = json.loads(path.read_text())
    circuit = data["circuit"]

    # highlighting matches the ladder's "selected" - the optimiser's current RECOMMENDATION under
    # whatever budget/squad the user is exploring in the controls above, not the raw committed
    # squad. only meaningful for the live round - there's no retroactive recommendation for a past
    # one, so historical rounds always show unhighlighted regardless of the squad-controls state
    selected_ids = set()
    if is_latest or (season == latest_season and round_num == latest_round):
        pred = load_predictions()
        model_state = load_state(TEAM_STATE_FILE)
        resolved_budget = budget if budget is not None else model_default_budget(model_state, pred["prices_index"])
        state = resolve_state(squad_mode, drivers, constructors, free_transfers, resolved_budget, pred["prices_index"], model_state, pred["driver_team"])
        team = cached_optimiser(
            pred["season"], pred["round"], pred["driver_df"], pred["constructor_df"], pred["prices_df"], resolved_budget, state,
            pred["price_delta"], pred["price_lambda"],
        )
        selected_ids = set(team["drivers"] + team["constructors"])

    mc = load_or_build_mc(season, round_num, circuit)
    dnf_prob = {aid: q.get("dnf_prob", 0.0) for aid, q in mc["drivers"].items()}

    rows = []
    for d in data["drivers"]:
        bd = d.get("points_breakdown", {})
        quali_pos = d["predicted_quali_position"]
        finish_pos = d["predicted_finish_position"]
        rows.append({
            "id": d["driver_id"],
            "name": surname(d["driver_id"]),
            "color": TEAM_COLORS.get(d.get("constructor_id", ""), "#888888"),
            "selected": d["driver_id"] in selected_ids,
            "quali_position": quali_pos,
            "finish_position": finish_pos,
            "positions_gained": quali_pos - finish_pos,
            "overtakes": round(bd.get("overtakes", 0.0), 1),
            "prob_fl": round(bd.get("prob_fl", 0.0), 3),
            "prob_dotd": round(bd.get("prob_dotd", 0.0), 3),
            "dnf_prob": round(dnf_prob.get(d["driver_id"], 0.0), 3),
            "expected_points": round(d["expected_points"], 1),
        })
    rows.sort(key=lambda r: -r["expected_points"])

    return {
        "season": season,
        "round": round_num,
        "circuit": circuit,
        "available_rounds": available.get(season, [round_num]),
        "rows": rows,
    }
