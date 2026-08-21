"""Builds the track record payload - model vs oracle performance across the season's backtested
rounds. Pure season-level stats, independent of the current squad selection, so unlike the other
sections this takes no budget/squad params. Mirrors app/dashboard.py's track-record section
(search for "track record" there).
"""

import json

from app.config import REPORTS_DIR

from api.common import load_predictions


def build_track_record():
    pred = load_predictions()
    season = pred["season"]

    bt_rows = []
    for f in sorted(REPORTS_DIR.glob("backtest_*.json")):
        season_num = int(f.stem.split("_")[1])
        if season_num >= 2026:
            bt_rows.extend(json.loads(f.read_text()))

    season_bt = [r for r in bt_rows if r.get("season") == season and r.get("oracle")]
    if not season_bt:
        return {"available": False}

    model_total = sum(r["model"] for r in season_bt)
    oracle_total = sum(r["oracle"] for r in season_bt)
    pcts = [(r["round"], r["model"] / r["oracle"] * 100) for r in season_bt]
    avg = sum(p for _, p in pcts) / len(pcts)
    best_round, best_pct = max(pcts, key=lambda x: x[1])
    worst_round, worst_pct = min(pcts, key=lambda x: x[1])
    last_round = max(r["round"] for r in season_bt)

    return {
        "available": True,
        "season": season,
        "last_round": last_round,
        "model_total": round(model_total),
        "oracle_total": round(oracle_total),
        "average_pct": avg,
        "best_round": best_round,
        "best_pct": best_pct,
        "worst_round": worst_round,
        "worst_pct": worst_pct,
        "rounds": [
            {"round": r, "pct": p, "is_worst": r == worst_round}
            for r, p in sorted(pcts)
        ],
    }
