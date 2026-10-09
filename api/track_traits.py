"""Builds the track traits payload - the circuit-level stats the pipeline already computes for the
upcoming race (data/processed/circuit_features/), each set against its average across the season's
calendar so a bare rate reads as high or low for a track. Independent of the squad selection, so like
the track record section this takes no budget/squad params.
"""

import pandas as pd

from app.config import PROCESSED_CIRCUIT_FEATURES_DIR, INTERIM_EVENTS_DIR, INTERIM_RACE_OVERTAKES_DIR

from api.common import load_predictions

# feature column -> payload key, in display order. every one is a rate in [0, 1] except the overtake
# index, which is this circuit's overtakes as a ratio to the season average (1.0 = average) and is
# turned into overtakes per driver below
_TRAITS = [
    ("circuit_overtake_index", "overtakes"),
    ("circuit_pole_to_win_rate", "pole_to_win"),
    ("circuit_top3_grid_to_podium_rate", "top3_to_podium"),
    ("circuit_fp3_top3_to_quali_top3_rate", "fp3_to_quali"),
    ("circuit_sc_vsc_rate", "safety_car"),
    ("circuit_dnf_rate", "dnf"),
]


def _value(x):
    return None if x is None or pd.isna(x) else float(x)


def build_track_traits():
    pred = load_predictions()
    season, rnd, circuit = pred["season"], pred["round"], pred["circuit"]
    race_id = f"{season}_{rnd:02d}"

    path = PROCESSED_CIRCUIT_FEATURES_DIR / f"{race_id}.parquet"
    if not path.exists():
        return {"available": False}
    row = pd.read_parquet(path).iloc[0]

    # the calendar average is over every round of this season that has circuit features, new venues
    # (all NaN) dropping out - they're features of the venue, not the race, so future rounds count too
    calendar = pd.concat(pd.read_parquet(f) for f in sorted(PROCESSED_CIRCUIT_FEATURES_DIR.glob(f"{season}_*.parquet")))

    event_path = INTERIM_EVENTS_DIR / f"{race_id}.parquet"
    event = pd.read_parquet(event_path).iloc[0] if event_path.exists() else None

    # overtakes per driver in this season's races so far - the average race the overtake index is
    # relative to. the track's figure is that scaled by its index, and the calendar figure is that
    # average itself (index 1.0 by definition), rather than a mean of per-track indices, which drifts
    # off 1.0 because each track's index is averaged over its own seasons
    ot_files = sorted(f for f in INTERIM_RACE_OVERTAKES_DIR.glob(f"{season}_*.parquet") if f.stem < race_id)
    season_ot = pd.concat(pd.read_parquet(f) for f in ot_files)["race_overtakes"].mean() if ot_files else None

    traits = []
    for col, key in _TRAITS:
        # a sprint weekend has no FP3, so how well FP3 predicts qualifying says nothing about this one
        if key == "fp3_to_quali" and event is not None and event["is_sprint"]:
            continue
        if key == "overtakes":
            index = _value(row.get(col))
            if season_ot is None or index is None:
                continue
            traits.append({"key": key, "value": index * float(season_ot), "calendar_avg": float(season_ot)})
            continue
        traits.append({"key": key, "value": _value(row.get(col)), "calendar_avg": _value(calendar[col].mean())})

    return {
        "available": True,
        "season": season,
        "round": rnd,
        "circuit": circuit,
        "event_name": None if event is None else event["event_name"],
        "is_street_circuit": None if event is None else bool(event["is_street_circuit"]),
        "is_sprint": None if event is None else bool(event["is_sprint"]),
        "n_seasons": int(row.get("circuit_data_n_seasons", 0) or 0),
        "traits": traits,
    }
