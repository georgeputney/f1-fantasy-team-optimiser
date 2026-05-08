"""Persist and load team state between rounds for transfer-constrained optimisation."""

import json

from pathlib import Path


# load team state from a JSON file; returns None if no state file exists (e.g. start of season)
def load_state(path):
    path = Path(path)

    if not path.exists():
        return None
    
    with open(path) as f: 
        return json.load(f)


# write team state to disk after each round, overwriting the previous state
def save_state(path, season, round_num, drivers, constructors, doubled_driver, budget_remaining, free_transfers_carried):
    path = Path(path)

    state = {
        "season": season,
        "round": round_num,
        "drivers": list(drivers),
        "constructors": list(constructors),
        "doubled_driver": doubled_driver,
        "budget_remaining": round(budget_remaining, 1),
        "free_transfers_carried": free_transfers_carried,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(state, f, indent=2)
