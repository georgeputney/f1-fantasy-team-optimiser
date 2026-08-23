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
# driver_teams snapshots which constructor each held driver was actually racing for at save time -
# a driver's id is permanent and team-independent, but their constructor can change round to round
# (grid seat swaps), so without this the only way to find "their team" is joining against whatever
# the CURRENT round's roster says, which silently relabels a driver under their new team even when
# looking at a squad from before the swap happened
def save_state(path, season, round_num, drivers, constructors, doubled_driver, prices, budget_remaining, free_transfers_carried, driver_teams=None):
    path = Path(path)

    state = {
        "season": season,
        "round": round_num,
        "drivers": list(drivers),
        "constructors": list(constructors),
        "doubled_driver": doubled_driver,
        "prices": prices,
        "budget_remaining": round(budget_remaining, 1),
        "free_transfers_carried": free_transfers_carried,
        "driver_teams": driver_teams or {},
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(state, f, indent=2)
