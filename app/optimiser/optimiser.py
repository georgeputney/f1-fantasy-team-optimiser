"""ILP team optimiser - selects the optimal F1 fantasy team under budget and roster constraints using PuLP."""

import pulp

from app.config import BUDGET_CAP, DRIVER_ROSTER_SIZE, CONSTRUCTOR_ROSTER_SIZE


# selects the optimal fantasy team using ILP, returns selected drivers, constructors, and the doubled driver
def optimiser(driver_points, constructor_points, prices, budget=BUDGET_CAP, state=None):
    # objective: maximise total expected fantasy points including doubled driver bonus
    prob = pulp.LpProblem("f1_fantasy", pulp.LpMaximize)

    prices_index = prices.set_index("asset_id")["price"]
    priced_assets = set(prices_index.index)

    drivers = [d for d in driver_points["driver_id"] if d in priced_assets]
    constructors = [c for c in constructor_points["constructor_id"] if c in priced_assets]

    # compute available budget and transfer allowance from previous team state
    if state is not None:
        prev_team = set(state["drivers"] + state["constructors"])
        free_transfers = 2 + state["free_transfers_carried"]
        available_budget = state["budget_remaining"]

        dropped = []
        for i in prev_team:
            if i in prices_index:
                available_budget += prices_index[i]
            else:
                available_budget += state["prices"][i]
                dropped.append(i)
    else:
        prev_team = set()
        available_budget = budget
        dropped = []

    # binary so either prob = 1 (selected) or prob = 0 (not selected)
    selected = pulp.LpVariable.dicts("selected", drivers + constructors, cat="Binary")
    doubled = pulp.LpVariable.dicts("doubled", drivers, cat="Binary")
    # penalty variable: each transfer beyond free allowance costs 10 points
    penalty_transfers = pulp.LpVariable("penalty_transfers", lowBound=0, cat="Continuous")

    prob += (
        pulp.lpSum(driver_points.set_index("driver_id")["expected_fantasy_points"][d] * selected[d] for d in drivers)
        + pulp.lpSum(constructor_points.set_index("constructor_id")["expected_fantasy_points"][c] * selected[c] for c in constructors)
        + pulp.lpSum(driver_points.set_index("driver_id")["expected_fantasy_points"][d] * doubled[d] for d in drivers)  # doubled driver scores an extra time
        - 10 * penalty_transfers
    )

    prob += pulp.lpSum(selected[d] for d in drivers) == DRIVER_ROSTER_SIZE
    prob += pulp.lpSum(selected[c] for c in constructors) == CONSTRUCTOR_ROSTER_SIZE

    prob += pulp.lpSum(doubled[d] for d in drivers) == 1  # exactly one doubled driver
    for d in drivers:
        prob += doubled[d] <= selected[d]  # can only double a selected driver

    prob += (
        pulp.lpSum(prices_index[d] * selected[d] for d in drivers)
        + pulp.lpSum(prices_index[c] * selected[c] for c in constructors)
    ) <= available_budget

    # transfers_in counts new assets not held last round; penalty kicks in beyond free allowance
    if state is not None:
        transfers_in = pulp.lpSum(selected[i] for i in drivers + constructors if i not in prev_team)
        prob += penalty_transfers >= transfers_in - free_transfers

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected_drivers = [d for d in drivers if pulp.value(selected[d]) == 1]
    selected_constructors = [c for c in constructors if pulp.value(selected[c]) == 1]
    doubled_driver = [d for d in drivers if pulp.value(doubled[d]) == 1][0]
    
    return {
        "drivers": selected_drivers,
        "constructors": selected_constructors,
        "doubled_driver": doubled_driver,
        "transfers_made": sum(1 for i in selected_drivers + selected_constructors if i not in prev_team),
        "transfer_penalty": round(pulp.value(penalty_transfers)),
        "dropped": dropped,
    }
