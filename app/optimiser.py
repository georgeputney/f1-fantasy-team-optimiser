"""ILP team optimiser - selects the optimal F1 fantasy team under budget and roster constraints using PuLP."""

import pulp

from app.config import BUDGET_CAP, DRIVER_ROSTER_SIZE, CONSTRUCTOR_ROSTER_SIZE


# selects the optimal fantasy team using ILP, returns selected drivers, constructors, and the doubled driver
def optimiser(driver_points, constructor_points, prices, budget=BUDGET_CAP):

    # objective: maximise total expected fantasy points including doubled driver bonus
    prob = pulp.LpProblem("f1_fantasy", pulp.LpMaximize)

    drivers = driver_points["driver_id"].tolist()
    constructors = constructor_points["constructor_id"].tolist()

    # binary so either prob = 1 (selected) or prob = 0 (not selected)
    selected = pulp.LpVariable.dicts("selected", drivers + constructors, cat="Binary")
    doubled = pulp.LpVariable.dicts("doubled", drivers, cat="Binary")

    prob += (
        pulp.lpSum(driver_points.set_index("driver_id")["expected_fantasy_points"][d] * selected[d] for d in drivers)
        + pulp.lpSum(constructor_points.set_index("constructor_id")["expected_fantasy_points"][c] * selected[c] for c in constructors)
        + pulp.lpSum(driver_points.set_index("driver_id")["expected_fantasy_points"][d] * doubled[d] for d in drivers)  # doubled driver scores an extra time
    )

    prob += pulp.lpSum(selected[d] for d in drivers) == DRIVER_ROSTER_SIZE
    prob += pulp.lpSum(selected[c] for c in constructors) == CONSTRUCTOR_ROSTER_SIZE

    prob += pulp.lpSum(doubled[d] for d in drivers) == 1  # exactly one doubled driver
    for d in drivers:
        prob += doubled[d] <= selected[d]  # can only double a selected driver

    prob += (
        pulp.lpSum(prices.set_index("asset_id")["price"][d] * selected[d] for d in drivers)
        + pulp.lpSum(prices.set_index("asset_id")["price"][c] * selected[c] for c in constructors)
    ) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected_drivers = [d for d in drivers if pulp.value(selected[d]) == 1]
    selected_constructors = [c for c in constructors if pulp.value(selected[c]) == 1]
    doubled_driver = [d for d in drivers if pulp.value(doubled[d]) == 1][0]
    
    return {
        "drivers": selected_drivers,
        "constructors": selected_constructors,
        "doubled_driver": doubled_driver,
    }
