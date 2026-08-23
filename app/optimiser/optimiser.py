"""ILP team optimiser - selects the optimal F1 fantasy team under budget and roster constraints using PuLP."""

import pulp

from app.config import BUDGET_CAP, DRIVER_ROSTER_SIZE, CONSTRUCTOR_ROSTER_SIZE


# selects the optimal fantasy team using ILP, returns selected drivers, constructors, and the doubled driver
# price_lambda weights expected next-round price gain (price_delta) into the objective, trading current
# points for future buying power; defaults to 0.0 so behaviour is unchanged unless a caller opts in
def optimiser(driver_points, constructor_points, prices, budget=BUDGET_CAP, state=None, price_delta=None, price_lambda=0.0):
    price_delta = price_delta or {}

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
        + price_lambda * pulp.lpSum(price_delta.get(i, 0.0) * selected[i] for i in drivers + constructors)  # reward holding assets due to rise in price
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


# enumerates the top-n legal teams by the same objective as optimiser(), each strictly different
# from every earlier one (no-good cuts: each prior team must lose at least one asset). Used for the
# ladder's "alternative teams" list - re-solves the ILP from scratch per team since PuLP has no
# native solution-pool API, so this costs one CBC solve per alternative
def enumerate_teams(driver_points, constructor_points, prices, budget=BUDGET_CAP, state=None, price_delta=None, price_lambda=0.0, n=5):
    price_delta = price_delta or {}

    prices_index = prices.set_index("asset_id")["price"]
    priced_assets = set(prices_index.index)
    drivers = [d for d in driver_points["driver_id"] if d in priced_assets]
    constructors = [c for c in constructor_points["constructor_id"] if c in priced_assets]
    driver_pts_map = driver_points.set_index("driver_id")["expected_fantasy_points"]
    constructor_pts_map = constructor_points.set_index("constructor_id")["expected_fantasy_points"]

    if state is not None:
        prev_team = set(state["drivers"] + state["constructors"])
        free_transfers = 2 + state["free_transfers_carried"]
        available_budget = state["budget_remaining"]
        for i in prev_team:
            available_budget += prices_index[i] if i in prices_index else state["prices"][i]
    else:
        prev_team = set()
        free_transfers = None
        available_budget = budget

    teams = []
    excluded_solutions = []
    for _ in range(n):
        prob = pulp.LpProblem("f1_fantasy_alt", pulp.LpMaximize)
        selected = pulp.LpVariable.dicts("selected", drivers + constructors, cat="Binary")
        doubled = pulp.LpVariable.dicts("doubled", drivers, cat="Binary")
        penalty_transfers = pulp.LpVariable("penalty_transfers", lowBound=0, cat="Continuous")

        prob += (
            pulp.lpSum(driver_pts_map[d] * selected[d] for d in drivers)
            + pulp.lpSum(constructor_pts_map[c] * selected[c] for c in constructors)
            + pulp.lpSum(driver_pts_map[d] * doubled[d] for d in drivers)
            + price_lambda * pulp.lpSum(price_delta.get(i, 0.0) * selected[i] for i in drivers + constructors)
            - 10 * penalty_transfers
        )
        prob += pulp.lpSum(selected[d] for d in drivers) == DRIVER_ROSTER_SIZE
        prob += pulp.lpSum(selected[c] for c in constructors) == CONSTRUCTOR_ROSTER_SIZE
        prob += pulp.lpSum(doubled[d] for d in drivers) == 1
        for d in drivers:
            prob += doubled[d] <= selected[d]
        prob += (
            pulp.lpSum(prices_index[d] * selected[d] for d in drivers)
            + pulp.lpSum(prices_index[c] * selected[c] for c in constructors)
        ) <= available_budget
        if state is not None:
            transfers_in = pulp.lpSum(selected[i] for i in drivers + constructors if i not in prev_team)
            prob += penalty_transfers >= transfers_in - free_transfers
        # no-good cut: each earlier solution must be missing at least one of its own assets here
        for prev_assets in excluded_solutions:
            prob += pulp.lpSum(selected[i] for i in prev_assets) <= len(prev_assets) - 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != "Optimal":
            break

        selected_drivers = [d for d in drivers if pulp.value(selected[d]) == 1]
        selected_constructors = [c for c in constructors if pulp.value(selected[c]) == 1]
        doubled_driver = [d for d in drivers if pulp.value(doubled[d]) == 1][0]
        transfer_penalty = round(pulp.value(penalty_transfers))
        total_points = (
            sum(driver_pts_map[d] for d in selected_drivers)
            + sum(constructor_pts_map[c] for c in selected_constructors)
            + driver_pts_map[doubled_driver]
            - 10 * transfer_penalty
        )
        teams.append({
            "drivers": selected_drivers,
            "constructors": selected_constructors,
            "doubled_driver": doubled_driver,
            "total_points": total_points,  # net of the transfer penalty - matches the ILP objective
            "transfer_penalty": transfer_penalty,
        })
        excluded_solutions.append(selected_drivers + selected_constructors)

    return teams
