"""Computes fantasy prices from targets and starting prices using the rolling PPM rule."""

import pandas as pd

from app.config import PROCESSED_TARGETS_DIR, PROCESSED_PRICES_DIR, STARTING_PRICES_DIR, PRICE_FLOOR, PRICE_CEILING

# imported lazily inside the two offline-pipeline functions that validate against it below -
# expected_price_delta (the only function the live API calls) never needs it, and pandera is
# ~200ms of import time the API would otherwise pay at every cold start for no reason


# mid-season seat changes, which the rolling PPM rule can't derive from points alone. Each listed
# driver races a sequence of seat stints (first round, last round or None for open-ended, entry
# price); a driver not listed holds one seat all season. Outside every stint the driver is off the
# grid and out of the price table. A stint after the first is hand-priced by the game at its first
# round rather than inheriting the previous seat's price, and a price only moves on the points scored
# in its own stint - see pricing_points. Drivers only - a constructor never changes seat. Keyed by season.
SEAT_STINTS = {
    2026: {
        "isack_hadjar": [(1, 11, None), (15, None, 14.5)],                 # out injured r12-r14
        "liam_lawson":  [(1, 11, None), (12, 14, 14.5), (15, None, 9.7)],  # covered Hadjar's Red Bull seat r12-r14
        "yuki_tsunoda": [(12, 14, 10.3)],                                  # filled Lawson's Racing Bulls seat r12-r14
    },
}


# prices the game set off-rule, as published, keyed by season then round. These are steps the rolling PPM
# rule can't produce from the final points: at r9 the game moved Gasly +0.6 and Lawson +0.2 where their
# r6-r8 points give -0.2 and +0.6. Both moves fit their pre-penalty r6 scores (Gasly 25, Lawson 13)
# instead, so the game looks to have priced r9 off r6 as first scored, before Gasly's post-race
# penalty - yet r7 was priced off the corrected scores. Every later round falls out of the rule unaided
PRICE_ANCHORS = {
    2026: {
        9: {"pierre_gasly": 12.8, "liam_lawson": 8.9},
    },
}


# PPM thresholds and step sizes for price changes
PPM_THRESHOLDS = [0.6, 0.9, 1.2]
LOW_PRICE_STEPS = [-0.6, -0.2, 0.2, 0.6]    # price < 20
HIGH_PRICE_STEPS = [-0.3, -0.1, 0.1, 0.3]   # price >= 20
PRICE_BRACKET_CUTOFF = 18.5


# index of the seat stint an asset is in at a round - 0 for an asset with no listed stints, None for
# a round a listed driver spends off the grid
def seat_stint(season, round_num, asset_id):
    stints = SEAT_STINTS.get(season, {}).get(asset_id)
    if stints is None:
        return 0

    for i, (first, last, _) in enumerate(stints):
        if first <= round_num and (last is None or round_num <= last):
            return i
    return None


# the points one round contributes to the rolling window that moves an asset's price at stint_round -
# None for a round spent in a different stint (or off the grid), and 0 (not "skipped") for a round in
# the same stint with no result at all.
# None drops out of the window entirely rather than scoring 0: the game prices a seat off only the
# rounds its current driver has actually raced it, so a driver's first race in a new seat is averaged
# over one round, not padded to three with zeros or mixed with points banked in the old seat. Lawson's
# r14 Red Bull price (14.3 -> 14.9) only comes out of the rule from his Red Bull rounds alone
def pricing_points(season, round_num, asset_id, points, stint_round):
    if seat_stint(season, round_num, asset_id) != seat_stint(season, stint_round, asset_id):
        return None

    return 0.0 if points is None or pd.isna(points) else float(points)


# mean of a rolling window, ignoring the rounds pricing_points marked as out-of-seat
def window_average(window):
    scored = [p for p in window if p is not None]
    return sum(scored) / len(scored) if scored else 0.0


# applies this round's hand-set prices to a freshly computed price table: a driver starting a new stint
# takes its hand-set price (and is added if the previous round's table didn't have them), a driver
# off the grid this round is removed outright, so the optimiser never offers someone who isn't racing,
# and any off-rule price the game published this round replaces the rule's (see PRICE_ANCHORS)
def apply_hand_set_prices(season, round_num, prices, asset_types):
    for asset_id, stints in SEAT_STINTS.get(season, {}).items():
        stint = seat_stint(season, round_num, asset_id)
        if stint is None:
            prices.pop(asset_id, None)
            continue

        first, _, entry_price = stints[stint]
        if first == round_num and entry_price is not None:
            prices[asset_id] = entry_price
            asset_types.setdefault(asset_id, "driver")

    for asset_id, price in PRICE_ANCHORS.get(season, {}).get(round_num, {}).items():
        prices[asset_id] = price

    return prices


# compute the price change for an asset given its rolling avg points and current price
def compute_price_change(avg_pts, price, floor, ceiling=None):
    ppm = avg_pts / price
    steps = LOW_PRICE_STEPS if price < PRICE_BRACKET_CUTOFF else HIGH_PRICE_STEPS

    if ppm < PPM_THRESHOLDS[0]:
        change = steps[0]
    elif ppm < PPM_THRESHOLDS[1]:
        change = steps[1]
    elif ppm < PPM_THRESHOLDS[2]:
        change = steps[2]
    else:
        change = steps[3]

    new_price = round(price + change, 1)
    if new_price < floor:
        new_price = floor
    if ceiling is not None and new_price > ceiling:
        new_price = ceiling

    return new_price


# compute prices for a single round from the previous round's prices and recent targets
def compute_price_round(season, round_num):
    import app.data.schemas as schemas

    floor = PRICE_FLOOR.get(season, 3.5)
    ceiling = PRICE_CEILING.get(season)

    # load previous round's prices as the base
    prev_path = PROCESSED_PRICES_DIR / f"{season}_{round_num - 1:02d}.parquet"
    if not prev_path.exists():
        raise FileNotFoundError(f"Previous round prices not found: {prev_path}")
    prev_prices = pd.read_parquet(prev_path)
    current_prices = prev_prices.set_index("asset_id")["price"].to_dict()
    asset_types = prev_prices.set_index("asset_id")["asset_type"].to_dict()

    # collect targets for the rolling window (up to 3 most recent rounds before round_num)
    target_files = sorted(PROCESSED_TARGETS_DIR.glob(f"{season}_*.parquet"))
    targets_by_round = {}
    for f in target_files:
        rnd = int(f.stem.split("_")[1])
        if rnd < round_num:
            targets_by_round[rnd] = pd.read_parquet(f).set_index("asset_id")["actual_fantasy_points"]

    recent_rounds = sorted(targets_by_round.keys())[-3:]

    next_prices = {}
    for asset_id, price in current_prices.items():
        recent_pts = [pricing_points(season, r, asset_id, targets_by_round[r].get(asset_id, 0), round_num - 1) for r in recent_rounds]
        avg_pts = window_average(recent_pts)
        next_prices[asset_id] = compute_price_change(avg_pts, price, floor, ceiling)

    next_prices = apply_hand_set_prices(season, round_num, next_prices, asset_types)

    price_df = pd.DataFrame({
        "race_id": f"{season}_{round_num:02d}",
        "asset_id": list(next_prices.keys()),
        "asset_type": [asset_types[a] for a in next_prices],
        "price": [next_prices[a] for a in next_prices],
    })
    schemas.fantasy_prices.validate(price_df)

    PROCESSED_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_PRICES_DIR / f"{season}_{round_num:02d}.parquet"
    price_df.to_parquet(out_path, index=False)

    return price_df


# expected next-round price change per asset, using predicted points for the current round
# mirrors the rolling PPM rule (compute_price_round) but substitutes predicted points for this
# round in place of the not-yet-known actual, so it uses no look-ahead - safe for the optimiser
def expected_price_delta(season, round_num, current_prices, predicted_points):
    floor = PRICE_FLOOR.get(season, 3.5)
    ceiling = PRICE_CEILING.get(season)

    # prior actual points per round for rounds before this one
    history = {}
    for f in sorted(PROCESSED_TARGETS_DIR.glob(f"{season}_*.parquet")):
        rnd = int(f.stem.split("_")[1])
        if rnd < round_num:
            history[rnd] = pd.read_parquet(f).set_index("asset_id")["actual_fantasy_points"]

    # next round is priced off the last 3 rounds' points, this round's being the prediction. Indexed
    # by round rather than by the rows an asset happens to have, so a round a driver sat out counts
    # as the 0 compute_price_round gives it instead of silently pulling in an older round in its place
    prior_rounds = sorted(history)[-2:]

    delta = {}
    for asset_id, price in dict(current_prices).items():
        window = [pricing_points(season, r, asset_id, history[r].get(asset_id, 0), round_num) for r in prior_rounds]
        window.append(float(predicted_points.get(asset_id, 0)))
        avg_pts = window_average(window)
        delta[asset_id] = compute_price_change(avg_pts, float(price), floor, ceiling) - float(price)

    return delta


# compute prices for all rounds of a season from starting prices and targets
def compute_prices(season):
    import app.data.schemas as schemas

    starting_prices = pd.read_csv(STARTING_PRICES_DIR / f"{season}.csv")
    floor = PRICE_FLOOR.get(season, 3.5)
    ceiling = PRICE_CEILING.get(season)

    # collect all available targets for this season
    target_files = sorted(PROCESSED_TARGETS_DIR.glob(f"{season}_*.parquet"))
    targets_by_round = {}
    for f in target_files:
        rnd = int(f.stem.split("_")[1])
        targets_by_round[rnd] = pd.read_parquet(f).set_index("asset_id")["actual_fantasy_points"]

    rounds = sorted(targets_by_round.keys())
    if not rounds:
        return []

    # initialise current prices from starting prices
    current_prices = starting_prices.set_index("asset_id")["price"].to_dict()
    asset_types = starting_prices.set_index("asset_id")["asset_type"].to_dict()

    all_price_frames = []

    # round 1 prices are the starting prices
    r1_df = pd.DataFrame({
        "race_id": f"{season}_{rounds[0]:02d}",
        "asset_id": list(current_prices.keys()),
        "asset_type": [asset_types[a] for a in current_prices],
        "price": [current_prices[a] for a in current_prices],
    })
    schemas.fantasy_prices.validate(r1_df)
    all_price_frames.append(r1_df)

    for i, rnd in enumerate(rounds):
        # compute next round's prices off the last 3 rounds up to and including this one, indexed by
        # round so a stint change can drop rounds out of the window (see pricing_points)
        next_rnd = rounds[i + 1] if i + 1 < len(rounds) else rnd + 1
        recent_rounds = rounds[max(0, i - 2):i + 1]
        next_prices = {}

        for asset_id, price in current_prices.items():
            recent = [pricing_points(season, r, asset_id, targets_by_round[r].get(asset_id, 0), rnd) for r in recent_rounds]
            avg_pts = window_average(recent)
            next_prices[asset_id] = compute_price_change(avg_pts, price, floor, ceiling)

        current_prices = apply_hand_set_prices(season, next_rnd, next_prices, asset_types)

        price_df = pd.DataFrame({
            "race_id": f"{season}_{next_rnd:02d}",
            "asset_id": list(current_prices.keys()),
            "asset_type": [asset_types[a] for a in current_prices],
            "price": [current_prices[a] for a in current_prices],
        })
        schemas.fantasy_prices.validate(price_df)
        all_price_frames.append(price_df)

    # write all rounds
    PROCESSED_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    for price_df in all_price_frames:
        race_id = price_df["race_id"].iloc[0]
        price_df.to_parquet(PROCESSED_PRICES_DIR / f"{race_id}.parquet", index=False)

    return all_price_frames
