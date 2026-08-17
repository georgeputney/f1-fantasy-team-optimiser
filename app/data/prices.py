"""Computes fantasy prices from targets and starting prices using the rolling PPM rule."""

import pandas as pd
import app.data.schemas as schemas

from app.config import PROCESSED_TARGETS_DIR, PROCESSED_PRICES_DIR, STARTING_PRICES_DIR, PRICE_FLOOR


# PPM thresholds and step sizes for price changes
PPM_THRESHOLDS = [0.6, 0.9, 1.2]
LOW_PRICE_STEPS = [-0.6, -0.2, 0.2, 0.6]    # price < 20
HIGH_PRICE_STEPS = [-0.3, -0.1, 0.1, 0.3]   # price >= 20
PRICE_BRACKET_CUTOFF = 18.5


# compute the price change for an asset given its rolling avg points and current price
def compute_price_change(avg_pts, price, floor):
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

    return new_price


# compute prices for a single round from the previous round's prices and recent targets
def compute_price_round(season, round_num):
    floor = PRICE_FLOOR.get(season, 3.5)

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
        recent_pts = [targets_by_round[r].get(asset_id, 0) for r in recent_rounds]
        avg_pts = sum(recent_pts) / len(recent_pts) if recent_pts else 0
        next_prices[asset_id] = compute_price_change(avg_pts, price, floor)

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

    # prior actual points per asset for rounds before this one
    history = {}
    for f in sorted(PROCESSED_TARGETS_DIR.glob(f"{season}_*.parquet")):
        rnd = int(f.stem.split("_")[1])
        if rnd < round_num:
            pts = pd.read_parquet(f).set_index("asset_id")["actual_fantasy_points"]
            for asset_id, v in pts.items():
                history.setdefault(asset_id, {})[rnd] = 0 if pd.isna(v) else v

    delta = {}
    for asset_id, price in dict(current_prices).items():
        prior = [history[asset_id][r] for r in sorted(history.get(asset_id, {}))]
        # next round is priced off the last 3 rounds' points, this round's being the prediction
        window = (prior + [float(predicted_points.get(asset_id, 0))])[-3:]
        avg_pts = sum(window) / len(window)
        delta[asset_id] = compute_price_change(avg_pts, float(price), floor) - float(price)

    return delta


# compute prices for all rounds of a season from starting prices and targets
def compute_prices(season):
    starting_prices = pd.read_csv(STARTING_PRICES_DIR / f"{season}.csv")
    floor = PRICE_FLOOR.get(season, 3.5)

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
    points_history = {asset_id: [] for asset_id in current_prices}

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
        # record this round's points
        pts = targets_by_round[rnd]
        for asset_id in current_prices:
            points_history[asset_id].append(pts.get(asset_id, 0))

        # compute next round's prices
        next_rnd = rounds[i + 1] if i + 1 < len(rounds) else rnd + 1
        next_prices = {}

        for asset_id, price in current_prices.items():
            recent = points_history[asset_id][-3:]
            avg_pts = sum(recent) / len(recent)
            next_prices[asset_id] = compute_price_change(avg_pts, price, floor)

        current_prices = next_prices

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
