"""CLI entry points for the F1 fantasy optimiser pipeline."""

import json
import logging
import time
import typer
import fastf1
import pandas as pd
import matplotlib.pyplot as plt

from app.data.ingest import (
    get_event_metadata, get_practice_results,
    get_race_laps, get_race_results, get_qualifying_results,
    get_sprint_laps, get_sprint_results, get_sprint_qualifying_results,
    get_dhl_pitstops,
)
from app.data.clean import (
    clean_events, clean_practice_results,
    clean_race_laps, clean_race_results, clean_qualifying_results,
    clean_sprint_laps, clean_sprint_results, clean_sprint_qualifying_results,
    clean_pitstops,
)
from app.data.targets import compute_targets
from app.data.prices import compute_prices

from app.features.build_historic_features import build_historic_features
from app.features.build_practice_features import build_practice_features
from app.features.build_circuit_features import build_circuit_features

from app.models.configs import FINISH_POSITION_MODEL, QUALI_POSITION_MODEL
from app.models.train import main as train_main, load as load_model, load_data_upto, train_walk_forward, save_season
from app.models.predict import load_season_model, predict as run_predict
from app.models.compose import compose_drivers, compose_constructor
from app.data.overtakes import build_overtake_predictor
from app.data.dotd import build_dotd_predictor

from app.optimiser.optimiser import optimiser
from app.optimiser.state import load_state, save_state

from app.backtest import get_actual_team_points, oracle_baseline, random_baseline, lagged_baseline, mean_prior_baseline

from app.config import (
    ALL_SEASONS, VAL_SEASONS, BUDGET_CAP,
    INTERIM_EVENTS_DIR, INTERIM_FP1_DIR, INTERIM_FP2_DIR, INTERIM_FP3_DIR,
    INTERIM_SPRINT_QUALIFYING_DIR, INTERIM_SPRINT_DIR, INTERIM_QUALI_DIR, INTERIM_RACES_DIR,
    INTERIM_RACE_LAPS_DIR,
    PROCESSED_TARGETS_DIR, PROCESSED_PRICES_DIR, PROCESSED_HISTORIC_FEATURES_DIR,
    RACE_OVERTAKES_DIR,
    REPORTS_DIR, PREDICTIONS_DIR, TEAM_STATE_FILE
)

logging.getLogger("fastf1").setLevel(logging.WARNING)

app = typer.Typer(no_args_is_help=True)


# fetch raw race, qualifying, practice, and event metadata from FastF1 for the given seasons and rounds and write to data/raw/
@app.command()
def ingest_data(season: list[int] = typer.Option(ALL_SEASONS), round: list[int] = typer.Option(None)):
    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        sprint_rounds = set(schedule[schedule["EventFormat"].isin(["sprint", "sprint_qualifying", "sprint_shootout"])]["RoundNumber"])  # FP1/sprint sessions only exist on sprint weekends

        if round:
            schedule = schedule[schedule["RoundNumber"].isin(round)]

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            location = event.get("Location", str(round_num))

            typer.echo(f"Ingesting season {s}, round {round_num:02d} - {location}...")

            try:
                get_event_metadata(s, round_num)
                time.sleep(0.5)
            except Exception as e:
                typer.echo(f"  Skipping round {round_num:02d}: event metadata not available ({e})")
                continue

            for fetch_fn, label in [
                (lambda: get_race_results(s, round_num), "race results"),
                (lambda: get_race_laps(s, round_num), "race laps"),
                (lambda: get_qualifying_results(s, round_num), "qualifying"),
            ]:
                try:
                    fetch_fn()
                    time.sleep(0.5)
                except Exception as e:
                    typer.echo(f"  Warning: could not fetch {label} ({e})")

            try:
                get_dhl_pitstops(s, round_num)
            except Exception:
                pass  # DHL data may not be available yet for recent/future races

            for session_name in ["FP2", "FP3"]:
                try:
                    get_practice_results(s, round_num, session_name)
                    time.sleep(0.5)
                except Exception:
                    pass  # sprint weekends don't have FP2/FP3

            if round_num in sprint_rounds:
                try:
                    get_practice_results(s, round_num, "FP1")
                    time.sleep(0.5)
                except Exception:
                    pass

                try:
                    get_sprint_laps(s, round_num)
                    time.sleep(0.5)
                except Exception:
                    pass

                try:
                    get_sprint_qualifying_results(s, round_num)
                    time.sleep(0.5)
                except Exception:
                    pass  # non-sprint weekends

                try:
                    get_sprint_results(s, round_num)
                    time.sleep(0.5)
                except Exception:
                    pass  # non-sprint weekends


# clean raw parquet files for the given seasons and write validated tables to data/interim/
@app.command()
def clean_data(season: list[int] = typer.Option(ALL_SEASONS), round: list[int] = typer.Option(None)):
    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        sprint_rounds = set(schedule[schedule["EventFormat"].isin(["sprint", "sprint_qualifying", "sprint_shootout"])]["RoundNumber"])  # FP1/sprint sessions only exist on sprint weekends

        if round:
            schedule = schedule[schedule["RoundNumber"].isin(round)]
            
        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            location = event.get("Location", str(round_num))

            typer.echo(f"Cleaning season {s}, round {round_num:02d} - {location}...")

            try:
                clean_events(s, round_num)
                clean_race_laps(s, round_num)
                clean_race_results(s, round_num)
                clean_qualifying_results(s, round_num)
            except FileNotFoundError:
                typer.echo(f"  Skipping round {round_num:02d}: raw data not found (run ingest-data first)")
                continue

            try:
                clean_pitstops(s, round_num)
            except Exception:
                pass  # DHL data may not be available for all races

            for session_name in ["FP2", "FP3"]:
                try:
                    clean_practice_results(s, round_num, session_name)
                except Exception:
                    pass  # sprint weekends or rounds without practice data

            if round_num in sprint_rounds:
                try:
                    clean_practice_results(s, round_num, "FP1")
                    time.sleep(0.5)
                except Exception:
                    pass

                try:
                    clean_sprint_laps(s, round_num)
                except Exception:
                    pass

                try:
                    clean_sprint_qualifying_results(s, round_num)
                except Exception:
                    pass  # non-sprint weekends

                try:
                    clean_sprint_results(s, round_num)
                except Exception:
                    pass  # non-sprint weekends


# compute actual fantasy points from cleaned results and write to data/processed/targets/
@app.command()
def build_targets(season: list[int] = typer.Option(ALL_SEASONS), round: list[int] = typer.Option(None)):
    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        if round:
            schedule = schedule[schedule["RoundNumber"].isin(round)]

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            location = event.get("Location", str(round_num))

            typer.echo(f"Building targets for season {s}, round {round_num:02d} - {location}...")

            try:
                compute_targets(s, round_num)
            except (FileNotFoundError, ValueError):
                typer.echo(f"  Skipping round {round_num:02d}: cleaned data not found or incomplete (run clean-data first)")
                continue


# compute fantasy prices from starting prices and targets using rolling PPM rule, write to data/processed/prices/
@app.command()
def build_prices(season: list[int] = typer.Option(ALL_SEASONS)):
    for s in season:
        typer.echo(f"Computing prices for season {s}...")
        try:
            frames = compute_prices(s)
            typer.echo(f"  {len(frames)} rounds computed")
        except FileNotFoundError as e:
            typer.echo(f"  Skipping season {s}: {e}")


# build historic rolling features and practice session features for the given seasons and write to data/processed/
@app.command()
def build_features(season: list[int] = typer.Option(ALL_SEASONS), round: list[int] = typer.Option(None)):

    race_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_RACES_DIR.glob("*.parquet"))])
    quali_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_QUALI_DIR.glob("*.parquet"))])
    events = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_EVENTS_DIR.glob("*.parquet"))])
    fantasy_targets = pd.concat([pd.read_parquet(f) for f in sorted(PROCESSED_TARGETS_DIR.glob("*.parquet"))])

    race_laps_files = sorted(INTERIM_RACE_LAPS_DIR.glob("*.parquet"))
    race_laps_all = pd.concat([pd.read_parquet(f) for f in race_laps_files]) if race_laps_files else None

    fp3_files = sorted(INTERIM_FP3_DIR.glob("*.parquet"))
    fp3_all = pd.concat([pd.read_parquet(f) for f in fp3_files]) if fp3_files else None

    overtake_files = sorted(RACE_OVERTAKES_DIR.glob("*.csv"))
    if overtake_files:
        _ot_frames = []
        for _f in overtake_files:
            _s, _r = _f.stem.split("_")
            _df = pd.read_csv(_f)
            _df["season"] = int(_s)
            _df["round"] = int(_r)
            _ot_frames.append(_df)
        overtake_history = pd.concat(_ot_frames).reset_index(drop=True)
    else:
        overtake_history = pd.DataFrame(columns=["driver_id", "race_overtakes", "season", "round"])

    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        if round:
            schedule = schedule[schedule["RoundNumber"].isin(round)]

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            location = event.get("Location", str(round_num))

            typer.echo(f"Building features for season {s}, round {round_num:02d} - {location}...")

            try:
                result = build_historic_features(race_results, quali_results, fantasy_targets, events, overtake_history, s, round_num)
            except FileNotFoundError:
                typer.echo(f"  Skipping round {round_num:02d}: processed data not found (run build-targets first)")
                continue

            if result is None:
                typer.echo(f"  Skipping round {round_num:02d}: no race data available yet")
                continue

            if (INTERIM_FP3_DIR / f"{s}_{round_num:02d}.parquet").exists() or (INTERIM_FP1_DIR / f"{s}_{round_num:02d}.parquet").exists():
                build_practice_features(s, round_num)

            build_circuit_features(race_results, quali_results, events, race_laps_all, fp3_all, overtake_history, s, round_num)


# train models - without --season runs dev training with eval metrics; with --season trains walk-forward season artifacts
@app.command()
def train_model(season: int = typer.Option(None)):
    if season is not None:
        typer.echo(f"Training season {season} model on all data before season {season}...")

        X_quali, y_quali = load_data_upto(QUALI_POSITION_MODEL, season, 1)
        quali_model = train_walk_forward(QUALI_POSITION_MODEL, X_quali, y_quali)

        X_finish, y_finish = load_data_upto(FINISH_POSITION_MODEL, season, 1, quali_model=quali_model, quali_config=QUALI_POSITION_MODEL)
        finish_model = train_walk_forward(FINISH_POSITION_MODEL, X_finish, y_finish)

        save_season(quali_model, QUALI_POSITION_MODEL, season)
        save_season(finish_model, FINISH_POSITION_MODEL, season)

        typer.echo(f"Saved season {season} artifacts.")
    else:
        typer.echo(f"Training quali position model...")
        train_main(QUALI_POSITION_MODEL)

        typer.echo(f"\nTraining finish position model...")
        quali_model = load_model(QUALI_POSITION_MODEL)
        train_main(FINISH_POSITION_MODEL, quali_model, QUALI_POSITION_MODEL)



# generate reports/predictions_latest.json for the Streamlit app
@app.command()
def generate_reports(season: int = typer.Option(...), round: int = typer.Option(...)):
    from datetime import datetime

    prices_path = PROCESSED_PRICES_DIR / f"{season}_{round:02d}.parquet"
    if not prices_path.exists():
        typer.echo(f"No prices file found: {prices_path} (run build-prices first)")
        raise typer.Exit(1)

    prices = pd.read_parquet(prices_path)
    prices_index = prices.set_index("asset_id")["price"]

    events_path = INTERIM_EVENTS_DIR / f"{season}_{round:02d}.parquet"
    circuit = pd.read_parquet(events_path)["location"].iloc[0] if events_path.exists() else f"Round {round}"

    quali_model = load_season_model(QUALI_POSITION_MODEL, season)
    finish_model = load_season_model(FINISH_POSITION_MODEL, season)

    typer.echo(f"Generating reports for season {season}, round {round:02d} - {circuit}...")

    predict_overtakes = build_overtake_predictor()
    predict_dotd = build_dotd_predictor()
    predictions = run_predict(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, season, round)
    driver_pts = compose_drivers(predictions, location=circuit, season=season, predict_overtakes=predict_overtakes, predict_dotd=predict_dotd)
    constructor_pts = compose_constructor(driver_pts)

    drivers_out = [
        {
            "driver_id": row["driver_id"],
            "constructor_id": row["constructor_id"],
            "expected_points": float(row["expected_fantasy_points"]),
            "price": float(prices_index.get(row["driver_id"], 0)),
            "predicted_quali_position": int(row["predicted_quali_position"]),
            "predicted_finish_position": int(row["predicted_finish_position"]),
            "points_breakdown": {
                "quali": float(row.get("points_quali", 0)),
                "finish": float(row.get("points_finish", 0)),
                "positions_gained": float(row.get("points_positions_gained", 0)),
                "overtakes": float(row.get("expected_overtakes", 0)),
                "prob_fl": float(row.get("prob_fl", 0)),
                "prob_dotd": float(row.get("prob_dotd", 0)),
                "sprint": float(row.get("points_sprint", 0)),
            },
        }
        for _, row in driver_pts.iterrows()
    ]

    constructors_out = [
        {
            "constructor_id": row["constructor_id"],
            "expected_points": float(row["expected_fantasy_points"]),
            "price": float(prices_index.get(row["constructor_id"], 0)),
        }
        for _, row in constructor_pts.iterrows()
    ]

    output = {
        "season": season,
        "round": round,
        "circuit": str(circuit),
        "generated_at": datetime.now().isoformat(),
        "drivers": sorted(drivers_out, key=lambda x: -x["expected_points"]),
        "constructors": sorted(constructors_out, key=lambda x: -x["expected_points"]),
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    versioned_path = PREDICTIONS_DIR / f"predictions_{season}_{round:02d}.json"
    with open(versioned_path, "w") as f:
        json.dump(output, f, indent=2)

    typer.echo(f"Saved to {versioned_path}")


# generate versioned predictions for all historical races that have prices but no saved file yet
@app.command()
def backfill_predictions(from_season: int = typer.Option(2026), prod: bool = typer.Option(True), overwrite: bool = typer.Option(False)):
    from datetime import datetime

    predict_overtakes = build_overtake_predictor()
    predict_dotd = build_dotd_predictor()

    prices_files = sorted(PROCESSED_PRICES_DIR.glob("*.parquet"))
    prices_files = [p for p in prices_files if int(p.stem.split("_")[0]) >= from_season]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    skipped, saved = 0, 0
    _current_season = None
    walk_quali = walk_finish = None

    for prices_path in prices_files:
        stem = prices_path.stem  # e.g. 2024_01
        season_str, round_str = stem.split("_")
        s, r = int(season_str), int(round_str)

        out_path = PREDICTIONS_DIR / f"predictions_{s}_{r:02d}.json"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        # load season artifact once per season (run train-model --season S first)
        if s != _current_season:
            typer.echo(f"Loading season {s} model artifacts...")
            walk_quali = load_season_model(QUALI_POSITION_MODEL, s)
            walk_finish = load_season_model(FINISH_POSITION_MODEL, s)
            _current_season = s

        events_path = INTERIM_EVENTS_DIR / f"{s}_{r:02d}.parquet"
        circuit = pd.read_parquet(events_path)["location"].iloc[0] if events_path.exists() else f"Round {r}"

        typer.echo(f"Generating {s} R{r:02d} - {circuit}...")
        try:

            prices = pd.read_parquet(prices_path)
            prices_index = prices.set_index("asset_id")["price"]

            predictions = run_predict(walk_quali, QUALI_POSITION_MODEL, walk_finish, FINISH_POSITION_MODEL, s, r)
            driver_pts = compose_drivers(predictions, location=circuit, season=s, predict_overtakes=predict_overtakes, predict_dotd=predict_dotd)
            constructor_pts = compose_constructor(driver_pts)

            drivers_out = [
                {
                    "driver_id": row["driver_id"],
                    "constructor_id": row["constructor_id"],
                    "expected_points": float(row["expected_fantasy_points"]),
                    "price": float(prices_index.get(row["driver_id"], 0)),
                    "predicted_quali_position": int(row["predicted_quali_position"]),
                    "predicted_finish_position": int(row["predicted_finish_position"]),
                    "points_breakdown": {
                        "quali": float(row.get("points_quali", 0)),
                        "finish": float(row.get("points_finish", 0)),
                        "positions_gained": float(row.get("points_positions_gained", 0)),
                        "overtakes": float(row.get("expected_overtakes", 0)),
                        "prob_fl": float(row.get("prob_fl", 0)),
                        "prob_dotd": float(row.get("prob_dotd", 0)),
                        "sprint": float(row.get("points_sprint", 0)),
                    },
                }
                for _, row in driver_pts.iterrows()
            ]
            constructors_out = [
                {
                    "constructor_id": row["constructor_id"],
                    "expected_points": float(row["expected_fantasy_points"]),
                    "price": float(prices_index.get(row["constructor_id"], 0)),
                }
                for _, row in constructor_pts.iterrows()
            ]
            output = {
                "season": s, "round": r, "circuit": str(circuit),
                "generated_at": datetime.now().isoformat(),
                "drivers": sorted(drivers_out, key=lambda x: -x["expected_points"]),
                "constructors": sorted(constructors_out, key=lambda x: -x["expected_points"]),
            }
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            saved += 1
        except Exception as e:
            typer.echo(f"  skipped ({e})")

    typer.echo(f"\nDone - {saved} saved, {skipped} already existed (use --overwrite to regenerate).")


# load the trained model, predict finish positions, and print expected fantasy points for drivers and constructors
@app.command()
def predict_race(season: int = typer.Option(...), round: int = typer.Option(...)):
    quali_model = load_season_model(QUALI_POSITION_MODEL, season)
    finish_model = load_season_model(FINISH_POSITION_MODEL, season)

    events_path = INTERIM_EVENTS_DIR / f"{season}_{round:02d}.parquet"
    location = pd.read_parquet(events_path)["location"].iloc[0] if events_path.exists() else None

    predict_overtakes = build_overtake_predictor()
    predict_dotd = build_dotd_predictor()
    predictions = run_predict(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, season, round)

    driver_points = compose_drivers(predictions, location=location, season=season, predict_overtakes=predict_overtakes, predict_dotd=predict_dotd)
    constructor_points = compose_constructor(driver_points)

    typer.echo(f"Predicting season {season}, round {round:02d} - {location}...")
    
    typer.echo("\nDrivers:")
    typer.echo(driver_points[["driver_id", "predicted_quali_position", "predicted_finish_position", "expected_overtakes", "expected_fantasy_points"]].to_string())
    typer.echo("\nConstructors:")
    typer.echo(constructor_points.to_string())


# load predictions, compose expected points, and select the optimal team under budget and transfer constraints
# loads team state from data/team_state.json if it exists; saves updated state after each run unless --no-state is passed
# dropped/inactive assets are sold at last known price and warned to the user
@app.command()
def optimise_team(season: int = typer.Option(...), round: int = typer.Option(...), budget: float = typer.Option(BUDGET_CAP), no_state: bool = typer.Option(False)):
    prices = pd.read_parquet(PROCESSED_PRICES_DIR / f"{season}_{round:02d}.parquet")
    state = None if no_state else load_state(TEAM_STATE_FILE)

    quali_model = load_season_model(QUALI_POSITION_MODEL, season)
    finish_model = load_season_model(FINISH_POSITION_MODEL, season)

    events_path = INTERIM_EVENTS_DIR / f"{season}_{round:02d}.parquet"
    location = pd.read_parquet(events_path)["location"].iloc[0] if events_path.exists() else None

    typer.echo(f"Optimising team for season {season}, round {round:02d} - {location}...")

    predict_overtakes = build_overtake_predictor()
    predict_dotd = build_dotd_predictor()
    predictions = run_predict(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, season, round)

    driver_points = compose_drivers(predictions, location=location, season=season, predict_overtakes=predict_overtakes, predict_dotd=predict_dotd)
    constructor_points = compose_constructor(driver_points)

    team = optimiser(driver_points, constructor_points, prices, budget, state)

    for d in team["dropped"]:
        typer.echo(f"\n  [!] {d} is inactive - sold at last known price £{state['prices'][d]:.1f}M")

    driver_points = driver_points.set_index("driver_id")["expected_fantasy_points"]
    constructor_points = constructor_points.set_index("constructor_id")["expected_fantasy_points"]
    
    asset_prices = prices.set_index("asset_id")["price"]
    available_budget = (state["budget_remaining"] + sum(asset_prices.get(i, state["prices"][i]) for i in state["drivers"] + state["constructors"])) if state else budget

    total = 0.0

    typer.echo("\nDrivers:")
    for d in team["drivers"]:
        points = driver_points[d] * (2 if d == team["doubled_driver"] else 1)
        price = asset_prices[d]

        doubled_marker = " [x2]" if d == team["doubled_driver"] else ""

        typer.echo(f"  {d:<30} {points:>6.1f} points    £{price:.1f}M{doubled_marker}")
        total += points

    typer.echo("\nConstructors:")
    for c in team["constructors"]:
        points = constructor_points[c]
        price = asset_prices[c]

        typer.echo(f"  {c:<30} {points:>6.1f} points    £{price:.1f}M")
        total += points

    total_price = sum(asset_prices[d] for d in team["drivers"]) + sum(asset_prices[c] for c in team["constructors"])

    typer.echo(f"\nTotal projected points: {total:.1f}")
    typer.echo(f"Total cost: £{total_price:.1f}M / £{available_budget:.1f}M")

    new_budget_remaining = available_budget - total_price
    typer.echo(f"Budget remaining: £{new_budget_remaining:.1f}M")

    if not no_state:
        free_transfers = 2 + (state["free_transfers_carried"] if state else 0)
        free_transfers_carried = 1 if team["transfers_made"] < free_transfers else 0

        team_prices = {i: float(asset_prices[i]) for i in team["drivers"] + team["constructors"]}
        save_state(TEAM_STATE_FILE, season, round, team["drivers"], team["constructors"], team["doubled_driver"], team_prices, new_budget_remaining, free_transfers_carried)

        typer.echo(f"\nTransfers made: {team['transfers_made']} ({team['transfer_penalty']} point penalty)")
        typer.echo(f"Free transfers carried to next round: {free_transfers_carried}")


# runs walk-forward backtest comparing model, oracle, and random strategies over historical seasons, prints per-round results and saves a cumulative points plot
# model and oracle are transfer-constrained with state carried forward each round; random is unconstrained
@app.command()
def backtest(season: list[int] = typer.Option(VAL_SEASONS), budget: float = typer.Option(BUDGET_CAP)):
    predict_overtakes = build_overtake_predictor()
    predict_dotd = build_dotd_predictor()

    for s in season:
        results = []

        model_state = None
        oracle_state = None  # reset at start of each season - no carry-over between seasons
        mean_state = None


        typer.echo(f"Loading season {s} model artifacts (run train-model --season {s} first if missing)...")
        quali_model = load_season_model(QUALI_POSITION_MODEL, s)
        finish_model = load_season_model(FINISH_POSITION_MODEL, s)

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0]

        for _, event in schedule.iterrows():
            round_num = event["RoundNumber"]
            location = event.get("Location", str(round_num))
            prices_path = PROCESSED_PRICES_DIR / f"{s}_{round_num:02d}.parquet"
            features_path = PROCESSED_HISTORIC_FEATURES_DIR / f"{s}_{round_num:02d}.parquet"
            targets_path = PROCESSED_TARGETS_DIR / f"{s}_{round_num:02d}.parquet"

            if not (prices_path.exists() and features_path.exists() and targets_path.exists()):
                continue

            typer.echo(f"Backtesting season {s}, round {round_num:02d} - {location}...")

            prices = pd.read_parquet(prices_path)
            asset_prices_index = prices.set_index("asset_id")["price"]

            events_path = INTERIM_EVENTS_DIR / f"{s}_{round_num:02d}.parquet"
            bt_location = pd.read_parquet(events_path)["location"].iloc[0] if events_path.exists() else None

            predictions = run_predict(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, s, round_num)

            driver_points = compose_drivers(predictions, location=bt_location, season=s, predict_overtakes=predict_overtakes, predict_dotd=predict_dotd)
            constructor_points = compose_constructor(driver_points)

            model_team = optimiser(driver_points, constructor_points, prices, budget, model_state)
            model_points = get_actual_team_points(model_team, s, round_num, model_team["transfer_penalty"]) # deduct transfer penalty from actual points
            model_state = _build_state(model_team, model_state, asset_prices_index, budget) # carry state forward to next round

            oracle_team = oracle_baseline(s, round_num, prices, budget, oracle_state)
            oracle_points = get_actual_team_points(oracle_team, s, round_num, oracle_team["transfer_penalty"]) # oracle is also transfer-constrained - true upper bound
            oracle_state = _build_state(oracle_team, oracle_state, asset_prices_index, budget)

            random_points = random_baseline(s, round_num, prices, budget)

            lagged_team = lagged_baseline(s, round_num, prices, budget)
            lagged_points = get_actual_team_points(lagged_team, s, round_num) if lagged_team else None

            mean_team = mean_prior_baseline(s, round_num, prices, budget, mean_state)
            if mean_team:
                mean_points = get_actual_team_points(mean_team, s, round_num, mean_team["transfer_penalty"])
                mean_state = _build_state(mean_team, mean_state, asset_prices_index, budget)
            else:
                mean_points = None

            results.append({"season": s, "round": round_num, "location": location, "model": model_points, "oracle": oracle_points, "random": random_points, "lagged": lagged_points, "mean": mean_points})

        df = pd.DataFrame(results)

        typer.echo(f"\n{'Round':<6} {'Location':<18} {'Model':>8} {'Oracle':>8} {'Mean':>8} {'Lagged':>8} {'Random':>8}")
        for _, row in df.iterrows():
            lagged_str = f"{row['lagged']:>8.1f}" if pd.notna(row["lagged"]) else f"{'N/A':>8}"
            mean_str = f"{row['mean']:>8.1f}" if pd.notna(row["mean"]) else f"{'N/A':>8}"
            typer.echo(f"  {int(row['round']):<4} {row['location']:<18} {row['model']:>8.1f} {row['oracle']:>8.1f} {mean_str} {lagged_str} {row['random']:>8.1f}")

        typer.echo(f"\n{'Total':<8} {df['model'].sum():>8.1f} {df['oracle'].sum():>8.1f} {df['mean'].sum():>8.1f} {df['lagged'].sum():>8.1f} {df['random'].sum():>8.1f}")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        df[["model", "oracle", "mean", "lagged", "random"]].cumsum().plot(title="Cumulative fantasy points by strategy", color=["blue", "orange", "purple", "red", "green"])

        plt.xlabel("Round")
        plt.ylabel("Cumulative points")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"backtest_{s}.png")
        plt.close()

        with open(REPORTS_DIR / f"backtest_{s}.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)

        typer.echo(f"\nPlot saved to reports/\n")


# builds in-memory team state after each backtest round - mirrors save_state but without file I/O
def _build_state(team, previous_state, asset_prices_index, budget):
    # sell previous team at current prices; fall back to stored price for dropped/inactive assets
    available = (previous_state["budget_remaining"] + sum(asset_prices_index.get(i, previous_state["prices"][i]) for i in previous_state["drivers"] + previous_state["constructors"])) if previous_state else budget
    
    new_budget = available - sum(asset_prices_index[i] for i in team["drivers"] + team["constructors"])
    free_transfers = 2 + (previous_state["free_transfers_carried"] if previous_state else 0)
    
    return {
        "drivers": team["drivers"],
        "constructors": team["constructors"],
        "prices": {i: float(asset_prices_index[i]) for i in team["drivers"] + team["constructors"]},
        "budget_remaining": round(new_budget, 1),
        "free_transfers_carried": 1 if team["transfers_made"] < free_transfers else 0,  # carry 1 if transfers unused
    }


if __name__ == "__main__": app()