"""CLI entry points for the F1 fantasy optimiser pipeline."""

import logging
import typer
import fastf1
import pandas as pd
import matplotlib.pyplot as plt

from app.data.ingest import get_event_metadata, get_race_results, get_qualifying_results, get_practice_results
from app.data.clean import clean_events, clean_race_results, clean_qualifying_results
from app.data.targets import compute_targets

from app.features.build_driver_features import build_driver_features

from app.models.configs import FINISH_POSITION_MODEL, QUALI_POSITION_MODEL
from app.models.train import main as train_main
from app.models.predict import load_model, predict as run_predict
from app.models.compose import compose_drivers, compose_constructor

from app.optimiser import optimiser

from app.backtest import get_actual_team_points, oracle_baseline, random_baseline

from app.config import ALL_SEASONS, VAL_SEASONS, BUDGET_CAP, FANTASY_PRICES_DIR, INTERIM_EVENTS_DIR, INTERIM_QUALI_DIR, INTERIM_RACES_DIR, PROCESSED_TARGETS_DIR, PROCESSED_DRIVER_FEATURES_DIR, REPORTS_DIR

logging.getLogger("fastf1").setLevel(logging.WARNING)

app = typer.Typer(no_args_is_help=True)


# fetch raw race, qualifying, practice, and event metadata from FastF1 for the given seasons and rounds and write to data/raw/
@app.command()
def ingest_data(season: list[int] = typer.Option(ALL_SEASONS), round: list[int] = typer.Option(None)):
    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        if round:
            schedule = schedule[schedule["RoundNumber"].isin(round)]

        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Ingesting season {s}, round {round_num:02d}...")

            get_event_metadata(s, round_num)
            get_race_results(s, round_num)
            get_qualifying_results(s, round_num)

            for session_name in ["FP2", "FP3"]:
                try:
                    get_practice_results(s, round_num, session_name)
                except Exception:
                    pass  # sprint weekends don't have FP2/FP3


# clean raw parquet files for the given seasons and write validated tables to data/interim/
@app.command()
def clean_data(season: list[int] = typer.Option(ALL_SEASONS)):
    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Cleaning season {s}, round {round_num:02d}...")

            clean_events(s, round_num)
            clean_race_results(s, round_num)
            clean_qualifying_results(s, round_num)


# compute actual fantasy points from cleaned results and write to data/processed/targets/
@app.command()
def build_targets(season: list[int] = typer.Option(ALL_SEASONS)):
    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Building targets for season {s}, round {round_num:02d}...")

            compute_targets(s, round_num)


# build driver and constructor features for the given seasons and write to data/processed/driver_features/
@app.command()
def build_features(season: list[int] = typer.Option(ALL_SEASONS)):

    race_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_RACES_DIR.glob("*.parquet"))])
    quali_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_QUALI_DIR.glob("*.parquet"))])
    events = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_EVENTS_DIR.glob("*.parquet"))])
    fantasy_targets = pd.concat([pd.read_parquet(f) for f in sorted(PROCESSED_TARGETS_DIR.glob("*.parquet"))])

    for s in season:

        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Building features for season {s}, round {round_num:02d}...")

            build_driver_features(race_results, quali_results, fantasy_targets, events, s, round_num)


# train the race finish position model
@app.command()
def train_model():
    typer.echo(f"Training quali position model...")
    train_main(QUALI_POSITION_MODEL)

    typer.echo(f"\nTraining finsih position model...")

    quali_model = load_model(QUALI_POSITION_MODEL)
    train_main(FINISH_POSITION_MODEL, quali_model, QUALI_POSITION_MODEL)


# load the trained model, predict finish positions, and print expected fantasy points for drivers and constructors
@app.command()
def predict_race(season: int = typer.Option(...), round: int = typer.Option(...)):
    quali_model = load_model(QUALI_POSITION_MODEL)
    finish_model = load_model(FINISH_POSITION_MODEL)

    predictions = run_predict(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, season, round)
    
    driver_points = compose_drivers(predictions)
    constructor_points = compose_constructor(driver_points)

    typer.echo(f"Predicting season {season}, round {round:02d}...")
    
    typer.echo("\nDrivers:")
    typer.echo(driver_points[["driver_id", "predicted_finish_position", "expected_fantasy_points"]].to_string())
    typer.echo("\nConstructors:")
    typer.echo(constructor_points.to_string())


# load predictions, compose expected points, and select the optimal team under budget constraints
@app.command()
def optimise_team(season: int = typer.Option(...), round: int = typer.Option(...), budget: float = typer.Option(BUDGET_CAP)):
    typer.echo(f"Optimising team for season {season}, round {round:02d}, budget {budget}...")

    prices = pd.read_csv(FANTASY_PRICES_DIR / f"{season}_{round:02d}.csv")
    
    quali_model = load_model(QUALI_POSITION_MODEL)
    finish_model = load_model(FINISH_POSITION_MODEL)

    predictions = run_predict(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, season, round)

    driver_points = compose_drivers(predictions)
    constructor_points = compose_constructor(driver_points)
    
    team = optimiser(driver_points, constructor_points, prices, budget)

    driver_points = driver_points.set_index("driver_id")["expected_fantasy_points"]
    constructor_points = constructor_points.set_index("constructor_id")["expected_fantasy_points"]
    driver_prices = prices.set_index("asset_id")["price"]

    total = 0.0

    typer.echo("\nDrivers:")
    for d in team["drivers"]:
        points = driver_points[d] * (2 if d == team["doubled_driver"] else 1)
        price = driver_prices[d]

        doubled_marker = " [x2]" if d == team["doubled_driver"] else ""

        typer.echo(f"  {d:<30} {points:>6.1f} points    £{price:.1f}M{doubled_marker}")
        total += points

    typer.echo("\nConstructors:")
    for c in team["constructors"]:
        points = constructor_points[c]
        price = driver_prices[c]

        typer.echo(f"  {c:<30} {points:>6.1f} points    £{price:.1f}M")
        total += points

    total_price = sum(driver_prices[d] for d in team["drivers"]) + sum(driver_prices[c] for c in team["constructors"])

    typer.echo(f"\nTotal projected points: {total:.1f}")
    typer.echo(f"Total cost: £{total_price:.1f}M / £{budget:.1f}M")


# runs walk-forward backtest comparing model, oracle, and random strategies over historical seasons, prints per-round results and saves a cumulative points plot
@app.command()
def backtest(season: list[int] = typer.Option(VAL_SEASONS), budget: float = typer.Option(BUDGET_CAP)):
    quali_model = load_model(QUALI_POSITION_MODEL)
    finish_model = load_model(FINISH_POSITION_MODEL)
    results = []

    for s in season:
        schedule = fastf1.get_event_schedule(s)
        schedule = schedule[schedule["RoundNumber"] > 0]

        for round_num in schedule["RoundNumber"]:
            prices_path = FANTASY_PRICES_DIR / f"{s}_{round_num:02d}.csv"
            features_path = PROCESSED_DRIVER_FEATURES_DIR / f"{s}_{round_num:02d}.parquet"
            targets_path = PROCESSED_TARGETS_DIR / f"{s}_{round_num:02d}.parquet"

            if not (prices_path.exists() and features_path.exists() and targets_path.exists()):
                continue

            typer.echo(f"Backtesting season {s}, round {round_num:02d}...")

            prices = pd.read_csv(prices_path)
            predictions = run_predict(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, s, round_num)

            driver_points = compose_drivers(predictions)
            constructor_points = compose_constructor(driver_points)

            model_team = optimiser(driver_points, constructor_points, prices, budget)
            model_points = get_actual_team_points(model_team, s, round_num)

            oracle_team = oracle_baseline(s, round_num, prices, budget)
            oracle_points = get_actual_team_points(oracle_team, s, round_num)

            random_points = random_baseline(s, round_num, prices, budget)

            results.append({"season": s, "round": round_num, "model": model_points, "oracle": oracle_points, "random": random_points})

    df = pd.DataFrame(results)

    typer.echo(f"\n{'Round':<8} {'Model':>8} {'Oracle':>8} {'Random':>8}")
    for _, row in df.iterrows():
        typer.echo(f"  {int(row['round']):<6} {row['model']:>8.1f} {row['oracle']:>8.1f} {row['random']:>8.1f}")

    typer.echo(f"\n{'Total':<8} {df['model'].sum():>8.1f} {df['oracle'].sum():>8.1f} {df['random'].sum():>8.1f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df[["model", "oracle", "random"]].cumsum().plot(title="Cumulative fantasy points by strategy")

    plt.xlabel("Round")
    plt.ylabel("Cumulative points")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"backtest_{'_'.join(str(s) for s in season)}.png")

    typer.echo(f"\nPlot saved to reports/")



if __name__ == "__main__": app()