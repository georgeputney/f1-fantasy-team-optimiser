"""CLI entry points for the F1 fantasy optimiser pipeline."""

import logging
import typer
import fastf1
import pandas as pd

from app.data.ingest import get_event_metadata, get_race_results, get_qualifying_results
from app.data.clean import clean_events, clean_race_results, clean_qualifying_results
from app.data.targets import compute_targets

from app.features.build_driver_features import build_driver_features

from app.models.configs import FINISH_POSITION_MODEL
from app.models.train import main as train_main
from app.models.predict import load_model, predict as run_predict
from app.models.compose import compose_drivers, compose_constructor

from app.optimiser.team_optimiser import optimiser

from app.config import ALL_SEASONS, BUDGET_CAP, FANTASY_PRICES_DIR, INTERIM_EVENTS_DIR, INTERIM_QUALI_DIR, INTERIM_RACES_DIR, PROCESSED_TARGETS_DIR

logging.getLogger("fastf1").setLevel(logging.WARNING)

app = typer.Typer(no_args_is_help=True)


# fetch raw race, qualifying, and event metadata from FastF1 for the given seasons and write to data/raw/
@app.command()
def ingest_data(seasons: list[int] = typer.Option(ALL_SEASONS)):
    for season in seasons:

        schedule = fastf1.get_event_schedule(season)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)
        
        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Ingesting season {season}, round {round_num:02d}...")

            get_event_metadata(season, round_num)
            get_race_results(season, round_num)
            get_qualifying_results(season, round_num)


# clean raw parquet files for the given seasons and write validated tables to data/interim/
@app.command()
def clean_data(seasons: list[int] = typer.Option(ALL_SEASONS)):
    for season in seasons:

        schedule = fastf1.get_event_schedule(season)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)
        
        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Cleaning season {season}, round {round_num:02d}...")

            clean_events(season, round_num)
            clean_race_results(season, round_num)
            clean_qualifying_results(season, round_num)


# compute actual fantasy points from cleaned results and write to data/processed/targets/
@app.command()
def build_targets(seasons: list[int] = typer.Option(ALL_SEASONS)):
    for season in seasons:

        schedule = fastf1.get_event_schedule(season)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)
        
        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Building targets for season {season}, round {round_num:02d}...")

            compute_targets(season, round_num)


# build driver and constructor features for the given seasons and write to data/processed/driver_features/
@app.command()
def build_features(seasons: list[int] = typer.Option(ALL_SEASONS)):

    race_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_RACES_DIR.glob("*.parquet"))])
    quali_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_QUALI_DIR.glob("*.parquet"))])
    events = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_EVENTS_DIR.glob("*.parquet"))])
    fantasy_targets = pd.concat([pd.read_parquet(f) for f in sorted(PROCESSED_TARGETS_DIR.glob("*.parquet"))])

    for season in seasons:

        schedule = fastf1.get_event_schedule(season)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)

        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Building features for season {season}, round {round_num:02d}...")

            build_driver_features(race_results, quali_results, fantasy_targets, events, season, round_num)


# train the race finish position model
@app.command()
def train_model():
    typer.echo(f"Training model...")

    train_main(FINISH_POSITION_MODEL)


# load the trained model, predict finish positions, and print expected fantasy points for drivers and constructors
@app.command()
def predict_race(season: int = typer.Option(...), round: int = typer.Option(...)):
    model = load_model(FINISH_POSITION_MODEL)
    predictions = run_predict(model, FINISH_POSITION_MODEL, season, round)
    
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
    prices = pd.read_csv(FANTASY_PRICES_DIR / f"{season}_{round:02d}.csv")
    
    model = load_model(FINISH_POSITION_MODEL)
    predictions = run_predict(model, FINISH_POSITION_MODEL, season, round)

    driver_points = compose_drivers(predictions)
    constructor_points = compose_constructor(driver_points)
    
    team = optimiser(driver_points, constructor_points, prices, budget)

    typer.echo(f"Optimising team for season {season}, round {round:02d}, budget {budget}...")
    
    typer.echo(f"\nDrivers: {team['drivers']}")
    typer.echo(f"Constructors: {team['constructors']}")
    typer.echo(f"Doubled: {team['doubled_driver']}")



if __name__ == "__main__": app()