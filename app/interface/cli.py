"""CLI entry points for the F1 fantasy optimiser pipeline."""

import logging
import typer
import fastf1

from app.data.ingest import get_event_metadata, get_race_results, get_qualifying_results
from app.data.clean import clean_events, clean_race_results, clean_qualifying_results
from app.data.targets import compute_targets
from app.config import ALL_SEASONS

logging.getLogger("fastf1").setLevel(logging.WARNING)

app = typer.Typer(no_args_is_help=True)


# fetch raw race, qualifying, and event metadata from FastF1 for the given seasons and write to data/raw/
@app.command()
def ingest_data(seasons: list[int] = typer.Option(ALL_SEASONS)):
    for season in seasons:

        schedule = fastf1.get_event_schedule(season)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)
        
        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Ingesting season {season}, round {round_num}...")

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

            typer.echo(f"Cleaning season {season}, round {round_num}...")

            clean_events(season, round_num)
            clean_race_results(season, round_num)
            clean_qualifying_results(season, round_num)


# 
@app.command()
def build_targets(seasons: list[int] = typer.Option(ALL_SEASONS)):
    for season in seasons:

        schedule = fastf1.get_event_schedule(season)
        schedule = schedule[schedule["RoundNumber"] > 0] # exclude testing events (round 0) (for now)
        
        for round_num in schedule["RoundNumber"]:

            typer.echo(f"Computing targets for season {season}, round {round_num}...")

            compute_targets(season, round_num)


if __name__ == "__main__": app()