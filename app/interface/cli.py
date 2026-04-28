"""CLI entry points for the F1 fantasy optimiser pipeline."""

import logging
import typer
import fastf1

from app.data.ingest import get_race_metadata, get_race_results, get_qualifying_results
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

            get_race_metadata(season, round_num)
            get_race_results(season, round_num)
            get_qualifying_results(season, round_num)


if __name__ == "__main__": app()