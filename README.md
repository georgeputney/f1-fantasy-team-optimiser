# F1 Fantasy Team Optimiser

ML pipeline for F1 fantasy team selection. Ingests historical race data, engineers rolling features, predicts finish positions with XGBoost, and selects the optimal team under budget and roster constraints using integer linear programming.

## Pipeline

```
FastF1 API → ingest → clean → targets → features → train → predict → optimise
```

## CLI Commands

```bash
# fetch raw data from FastF1
python -m app.interface.cli ingest-data --season 2023 --season 2024 --season 2025

# clean and normalise
python -m app.interface.cli clean-data --season 2023 --season 2024 --season 2025

# compute fantasy point targets
python -m app.interface.cli build-targets --season 2023 --season 2024 --season 2025

# build driver and constructor features
python -m app.interface.cli build-features --season 2023 --season 2024 --season 2025

# train the finish position model
python -m app.interface.cli train-model

# preview predicted fantasy points for a race
python -m app.interface.cli predict-race --season 2025 --round 6

# select optimal team for a race
python -m app.interface.cli optimise-team --season 2025 --round 6 --budget 100

# walk-forward backtest over one or more seasons
python -m app.interface.cli backtest --season 2024 --season 2025
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
```

## Data

Race and qualifying data is fetched from [FastF1](https://github.com/theOehrly/Fast-F1). Fantasy prices are manually maintained CSVs in `data/raw/fantasy_prices/` — these must be updated before each race weekend.

## Stack

Python · pandas · XGBoost · PuLP (CBC) · FastF1 · Typer · pandera
