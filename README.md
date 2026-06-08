# F1 Fantasy Team Optimiser
**Live at:** [f1fantasy.georgeputney.com](https://f1fantasy.georgeputney.com)

ML pipeline for F1 fantasy team selection. Ingests historical race, qualifying, practice, sprint, and pit stop data via FastF1, engineers rolling features across multiple timeframes, predicts finish positions and qualifying positions with XGBoost, then selects the optimal team under budget and roster constraints using integer linear programming. Probabilistic models for fastest lap, Driver of the Day, and overtakes feed into expected fantasy point calculations.

## Pipeline

```
FastF1 API -> ingest -> clean -> targets -> prices -> features -> train -> predict -> compose -> optimise
```

## Structure

```
app/
  config.py                         # seasons, paths, budget cap, all constants
  dashboard.py                      # Streamlit dashboard for viewing predictions
  backtest.py                       # walk-forward backtesting with baselines
  data/
    ingest.py                       # fetch race, quali, practice, sprint, pit stop data
    clean.py                        # validate and normalise raw parquets
    targets.py                      # compute fantasy point targets from results
    prices.py                       # compute rolling price-per-million values
    scoring_rules.py                # official F1 fantasy scoring breakdown
    schemas.py                      # pandera schemas for data validation
    overtakes.py                    # expected overtakes per driver
    dotd.py                         # Driver of the Day probability model
  features/
    build_historic_features.py      # rolling driver and constructor features
    build_practice_features.py      # FP1/FP3 session features
    build_circuit_features.py       # circuit-level features
  models/
    configs.py                      # XGBoost model definitions
    train.py                        # training loop, walk-forward season artifacts
    predict.py                      # inference from trained models
    compose.py                      # combine predictions into expected fantasy points
    evaluation.py                   # model evaluation metrics
  optimiser/
    optimiser.py                    # ILP team selection under constraints
    state.py                        # persist budget, prices, and free transfers across rounds
  interface/
    cli.py                          # Typer CLI entry points
```

## Requirements

Python 3.10+

```bash
pip install -e .
```

## Usage

**Run the full pipeline:**

```bash
# fetch raw data from FastF1
python -m app.interface.cli ingest-data --season 2026

# clean and validate
python -m app.interface.cli clean-data --season 2026

# compute fantasy point targets
python -m app.interface.cli build-targets --season 2026

# compute rolling prices
python -m app.interface.cli build-prices --season 2026

# build driver, constructor, practice, and circuit features
python -m app.interface.cli build-features --season 2026

# train finish position and qualifying position models
python -m app.interface.cli train-model

# generate prediction reports for the dashboard
python -m app.interface.cli generate-reports --season 2026 --round 7

# select optimal team for a race
python -m app.interface.cli optimise-team --season 2026 --round 7
```

**Backtest:**

```bash
python -m app.interface.cli backtest --season 2025
```

Runs walk-forward evaluation against oracle, random, lagged, and mean-prior baselines.

**Backfill prediction reports:**

```bash
python -m app.interface.cli backfill-predictions --from-season 2026
```

**Dashboard:**

```bash
streamlit run app/dashboard.py
```

## CI/CD

A GitHub Actions pipeline triggers on pushes to `main` that touch `data/interim/**`. It runs `build-targets -> build-prices -> build-features -> generate-reports` and commits the results back to the repo. Can also be triggered manually via `workflow_dispatch` with a season and round number.

## Data

Race, qualifying, practice, sprint, and pit stop data is fetched from [FastF1](https://github.com/theOehrly/Fast-F1). Fantasy prices are manually maintained CSVs in `data/raw/fantasy_prices/` and must be updated before each race weekend.

## Stack

Python, pandas, XGBoost, PuLP (CBC), FastF1, Typer, pandera, Streamlit, Plotly
