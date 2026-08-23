# F1 Fantasy Team Optimiser
**Live at:** [pitwall.georgeputney.com](https://pitwall.georgeputney.com)

ML pipeline and web app for F1 fantasy team selection. The pipeline ingests historical race, qualifying, practice, sprint, and pit stop data via FastF1, engineers rolling features across multiple timeframes, predicts finish positions and qualifying positions with XGBoost, then selects the optimal team under budget and roster constraints using integer linear programming. Probabilistic models for fastest lap, Driver of the Day, and overtakes feed into expected fantasy point calculations, and a calibrated Monte Carlo simulation supplies each asset's likely points range. A FastAPI service (`api/`) exposes the results to a React frontend (`web/`), [Pitwall](https://pitwall.georgeputney.com).

## Pipeline

```
FastF1 API -> ingest -> clean -> targets -> prices -> features -> train -> predict -> compose -> optimise
```

## Structure

```
app/                                 # ML pipeline: ingest, clean, engineer features, train, predict, optimise
├── config.py                       # seasons, paths, budget cap, all constants
├── data/
│   ├── ingest.py                   # fetch race, quali, practice, sprint, pit stop data
│   ├── clean.py                    # validate and normalise raw parquets
│   ├── targets.py                  # compute fantasy point targets from results
│   ├── prices.py                   # compute rolling price-per-million values
│   ├── scoring_rules.py            # official F1 fantasy scoring rules (2026+)
│   ├── schemas.py                  # pandera schemas for data validation
│   ├── overtakes.py                # expected overtakes per driver
│   └── dotd.py                     # Driver of the Day probability model
├── features/
│   ├── build_historic_features.py  # rolling driver and constructor features
│   ├── build_practice_features.py  # FP2/FP3 pace, gaps, sector deltas
│   └── build_circuit_features.py   # circuit-level features (overtake index, DNF rate, etc.)
├── models/
│   ├── configs.py                  # XGBoost model definitions
│   ├── train.py                    # training loop, walk-forward season artifacts
│   ├── predict.py                  # inference from trained models
│   ├── compose.py                  # combine predictions into expected fantasy points
│   ├── monte_carlo.py              # calibrated Monte Carlo race-outcome simulation (display only)
│   ├── evaluation.py               # model evaluation metrics
│   └── backtest.py                 # walk-forward backtesting with baselines
├── optimiser/
│   ├── optimiser.py                # ILP team selection under constraints
│   ├── budget_range.py             # theoretical max/min reachable budget by a given round
│   └── state.py                    # persist budget, prices, and free transfers across rounds
└── interface/
    └── cli.py                      # Typer CLI entry points

api/                                 # FastAPI service backing the frontend (thin wrapper over app/)
├── main.py                         # routes; in production also serves web/dist as static files
├── common.py                       # shared payload helpers, cached prediction/MC snapshot loading
├── ladder.py                       # driver/constructor ranking section
├── team.py                         # squad controls, projected points, lineup, suggested transfers
├── breakdown.py                    # per-driver points breakdown, browsable by round
├── value.py                        # budget waterfall and value-map scatter data
└── track_record.py                 # model vs oracle performance across the season

web/                                 # React + TypeScript + Vite frontend (Pitwall)
```

## Requirements

Python 3.10+

```bash
# API only (what the web service installs)
pip install -e ".[api]"

# full data + training + report pipeline
pip install -e ".[pipeline]"
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

# generate prediction reports for the web app
python -m app.interface.cli generate-reports --season 2026 --round 7

# select optimal team for a race
python -m app.interface.cli optimise-team --season 2026 --round 7
```

**Backtest:**

```bash
python -m app.interface.cli backtest --season 2025
```

Runs walk-forward evaluation against oracle, lagged, and mean-prior baselines. All strategies are transfer-constrained and cumulative results are saved to `reports/`.

**Backfill prediction reports:**

```bash
python -m app.interface.cli backfill-predictions --from-season 2026
```

**Web app (local dev):**

```bash
# API - reads reports/predictions/*.json and data/processed/prices/*.parquet
uvicorn api.main:app --reload --port 8000

# frontend - proxies /api to the port above
cd web && npm install && npm run dev
```

## CI/CD

A GitHub Actions workflow (`checker.yml`) runs on a cron schedule every 20 minutes on Fridays, Saturdays, and Mondays. It checks whether a session has become available via FastF1 and triggers the pipeline (`pipeline.yml`) at three points during a race weekend:

1. **Post-FP2** -- preliminary predictions with early practice data (conventional weekends only)
2. **Post-FP3** -- full predictions with all practice data (or post-Sprint Qualifying for sprint weekends)
3. **Post-race** -- preliminary predictions for the next round using historical features only

The pipeline re-ingests the previous round, processes the current round, builds features, generates prediction reports, and commits the results back to the repo. After a race, it also runs the backtest. The workflow can be triggered manually via `workflow_dispatch`.

## Deployment

`Dockerfile` builds the React frontend, then bundles it into a single Python image that serves both the built static files and the FastAPI routes from one `uvicorn` process (see the `WEB_DIST` mount in `api/main.py`). Deployed as one Render web service - no separate frontend host or build step needed there. New predictions land via the CI pipeline committing straight to the repo, which triggers a redeploy.

## Data

Race, qualifying, practice, sprint, and pit stop data is fetched from [FastF1](https://github.com/theOehrly/Fast-F1). Fantasy starting prices are manually maintained CSVs in `data/manual/starting_prices/`. Rolling prices are computed from targets and stored in `data/processed/prices/`.

## Stack

Python, pandas, XGBoost, PuLP (CBC), FastF1, Typer, pandera, FastAPI, React, TypeScript, Vite, Docker