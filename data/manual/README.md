# data/manual/

Manually maintained inputs - files that cannot be derived from the FastF1 API and must be updated by hand after each race weekend.

## Contents

| File/Directory | Updated | Description |
|---|---|---|
| `fantasy_points/` | Each race weekend | Official F1 Fantasy points per round (2026+) |
| `starting_prices/` | Start of season | Round 1 asset prices for price computation |
| `race_overtakes/` | Each race weekend | Per-driver race overtake counts (2023+) |
| `sprint_overtakes/` | Sprint race weekends | Per-driver sprint overtake counts |

---

## fantasy_points/

One CSV per race weekend, named `{year}_{round:02d}.csv` (e.g. `2026_03.csv`).

### Schema

```
asset_id, asset_type, actual_fantasy_points
```

- `asset_id`: canonical, season-stable identifier (see ID conventions below)
- `asset_type`: `driver` or `constructor`
- `actual_fantasy_points`: official F1 Fantasy points for that round (integer)

Sort order: drivers first (alphabetical `asset_id`), then constructors (alphabetical `asset_id`). Header always present. LF line endings.

### Source

F1 Fantasy website, after each race weekend.

---

## starting_prices/

One CSV per season, named `{year}.csv` (e.g. `2026.csv`). Contains round 1 prices used by the automatic price computation pipeline.

### Schema

```
asset_id, asset_type, price
```

- `asset_id`: canonical identifier
- `asset_type`: `driver` or `constructor`
- `price`: round 1 fantasy price (float, in millions)

### Source

F1 Fantasy website, at the start of each season.

---

## race_overtakes/

One CSV per race weekend, named `{year}_{round:02d}.csv` (e.g. `2026_05.csv`).

### Schema

```
driver_id, race_overtakes
```

- `driver_id`: canonical driver identifier (see ID conventions below)
- `race_overtakes`: number of on-track overtakes scored in the race (integer)

Sort order: no required order. Header always present. LF line endings.

### Source

F1 Fantasy statistics page, scraped after each race weekend. Sprint overtake counts are **not** included here.

### Coverage

2023-2026 (ongoing). All rounds for completed seasons; current season updated race by race.

---

## sprint_overtakes/

One CSV per sprint race weekend, named `{year}_{round:02d}.csv`.

### Schema

```
driver_id, sprint_overtakes
```

Same conventions as `race_overtakes/`.

---

## ID conventions

### Drivers

`firstname_lastname` in lowercase snake_case, ASCII-folded (no diacritics), full legal name:

- `lando_norris`, `sergio_perez`, `andrea_kimi_antonelli`, `zhou_guanyu`, `nyck_de_vries`

A driver keeps the same `asset_id` across team changes.

### Constructors

Short canonical team name in snake_case, following current season branding:

| Season | `asset_id` |
|-------:|------------|
| 2023 | `alphatauri` |
| 2024 | `rb` |
| 2025+ | `racing_bulls` |
| 2023 | `alfa_romeo` |
| 2024-25 | `kick_sauber` |
| 2026+ | `audi` |
| 2026+ | `cadillac` |

`red_bull`, `ferrari`, `mercedes`, `mclaren`, `aston_martin`, `williams`, `alpine`, and `haas` are stable across all seasons.
