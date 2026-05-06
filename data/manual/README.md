# data/manual/

Manually maintained inputs - files that cannot be derived from the FastF1 API and must be updated by hand after each race weekend.

## Contents

| File/Directory | Updated | Description |
|---|---|---|
| `fantasy_prices/` | Each race weekend | F1 Fantasy asset prices per round |
| `dnf_classification_patch.csv` | Each race weekend (if a crash DNF occurred) | Crash vs mechanical DNF overrides for 2023+ |

---

## fantasy_prices/

One CSV per race weekend, named `{year}_{round:02d}.csv` (e.g. `2025_03.csv`).

### Schema

```
race_id, asset_id, asset_type, price
```

- `race_id`: `{season}_{round:02d}`, e.g. `2025_03`
- `asset_id`: canonical, season-stable identifier (see ID conventions below)
- `asset_type`: `driver` or `constructor`
- `price`: fantasy game price for that race (float, in millions)

Sort order: drivers first (alphabetical `asset_id`), then constructors (alphabetical `asset_id`). Header always present. LF line endings.

### Source

F1 Fantasy price PDFs, available on the F1 Fantasy website each race weekend.

---

## dnf_classification_patch.csv

FastF1 returns a generic `"Retired"` status for all DNFs in 2023+, making it impossible to distinguish crashes from mechanical failures automatically. This file lists confirmed crash DNFs so `clean.py` can set `crash_dnf_flag` correctly. All other DNFs default to mechanical.

### Schema

```
race_id, driver_id, dnf_type
```

- `race_id`: `{season}_{round:02d}`
- `driver_id`: canonical driver identifier (see ID conventions below)
- `dnf_type`: always `crash` (only crash entries are listed)

Lines beginning with `#` are comments and are ignored by the pipeline.

### Source

Wikipedia race result tables.

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
| 2024–25 | `kick_sauber` |
| 2026+ | `audi` |
| 2026+ | `cadillac` |

`red_bull`, `ferrari`, `mercedes`, `mclaren`, `aston_martin`, `williams`, `alpine`, and `haas` are stable across all seasons.
