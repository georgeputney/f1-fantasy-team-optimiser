"""
Scrape race/sprint overtakes and DOTD from f1fantasytools.com/statistics.

The site is a Next.js SPA that embeds all data in the initial page load.
Uses Playwright to render the page, select the stat type from a combobox
dropdown, and extract the tables from the DOM. Writes per-round CSVs to
data/raw/race_overtakes/, data/raw/sprint_overtakes/, and a per-season
DOTD CSV to data/raw/race_dotd/.

Usage:
    python scraping/scrape_overtakes_and_dotd.py                    # scrape current season (2026)
    python scraping/scrape_overtakes_and_dotd.py --season 2025      # scrape a specific season
    python scraping/scrape_overtakes_and_dotd.py --force             # overwrite existing files

Requires: playwright (pip install playwright && playwright install chromium)
"""

import argparse
import time

import fastf1
import pandas as pd
from playwright.sync_api import sync_playwright

from app.config import (
    FASTF1_CACHE_DIR, RAW_RACE_OVERTAKES_DIR, RAW_SPRINT_OVERTAKES_DIR, RAW_DOTD_DIR,
)

fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

URL = "https://f1fantasytools.com/statistics"

# 3-letter abbreviation -> driver_id mapping, built from FastF1 at runtime
_ABBREV_CACHE: dict[int, dict[str, str]] = {}

DRIVER_ID_NORMALISATION = {
    "kimi_antonelli": "andrea_kimi_antonelli",
}

# fallback for mid-season replacements not captured by the first/mid/last round check
ABBREV_FALLBACK = {
    "LAW": "liam_lawson",
    "DEV": "nyck_de_vries",
    "RIC": "daniel_ricciardo",
    "COL": "franco_colapinto",
    "SAR": "logan_sargeant",
    "DOO": "jack_doohan",
    "LIN": "arvid_lindblad",
    "BEA": "oliver_bearman",
}


def _build_abbrev_map(season: int) -> dict[str, str]:
    """build a mapping of 3-letter abbreviations to driver_ids for a season.

    Loads multiple rounds to catch mid-season driver changes (e.g. Colapinto
    replacing Sargeant in 2025).
    """
    if season in _ABBREV_CACHE:
        return _ABBREV_CACHE[season]

    schedule = fastf1.get_event_schedule(season, include_testing=False)
    # check first, middle, and last rounds to catch mid-season driver changes
    round_numbers = [int(r) for r in schedule["RoundNumber"]]
    rounds_to_check = [round_numbers[0],
                       round_numbers[len(round_numbers) // 2],
                       round_numbers[-1]]

    mapping = {}
    for round_num in rounds_to_check:
        try:
            session = fastf1.get_session(season, round_num, "R")
            session.load(telemetry=False, weather=False, messages=False)
            if session.results.empty:
                continue
        except Exception:
            continue

        for _, row in session.results.iterrows():
            abbrev = row["Abbreviation"].upper()
            if abbrev in mapping:
                continue
            first = row["FirstName"].lower().replace(" ", "_")
            last = row["LastName"].lower().replace(" ", "_")
            driver_id = f"{first}_{last}"
            driver_id = DRIVER_ID_NORMALISATION.get(driver_id, driver_id)
            mapping[abbrev] = driver_id

    # add fallback abbreviations for mid-season replacements
    for abbrev, driver_id in ABBREV_FALLBACK.items():
        if abbrev not in mapping:
            mapping[abbrev] = driver_id

    _ABBREV_CACHE[season] = mapping
    return mapping


def _select_stat_type(page, stat_name: str):
    """open the stat-type combobox and select an option by text."""
    combo = page.locator("button[role='combobox']").nth(1)
    combo.click()
    time.sleep(0.5)

    option = page.locator("[role='option']", has_text=stat_name)
    option.click()
    time.sleep(0.5)


def _select_season(page, season: int):
    """open the season combobox and select a year."""
    combo = page.locator("button[role='combobox']").nth(0)
    current = combo.inner_text().strip()
    if current == str(season):
        return

    combo.click()
    time.sleep(0.5)
    option = page.locator("[role='option']", has_text=str(season))
    option.click()
    time.sleep(0.5)


def _extract_driver_table(page) -> list[dict]:
    """extract the first (driver) table from the current page view."""
    tables = page.query_selector_all("table")
    if not tables:
        return []

    rows = tables[0].query_selector_all("tr")
    if len(rows) < 2:
        return []

    headers = []
    for th in rows[0].query_selector_all("th, td"):
        headers.append(th.inner_text().strip())

    data = []
    for row in rows[1:]:
        cells = row.query_selector_all("th, td")
        values = [c.inner_text().strip() for c in cells]
        if not values or values[0] == "AVG":
            continue
        data.append(dict(zip(headers, values)))

    return data


def _save_overtakes(
    data: list[dict],
    season: int,
    abbrev_map: dict[str, str],
    kind: str,
    force: bool,
) -> int:
    """save per-round CSV files from the extracted table data. returns rounds written."""
    if kind == "race":
        out_dir = RAW_RACE_OVERTAKES_DIR
        col_name = "race_overtakes"
    else:
        out_dir = RAW_SPRINT_OVERTAKES_DIR
        col_name = "sprint_overtakes"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not data:
        return 0

    sample_keys = list(data[0].keys())
    round_cols = sorted(
        [k for k in sample_keys if k.startswith("R") and k[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )

    written = 0
    for rcol in round_cols:
        round_num = int(rcol[1:])
        out_path = out_dir / f"{season}_{round_num:02d}.csv"

        if out_path.exists() and not force:
            continue

        # skip future rounds with no data
        values = [row.get(rcol, "").strip() for row in data]
        if all(v == "" for v in values):
            continue

        rows = []
        dr_key = sample_keys[0]
        for row in data:
            abbrev = row[dr_key].strip().upper()
            overtakes_str = row.get(rcol, "").strip()
            if not overtakes_str:
                continue

            driver_id = abbrev_map.get(abbrev)
            if driver_id is None:
                print(f"    warning: unknown abbreviation '{abbrev}', skipping")
                continue

            try:
                overtakes = int(overtakes_str)
            except ValueError:
                continue

            rows.append({"driver_id": driver_id, col_name: overtakes})

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)
            print(f"    wrote {out_path.name}")
            written += 1

    return written


def _save_dotd(
    data: list[dict],
    season: int,
    abbrev_map: dict[str, str],
    force: bool,
) -> int:
    """save DOTD winners from the checkmark table to a single per-season CSV. returns rounds written."""
    RAW_DOTD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DOTD_DIR / f"{season}.csv"

    if out_path.exists() and not force:
        return 0

    if not data:
        return 0

    sample_keys = list(data[0].keys())
    round_cols = sorted(
        [k for k in sample_keys if k.startswith("R") and k[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )

    rows = []
    dr_key = sample_keys[0]
    for rcol in round_cols:
        round_num = int(rcol[1:])
        for row in data:
            cell = row.get(rcol, "").strip()
            # checkmark indicates DOTD winner
            if "\u2713" in cell or "✓" in cell:
                abbrev = row[dr_key].strip().upper()
                driver_id = abbrev_map.get(abbrev)
                if driver_id is None:
                    print(f"    warning: unknown abbreviation '{abbrev}' for DOTD, skipping")
                    continue
                rows.append({
                    "race_id": f"{season}_{round_num:02d}",
                    "driver_id": driver_id,
                })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"    wrote {out_path.name} ({len(rows)} rounds)")

    return len(rows)


def scrape_season(season: int, force: bool = False) -> dict[str, int]:
    """scrape race/sprint overtakes and DOTD for a full season."""
    abbrev_map = _build_abbrev_map(season)
    counts = {"race": 0, "sprint": 0, "dotd": 0}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"loading {URL} ...")
        page.goto(URL, wait_until="networkidle")
        time.sleep(2)

        _select_season(page, season)

        # --- race overtakes ---
        print(f"  selecting 'Race Overtakes'...")
        _select_stat_type(page, "Race Overtakes")
        time.sleep(0.5)

        race_data = _extract_driver_table(page)
        if race_data:
            counts["race"] = _save_overtakes(race_data, season, abbrev_map, "race", force)

        # --- sprint overtakes ---
        print(f"  selecting 'Sprint Overtakes'...")
        _select_stat_type(page, "Sprint Overtakes")
        time.sleep(0.5)

        sprint_data = _extract_driver_table(page)
        if sprint_data:
            counts["sprint"] = _save_overtakes(sprint_data, season, abbrev_map, "sprint", force)

        # --- driver of the day ---
        print(f"  selecting 'Race Driver of the Day'...")
        _select_stat_type(page, "Race Driver of the Day")
        time.sleep(0.5)

        dotd_data = _extract_driver_table(page)
        if dotd_data:
            counts["dotd"] = _save_dotd(dotd_data, season, abbrev_map, force)

        browser.close()

    return counts


def main():
    parser = argparse.ArgumentParser(description="Scrape overtakes from f1fantasytools.com")
    parser.add_argument("--season", type=int, default=2026, help="season to scrape (default: 2026)")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    counts = scrape_season(args.season, force=args.force)
    print(f"\ndone. wrote {counts['race']} race round(s), {counts['sprint']} sprint round(s), {counts['dotd']} DOTD round(s).")


if __name__ == "__main__":
    main()
