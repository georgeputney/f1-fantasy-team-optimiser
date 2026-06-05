"""
Scrape pit stop stationary times from the DHL Fastest Pit Stop Award API.

Uses the inmotion.dhl JSON API to fetch per-race pit stop data with official
stationary (box) times measured to 0.01s precision. Writes one parquet file
per race to data/raw/pitstops/, matching the schema used by the rest of the pipeline.

Usage:
    python scraping/scrape_dhl_pitstops.py              # scrape all available races
    python scraping/scrape_dhl_pitstops.py --season 2026 # scrape a single season
    python scraping/scrape_dhl_pitstops.py --force       # overwrite existing files

To add a new season, use --discover-season to find the DHL element IDs:
    python scraping/scrape_dhl_pitstops.py --discover-season 2024
"""

import argparse
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import fastf1
import pandas as pd
import requests

from app.config import FASTF1_CACHE_DIR

fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

# DHL API base
API_BASE = "https://inmotion.dhl/api/f1-award-element-data"

# 2023+: dedicated archive pages with an events-list endpoint and a race-data endpoint
SEASON_CONFIG = {
    2023: {"events_element_id": 6284, "race_element_id": 6282},
    2024: {"events_element_id": 6276, "race_element_id": 6273},
    2025: {"events_element_id": 6367, "race_element_id": 6365},
    2026: {"events_element_id": 7375, "race_element_id": 7373},
}

# 2018-2022: overview page endpoints (no events-list, need to scan event IDs)
# race_element_id: the overview endpoint for per-race data
# scan_range: range of DHL event IDs to try for this season
OVERVIEW_CONFIG = {
    2018: {"race_element_id": 7613, "scan_range": (90, 140)},
    2019: {"race_element_id": 7609, "scan_range": (275, 320)},
    2020: {"race_element_id": 7604, "scan_range": (490, 530)},
    2021: {"race_element_id": 7601, "scan_range": (650, 700)},
    2022: {"race_element_id": 7598, "scan_range": (730, 770)},
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pitstops"


# parse an HTML table into a list of row-lists
class _TableParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell = ""
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data


# parse a DHL HTML table into a DataFrame with standard columns
def _parse_html_table(html: str) -> pd.DataFrame | None:
    if not html:
        return None

    parser = _TableParser()
    parser.feed(html)

    if len(parser.rows) < 2:
        return None

    headers = [h.lower().replace(".", "").strip() for h in parser.rows[0]]
    rows = parser.rows[1:]
    df = pd.DataFrame(rows, columns=headers)

    expected = {"pos", "team", "driver", "time (sec)", "lap", "points"}
    if not expected.issubset(set(df.columns)):
        return None

    df = df.rename(columns={"time (sec)": "time_s", "pos": "position"})
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["lap"] = pd.to_numeric(df["lap"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")

    return df


# extract the GP name from the DHL header HTML (e.g. 'FORMULA 1 GRAND PRIX DE MONACO 2018')
def _extract_gp_name(header_html: str) -> str | None:
    match = re.search(r"<h[12][^>]*>\s*(.*?)\s*</h[12]>", header_html, re.DOTALL)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


# build a mapping from lowercase location/country keywords to round numbers using FastF1
def _build_round_map(season: int) -> dict[str, int]:
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    round_map = {}
    for _, row in schedule.iterrows():
        rnd = int(row["RoundNumber"])
        if rnd == 0:
            continue
        # index by multiple keys: country, location, and event name fragments
        for field in ["Country", "Location", "EventName"]:
            val = str(row[field]).lower().strip()
            if val and val != "nan":
                round_map[val] = rnd
    return round_map


# DHL GP names use local languages; map non-English fragments to FastF1 country/location names
_GP_ALIASES = {
    "österreich": "austria", "osterreich": "austria", "steiermark": "austria",
    "magyar": "hungary", "hungaroring": "hungary",
    "italia": "italy", "d'italia": "italy", "monza": "italy",
    "méxico": "mexico", "mexico": "mexico",
    "brasil": "brazil", "são paulo": "brazil", "interlagos": "brazil",
    "españa": "spain", "espana": "spain", "barcelona": "spain",
    "belgique": "belgium", "spa": "belgium",
    "deutschland": "germany", "hockenheim": "germany", "nürburgring": "germany",
    "türkei": "turkey", "istanbul": "turkey",
    "emirates": "united arab emirates", "abu dhabi": "united arab emirates",
    "bahrain": "bahrain", "sakhir": "bahrain",
    "monaco": "monaco",
    "canada": "canada", "montréal": "canada", "montreal": "canada",
    "france": "france",
    "british": "great britain", "silverstone": "great britain",
    "singapore": "singapore",
    "japan": "japan", "suzuka": "japan",
    "united states": "united states", "austin": "united states",
    "russia": "russia", "sochi": "russia",
    "china": "china", "shanghai": "china",
    "australia": "australia", "melbourne": "australia",
    "azerbaijan": "azerbaijan", "baku": "azerbaijan",
    "portugal": "portugal", "portimão": "portugal", "portimao": "portugal",
    "saudi": "saudi arabia", "jeddah": "saudi arabia",
    "netherlands": "netherlands", "zandvoort": "netherlands",
    "qatar": "qatar", "lusail": "qatar",
    "miami": "miami",
    "las vegas": "las vegas",
    "imola": "emilia romagna", "emilia romagna": "emilia romagna",
    "made in italy": "emilia romagna",
}


# fuzzy match a DHL GP name to a round number.
# checks aliases first (longest match), then direct FastF1 keys (longest match).
# aliases take priority because DHL names use local languages and contain
# misleading substrings (e.g. 'Made in Italy' for the Emilia Romagna GP).
def _match_round(gp_name: str, round_map: dict[str, int]) -> int | None:
    gp_lower = gp_name.lower()

    # try aliases first (longest match), since they handle known edge cases
    for alias, canonical in sorted(_GP_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in gp_lower:
            for key, rnd in round_map.items():
                if canonical in key or key in canonical:
                    return rnd

    # fall back to direct match against FastF1 keys, longest first
    for key, rnd in sorted(round_map.items(), key=lambda x: -len(x[0])):
        if key in gp_lower:
            return rnd

    return None


# --- 2023+ seasons: events list available ---

# fetch the list of events for a 2023+ season from the DHL API
def get_events(season: int) -> list[dict]:
    cfg = SEASON_CONFIG[season]
    r = requests.get(f"{API_BASE}/{cfg['events_element_id']}", timeout=15)
    r.raise_for_status()
    return r.json()["data"]["chart"]["events"]


# fetch all pit stop times for a single race. parses htmlList table (all stops, not just top 10).
def get_race_pitstops(element_id: int, event_id: int) -> pd.DataFrame | None:
    r = requests.get(f"{API_BASE}/{element_id}?event={event_id}", timeout=15)
    r.raise_for_status()
    data = r.json()
    return _parse_html_table(data.get("htmlList", {}).get("table", ""))


# scrape a 2023+ season using the events-list endpoint
def scrape_season_modern(season: int, force: bool = False) -> int:
    cfg = SEASON_CONFIG[season]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events = get_events(season)
    print(f"\n{season}: {len(events)} events on calendar")

    scraped = 0
    for round_num, event in enumerate(events, start=1):
        out_path = OUTPUT_DIR / f"{season}_{round_num:02d}.parquet"

        if out_path.exists() and not force:
            print(f"  R{round_num:02d} {event['short_title']}: already exists, skipping")
            scraped += 1
            continue

        df = get_race_pitstops(cfg["race_element_id"], event["id"])
        if df is None or len(df) == 0:
            print(f"  R{round_num:02d} {event['short_title']}: no data (future race?)")
            continue

        out = pd.DataFrame({
            "season": season,
            "round": round_num,
            "driver": df["driver"],
            "team": df["team"],
            "lap": df["lap"],
            "stationary_s": df["time_s"],
            "dhl_position": df["position"],
            "dhl_points": df["points"],
        })

        out.to_parquet(out_path, index=False)
        print(f"  R{round_num:02d} {event['short_title']}: {len(out)} stops")
        scraped += 1
        time.sleep(0.5)

    return scraped


# 2018-2022 seasons: overview page, scan for events 
# scrape a 2018-2022 season by scanning event IDs on the overview endpoint
def scrape_season_overview(season: int, force: bool = False) -> int:
    cfg = OVERVIEW_CONFIG[season]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    element_id = cfg["race_element_id"]
    lo, hi = cfg["scan_range"]

    # build round mapping from FastF1
    print(f"\n{season}: loading FastF1 schedule for round mapping...")
    round_map = _build_round_map(season)

    # scan for valid events
    print(f"{season}: scanning event IDs {lo}-{hi}...")
    found_events = []
    for ev_id in range(lo, hi):
        r = requests.get(f"{API_BASE}/{element_id}?event={ev_id}", timeout=15)
        resp = r.json()
        data = resp["data"]
        if not data["chart"] or not data.get("event_id"):
            continue
        if int(data["event_id"]) != ev_id:
            continue  # API returned a different event (default fallback)
        header = resp.get("htmlList", {}).get("header", "")
        gp_name = _extract_gp_name(header)
        if not gp_name:
            continue
        round_num = _match_round(gp_name, round_map)
        found_events.append({"event_id": ev_id, "gp_name": gp_name, "round": round_num})

    # deduplicate: if DHL returns the same GP for multiple event IDs, keep the first
    seen_rounds = set()
    deduped = []
    for event in found_events:
        if event["round"] is not None and event["round"] in seen_rounds:
            continue
        if event["round"] is not None:
            seen_rounds.add(event["round"])
        deduped.append(event)

    print(f"{season}: found {len(found_events)} events ({len(deduped)} unique rounds)")

    scraped = 0
    for event in deduped:
        round_num = event["round"]
        if round_num is None:
            print(f"  ?? {event['gp_name']}: could not match to round number, skipping")
            continue

        out_path = OUTPUT_DIR / f"{season}_{round_num:02d}.parquet"
        if out_path.exists() and not force:
            print(f"  R{round_num:02d} {event['gp_name']}: already exists, skipping")
            scraped += 1
            continue

        df = get_race_pitstops(element_id, event["event_id"])
        if df is None or len(df) == 0:
            print(f"  R{round_num:02d} {event['gp_name']}: no data")
            continue

        out = pd.DataFrame({
            "season": season,
            "round": round_num,
            "driver": df["driver"],
            "team": df["team"],
            "lap": df["lap"],
            "stationary_s": df["time_s"],
            "dhl_position": df["position"],
            "dhl_points": df["points"],
        })

        out.to_parquet(out_path, index=False)
        print(f"  R{round_num:02d} {event['gp_name']}: {len(out)} stops")
        scraped += 1
        time.sleep(0.3)

    return scraped


# scrape a season using the appropriate method
def scrape_season(season: int, force: bool = False) -> int:
    if season in SEASON_CONFIG:
        return scrape_season_modern(season, force)
    elif season in OVERVIEW_CONFIG:
        return scrape_season_overview(season, force)
    else:
        print(f"season {season} not configured.")
        return 0


# use playwright to find the DHL element IDs for a given season.
# loads the DHL archive page for that season and captures the API calls it makes.
def discover_season(season: int):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is required for --discover-season. install with: pip install playwright")
        return

    url = f"https://inmotion.dhl/en/formula-1/fastest-pit-stop-award-{season}"
    print(f"loading {url} ...")

    api_calls = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def on_response(response):
            if "f1-award-element-data" in response.url:
                api_calls.append(response.url)

        page.on("response", on_response)
        page.goto(url, wait_until="networkidle")
        browser.close()

    if not api_calls:
        print(f"no API calls found. the page might not exist for {season}.")
        return

    print(f"\nAPI endpoints found for {season}:")
    for url in api_calls:
        element_id = url.split("/")[-1].split("?")[0]
        print(f"  element_id={element_id}  ({url})")

    print(f"\ntest each endpoint to identify which is events vs race data:")
    for url in api_calls:
        element_id = url.split("/")[-1].split("?")[0]
        r = requests.get(url, timeout=15)
        data = r.json()["data"]
        chart = data["chart"]
        if isinstance(chart, dict) and "events" in chart:
            print(f"  {element_id} -> events list ({len(chart['events'])} events)")
        elif isinstance(chart, list):
            print(f"  {element_id} -> race data ({len(chart)} entries)")
        elif isinstance(chart, dict) and "standings" in chart:
            print(f"  {element_id} -> standings")
        else:
            print(f"  {element_id} -> unknown structure: {list(chart.keys()) if isinstance(chart, dict) else type(chart)}")

    print(f"\nadd the correct IDs to SEASON_CONFIG in this script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape DHL pit stop stationary times")
    parser.add_argument("--season", type=int, help="scrape a single season (default: all configured)")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--discover-season", type=int, metavar="YEAR", help="find DHL element IDs for a season")
    args = parser.parse_args()

    if args.discover_season:
        discover_season(args.discover_season)
    else:
        all_seasons = sorted(set(SEASON_CONFIG.keys()) | set(OVERVIEW_CONFIG.keys()))
        seasons = [args.season] if args.season else all_seasons
        total = 0
        for s in seasons:
            total += scrape_season(s, force=args.force)
        print(f"\ndone. {total} races scraped.")
