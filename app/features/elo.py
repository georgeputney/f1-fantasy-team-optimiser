"""Decomposed driver + constructor Elo ratings as features - loads pre-2018 seed ratings and processes interim race/quali data."""

import math

import pandas as pd

from app.config import MANUAL_DIR, INTERIM_RACES_DIR, INTERIM_QUALI_DIR, INTERIM_EVENTS_DIR

SEED_DIR = MANUAL_DIR / "elo_seeds"

# tuned via grid search in notebooks/elo_analysis.ipynb (576 combinations, holdout=2024)
TUNED_PARAMS = {
    "k_base": 150,
    "alpha": 0.65,
    "teammate_multiplier": 2.0,
    "season_reversion": 0.25,
    "reg_reversion": 0.75,
    "provisional_races": 30,
    "max_boost": 3.0,
    "margin_cap": 2.0,
}

# years after which a major reg change resets constructor ratings at the start of the next season
REG_BOUNDARY_YEARS = {2008, 2013, 2016, 2021, 2025}

INITIAL_RATING = 1500


class EloSystem:

    def __init__(self, k_base, alpha, teammate_multiplier, margin_cap,
                 provisional_races, max_boost, season_reversion, reg_reversion):
        self.k_base = k_base
        self.alpha = alpha
        self.teammate_multiplier = teammate_multiplier
        self.margin_cap = margin_cap
        self.provisional_races = provisional_races
        self.max_boost = max_boost
        self.season_reversion = season_reversion
        self.reg_reversion = reg_reversion

        self._driver_ratings = {}
        self._constructor_ratings = {}
        self._driver_races = {}


    def driver_rating(self, driver_id):
        return self._driver_ratings.setdefault(driver_id, INITIAL_RATING)


    def constructor_rating(self, constructor_id):
        return self._constructor_ratings.setdefault(constructor_id, INITIAL_RATING)


    def entry_strength(self, driver_id, constructor_id):
        return self.driver_rating(driver_id) + self.constructor_rating(constructor_id)


    # load pre-computed ratings and race counts as starting point (burned in from 2000-2017)
    def seed(self, driver_seeds, constructor_seeds, driver_race_counts=None):
        self._driver_ratings.update(driver_seeds)
        self._constructor_ratings.update(constructor_seeds)
        
        if driver_race_counts:
            self._driver_races.update(driver_race_counts)


    # pull constructor ratings toward INITIAL_RATING by the given fraction
    def apply_reversion(self, fraction):
        for cid in self._constructor_ratings:
            r = self._constructor_ratings[cid]
            self._constructor_ratings[cid] = r + fraction * (INITIAL_RATING - r)

    def _expected_score(self, strength_a, strength_b):
        return 1.0 / (1.0 + 10.0 ** ((strength_b - strength_a) / 400.0))


    def _margin_multiplier(self, pos_a, pos_b, n_classified):
        if n_classified < 2:
            return 1.0
        
        raw = math.log1p(abs(pos_a - pos_b)) / math.log(n_classified)
        return min(raw, self.margin_cap)


    def _rookie_k_multiplier(self, driver_id):
        races = self._driver_races.get(driver_id, 0)

        if races >= self.provisional_races:
            return 1.0
        
        return self.max_boost - (self.max_boost - 1.0) * (races / self.provisional_races)

    # process one race: pairwise elo updates for all classified finishers
    def update_race(self, race_df):
        entries = race_df.to_dict("records")
        n_classified = len(entries)
        n_opponents = max(n_classified - 1, 1)

        strengths = {
            r["driver_id"]: self.entry_strength(r["driver_id"], r["constructor_id"])
            for r in entries
        }
        rookie_mult = {r["driver_id"]: self._rookie_k_multiplier(r["driver_id"]) for r in entries}

        driver_delta = {r["driver_id"]: 0.0 for r in entries}
        constructor_delta = {r["constructor_id"]: 0.0 for r in entries}

        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                d_a, d_b = a["driver_id"], b["driver_id"]
                c_a, c_b = a["constructor_id"], b["constructor_id"]
                is_teammate = c_a == c_b

                e_a = self._expected_score(strengths[d_a], strengths[d_b])
                score_a = 1.0 if a["position"] < b["position"] else (
                    0.5 if a["position"] == b["position"] else 0.0)

                m_mult = self._margin_multiplier(a["position"], b["position"], n_classified)
                k_pair = (self.k_base / n_opponents) * m_mult
                delta = k_pair * (score_a - e_a)

                if is_teammate:
                    tm_delta = delta * self.teammate_multiplier
                    driver_delta[d_a] += tm_delta * rookie_mult[d_a]
                    driver_delta[d_b] -= tm_delta * rookie_mult[d_b]
                else:
                    driver_delta[d_a] += self.alpha * delta * rookie_mult[d_a]
                    driver_delta[d_b] -= self.alpha * delta * rookie_mult[d_b]
                    constructor_delta[c_a] += (1.0 - self.alpha) * delta
                    constructor_delta[c_b] -= (1.0 - self.alpha) * delta

        for driver_id, d in driver_delta.items():
            self._driver_ratings[driver_id] = self.driver_rating(driver_id) + d

        for constructor_id, d in constructor_delta.items():
            self._constructor_ratings[constructor_id] = self.constructor_rating(constructor_id) + d

        for driver_id in driver_delta:
            self._driver_races[driver_id] = self._driver_races.get(driver_id, 0) + 1


# load seed csvs burned in from 2000-2017 in the elo_analysis notebook
def _load_seeds():
    driver_path = SEED_DIR / "driver_elo_seeds.csv"
    constructor_path = SEED_DIR / "constructor_elo_seeds.csv"

    driver_seeds = {"race": {}, "quali": {}, "race_counts": {}}
    constructor_seeds = {"race": {}, "quali": {}}

    if driver_path.exists():
        df = pd.read_csv(driver_path)
        driver_seeds["race"] = df.set_index("driver_id")["race_elo"].to_dict()
        driver_seeds["quali"] = df.set_index("driver_id")["quali_elo"].to_dict()

        if "race_count" in df.columns:
            driver_seeds["race_counts"] = df.set_index("driver_id")["race_count"].astype(int).to_dict()

    if constructor_path.exists():
        df = pd.read_csv(constructor_path)
        constructor_seeds["race"] = df.set_index("constructor_id")["race_elo"].to_dict()
        constructor_seeds["quali"] = df.set_index("constructor_id")["quali_elo"].to_dict()

    return driver_seeds, constructor_seeds


# load all interim race, quali, and event parquets into sorted dataframes
def _load_all_interim_data():
    race_files = sorted(INTERIM_RACES_DIR.glob("*.parquet"))
    quali_files = sorted(INTERIM_QUALI_DIR.glob("*.parquet"))
    event_files = sorted(INTERIM_EVENTS_DIR.glob("*.parquet"))

    races = pd.concat([pd.read_parquet(f) for f in race_files], ignore_index=True)
    qualis = pd.concat([pd.read_parquet(f) for f in quali_files], ignore_index=True)
    events = pd.concat([pd.read_parquet(f) for f in event_files], ignore_index=True)

    date_map = events.set_index("race_id")["event_date"]
    races["date"] = races["race_id"].map(date_map)
    qualis["date"] = qualis["race_id"].map(date_map)

    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    qualis = qualis.sort_values(["season", "round"]).reset_index(drop=True)

    return races, qualis


# filter to classified finishers and normalise the position column
def _prepare_for_elo(df, position_col):
    out = df[~df["dnf_flag"]].copy() if "dnf_flag" in df.columns else df.copy()
    out = out.rename(columns={position_col: "position"})
    out = out.dropna(subset=["position"])
    out["position"] = out["position"].astype(int)

    return out[["race_id", "season", "round", "date", "driver_id", "constructor_id", "position"]]


# run elo from seeds through all interim data, returning pre-race ratings as features
# ratings are snapshotted BEFORE each race is processed to avoid leakage
def compute_elo_features():
    driver_seeds, constructor_seeds = _load_seeds()

    race_elo = EloSystem(**TUNED_PARAMS)
    quali_elo = EloSystem(**TUNED_PARAMS)

    if driver_seeds["race"]:
        race_elo.seed(driver_seeds["race"], constructor_seeds["race"], driver_seeds["race_counts"])
        quali_elo.seed(driver_seeds["quali"], constructor_seeds["quali"], driver_seeds["race_counts"])

    races, qualis = _load_all_interim_data()

    race_input = _prepare_for_elo(races, "finish_position")
    quali_input = _prepare_for_elo(qualis, "quali_position")

    # driver->constructor lookup per race from full data (including DNFs)
    driver_constructor = races.set_index(["race_id", "driver_id"])["constructor_id"].to_dict()

    rows = []
    prev_season = None

    race_ids_ordered = (
        race_input.drop_duplicates("race_id")[["race_id", "season", "round"]]
        .sort_values(["season", "round"])["race_id"]
        .tolist()
    )

    for race_id in race_ids_ordered:
        race_df = race_input[race_input["race_id"] == race_id]
        season = race_df["season"].iloc[0]

        # season/regulation reversion before first race of new season
        if prev_season is not None and season != prev_season:
            reversion = (
                TUNED_PARAMS["reg_reversion"] if prev_season in REG_BOUNDARY_YEARS
                else TUNED_PARAMS["season_reversion"]
            )
            if reversion > 0:
                race_elo.apply_reversion(reversion)
                quali_elo.apply_reversion(reversion)

        # snapshot ratings BEFORE this race (no leakage)
        all_drivers = races[races["race_id"] == race_id]["driver_id"].unique()
        for driver_id in all_drivers:
            constructor_id = driver_constructor.get((race_id, driver_id))
            
            if constructor_id is None:
                continue

            rows.append({
                "race_id": race_id,
                "driver_id": driver_id,
                "driver_race_elo": race_elo.driver_rating(driver_id),
                "driver_quali_elo": quali_elo.driver_rating(driver_id),
                "constructor_race_elo": race_elo.constructor_rating(constructor_id),
                "constructor_quali_elo": quali_elo.constructor_rating(constructor_id),
                "entry_race_strength": race_elo.entry_strength(driver_id, constructor_id),
                "entry_quali_strength": quali_elo.entry_strength(driver_id, constructor_id),
            })

        # update with this race's results
        race_elo.update_race(race_df)

        # update with this race's qualifying
        quali_df = quali_input[quali_input["race_id"] == race_id]
        if not quali_df.empty:
            quali_elo.update_race(quali_df)

        prev_season = season

    return pd.DataFrame(rows)
