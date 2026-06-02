"""DOTD probability model - per-driver normalised win probabilities based on historical vote rates."""

import pandas as pd

from app.config import MANUAL_DIR, INTERIM_RACES_DIR

DOTD_FILE = MANUAL_DIR / "driver_of_the_day.csv"

# laplace smoothing weight - equivalent to this many races at the global mean rate.
# higher = shrinks new drivers faster toward the mean; 5 means ~5 races before personal rate dominates.
_SMOOTHING = 5.0


# returns a predict_dotd(driver_ids) function built from historical DOTD win rates
# uses per-driver Bayesian smoothing: (wins + k * global_mean) / (driver_races + k)
# where driver_races is how many races each driver actually competed in - so a new
# driver with 3 wins in 10 races gets ~27% rather than 3/177 from dividing by the
# full dataset size. probabilities are normalised to sum to 1.0 across the field
def build_dotd_predictor():
    dotd = pd.read_csv(DOTD_FILE)
    dotd = dotd.dropna(subset=["driver_id"])
    dotd = dotd[dotd["driver_id"].str.strip() != ""]

    global_mean = 1.0 / 20
    win_counts = dotd["driver_id"].value_counts()

    # per-driver race counts from interim race results
    driver_race_counts = (
        pd.concat([pd.read_parquet(f, columns=["driver_id"]) for f in sorted(INTERIM_RACES_DIR.glob("*.parquet"))])
        ["driver_id"].value_counts()
    )

    # smoothed per-driver rate: (wins + k * global_mean) / (driver_races + k)
    all_drivers = driver_race_counts.index.union(win_counts.index)
    wins  = win_counts.reindex(all_drivers, fill_value=0)
    races = driver_race_counts.reindex(all_drivers, fill_value=1)  # min 1 to avoid div/0
    smoothed = (wins + _SMOOTHING * global_mean) / (races + _SMOOTHING)

    def predict_dotd(driver_ids: pd.Series) -> pd.Series:
        # returns per-driver DOTD probabilities normalised to sum to 1.0 across the field
        raw = driver_ids.map(smoothed).fillna(global_mean)
        return raw / raw.sum()

    return predict_dotd
