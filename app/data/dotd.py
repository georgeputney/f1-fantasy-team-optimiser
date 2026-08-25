"""DOTD probability model - per-driver normalised win probabilities based on historical vote rates."""

import pandas as pd

from app.config import INTERIM_DOTD_DIR, INTERIM_RACES_DIR

# laplace smoothing weight - equivalent to this many races at the global mean rate.
# higher = shrinks new drivers faster toward the mean; 5 means ~5 races before personal rate dominates.
_SMOOTHING = 5.0


# race_id is "{season}_{round:02d}" - true if it's strictly before the cutoff, so a walk-forward
# caller (e.g. backtest) can exclude any race that hadn't happened yet as of the round being predicted
def _is_before(race_id, before):
    if before is None:
        return True
    season, round_num = (int(x) for x in race_id.split("_"))
    return (season, round_num) < before


# returns a predict_dotd(driver_ids) function built from historical DOTD win rates
# uses per-driver Bayesian smoothing: (wins + k * global_mean) / (driver_races + k)
# where driver_races is how many races each driver actually competed in - so a new
# driver with 3 wins in 10 races gets ~27% rather than 3/177 from dividing by the
# full dataset size. probabilities are normalised to sum to 1.0 across the field.
# `before` restricts to races strictly before (season, round) - used by the walk-forward backtest
# so a later round's DOTD result can't leak into an earlier round's prediction; live callers
# (generate-reports, optimise-team) omit it and use everything available.
def build_dotd_predictor(before=None):
    dotd = pd.concat([pd.read_csv(f) for f in sorted(INTERIM_DOTD_DIR.glob("*.csv"))])
    dotd = dotd.dropna(subset=["driver_id"])
    dotd = dotd[dotd["driver_id"].str.strip() != ""]
    dotd = dotd[dotd["race_id"].apply(lambda r: _is_before(r, before))]

    global_mean = 1.0 / 20
    win_counts = dotd["driver_id"].value_counts()

    # per-driver race counts from interim race results
    race_frames = []
    for f in sorted(INTERIM_RACES_DIR.glob("*.parquet")):
        season, round_num = (int(x) for x in f.stem.split("_"))
        if before is None or (season, round_num) < before:
            race_frames.append(pd.read_parquet(f, columns=["driver_id"]))
    driver_race_counts = pd.concat(race_frames)["driver_id"].value_counts()

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
