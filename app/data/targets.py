"""Loads actual fantasy points from manually uploaded CSVs in data/manual/fantasy_points/."""

import pandas as pd

from app.config import FANTASY_POINTS_DIR


# load a single race's fantasy points CSV and return a DataFrame matching the targets schema
def load_fantasy_targets(season, round_num):
    path = FANTASY_POINTS_DIR / f"{season}_{round_num:02d}.csv"
    df = pd.read_csv(path)
    
    df["season"] = season
    df["round"] = round_num
    df = df.rename(columns={"fantasy_points": "actual_fantasy_points"})

    return df[["race_id", "season", "round", "asset_id", "asset_type", "actual_fantasy_points"]]


# load all fantasy points CSVs and return a single concatenated DataFrame
def load_all_fantasy_targets():
    files = sorted(FANTASY_POINTS_DIR.glob("*.csv"))
    frames = []

    for f in files:
        season, round_num = f.stem.split("_")
        frames.append(load_fantasy_targets(int(season), int(round_num)))

    return pd.concat(frames).reset_index(drop=True)
