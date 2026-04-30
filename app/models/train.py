"""Training loop for XGBoost models - loads data, fits model, evaluates, and saves artifact to data/artifacts/."""

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier

from app.config import (
    PROCESSED_DRIVER_FEATURES_DIR, INTERIM_RACES_DIR, ARTIFACTS_DIR,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS
)
from app.models.evaluation import evaluate


MODEL_CLASSES = {
    "XGBRegressor": XGBRegressor,
    "XGBClassifier": XGBClassifier,
}


# loads driver features, joins finish and qualifying position, and returns train/val/test splits
def load_data(config):
    driver_features = pd.concat([pd.read_parquet(f) for f in sorted(PROCESSED_DRIVER_FEATURES_DIR.glob("*.parquet"))])
    race_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_RACES_DIR.glob("*.parquet"))])

    df = driver_features.merge( # TODO V2: replace grid_position with Model 1 predicted quali position
        race_results[["race_id", "driver_id", "finish_position", "grid_position"]].rename(columns={"grid_position": "quali_position"}),
        on=["race_id", "driver_id"],
        how="left"
    )

    df = df.dropna(subset=[config["target"]])  # drop rows with no finish position (DNS/early retirement before classification)

    X = df[config["features"]]
    y = df[config["target"]]

    X_train = X[df["season"].isin(TRAIN_SEASONS)]
    y_train = y[df["season"].isin(TRAIN_SEASONS)]

    X_val = X[df["season"].isin(VAL_SEASONS)]
    y_val = y[df["season"].isin(VAL_SEASONS)]

    X_test = X[df["season"].isin(TEST_SEASONS)]
    y_test = y[df["season"].isin(TEST_SEASONS)]

    return X_train, y_train, X_val, y_val, X_test, y_test


# instantiates and fits an XGBoost model using the config, with early stopping on the validation set
def train(config, X_train, y_train, X_val, y_val):
    model = MODEL_CLASSES[config["model_type"]](**config["hyperparams"])

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


# saves the fitted model to data/artifacts/
def save(model, config):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACTS_DIR / f"{config['name']}.joblib")
    

# orchestrates the full training pipeline - loads data, trains, saves, and evaluates
def main(config):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config)
    model = train(config, X_train, y_train, X_val, y_val)
    save(model, config)

    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")
