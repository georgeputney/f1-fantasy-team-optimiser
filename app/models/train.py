"""Training loop for XGBoost models - loads data, fits model, evaluates, and saves artifact to data/artifacts/."""

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier

from app.config import (
    PROCESSED_HISTORIC_FEATURES_DIR, INTERIM_RACES_DIR, INTERIM_QUALI_DIR, ARTIFACTS_DIR,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, PROCESSED_PRACTICE_FEATURES_DIR
)
from app.models.evaluation import evaluate


MODEL_CLASSES = {
    "XGBRegressor": XGBRegressor,
    "XGBClassifier": XGBClassifier,
}


# loads historic & practice features, joins finish and qualifying position, and returns train/val/test splits
# if quali_model and quali_config are provided, replaces actual quali_position with model predictions
# so the finish model trains on the same noisy input it will see at inference time
def load_data(config, quali_model=None, quali_config=None):
    historic_features = pd.concat([pd.read_parquet(f) for f in sorted(PROCESSED_HISTORIC_FEATURES_DIR.glob("*.parquet"))])
    practice_features = pd.concat([pd.read_parquet(f) for f in sorted(PROCESSED_PRACTICE_FEATURES_DIR.glob("*.parquet"))])

    race_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_RACES_DIR.glob("*.parquet"))])
    quali_results = pd.concat([pd.read_parquet(f) for f in sorted(INTERIM_QUALI_DIR.glob("*.parquet"))])

    df = historic_features.merge(
        race_results[["race_id", "driver_id", "finish_position", "dnf_flag"]],
        on=["race_id", "driver_id"],
        how="left"
    ).merge(
        quali_results[["race_id", "driver_id", "quali_position"]],
        on=["race_id", "driver_id"],
        how="left"
    ).merge(
        practice_features,
        on=["race_id", "driver_id"],
        how="left"
    )

    if quali_model is not None:
        # replace actual quali positions with model predictions so training distribution matches inference
        X_quali = df[quali_config["features"]]
        raw_preds = quali_model.predict(X_quali)
        # rank within each race so predictions are valid positions (1-N) not raw regressor outputs
        df["predicted_quali_position"] = (
            df.assign(_pred=raw_preds)
            .groupby("race_id")["_pred"]
            .rank(method="first")
            .astype(int)
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
    

# orchestrates the full training pipeline - loads data, trains, saves, and evaluates.
# pass quali_model and quali_config when training the finish model so it trains on predicted quali positions.
def main(config, quali_model=None, quali_config=None):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config, quali_model, quali_config)
    model = train(config, X_train, y_train, X_val, y_val)
    save(model, config)

    evaluate(model, X_val, y_val, "val", config["eval_metrics"])
    evaluate(model, X_test, y_test, "test", config["eval_metrics"])

