"""Loads a trained model and generates finish position predictions for a given race."""

import joblib
import pandas as pd

from app.models.configs import FINISH_POSITION_MODEL

from app.config import PROCESSED_DRIVER_FEATURES_DIR, ARTIFACTS_DIR, INTERIM_RACES_DIR


# loads a trained model artifact from data/artifacts/
def load_model(config):
    return joblib.load(ARTIFACTS_DIR / f"{config['name']}.joblib")


# loads driver features for a given race and returns predicted finish positions per driver
def predict(model, config, season, round_num):
    features = pd.read_parquet(PROCESSED_DRIVER_FEATURES_DIR / f"{season}_{round_num:02d}.parquet")
    
    race_results = pd.read_parquet(INTERIM_RACES_DIR / f"{season}_{round_num:02d}.parquet")
    features = features.merge( # TODO V2: replace grid_position with Model 1 predicted quali position once Model 1 is built
        race_results[["driver_id", "grid_position"]].rename(columns={"grid_position": "quali_position"}),
        on="driver_id",
        how="left"
    )

    X = features[config["features"]]
    predictions = model.predict(X)

    predictions = pd.Series(predictions).rank().astype(int).values  # convert to integer ranks 1-20
        
    return pd.DataFrame({
        "driver_id": features["driver_id"],
        "constructor_id": features["constructor_id"],
        "predicted_finish_position": predictions,
    }).sort_values("predicted_finish_position").reset_index(drop=True)
