"""Loads trained models and generates predictions for a given race.

Runs the quali position model first, then feeds predicted_quali_position
into the finish position model as a feature — always chained, no fallback.
"""

import joblib
import pandas as pd

from app.config import PROCESSED_HISTORIC_FEATURES_DIR, PROCESSED_PRACTICE_FEATURES_DIR, ARTIFACTS_DIR


# loads a trained model artifact from data/artifacts/
def load_model(config):
    return joblib.load(ARTIFACTS_DIR / f"{config['name']}.joblib")


# loads historic & practice features for a given race, runs the quali model, then the finish model,
# and returns a DataFrame with predicted_quali_position and predicted_finish_position per driver
def predict(quali_model, quali_config, finish_model, finish_config, season, round_num):
    historic_features = pd.read_parquet(PROCESSED_HISTORIC_FEATURES_DIR / f"{season}_{round_num:02d}.parquet")
    
    practice_path = PROCESSED_PRACTICE_FEATURES_DIR / f"{season}_{round_num:02d}.parquet"
    practice_features = pd.read_parquet(practice_path) if practice_path.exists() else None

    if practice_features is not None:
        features = historic_features.merge(practice_features, on=["race_id", "driver_id"], how="left")
    else:
        features = historic_features.copy()
        practice_cols = set(c for c in quali_config["features"] + finish_config["features"] if c.startswith("fp"))
        for col in practice_cols:
            features[col] = float("nan")

    # step 1: predict qualifying position (no quali input by design)
    X_quali = features[quali_config["features"]]
    quali_preds = quali_model.predict(X_quali)
    features["predicted_quali_position"] = pd.Series(quali_preds).rank().astype(int).values

    # step 2: predict finish position using predicted quali position as a feature
    X_finish = features[finish_config["features"]]
    finish_preds = finish_model.predict(X_finish)
    features["predicted_finish_position"] = pd.Series(finish_preds).rank().astype(int).values

    return pd.DataFrame({
        "driver_id": features["driver_id"],
        "constructor_id": features["constructor_id"],
        "predicted_quali_position": features["predicted_quali_position"],
        "predicted_finish_position": features["predicted_finish_position"],
    }).sort_values("predicted_finish_position").reset_index(drop=True)
