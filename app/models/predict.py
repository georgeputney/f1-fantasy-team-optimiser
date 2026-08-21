"""Loads trained models and generates predictions for a given race.

Runs the quali position model first, then feeds predicted_quali_position
into the finish position model as a feature - always chained, no fallback.
"""

import joblib
import pandas as pd

from app.config import PROCESSED_HISTORIC_FEATURES_DIR, PROCESSED_PRACTICE_FEATURES_DIR, PROCESSED_CIRCUIT_FEATURES_DIR, ARTIFACTS_DIR, INTERIM_SPRINT_QUALIFYING_DIR, INTERIM_EVENTS_DIR


# true if the round is a sprint weekend, read from the interim events file (is_sprint flag).
# lets us project sprint points before the sprint qualifying session has actually run.
def _is_sprint_weekend(season, round_num):
    events_path = INTERIM_EVENTS_DIR / f"{season}_{round_num:02d}.parquet"
    if not events_path.exists():
        return False
    events = pd.read_parquet(events_path)
    return bool(events["is_sprint"].iloc[0]) if "is_sprint" in events.columns else False


# on a sprint weekend the sprint grid predicts sprint points (compose uses sprint_quali_position as a
# stand-in for the sprint finish). before that session runs the file is absent, so fall back to the
# predicted quali position as the best available proxy - otherwise sprint points are silently dropped
# from every projection made while teams are still being locked in.
def _fill_sprint_proxy(features, season, round_num):
    if features["sprint_quali_position"].isna().all() and _is_sprint_weekend(season, round_num):
        features["sprint_quali_position"] = features["predicted_quali_position"].astype(float)


# loads a trained model artifact - pass prod=True to load the production artifact trained on all historical data
def load_model(config, prod=False):
    if prod:
        return joblib.load(ARTIFACTS_DIR / f"{config['name']}_prod.joblib")
    return joblib.load(ARTIFACTS_DIR / f"{config['name']}.joblib")


# loads a per-season walk-forward artifact; falls back to prod then dev if not found
def load_season_model(config, season):
    season_path = ARTIFACTS_DIR / f"{config['name']}_{season}.joblib"
    if season_path.exists():
        return joblib.load(season_path)
    prod_path = ARTIFACTS_DIR / f"{config['name']}_prod.joblib"
    if prod_path.exists():
        return joblib.load(prod_path)
    return joblib.load(ARTIFACTS_DIR / f"{config['name']}.joblib")


# loads historic, practice, and circuit features for a race and fills any feature columns the
# model configs expect but this race's data doesn't have (e.g. added after older files were processed)
def _load_features(quali_config, finish_config, season, round_num):
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

    circuit_path = PROCESSED_CIRCUIT_FEATURES_DIR / f"{season}_{round_num:02d}.parquet"
    if circuit_path.exists():
        circuit_features = pd.read_parquet(circuit_path)
        features = features.merge(circuit_features, on="race_id", how="left")
    else:
        circuit_cols = set(c for c in quali_config["features"] + finish_config["features"] if c.startswith("circuit_"))
        for col in circuit_cols:
            features[col] = float("nan")

    # load sprint qualifying position if this is a sprint weekend
    sq_path = INTERIM_SPRINT_QUALIFYING_DIR / f"{season}_{round_num:02d}.parquet"
    if sq_path.exists():
        sq = pd.read_parquet(sq_path).set_index("driver_id")["sprint_quali_position"]
        features["sprint_quali_position"] = features["driver_id"].map(sq)
    else:
        features["sprint_quali_position"] = float("nan")

    for col in set(quali_config["features"] + finish_config["features"]):
        if col not in features.columns:
            features[col] = float("nan")

    return features


# loads historic & practice features for a given race, runs the quali model, then the finish model,
# and returns a DataFrame with predicted positions per driver
def predict(quali_model, quali_config, finish_model, finish_config, season, round_num):
    features = _load_features(quali_config, finish_config, season, round_num)

    # step 1: predict qualifying position (no quali input by design)
    X_quali = features[quali_config["features"]]
    quali_preds = quali_model.predict(X_quali)
    features["predicted_quali_position"] = pd.Series(quali_preds).rank(method="first").astype(int).values

    # step 2: predict finish position using predicted quali position as a feature
    X_finish = features[finish_config["features"]]
    finish_preds = finish_model.predict(X_finish)
    features["predicted_finish_position"] = pd.Series(finish_preds).rank(method="first").astype(int).values

    _fill_sprint_proxy(features, season, round_num)

    return pd.DataFrame({
        "driver_id": features["driver_id"],
        "constructor_id": features["constructor_id"],
        "predicted_quali_position": features["predicted_quali_position"],
        "predicted_finish_position": features["predicted_finish_position"],
        "sprint_quali_position": features["sprint_quali_position"],
    }).sort_values("predicted_finish_position").reset_index(drop=True)


# same as predict() but also keeps the raw (pre-ranking) XGBoost outputs, used by the Monte Carlo
# simulator - residual pools are calibrated against these raw scores, not the ranked positions
def predict_with_raw(quali_model, quali_config, finish_model, finish_config, season, round_num):
    features = _load_features(quali_config, finish_config, season, round_num)

    X_quali = features[quali_config["features"]]
    raw_quali = quali_model.predict(X_quali)
    features["predicted_quali_position"] = pd.Series(raw_quali).rank(method="first").astype(int).values

    X_finish = features[finish_config["features"]]
    raw_finish = finish_model.predict(X_finish)
    features["predicted_finish_position"] = pd.Series(raw_finish).rank(method="first").astype(int).values

    _fill_sprint_proxy(features, season, round_num)

    return pd.DataFrame({
        "driver_id": features["driver_id"],
        "constructor_id": features["constructor_id"],
        "predicted_quali_position": features["predicted_quali_position"],
        "predicted_finish_position": features["predicted_finish_position"],
        "raw_quali": raw_quali,
        "raw_finish": raw_finish,
        "sprint_quali_position": features["sprint_quali_position"],
    }).sort_values("predicted_finish_position").reset_index(drop=True)

