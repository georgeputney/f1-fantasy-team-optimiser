"""Model config definitions for each prediction target — features, hyperparameters, and evaluation settings."""

# XGBoost regressor predicting race finish position (1-20) - quali_position feature uses actual qualifying position at training time
FINISH_POSITION_MODEL = {
    "name": "finish_position",
    "target": "finish_position",
    "model_type": "XGBRegressor",
    "features": [
        "rolling_quali_pos_last_3",
        "rolling_quali_pos_last_5",
        "rolling_finish_pos_last_3",
        "rolling_finish_pos_last_5",
        "rolling_fantasy_points_last_3",
        "rolling_fantasy_points_last_5",
        "rolling_dnf_rate_last_5",
        "circuit_rolling_quali_pos_last_3",
        "circuit_rolling_quali_pos_last_5",
        "circuit_rolling_finish_pos_last_3",
        "circuit_rolling_finish_pos_last_5",
        "season_points_to_date",
        "round_number",
        "is_street_circuit",
        "constructor_rolling_fantasy_points_last_3",
        "constructor_rolling_fantasy_points_last_5",
        "constructor_rolling_dnf_rate_last_5",
        "constructor_rolling_quali_pos_last_3",
        "constructor_form_trend_last_5",
        "quali_position", # supplied as predicted_quali_position from the quali model at inference time
    ],
    "hyperparams": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 20,
    },
    "eval_metrics": ["mae", "spearman"],
    "feature_importance_top_n": 10, 
}


# XGBoost regressor predicting qualifying position (1-20) - no quali input by design, only pre-session features
QUALI_POSITION_MODEL = {
    "name": "quali_position",
    "target": "quali_position",
    "model_type": "XGBRegressor",
    "features": [
        "rolling_finish_pos_last_3",
        "rolling_finish_pos_last_5",
        "rolling_quali_pos_last_3",
        "rolling_quali_pos_last_5",
        "circuit_rolling_quali_pos_last_3",
        "circuit_rolling_quali_pos_last_5",
        "season_points_to_date",
        "round_number",
        "is_street_circuit",
        "constructor_rolling_quali_pos_last_3",
        "constructor_form_trend_last_5",
    ],
    "hyperparams": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 20,
    },
    "eval_metrics": ["mae", "spearman"],
    "feature_importance_top_n": 10, 
}