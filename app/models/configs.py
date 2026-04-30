"""Model config definitions for each prediction target — features, hyperparameters, and evaluation settings."""

#
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
        "quali_position", #train on real position then substitute for predicted
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