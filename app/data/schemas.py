"""Pandera schemas defining the internal table contracts for the F1 fantasy pipeline."""

import pandera.pandas as pa


# event-level metadata, one row per race weekend
# primary key: race_id 
# all other tables join here for circuit and event context
events = pa.DataFrameSchema(
    {
        "race_id": pa.Column(str, 
            checks=pa.Check.str_matches(r"^\d{4}_\d{1,2}$"),
            unique=True
        ),
        "season": pa.Column(int, 
            checks=[
                pa.Check.greater_than_or_equal_to(2018), 
                pa.Check.less_than_or_equal_to(2030)
            ]
        ),
        "round": pa.Column(int),
        "event_name": pa.Column(str),
        "location": pa.Column(str),
        "country": pa.Column(str),
        "event_date": pa.Column(pa.DateTime),
        "is_street_circuit": pa.Column(bool),
        "is_sprint": pa.Column(bool),
    },
    strict=True,
)


# driver-level race outcomes, one row per driver per race
# primary key: race_id + driver_id
# source of truth for driver-constructor relationships per race
race_results = pa.DataFrameSchema(
    {
        "race_id": pa.Column(str, 
            checks=pa.Check.str_matches(r"^\d{4}_\d{1,2}$")
        ),
        "driver_id": pa.Column(str),
        "constructor_id": pa.Column(str),
        "grid_position": pa.Column(float, 
            checks=[
                pa.Check.greater_than_or_equal_to(0), 
                pa.Check.less_than_or_equal_to(22),
            ],
            nullable=True,
        ),
        "finish_position": pa.Column(float, 
            checks=[
                pa.Check.greater_than_or_equal_to(1), 
                pa.Check.less_than_or_equal_to(22),
            ],
            nullable=True,
        ),
        "status": pa.Column(str),
        "dnf_flag": pa.Column(bool),
        "points": pa.Column(float),
        "positions_gained": pa.Column(float, nullable=True),
        "fastest_lap_flag": pa.Column(bool),
    },
    strict=True,
)


# driver-level qualifying outcomes, one row per driver per race
# primary key: race_id + driver_id
# q2_time and q3_time are null for drivers eliminated in q1/q2 respectively
quali_results = pa.DataFrameSchema(
    {
        "race_id": pa.Column(str, 
            checks=pa.Check.str_matches(r"^\d{4}_\d{1,2}$")
        ),
        "driver_id": pa.Column(str),
        "constructor_id": pa.Column(str),
        "quali_position": pa.Column(float, 
            checks=[
                pa.Check.greater_than_or_equal_to(0), 
                pa.Check.less_than_or_equal_to(22),
            ],
            nullable=True,
        ),
        "q1_time": pa.Column(float, nullable=True),
        "q2_time": pa.Column(float, nullable=True),
        "q3_time": pa.Column(float, nullable=True),
    },
    strict=True,
)


# fantasy prices per driver and constructor per race, one row per asset per race
# primary key: race_id + asset_id
# manually maintained - must be updated before each race weekend
fantasy_prices = pa.DataFrameSchema(
    {
        "race_id": pa.Column(str, 
            checks=pa.Check.str_matches(r"^\d{4}_\d{1,2}$")
        ),
        "asset_id": pa.Column(str),
        "asset_type": pa.Column(str, pa.Check.isin(["driver", "constructor"])),
        "price": pa.Column(float, pa.Check.greater_than_or_equal_to(3.5)),
    },
    strict=True,
)


# actual fantasy points scored per asset per race, one row per asset per race
# primary key: race_id + asset_id
# post-race only - used as training labels for the models
fantasy_targets = pa.DataFrameSchema(
    {
        "race_id": pa.Column(str, 
            checks=pa.Check.str_matches(r"^\d{4}_\d{1,2}$")
        ),
        "asset_id": pa.Column(str),
        "asset_type": pa.Column(str, pa.Check.isin(["driver", "constructor"])),
        "actual_fantasy_points": pa.Column(float),
    },
    strict=True,
)


# engineered features per driver per race, one row per driver per race
# primary key: race_id + driver_id + prediction_stage
# strict=False - feature columns are not enumerated here, validated by the feature store
driver_features = pa.DataFrameSchema(
    {
        "race_id": pa.Column(str, 
            checks=pa.Check.str_matches(r"^\d{4}_\d{1,2}$")
        ),
        "driver_id": pa.Column(str),
        "prediction_stage": pa.Column(str,
            pa.Check.isin(["pre_quali", "post_quali", "training"])                          
        ),
    },
    strict=False,
)


# model predictions per asset per race, one row per asset per race
# primary key: race_id + asset_id
# predicted_quali_pos is null for constructor rows
predictions = pa.DataFrameSchema(
    {
        "race_id": pa.Column(str, 
            checks=pa.Check.str_matches(r"^\d{4}_\d{1,2}$")
        ),
        "asset_id": pa.Column(str),
        "asset_type": pa.Column(str, pa.Check.isin(["driver", "constructor"])),
        "predicted_quali_pos": pa.Column(float, 
            checks=[
                pa.Check.greater_than_or_equal_to(1), 
                pa.Check.less_than_or_equal_to(22),
            ],
            nullable=True,
        ),
        "predicted_race_pos": pa.Column(float, 
            checks=[
                pa.Check.greater_than_or_equal_to(1), 
                pa.Check.less_than_or_equal_to(22),
            ],
            nullable=True,
        ),
        "dnf_prob": pa.Column(float, 
            checks=[
                pa.Check.greater_than_or_equal_to(0), 
                pa.Check.less_than_or_equal_to(1),
            ],
        ),
        "fastest_lap_prob": pa.Column(float, 
            checks=[
                pa.Check.greater_than_or_equal_to(0), 
                pa.Check.less_than_or_equal_to(1),
            ],
        ),
        "expected_fantasy_points": pa.Column(float),
    },
    strict=True,
)