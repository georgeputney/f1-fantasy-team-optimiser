"""Shared utilities for feature engineering."""

# filters a results DataFrame to rows for a given asset strictly before the current race, 
# preserving temporal ordering with no leakage
def _get_prior_results(results, driver_id, season, round_num, id_col="driver_id"):

    return results[
        (results[id_col] == driver_id) &
        ((results["season"] < season) | 
        ((results["season"] == season) & (results["round"] < round_num)))
    ]