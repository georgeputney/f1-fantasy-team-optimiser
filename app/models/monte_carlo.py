"""Monte Carlo race-outcome simulation - calibrated per-driver/constructor fantasy point distributions."""

import numpy as np
import pandas as pd
import joblib

from app.config import (
    PROCESSED_HISTORIC_FEATURES_DIR, INTERIM_RACES_DIR, INTERIM_QUALI_DIR, ARTIFACTS_DIR, TRAIN_SEASONS,
)
from app.models.configs import QUALI_POSITION_MODEL, FINISH_POSITION_MODEL
from app.models.predict import load_season_model, predict_with_raw
from app.data.overtakes import build_overtake_predictor
from app.data.dotd import build_dotd_predictor
from app.data.scoring_rules import (
    DRIVER_RACE_POSITION_POINTS, DRIVER_QUALI_POSITION_POINTS, CONSTRUCTOR_QUALI_BONUS,
    FASTEST_LAP_POINTS, DOTD_POINTS, POSITION_GAINED_POINTS, RACE_PENALTY, OVERTAKE_MADE_POINTS,
    DRIVER_SPRINT_POSITION_POINTS, SPRINT_FASTEST_LAP_POINTS, SPRINT_OVERTAKE_MADE_POINTS,
)
from app.models.compose import FASTEST_LAP_PROB, expected_pitstop_points

# display only - team selection stays on the ranked pipeline (app.optimiser). MC was validated against
# the season backtest and loses at team selection (see project memory: MC's driver-level MAE win never
# survives the team-level task, and multi-season sweeps found no calibration that beats ranked). Its one
# validated use is rendering a calibrated per-asset range (~79-80% P10-P90 coverage) alongside the ranked
# expected-points figure the optimiser actually uses.
# ported from notebooks/monte_carlo_scoring.ipynb (2026-08-08 experiment, see that notebook for the full
# evaluation) - this module keeps only what production needs: calibrate once from historical residuals,
# then simulate the current round on demand.

CALIBRATION_ARTIFACT = ARTIFACTS_DIR / "mc_calibration.joblib"
CALIBRATION_SEASONS = TRAIN_SEASONS  # held out from the live/eval seasons, matches the notebook split

GRID = 41  # headroom past 20 - a few historical races ran reserve/substitute entries past P20
TIER_EDGES = [0.0, 3.5, 7.5, 12.5, 21.0]     # predicted-position tiers used to bucket residual pools
POS_BINS = [0, 5, 10, 15, GRID]              # predicted-finish bins used to calibrate P(DNF)

RACE_PTS = np.zeros(GRID)
QUALI_PTS = np.zeros(GRID)
for _p, _v in DRIVER_RACE_POSITION_POINTS.items():
    RACE_PTS[_p] = _v
for _p, _v in DRIVER_QUALI_POSITION_POINTS.items():
    QUALI_PTS[_p] = _v
FL_PROB_ARR = np.zeros(GRID)
for _p, _v in FASTEST_LAP_PROB.items():
    FL_PROB_ARR[_p] = _v


def _tier_of(raw):
    return np.clip(np.digitize(raw, TIER_EDGES[1:-1]), 0, 3)


# bucket a predicted (ranked) finishing/qualifying position into the 4 residual-pool tiers.
# we key on the RANKED position, not the raw value: raw predictions are compressed (~P5-15 even for the
# leader), so a raw-value cutoff would file the predicted winner into a midfield-noise pool.
def _pos_tier(positions):
    return np.clip(np.digitize(np.asarray(positions), POS_BINS[1:-1]), 0, 3)


# dense competition-free ranks 1..n along axis=1 (argsort of argsort)
def _ranks(x):
    return np.argsort(np.argsort(x, axis=1), axis=1) + 1


# calibration - walks CALIBRATION_SEASONS once, saves residual pools / DNF curve / frailty to disk.
# entirely local (reads already-built historic features + interim results), no network calls.
def _collect_calibration_rows():
    rows = []
    for season in CALIBRATION_SEASONS:
        quali_model = load_season_model(QUALI_POSITION_MODEL, season)
        finish_model = load_season_model(FINISH_POSITION_MODEL, season)

        round_nums = sorted(
            int(f.stem.split("_")[1]) for f in PROCESSED_HISTORIC_FEATURES_DIR.glob(f"{season}_*.parquet")
        )
        for rnd in round_nums:
            races_path = INTERIM_RACES_DIR / f"{season}_{rnd:02d}.parquet"
            quali_path = INTERIM_QUALI_DIR / f"{season}_{rnd:02d}.parquet"
            if not (races_path.exists() and quali_path.exists()):
                continue

            preds = predict_with_raw(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, season, rnd)
            race_results = pd.read_parquet(races_path).set_index("driver_id")
            quali_results = pd.read_parquet(quali_path).set_index("driver_id")

            for _, row in preds.iterrows():
                did = row["driver_id"]
                if did not in race_results.index:
                    continue
                actual_finish = race_results.loc[did, "finish_position"]
                actual_quali = quali_results.loc[did, "quali_position"] if did in quali_results.index else np.nan
                if pd.isna(actual_finish) or pd.isna(actual_quali):
                    continue

                rows.append({
                    "season": season, "round": rnd, "driver_id": did,
                    "raw_finish": row["raw_finish"], "raw_quali": row["raw_quali"],
                    "ranked_finish": row["predicted_finish_position"],
                    "ranked_quali": row["predicted_quali_position"],
                    "actual_finish": int(actual_finish), "actual_quali": int(actual_quali),
                    "dnf_flag": bool(race_results.loc[did, "dnf_flag"]),
                })

    return pd.DataFrame(rows)


# recomputes residual pools, the DNF-by-tier curve, and frailty phi from historical data, and saves
# them to ARTIFACTS_DIR. run this once (or whenever CALIBRATION_SEASONS' underlying data or models
# change) - simulate_round() loads the cached result rather than recalibrating per request
def build_calibration():
    data = _collect_calibration_rows()
    # residual is measured against the RANKED (integer) prediction, not the raw XGBoost score: raw outputs
    # are heavily compressed (the predicted winner scores raw ~4, not 1), so a raw-centred sim pulls
    # front-runners down to ~P3 and disagrees with the ranked headline. the ranked position tracks the
    # actual median finish almost exactly (P1->1, P5->5), so ranked + (actual-ranked) residual is both
    # better calibrated (coverage ~0.82 vs ~0.88) and coherent with the displayed expected points.
    data["finish_residual"] = data["actual_finish"] - data["ranked_finish"]
    data["quali_residual"] = data["actual_quali"] - data["ranked_quali"]

    finish_pools, quali_pools = {}, {}
    tf_all = _pos_tier(data["ranked_finish"].values)
    tq_all = _pos_tier(data["ranked_quali"].values)
    for t in range(4):
        finish_pools[t] = data.loc[tf_all == t, "finish_residual"].dropna().values
        quali_pools[t] = data.loc[tq_all == t, "quali_residual"].dropna().values

    cal_bin = np.clip(np.digitize(data["ranked_finish"].values, POS_BINS[1:-1]), 0, 3)
    dnf_by_bin = np.array([data.loc[cal_bin == b, "dnf_flag"].mean() for b in range(4)])
    dnf_prob_by_pos = np.zeros(GRID)
    for pos in range(1, GRID):
        dnf_prob_by_pos[pos] = dnf_by_bin[np.clip(np.digitize(pos, POS_BINS[1:-1]), 0, 3)]

    # fit frailty over-dispersion phi so simulated per-race DNF-count variance matches the observed
    # variance (a shared per-race shock - see run_race_mc - correlates retirements across the grid)
    dnf_counts = data.groupby(["season", "round"])["dnf_flag"].sum()
    var_n = dnf_counts.var()
    probs20 = dnf_prob_by_pos[1:21]
    rng = np.random.default_rng(0)

    def _sim_dnf_var(phi, n=60000):
        z = np.ones(n) if phi <= 0 else rng.gamma(1.0 / phi, phi, n)
        return float(((rng.random((n, 20)) < np.clip(probs20[None, :] * z[:, None], 0, 1)).sum(axis=1)).var())

    phi_grid = np.linspace(0.0, 0.6, 61)
    frailty_phi = float(min(phi_grid, key=lambda p: abs(_sim_dnf_var(p) - var_n)))

    calibration = {
        "finish_pools": finish_pools,
        "quali_pools": quali_pools,
        "dnf_prob_by_pos": dnf_prob_by_pos,
        "frailty_phi": frailty_phi,
        "overall_dnf": float(data["dnf_flag"].mean()),
        "seasons": CALIBRATION_SEASONS,
        "n_records": len(data),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibration, CALIBRATION_ARTIFACT)
    return calibration


def load_calibration():
    if not CALIBRATION_ARTIFACT.exists():
        return build_calibration()
    return joblib.load(CALIBRATION_ARTIFACT)


# simulates one race n_sims times, centred on the ranked (integer) predicted positions. race is a
# DataFrame with driver_id, constructor_id, ranked_finish, ranked_quali, location, season (one row per
# driver on the current grid). returns (totals, driver_ids, quali_pos, dnf): totals is the
# (n_sims, n_drivers) fantasy-points matrix, quali_pos the sampled qualifying positions and dnf the
# boolean retirement mask (both n_sims x n_drivers, used by callers for constructor bonus and DNF risk)
def run_race_mc(race, calibration, n_sims=1000, rng=None, dnf_rate=None):
    rng = rng or np.random.default_rng()
    finish_pools = calibration["finish_pools"]
    quali_pools = calibration["quali_pools"]
    dnf_prob_by_pos = calibration["dnf_prob_by_pos"]
    frailty_phi = calibration["frailty_phi"]

    predict_overtakes = build_overtake_predictor()
    predict_dotd = build_dotd_predictor()

    race = race.reset_index(drop=True)
    n = len(race)
    # sims are centred on the RANKED (integer) predicted positions and perturbed by the empirical
    # actual-minus-ranked residuals - see build_calibration for why raw scores are not used
    ranked_f = race["ranked_finish"].values.astype(float)
    ranked_q = race["ranked_quali"].values.astype(float)
    # per-driver DNF probability - a caller (simulate_round) may pass a blended driver-specific rate;
    # otherwise fall back to the position-tier prior
    if dnf_rate is None:
        dnf_rate = dnf_prob_by_pos[np.clip(race["ranked_finish"].values, 1, GRID - 1)]
    else:
        dnf_rate = np.asarray(dnf_rate, dtype=float)
    tf = _pos_tier(race["ranked_finish"].values)
    tq = _pos_tier(race["ranked_quali"].values)

    resid_f = np.empty((n_sims, n))
    resid_q = np.empty((n_sims, n))
    for i in range(n):
        resid_f[:, i] = rng.choice(finish_pools[tf[i]], size=n_sims)
        resid_q[:, i] = rng.choice(quali_pools[tq[i]], size=n_sims)

    latent_f = ranked_f[None, :] + resid_f
    latent_q = ranked_q[None, :] + resid_q

    # shared per-race Gamma(mean 1, var frailty_phi) shock correlates DNF odds across the grid
    if frailty_phi > 0:
        z = rng.gamma(1.0 / frailty_phi, frailty_phi, size=n_sims)
        dnf_rate_sim = np.clip(dnf_rate[None, :] * z[:, None], 0.0, 1.0)
    else:
        dnf_rate_sim = np.broadcast_to(dnf_rate[None, :], (n_sims, n))
    dnf = rng.random((n_sims, n)) < dnf_rate_sim

    # DNF-aware finish ranking: survivors first (small key), retirees pushed to the back
    BIG = 1e6
    finish_pos = _ranks(latent_f + BIG * dnf)
    quali_pos = _ranks(latent_q)
    positions_gained = quali_pos - finish_pos

    # fastest lap: weighted single winner among survivors, weighted by QUALIFYING position
    # (matches compose_drivers, which derives fastest-lap probability from quali position)
    fl_w = FL_PROB_ARR[quali_pos] * (~dnf)
    g = -np.log(-np.log(rng.random((n_sims, n)) + 1e-12) + 1e-12)
    fl_winner = np.argmax(np.log(fl_w + 1e-12) + g, axis=1)
    fl_flag = np.zeros((n_sims, n), dtype=bool)
    fl_flag[np.arange(n_sims), fl_winner] = True

    # driver of the day: weighted single winner across the field
    dotd_p = np.asarray(predict_dotd(race["driver_id"])).astype(float)
    dotd_p = dotd_p / dotd_p.sum() if dotd_p.sum() > 0 else np.full(n, 1.0 / n)
    g2 = -np.log(-np.log(rng.random((n_sims, n)) + 1e-12) + 1e-12)
    dotd_winner = np.argmax(np.log(dotd_p[None, :] + 1e-12) + g2, axis=1)
    dotd_flag = np.zeros((n_sims, n), dtype=bool)
    dotd_flag[np.arange(n_sims), dotd_winner] = True

    # overtakes: Poisson around the predictor's expectation at the ranked quali slot
    exp_ot = np.array([
        predict_overtakes(race.loc[i, "driver_id"], race.loc[i, "location"],
                          int(race.loc[i, "season"]), int(race.loc[i, "ranked_quali"]))
        for i in range(n)
    ])
    exp_ot = np.clip(np.nan_to_num(exp_ot, nan=0.0), 0, None)
    overtakes = rng.poisson(exp_ot[None, :] * np.ones((n_sims, 1)))

    race_base = np.where(dnf, RACE_PENALTY, RACE_PTS[finish_pos] + positions_gained * POSITION_GAINED_POINTS)
    race_total = (race_base + fl_flag * FASTEST_LAP_POINTS + dotd_flag * DOTD_POINTS
                  + overtakes * OVERTAKE_MADE_POINTS)
    quali_total = QUALI_PTS[quali_pos]
    total = quali_total + race_total  # (n_sims, n)

    # sprint weekends: add the sprint component. sprint qualifying position stands in for sprint finish
    # (mirrors compose_drivers), so this is deterministic per driver - broadcast across sims as an offset
    if "sprint_quali_position" in race.columns and race["sprint_quali_position"].notna().any():
        sq = race["sprint_quali_position"].fillna(20).astype(int).values
        sprint_finish = np.array([DRIVER_SPRINT_POSITION_POINTS.get(int(p), 0) for p in sq], dtype=float)
        sprint_fl = FL_PROB_ARR[np.clip(sq, 0, GRID - 1)]
        sprint_pts = (sprint_finish + sprint_fl * SPRINT_FASTEST_LAP_POINTS
                      + (exp_ot / 3.0) * SPRINT_OVERTAKE_MADE_POINTS)
        total = total + sprint_pts[None, :]

    return total, race["driver_id"].values, quali_pos, dnf


def _quantiles(totals):
    return {
        "p10": float(np.percentile(totals, 10)),
        "p25": float(np.percentile(totals, 25)),
        "median": float(np.median(totals)),
        "p75": float(np.percentile(totals, 75)),
        "p90": float(np.percentile(totals, 90)),
    }


DNF_RECENT_WINDOW = 30   # trailing calendar races used for recent DNF rates (~1.3 seasons)
DNF_BLEND_K = 8          # shrinkage strength: a driver's own rate gets weight n / (n + K)


# per-driver retirement rates from the most recent races before (season, round_num), from the reliable
# status-based dnf_flag. returns (driver_rates {driver_id: (rate, n_races)}, field_wide_rate). recency is
# deliberate: reliability has changed a lot since 2018, so a driver's own recent record and the current
# field-wide rate are better than the frozen 2018-23 tier curve.
def _recent_dnf(season, round_num, window=DNF_RECENT_WINDOW):
    races = []
    for f in INTERIM_RACES_DIR.glob("*.parquet"):
        try:
            s, r = map(int, f.stem.split("_"))
        except ValueError:
            continue
        if (s, r) < (season, round_num):
            races.append((s, r, f))
    races.sort()
    races = races[-window:]
    if not races:
        return {}, None
    df = pd.concat([pd.read_parquet(f) for _, _, f in races])
    driver_rates = {d: (float(v.mean()), int(v.count())) for d, v in df.groupby("driver_id")["dnf_flag"]}
    return driver_rates, float(df["dnf_flag"].mean())


# blends each driver's recent retirement rate with the position-tier prior (whose LEVEL is rescaled to the
# recent field-wide rate so it isn't anchored to stale 2018-23 reliability). shrinkage weights by how many
# recent races the driver has, so a rookie falls back to the tier prior. returns a per-driver array aligned
# to race row order. display only.
def _blended_dnf(race, calibration, season, round_num):
    driver_rates, recent_overall = _recent_dnf(season, round_num)
    cal_overall = calibration.get("overall_dnf") or float(np.mean(calibration["dnf_prob_by_pos"][1:21]))
    level = (recent_overall / cal_overall) if (recent_overall and cal_overall) else 1.0
    prior = np.clip(calibration["dnf_prob_by_pos"][np.clip(race["ranked_finish"].values, 1, GRID - 1)] * level, 0.0, 1.0)
    out = np.empty(len(race))
    for i, did in enumerate(race["driver_id"].values):
        rate, n = driver_rates.get(did, (float(prior[i]), 0))
        w = n / (n + DNF_BLEND_K)
        out[i] = float(np.clip(w * rate + (1 - w) * prior[i], 0.0, 1.0))
    return out


# simulates the given round from the CURRENT (live) round's raw model outputs.
# returns per-asset quantiles for display PLUS the raw (n_sims x n_drivers) matrix and the driver_id
# order it corresponds to - callers that need a JOINT distribution for an arbitrary selected lineup
# (e.g. a team's "likely range", which is not just the sum of per-driver quantiles - captains double,
# and correlation between a driver and their constructor matters) sum the raw columns themselves
# rather than resimulating. display only - does not feed the optimiser
def simulate_round(season, round_num, location, n_sims=10000, seed=42):
    calibration = load_calibration()
    rng = np.random.default_rng(seed)

    quali_model = load_season_model(QUALI_POSITION_MODEL, season)
    finish_model = load_season_model(FINISH_POSITION_MODEL, season)
    raw_preds = predict_with_raw(quali_model, QUALI_POSITION_MODEL, finish_model, FINISH_POSITION_MODEL, season, round_num)

    race = raw_preds.rename(columns={
        "predicted_quali_position": "ranked_quali",
        "predicted_finish_position": "ranked_finish",
    })
    race["season"] = season
    race["location"] = location

    dnf_rate = _blended_dnf(race, calibration, season, round_num)
    totals, driver_ids, quali_pos, dnf = run_race_mc(race, calibration, n_sims=n_sims, rng=rng, dnf_rate=dnf_rate)

    dnf_prob = dnf.mean(axis=0)
    driver_dist = {
        did: {**_quantiles(totals[:, i]), "dnf_prob": float(dnf_prob[i])}
        for i, did in enumerate(driver_ids)
    }

    # constructor per-sim points = both drivers' points + Q2/Q3 qualifying bonus (computed per sim from
    # the sampled quali positions) + expected pitstop bracket points (deterministic) - matches compose_constructor
    col = {did: i for i, did in enumerate(driver_ids)}
    members = {}
    constructor_of = race.set_index("driver_id")["constructor_id"].to_dict()
    for did in driver_ids:
        members.setdefault(constructor_of[did], []).append(col[did])

    pitstop_pts = expected_pitstop_points(season, round_num)
    q2_cutoff = (len(driver_ids) + 10) // 2
    constructor_ids, constructor_cols = [], []
    for cid, cols in members.items():
        driver_sum = totals[:, cols].sum(axis=1)
        q2 = (quali_pos[:, cols] <= q2_cutoff).sum(axis=1)
        q3 = (quali_pos[:, cols] <= 10).sum(axis=1)
        bonus = np.array([CONSTRUCTOR_QUALI_BONUS.get((int(a), int(b)), 0) for a, b in zip(q2, q3)], dtype=float)
        constructor_ids.append(cid)
        constructor_cols.append(driver_sum + bonus + float(pitstop_pts.get(cid, 0.0)))
    constructor_totals = np.column_stack(constructor_cols) if constructor_cols else np.zeros((n_sims, 0))
    constructor_dist = {cid: _quantiles(constructor_totals[:, i]) for i, cid in enumerate(constructor_ids)}

    return {
        "drivers": driver_dist,
        "constructors": constructor_dist,
        "raw_totals": totals.tolist(),
        "raw_driver_ids": list(driver_ids),
        "raw_constructor_totals": constructor_totals.tolist(),
        "raw_constructor_ids": constructor_ids,
    }


# sums the raw per-sim matrices for an arbitrary selected lineup (captain doubled) into a joint team
# points distribution. sim is simulate_round()'s output - no resimulation needed, so this is cheap
# enough to call on every budget/squad interaction. constructors use their full per-sim points
# (drivers + quali bonus + pitstop), so the team range matches the displayed constructor figures
def team_distribution(sim, selected_drivers, captain, selected_constructors):
    totals = np.asarray(sim["raw_totals"])
    driver_idx = {did: i for i, did in enumerate(sim["raw_driver_ids"])}
    cons_totals = np.asarray(sim["raw_constructor_totals"])
    cons_idx = {cid: i for i, cid in enumerate(sim["raw_constructor_ids"])}

    team_sim = np.zeros(totals.shape[0])
    for d in selected_drivers:
        if d in driver_idx:
            team_sim += (2 if d == captain else 1) * totals[:, driver_idx[d]]
    for c in selected_constructors:
        if c in cons_idx:
            team_sim += cons_totals[:, cons_idx[c]]

    return _quantiles(team_sim)