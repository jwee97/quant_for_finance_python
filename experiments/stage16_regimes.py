"""Stage 16 - Regime detection (Generation 2, Priority 1).

Questions, in the order they can be trusted:

1. DETECTION. Do hidden-Markov, mixture and rule-based regimes carry information
   about the NEXT day, once the persistence of both the labels and the returns
   is respected (circular-shift permutation tests, Benjamini-Hochberg control)?
2. CHANGE POINTS. Does Bayesian online change-point detection flag dated
   volatility shocks quickly, and at what false-alarm cost, against a simple
   volatility-jump rule?
3. STRATEGIES. Do three regime-aware rules (a de-risking overlay, a blend of
   two books, a gate learned on training data) beat the static books they are
   built on, after costs, on a paired bootstrap?
4. THE TRAP. How much would a backtest on SMOOTHED probabilities overstate, and
   does the automated look-ahead test catch it while passing the honest rules?

Everything tradable uses filtered probabilities and parameters estimated only
on data before each refit date. Decision rules were committed to
``config/regimes.yaml`` before any of this was computed.

Figures 30-33.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.execution import vol_target_scaling
from src.backtest.metrics import performance_summary
from src.data.macro import ensure_macro_raw
from src.features.macro import macro_feature_panel
from src.features.regime_rules import (
    bear_market_state,
    inflation_shock,
    liquidity_crisis,
    volatility_regime,
)
from src.features.sleeves import sleeve_returns
from src.models.regimes import (
    alarm_episodes,
    bocpd,
    evaluate_detector,
    full_sample_hmm,
    naive_shock_detector,
    walk_forward_gmm,
    walk_forward_hmm,
)
from src.portfolio.regime_aware import (
    blend_books,
    derisk_overlay,
    full_sample_gate,
    gate_book,
    training_gate,
)
from src.utils.dates import slice_dates
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from src.validation.leakage import check_no_lookahead
from src.validation.permutation import circular_shift_test, lagged_pair, shuffle_test
from src.validation.robustness import multiple_testing_penalty, paired_sharpe_test
from experiments.context import build_context
from experiments.strategies import build_ladder

STAGE = "stage16_regimes"
LAG_NOTE = "state at the close of t against the return on t+1"


# ---------------------------------------------------------------------------
# Inputs and model fitting
# ---------------------------------------------------------------------------
def hmm_settings(cfg) -> dict:
    node = cfg.get("regimes.detection.hmm", {}) or {}
    walk = cfg.get("regimes.detection.walk_forward", {}) or {}
    return {
        "min_train": int(walk.get("min_train_days", 750)),
        "refit_every": int(walk.get("refit_every", 126)),
        "n_init": int(node.get("n_init", 5)),
        "n_init_refit": int(node.get("n_init_refit", 1)),
        "n_iter": int(node.get("n_iter", 300)),
        "tol": float(node.get("tol", 1e-4)),
        "seed": int(node.get("seed", 11)),
    }


def emission_matrix(cfg, returns: pd.DataFrame, investable: pd.DataFrame) -> pd.DataFrame:
    """Daily sleeve returns in PERCENT (the library's covariance floor assumes that scale)."""
    sleeves = {k: list(v) for k, v in (cfg.get("regimes.detection.sleeves", {}) or {}).items()}
    return (sleeve_returns(returns, sleeves, investable).dropna() * 100.0)


def fit_regime_models(cfg, X: pd.DataFrame, cache_dir, logger) -> dict:
    """Walk-forward HMMs (2 and 3 states), GMMs, and the full-sample look-ahead controls.

    Cached on disk under a digest of the data and settings. The cache also keeps
    every individual HMM fit, which lets the look-ahead test skip refits whose
    training data are untouched by a perturbation of the future.
    """
    settings = hmm_settings(cfg)
    gmm_node = cfg.get("regimes.detection.gmm", {}) or {}
    hmm_node = cfg.get("regimes.detection.hmm", {}) or {}
    key_source = json.dumps({"settings": settings, "gmm": gmm_node, "hmm": hmm_node}, sort_keys=True, default=str)
    digest = hashlib.blake2b(np.ascontiguousarray(X.to_numpy()).tobytes() + key_source.encode(),
                             digest_size=8).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"regime_models_{digest}.pkl"
    if path.exists():
        logger.info("regime models: using the cache %s", path.name)
        return pickle.loads(path.read_bytes())

    fit_cache: dict = {}
    n_primary = int(hmm_node.get("n_states_primary", 2))
    n_sensitivity = int(hmm_node.get("n_states_sensitivity", 3))
    out: dict = {"fit_cache": fit_cache, "settings": settings, "n_primary": n_primary,
                 "n_sensitivity": n_sensitivity}
    for name, k in (("hmm_primary", n_primary), ("hmm_sensitivity", n_sensitivity)):
        logger.info("walk-forward HMM, %d states (refit every %d days, first fit on %d days)",
                    k, settings["refit_every"], settings["min_train"])
        out[name] = walk_forward_hmm(X, n_states=k, cache=fit_cache, **settings)
    logger.info("full-sample HMM controls (look-ahead by design)")
    out["full_primary"] = full_sample_hmm(X, n_primary, settings["n_init"], settings["n_iter"],
                                          settings["tol"], settings["seed"])
    components = [int(c) for c in gmm_node.get("n_components", [2, 3])]
    for c in components:
        out[f"gmm_{c}"] = walk_forward_gmm(X, c, settings["min_train"], settings["refit_every"],
                                           int(gmm_node.get("n_init", 5)), settings["seed"])
    path.write_bytes(pickle.dumps(out))
    return out


def hmm_refit_table(result, label: str) -> pd.DataFrame:
    rows = []
    for w in result.windows:
        fit = w["fit"]
        diagonal = np.diag(fit.transmat)
        rows.append({
            "model": label, "refit_date": w["date"].date().isoformat(), "train_days": w["start"],
            "loglik": fit.loglik, "converged": fit.converged,
            **{f"stay_prob_state_{k}": float(diagonal[k]) for k in range(len(diagonal))},
            **{f"mean_duration_state_{k}": float(1.0 / max(1.0 - diagonal[k], 1e-9)) for k in range(len(diagonal))},
            **{f"equity_vol_state_{k}": float(np.sqrt(fit.covars[k][0, 0] * 252) / 100.0) for k in range(len(diagonal))},
            **{f"equity_mean_ann_state_{k}": float(fit.means[k][0] * 252 / 100.0) for k in range(len(diagonal))},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime variables
# ---------------------------------------------------------------------------
def rule_regimes(cfg, market, returns: pd.DataFrame, cpi_yoy: pd.Series) -> dict[str, pd.DataFrame]:
    rules = cfg.get("regimes.rules", {}) or {}
    bear_cfg = rules.get("bear", {}) or {}
    vol_cfg = rules.get("volatility", {}) or {}
    infl_cfg = rules.get("inflation_shock", {}) or {}
    liq_cfg = rules.get("liquidity_crisis", {}) or {}
    index_name = str(bear_cfg.get("index", "SPY"))
    bear = bear_market_state(market.prices[index_name], float(bear_cfg.get("drawdown", 0.20)),
                             float(bear_cfg.get("recovery", 0.20)))
    vol = volatility_regime(returns[index_name], float(vol_cfg.get("halflife", 21)),
                            float(vol_cfg.get("high_percentile", 0.80)),
                            float(vol_cfg.get("low_percentile", 0.20)), int(vol_cfg.get("min_history", 252)))
    inflation = inflation_shock(cpi_yoy, returns[index_name], returns["IEF"],
                                float(infl_cfg.get("cpi_yoy_min", 0.04)),
                                float(infl_cfg.get("stock_bond_corr_min", 0.0)),
                                int(infl_cfg.get("corr_window", 126)))
    credit, safe = [str(t) for t in liq_cfg.get("credit_pair", ["HYG", "IEF"])]
    liquidity = liquidity_crisis(vol["percentile"], returns[credit], returns[safe],
                                 float(liq_cfg.get("vol_percentile_min", 0.90)),
                                 int(liq_cfg.get("credit_window", 21)),
                                 float(liq_cfg.get("credit_max", -0.05)))
    return {"bear": bear.to_frame("bear"), "volatility": vol, "inflation": inflation, "liquidity": liquidity}


def regime_flags(models: dict, rules: dict) -> pd.DataFrame:
    """Every regime variable as a 0/1 flag known at the close of day t (NaN if undefined)."""
    def high(prob: pd.DataFrame) -> pd.Series:
        valid = prob.notna().all(axis=1)
        return (prob.iloc[:, -1] > 0.5).astype(float).where(valid)

    flags = pd.DataFrame({
        "hmm2_high_vol": high(models["hmm_primary"].prob),
        "hmm3_high_vol": (models["hmm_sensitivity"].prob.iloc[:, -1]
                          == models["hmm_sensitivity"].prob.max(axis=1)).astype(float)
        .where(models["hmm_sensitivity"].prob.notna().all(axis=1)),
        "gmm2_high_vol": high(models["gmm_2"]),
        "bear_market": rules["bear"]["bear"],
        "vol_high": rules["volatility"]["high"],
        "vol_low": rules["volatility"]["low"],
        "inflation_shock": rules["inflation"]["shock"],
        "liquidity_crisis": rules["liquidity"]["crisis"],
    })
    return flags


FLAG_LABELS = {
    "hmm2_high_vol": "HMM (2 states): high-vol",
    "hmm3_high_vol": "HMM (3 states): high-vol",
    "gmm2_high_vol": "Mixture (2): high-vol",
    "bear_market": "Bear market (-20%)",
    "vol_high": "Volatility, top quintile",
    "vol_low": "Volatility, bottom quintile",
    "inflation_shock": "Inflation shock",
    "liquidity_crisis": "Liquidity crisis",
}


# ---------------------------------------------------------------------------
# Conditional statistics and permutation tests
# ---------------------------------------------------------------------------
def conditional_table(flags: pd.DataFrame, sleeve_daily: pd.DataFrame, bond_stock: pd.DataFrame,
                      span: slice, min_days: int) -> pd.DataFrame:
    """Annualised return, volatility and Sharpe inside and outside each regime (same-day, descriptive)."""
    rows = []
    daily = sleeve_daily.loc[span]
    for name in flags.columns:
        flag = flags[name].reindex(daily.index)
        valid = flag.notna()
        inside = valid & (flag > 0.5)
        outside = valid & (flag <= 0.5)
        for sleeve in daily.columns:
            r = daily[sleeve]
            for label, mask in (("inside", inside), ("outside", outside)):
                x = r[mask].dropna()
                rows.append({
                    "regime": name, "sleeve": sleeve, "where": label, "n_days": int(len(x)),
                    "ann_return": float(x.mean() * 252) if len(x) else np.nan,
                    "ann_vol": float(x.std(ddof=1) * np.sqrt(252)) if len(x) > 1 else np.nan,
                    "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan,
                    "enough_days": bool(len(x) >= min_days),
                })
        for label, mask in (("inside", inside), ("outside", outside)):
            pair = bond_stock.loc[span][mask.reindex(bond_stock.loc[span].index, fill_value=False)].dropna()
            rows.append({"regime": name, "sleeve": "stock_bond_corr", "where": label,
                         "n_days": int(len(pair)),
                         "ann_return": float(pair["stock"].corr(pair["bond"])) if len(pair) > 2 else np.nan,
                         "ann_vol": np.nan, "sharpe": np.nan, "enough_days": bool(len(pair) >= min_days)})
    return pd.DataFrame(rows)


def run_permutation_family(flags: pd.DataFrame, sleeve_daily: pd.DataFrame, span: slice,
                           analysis: dict, fdr: float) -> pd.DataFrame:
    """Every regime against next-day mean returns of each sleeve and next-day |equity return|.

    The primary p-value is the studentised circular-shift test; the plain
    difference-of-means version and the naive shuffle are kept alongside so the
    effect of each correction can be read off the table.
    """
    permutation = analysis.get("permutation", {}) or {}
    n_shifts = int(permutation.get("n_shifts", 5000))
    min_shift = int(permutation.get("min_shift_days", 63))
    seed = int(permutation.get("seed", 7))
    min_days = int(analysis.get("min_days_per_regime", 60))
    outcomes = {f"{sleeve}_next_day_mean": sleeve_daily[sleeve] for sleeve in sleeve_daily.columns}
    outcomes["equity_next_day_abs"] = sleeve_daily["equity"].abs()
    rows = []
    for name in flags.columns:
        flag = flags[name].loc[span]
        for outcome_name, outcome in outcomes.items():
            f, o = lagged_pair(flag.dropna(), outcome.loc[span], lag=1)
            result = circular_shift_test(o.to_numpy(), f.to_numpy(), n_shifts, min_shift, seed)
            row = {"regime": name, "outcome": outcome_name, "n_pairs": int(len(f))}
            if not result or result["n_in"] < min_days or result["n_out"] < min_days:
                row.update({"tested": False, "n_in": int(f.sum()), "n_out": int(len(f) - f.sum())})
            else:
                naive = shuffle_test(o.to_numpy(), f.to_numpy(), 500, seed)
                row.update({"tested": True,
                            **{k: result[k] for k in ("statistic", "t_statistic", "p_value",
                                                      "p_value_difference", "n_in", "n_out",
                                                      "mean_in", "mean_out")},
                            "naive_shuffle_p_value": naive["p_value"]})
            rows.append(row)
    table = pd.DataFrame(rows)
    tested = table["tested"].fillna(False).astype(bool)
    table["bh_significant"] = False
    table["bh_significant_plain_difference"] = False
    if tested.any():
        table.loc[tested, "bh_significant"] = benjamini_hochberg(table.loc[tested, "p_value"], fdr)
        table.loc[tested, "bh_significant_plain_difference"] = benjamini_hochberg(
            table.loc[tested, "p_value_difference"], fdr)
    table["significant_raw_5pct"] = tested & (table["p_value"] < 0.05)
    table["naive_significant_5pct"] = tested & (table["naive_shuffle_p_value"] < 0.05)
    return table


def _markov_flag(T: int, rng, p_enter: float, p_stay: float) -> np.ndarray:
    flag = np.zeros(T, dtype=bool)
    flag[0] = rng.random() < p_enter / max(p_enter + 1.0 - p_stay, 1e-12)
    for t in range(1, T):
        flag[t] = (rng.random() < p_stay) if flag[t - 1] else (rng.random() < p_enter)
    return flag


def size_demonstration(n_reps: int = 150, seed: int = 5) -> dict:
    """Rejection rates of the tests when NO relationship exists.

    Scenario A: persistent random labels against persistent random outcomes.
    Only a test that respects persistence holds its level; shuffling does not.

    Scenario B: a rare, much more volatile regime (the shape of a crisis) with
    the mean return identical in and out of it. The plain difference of means
    is badly oversized; the studentised statistic is not.
    """
    rng = np.random.default_rng(seed)
    a_circular, a_naive = [], []
    for rep in range(n_reps):
        flag = _markov_flag(1200, rng, 0.015, 0.985)
        shocks = rng.standard_normal(1200)
        outcome = np.empty(1200)
        outcome[0] = 0.0
        for t in range(1, 1200):
            outcome[t] = 0.98 * outcome[t - 1] + 0.2 * shocks[t]
        a_circular.append(circular_shift_test(outcome[1:], flag[:-1], 400, 40, rep)["p_value"])
        a_naive.append(shuffle_test(outcome[1:], flag[:-1], 200, rep)["p_value"])
    b_plain, b_student = [], []
    for rep in range(n_reps):
        flag = _markov_flag(1500, rng, 0.01, 0.95)
        if flag.sum() < 30:
            continue
        returns = np.where(flag, 3.0, 0.5) * rng.standard_normal(1500)
        result = circular_shift_test(returns[1:], flag[:-1], 400, 40, rep)
        b_plain.append(result["p_value_difference"])
        b_student.append(result["p_value"])
    rate = lambda p: float(np.mean(np.asarray(p) < 0.05))
    summary = pd.DataFrame([
        {"scenario": "persistent label, persistent outcome", "test": "naive shuffle", "rejection_rate_5pct": rate(a_naive)},
        {"scenario": "persistent label, persistent outcome", "test": "circular shift", "rejection_rate_5pct": rate(a_circular)},
        {"scenario": "rare high-variance regime, no mean effect", "test": "circular shift, plain difference", "rejection_rate_5pct": rate(b_plain)},
        {"scenario": "rare high-variance regime, no mean effect", "test": "circular shift, studentised", "rejection_rate_5pct": rate(b_student)},
    ])
    return {"summary": summary, "persistence": pd.DataFrame({"circular_shift": a_circular, "naive_shuffle": a_naive}),
            "heteroskedastic": pd.DataFrame({"plain_difference": b_plain, "studentised": b_student})}


# ---------------------------------------------------------------------------
# Change-point study
# ---------------------------------------------------------------------------
def bocpd_study(cfg, equity_pct: pd.Series) -> dict:
    """BOCPD against dated shocks, with the naive volatility-jump rule beside it."""
    node = cfg.get("regimes.bocpd", {}) or {}
    decisions = cfg.get("regimes.decisions", {}) or {}
    merge_gap = int((decisions.get("h_bocpd", {}) or {}).get("merge_gap_days", 5))
    naive_cfg = decisions.get("naive_detector", {}) or {}
    short_run = int(node.get("short_run_days", 10))
    prior = node.get("prior", {}) or {}
    result = bocpd(equity_pct, hazard_lambda=float(node.get("hazard_lambda", 250)),
                   burn_in=int(node.get("burn_in_days", 250)), kappa0=float(prior.get("kappa0", 1.0)),
                   alpha0=float(prior.get("alpha0", 1.0)), short_runs=(short_run,))
    mass = result.short_mass[f"le_{short_run}"]
    events = {k: str(v) for k, v in (node.get("events", {}) or {}).items()}
    search = int(node.get("search_days", 60))
    threshold = float(node.get("alarm_threshold", 0.5))

    def score_bocpd(thr: float) -> dict:
        return evaluate_detector(mass >= thr, events, search, merge_gap)

    def naive_alarm(ratio: float) -> pd.Series:
        alarm = naive_shock_detector(equity_pct, int(naive_cfg.get("short_window", 5)),
                                     int(naive_cfg.get("long_window", 250)), ratio)
        return alarm.reindex(mass.index).fillna(False)

    primary = score_bocpd(threshold)
    naive_ratio = float(naive_cfg.get("ratio", 2.0))
    naive_primary = evaluate_detector(naive_alarm(naive_ratio), events, search, merge_gap)
    pre_days = 5
    primary_lenient = lenient_detection(mass >= threshold, events, pre_days, search, merge_gap)
    naive_lenient = lenient_detection(naive_alarm(naive_ratio), events, pre_days, search, merge_gap)
    sweep = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        strict = score_bocpd(thr)
        lenient = lenient_detection(mass >= thr, events, pre_days, search, merge_gap)
        sweep.append({"detector": "BOCPD", "setting": thr, "n_detected": strict["n_detected"],
                      "mean_delay_days": strict["mean_delay_days"],
                      "false_alarms_per_year": strict["false_alarms_per_year"],
                      "n_detected_lenient": lenient["n_detected"],
                      "false_alarms_per_year_lenient": lenient["false_alarms_per_year"]})
    for ratio in (1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
        strict = evaluate_detector(naive_alarm(ratio), events, search, merge_gap)
        lenient = lenient_detection(naive_alarm(ratio), events, pre_days, search, merge_gap)
        sweep.append({"detector": "naive volatility jump", "setting": ratio, "n_detected": strict["n_detected"],
                      "mean_delay_days": strict["mean_delay_days"],
                      "false_alarms_per_year": strict["false_alarms_per_year"],
                      "n_detected_lenient": lenient["n_detected"],
                      "false_alarms_per_year_lenient": lenient["false_alarms_per_year"]})
    return {"result": result, "mass": mass, "threshold": threshold, "short_run": short_run,
            "events": events, "search_days": search, "pre_days": pre_days, "primary": primary,
            "naive": naive_primary, "primary_lenient": primary_lenient, "naive_lenient": naive_lenient,
            "naive_alarm": naive_alarm(naive_ratio), "sweep": pd.DataFrame(sweep)}


# ---------------------------------------------------------------------------
# Strategy factory: ONE code path for the reported strategies and the tested ones
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _State:
    returns: pd.DataFrame
    X: pd.DataFrame
    hmm2: object
    hmm3: object
    full: dict
    base: dict
    vol_high: pd.Series
    mom_scaled: pd.DataFrame
    mom_net: pd.Series
    gates: dict
    gate_log: dict
    probabilities: dict


class RegimeStrategyFactory:
    """Builds every regime-aware book from a ``MarketData``.

    The leakage test needs ``market -> weights`` functions, and the strategies
    reported in the study must be the SAME functions, or the test certifies
    something other than what was run. State per market is memoised by object
    identity (the market objects are kept alive so identities are not reused).
    """

    def __init__(self, cfg, engine: BacktestEngine, fit_cache: dict, ladder_provider):
        self.cfg, self.engine, self.fit_cache = cfg, engine, fit_cache
        self.ladder_provider = ladder_provider
        self._memo: dict[int, tuple[object, _State]] = {}
        strategies = cfg.get("regimes.strategies", {}) or {}
        self.cash = str(strategies.get("cash", "SHY"))
        self.derisk_max = float(strategies.get("derisk_max", 0.5))
        self.gate_min_days = int(strategies.get("gate_min_days", 60))

    def scaled_momentum(self, market, momentum: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """The momentum book after the engine's own masking and volatility targeting."""
        weights = momentum.copy()
        mask = market.investable.reindex_like(weights).fillna(False)
        weights = weights.where(mask, 0.0)
        weights = weights.where(mask.sum(axis=1) >= self.engine.min_assets, 0.0)
        aligned = returns.reindex(weights.index).reindex(columns=weights.columns)
        scaled, _ = vol_target_scaling(weights, aligned, self.engine.target_vol or 0.10,
                                       self.engine.vol_lookback, self.engine.max_leverage,
                                       self.engine.annualisation)
        return scaled

    def state(self, market, models: dict | None = None, controls: bool = True) -> _State:
        key = id(market)
        if key not in self._memo:
            self._memo[key] = (market, self._honest_state(market, models))
        state = self._memo[key][1]
        if controls and "|smoothed" not in state.probabilities:
            self._add_controls(market, state, models)
        return state

    def _honest_state(self, market, models: dict | None) -> _State:
        cfg, settings = self.cfg, hmm_settings(self.cfg)
        returns = market.returns()
        X = emission_matrix(cfg, returns, market.investable)
        if models is None:
            hmm2 = walk_forward_hmm(X, n_states=2, cache=self.fit_cache, **settings)
            hmm3 = None
        else:
            hmm2, hmm3 = models["hmm_primary"], models["hmm_sensitivity"]
        base = self.ladder_provider(market)
        mom_scaled = self.scaled_momentum(market, base["M3_momentum"], returns)
        mom_run = self.engine.run(mom_scaled, returns, "momentum_ungated", market.investable,
                                  apply_vol_target=False)
        mom_net = mom_run.net_returns
        horizon = self.engine.signal_lag + 1
        gates, logs, probs = {}, {}, {}
        for suffix, model in (("", hmm2), ("_3state", hmm3)):
            if model is None:
                continue
            gates[suffix], logs[suffix] = training_gate(mom_net, model.windows, X.index,
                                                        self.gate_min_days, horizon)
            probs[suffix] = model.p_high
        rules = cfg.get("regimes.rules.volatility", {}) or {}
        vol = volatility_regime(returns["SPY"], float(rules.get("halflife", 21)),
                                float(rules.get("high_percentile", 0.80)),
                                float(rules.get("low_percentile", 0.20)),
                                int(rules.get("min_history", 252)))
        return _State(returns=returns, X=X, hmm2=hmm2, hmm3=hmm3, full=None, base=base,
                      vol_high=vol["high"], mom_scaled=mom_scaled, mom_net=mom_net, gates=gates,
                      gate_log=logs, probabilities=probs)

    def _add_controls(self, market, state: _State, models: dict | None) -> None:
        """The look-ahead controls: one fit on the whole sample, filtered or smoothed."""
        settings = hmm_settings(self.cfg)
        full = (models["full_primary"] if models is not None else
                full_sample_hmm(state.X, 2, settings["n_init"], settings["n_iter"], settings["tol"],
                                settings["seed"]))
        horizon = self.engine.signal_lag + 1
        state.full = full
        for suffix, frame in (("|full_theta", full["filtered_full_theta"]), ("|smoothed", full["smoothed"])):
            state.gates[suffix], state.gate_log[suffix] = full_sample_gate(
                state.mom_net, frame, self.gate_min_days, horizon)
            state.probabilities[suffix] = frame.iloc[:, -1].rename("p_high")

    def weights(self, market, models: dict | None = None, variants: tuple[str, ...] | None = None) -> dict:
        wanted = tuple(variants) if variants is not None else None
        need_controls = wanted is None or any(v.startswith("|") for v in wanted)
        st = self.state(market, models, controls=need_controls)
        index = market.prices.index
        ew, rp, mom = st.base["M0_equal_weight"], st.base["M2_risk_parity"], st.mom_scaled
        out = {"equal_weight": ew, "risk_parity": rp, "momentum": mom,
               "naive_vol_timing": derisk_overlay(ew, st.vol_high.reindex(index), self.cash, self.derisk_max)}
        for suffix in (wanted if wanted is not None else list(st.probabilities)):
            if suffix not in st.probabilities:
                continue
            p = st.probabilities[suffix].reindex(index)
            g = st.gates[suffix].reindex(index)
            out[f"RA1_derisk{suffix}"] = derisk_overlay(ew, p, self.cash, self.derisk_max)
            out[f"RA2_blend{suffix}"] = blend_books(ew, rp, p)
            out[f"RA3_gated_momentum{suffix}"] = gate_book(mom, g)
        return out


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------
def run_books(engine: BacktestEngine, books: dict, returns: pd.DataFrame, investable: pd.DataFrame) -> dict:
    return {name: engine.run(weights, returns, name, investable, apply_vol_target=False)
            for name, weights in books.items()}


def performance_table(results: dict, start: pd.Timestamp) -> pd.DataFrame:
    rows = {}
    for name, result in results.items():
        net = slice_dates(result.net_returns, start, None)
        stats = performance_summary(net, turnover=slice_dates(result.turnover, start, None),
                                    costs=slice_dates(result.costs, start, None))
        stats["ann_turnover"] = float(slice_dates(result.turnover, start, None).sum()
                                      / max(len(net) / 252.0, 1e-9))
        rows[name] = stats
    return pd.DataFrame(rows).T


def strategy_family_tests(streams: dict, family: dict, test_cfg: dict, fdr: float) -> pd.DataFrame:
    """The pre-declared family: each regime-aware rule against the books it is built on."""
    rows = []
    for strategy, benchmarks in family.items():
        for benchmark in benchmarks:
            result = paired_sharpe_test(streams[strategy], streams[benchmark],
                                        n_samples=int(test_cfg.get("n_samples", 2000)),
                                        block_length=int(test_cfg.get("block_length", 21)),
                                        seed=int(test_cfg.get("seed", 7)))
            rows.append({"strategy": strategy, "benchmark": benchmark, **result})
    table = pd.DataFrame(rows)
    table["bh_significant"] = benjamini_hochberg(table["p_value"], fdr)
    table["passes"] = table["bh_significant"] & (table["difference"] > 0)
    return table


# ---------------------------------------------------------------------------
# Look-ahead tests
# ---------------------------------------------------------------------------
LEAKAGE_STRATEGIES = {
    # honest rules: the automated test must PASS them
    "RA1_derisk": "honest", "RA2_blend": "honest", "RA3_gated_momentum": "honest",
    # smoothed / full-sample controls: the test must FAIL them, or it is blind
    "RA1_derisk|smoothed": "control", "RA2_blend|smoothed": "control",
    "RA1_derisk|full_theta": "control", "RA2_blend|full_theta": "control",
}


def leakage_study(cfg, market, factory: RegimeStrategyFactory, models: dict, logger) -> pd.DataFrame:
    """Run the platform's look-ahead test on every regime-aware book and on the smoothed controls.

    The builders are the factory's own functions: the weights that were tested
    are the weights that were backtested. The original market reuses the state
    already computed for the study; each perturbed market is rebuilt from its
    own perturbed returns, with refits whose training data are untouched served
    from the fit cache (the cache is keyed on the data actually passed to the fit).
    """
    node = (cfg.get("regimes.decisions.leakage", {}) or {})
    splits = [str(d) for d in node.get("split_dates", ["2014-06-30", "2019-06-28"])]
    modes = [str(m) for m in node.get("modes", ["shock", "reverse"])]
    factory.state(market, models)                      # seed the memo with the study's own state

    def own(m):
        return models if m is market else None

    def builder_for(name: str, kind: str):
        variants = ("",) if kind == "honest" else ("|" + name.split("|", 1)[1],)

        def build(m):
            return factory.weights(m, own(m), variants=variants)[name]
        return build

    def signal_builder(variant: str):
        def build(m):
            return factory.state(m, own(m), controls=variant != "").probabilities[variant] \
                .reindex(m.prices.index).to_frame("p_high")
        return build

    jobs = [(name, kind, builder_for(name, "honest" if kind == "honest" else "control"))
            for name, kind in LEAKAGE_STRATEGIES.items()]
    jobs += [("P_high (filtered)", "honest", signal_builder("")),
             ("P_high (smoothed)", "control", signal_builder("|smoothed"))]
    rows = []
    for name, kind, build in jobs:
        for split in splits:
            for mode in modes:
                try:
                    outcome = check_no_lookahead(market, build, split, mode=mode)
                    row = outcome.to_row()
                    row["error"] = ""
                except Exception as exc:                     # recorded in the table, and fatal after the loop
                    logger.error("look-ahead test %s %s %s failed to run: %s", name, split, mode, exc)
                    row = {"split_date": split, "passed": False, "max_weight_difference": np.nan,
                           "n_differing_cells": -1, "n_cells_checked": 0, "future_weights_changed": False,
                           "message": f"error: {exc}", "first_difference_date": "", "error": str(exc)}
                row.update({"strategy": name, "kind": kind, "mode": mode})
                rows.append(row)
                logger.info("look-ahead %-24s %-8s %s %-7s -> %s", name, kind, split, mode,
                            "PASS" if row["passed"] else ("ERROR" if row["error"] else "FAIL"))
    frame = pd.DataFrame(rows)
    flagged = (~frame["passed"].astype(bool)) & frame["future_weights_changed"].astype(bool) & (frame["error"] == "")
    frame["as_expected"] = np.where(frame["kind"] == "honest", frame["passed"].astype(bool), flagged)
    if (frame["error"] != "").any():
        raise RuntimeError(f"{int((frame['error'] != '').sum())} look-ahead runs crashed: "
                           f"{frame.loc[frame['error'] != '', 'message'].unique().tolist()}")
    front = ["strategy", "kind", "mode", "split_date", "passed", "as_expected", "max_weight_difference",
             "n_differing_cells", "future_weights_changed", "message"]
    return frame.loc[:, front + [c for c in frame.columns if c not in front]]


def lenient_detection(alarm: pd.Series, events: dict, pre_days: int = 5, search_days: int = 60,
                      merge_gap: int = 5) -> dict:
    """POST-HOC reading: credit an alarm that is active anywhere in [event - pre_days, event + search_days].

    The pre-declared rule only credits an alarm episode that STARTS on or after
    the dated event. Dated events are chronology labels for sell-offs that often
    began a few sessions earlier, so this reading also credits an alarm that fired
    a few days ahead of the date. It is reported beside the strict result and
    never used to change a decision.
    """
    index = alarm.index
    flags = alarm.fillna(False).to_numpy(dtype=bool)
    detected, delays, windows = [], [], []
    for date in events.values():
        position = int(index.searchsorted(pd.Timestamp(date)))
        lo, hi = max(position - pre_days, 0), min(position + search_days, len(index) - 1)
        windows.append((lo, hi))
        hits = np.flatnonzero(flags[lo:hi + 1])
        detected.append(bool(len(hits)))
        delays.append(float(lo + hits[0] - position) if len(hits) else np.nan)
    episodes = alarm_episodes(alarm, merge_gap)
    false = [s for s, e in episodes
             if not any(not (int(index.get_loc(e)) < lo or int(index.get_loc(s)) > hi) for lo, hi in windows)]
    years = max((index[-1] - index[0]).days / 365.25, 1e-9)
    return {"detected": dict(zip(events, detected)), "delay_days": dict(zip(events, delays)),
            "n_detected": int(sum(detected)), "n_false_alarms": len(false),
            "false_alarms_per_year": len(false) / years}


# ---------------------------------------------------------------------------
# Small helpers for the tables and figures
# ---------------------------------------------------------------------------
def spell_lengths(flag: pd.Series) -> np.ndarray:
    """Lengths of the consecutive runs of 1s in a 0/1 flag."""
    values = flag.dropna().to_numpy(dtype=float) > 0.5
    if not values.any():
        return np.array([], dtype=int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], values.astype(int), [0]))))
    return (edges[1::2] - edges[0::2]).astype(int)


def persistence_table(flags: pd.DataFrame, models: dict) -> pd.DataFrame:
    rows = []
    for name in flags.columns:
        flag = flags[name].dropna()
        spells = spell_lengths(flag)
        years = max((flag.index[-1] - flag.index[0]).days / 365.25, 1e-9)
        rows.append({"regime": name, "share_of_days_on": float((flag > 0.5).mean()),
                     "n_spells": int(len(spells)),
                     "mean_spell_days": float(spells.mean()) if len(spells) else np.nan,
                     "median_spell_days": float(np.median(spells)) if len(spells) else np.nan,
                     "longest_spell_days": int(spells.max()) if len(spells) else 0,
                     "spells_per_year": len(spells) / years})
    table = pd.DataFrame(rows).set_index("regime")
    fit = models["full_primary"]["fit"]
    stay = np.diag(fit.transmat)
    table.loc["full-sample HMM (implied)", "mean_spell_days"] = float(1.0 / max(1.0 - stay[-1], 1e-9))
    table.loc["full-sample HMM (implied) calm state", "mean_spell_days"] = float(1.0 / max(1.0 - stay[0], 1e-9))
    return table


def _shade(ax, flag: pd.Series, color: str, alpha: float = 0.25) -> None:
    on = (flag.reindex(flag.index).fillna(0.0) > 0.5).to_numpy()
    if not on.any():
        return
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(int), [0]))))
    for lo, hi in zip(edges[0::2], edges[1::2]):
        ax.axvspan(flag.index[lo], flag.index[min(hi, len(flag) - 1)], color=color, alpha=alpha, linewidth=0)


EVENT_COLOURS = "#555555"


def figure_timeline(X, models, flags, rules, persist, events, path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.5, 11.5))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.42, wspace=0.28,
                            top=0.935, bottom=0.10, left=0.06, right=0.985)
    first = models["hmm_primary"].first_date
    equity = (1.0 + X["equity"] / 100.0).loc[first:].cumprod()
    p_filtered = models["hmm_primary"].p_high
    p_smoothed = models["full_primary"]["smoothed"].iloc[:, -1]
    p_full_theta = models["full_primary"]["filtered_full_theta"].iloc[:, -1]
    p_gmm = models["gmm_2"].iloc[:, -1]

    ax = fig.add_subplot(grid[0, :])
    ax.semilogy(equity.index, equity.to_numpy(), color="#0072B2", linewidth=1.2)
    _shade(ax, flags["hmm2_high_vol"].loc[first:], "#D55E00", 0.28)
    top = ax.get_ylim()[1]
    for name, date in events.items():
        when = pd.Timestamp(date)
        if when < first:
            continue
        ax.axvline(when, color=EVENT_COLOURS, linestyle=":", linewidth=0.9)
        ax.text(when, top, name.replace("_", " "), rotation=90, va="top", ha="right", fontsize=7.5,
                color=EVENT_COLOURS)
    ax.set_ylabel("Equity sleeve, growth of 1 (log)")
    ax.set_title("Equity sleeve with the days the walk-forward HMM calls high-volatility (shaded)")

    ax = fig.add_subplot(grid[1, :])
    ax.plot(p_filtered.dropna().index, p_filtered.dropna(), color="#0072B2", linewidth=1.3,
            label="filtered, walk-forward (tradable)")
    ax.plot(p_smoothed.loc[first:].index, p_smoothed.loc[first:], color="#E69F00", linewidth=0.7,
            alpha=0.95, label="smoothed, full sample (look-ahead)")
    ax.set_ylabel("P(high-volatility state)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.set_title("The same model, two information sets: what was known on the day vs what hindsight says")

    def zoom(position, window, title, legend_loc):
        ax = fig.add_subplot(position)
        ax.plot(p_smoothed.loc[window].index, p_smoothed.loc[window], color="#E69F00", linewidth=1.4,
                label="HMM smoothed")
        ax.plot(p_filtered.loc[window].index, p_filtered.loc[window], color="#0072B2", linewidth=1.7,
                label="HMM filtered")
        ax.plot(p_full_theta.loc[window].index, p_full_theta.loc[window], color="#009E73", linewidth=1.0,
                linestyle="--", label="HMM filtered, full-sample parameters")
        ax.step(p_gmm.loc[window].index, p_gmm.loc[window], where="mid", color="#7F7F7F", linewidth=0.9,
                label="mixture (no chain)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        return ax

    zoom(grid[2, 0], slice("2009-01-02", "2009-06-30"),
         "The 2009 turn: the filter lags, hindsight does not", None)
    zoomed = zoom(grid[2, 1], slice("2020-01-15", "2020-05-29"),
                  "Early 2020: the chain removes the mixture's flicker", None)
    handles, labels = zoomed.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.33, 0.0), ncol=4, fontsize=8, frameon=False)

    ax = fig.add_subplot(grid[2, 2])
    rows = [("HMM, 2 states", "hmm2_high_vol"), ("HMM, 3 states", "hmm3_high_vol"),
            ("Mixture, 2", "gmm2_high_vol"), ("Vol top quintile", "vol_high"),
            ("Bear market", "bear_market"), ("Inflation shock", "inflation_shock"),
            ("Liquidity crisis", "liquidity_crisis")]
    values = [persist.loc[key, "mean_spell_days"] for _, key in rows]
    y = np.arange(len(rows))
    ax.barh(y, values, color=[PALETTE[i % len(PALETTE)] for i in range(len(rows))])
    ax.set_yticks(y, [label for label, _ in rows], fontsize=8.5)
    ax.invert_yaxis()
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:.0f}", va="center", fontsize=8.5)
    ax.set_xlabel("Mean spell in the regime (trading days)", fontsize=9)
    ax.set_title("How long does a regime last?", fontsize=10)
    fig.suptitle("Figure 30. Regimes through time: filtered, smoothed, and rule-based",
                 fontsize=13, fontweight="bold", y=0.985)
    save_figure(fig, path, "What do the regime models say through time, how different is the tradable "
                           "(filtered) answer from the hindsight (smoothed) one, and how long do "
                           "the regimes last?", 30)


def figure_conditional(cond, family, size_summary, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    colours = {"inside": "#D55E00", "outside": "#0072B2"}

    ax = axes[0, 0]
    subset = cond[(cond["regime"] == "hmm2_high_vol") & (cond["sleeve"] != "stock_bond_corr")]
    sleeves = list(subset["sleeve"].unique())
    x = np.arange(len(sleeves))
    for j, where in enumerate(("outside", "inside")):
        values = subset[subset["where"] == where].set_index("sleeve").reindex(sleeves)
        ax.bar(x + (j - 0.5) * 0.36, 100 * values["ann_return"], width=0.36, color=colours[where],
               label=f"{'calm' if where == 'outside' else 'high-vol'}: return")
        for xi, v in zip(x + (j - 0.5) * 0.36, 100 * values["ann_vol"]):
            ax.plot([xi - 0.17, xi + 0.17], [v, v], color="black", linewidth=1.8)
    ax.plot([], [], color="black", linewidth=1.8, label="annualised volatility")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, sleeves)
    ax.set_ylabel("% per year")
    ax.set_title("Same-day returns and risk in the HMM's two states (descriptive)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    corr = cond[cond["sleeve"] == "stock_bond_corr"]
    names = [n for n in FLAG_LABELS if n in set(corr["regime"])]
    y = np.arange(len(names))
    for j, where in enumerate(("outside", "inside")):
        values = corr[corr["where"] == where].set_index("regime").reindex(names)["ann_return"]
        ax.barh(y + (j - 0.5) * 0.38, values, height=0.38, color=colours[where],
                label="outside regime" if where == "outside" else "inside regime")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [FLAG_LABELS[n] + (" (corr >= 0 by definition)" if n == "inflation_shock" else "")
                      for n in names], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Correlation of daily equity and bond returns")
    ax.set_title("Does the diversification between stocks and bonds survive the regime?")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    tested = family.copy()
    outcomes = list(dict.fromkeys(tested["outcome"]))
    regimes = [n for n in FLAG_LABELS if n in set(tested["regime"])]
    grid_t = pd.DataFrame(np.nan, index=regimes, columns=outcomes)
    marks = pd.DataFrame("", index=regimes, columns=outcomes)
    for _, row in tested.iterrows():
        if bool(row["tested"]):
            grid_t.loc[row["regime"], row["outcome"]] = row["t_statistic"]
            marks.loc[row["regime"], row["outcome"]] = "*" if bool(row["bh_significant"]) else ""
        else:
            marks.loc[row["regime"], row["outcome"]] = "n/a"
    image = ax.imshow(grid_t.to_numpy(dtype=float), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(len(outcomes)), [o.replace("_next_day_", "\nnext day: ").replace("_", " ") for o in outcomes], fontsize=8)
    ax.set_yticks(range(len(regimes)), [FLAG_LABELS[r] for r in regimes], fontsize=8.5)
    for i, r in enumerate(regimes):
        for j, o in enumerate(outcomes):
            value = grid_t.loc[r, o]
            text = marks.loc[r, o]
            label = text if text == "n/a" else (f"{value:+.1f}{text}" if np.isfinite(value) else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                    color="white" if np.isfinite(value) and abs(value) > 2.6 else "black",
                    fontweight="bold" if text == "*" else "normal")
    ax.grid(False)
    fig = ax.figure
    fig.colorbar(image, ax=ax, shrink=0.7, label="studentised statistic")
    ax.set_title("Does the state at the close of t predict day t+1?  (* = BH-significant)")

    ax = axes[1, 1]
    labels = [f"{r['test']}" for _, r in size_summary.iterrows()]
    y = np.arange(len(labels))
    colours_bar = ["#CC79A7", "#009E73", "#CC79A7", "#009E73"]
    ax.barh(y, 100 * size_summary["rejection_rate_5pct"], color=colours_bar)
    ax.axvline(5, color="black", linestyle="--", linewidth=1.0)
    ax.text(5.5, -0.45, "nominal 5%", fontsize=8)
    ax.set_yticks(y, [textwrap_label(l, s) for l, s in zip(labels, size_summary["scenario"])], fontsize=8)
    ax.invert_yaxis()
    for yi, v in zip(y, 100 * size_summary["rejection_rate_5pct"]):
        ax.text(v, yi, f" {v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("% of true-null simulations rejected at the 5% level")
    ax.set_title("Why the test is built this way: size when there is NO effect")
    fig.suptitle("Figure 31. What the regimes mean, and what can honestly be said about them",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Do regimes differ in risk and diversification, does any regime state "
                           "predict the NEXT day once persistence and variance are respected, and "
                           "why does the test need to be built this way?", 31)


def textwrap_label(test: str, scenario: str) -> str:
    short = "persistent label" if scenario.startswith("persistent") else "rare high-variance regime"
    return f"{short}\n{test}"


def figure_changepoints(study: dict, path):
    import matplotlib.dates as mdates

    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    result, mass, events = study["result"], study["mass"], study["events"]
    event_dates = {k: pd.Timestamp(v) for k, v in events.items()}

    ax = axes[0, 0]
    bins = result.run_length_bins
    x0, x1 = mdates.date2num(bins.index[0]), mdates.date2num(bins.index[-1])
    image = ax.imshow(np.sqrt(bins.T.to_numpy()), aspect="auto", origin="lower", cmap="magma_r",
                      extent=[x0, x1, -0.5, bins.shape[1] - 0.5], vmin=0, vmax=1)
    ax.xaxis_date()
    ax.set_yticks(range(bins.shape[1]), list(bins.columns))
    for name, date in event_dates.items():
        ax.axvline(date, color="#009E73", linestyle=":", linewidth=1.2)
    ax.grid(False)
    ax.set_ylabel("Run length so far (trading days)")
    ax.set_title("Run-length posterior (sqrt of probability); green = dated shocks")
    fig.colorbar(image, ax=ax, shrink=0.7)

    ax = axes[0, 1]
    ax.plot(mass.index, mass.to_numpy(), color="#0072B2", linewidth=0.7)
    ax.axhline(study["threshold"], color="#CC0000", linestyle="--", linewidth=1.0,
               label=f"alarm threshold {study['threshold']:.1f}")
    naive_days = study["naive_alarm"][study["naive_alarm"]].index
    ax.vlines(naive_days, -0.06, -0.01, color="#7F7F7F", linewidth=0.5, label="naive volatility-jump alarms")
    for name, date in event_dates.items():
        ax.axvline(date, color="#009E73", linestyle=":", linewidth=1.2)
    ax.set_ylim(-0.08, 1.05)
    ax.set_ylabel(f"P(run length <= {study['short_run']} days)")
    ax.set_title("The informative quantity: mass on SHORT runs")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 0]
    names = list(events)
    y = np.arange(len(names))
    strict_b = study["primary"]["events"].set_index("event")["delay_days"]
    strict_n = study["naive"]["events"].set_index("event")["delay_days"]
    for offset, (label, strict, lenient, colour) in enumerate([
            ("BOCPD", strict_b, study["primary_lenient"]["delay_days"], "#0072B2"),
            ("naive volatility jump", strict_n, study["naive_lenient"]["delay_days"], "#D55E00")]):
        yy = y + (offset - 0.5) * 0.3
        ax.scatter([strict.get(n, np.nan) for n in names], yy, s=70, color=colour, label=f"{label}, pre-declared rule")
        ax.scatter([lenient.get(n, np.nan) for n in names], yy, s=70, facecolors="none", edgecolors=colour,
                   linewidths=1.6, label=f"{label}, alarm credited up to {study['pre_days']} days early (post-hoc)")
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y, [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Trading days from the dated event to the first alarm (negative = before the date)")
    ax.set_title("Detection delay per event (missing marker = not detected)")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1, 1]
    sweep = study["sweep"]
    for k, (detector, colour) in enumerate((("BOCPD", "#0072B2"), ("naive volatility jump", "#D55E00"))):
        part = sweep[sweep["detector"] == detector].sort_values("false_alarms_per_year")
        ax.plot(part["false_alarms_per_year"], part["n_detected"] + (k - 0.5) * 0.06, "-o", color=colour,
                markersize=5, label=f"{detector}, pre-declared rule")
        ax.plot(part["false_alarms_per_year_lenient"], part["n_detected_lenient"] + (k - 0.5) * 0.06, "--s",
                color=colour, markerfacecolor="none", markersize=6, alpha=0.8, label=f"{detector}, credited early")
    ax.set_xlabel("False-alarm episodes per year")
    ax.set_ylabel(f"Dated shocks detected (of {len(events)})")
    ax.set_yticks(range(0, len(events) + 1))
    ax.set_title("Operating characteristics (thresholds swept post-hoc)")
    ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Figure 32. Online change-point detection against dated volatility shocks",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Does Bayesian online change-point detection flag dated volatility shocks "
                           "quickly, at what false-alarm cost, and does it beat a one-line volatility-"
                           "jump rule?", 32)


def figure_strategies(results: dict, start, forest: pd.DataFrame, inflation: pd.DataFrame, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    show = [("equal_weight", "equal weight", "#7F7F7F"), ("risk_parity", "risk parity", "#56B4E9"),
            ("naive_vol_timing", "naive vol timing (rule)", "#E69F00"),
            ("RA1_derisk", "RA1: HMM de-risk overlay", "#0072B2"), ("RA2_blend", "RA2: HMM blend EW/RP", "#D55E00")]

    ax = axes[0, 0]
    for name, label, colour in show:
        net = slice_dates(results[name].net_returns, start, None)
        ax.semilogy(net.index, (1.0 + net).cumprod().to_numpy(), color=colour, linewidth=1.3, label=label)
    ax.set_ylabel("Growth of 1 unit, net (log)")
    ax.set_title("Static books and regime-aware books, after costs")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    for name, label, colour in [show[0], show[2], show[3]]:
        net = slice_dates(results[name].net_returns, start, None)
        curve = (1.0 + net).cumprod()
        ax.plot(curve.index, 100 * (curve / curve.cummax() - 1.0).to_numpy(), color=colour, linewidth=1.0, label=label)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdowns: where a de-risking rule is supposed to help")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 0]
    forest = forest.reset_index(drop=True)
    y = np.arange(len(forest))
    for i, row in forest.iterrows():
        colour = "#009E73" if row.get("passes", False) else ("#0072B2" if row["kind"] == "pre-declared family" else "#7F7F7F")
        ax.errorbar(row["difference"], i, xerr=[[row["difference"] - row["lo"]], [row["hi"] - row["difference"]]],
                    fmt="o", color=colour, capsize=3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y, forest["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe difference, strategy minus benchmark (90% bootstrap interval)")
    ax.set_title("Sharpe differences: pre-declared (blue), secondary (grey)")

    ax = axes[1, 1]
    groups = list(dict.fromkeys(inflation["group"]))
    variants = ["honest (filtered, walk-forward)", "filtered, full-sample parameters", "smoothed (look-ahead)"]
    colours = ["#0072B2", "#E69F00", "#CC79A7"]
    x = np.arange(len(groups))
    for j, variant in enumerate(variants):
        values = inflation[inflation["variant"] == variant].set_index("group").reindex(groups)["sharpe"]
        bars = ax.bar(x + (j - 1) * 0.27, values, width=0.27, color=colours[j], label=variant)
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, groups, fontsize=8.5)
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_ylim(0.0, float(inflation["sharpe"].max()) * 1.32)
    ax.set_title("What hindsight adds to the Sharpe ratio")
    ax.legend(fontsize=8, loc="upper left", ncol=1)
    fig.suptitle("Figure 33. Regime-aware strategies against the books they are built on",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Do regime-aware rules beat static allocations after costs, and how much "
                           "would a backtest on hindsight probabilities have overstated?", 33)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def log_registry(context, cfg, models, flags, family, study, perf, family_tests, secondary, inflation,
                 weekly, weekly_test, leakage, persist, size, first, engine, logger) -> None:
    registry = context.registry
    cost_bps = float(cfg.get("backtest.costs.cost_bps", 10.0))
    settings = hmm_settings(cfg)
    decisions = cfg.get("regimes.decisions", {}) or {}
    test_period = f"{first.date()} onward, one day ahead"
    train_period = (f"expanding window from the start of data, refit every {settings['refit_every']} days "
                    f"after the first {settings['min_train']}")
    hmm_parameters = {"states": models["n_primary"], "refit_every": settings["refit_every"],
                      "min_train_days": settings["min_train"], "test": "studentised circular shift",
                      "n_shifts": int(((cfg.get("regimes.analysis", {}) or {}).get("permutation", {}) or {}).get("n_shifts", 5000)),
                      "fdr": float((cfg.get("regimes.analysis", {}) or {}).get("fdr", 0.10))}
    n_tested = int(family["tested"].sum())
    studentised_note = (
        "Decision rules were committed to config/regimes.yaml (commit 43522cb) before any regime result "
        "existed. One methodological change was made after the first look at the results: the circular-shift "
        "statistic was studentised. The plain difference of means had put three rare, high-variance flags "
        "(3-state high-vol, top-quintile volatility, bear market) across the line; a size simulation then showed "
        f"the plain statistic rejecting a TRUE null {100 * float(size['summary'].iloc[2]['rejection_rate_5pct']):.0f}% of the time for a rare "
        f"high-variance regime against {100 * float(size['summary'].iloc[3]['rejection_rate_5pct']):.0f}% for the studentised one. "
        "Decisions follow the studentised statistic; both p-values are in stage16_permutation_tests.csv.")

    # 1. does the filtered HMM state carry information about the next day's VOLATILITY?
    cell = _cell(family, "hmm2_high_vol", "equity_next_day_abs")
    registry.log(
        "A filtered high-volatility HMM state at the close of day t predicts larger absolute equity-sleeve "
        "returns on day t+1.",
        stage=STAGE, parameters=hmm_parameters, train_period=train_period, test_period=test_period,
        results={"mean_abs_return_in_state": float(cell["mean_in"]), "mean_abs_return_out_of_state": float(cell["mean_out"]),
                 "t_statistic": float(cell["t_statistic"]), "p_value": float(cell["p_value"]),
                 "bh_significant": bool(cell["bh_significant"]), "n_days_in_state": int(cell["n_in"]),
                 "n_tests_in_family": n_tested},
        decision="retain" if bool(cell["bh_significant"]) else "reject",
        notes=("The easy case: volatility clusters, so a state defined by volatility should forecast it, and a "
               "model that could not would be broken. Retained as a sanity check, not as a trading result. "
               "With daily data the model's two states are short volatility clusters, not bull and bear "
               "markets: the full-sample fit implies a mean spell of "
               f"{float(persist.loc['full-sample HMM (implied)', 'mean_spell_days']):.0f} days in the high-volatility "
               f"state and {float(persist.loc['full-sample HMM (implied) calm state', 'mean_spell_days']):.0f} in the calm one. "
               + studentised_note),
    )

    # 2. ... and about the next day's MEAN return?
    cell = _cell(family, "hmm2_high_vol", "equity_next_day_mean")
    three = _cell(family, "hmm3_high_vol", "equity_next_day_mean")
    registry.log(
        "A filtered high-volatility HMM state at the close of day t predicts a different mean equity-sleeve "
        "return on day t+1.",
        stage=STAGE, parameters=hmm_parameters, train_period=train_period, test_period=test_period,
        results={"mean_return_in_state": float(cell["mean_in"]), "mean_return_out_of_state": float(cell["mean_out"]),
                 "t_statistic": float(cell["t_statistic"]), "p_value": float(cell["p_value"]),
                 "bh_significant": bool(cell["bh_significant"]),
                 "three_state_t_statistic": float(three["t_statistic"]), "three_state_p_value": float(three["p_value"]),
                 "three_state_p_value_plain_difference": float(three["p_value_difference"])},
        decision="retain" if bool(cell["bh_significant"]) else "reject",
        notes=("The hard case, and the one a trading rule needs. Not rejected for lack of effect size: the "
               f"point estimates are positive in the high-volatility state ({100 * float(cell['mean_in']):.3f}% vs "
               f"{100 * float(cell['mean_out']):.3f}% a day), the opposite sign to what a de-risking rule assumes, "
               "but they are far inside the noise. The 3-state variant's apparent rebound after crisis days "
               f"(plain-difference p={float(three['p_value_difference']):.3f}) disappears once studentised "
               f"(p={float(three['p_value']):.2f}). " + studentised_note),
    )

    # 3. named rule regimes
    named = ["bear_market", "vol_high", "inflation_shock", "liquidity_crisis"]
    rule_rows = family[family["regime"].isin(named) & family["outcome"].str.endswith("_next_day_mean")]
    passed = bool(rule_rows["bh_significant"].any())
    plain_passed = bool(rule_rows["bh_significant_plain_difference"].any())
    exploratory = family[(family["regime"] == "vol_low") & family["outcome"].str.endswith("_next_day_mean")
                         & family["bh_significant"].astype(bool)]
    vol_high_row = _cell(family, "vol_high", "equity_next_day_mean")
    registry.log(
        "Named rule-based regimes (bear market, high volatility, inflation shock, liquidity crisis) are "
        "followed by different mean sleeve returns on the next day.",
        stage=STAGE, parameters={**hmm_parameters, "regimes": named, "n_outcomes": 3},
        train_period="none: rules use only trailing data", test_period=test_period,
        results={"n_regime_outcome_tests": int(len(rule_rows)), "n_untestable": int((~rule_rows["tested"].astype(bool)).sum()),
                 "any_bh_significant": passed, "any_bh_significant_with_plain_difference": plain_passed,
                 "vol_high_equity_t_statistic": float(vol_high_row["t_statistic"]),
                 "vol_high_equity_p_value": float(vol_high_row["p_value"]),
                 "vol_high_equity_p_value_plain_difference": float(vol_high_row["p_value_difference"])},
        decision="retain" if passed else "reject",
        notes=("The liquidity-crisis rule is on for only "
               f"{int(_cell(family, 'liquidity_crisis', 'equity_next_day_mean')['n_in'])} days of the "
               f"{int(analysis_days(family))} tested, below the pre-declared 60-day minimum, so it is reported "
               "but not tested. With the plain difference of means the answer would have been 'retain' "
               f"(high-volatility days, p={float(vol_high_row['p_value_difference']):.3f}); the studentised statistic, "
               f"the correctly sized one, gives p={float(vol_high_row['p_value']):.3f}. "
               + ("Outside the named hypothesis, the same BH family also flags the LOW-volatility state "
                  "(bottom volatility quintile) as followed by weaker returns for "
                  + ", ".join(exploratory["outcome"].str.replace("_next_day_mean", "").tolist())
                  + ": exploratory, not part of this decision, but family-wise controlled. " if len(exploratory) else "")
               + studentised_note),
    )

    # 4. change points
    rule = decisions.get("h_bocpd", {}) or {}
    need, cap = int(rule.get("min_events_detected", 4)), float(rule.get("max_false_alarms_per_year", 3.0))
    strict = study["primary"]
    ok = strict["n_detected"] >= need and strict["false_alarms_per_year"] <= cap
    early = study["primary_lenient"]
    registry.log(
        "Bayesian online change-point detection flags dated volatility shocks within 60 trading days with "
        "few false alarms.",
        stage=STAGE,
        parameters={"hazard_lambda": float(cfg.get("regimes.bocpd.hazard_lambda", 250)),
                    "alarm": f"P(run <= {study['short_run']}) >= {study['threshold']}",
                    "events": study["events"], "search_days": study["search_days"],
                    "min_events_detected": need, "max_false_alarms_per_year": cap},
        train_period=f"prior from the first {int(cfg.get('regimes.bocpd.burn_in_days', 250))} days only", test_period="every later day",
        results={"events_detected": strict["n_detected"], "n_events": strict["n_events"],
                 "mean_delay_days": strict["mean_delay_days"], "false_alarm_episodes_per_year": strict["false_alarms_per_year"],
                 "naive_events_detected": study["naive"]["n_detected"], "naive_mean_delay_days": study["naive"]["mean_delay_days"],
                 "naive_false_alarms_per_year": study["naive"]["false_alarms_per_year"],
                 "posthoc_events_detected_if_alarms_credited_early": early["n_detected"],
                 "posthoc_false_alarms_per_year_if_credited_early": early["false_alarms_per_year"]},
        decision="retain" if ok else "reject",
        notes=("By the pre-declared rule an alarm episode must START on or after the dated event. Three events "
               "were missed only in that sense: the detector fired one to four trading days BEFORE the date, on "
               "the same sell-offs, and those episodes are scored as false alarms. Crediting alarms up to "
               f"{study['pre_days']} days early (post-hoc, labelled, not used for the decision) gives "
               f"{early['n_detected']}/{strict['n_events']} detected at {early['false_alarms_per_year']:.2f} false "
               "alarms a year. Most 'false alarms' are real market stress that simply is not on the list of five. "
               "The one-line volatility-jump rule detects all five at its pre-declared setting with longer delays and more "
               "alarms, and at a stricter post-hoc setting matches BOCPD's false-alarm rate with a shorter delay: "
               "the Bayesian machinery is not shown to beat it on these events."),
    )

    # 5-7. strategies
    honest_perf = perf.loc[HONEST_CANDIDATES]
    best = honest_perf["sharpe"].idxmax()
    n_obs = int(perf.loc[best, "n_obs"])
    penalty = multiple_testing_penalty(float(perf.loc[best, "sharpe"]), len(HONEST_CANDIDATES), n_obs,
                                       float(perf.loc[best, "skew"]), float(perf.loc[best, "excess_kurtosis"]) + 3.0)
    texts = {
        "RA1_derisk": ("De-risking an equal-weight book toward cash as the filtered HMM high-volatility "
                       "probability rises beats equal weight after costs.",
                       "shift up to 50% of the book to SHY as P(high-vol) -> 1"),
        "RA2_blend": ("Blending equal weight and risk parity by the filtered high-volatility probability "
                      "beats BOTH constant allocations after costs.", "(1-p) equal weight + p risk parity; no free parameter"),
        "RA3_gated_momentum": ("Gating the momentum book off in regimes where it lost money on the training "
                               "sample beats the ungated book after costs.",
                               "gate learned on each window's training sample; open if too little evidence"),
    }
    for name, (hypothesis, rule_text) in texts.items():
        rows = family_tests[family_tests["strategy"] == name]
        passes = bool(rows["passes"].all())
        results = {"net_sharpe": float(perf.loc[name, "sharpe"]), "max_drawdown": float(perf.loc[name, "max_drawdown"]),
                   "n_trials_in_stage": len(HONEST_CANDIDATES),
                   "deflated_sharpe_probability_best_candidate": float(penalty["deflated_sharpe_probability"]),
                   "best_candidate": best}
        for _, r in rows.iterrows():
            results[f"sharpe_difference_vs_{r['benchmark']}"] = float(r["difference"])
            results[f"paired_p_value_vs_{r['benchmark']}"] = float(r["p_value"])
            results[f"net_sharpe_{r['benchmark']}"] = float(r["sharpe_b"])
            results[f"max_drawdown_{r['benchmark']}"] = float(perf.loc[r["benchmark"], "max_drawdown"])
        extra = ""
        if name == "RA1_derisk":
            vs_naive = secondary[(secondary["strategy"] == "RA1_derisk") & (secondary["benchmark"] == "naive_vol_timing")].iloc[0]
            naive_vs_ew = secondary[(secondary["strategy"] == "naive_vol_timing") & (secondary["benchmark"] == "equal_weight")].iloc[0]
            results["sharpe_difference_vs_naive_vol_timing"] = float(vs_naive["difference"])
            results["paired_p_value_vs_naive_vol_timing"] = float(vs_naive["p_value"])
            results["naive_vol_timing_net_sharpe"] = float(perf.loc["naive_vol_timing", "sharpe"])
            extra = (f" The same overlay driven by a one-line rule (expanding volatility percentile above the 80th) "
                     f"reaches a net Sharpe of {float(perf.loc['naive_vol_timing', 'sharpe']):.3f} against the HMM's "
                     f"{float(perf.loc[name, 'sharpe']):.3f}, with a maximum drawdown of "
                     f"{100 * float(perf.loc['naive_vol_timing', 'max_drawdown']):.1f}% against "
                     f"{100 * float(perf.loc[name, 'max_drawdown']):.1f}%: a slower state estimate suits a monthly overlay "
                     "better than the HMM's one-to-three-week clusters.")
        registry.log(
            hypothesis, stage=STAGE,
            parameters={"rule": rule_text, "derisk_max": float(cfg.get("regimes.strategies.derisk_max", 0.5)),
                        "cash": str(cfg.get("regimes.strategies.cash", "SHY")), "rebalance": engine.rebalance,
                        "states": models["n_primary"], "decision_rule": "paired bootstrap, BH across the family, positive difference"},
            train_period=train_period, test_period=f"{first.date()} onward, net of costs",
            cost_bps=cost_bps, results=results, decision="retain" if passes else "reject",
            notes=("Judged by the pre-declared family (RA1 vs equal weight; RA2 vs equal weight and vs risk "
                   "parity; RA3 vs momentum), paired stationary bootstrap, Benjamini-Hochberg across the four "
                   "comparisons; a strategy is retained only if every comparison it appears in is significant with a "
                   "POSITIVE difference. A higher point estimate without significance is not an improvement."
                   + extra),
        )

    # 8. the trap
    honest_ok = int(leakage[leakage["kind"] == "honest"]["passed"].sum())
    honest_n = int((leakage["kind"] == "honest").sum())
    control_flagged = int((~leakage[leakage["kind"] == "control"]["passed"].astype(bool)).sum())
    control_n = int((leakage["kind"] == "control").sum())
    monthly = inflation[inflation["group"] == "RA1, monthly overlay"].set_index("variant")["sharpe"]
    weekly_row = inflation[inflation["group"].str.startswith("RA1, weekly")].set_index("variant")["sharpe"]
    registry.log(
        "A backtest on smoothed (full-sample) HMM probabilities overstates the honest, filtered one, and the "
        "automated look-ahead test tells them apart.",
        stage=STAGE, parameters={"controls": ["filtered with full-sample parameters", "smoothed"],
                                 "leakage_splits": decisions.get("leakage", {}).get("split_dates"),
                                 "leakage_modes": decisions.get("leakage", {}).get("modes")},
        train_period="honest: walk-forward; controls: the whole sample", test_period=f"{first.date()} onward",
        cost_bps=cost_bps,
        results={"look_ahead_tests_honest_passed": honest_ok, "look_ahead_tests_honest_total": honest_n,
                 "look_ahead_tests_controls_flagged": control_flagged, "look_ahead_tests_controls_total": control_n,
                 "ra1_monthly_sharpe_honest": float(monthly[VARIANT_LABELS[0]]),
                 "ra1_monthly_sharpe_smoothed": float(monthly[VARIANT_LABELS[2]]),
                 "ra1_weekly_sharpe_honest": float(weekly_row[VARIANT_LABELS[0]]),
                 "ra1_weekly_sharpe_smoothed": float(weekly_row[VARIANT_LABELS[2]]),
                 "ra1_weekly_smoothed_minus_honest": float(weekly_test.get("difference", np.nan)),
                 "ra1_weekly_paired_p_value": float(weekly_test.get("p_value", np.nan))},
        decision="record",
        notes=("Two questions with different answers. (1) Does the automated test work? Every honest rule passed "
               f"({honest_ok}/{honest_n}) and every look-ahead control (smoothed, or filtered with full-sample "
               f"parameters) was flagged ({control_flagged}/{control_n}), with pre-split weights that moved when only "
               "the future was replaced. (2) Does hindsight "
               f"inflate the Sharpe ratio? At the monthly overlay it did not here (honest {float(monthly[VARIANT_LABELS[0]]):.3f}, "
               f"smoothed {float(monthly[VARIANT_LABELS[2]]):.3f}): regimes last one to three weeks, a monthly overlay "
               "samples one day in twenty, and the smoothed probabilities see only a few days past it. At a weekly "
               f"overlay (post-hoc) honest {float(weekly_row[VARIANT_LABELS[0]]):.3f}, smoothed {float(weekly_row[VARIANT_LABELS[2]]):.3f} "
               f"(difference {float(weekly_test.get('difference', np.nan)):+.3f}, p={float(weekly_test.get('p_value', np.nan)):.2f}). "
               "A leak need not show up in the headline number to be a leak; that is why the test exists."),
    )

    # 9. persistence
    spells = persist["mean_spell_days"]
    registry.log(
        "A Markov chain makes regime labels more persistent than classifying each day on its own.",
        stage=STAGE, parameters={"model": "2-component Gaussian emissions, with and without the chain"},
        train_period=train_period, test_period=test_period,
        results={"hmm2_mean_spell_days": float(spells["hmm2_high_vol"]), "gmm2_mean_spell_days": float(spells["gmm2_high_vol"]),
                 "hmm3_mean_spell_days": float(spells["hmm3_high_vol"]), "rule_vol_high_mean_spell_days": float(spells["vol_high"]),
                 "bear_market_mean_spell_days": float(spells["bear_market"])},
        decision="record",
        notes=("Descriptive. Spells are the lengths of consecutive days in the high-volatility state. Persistence is "
               "what makes a label tradable at a monthly horizon, and the HMM's spells are short either way: "
               "regimes found in daily returns are volatility clusters."),
    )


def analysis_days(family: pd.DataFrame) -> int:
    return int(family["n_pairs"].max())


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
HONEST_CANDIDATES = ["RA1_derisk", "RA2_blend", "RA3_gated_momentum",
                     "RA1_derisk_3state", "RA2_blend_3state", "RA3_gated_momentum_3state"]
VARIANT_LABELS = ["honest (filtered, walk-forward)", "filtered, full-sample parameters", "smoothed (look-ahead)"]


def _cell(family: pd.DataFrame, regime: str, outcome: str) -> pd.Series:
    return family[(family["regime"] == regime) & (family["outcome"] == outcome)].iloc[0]


def _secondary_tests(streams: dict, pairs: list[tuple[str, str]], test_cfg: dict) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        result = paired_sharpe_test(streams[a], streams[b], n_samples=int(test_cfg.get("n_samples", 2000)),
                                    block_length=int(test_cfg.get("block_length", 21)),
                                    seed=int(test_cfg.get("seed", 7)))
        rows.append({"strategy": a, "benchmark": b, **result})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 16: regime detection")
    parser.add_argument("--skip-leakage", action="store_true",
                        help="reuse the last look-ahead table if there is one (development only)")
    args = parser.parse_args(argv)

    context, logger = build_context(STAGE, generation=2)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 16 | regime detection (Generation 2, Priority 1)")
    logger.info("=" * 72)
    market = context.market_data()
    returns = market.returns()
    index = pd.DatetimeIndex(market.index)
    X = emission_matrix(cfg, returns, market.investable)
    logger.info("emission matrix: %d days x %s, in percent", len(X), list(X.columns))

    models = fit_regime_models(cfg, X, cfg.path("features") / "regimes", logger)
    first = models["hmm_primary"].first_date
    n_refits = len(models["hmm_primary"].windows)
    logger.info("first out-of-sample probability: %s after %d refits", first.date(), n_refits)
    context.save_table(pd.DataFrame({"hmm2_p_high": models["hmm_primary"].p_high,
                                     "hmm3_p_high": models["hmm_sensitivity"].p_high}).dropna(how="all"),
                       "stage16_filtered_probabilities.csv")
    context.save_table(pd.concat([hmm_refit_table(models["hmm_primary"], "hmm_2_states"),
                                  hmm_refit_table(models["hmm_sensitivity"], "hmm_3_states")]),
                       "stage16_hmm_refits.csv", index=False)
    full_fit = models["full_primary"]["fit"]
    logger.info("full-sample 2-state HMM (look-ahead, for reference): stay probabilities %s, "
                "equity volatility by state %s", np.round(np.diag(full_fit.transmat), 3).tolist(),
                [round(float(np.sqrt(c[0, 0] * 252) / 100.0), 3) for c in full_fit.covars])

    # ------------------------------------------------------------ rule-based regimes
    specs, raw = ensure_macro_raw(cfg)
    panel = macro_feature_panel(raw, specs, index, cfg)
    rules = rule_regimes(cfg, market, returns, panel["levels"]["CPI_YOY"])
    flags = regime_flags(models, rules)
    context.save_table(flags.loc[first:], "stage16_regime_flags.csv")
    persist = persistence_table(flags.loc[first:], models)
    context.save_table(persist, "stage16_regime_persistence.csv")
    logger.info("regimes on the out-of-sample span:\n%s", persist.round(2).to_string())

    # ------------------------------------------------------------ conditional analysis
    analysis = cfg.get("regimes.analysis", {}) or {}
    fdr = float(analysis.get("fdr", 0.10))
    min_days = int(analysis.get("min_days_per_regime", 60))
    sleeves = {k: list(v) for k, v in (cfg.get("regimes.detection.sleeves", {}) or {}).items()}
    sleeve_daily = sleeve_returns(returns, sleeves, market.investable)
    span = slice(first, None)
    bond_stock = pd.DataFrame({"stock": returns["SPY"], "bond": returns["IEF"]})
    cond = conditional_table(flags, sleeve_daily, bond_stock, span, min_days)
    context.save_table(cond, "stage16_regime_conditional.csv", index=False)
    family = run_permutation_family(flags, sleeve_daily, span, analysis, fdr)
    context.save_table(family, "stage16_permutation_tests.csv", index=False)
    show = ["regime", "outcome", "n_in", "mean_in", "mean_out", "t_statistic", "p_value",
            "p_value_difference", "bh_significant", "bh_significant_plain_difference"]
    logger.info("circular-shift tests (%s), BH FDR %.0f%% across %d tests:\n%s", LAG_NOTE, 100 * fdr,
                int(family["tested"].sum()), family[family["tested"].astype(bool)][show].round(4).to_string(index=False))
    untested = family[~family["tested"].astype(bool)][["regime", "outcome", "n_in", "n_out"]]
    if len(untested):
        logger.info("not tested (fewer than %d days in or out of the regime):\n%s", min_days,
                    untested.drop_duplicates("regime").to_string(index=False))
    size = size_demonstration()
    context.save_table(size["summary"], "stage16_test_size_simulation.csv", index=False)
    logger.info("size when there is no effect:\n%s", size["summary"].round(3).to_string(index=False))

    # ------------------------------------------------------------ change points
    study = bocpd_study(cfg, X["equity"])
    events_table = study["primary"]["events"].merge(
        study["naive"]["events"][["event", "detected", "alarm_date", "delay_days"]], on="event",
        suffixes=("_bocpd", "_naive"))
    events_table["bocpd_delay_if_credited_early"] = events_table["event"].map(study["primary_lenient"]["delay_days"])
    events_table["naive_delay_if_credited_early"] = events_table["event"].map(study["naive_lenient"]["delay_days"])
    context.save_table(events_table, "stage16_bocpd_events.csv", index=False)
    context.save_table(study["sweep"], "stage16_detector_operating_characteristics.csv", index=False)
    context.save_table(study["result"].short_mass.join(study["result"].expected_run_length), "stage16_bocpd_posterior_summary.csv")
    logger.info("BOCPD (alarm: P(run<=%d) >= %.1f): detected %d/%d, mean delay %.1f days, %.2f false-alarm "
                "episodes/yr;  naive jump rule: %d/%d, %.1f days, %.2f/yr",
                study["short_run"], study["threshold"], study["primary"]["n_detected"], study["primary"]["n_events"],
                study["primary"]["mean_delay_days"], study["primary"]["false_alarms_per_year"],
                study["naive"]["n_detected"], study["naive"]["n_events"], study["naive"]["mean_delay_days"],
                study["naive"]["false_alarms_per_year"])
    logger.info("post-hoc, alarms credited up to %d days early: BOCPD %d/%d (%.2f/yr), naive %d/%d (%.2f/yr)",
                study["pre_days"], study["primary_lenient"]["n_detected"], study["primary"]["n_events"],
                study["primary_lenient"]["false_alarms_per_year"], study["naive_lenient"]["n_detected"],
                study["primary"]["n_events"], study["naive_lenient"]["false_alarms_per_year"])

    # ------------------------------------------------------------ strategies
    engine = BacktestEngine.from_config(cfg)

    def provider(m):
        if m is market:
            from experiments.strategies import cached_ladder
            return cached_ladder(m, cfg, context.processed, ["M0_equal_weight", "M2_risk_parity", "M3_momentum"])
        return build_ladder(m, cfg, ["M0_equal_weight", "M2_risk_parity", "M3_momentum"])

    factory = RegimeStrategyFactory(cfg, engine, models["fit_cache"], provider)
    books = factory.weights(market, models)
    results = run_books(engine, books, returns, market.investable)
    perf = performance_table(results, first)
    context.save_table(perf, "stage16_strategy_performance.csv")
    logger.info("strategy performance from %s (net of costs):\n%s", first.date(),
                perf[["sharpe", "cagr", "ann_vol", "max_drawdown", "calmar", "ann_turnover"]].round(3).to_string())
    streams = {name: slice_dates(r.net_returns, first, None) for name, r in results.items()}
    spec = cfg.get("regimes.decisions.h_strategies", {}) or {}
    strategies_cfg = cfg.get("regimes.strategies", {}) or {}
    family_tests = strategy_family_tests(streams, spec.get("family", {}), spec.get("paired_test", {}), fdr)
    context.save_table(family_tests, "stage16_strategy_family_tests.csv", index=False)
    logger.info("pre-declared family (paired bootstrap, BH FDR %.0f%%):\n%s", 100 * fdr,
                family_tests[["strategy", "benchmark", "sharpe_a", "sharpe_b", "difference", "p_value",
                              "bh_significant", "passes"]].round(3).to_string(index=False))
    secondary_pairs = [("RA1_derisk", "naive_vol_timing"), ("naive_vol_timing", "equal_weight"),
                       ("RA1_derisk_3state", "equal_weight"), ("RA2_blend_3state", "risk_parity"),
                       ("RA3_gated_momentum_3state", "momentum"),
                       ("RA1_derisk|smoothed", "RA1_derisk"), ("RA1_derisk|full_theta", "RA1_derisk"),
                       ("RA2_blend|smoothed", "RA2_blend"), ("RA2_blend|full_theta", "RA2_blend")]
    secondary = _secondary_tests(streams, secondary_pairs, spec.get("paired_test", {}))
    context.save_table(secondary, "stage16_strategy_secondary_tests.csv", index=False)
    logger.info("secondary comparisons (reported, not decisions):\n%s",
                secondary[["strategy", "benchmark", "sharpe_a", "sharpe_b", "difference", "p_value"]].round(3).to_string(index=False))
    gate_log = pd.concat({k or "2_states": v for k, v in factory.state(market, models).gate_log.items()},
                         names=["variant"]).reset_index(level=0)
    context.save_table(gate_log, "stage16_gate_log.csv", index=False)

    # post-hoc sensitivity: a faster (weekly) overlay, where hindsight has more days to help
    weekly_engine = dataclasses.replace(engine, rebalance=str(cfg.get("regimes.decisions.sensitivity.overlay_rebalance", "weekly")))
    weekly_names = ["equal_weight", "naive_vol_timing", "RA1_derisk", "RA1_derisk|full_theta", "RA1_derisk|smoothed"]
    weekly_results = run_books(weekly_engine, {k: books[k] for k in weekly_names}, returns, market.investable)
    weekly = performance_table(weekly_results, first)
    context.save_table(weekly, "stage16_weekly_overlay_sensitivity.csv")
    weekly_streams = {k: slice_dates(r.net_returns, first, None) for k, r in weekly_results.items()}
    weekly_test = paired_sharpe_test(weekly_streams["RA1_derisk|smoothed"], weekly_streams["RA1_derisk"],
                                     n_samples=int(spec.get("paired_test", {}).get("n_samples", 2000)),
                                     block_length=int(spec.get("paired_test", {}).get("block_length", 21)),
                                     seed=int(spec.get("paired_test", {}).get("seed", 7)))
    logger.info("weekly overlay (post-hoc): Sharpe honest %.3f, full-theta %.3f, smoothed %.3f; "
                "smoothed minus honest %+.3f (p=%.3f)", weekly.loc["RA1_derisk", "sharpe"],
                weekly.loc["RA1_derisk|full_theta", "sharpe"], weekly.loc["RA1_derisk|smoothed", "sharpe"],
                weekly_test.get("difference", np.nan), weekly_test.get("p_value", np.nan))

    inflation_rows = []
    for group, names, table in (("RA1, monthly overlay", ("RA1_derisk", "RA1_derisk|full_theta", "RA1_derisk|smoothed"), perf),
                                ("RA2, monthly blend", ("RA2_blend", "RA2_blend|full_theta", "RA2_blend|smoothed"), perf),
                                ("RA1, weekly overlay\n(post-hoc)", ("RA1_derisk", "RA1_derisk|full_theta", "RA1_derisk|smoothed"), weekly)):
        for label, name in zip(VARIANT_LABELS, names):
            inflation_rows.append({"group": group, "variant": label, "strategy": name, "sharpe": float(table.loc[name, "sharpe"])})
    inflation = pd.DataFrame(inflation_rows)
    context.save_table(inflation, "stage16_lookahead_inflation.csv", index=False)

    # ------------------------------------------------------------ the automated look-ahead test
    leak_path = context.tables / "stage16_leakage.csv"
    if args.skip_leakage and leak_path.exists():
        leakage = pd.read_csv(leak_path)
        logger.info("look-ahead tests: reusing %s", leak_path.name)
    else:
        leakage = leakage_study(cfg, market, factory, models, logger)
        context.save_table(leakage, "stage16_leakage.csv", index=False)
    honest = leakage[leakage["kind"] == "honest"]
    controls = leakage[leakage["kind"] == "control"]
    logger.info("look-ahead test: honest rules passed %d/%d, controls flagged %d/%d",
                int(honest["passed"].sum()), len(honest), int((~controls["passed"].astype(bool)).sum()), len(controls))

    # ------------------------------------------------------------ figures
    figure_timeline(X, models, flags, rules, persist, study["events"], context.figure("fig30_regimes_timeline.png"))
    figure_conditional(cond, family, size["summary"], context.figure("fig31_regime_conditional.png"))
    figure_changepoints(study, context.figure("fig32_changepoints.png"))
    forest_rows = []
    for _, r in family_tests.iterrows():
        forest_rows.append({"label": f"{r['strategy']} vs {r['benchmark']}", "difference": r["difference"],
                            "lo": r["ci_lower_5pct"], "hi": r["ci_upper_95pct"], "kind": "pre-declared family",
                            "passes": bool(r["passes"])})
    for _, r in secondary.iterrows():
        forest_rows.append({"label": f"{r['strategy']} vs {r['benchmark']}", "difference": r["difference"],
                            "lo": r["ci_lower_5pct"], "hi": r["ci_upper_95pct"], "kind": "secondary", "passes": False})
    figure_strategies(results, first, pd.DataFrame(forest_rows), inflation, context.figure("fig33_regime_strategies.png"))

    # ------------------------------------------------------------ registry
    log_registry(context, cfg, models, flags, family, study, perf, family_tests, secondary, inflation,
                 weekly, weekly_test, leakage, persist, size, first, engine, logger)
    logger.info("STAGE 16 complete")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
