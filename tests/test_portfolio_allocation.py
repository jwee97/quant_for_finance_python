"""Forecast-driven, risk-aware allocation (``src/ml/portfolio_alloc.py``): deterministic cases whose answer is known, the point-in-time property, the reconciliation of the book through the authoritative backtest, and that the
default and the existing experiments are untouched.

Each case builds a panel of daily returns and forecasts whose properties are set by the test (volatilities, correlations, the size of the forecast) and reads the weights an allocator gives, so that a wrong sign, a lost
constraint or a forecast that is silently ignored cannot pass."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.ml import allocator_compare as ac
from src.ml import config as mlcfg
from src.ml import portfolio_alloc as pa
from src.ml.backtest import run_backtest

N = 700
DATES = pd.bdate_range("2018-01-01", periods=N)


def panel(vols=(0.01, 0.01), corr=0.0, seed=0, drift=0.0, n=N):
    rng = np.random.default_rng(seed)
    k = len(vols)
    c = np.full((k, k), corr)
    np.fill_diagonal(c, 1.0)
    chol = np.linalg.cholesky(c)
    z = rng.standard_normal((n, k)) @ chol.T
    r = z * np.asarray(vols) + np.asarray(drift)
    return pd.DataFrame(r, index=DATES[:n], columns=[f"S{i}" for i in range(k)])


def signal(allocator, forecast=None, returns=None, **kw):
    lim = kw.pop("limits", {})
    alloc = {"allocator": allocator, "calibrate": "none", "cov_method": "sample", "cov_window": 252, "cost_bps": 0.0, "limits": {"max_weight": 1.0, **lim}, **kw}
    return {"method": "portfolio_allocation", "holding_period": 1, "allocation": alloc, "window_days": 5, "lag_total": 1}


def score_of(returns, mu):
    return pd.DataFrame({c: float(m) for c, m in zip(returns.columns, mu)}, index=returns.index)


def last_weights(allocator, returns, mu, **kw):
    w, log = pa.run(score_of(returns, mu), returns, None, signal(allocator, **kw))
    return w.iloc[-1], log


# ------------------------------------------------------------------------------------------------ the two-stock cases
def test_equal_forecasts_and_unequal_risk_overweight_the_calm_stock_and_equal_weight_stays_half_half():
    r = panel(vols=(0.01, 0.03), seed=1)
    eq, _ = last_weights("equal_weight", r, (0.01, 0.01))
    iv, _ = last_weights("inverse_vol", r, (0.01, 0.01))
    mv, _ = last_weights("mean_variance", r, (0.01, 0.01), risk_aversion=5.0)
    assert eq.to_dict() == pytest.approx({"S0": 0.5, "S1": 0.5})
    assert iv["S0"] / iv["S1"] == pytest.approx(3.0, rel=0.15) and iv.sum() == pytest.approx(1.0)
    assert mv["S0"] > mv["S1"] and mv.sum() == pytest.approx(1.0, abs=1e-6), "the same forecast for both: the lower-risk stock earns more per unit of risk"


def test_unequal_forecasts_and_equal_risk_overweight_the_better_forecast_while_risk_only_rules_stay_half_half():
    r = panel(vols=(0.01, 0.01), seed=2)
    mv, _ = last_weights("mean_variance", r, (0.02, 0.0), risk_aversion=5.0)
    iv, _ = last_weights("inverse_vol", r, (0.02, 0.0))
    erc, _ = last_weights("erc", r, (0.02, 0.0))
    assert mv["S0"] > 0.6 > mv["S1"]
    assert iv["S0"] == pytest.approx(0.5, abs=0.06) and erc["S0"] == pytest.approx(0.5, abs=0.06), "inverse volatility and ERC never read the forecast"


def test_a_stronger_forecast_gives_a_larger_overweight_and_a_larger_risk_aversion_a_smaller_one():
    r = panel(vols=(0.01, 0.01), seed=3)
    small, _ = last_weights("mean_variance", r, (0.0004, 0.0), risk_aversion=5.0)
    big, _ = last_weights("mean_variance", r, (0.0012, 0.0), risk_aversion=5.0)
    timid, _ = last_weights("mean_variance", r, (0.0012, 0.0), risk_aversion=50.0)
    assert big["S0"] > small["S0"] > 0.5
    assert 0.5 <= timid["S0"] < big["S0"]


def test_highly_correlated_stocks_make_a_forecast_difference_a_bet_on_the_spread():
    r = panel(vols=(0.01, 0.01), corr=0.97, seed=4)
    mv, _ = last_weights("mean_variance", r, (0.012, 0.010), risk_aversion=5.0, limits={"long_only": False, "gross_max": 2.0, "net_min": 1.0, "net_max": 1.0, "max_weight": 2.0})
    erc, _ = last_weights("erc", r, (0.012, 0.010))
    assert erc["S0"] == pytest.approx(0.5, abs=0.05)
    assert mv["S0"] > 1.0 and mv["S1"] < 0.0, "near-identical stocks with a different forecast: long the better one against the other"


def test_a_singular_covariance_is_repaired_or_reported_never_a_crash_and_the_book_is_valid():
    r = panel(vols=(0.01, 0.01, 0.01), seed=5)
    r["S2"] = r["S0"]                                                                          # an exact duplicate: the sample covariance is singular
    w, log = pa.run(score_of(r, (0.01, 0.0, 0.01)), r, None, signal("mean_variance"))
    book = w.iloc[-1]
    assert np.isfinite(book).all() and book.sum() == pytest.approx(1.0, abs=1e-6) and (book >= -1e-9).all()
    assert log["counts"]["decisions"] > 0


def test_negative_expected_returns_long_only_fully_invested_cash_allowed_and_long_short():
    r = panel(vols=(0.01, 0.01), seed=6)
    full, _ = last_weights("mean_variance", r, (-0.02, -0.01), risk_aversion=5.0)
    assert full.sum() == pytest.approx(1.0, abs=1e-6) and (full >= -1e-9).all(), "fully invested long only: the least bad mix, not a short"
    cash, _ = last_weights("mean_variance", r, (-0.02, -0.01), risk_aversion=5.0, limits={"cash_allowed": True, "gross_max": 1.0})
    assert cash.sum() < 0.05, "cash allowed and every expected return negative: stay in cash"
    short, _ = last_weights("mean_variance", r, (-0.02, 0.02), risk_aversion=5.0, limits={"long_only": False, "gross_max": 1.0, "max_weight": 1.0, "cash_allowed": True})
    assert short["S0"] < -0.05 and short["S1"] > 0.05 and short.abs().sum() <= 1.0 + 1e-6


def test_position_caps_hold_and_an_infeasible_set_of_limits_is_reported_not_relaxed():
    r = panel(vols=(0.01, 0.01), seed=7)
    capped, _ = last_weights("mean_variance", r, (0.05, 0.0), risk_aversion=1.0, limits={"max_weight": 0.65})
    assert capped.max() <= 0.65 + 1e-9 and capped.sum() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(pa.InfeasibleConstraints):
        pa.require_feasible(pa.Limits(max_weight=0.3), ["S0", "S1"], None, "mean_variance")
    w, log = pa.run(score_of(r, (0.01, 0.0)), r, None, signal("mean_variance", limits={"max_weight": 0.3}))
    assert log["counts"]["optimal"] == 0 and log["counts"]["infeasible"] == log["counts"]["decisions"] > 0
    assert (w.abs().sum(axis=1) < 1e-9).all() or (w.max(axis=1) <= 0.3 + 1e-9).all(), "a limit is never relaxed to make a book"
    assert all(rec["status"] == "infeasible" and rec["failure_policy"] for rec in log["records"] if "status" in rec)


def test_a_sector_cap_without_a_sector_map_is_refused_not_ignored_and_with_one_it_binds():
    with pytest.raises(pa.InfeasibleConstraints, match="sector map"):
        pa.require_feasible(pa.Limits(max_sector=0.5), ["S0", "S1"], None, "mean_variance")
    r = panel(vols=(0.01, 0.01, 0.01), seed=18)
    groups = {"S0": "tech", "S1": "tech", "S2": "bank"}
    w, _ = pa.run(score_of(r, (0.002, 0.002, 0.0)), r, None, signal("mean_variance", limits={"max_sector": 0.6}), groups)
    book = w.iloc[-1]
    assert book[["S0", "S1"]].sum() <= 0.6 + 1e-6 and book.sum() == pytest.approx(1.0, abs=1e-6)


def test_transaction_costs_and_a_turnover_penalty_slow_the_book_down():
    r = panel(vols=(0.01, 0.01), seed=8)
    s = pd.DataFrame({"S0": 0.01 * np.sin(np.arange(N) / 15.0), "S1": -0.01 * np.sin(np.arange(N) / 15.0)}, index=r.index)
    free, _ = pa.run(s, r, None, signal("mean_variance", cost_bps=0.0, risk_aversion=3.0))
    dear, _ = pa.run(s, r, None, signal("mean_variance", cost_bps=150.0, turnover_penalty=5.0, risk_aversion=3.0))
    assert dear.diff().abs().sum().sum() < free.diff().abs().sum().sum()


def test_a_turnover_limit_caps_the_trade_at_every_decision():
    r = panel(vols=(0.01, 0.01), seed=9)
    s = pd.DataFrame({"S0": 0.03 * np.sign(np.sin(np.arange(N) / 20.0)), "S1": 0.0}, index=r.index)
    w, log = pa.run(s, r, None, signal("mean_variance", risk_aversion=2.0, limits={"turnover_max": 0.10}))
    first = w.abs().sum(axis=1).gt(0).idxmax()                                                  # entering from a flat book is not a rebalance of a held book
    moves = (w.diff().abs().sum(axis=1) / 2.0).loc[first:].iloc[1:]
    assert moves.max() <= 0.10 + 1e-6 and log["counts"]["turnover_limited"] > 0


def test_forecast_uncertainty_shrinks_the_black_litterman_tilt_and_the_robust_book_below_plain_mean_variance():
    r = panel(vols=(0.01, 0.01), seed=10)
    rng = np.random.default_rng(10)
    # a noisy but informative forecast: its out-of-sample error is large, so the measured view variance is large
    truth = r.rolling(5).sum().shift(-1)
    s = pd.DataFrame({c: 0.3 * truth[c].fillna(0.0) + rng.normal(0, 0.03, N) for c in r.columns}, index=r.index)
    sig_cal = lambda a: {**signal(a), "allocation": {**signal(a)["allocation"], "calibrate": "slope", "calib_min_t": 0.0, "calib_min_obs": 60}}
    mv, _ = pa.run(s, r, None, sig_cal("mean_variance"))
    bl, _ = pa.run(s, r, None, sig_cal("black_litterman"))
    dev = lambda w: (w.iloc[-200:] - 0.5).abs().mean().mean()
    assert dev(bl) <= dev(mv) + 1e-9, "a forecast with a large measured error moves the Black-Litterman book less than it moves plain mean-variance"
    cash = lambda a: {**sig_cal(a), "allocation": {**sig_cal(a)["allocation"], "limits": {"cash_allowed": True, "max_weight": 1.0}}}
    mvc, _ = pa.run(s, r, None, cash("mean_variance"))
    robc, _ = pa.run(s, r, None, cash("robust_mv"))
    assert robc.iloc[-200:].sum(axis=1).mean() <= mvc.iloc[-200:].sum(axis=1).mean() + 1e-9, "the robust book holds no more of an uncertain forecast than the plain one when it may hold cash"


def test_hrp_is_refused_for_fewer_than_three_assets_and_the_probability_is_never_an_expected_return():
    r = panel(vols=(0.01, 0.01), seed=11)
    with pytest.raises(pa.AllocationError, match="at least 3 assets|three assets"):
        pa.run(score_of(r, (0.0, 0.0)), r, None, signal("hrp"))
    with pytest.raises(pa.AllocationError, match="not an expected return"):
        pa.to_expected_return(pd.DataFrame({"S0": [0.6]}), "probability")


def test_the_forecast_contract_refuses_another_horizon_or_currency_and_a_price_becomes_a_return():
    cfg = pa.config_from_signal({"allocation": {"forecast_horizon": 21, "forecast_currency": "EUR"}})
    with pytest.raises(pa.AllocationError, match="21 trading days"):
        pa.check_forecast_contract(cfg, 5, "EUR")
    cfg2 = pa.config_from_signal({"allocation": {"forecast_horizon": 5, "forecast_currency": "EUR"}})
    with pytest.raises(pa.AllocationError, match="EUR.*USD|currencies"):
        pa.check_forecast_contract(cfg2, 5, "USD")
    prices = pd.DataFrame({"A": [100.0, 110.0]})
    out = pa.to_expected_return(pd.DataFrame({"A": [105.0, 99.0]}), "price", prices)
    assert out["A"].tolist() == pytest.approx([0.05, -0.1])
    with pytest.raises(pa.AllocationError, match="unknown allocation setting"):
        pa.config_from_signal({"allocation": {"allocatr": "erc"}})


# ------------------------------------------------------------------------------------------------ point in time
@pytest.mark.parametrize("allocator", ["mean_variance", "black_litterman", "cvar", "kelly", "inverse_vol", "erc"])
def test_weights_at_a_date_do_not_depend_on_anything_after_it(allocator):
    r = panel(vols=(0.01, 0.02, 0.015), corr=0.3, seed=12)
    rng = np.random.default_rng(12)
    s = pd.DataFrame(0.2 * r.rolling(5).sum().shift(-1).fillna(0.0).to_numpy() + rng.normal(0, 0.01, r.shape), index=r.index, columns=r.columns)
    cut = 500
    sig = {**signal(allocator), "allocation": {**signal(allocator)["allocation"], "calibrate": "slope", "calib_min_t": 0.0, "kelly_min_t": 0.0}}
    base, _ = pa.run(s, r, None, sig)
    r2, s2 = r.copy(), s.copy()
    r2.iloc[cut + 1:] = r2.iloc[cut + 1:] * -3.0 + 0.05                                        # rewrite the future
    s2.iloc[cut + 1:] = s2.iloc[cut + 1:] * -3.0
    other, _ = pa.run(s2, r2, None, sig)
    assert np.allclose(base.iloc[:cut + 1].to_numpy(), other.iloc[:cut + 1].to_numpy(), atol=1e-9, equal_nan=True), "a weight at date t used information from after t"
    truncated, _ = pa.run(s.iloc[:cut + 1], r.iloc[:cut + 1], None, sig)
    assert np.allclose(base.iloc[:cut + 1].to_numpy(), truncated.to_numpy(), atol=1e-9, equal_nan=True), "the weights change when the future is simply not there"


def test_the_calibration_uses_only_forecasts_whose_outcome_has_ended_before_the_decision():
    r = panel(vols=(0.01, 0.01), seed=13)
    s = pd.DataFrame(0.2 * r.rolling(5).sum().shift(-1).fillna(0.0).to_numpy(), index=r.index, columns=r.columns)
    cfg = pa.config_from_signal({"allocation": {"calibrate": "slope", "calib_min_obs": 30, "calib_min_t": 0.0}})
    fm = pa.ForecastModel(s, r, cfg, horizon=5, lag=1)
    i = 400
    a = fm.asof(i)
    s2, r2 = s.copy(), r.copy()
    r2.iloc[i + 1:] = 9.0                                                                      # everything after the decision date (the newest outcome used ends at row i)
    s2.iloc[i + 1:] = -1.0
    b = pa.ForecastModel(s2, r2, cfg, horizon=5, lag=1).asof(i)
    assert a.slope == pytest.approx(b.slope) and a.n_obs == b.n_obs


def test_weights_are_recomputed_at_each_rebalance_from_the_forecasts_then_available():
    r = panel(vols=(0.01, 0.01), seed=14)
    s = pd.DataFrame({"S0": 0.02 * np.where(np.arange(N) < 400, 1.0, -1.0), "S1": 0.0}, index=r.index)
    w, _ = pa.run(s, r, None, signal("mean_variance", risk_aversion=3.0))
    assert w["S0"].iloc[300] > 0.6 and w["S0"].iloc[-1] < 0.4, "the forecast flipped sign at row 400 and the weights followed it"


# ------------------------------------------------------------------------------------------------ accounting
def test_the_backtest_reconciles_pnl_costs_and_the_allocator_weights():
    r = panel(vols=(0.01, 0.02), seed=15)
    s = pd.DataFrame({"S0": 0.01 * np.sin(np.arange(N) / 30.0), "S1": 0.0}, index=r.index)
    w, _ = pa.run(s, r, None, signal("mean_variance", risk_aversion=3.0))
    bt = {"rebalance": "weekly", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": None, "max_leverage": 1.5, "extra_lag": 0}
    res = run_backtest(w, r, None, bt, 1, "t")
    assert np.allclose(res.net_returns, res.gross_returns - res.costs.reindex(res.gross_returns.index).fillna(0.0), atol=1e-12)
    held = res.weights.shift(1).reindex(r.index).fillna(0.0)                                   # the book held through day t is the one the engine had at the end of t - 1
    assert np.allclose((held * r.fillna(0.0)).sum(axis=1).reindex(res.gross_returns.index), res.gross_returns, atol=1e-10), "gross P&L is the held book times the day's returns"
    assert (res.costs.fillna(0.0) >= 0).all() and res.costs.sum() > 0
    zero = run_backtest(w, r, None, {**bt, "cost_bps": 0.0, "slippage_bps": 0.0}, 1, "t")
    assert zero.net_returns.sum() > res.net_returns.sum(), "costs reduce the net result"


def test_the_weight_decomposition_adds_up_and_names_what_drove_the_weights():
    r = panel(vols=(0.01, 0.03), seed=16)
    _, log = last_weights("mean_variance", r, (0.02, 0.0), risk_aversion=5.0)
    rec = [x for x in log["records"] if x.get("explain")][-1]
    ex, a = rec["explain"], ["S0", "S1"]
    for k in a:
        assert ex["equal"][k] + ex["risk_effect"][k] + ex["forecast_effect"][k] == pytest.approx(rec["weights"][k], abs=1e-9)
    assert ex["driver"] in ("forecast", "risk", "constraints", "equal")
    _, log2 = last_weights("equal_weight", r, (0.02, 0.0))
    assert [x for x in log2["records"] if x.get("explain")][-1]["explain"]["driver"] == "equal"


# ------------------------------------------------------------------------------------------------ comparison and selection discipline
def _compare_inputs(n_assets=2, seed=17):
    r = panel(vols=tuple([0.01] * n_assets), seed=seed)
    rng = np.random.default_rng(seed)
    s = pd.DataFrame(0.3 * r.rolling(5).sum().shift(-1).fillna(0.0).to_numpy() + rng.normal(0, 0.02, r.shape), index=r.index, columns=r.columns)
    labels = pd.Series(np.where(np.arange(N) < 400, "validation", np.where(np.arange(N) < 550, "test", "forward")), index=r.index)
    sig = {**signal("mean_variance"), "allocation": {**signal("mean_variance")["allocation"], "calibrate": "slope", "calib_min_t": 0.0, "cost_bps": 7.0}}
    bt = {"rebalance": "weekly", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": None, "max_leverage": 1.5, "extra_lag": 0}
    return s, r, sig, bt, labels


def test_the_comparison_uses_identical_forecasts_selects_on_validation_and_hides_protected_segments_unless_asked():
    s, r, sig, bt, labels = _compare_inputs()
    out = ac.compare(s, r, None, sig, bt, 1, labels, allocators=("equal_weight", "inverse_vol", "mean_variance", "hrp"))
    assert "hrp" in out["skipped"] and set(out["allocators"]) == {"equal_weight", "inverse_vol", "mean_variance"}
    assert out["select_on"] == "validation" and out["evaluated_segments"] == []
    assert all(set(v["segments"]) == {"validation"} for v in out["table"].values()), "test and forward are not computed unless asked for"
    assert set(out["curves"]["segments"]) == {"validation"} and out["selected"] in out["allocators"]
    assert out["table"]["equal_weight"]["segments"]["validation"]["effective_n"] == pytest.approx(2.0)
    assert out["paired_vs_equal_weight"] and all("p_value_holm" in v for v in out["paired_vs_equal_weight"].values())
    rich = ac.compare(s, r, None, sig, bt, 1, labels, allocators=("equal_weight", "mean_variance"), evaluate=("test",))
    assert rich["evaluated_segments"] == ["test"] and all(v["segments"]["test"]["used_for_selection"] is False for v in rich["table"].values())
    assert rich["selected"] == ac.compare(s, r, None, sig, bt, 1, labels, allocators=("equal_weight", "mean_variance"))["selected"], "looking at the test segment does not change the choice"
    with pytest.raises(pa.AllocationError, match="cannot also be listed"):
        ac.compare(s, r, None, sig, bt, 1, labels, allocators=("equal_weight",), evaluate=("validation",))


def test_prediction_quality_estimates_no_cross_sectional_statistic_for_two_stocks_and_one_for_six():
    s, r, *_ = _compare_inputs()
    labels = pd.Series("validation", index=r.index)
    pq = ac.prediction_quality(s, r, 5, 1, labels)["validation"]
    assert pq["cross_sectional_rank_ic"] is None and "at least five assets" in pq["cross_sectional_note"] and pq["pooled_correlation"] is not None
    s6, r6, *_ = _compare_inputs(n_assets=6)
    assert ac.prediction_quality(s6, r6, 5, 1, pd.Series("validation", index=r6.index))["validation"]["cross_sectional_rank_ic"] is not None


# ------------------------------------------------------------------------------------------------ the default and history are unchanged
def test_the_default_signal_method_and_the_config_hashes_of_existing_experiments_are_unchanged():
    from src.ml import experiments as ex

    assert mlcfg.experiment_config({})["signal"]["method"] != "portfolio_allocation"
    assert ex.config_hash(mlcfg.experiment_config({})) == "c1b756f10a81"
    assert ex.config_hash(mlcfg.experiment_config(mlcfg.RECOMMENDED)) == "f2fc7c3f5c4e"
    assert ex.config_hash(mlcfg.experiment_config({"signal": {"method": "cost_aware_mean_variance", "holding_period": 1}, "dataset": {"target": "fwd_ret_5", "horizon": 5}})) == "44ead79c14a8"


def test_existing_position_rules_reproduce_their_recorded_numbers():
    """Golden values recorded from the code before portfolio allocation existed (6 assets, seed 4, weekly, ridge)."""
    from src.ml import experiments as ex
    from tests.ml_helpers import make_bundle

    golden = {"top_n_long_short": {"weights_sum": 495.4, "net_sum": 0.252029060303, "turnover_sum": 53.9481654002}, "benchmark_tilt": {"weights_sum": 522.0, "net_sum": 0.724470988618, "turnover_sum": 16.0336147051}}
    b = make_bundle(n_assets=6, n=1100, seed=4, alpha=1.0, macro=False)
    base = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5", "horizon": 5, "min_history": 60}, "split": {"min_train_days": 300, "refit_every": 63, "validation_fraction": 0.2, "test_fraction": 0.2,
            "forward_fraction": 0.1}, "model": {"kind": "ridge"}, "backtest": {"rebalance": "weekly", "target_vol": None}}
    for method, extra in (("top_n_long_short", {"n": 2}), ("benchmark_tilt", {"tilt": 0.5})):
        cfg = copy.deepcopy(base)
        cfg["signal"] = {"method": method, **extra}
        o = ex.run_experiment(cfg, bundle=b, save=False, explain=False, final_model=False)
        g = golden[method]
        assert round(float(o.weights.abs().sum().sum()), 10) == pytest.approx(g["weights_sum"], abs=1e-6)
        assert round(float(o.net.sum()), 12) == pytest.approx(g["net_sum"], abs=1e-9)
        assert round(float(o.turnover.sum()), 10) == pytest.approx(g["turnover_sum"], abs=1e-8)
        assert "portfolio_allocation" not in o.report, "an existing method records no allocation section"


def test_a_rank_forecast_is_refused_by_the_output_contract_with_what_to_change():
    from src.ml import experiments as ex
    from tests.ml_helpers import make_bundle

    b = make_bundle(n_assets=2, n=1500, seed=3, alpha=1.0, macro=False)
    cfg = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5", "horizon": 5, "min_history": 60, "normalize": "cross_sectional_z"}, "split": {"min_train_days": 300, "refit_every": 126}, "model": {"kind": "ridge"},
           "signal": {"method": "portfolio_allocation", "holding_period": 1}}
    o = ex.run_experiment(cfg, bundle=b, save=False, explain=False, final_model=False)
    assert o.report["portfolio_allocation"]["counts"]["decisions"] > 0
    bad = copy.deepcopy(cfg)
    bad["dataset"]["target"] = "rank_5"
    with pytest.raises(Exception, match="rank|relative|fwd_ret"):
        ex.run_experiment(bad, bundle=b, save=False, explain=False, final_model=False)
