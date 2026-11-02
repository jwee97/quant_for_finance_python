"""Release 6, P2-04 and P2-16: the cost-aware optimiser as a position rule of the experiment pipeline.

The optimiser itself is checked against closed-form solutions in ``test_ml_allocation.py``. Here the wiring is: a forecast in return units and a covariance fitted as of the decision date, scaled to the same window;
the book held before each solve is the previous solution; the cost comes from the backtest section; a solve that gives no book is counted and reported; and a model whose output is not a return is refused.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, contracts as ct, experiments as xp, registry as rg, risk_models as rm, signals as sg
from tests.ml_helpers import make_bundle

ASSETS = list("ABCDE")


def market(n=330, seed=1, sigma=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(rng.normal(0.0, sigma, (n, len(ASSETS))), index=idx, columns=ASSETS)


def cfg(**over):
    base = {"method": "cost_aware_mean_variance", "gross": 1.0, "net": 0.0, "max_weight": 0.35, "holding_period": 1, "cost_bps": 0.0, "window_days": 5, "opt_every": 5, "risk_aversion": 5.0,
            "cov_method": "sample", "cov_window": 252}
    base.update(over)
    return base


def scores_on(rets, rows: dict):
    s = pd.DataFrame(np.nan, index=rets.index, columns=ASSETS)
    for i, v in rows.items():
        s.iloc[i] = v
    return s


MU = np.array([0.0004, 0.0002, -0.0002, -0.0004, 0.0])


def test_without_costs_the_book_is_the_closed_form_mean_variance_solution_in_the_forecast_windows_units():
    rets = market()
    s = scores_on(rets, {260: MU, 270: MU})
    w = sg.build_weights(s, cfg(), returns=rets)
    cov = rm.fit_asof(rets, rets.index[260], rm.CovSpec("sample", window=252)).matrix.loc[ASSETS, ASSETS] * 5.0         # per-day covariance scaled to the five-session forecast window
    inv = np.linalg.inv(cov.to_numpy())
    nu = (np.ones(5) @ inv @ MU) / (np.ones(5) @ inv @ np.ones(5))                                                      # the multiplier of the dollar-neutral constraint
    expected = inv @ (MU - nu) / 5.0                                                                                    # w = Sigma^-1 (mu - nu) / lambda
    assert np.abs(expected).max() < 0.35 and np.abs(expected).sum() < 1.0, "the fixture leaves the caps slack, so the closed form is the answer"
    assert np.allclose(w.iloc[260].to_numpy(), expected, atol=2e-5), (w.iloc[260].to_numpy(), expected)
    assert abs(w.iloc[260].sum()) < 1e-7 and np.allclose(w.iloc[260:265].to_numpy(), w.iloc[260].to_numpy()), "dollar neutral, and held between solves"
    assert np.allclose(w.iloc[:260].to_numpy(), 0.0), "no forecast, no position"
    log = w.attrs["allocation"]
    assert log["solves"] == 2 and log["optimal"] == 2 and log["failed"] == 0 and log["infeasible"] == 0 and log["opt_every"] == 5 and log["window_days"] == 5


def test_a_trading_cost_creates_a_no_trade_region_and_the_book_held_before_is_the_previous_solution():
    rets = market()
    s = scores_on(rets, {260: MU, 265: MU * 1.0, 270: MU * 1.0})
    dear = sg.build_weights(s, cfg(cost_bps=500.0), returns=rets)
    assert np.allclose(dear.to_numpy(), 0.0) and dear.attrs["allocation"]["no_trade"] == 3, "a cost of 5% of the traded weight is more than the forecast edge: nothing is traded, and each solve says so"
    cheap = sg.build_weights(s, cfg(cost_bps=0.0), returns=rets)
    mild = sg.build_weights(s, cfg(cost_bps=2.0), returns=rets)
    assert np.abs(cheap.iloc[260]).sum() > np.abs(mild.iloc[260]).sum() > 0.0, "a small cost trades a little less than no cost"
    flip = scores_on(rets, {260: MU, 265: -MU * 0.1})                                                                   # a small change of view
    small = sg.build_weights(flip, cfg(cost_bps=2.0), returns=rets)
    free = sg.build_weights(flip, cfg(cost_bps=0.0), returns=rets)
    move_cost = np.abs(small.iloc[265] - small.iloc[260]).sum()
    move_free = np.abs(free.iloc[265] - free.iloc[260]).sum()
    assert move_cost < move_free, "with a cost the second solve moves less from the book actually held than it would without one"


def test_a_solve_with_no_book_holds_the_previous_book_and_is_counted():
    rets = market()
    s = scores_on(rets, {260: MU, 265: MU})
    impossible = sg.build_weights(s, cfg(max_weight=0.1, gross=1.0, net=0.9), returns=rets)                             # net 0.9 with five names capped at 0.1: nothing satisfies the constraints
    log = impossible.attrs["allocation"]
    assert log["infeasible"] == 2 and log["held_after_failure"] == 2 and np.allclose(impossible.to_numpy(), 0.0), log
    rescued = sg.build_weights(s, cfg(max_weight=0.1, gross=1.0, net=0.9, alloc_fallback="equal_weight"), returns=rets)
    assert rescued.attrs["allocation"]["fallback_used"] == 2 and rescued.attrs["allocation"]["infeasible"] == 2, "a configured fallback is labelled and counted, never presented as an optimum"
    thin = scores_on(rets, {260: [0.0004, -0.0004, np.nan, np.nan, np.nan]})
    assert sg.build_weights(thin, cfg(), returns=rets).attrs["allocation"]["skipped_too_few_assets"] == 1
    young = scores_on(rets.iloc[:50], {10: MU})
    assert sg.build_weights(young, cfg(), returns=rets.iloc[:50]).attrs["allocation"]["skipped_no_covariance"] == 1, "no covariance can be fitted on ten days of history: nothing is traded on a guess"


def test_the_configuration_derives_the_cost_and_the_window_and_refuses_what_changes_the_units():
    c = mc.experiment_config({"signal": {"method": "cost_aware_mean_variance", "holding_period": 1}, "backtest": {"cost_bps": 3.0, "slippage_bps": 2.0}, "dataset": {"horizon": 10, "target": "fwd_ret_10"}}, defaults=mc.DEFAULTS)
    assert c["signal"]["cost_bps"] == 5.0 and c["signal"]["window_days"] == 10
    other = mc.experiment_config({"signal": {"method": "cost_aware_mean_variance", "holding_period": 1}, "backtest": {"cost_bps": 8.0, "slippage_bps": 2.0}, "dataset": {"horizon": 10, "target": "fwd_ret_10"}}, defaults=mc.DEFAULTS)
    assert mc.config_hash(c) != mc.config_hash(other), "the cost is part of what decides the book, so it is part of the hash"
    plain = mc.experiment_config({"signal": {"method": "proportional"}}, defaults=mc.DEFAULTS)
    assert "cost_bps" not in plain["signal"] and "window_days" not in plain["signal"], "the derived settings exist only for the method that uses them"
    for bad, msg in (({"holding_period": 5}, "holding_period to 1"), ({"risk_aversion": -1}, "risk_aversion"), ({"cov_method": "magic"}, "cov_method"), ({"cov_window": 20}, "cov_window"),
                     ({"alloc_fallback": "all_in"}, "alloc_fallback")):
        with pytest.raises(mc.MLConfigError, match=msg):
            mc.experiment_config({"signal": {"method": "cost_aware_mean_variance", "holding_period": 1, **bad}}, defaults=mc.DEFAULTS)
    assert mc.config_hash(mc.experiment_config({}, defaults=mc.DEFAULTS)) == mc.config_hash(mc.experiment_config({"signal": {}}, defaults=mc.DEFAULTS)), "no existing configuration hash moved"


def test_only_a_return_forecast_may_be_read_by_the_optimiser():
    sigcfg = {"method": "cost_aware_mean_variance", "net": 0.0}
    assert ct.check_signal_use("ridge", "regression", "fwd_ret_5", sigcfg) == "expected_return"
    assert ct.check_signal_use("ridge", "regression", "excess_ret_5", sigcfg) == "expected_return"
    with pytest.raises(ct.OutputContractError, match="not a return"):
        ct.check_signal_use("ridge", "regression", "rank_5", sigcfg)
    with pytest.raises(ct.OutputContractError, match="not a return"):
        ct.check_signal_use("lgbm_ranker", "regression", "fwd_ret_5", sigcfg)
    with pytest.raises(ct.OutputContractError, match="dollar neutral"):
        ct.check_signal_use("ridge", "regression", "resid_ret_5", {**sigcfg, "net": 0.3})
    assert ct.check_signal_use("ridge", "regression", "fwd_ret_5", {**sigcfg, "net": 0.3}) == "expected_return", "an absolute return forecast may carry a net exposure"
    assert ct.check_signal_use("lgbm_ranker", "regression", "fwd_ret_5", {**sigcfg, "reinterpret_outputs": True}) == "ranking_score", "the override exists and is recorded elsewhere"


def test_in_an_experiment_the_optimiser_respects_its_limits_reports_every_solve_and_trades_less_when_trading_is_dearer(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    bundle = make_bundle(10, 1150, seed=71, alpha=1.0)
    over = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
            "signal": {"method": "cost_aware_mean_variance", "holding_period": 1, "gross": 1.0, "net": 0.0, "max_weight": 0.3, "risk_aversion": 10.0, "cov_window": 126, "opt_every": 5},
            "backtest": {"cost_bps": 1.0, "slippage_bps": 1.0}}
    exp = xp.run_experiment(over, bundle=bundle, registry=reg, explain=False, final_model=False, robustness=False)
    rep = reg.report(exp.id)
    a = rep["allocation"]
    assert a["solves"] > 50 and a["optimal"] + a["infeasible"] + a["failed"] == a["solves"] and a["spec"]["cost"] == pytest.approx(2e-4) and a["window_days"] == 5
    pos = pd.read_parquet(reg.root / "experiments" / exp.id / "positions.parquet")
    assert pos.abs().sum(axis=1).max() <= 1.0 + 1e-6 and pos.sum(axis=1).abs().max() <= 1e-6 and pos.abs().max().max() <= 0.3 + 1e-9, "gross, net and position limits hold on every date"
    dear = xp.rebacktest(reg, exp.id, {"backtest": {"cost_bps": 60.0, "slippage_bps": 1.0}, "name": "dear"})
    cheap_t, dear_t = exp.metrics["oos"]["ann_turnover"], dear.metrics["oos"]["ann_turnover"]
    assert dear_t < cheap_t, (cheap_t, dear_t)
    assert reg.report(dear.id)["allocation"]["spec"]["cost"] == pytest.approx(61e-4), "the cost the optimiser weighs follows the backtest section it was re-run with"
