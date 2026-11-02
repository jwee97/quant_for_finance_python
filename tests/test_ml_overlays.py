"""Release 7, P2-05 and P2-09: the no-trade region and the net-edge trade filter.

The decision rule is worked by hand on a two-asset book where the gain of a move and its cost are known numbers, so each threshold is exact. The overlays are then run through time on a small market to check the
bookkeeping: what traded, what was held, what a hard control forced, and that neither rule ever reads an outcome that had not settled."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.execution import rebalance_dates
from src.ml import config as mc, contracts as ct, experiments as xp, monitoring as mon, notrade as nt, registry as rg, signals as sg, tradefilter as tf
from tests.ml_helpers import make_bundle

A2 = ["A", "B"]
COV = pd.DataFrame(np.diag([0.04, 0.04]), index=A2, columns=A2)           # variance of each name over the holding window
ZERO = pd.Series([0.0, 0.0], index=A2)


def decide(mu=0.02, target=(0.1, 0.0), current=(0.0, 0.0), **kw):
    kw = {"cost": 0.001, "margin": 0.0, "risk_aversion": 5.0, **kw}
    return nt.decide(pd.Series(current, index=A2), pd.Series(target, index=A2), pd.Series([mu, 0.0], index=A2), COV, **kw)


def test_the_move_is_made_only_when_its_gain_exceeds_its_cost_with_the_margin():
    d = decide(mu=0.02)                                                                          # U(0.1, 0) = 0.02 * 0.1 - 2.5 * 0.04 * 0.01 = 0.001; cost = 0.001 * 0.1 = 0.0001
    assert d["gain"] == pytest.approx(0.001) and d["cost"] == pytest.approx(1e-4) and d["threshold"] == pytest.approx(1e-4) and d["traded"] is True
    assert d["utility_hold"] == 0.0 and d["utility_target"] == pytest.approx(0.001) and list(d["weights"]) == [0.1, 0.0]
    for m, traded in ((5.0, True), (8.9, True), (9.5, False), (20.0, False)):                   # the gain is ten times the cost: a margin of nine times is the last that passes
        d = decide(mu=0.02, margin=m)
        assert d["traded"] is traded, (m, d["reason"])
        assert d["threshold"] == pytest.approx((1 + m) * 1e-4)
        assert list(d["weights"]) == ([0.1, 0.0] if traded else [0.0, 0.0])
    low = decide(mu=0.0005)                                                                       # a low edge: the risk of the position costs more than it earns
    assert low["gain"] < 0 and low["traded"] is False and "does not exceed" in low["reason"]
    high = decide(mu=0.05, margin=3.0)
    assert high["traded"] is True and high["gain"] == pytest.approx(0.05 * 0.1 - 0.001)
    assert decide(mu=0.02, target=(0.0, 0.0))["reason"] == "already at the target"


def test_hysteresis_a_trading_episode_continues_on_less_than_it_took_to_start():
    at_rest = decide(mu=0.02, margin=9.5, hold_margin=5.0, active=False)
    in_episode = decide(mu=0.02, margin=9.5, hold_margin=5.0, active=True)
    assert at_rest["traded"] is False and at_rest["active"] is False and at_rest["threshold"] == pytest.approx(10.5e-4)
    assert in_episode["traded"] is True and in_episode["active"] is True and in_episode["threshold"] == pytest.approx(6e-4)
    held = decide(mu=0.0005, margin=9.5, hold_margin=5.0, active=True)
    assert held["traded"] is False and "hysteresis" in held["reason"] and held["active"] is False, "when the gain falls away the episode ends"


def test_hard_controls_override_the_rule_and_are_recorded_with_their_reason():
    forced = decide(mu=0.0, current=(0.3, 0.3), target=(0.3, 0.3), tradable=pd.Series([True, False], index=A2))
    assert list(forced["weights"]) == [0.3, 0.0] and forced["forced"] == [{"asset": "B", "from": 0.3, "to": 0.0, "reason": "the name can no longer be held"}], "closed whether or not it pays"
    cut = decide(current=(0.5, 0.0), target=(0.3, 0.0), position_max=0.35, margin=100.0)
    assert [f["reason"] for f in cut["forced"]] == ["above the position limit"] and cut["weights"]["A"] == pytest.approx(0.3) or cut["weights"]["A"] == pytest.approx(0.35)
    scaled = decide(current=(0.4, 0.4), target=(0.4, 0.4), gross_max=0.6, position_max=0.5)
    assert scaled["weights"].abs().sum() == pytest.approx(0.6) and scaled["forced"][0]["asset"] == "*" and "scaled" in scaled["forced"][0]["reason"]
    infeasible = decide(mu=0.5, current=(0.2, 0.0), target=(0.6, 0.0), position_max=0.35)
    assert infeasible["target_feasible"] is False and infeasible["traded"] is False and infeasible["reason"].startswith("target infeasible") and list(infeasible["weights"]) == [0.2, 0.0]
    both = decide(mu=0.5, current=(0.5, 0.0), target=(0.6, 0.0), position_max=0.35)
    assert list(both["weights"]) == [0.35, 0.0] and "forced trades were still made" in both["reason"], "an infeasible target never blocks the forced trade"
    held = decide(mu=0.5, target=(0.0, 0.2), tradable=pd.Series([True, False], index=A2))
    assert held["target_feasible"] is False and "cannot be held" in held["infeasible_reasons"][0]
    with pytest.raises(ValueError):
        decide(margin=-1.0)


def market(n=330, seed=3, k=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    cols = [f"S{i}" for i in range(k)]
    return pd.DataFrame(rng.normal(0.0, 0.01, (n, k)), index=idx, columns=cols)


def cfg(**over):
    base = {"method": "top_n_long_short", "gross": 1.0, "net": 0.0, "max_weight": 0.5, "holding_period": 1, "cost_bps": 10.0, "window_days": 5, "label_offset": 6, "rebalance": "daily", "lag_total": 1,
            "cov_method": "sample", "cov_window": 252, "risk_aversion": 5.0}
    base.update(over)
    return base


def flipping_targets(rets):
    t = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    t.iloc[::2, 0], t.iloc[::2, 1] = 0.5, -0.5                                                    # one pairing on even days,
    t.iloc[1::2, 2], t.iloc[1::2, 3] = 0.5, -0.5                                                  # a different one on odd days
    return t


def flipping_scores(rets, big=0.005):
    s = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    s.iloc[::2, 0], s.iloc[::2, 1] = big, -big                                                    # on even days the forecast favours the first pairing,
    s.iloc[1::2, 2], s.iloc[1::2, 3] = big, -big                                                  # on odd days the other
    s.iloc[:260] = np.nan                                                                         # no forecast, no decision
    return s


def test_through_time_a_dear_cost_stops_the_book_flipping_and_every_decision_is_counted():
    rets = market()
    targets = flipping_targets(rets)
    scores = flipping_scores(rets)
    free, flog = nt.apply(targets, scores, rets, None, cfg(no_trade_margin=0.0, cost_bps=0.0))
    dear, dlog = nt.apply(targets, scores, rets, None, cfg(no_trade_margin=0.5, cost_bps=20.0))
    assert dlog["decisions"] == flog["decisions"] == 69 and np.allclose(free.iloc[:260].to_numpy(), 0.0), "the same decisions are taken on the same dates (every row from 260 to the penultimate)"
    turn = lambda w: float(w.diff().abs().sum().sum())                                              # noqa: E731
    assert flog["traded"] == 69, "at no cost the book follows the forecast every day"
    assert dlog["traded"] == 1 and dlog["held_below_threshold"] == 34, "at 20 bps with a margin of one half the first move pays (gain 0.0044 against 0.003) and no flip does (0.005 against 0.006)"
    assert dlog["decisions"] == dlog["traded"] + dlog["held_below_threshold"] + 34, "the other 34 are the days the proposal is the book already held"
    assert turn(dear) < turn(free) / 10 and dlog["cost_avoided"] == pytest.approx(34 * 0.004) and dlog["utility_given_up"] == pytest.approx(34 * 0.005, rel=0.05)
    assert dlog["window_days"] == 5 and dlog["cost_per_unit"] == pytest.approx(0.002) and dlog["version"] == "notrade-v1"


def test_a_name_that_stops_being_investable_is_closed_by_force_and_the_log_says_why():
    rets = market()
    targets = flipping_targets(rets)
    targets.iloc[:, 0], targets.iloc[:, 1] = 0.0, 0.0
    targets.iloc[:, 2], targets.iloc[:, 3] = 0.5, -0.5
    scores = pd.DataFrame(0.005, index=rets.index, columns=rets.columns)
    scores.iloc[:, 3] = -0.005
    scores.iloc[:260] = np.nan
    inv = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    inv.iloc[290:, 2] = False                                                                     # S2 is delisted at row 290
    out, log = nt.apply(targets.mask(~inv, 0.0), scores, rets, inv, cfg(no_trade_margin=1.0))
    assert out.iloc[289, 2] != 0.0 and (out.iloc[290:, 2] == 0.0).all(), "closed at the first decision after it could not be held"
    assert log["forced_trades"] >= 1 and log["forced_reasons"].get("the name can no longer be held", 0) >= 1 and log["events"][0]["forced"][0]["asset"] == "S2"


def test_decisions_fall_on_the_dates_the_engine_trades_lag_sessions_later():
    rets = market()
    for freq in ("weekly", "monthly"):
        marks = set(rebalance_dates(rets.index, freq))
        dates = nt.decision_dates(rets.index, freq, 2)
        pos = rets.index.get_indexer(dates)
        assert len(dates) > 5 and all(rets.index[p + 2] in marks for p in pos), freq
    daily = nt.decision_dates(rets.index, "daily", 1)
    assert len(daily) == len(rets) - 1


def test_the_trade_verdicts_are_worked_by_hand():
    mu = pd.Series([0.01, 0.0005, -0.01], index=list("ABC"))
    current = pd.Series([0.0, 0.0, 0.2], index=list("ABC"))
    target = pd.Series([0.3, 0.3, 0.0], index=list("ABC"))
    v = tf.accept(mu, current, target, cost=0.001, cost_margin=1.0, scale=0.004, k=1.0)
    assert list(v["change"]) == pytest.approx([0.3, 0.3, -0.2]) and list(v["edge_per_unit"]) == pytest.approx([0.01, 0.0005, 0.01])
    assert list(v["net_edge_per_unit"]) == pytest.approx([0.008, -0.0015, 0.008]) and list(v["z"]) == pytest.approx([2.0, -0.375, 2.0])
    assert list(v["accepted"]) == [True, False, True], "selling a name the forecast turned against passes; buying one whose edge is below the margined cost does not"
    strict = tf.accept(mu, current, target, cost=0.001, cost_margin=1.0, scale=0.004, k=3.0)
    assert list(strict["accepted"]) == [False, False, False]
    unchanged = tf.accept(mu, target, target, cost=0.001, cost_margin=1.0, scale=0.004, k=3.0)
    assert unchanged["accepted"].all(), "a name that is not traded is never rejected"


def test_the_forecast_error_scale_uses_only_settled_outcomes():
    rets = market(n=400, k=6)
    rng = np.random.default_rng(5)
    scores = pd.DataFrame(rng.normal(0, 0.01, rets.shape), index=rets.index, columns=rets.columns)
    base = tf.forecast_error_scale(scores, rets, 5, 1, window=120, min_dates=40)
    prices = (1 + rets).cumprod()
    fwd = prices.shift(-6) / prices.shift(-1) - 1.0
    err2 = ((scores - fwd) ** 2).mean(axis=1)
    t = 300
    ref = np.sqrt(err2.iloc[: t - 6 + 1].iloc[-120:].mean())                                       # decisions s with s + 6 <= t, the last 120 of them
    assert base.iloc[t] == pytest.approx(ref, rel=1e-9)
    assert base.iloc[:40].isna().all(), "nothing is estimated before enough decisions have settled"
    changed = rets.copy()
    changed.iloc[330:] = rng.normal(0, 0.05, changed.iloc[330:].shape)                              # a different future
    again = tf.forecast_error_scale(scores, changed, 5, 1, window=120, min_dates=40)
    assert np.allclose(base.iloc[:330].to_numpy(), again.iloc[:330].to_numpy(), equal_nan=True), "the scale at t does not move when what follows t changes"


def test_through_time_the_filter_rejects_what_it_should_and_reports_coverage_and_diversification():
    rets = market(n=420, k=6)
    rng = np.random.default_rng(8)
    sig = pd.DataFrame(rng.normal(0, 0.004, rets.shape), index=rets.index, columns=rets.columns)
    prices = (1 + rets).cumprod()
    fwd = prices.shift(-6) / prices.shift(-1) - 1.0
    scores = (0.5 * fwd.fillna(0.0) + sig).where(pd.Series(rets.index >= rets.index[200], index=rets.index), axis=0)                    # forecasts that carry some information
    targets = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for i in range(len(rets)):
        row = scores.iloc[i]
        if row.notna().all():
            top, bot = list(row.nlargest(2).index), list(row.nsmallest(2).index)
            targets.loc[targets.index[i], top], targets.loc[targets.index[i], bot] = 0.25, -0.25
    out0, log0 = tf.apply(targets, scores, rets, None, cfg(trade_filter_k=0.0))
    out3, log3 = tf.apply(targets, scores, rets, None, cfg(trade_filter_k=3.0))
    assert log0["proposed"] > 100 and log3["proposed"] > 100, "the proposals are the changes from the book each filter actually held, so their numbers differ"
    assert 0.0 <= log3["coverage_by_count"] < log0["coverage_by_count"] <= 1.0 and log3["coverage_by_turnover"] < log0["coverage_by_turnover"] <= 1.0
    assert "lower coverage is not better performance" in log3["note"]
    assert 0 < log3["mean_hhi_filtered"] <= 1.0 and 0 < log3["mean_hhi_unfiltered"] <= 1.0 and log3["mean_positions_filtered"] > 0, "the concentration with and without the filter is reported; rejecting an exit can leave more names held, not fewer"
    assert log0["no_uncertainty_estimate"] > 0 and log0["dates"] > log0["no_uncertainty_estimate"], "before enough decisions have settled nothing is filtered, and that is counted"
    turn = lambda w: float(w.diff().abs().sum().sum())                                              # noqa: E731
    assert turn(out3) < turn(out0)


def test_the_configuration_derives_what_the_overlays_need_and_refuses_what_makes_no_sense():
    c = mc.experiment_config({"signal": {"no_trade_margin": 1.0, "trade_filter_k": 1.5}, "backtest": {"cost_bps": 3.0, "slippage_bps": 2.0, "rebalance": "weekly", "extra_lag": 1}, "dataset": {"horizon": 10, "target": "fwd_ret_10"}},
                              defaults=mc.DEFAULTS)
    s = c["signal"]
    assert s["cost_bps"] == 5.0 and s["window_days"] == 10 and s["label_offset"] == 11 and s["rebalance"] == "weekly" and s["lag_total"] == 2
    plain = mc.experiment_config({}, defaults=mc.DEFAULTS)
    assert not set(sg.DERIVED_KEYS) & set(plain["signal"]), "a configuration that uses no overlay carries none of the derived settings"
    for bad, msg in (({"no_trade_margin": -1.0}, "no_trade_margin"), ({"no_trade_margin": 1.0, "no_trade_hold_margin": 2.0}, "hold_margin"), ({"trade_filter_k": -1.0}, "trade_filter_k"),
                     ({"trade_filter_k": 1.0, "trade_filter_window": 5}, "trade_filter_window")):
        with pytest.raises(mc.MLConfigError, match=msg):
            mc.experiment_config({"signal": bad}, defaults=mc.DEFAULTS)


def test_only_a_return_forecast_may_feed_the_overlays():
    for key, val in (("no_trade_margin", 1.0), ("trade_filter_k", 1.0)):
        sigcfg = {"method": "top_n_long_short", "net": 0.0, key: val}
        assert ct.check_signal_use("ridge", "regression", "fwd_ret_5", sigcfg) == "expected_return"
        with pytest.raises(ct.OutputContractError, match="not a return"):
            ct.check_signal_use("lgbm_ranker", "regression", "fwd_ret_5", sigcfg)
        with pytest.raises(ct.OutputContractError, match="not a return"):
            ct.check_signal_use("ridge", "regression", "rank_5", sigcfg)
        with pytest.raises(ct.OutputContractError, match="dollar neutral"):
            ct.check_signal_use("ridge", "regression", "excess_ret_5", {**sigcfg, "net": 0.2})


def test_in_an_experiment_the_overlays_are_reported_and_a_forward_test_refuses_rules_that_need_state(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    bundle = make_bundle(10, 1150, seed=81, alpha=1.0)
    over = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
            "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1, "no_trade_margin": 1.0, "trade_filter_k": 0.5, "cov_window": 126}, "backtest": {"cost_bps": 5.0, "slippage_bps": 2.0, "rebalance": "weekly"}}
    exp = xp.run_experiment(over, bundle=bundle, registry=reg, explain=False, robustness=False)
    rep = reg.report(exp.id)
    assert rep["overlays"]["trade_filter"]["k"] == 0.5 and rep["overlays"]["no_trade"]["decisions"] > 20 and rep["overlays"]["no_trade"]["cost_per_unit"] == pytest.approx(7e-4)
    plain = xp.rebacktest(reg, exp.id, {"signal": {"no_trade_margin": None, "trade_filter_k": None}, "name": "plain"})
    assert "overlays" not in reg.report(plain.id) and plain.cfg["signal"].get("no_trade_margin") is None
    assert exp.metrics["oos"]["ann_turnover"] < plain.metrics["oos"]["ann_turnover"], "the overlays traded less than the same predictions without them"
    tracker = mon.ForwardTracker(reg, exp.id)
    with pytest.raises(ValueError, match="forward testing does not support"):
        tracker.step(bundle)
