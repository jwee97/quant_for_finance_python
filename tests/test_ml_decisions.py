"""The buy / sell / hold decision engine, its ledger and the engine's event-driven mode, on synthetic worlds whose answers are known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import decisions as D
from src.ml import trade_log as T
from src.ml.backtest import run_backtest
from src.ml.config import MLConfigError, experiment_config

BT = {"rebalance": "on_change", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": None, "max_leverage": 1.5, "extra_lag": 0}
COLS = list("ABCDEFGH")


def world(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    r = pd.DataFrame(rng.normal(0.0003, 0.01, (n, len(COLS))), idx, COLS)
    score = r.shift(-2).rolling(5).mean().fillna(0.0) + rng.normal(0, 0.001, r.shape)               # a noisy look at the future: only a test world, never a model
    return r, score


def pol(**kw):
    return D.policy_from({f"decision_{k}": v for k, v in kw.items()})


def test_long_only_policy_respects_limits_and_never_shorts():
    r, s = world()
    w, dec = D.run(s, r, pol(max_positions=3, max_position=0.4, long_exposure=0.9), vol=r.rolling(20).std(), lag=1)
    assert (w >= -1e-12).all().all() and w.sum(axis=1).max() <= 0.9 + 1e-9 and w.max().max() <= 0.4 + 1e-9
    assert ((w > 1e-9).sum(axis=1) <= 3).all()
    assert not dec["action"].isin(["SHORT", "COVER"]).any()


def test_selling_a_long_is_not_opening_a_short():
    r, s = world()
    _, dec = D.run(s, r, pol(allow_short=True, short_exposure=0.5, long_exposure=0.5), vol=r.rolling(20).std(), lag=1)
    assert {"BUY", "SELL", "SHORT", "COVER"} <= set(dec["action"])
    first = dec[dec["action"].isin(["SELL", "SHORT"])].groupby("action")["prev_weight"].first()
    assert first["SELL"] > 0 and first["SHORT"] == 0                          # a sell closes something held long; a short opens from flat


def test_entries_and_rule_exits_happen_only_on_rebalance_days_but_forced_exits_are_daily():
    r, s = world()
    _, dec = D.run(s, r, pol(rebalance_days=5, max_holding=7), vol=r.rolling(20).std(), lag=1)
    pos = {d: i for i, d in enumerate(r.index)}
    assert all(pos[d] % 5 == 0 for d in dec.loc[dec["action"].isin(["BUY", "SELL"]), "date"])
    assert any(pos[d] % 5 != 0 for d in dec.loc[dec["action"] == "TIME_EXIT", "date"])


def test_hysteresis_a_name_between_the_exit_and_entry_lines_is_held_not_sold():
    idx = pd.bdate_range("2021-01-01", periods=30)
    r = pd.DataFrame(0.0, idx, list("ABCDE"))
    s = pd.DataFrame(np.tile([5.0, 4.0, 3.0, 2.0, 1.0], (30, 1)), idx, r.columns)         # A is best
    s.loc[idx[5]:, "A"] = 3.6                                                            # from day 5 A drops to second place (rank 80%), still inside the exit line
    s.loc[idx[5]:, "B"] = 5.0
    _, dec = D.run(s, r, pol(entry_quantile=0.2, exit_quantile=0.5, max_positions=1, rebalance_days=5), lag=1)
    a = dec[dec["asset"] == "A"]
    assert a.iloc[0]["action"] == "BUY" and "SELL" not in set(a["action"])                # kept through the drop
    assert "HOLD" in set(a["action"])


def test_stop_loss_is_measured_from_the_executed_entry_price_and_blocks_reentry_for_the_cooldown():
    idx = pd.bdate_range("2021-01-01", periods=60)
    r = pd.DataFrame(0.0, idx, list("AB"))
    r.loc[idx[9], "A"] = -0.30                                                          # a crash on day 9
    s = pd.DataFrame({"A": 2.0, "B": 1.0}, index=idx)                                    # A is always the favourite: only the cooldown can keep it out
    vol = pd.DataFrame(0.01, idx, r.columns)
    _, dec = D.run(s, r, pol(entry_quantile=0.5, exit_quantile=0.5, max_positions=2, rebalance_days=5, stop_loss_vol=2.0, cooldown=10, horizon=1), vol=vol, lag=1)
    a = dec[dec["asset"] == "A"]
    stop = a[a["action"] == "STOP_LOSS"].iloc[0]
    assert stop["date"] == idx[9]                                                       # the first close that shows the loss, not a day later
    assert stop["entry_price"] == pytest.approx(1.0) and stop["price"] == pytest.approx(0.7)
    rebuy = a[(a["action"] == "BUY") & (a["date"] > stop["date"])].iloc[0]["date"]
    pos = {d: i for i, d in enumerate(idx)}
    assert pos[rebuy] - pos[stop["date"]] > 10                                          # not before the cooldown has run (and then only on a rebalance day)


def test_no_decision_depends_on_a_price_after_its_own_day():
    r, s = world()
    cut = 150
    w1, d1 = D.run(s, r, pol(stop_loss_vol=2.0, max_holding=12, allow_short=True, short_exposure=0.4, long_exposure=0.6), vol=r.rolling(20).std(), lag=1)
    r2, s2 = r.copy(), s.copy()
    r2.iloc[cut + 1:] = np.random.default_rng(9).normal(0, 0.05, r2.iloc[cut + 1:].shape)     # rewrite the future
    s2.iloc[cut + 1:] = np.random.default_rng(10).normal(0, 1.0, s2.iloc[cut + 1:].shape)
    w2, d2 = D.run(s2, r2, pol(stop_loss_vol=2.0, max_holding=12, allow_short=True, short_exposure=0.4, long_exposure=0.6), vol=r2.rolling(20).std(), lag=1)
    assert np.allclose(w1.iloc[:cut + 1].to_numpy(), w2.iloc[:cut + 1].to_numpy())
    assert d1[d1["date"] <= r.index[cut]].reset_index(drop=True).equals(d2[d2["date"] <= r.index[cut]].reset_index(drop=True))


def test_the_ledger_reconciles_to_the_engine_to_the_last_basis_point():
    r, s = world()
    p = pol(stop_loss_vol=2.0, max_holding=15, allow_short=True, short_exposure=0.5, long_exposure=0.5, max_positions=3)
    w, dec = D.run(s, r, p, vol=r.rolling(20).std(), lag=1)
    res = run_backtest(w, r, None, BT, 1)
    fills = T.executions(res.weights, res.trades, r, 7.0, dec, 1)
    trips = T.round_trips(res.weights, r, fills, 7.0)
    assert not fills.duplicated(["date", "asset"]).any()
    assert fills["cost"].sum() == pytest.approx(res.costs.sum(), abs=1e-12)
    assert trips["cost"].sum() == pytest.approx(res.costs.sum(), abs=1e-12)
    assert trips["contribution"].sum() == pytest.approx(res.gross_returns.sum(), abs=1e-10)
    assert trips["net_contribution"].sum() == pytest.approx(res.net_returns.sum(), abs=1e-10)


def test_every_executed_trade_names_its_decision_or_says_it_was_a_drift_correction():
    r, s = world()
    w, dec = D.run(s, r, pol(max_positions=3), vol=r.rolling(20).std(), lag=1)
    res = run_backtest(w, r, None, BT, 1)
    fills = T.executions(res.weights, res.trades, r, 7.0, dec, 1)
    assert fills["action"].notna().all() and fills["reason"].notna().all()
    assert set(fills["action"]) <= {*D.ACTIONS, "DRIFT"} | {a for a in fills["action"] if " + " in a}
    opens = fills[fills["kind"] == "OPEN_LONG"]
    assert (opens["action"].str.contains("BUY")).all()
    assert (pd.to_datetime(opens["date"]) > pd.to_datetime(opens["decision_date"])).all()     # executed strictly after it was decided


def test_event_driven_rebalance_trades_only_when_a_target_changes():
    idx = pd.bdate_range("2021-01-01", periods=40)
    r = pd.DataFrame(np.random.default_rng(0).normal(0.001, 0.01, (40, 2)), idx, ["A", "B"])
    w = pd.DataFrame({"A": 0.5, "B": 0.0}, index=idx)
    w.loc[idx[20]:, "B"] = 0.3                                                          # one change on day 20
    res = run_backtest(w, r, None, BT, 1)
    traded_days = res.turnover[res.turnover > 1e-12].index
    assert list(traded_days) == [idx[1], idx[21]]                                      # the entry and the change, each one lag later; nothing in between


def test_perfect_foresight_is_a_diagnostic_control_that_pays_only_at_the_right_delay():
    r, _ = world(400)
    for k, expect in [(2, True), (3, False), (1, False)]:
        s = r.shift(-k).rolling(1).mean().fillna(0.0)
        w, _dec = D.run(s, r, pol(max_positions=3, entry_quantile=0.4, exit_quantile=0.4, rebalance_days=1, long_exposure=0.6, allow_short=True, short_exposure=0.6, cooldown=0), lag=1)
        net = run_backtest(w, r, None, {**BT, "cost_bps": 0.0, "slippage_bps": 0.0}, 1).net_returns.dropna()
        sh = net.mean() / net.std() * np.sqrt(252)
        assert (sh > 8) if expect else (abs(sh) < 3)


def test_the_policy_is_validated_and_the_method_is_accepted_by_the_experiment_configuration():
    cfg = experiment_config({"signal": {"method": "decision_policy", "decision_entry_quantile": 0.2, "decision_exit_quantile": 0.4}, "backtest": {"rebalance": "on_change"}})
    assert cfg["signal"]["lag_total"] >= 1                                              # derived from the other sections, as for the other stateful rules
    for bad in ({"decision_entry_quantile": 0.6}, {"decision_exit_quantile": 0.1}, {"decision_sizing": "kelly"}, {"decision_short_exposure": 0.5}, {"decision_max_holding": 0}):
        with pytest.raises(MLConfigError):
            experiment_config({"signal": {"method": "decision_policy", **bad}})


def test_the_experiment_pipeline_hands_the_policy_annualised_volatility_and_the_stop_still_fires_at_the_daily_scale():
    """``build_weights`` is given the platform's annualised trailing volatility; the policy's stops are in daily units. A stop at 2 volatilities on a 1%-a-day asset must fire on a -3% day, not wait for -90%."""
    from src.ml.signals import build_weights

    idx = pd.bdate_range("2021-01-01", periods=60)
    r = pd.DataFrame(0.0, idx, list("ABC"))
    r.loc[idx[9], "A"] = -0.05
    s = pd.DataFrame({"A": 3.0, "B": 2.0, "C": 1.0}, index=idx)
    annual = pd.DataFrame(0.01 * np.sqrt(252.0), idx, r.columns)                       # what trailing_vol returns for a 1%-a-day asset
    cfg = {"method": "decision_policy", "decision_entry_quantile": 0.5, "decision_exit_quantile": 0.5, "decision_max_positions": 2, "decision_rebalance_days": 5, "decision_stop_loss_vol": 2.0,
           "decision_horizon": 1, "decision_cooldown": 0, "lag_total": 1}
    w = build_weights(s, cfg, vol=annual, returns=r)
    dec = w.attrs["decisions"]
    stop = dec[(dec["asset"] == "A") & (dec["action"] == "STOP_LOSS")]
    assert len(stop) == 1 and stop.iloc[0]["date"] == idx[9]


def test_random_scores_with_the_models_persistence_trade_at_the_models_pace_and_carry_no_skill():
    from src.ml import controls as ct

    r, s = world(900)
    smooth = s.rolling(10).mean().fillna(0.0)                                          # persistent scores
    rho = ct.rank_persistence(smooth)
    assert rho > 0.5 and abs(ct.rank_persistence(pd.DataFrame(np.random.default_rng(0).normal(size=s.shape), s.index, s.columns))) < 0.1
    rs = ct.random_scores(smooth, rho, np.random.default_rng(1))
    assert abs(ct.rank_persistence(rs) - rho) < 0.12 and rs.shape == smooth.shape
    ic = pd.concat([rs.shift(1).stack(), r.stack()], axis=1).dropna().corr(method="spearman").iloc[0, 1]
    assert abs(ic) < 0.05, "random scores carry no information about returns"
