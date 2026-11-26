"""From a signal to a trade to the money it made: the audit trail of a weight backtest reconciles to the result and shows the timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.audit import reconcile, trade_trail
from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine


def world(n=160, k=5, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = [f"A{i}" for i in range(k)]
    ret = pd.DataFrame(rng.normal(0.0004, 0.01, (n, k)), index=idx, columns=cols)
    signal = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=cols)
    ranks = signal.rank(axis=1)
    targets = (ranks.sub(ranks.mean(axis=1), axis=0)).div(ranks.sub(ranks.mean(axis=1), axis=0).abs().sum(axis=1), axis=0)          # dollar neutral, gross one
    return idx, ret, signal, targets


def run(rebalance="monthly", lag=1, bps=10.0, drift_basis="gross"):
    idx, ret, signal, targets = world()
    eng = BacktestEngine(signal_lag=lag, rebalance=rebalance, weight_drift=True, drift_basis=drift_basis, min_assets=1, cost_model=LinearCostModel(cost_bps=bps))
    res = eng.run(targets, ret)
    return res, targets, ret, signal


@pytest.mark.parametrize("rebalance,lag,basis", [("monthly", 1, "gross"), ("monthly", 2, "equity"), ("daily", 1, "gross"), ("weekly", 1, "equity")])
def test_the_trail_reconciles_to_the_results_gross_return_and_costs(rebalance, lag, basis):
    res, targets, ret, signal = run(rebalance, lag, 10.0, basis)
    trail = trade_trail(res, targets, ret, signal)
    rec = reconcile(trail, res, 1e-9)
    assert rec["reconciled"], rec
    assert rec["trades"] == len(trail) and rec["trades"] > 0


def test_every_trade_carries_its_decision_date_signal_and_target_and_the_decision_precedes_the_trade_by_the_lag():
    res, targets, ret, signal = run("monthly", 2)
    trail = trade_trail(res, targets, ret, signal)
    real = trail.dropna(subset=["trade_date"])
    idx = res.weights.index
    for _, r in real.iterrows():
        assert idx.get_loc(r["trade_date"]) - idx.get_loc(r["decision_date"]) == 2
        assert r["signal"] == pytest.approx(signal.at[r["decision_date"], r["asset"]]) and r["target"] == pytest.approx(targets.at[r["decision_date"], r["asset"]])
    # on a rebalance the post weight is the target of the decision date
    rebal = real[real["trade_date"] == real["trade_date"].iloc[len(real) // 2]]
    assert np.allclose(rebal["post_weight"], rebal["target"])
    assert np.allclose(rebal["pre_weight"] + rebal["traded"], rebal["post_weight"])


def test_a_cost_is_the_assets_rate_times_the_weight_traded_and_a_per_asset_rate_is_honoured():
    res, targets, ret, signal = run("monthly", 1, 10.0)
    trail = trade_trail(res, targets, ret, signal)
    assert np.allclose(trail["cost"], trail["traded"].abs() * 10.0 / 1e4)
    idx, ret2, sig2, tg2 = world()
    eng = BacktestEngine(signal_lag=1, rebalance="monthly", min_assets=1, cost_model=LinearCostModel(cost_bps=10.0, per_asset_bps={"A0": 50.0}))
    r2 = eng.run(tg2, ret2)
    rates = {c: (50.0 if c == "A0" else 10.0) / 1e4 for c in ret2.columns}
    t2 = trade_trail(r2, tg2, ret2, sig2, cost_rates=rates)
    assert reconcile(t2, r2)["reconciled"]
    assert reconcile(trade_trail(r2, tg2, ret2, sig2), r2)["cost_gap"] != 0.0, "without the per-asset rates the trail says it does not reconcile and by how much, instead of hiding it"


def test_a_trades_pnl_is_the_held_weight_times_the_return_until_the_next_trade_in_that_asset():
    res, targets, ret, signal = run("monthly", 1)
    trail = trade_trail(res, targets, ret, signal)
    a = "A1"
    rows = trail[trail["asset"] == a].dropna(subset=["trade_date"]).reset_index(drop=True)
    first, second = rows.iloc[0], rows.iloc[1]
    idx = res.weights.index
    lo, hi = idx.get_loc(first["trade_date"]), idx.get_loc(second["trade_date"])
    want = float((res.weights[a].shift(1).iloc[lo + 1: hi + 1] * ret[a].iloc[lo + 1: hi + 1]).sum())
    assert first["pnl"] == pytest.approx(want, abs=1e-12) and first["days_held"] == hi - lo


def test_a_signal_after_the_decision_date_changes_nothing_in_a_trades_row():
    res, targets, ret, signal = run("monthly", 1)
    t1 = trade_trail(res, targets, ret, signal)
    later = signal.copy()
    later.iloc[100:] = 99.0
    t2 = trade_trail(res, targets, ret, later)
    early = t1["decision_date"].notna() & (t1["decision_date"] < signal.index[100])
    assert np.allclose(t1.loc[early, "signal"], t2.loc[early, "signal"])
