"""A cost of holding a position (the rolls of a futures contract): charged every day on the size held, long or short, by the engine, and worked out per root from its contract terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import futures_analytics as fa
from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine
from src.utils.config import load_config


def _run(weights_value: float, holding):
    days = pd.bdate_range("2024-01-01", periods=40)
    prices = pd.DataFrame({"A": 100.0 * np.cumprod(np.full(40, 1.001)), "B": 50.0 * np.cumprod(np.full(40, 0.999))}, index=days)
    weights = pd.DataFrame({"A": weights_value, "B": -weights_value}, index=days)
    engine = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=False, min_assets=1, cost_model=LinearCostModel(cost_bps=0.0), holding_cost=holding)
    return engine.run(weights, prices.pct_change(), name="t")


def test_a_holding_cost_is_charged_daily_on_the_size_held_and_a_short_pays_it_too():
    free, charged = _run(0.5, None), _run(0.5, {"A": 0.0252, "B": 0.0504})
    gap = (free.net_returns - charged.net_returns).dropna()
    held = gap[gap > 0]
    assert len(held) >= 30 and held.to_numpy() == pytest.approx(0.5 * 0.0252 / 252 + 0.5 * 0.0504 / 252), "half the book long A, half short B: both pay their own rate"
    assert charged.gross_returns.equals(free.gross_returns) and float(charged.costs.sum()) == pytest.approx(float(gap.sum()))
    assert (free.net_returns - _run(0.5, {"C": 0.5}).net_returns).abs().max() == 0.0, "a cost for an asset that is not held changes nothing"
    assert (free.net_returns - _run(0.5, {"A": -0.5}).net_returns).abs().max() == 0.0, "a holding cost is never a rebate"
    double = (free.net_returns - _run(1.0, {"A": 0.0252, "B": 0.0504}).net_returns).dropna()
    assert double[double > 0].to_numpy() == pytest.approx(2 * held.iloc[0])


def test_the_roll_cost_of_a_root_is_its_spread_and_commissions_every_contract_month():
    curves = {r: fa.simulated_curve(r, "2026-09-18") for r in ("ES", "NG")}
    costs = fa.roll_costs({r: c.spec for r, c in curves.items()}, curves)
    for root, c in curves.items():
        spec, got = c.spec, costs[root]
        front = c.long.sort_values(["date", "ttm"]).groupby("date").first()["price"].iloc[-1]
        assert got["per_roll_dollars"] == pytest.approx(spec.spread_ticks * spec.tick_size * spec.multiplier + 2 * spec.commission) and got["rolls_a_year"] == len(spec.months)
        assert got["yearly"] == pytest.approx(got["per_roll_dollars"] / (front * spec.multiplier) * len(spec.months)) and 0 < got["yearly"] < 0.05
    assert costs["NG"]["yearly"] > costs["ES"]["yearly"], "a small contract that rolls every month costs more to hold than a large quarterly one"


def test_a_futures_backtest_pays_for_its_rolls_and_says_so():
    config = load_config()
    with_rolls = fa.run_futures_strategy(config, "carry_ts", ["CL", "NG", "ES"], as_of="2026-09-18")
    without = fa.run_futures_strategy(config, "carry_ts", ["CL", "NG", "ES"], as_of="2026-09-18", roll_cost=False)
    assert [r["root"] for r in with_rolls["roll_costs"]] == ["CL", "ES", "NG"] and without["roll_costs"] == []
    assert with_rolls["metrics"]["cagr"] < without["metrics"]["cagr"] and with_rolls["metrics"]["ann_cost_bps"] > without["metrics"]["ann_cost_bps"]
    assert with_rolls["metrics"]["gross_sharpe"] == pytest.approx(without["metrics"]["gross_sharpe"]), "the gross return is the same book"
    assert "less the cost of each roll" in " ".join(with_rolls["notes"]) and "less the cost of each roll" not in " ".join(without["notes"])


def test_a_short_earns_its_carry_and_a_long_does_not():
    """What a short held through a perpetual future collects (its funding) is credited to the short positions only, on the size held the day before, and is part of the return before costs."""
    days = pd.bdate_range("2024-01-01", periods=40)
    prices = pd.DataFrame({"A": 100.0 * np.cumprod(np.full(40, 1.001)), "B": 50.0 * np.cumprod(np.full(40, 0.999))}, index=days)
    weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=days)
    carry = pd.DataFrame({"A": 0.0004, "B": 0.0002}, index=days)

    def run(short_carry):
        engine = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=False, min_assets=1, cost_model=LinearCostModel(cost_bps=0.0), short_carry=short_carry)
        return engine.run(weights, prices.pct_change(), name="t")

    plain, funded = run(None), run(carry)
    gap = (funded.net_returns - plain.net_returns).dropna()
    assert gap[gap != 0].to_numpy() == pytest.approx(0.5 * 0.0002), "only B is short, so only B's carry is earned, on half the book"
    assert (funded.gross_returns - plain.gross_returns).dropna().abs().sum() == pytest.approx(gap.abs().sum()) and float(funded.costs.sum()) == 0.0
    paying = (run(-carry).net_returns - plain.net_returns).dropna()
    assert paying[paying != 0].to_numpy() == pytest.approx(-0.5 * 0.0002), "negative funding is paid by the short"
    long_only = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=False, min_assets=1, cost_model=LinearCostModel(cost_bps=0.0), short_carry=carry).run(weights.abs(), prices.pct_change(), name="t")
    assert (long_only.net_returns - BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=False, min_assets=1, cost_model=LinearCostModel(cost_bps=0.0)).run(weights.abs(), prices.pct_change(), name="t").net_returns).abs().max() == 0.0


def test_a_crypto_short_is_scored_through_the_perpetual_and_a_long_only_strategy_is_unchanged():
    from src.assets import crypto_analytics as ca
    from src.framework.crypto_bundle import load_crypto_bundle

    config = load_config()
    with_funding = ca.run_crypto_strategy(config, "spot_momentum", ["BTC", "ETH"])
    without = ca.run_crypto_strategy(config, "spot_momentum", ["BTC", "ETH"], short_funding=False)
    assert with_funding["short_funding"] is True and without["short_funding"] is False and "held through the perpetual" in " ".join(with_funding["notes"])
    net_short = min(with_funding["exposure"]["net"]) < -1e-6
    assert (with_funding["equity"] != without["equity"]) == net_short, "funding changes the result exactly when the strategy was ever net short"
    carry = ca.run_crypto_strategy(config, "funding_carry", ["BTC"])
    assert carry["short_funding"] is False, "the carry strategies already trade the index that holds the funding"
    returns = load_crypto_bundle(config).returns
    assert {"BTC_CARRY", "ETH_CARRY"} <= set(returns.columns) and returns["BTC_CARRY"].abs().max() < 0.2


def test_a_trend_strategy_runs_on_other_coins_and_the_carry_strategies_say_they_are_for_bitcoin_and_ether():

    from src.assets import crypto_analytics as ca
    from src.framework.crypto_bundle import load_crypto_bundle

    config = load_config()
    btc = load_crypto_bundle(config).prices["BTC"]
    alt = ca.CryptoSeries("SOL", btc * 0.05 * (1 + 0.1 * np.sin(np.arange(len(btc)) / 20.0)), None, None, "user_file")
    out = ca.run_crypto_strategy(config, "spot_momentum", ["BTC", "SOL"], series={"SOL": alt})
    assert out["assets"] == ["BTC", "SOL"] and out["n_days"] > 100 and "SOL: spot prices only" in " ".join(out["notes"]) and out["actions"]["units"] is True
    assert out["universe"] == ["BTC", "SOL"] and out["metrics"]["gross_exposure_mean"] > 0
    with pytest.raises(ValueError, match="written for BTC and ETH"):
        ca.run_crypto_strategy(config, "funding_carry", ["BTC", "SOL"], series={"SOL": alt})
    with pytest.raises(ValueError, match="covers"):
        ca.run_crypto_strategy(config, "spot_momentum", ["SOL"])
