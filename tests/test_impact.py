"""Execution model: impact costs, capacity, no-trade bands (Generation 3, Priority 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine
from src.backtest.impact import (
    ImpactSettings, banded_execution, banded_gross_returns, capacity_curve, capacity_from_curve, impact_cost_frame,
    impact_net_returns, market_state, scheduling_table)


@pytest.fixture(scope="module")
def book(synthetic_returns, tickers):
    rng = np.random.default_rng(3)
    weights = pd.DataFrame(rng.normal(size=synthetic_returns.shape), index=synthetic_returns.index, columns=tickers)
    weights = weights.div(weights.abs().sum(axis=1), axis=0)
    engine = BacktestEngine(signal_lag=1, rebalance="monthly", weight_drift=True,
                            cost_model=LinearCostModel(5.0, {"AAA": 3.0, "EEE": 9.0}))
    return engine, engine.run(weights, synthetic_returns, "t", apply_vol_target=False)


@pytest.fixture(scope="module")
def state(synthetic_returns):
    close = 100.0 * (1.0 + synthetic_returns).cumprod()
    volume = pd.DataFrame(2e6, index=close.index, columns=close.columns)
    return market_state(close, volume, synthetic_returns)


def test_market_state_uses_only_the_past(synthetic_returns):
    close = 100.0 * (1.0 + synthetic_returns).cumprod()
    volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    sigma, adv = market_state(close, volume, synthetic_returns)
    shocked = synthetic_returns.copy()
    shocked.iloc[500] *= 10.0
    sigma2, _ = market_state(100.0 * (1.0 + shocked).cumprod(), volume, shocked)
    assert sigma.iloc[500].equals(sigma2.iloc[500]) or np.allclose(sigma.iloc[500], sigma2.iloc[500])
    assert not np.allclose(sigma.iloc[501], sigma2.iloc[501])
    assert adv.iloc[:30].isna().all().all()


def test_zero_aum_is_exactly_the_generation1_linear_model(book, state, synthetic_returns):
    """The declared identity: as AUM -> 0 the impact model reduces to the linear one."""
    engine, result = book
    sigma, adv = state
    rates = engine.cost_model.rates(result.trades.columns)
    net, daily, stats = impact_net_returns(result.gross_returns, result.trades, 0.0, sigma, adv, rates)
    assert np.allclose(net.reindex(result.net_returns.index), result.net_returns, atol=1e-15)
    tiny, _, _ = impact_net_returns(result.gross_returns, result.trades, 1e-6, sigma, adv, rates)
    assert np.allclose(tiny.reindex(result.net_returns.index), result.net_returns, atol=1e-9)


def test_impact_cost_formula_on_one_trade(synthetic_returns):
    index = synthetic_returns.index[:3]
    cols = ["A"]
    trades = pd.DataFrame({"A": [0.0, 0.04, 0.0]}, index=index)
    sigma = pd.DataFrame({"A": [0.01] * 3}, index=index)
    adv = pd.DataFrame({"A": [1e8] * 3}, index=index)
    rates = pd.Series({"A": 0.0005})
    cost, part = impact_cost_frame(trades, 1e9, sigma, adv, rates, ImpactSettings(coefficient=2.0))
    participation = 0.04 * 1e9 / 1e8
    expected = 0.04 * (0.0005 + 2.0 * 0.01 * np.sqrt(min(participation, 1.0)))
    assert part["A"].iloc[1] == pytest.approx(participation)
    assert cost["A"].iloc[1] == pytest.approx(expected)
    assert cost["A"].iloc[0] == 0.0


def test_participation_is_capped_in_the_formula_but_reported_uncapped(synthetic_returns):
    index = synthetic_returns.index[:2]
    trades = pd.DataFrame({"A": [0.0, 0.5]}, index=index)
    sigma = pd.DataFrame({"A": [0.02, 0.02]}, index=index)
    adv = pd.DataFrame({"A": [1e7, 1e7]}, index=index)
    cost, part = impact_cost_frame(trades, 1e10, sigma, adv, pd.Series({"A": 0.0}), ImpactSettings(1.0, 1.0))
    assert part["A"].iloc[1] == pytest.approx(0.5 * 1e10 / 1e7)
    assert cost["A"].iloc[1] == pytest.approx(0.5 * 0.02 * 1.0)


def test_cost_is_increasing_in_aum_and_in_the_coefficient(book, state):
    engine, result = book
    sigma, adv = state
    rates = engine.cost_model.rates(result.trades.columns)
    totals = [impact_net_returns(result.gross_returns, result.trades, a, sigma, adv, rates)[1].sum() for a in (1e7, 1e8, 1e9)]
    assert totals[0] < totals[1] < totals[2]
    low = impact_net_returns(result.gross_returns, result.trades, 1e9, sigma, adv, rates, ImpactSettings(0.5))[1].sum()
    high = impact_net_returns(result.gross_returns, result.trades, 1e9, sigma, adv, rates, ImpactSettings(2.0))[1].sum()
    assert low < totals[2] < high


def test_missing_volume_never_makes_a_trade_free(synthetic_returns):
    index = synthetic_returns.index[:2]
    trades = pd.DataFrame({"A": [0.0, 0.1], "B": [0.0, 0.1]}, index=index)
    sigma = pd.DataFrame({"A": [np.nan] * 2, "B": [0.01] * 2}, index=index)
    adv = pd.DataFrame({"A": [np.nan] * 2, "B": [1e8] * 2}, index=index)
    cost, _ = impact_cost_frame(trades, 1e9, sigma, adv, pd.Series({"A": 0.0, "B": 0.0}))
    assert cost["A"].iloc[1] == pytest.approx(cost["B"].iloc[1])
    assert cost["A"].iloc[1] > 0


def test_capacity_interpolation_and_edge_cases():
    curve = pd.Series([1.0, 0.9, 0.6, 0.2], index=[1e6, 1e7, 1e8, 1e9])
    capacity = capacity_from_curve(curve, 1.0)
    assert 1e8 < capacity < 1e9
    assert capacity == pytest.approx(10 ** (8 + (0.6 - 0.5) / (0.6 - 0.2)))
    assert capacity_from_curve(curve, 0.3) == "above grid"
    assert capacity_from_curve(curve, -0.5) == "n/a"
    assert capacity_from_curve(pd.Series([0.1, 0.05], index=[1e6, 1e7]), 1.0) == "below grid"


def test_capacity_curve_is_monotone_non_increasing(book, state):
    engine, result = book
    sigma, adv = state
    rates = engine.cost_model.rates(result.trades.columns)
    curve = capacity_curve(result.gross_returns, result.trades, np.logspace(6, 11, 6), sigma, adv, rates)
    assert (curve.diff().dropna() <= 1e-12).all()


def test_zero_band_reproduces_the_engine_trades(book, synthetic_returns):
    _, result = book
    held, traded = banded_execution(result.weights, result.trades, synthetic_returns, 0.0)
    assert np.allclose(traded.to_numpy(), result.trades.to_numpy(), atol=1e-12)
    assert np.allclose(banded_gross_returns(held, synthetic_returns).dropna(), result.gross_returns, atol=1e-12)


def test_band_reduces_turnover_and_never_trades_inside_the_band(book, synthetic_returns):
    _, result = book
    base = result.trades.abs().sum().sum()
    previous = base
    for band in (0.005, 0.02, 0.08):
        held, traded = banded_execution(result.weights, result.trades, synthetic_returns, band)
        total = traded.abs().sum().sum()
        assert total <= previous + 1e-12
        previous = total
        later = traded.to_numpy()[1:]
        assert (np.abs(later[np.abs(later) > 1e-12]) > band - 1e-12).all()
    large = banded_execution(result.weights, result.trades, synthetic_returns, 10.0)[1]
    first = large.abs().sum(axis=1).ne(0).idxmax()
    assert large.abs().to_numpy().sum() == pytest.approx(large.loc[first].abs().sum())     # nothing but the initial build


def test_scheduling_cost_falls_with_days_and_timing_risk_rises():
    table = scheduling_table(0.05, 1e9, 0.01, 2e8, days=(1, 2, 5, 10), coefficient=1.0, linear_rate=0.0005)
    assert table["cost_bps_of_trade"].is_monotonic_decreasing
    assert table["timing_risk_bps_of_trade"].is_monotonic_increasing
    one = table.loc[1, "cost_bps_of_trade"]
    four = scheduling_table(0.05, 1e9, 0.01, 2e8, days=(4,), linear_rate=0.0)["cost_bps_of_trade"].iloc[0]
    assert 1e4 * 0.01 * np.sqrt(0.05 * 1e9 / 2e8) == pytest.approx(one - 5.0)
    assert four == pytest.approx(1e4 * 0.01 * np.sqrt(0.05 * 1e9 / 2e8 / 4))


def test_zero_band_matches_the_engine_when_the_book_starts_in_cash(synthetic_returns, tickers):
    """Leading all-zero target rows are a held (cash) book that earns 0, not missing data."""
    rng = np.random.default_rng(5)
    weights = pd.DataFrame(rng.normal(size=synthetic_returns.shape), index=synthetic_returns.index, columns=tickers)
    weights = weights.div(weights.abs().sum(axis=1), axis=0)
    weights.iloc[:150] = 0.0
    engine = BacktestEngine(signal_lag=1, rebalance="monthly", weight_drift=True, cost_model=LinearCostModel(5.0))
    result = engine.run(weights, synthetic_returns, "t", apply_vol_target=False)
    held, traded = banded_execution(result.weights, result.trades, synthetic_returns, 0.0)
    gross = banded_gross_returns(held, synthetic_returns).dropna()
    assert len(gross) == len(result.gross_returns)
    assert np.allclose(gross, result.gross_returns, atol=1e-12)
