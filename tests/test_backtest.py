"""Backtest engine: timing, costs and metrics (spec §59)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import LinearCostModel, breakeven_cost, cost_sensitivity
from src.backtest.engine import BacktestEngine, buy_and_hold
from src.backtest.execution import apply_execution_lag, build_held_weights, drift_weights
from src.backtest.metrics import (
    cagr,
    deflated_sharpe_ratio,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
)


def test_portfolio_return_is_previous_weights_times_this_return():
    """R_t = w_{t-1}' r_t, exactly."""
    index = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.DataFrame({"A": np.full(10, 0.01), "B": np.full(10, -0.004)}, index=index)
    weights = pd.DataFrame({"A": np.full(10, 0.5), "B": np.full(10, 0.5)}, index=index)
    engine = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=False,
                            cost_model=LinearCostModel(0.0))
    result = engine.run(weights, returns, "t", apply_vol_target=False)
    expected = 0.5 * 0.01 + 0.5 * -0.004
    assert result.gross_returns.iloc[-1] == pytest.approx(expected)


def test_execution_lag_shifts_weights_forward():
    index = pd.bdate_range("2020-01-01", periods=5)
    weights = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    lagged = apply_execution_lag(weights, 1)
    assert np.isnan(lagged["A"].iloc[0])
    assert lagged["A"].iloc[1] == 1.0
    assert lagged["A"].iloc[-1] == 4.0


def test_a_larger_lag_can_only_delay_the_book():
    index = pd.bdate_range("2020-01-01", periods=20)
    weights = pd.DataFrame({"A": np.arange(20, dtype=float)}, index=index)
    one = apply_execution_lag(weights, 1).dropna()
    two = apply_execution_lag(weights, 2).dropna()
    assert two["A"].iloc[0] == one["A"].iloc[0]
    assert len(two) == len(one) - 1


def test_a_constant_target_trades_only_once():
    """A book whose target never changes pays to be built and nothing after.

    The initial build lands on the first rebalance date after the execution
    lag, not on row 0, so the assertion is about the total and about what
    happens *after* the first trade.
    """
    index = pd.bdate_range("2020-01-01", periods=250)
    returns = pd.DataFrame({"A": np.full(250, 0.0004), "B": np.full(250, 0.0002)}, index=index)
    weights = pd.DataFrame({"A": np.full(250, 0.5), "B": np.full(250, 0.5)}, index=index)
    engine = BacktestEngine(signal_lag=1, rebalance="monthly", weight_drift=False,
                            cost_model=LinearCostModel(25.0))
    result = engine.run(weights, returns, "t", apply_vol_target=False)
    traded = result.turnover
    first = int((traded.abs() > 1e-12).argmax())
    assert traded.sum() == pytest.approx(1.0)              # the initial build only
    assert traded.iloc[first + 1:].sum() == pytest.approx(0.0, abs=1e-12)


def test_costs_reduce_returns_by_exactly_rate_times_turnover():
    index = pd.bdate_range("2020-01-01", periods=5)
    trades = pd.DataFrame({"A": [1.0, -0.5, 0.0, 0.25, 0.0]}, index=index)
    gross = pd.Series(np.full(5, 0.001), index=index)
    model = LinearCostModel(10.0)
    net, costs = model.apply(gross, trades)
    assert costs.iloc[0] == pytest.approx(1.0 * 10.0 / 1e4)
    assert costs.iloc[1] == pytest.approx(0.5 * 10.0 / 1e4)
    assert (gross - net).to_numpy() == pytest.approx(costs.to_numpy())


def test_per_asset_costs_are_applied_per_asset():
    index = pd.bdate_range("2020-01-01", periods=2)
    trades = pd.DataFrame({"CHEAP": [1.0, 0.0], "DEAR": [1.0, 0.0]}, index=index)
    model = LinearCostModel(10.0, {"CHEAP": 2.0, "DEAR": 50.0})
    costs = model.trade_costs(trades)
    assert costs.iloc[0] == pytest.approx((2.0 + 50.0) / 1e4)


def test_higher_costs_never_increase_net_performance():
    index = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(3)
    gross = pd.Series(rng.normal(0.0005, 0.01, 500), index=index)
    trades = pd.DataFrame({"A": rng.normal(0.0, 0.1, 500)}, index=index)
    table = cost_sensitivity(gross, trades, (0.0, 5.0, 10.0, 25.0, 50.0))
    assert table["net_ann_return"].is_monotonic_decreasing


def test_breakeven_cost_zeroes_the_net_return():
    index = pd.bdate_range("2020-01-01", periods=300)
    gross = pd.Series(np.full(300, 0.0004), index=index)
    trades = pd.DataFrame({"A": np.full(300, 0.2)}, index=index)
    breakeven = breakeven_cost(gross, trades)
    net, _ = LinearCostModel(breakeven).apply(gross, trades)
    assert net.mean() == pytest.approx(0.0, abs=1e-12)


def test_weight_drift_moves_weights_with_returns():
    index = pd.bdate_range("2020-01-01", periods=5)
    targets = pd.DataFrame({"A": [0.5, np.nan, np.nan, np.nan, np.nan],
                            "B": [0.5, np.nan, np.nan, np.nan, np.nan]}, index=index)
    returns = pd.DataFrame({"A": np.full(5, 0.10), "B": np.zeros(5)}, index=index)
    held = drift_weights(targets, returns)
    assert held["A"].iloc[1] > 0.5   # the winner grows
    assert held["B"].iloc[1] < 0.5
    assert held.iloc[1].sum() == pytest.approx(1.0)


def test_buy_and_hold_has_zero_ongoing_turnover(synthetic_returns):
    result = buy_and_hold(synthetic_returns, "AAA")
    assert result.turnover.iloc[1:].sum() == pytest.approx(0.0)
    assert result.net_returns.equals(result.gross_returns)


def test_cagr_uses_calendar_time_not_period_count():
    """CAGR is a *calendar* annual rate.

    252 business days span 351 calendar days, not 365, so compounding 0.1% a
    day over them annualises to more than (1.001^252 - 1). Asserting the
    period-count version would bake a subtly wrong convention into the suite.
    """
    index = pd.bdate_range("2020-01-01", periods=252)
    returns = pd.Series(np.full(252, 0.001), index=index)
    years = (index[-1] - index[0]).days / 365.25
    assert cagr(returns) == pytest.approx((1.001 ** 252) ** (1.0 / years) - 1.0, rel=1e-6)


def test_cagr_over_a_full_calendar_year_matches_total_growth():
    index = pd.bdate_range("2020-01-01", "2020-12-31")
    returns = pd.Series(np.full(len(index), 0.001), index=index)
    total = (1.001 ** len(index)) - 1.0
    assert cagr(returns) == pytest.approx(total, rel=0.06)


def test_metrics_on_a_constant_return_series():
    index = pd.bdate_range("2020-01-01", periods=252)
    returns = pd.Series(np.full(252, 0.001), index=index)
    assert np.isnan(sharpe_ratio(returns)) or sharpe_ratio(returns) > 100
    assert np.isnan(sortino_ratio(returns))   # no downside observations at all
    summary = performance_summary(returns)
    assert np.isnan(summary["skew"])          # undefined, not a warning
    assert np.isnan(summary["excess_kurtosis"])


def test_summary_reports_gross_and_net(synthetic_returns, synthetic_market):
    weights = pd.DataFrame(1.0 / 5, index=synthetic_returns.index, columns=synthetic_returns.columns)
    engine = BacktestEngine(signal_lag=1, rebalance="monthly", cost_model=LinearCostModel(20.0))
    result = engine.run(weights, synthetic_returns, "t", synthetic_market.investable,
                        apply_vol_target=False)
    summary = result.summary()
    assert summary["gross_sharpe"] >= summary["sharpe"]
    assert summary["ann_cost_drag"] >= 0.0


def test_deflated_sharpe_falls_as_trials_rise():
    single = deflated_sharpe_ratio(1.0, 1, 2520)
    many = deflated_sharpe_ratio(1.0, 500, 2520)
    assert single > many
    assert 0.0 <= many <= 1.0
