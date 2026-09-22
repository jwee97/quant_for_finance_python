"""Return arithmetic (spec §59)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.returns import (
    cumulative_returns,
    drawdown,
    forward_returns,
    log_returns,
    max_drawdown,
    simple_returns,
    to_log,
    to_simple,
)


def test_simple_return_100_to_110_is_10_percent():
    """The specification's own example: 100 -> 110 must be exactly 10%."""
    prices = pd.Series([100.0, 110.0], index=pd.bdate_range("2020-01-01", periods=2))
    assert simple_returns(prices).iloc[-1] == pytest.approx(0.10)


def test_log_and_simple_returns_are_consistent():
    prices = pd.Series([100.0, 110.0, 99.0, 150.0], index=pd.bdate_range("2020-01-01", periods=4))
    simple = simple_returns(prices).dropna()
    log = log_returns(prices).dropna()
    assert to_simple(log).to_numpy() == pytest.approx(simple.to_numpy())
    assert to_log(simple).to_numpy() == pytest.approx(log.to_numpy())


def test_log_returns_are_additive_over_time():
    prices = pd.Series([100.0, 110.0, 99.0, 150.0], index=pd.bdate_range("2020-01-01", periods=4))
    total = np.log(prices.iloc[-1] / prices.iloc[0])
    assert log_returns(prices).sum() == pytest.approx(total)


def test_cumulative_returns_compound_correctly():
    returns = pd.Series([0.10, -0.10, 0.05], index=pd.bdate_range("2020-01-01", periods=3))
    curve = cumulative_returns(returns)
    assert curve.iloc[-1] == pytest.approx(1.10 * 0.90 * 1.05)


def test_forward_returns_are_stamped_at_the_decision_date():
    """``forward_returns`` at t must equal the realised return over (t, t+h]."""
    index = pd.bdate_range("2020-01-01", periods=60)
    returns = pd.DataFrame({"A": np.linspace(0.001, 0.002, 60)}, index=index)
    forward = forward_returns(returns, 5)
    expected = float((1.0 + returns["A"].iloc[10:15]).prod() - 1.0)
    assert forward["A"].iloc[9] == pytest.approx(expected)


def test_forward_returns_do_not_leak_into_the_past():
    """Changing only the future must leave earlier forward returns untouched."""
    index = pd.bdate_range("2020-01-01", periods=100)
    returns = pd.DataFrame({"A": np.full(100, 0.001)}, index=index)
    altered = returns.copy()
    altered.iloc[60:] = 0.5
    base = forward_returns(returns, 5)
    changed = forward_returns(altered, 5)
    # Row 54 is the last whose (t, t+5] window ends at or before row 59.
    pd.testing.assert_frame_equal(base.iloc[:55], changed.iloc[:55])


def test_forward_returns_tail_is_missing_by_construction():
    index = pd.bdate_range("2020-01-01", periods=50)
    returns = pd.DataFrame({"A": np.full(50, 0.001)}, index=index)
    forward = forward_returns(returns, 10)
    assert forward["A"].iloc[-10:].isna().all()


def test_drawdown_is_zero_for_a_monotone_curve():
    curve = pd.Series(np.linspace(1.0, 2.0, 100), index=pd.bdate_range("2020-01-01", periods=100))
    assert drawdown(curve).max() == pytest.approx(0.0)
    assert max_drawdown(curve) == pytest.approx(0.0)


def test_known_drawdown_depth():
    curve = pd.Series([1.0, 2.0, 1.0, 1.5], index=pd.bdate_range("2020-01-01", periods=4))
    assert max_drawdown(curve) == pytest.approx(-0.5)
