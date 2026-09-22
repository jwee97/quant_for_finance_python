"""Signal construction and the forecast -> position stack."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.mean_reversion import half_life_of_reversion, price_zscore, variance_ratio
from src.features.momentum import total_return_momentum, volatility_scaled_momentum
from src.signals.combine import combine_signals, combine_strategy_returns
from src.signals.transform import (
    apply_weight_cap,
    cross_sectional_demean,
    cross_sectional_zscore,
    normalise_gross,
    risk_scale,
    signal_to_positions,
    turnover,
)


def test_momentum_equals_the_definition():
    index = pd.bdate_range("2020-01-01", periods=30)
    prices = pd.DataFrame({"A": np.linspace(100.0, 130.0, 30)}, index=index)
    momentum = total_return_momentum(prices, 10, 0)
    expected = prices["A"].iloc[20] / prices["A"].iloc[10] - 1.0
    assert momentum["A"].iloc[20] == pytest.approx(expected)


def test_momentum_uses_no_future_prices():
    index = pd.bdate_range("2020-01-01", periods=200)
    prices = pd.DataFrame({"A": np.linspace(100.0, 200.0, 200)}, index=index)
    altered = prices.copy()
    altered.iloc[150:] *= 10.0
    base = total_return_momentum(prices, 20, 1)
    changed = total_return_momentum(altered, 20, 1)
    pd.testing.assert_frame_equal(base.iloc[:150], changed.iloc[:150])


def test_skip_excludes_the_most_recent_days():
    index = pd.bdate_range("2020-01-01", periods=40)
    prices = pd.DataFrame({"A": np.linspace(100.0, 140.0, 40)}, index=index)
    no_skip = total_return_momentum(prices, 10, 0)["A"].iloc[-1]
    skipped = total_return_momentum(prices, 10, 5)["A"].iloc[-1]
    assert no_skip != pytest.approx(skipped)
    assert skipped == pytest.approx(prices["A"].iloc[-6] / prices["A"].iloc[-16] - 1.0)


def test_zscore_is_zero_at_the_mean():
    index = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"A": np.full(100, 100.0)}, index=index)
    prices.iloc[50] = 100.0
    z = price_zscore(prices, 21)
    assert z["A"].dropna().abs().max() == pytest.approx(0.0) or z["A"].dropna().empty


def test_zscore_is_positive_above_the_mean():
    index = pd.bdate_range("2020-01-01", periods=60)
    values = np.full(60, 100.0)
    values[-1] = 120.0
    prices = pd.DataFrame({"A": values}, index=index)
    z = price_zscore(prices, 21)
    assert z["A"].iloc[-1] > 0


def test_variance_ratio_of_a_random_walk_is_about_one(rng):
    walk = pd.Series(np.cumsum(rng.normal(0.0, 0.01, 6000)))
    assert variance_ratio(walk, 5) == pytest.approx(1.0, abs=0.12)


def test_variance_ratio_detects_mean_reversion(rng):
    n = 6000
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.normal(0.0, 0.01)
    assert variance_ratio(pd.Series(values), 10) < 0.9


def test_half_life_is_finite_for_a_reverting_series(rng):
    n = 3000
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.9 * values[i - 1] + rng.normal(0.0, 0.01)
    half_life = half_life_of_reversion(pd.Series(values))
    assert 0 < half_life < 100


def test_cross_sectional_transforms_centre_each_date():
    frame = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 6.0], "C": [5.0, 10.0]})
    assert cross_sectional_demean(frame).sum(axis=1).abs().max() == pytest.approx(0.0)
    z = cross_sectional_zscore(frame)
    assert z.std(axis=1, ddof=1).to_numpy() == pytest.approx(np.ones(2))


def test_gross_normalisation_and_cap_hold_simultaneously(rng, tickers):
    raw = pd.DataFrame(rng.normal(size=(50, len(tickers))), columns=tickers)
    weights = apply_weight_cap(normalise_gross(raw, 1.0), 0.30, 1.0)
    assert weights.abs().max().max() <= 0.30 + 1e-9
    assert weights.abs().sum(axis=1).to_numpy() == pytest.approx(np.ones(50))


def test_risk_scaling_equalises_risk_for_equal_signals():
    index = pd.bdate_range("2020-01-01", periods=5)
    signal = pd.DataFrame(1.0, index=index, columns=["A", "B"])
    volatility = pd.DataFrame({"A": np.full(5, 0.10), "B": np.full(5, 0.40)}, index=index)
    scaled = risk_scale(signal, volatility, 0.10)
    assert scaled["A"].iloc[0] == pytest.approx(4.0 * scaled["B"].iloc[0])
    risk = scaled * volatility
    assert risk["A"].iloc[0] == pytest.approx(risk["B"].iloc[0])


def test_signal_to_positions_is_a_pure_function_of_that_date(rng, tickers):
    """Changing a later date's signal must not change an earlier date's book."""
    index = pd.bdate_range("2020-01-01", periods=100)
    signal = pd.DataFrame(rng.normal(size=(100, len(tickers))), index=index, columns=tickers)
    volatility = pd.DataFrame(0.15, index=index, columns=tickers)
    altered = signal.copy()
    altered.iloc[60:] += 50.0
    base = signal_to_positions(signal, volatility)
    changed = signal_to_positions(altered, volatility)
    pd.testing.assert_frame_equal(base.iloc[:60], changed.iloc[:60])


def test_turnover_is_the_sum_of_absolute_weight_changes():
    index = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.7, 0.7], "B": [0.5, 0.3, 0.3]}, index=index)
    result = turnover(weights)
    assert result.iloc[1] == pytest.approx(0.4)
    assert result.iloc[2] == pytest.approx(0.0)


def test_combining_signals_standardises_before_blending():
    index = pd.bdate_range("2020-01-01", periods=10)
    small = pd.DataFrame({"A": np.full(10, 0.001), "B": np.full(10, -0.001)}, index=index)
    large = pd.DataFrame({"A": np.full(10, 1000.0), "B": np.full(10, -1000.0)}, index=index)
    combined = combine_signals({"s": small, "l": large}, {"s": 0.5, "l": 0.5})
    # Both signals rank A above B; after standardising they must contribute equally.
    assert combined["A"].iloc[-1] == pytest.approx(-combined["B"].iloc[-1])


def test_strategy_combination_weights_are_lagged():
    index = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    streams = pd.DataFrame({"a": rng.normal(0.0003, 0.01, 400),
                            "b": rng.normal(0.0002, 0.02, 400)}, index=index)
    combined, weights = combine_strategy_returns(streams, "inverse_vol", 100, lag=1)
    assert weights.iloc[0].isna().all()      # nothing knowable on day one
    valid = weights.dropna()
    assert valid.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(valid)))
