"""Pairs trading and PCA statistical arbitrage (Extensions E and F)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.pairs import (
    backtest_pair,
    build_spread,
    estimate_hedge_ratio,
    pairs_positions,
    screen_pairs,
    analyse_pair,
)
from src.signals.pca_strategy import (
    factor_exposure_check,
    pca_stat_arb_signal,
    residual_zscore,
    rolling_residuals,
)


@pytest.fixture(scope="module")
def cointegrated_pair():
    """Two prices sharing a stochastic trend plus a stationary spread."""
    rng = np.random.default_rng(42)
    n = 1500
    index = pd.bdate_range("2015-01-01", periods=n)
    common = np.cumsum(rng.normal(0.0003, 0.01, n))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.95 * spread[i - 1] + rng.normal(0.0, 0.01)   # stationary
    left = pd.Series(100.0 * np.exp(common + spread), index=index)
    right = pd.Series(100.0 * np.exp(common), index=index)
    return left, right


@pytest.fixture(scope="module")
def independent_pair():
    """Two independent random walks: correlated in levels by luck, never cointegrated."""
    rng = np.random.default_rng(7)
    n = 1500
    index = pd.bdate_range("2015-01-01", periods=n)
    left = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=index)
    right = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=index)
    return left, right


def test_hedge_ratio_recovers_a_known_relationship():
    index = pd.bdate_range("2020-01-01", periods=500)
    right = pd.Series(np.exp(np.linspace(4.0, 5.0, 500)), index=index)
    left = right ** 0.8 * 3.0
    beta, _ = estimate_hedge_ratio(left, right)
    assert beta == pytest.approx(0.8, abs=0.01)


def test_spread_of_a_perfect_relationship_is_constant():
    index = pd.bdate_range("2020-01-01", periods=300)
    right = pd.Series(np.exp(np.linspace(4.0, 5.0, 300)), index=index)
    left = right ** 0.8
    beta, alpha = estimate_hedge_ratio(left, right)
    spread = build_spread(left, right, beta, alpha)
    assert spread.std() == pytest.approx(0.0, abs=1e-8)


def test_cointegration_is_detected_when_it_exists(cointegrated_pair):
    left, right = cointegrated_pair
    result = analyse_pair(left, right, "L", "R")
    assert result.is_cointegrated
    assert result.adf_pvalue < 0.05
    assert 1.0 < result.half_life < 126.0
    assert result.tradable


def test_independent_random_walks_are_not_cointegrated(independent_pair):
    left, right = independent_pair
    result = analyse_pair(left, right, "L", "R")
    assert not result.is_cointegrated
    assert not result.tradable


def test_correlation_alone_does_not_imply_cointegration(independent_pair):
    """The distinction the screen exists to enforce."""
    left, right = independent_pair
    result = analyse_pair(left, right, "L", "R")
    level_correlation = float(np.corrcoef(np.log(left), np.log(right))[0, 1])
    assert abs(level_correlation) > 0.2      # levels can look related
    assert not result.is_cointegrated        # the spread still wanders


def test_pairs_positions_are_bounded_and_causal(cointegrated_pair):
    left, right = cointegrated_pair
    result = analyse_pair(left, right, "L", "R")
    signals = pairs_positions(left, right, result.hedge_ratio, result.intercept)
    assert set(signals["position"].unique()) <= {-1.0, 0.0, 1.0}

    altered_left = left.copy()
    altered_left.iloc[1000:] *= 5.0
    changed = pairs_positions(altered_left, right, result.hedge_ratio, result.intercept)
    pd.testing.assert_series_equal(signals["position"].iloc[:1000],
                                   changed["position"].iloc[:1000])


def test_entry_and_exit_thresholds_are_respected(cointegrated_pair):
    left, right = cointegrated_pair
    result = analyse_pair(left, right, "L", "R")
    signals = pairs_positions(left, right, result.hedge_ratio, result.intercept,
                              entry=2.0, exit_threshold=0.5).dropna()
    opened = signals[(signals["position"] != 0) & (signals["position"].shift() == 0)]
    assert (opened["zscore"].abs() > 2.0 - 1e-9).all()


def test_pair_backtest_charges_costs(cointegrated_pair):
    left, right = cointegrated_pair
    result = analyse_pair(left, right, "L", "R")
    free = backtest_pair(left, right, result, cost_bps=0.0)
    expensive = backtest_pair(left, right, result, cost_bps=50.0)
    assert expensive["sharpe"] <= free["sharpe"]


def test_screen_runs_on_training_data_only(synthetic_prices):
    cutoff = synthetic_prices.index[600]
    frame = screen_pairs(synthetic_prices, [("AAA", "BBB")], sample_end=cutoff)
    assert len(frame) == 1
    assert frame["n_obs"].iloc[0] <= 601


def test_pca_residuals_remove_the_common_factor(synthetic_returns):
    """Residuals must be far less correlated with each other than raw returns."""
    residuals = rolling_residuals(synthetic_returns, n_components=2, window=252, step=21).dropna(how="all")
    assert len(residuals.dropna(how="any")) > 100
    raw = synthetic_returns.loc[residuals.dropna(how="any").index].corr()
    residual_corr = residuals.dropna(how="any").corr()
    n = raw.shape[0]
    off = ~np.eye(n, dtype=bool)
    assert abs(residual_corr.to_numpy()[off]).mean() < abs(raw.to_numpy()[off]).mean()


def test_pca_residuals_are_causal(synthetic_returns):
    altered = synthetic_returns.copy()
    altered.iloc[900:] *= 20.0
    base = rolling_residuals(synthetic_returns, 2, 252, 21)
    changed = rolling_residuals(altered, 2, 252, 21)
    # Residuals up to the last refit before row 900 must be unaffected.
    pd.testing.assert_frame_equal(base.iloc[:800], changed.iloc[:800])


def test_stat_arb_signal_has_the_reversion_sign(synthetic_returns):
    residuals = rolling_residuals(synthetic_returns, 2, 252, 21)
    z = residual_zscore(residuals, 21)
    signal = pca_stat_arb_signal(synthetic_returns, 2, 252, 21, 21)
    aligned = pd.concat([z.stack().rename("z"), signal.stack().rename("s")], axis=1).dropna()
    assert len(aligned) > 100
    assert float(np.corrcoef(aligned["z"], aligned["s"])[0, 1]) == pytest.approx(-1.0, abs=1e-6)


def test_factor_exposure_check_detects_a_pure_factor_bet(synthetic_returns):
    from src.features.pca import pca_decomposition

    result = pca_decomposition(synthetic_returns.dropna(how="any"), use_correlation=True)
    pc1 = result.eigenvectors["PC1"]
    exposures = factor_exposure_check(pc1, synthetic_returns, 3)
    assert abs(exposures["PC1"]) > 0.9       # a PC1 book is all PC1
    assert abs(exposures["PC2"]) < 0.1
