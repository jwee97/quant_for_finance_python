"""Risk engine: VaR, CVaR and coverage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.risk.contributions import component_var, diversification_ratio
from src.risk.cvar import historical_cvar, parametric_cvar
from src.risk.stress import apply_scenario
from src.risk.var import (
    christoffersen_test,
    historical_var,
    kupiec_test,
    monte_carlo_var,
    parametric_var,
    rolling_var,
    rolling_var_backtest,
)


def test_historical_var_is_the_empirical_quantile():
    returns = pd.Series(np.linspace(-0.10, 0.10, 1001))
    var = historical_var(returns, 0.95)
    assert var == pytest.approx(-float(np.quantile(returns, 0.05)))
    assert var > 0                      # reported as a positive loss


def test_var_increases_with_confidence():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.01, 5000))
    assert historical_var(returns, 0.99) > historical_var(returns, 0.95)


def test_parametric_var_matches_theory_on_normal_data():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0, 0.01, 200000))
    expected = -(returns.mean() + stats.norm.ppf(0.05) * returns.std(ddof=1))
    assert parametric_var(returns, 0.95, 1, "normal") == pytest.approx(expected, rel=1e-6)


def test_cvar_is_always_at_least_var():
    """CVaR averages the tail beyond VaR, so it cannot be smaller."""
    rng = np.random.default_rng(2)
    for scale in (0.005, 0.02):
        returns = pd.Series(rng.standard_t(4, 20000) * scale)
        for alpha in (0.90, 0.95, 0.99):
            assert historical_cvar(returns, alpha) >= historical_var(returns, alpha) - 1e-12


def test_fat_tails_make_the_normal_model_understate_cvar():
    rng = np.random.default_rng(3)
    fat = pd.Series(rng.standard_t(3, 50000) * 0.01)
    assert historical_cvar(fat, 0.99) > parametric_cvar(fat, 0.99, 1, "normal")


def test_monte_carlo_var_preserves_portfolio_correlation():
    """Simulating assets independently would overstate diversification."""
    rng = np.random.default_rng(4)
    common = rng.normal(0.0, 0.01, 4000)
    frame = pd.DataFrame({"A": common + rng.normal(0, 0.002, 4000),
                          "B": common + rng.normal(0, 0.002, 4000)})
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    joint, _ = monte_carlo_var(frame, weights, 0.95, 1, 40000, "normal", seed=7)
    empirical = historical_var(frame @ weights, 0.95)
    assert joint == pytest.approx(empirical, rel=0.15)


def test_kupiec_accepts_a_correct_breach_count():
    _, p_value = kupiec_test(n_breaches=50, n_obs=1000, alpha=0.95)
    assert p_value > 0.10


def test_kupiec_rejects_far_too_many_breaches():
    _, p_value = kupiec_test(n_breaches=150, n_obs=1000, alpha=0.95)
    assert p_value < 0.01


def test_christoffersen_rejects_clustered_breaches():
    """Same number of breaches, but all consecutive."""
    clustered = pd.Series([0] * 950 + [1] * 50)
    _, p_value = christoffersen_test(clustered)
    assert p_value < 0.01


def test_christoffersen_accepts_independent_breaches():
    rng = np.random.default_rng(5)
    independent = pd.Series(rng.binomial(1, 0.05, 4000))
    _, p_value = christoffersen_test(independent)
    assert p_value > 0.05


def test_rolling_var_is_lagged_and_therefore_a_forecast():
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0.0, 0.01, 1200),
                        index=pd.bdate_range("2015-01-01", periods=1200))
    forecast = rolling_var(returns, 0.95, 250, "historical", 100)
    altered = returns.copy()
    altered.iloc[800:] *= 20.0
    changed = rolling_var(altered, 0.95, 250, "historical", 100)
    # A forecast at t may not react to returns at t or later.
    pd.testing.assert_series_equal(forecast.iloc[:801], changed.iloc[:801])


def test_var_backtest_passes_on_data_that_matches_its_own_model():
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0.0, 0.01, 4000),
                        index=pd.bdate_range("2010-01-01", periods=4000))
    result = rolling_var_backtest(returns, 0.95, 500, "historical", 250)
    assert result.breach_rate == pytest.approx(0.05, abs=0.02)
    assert result.kupiec_pvalue > 0.01


def test_component_var_sums_to_portfolio_var(synthetic_returns):
    from src.portfolio.covariance import estimate_covariance

    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    weights = pd.Series(1.0 / covariance.shape[0], index=covariance.columns)
    frame = component_var(weights, covariance, 0.95)
    assert frame["var_share"].sum() == pytest.approx(1.0)


def test_diversification_ratio_is_one_for_a_single_asset(synthetic_returns):
    from src.portfolio.covariance import estimate_covariance

    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    weights = pd.Series(0.0, index=covariance.columns)
    weights.iloc[0] = 1.0
    assert diversification_ratio(weights, covariance) == pytest.approx(1.0)


def test_diversification_ratio_exceeds_one_for_a_spread_book(synthetic_returns):
    from src.portfolio.covariance import estimate_covariance

    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    weights = pd.Series(1.0 / covariance.shape[0], index=covariance.columns)
    assert diversification_ratio(weights, covariance) > 1.0


def test_scenario_pnl_is_weight_times_shock():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    result = apply_scenario(weights, {"equity": -0.20}, {"A": "equity", "B": "rates"})
    assert result["total_pnl"] == pytest.approx(0.6 * -0.20)
    assert result["worst_contributor"] == "A"
