"""Portfolio construction invariants (spec §59)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.constraints import Constraints, apply_turnover_limit, project_frame
from src.portfolio.covariance import (
    estimate_covariance,
    is_positive_definite,
    nearest_positive_definite,
    portfolio_volatility,
)
from src.portfolio.equal_weight import equal_weight_book, equal_weights
from src.portfolio.inverse_vol import inverse_volatility_weights
from src.portfolio.mean_variance import mean_variance_weights, minimum_variance_weights
from src.portfolio.risk_parity import (
    risk_concentration,
    risk_contribution_frame,
    risk_contributions,
    risk_parity_weights,
)


def test_equal_weights_sum_to_one(tickers):
    weights = equal_weights(tickers)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.nunique() == 1


def test_equal_weight_book_respects_investability(synthetic_market):
    book = equal_weight_book(synthetic_market.investable)
    live = book.loc[book.abs().sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1), 1.0)
    # EEE is not investable for the first 200 rows and must hold nothing.
    assert book["EEE"].iloc[:200].abs().max() == pytest.approx(0.0)


def test_portfolio_return_matches_the_hand_calculation():
    """The specification's example: w = [0.5, 0.5], r = [0.1, -0.04] -> 0.03."""
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    returns = pd.Series([0.10, -0.04], index=["A", "B"])
    assert float(weights @ returns) == pytest.approx(0.03)


def test_inverse_volatility_weights_are_proportional_to_inverse_sigma():
    volatility = pd.Series([0.10, 0.20, 0.40], index=["A", "B", "C"])
    weights = inverse_volatility_weights(volatility)
    assert weights.sum() == pytest.approx(1.0)
    # Halving volatility must double the weight.
    assert weights["A"] == pytest.approx(2.0 * weights["B"])
    assert weights["B"] == pytest.approx(2.0 * weights["C"])


def test_risk_parity_equalises_risk_contributions(synthetic_returns):
    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    weights = risk_parity_weights(covariance)
    frame = risk_contribution_frame(weights, covariance)
    shares = frame["risk_share"].to_numpy()
    assert weights.sum() == pytest.approx(1.0)
    assert shares.max() - shares.min() < 1e-6
    assert shares == pytest.approx(np.full(len(shares), 1.0 / len(shares)), abs=1e-6)
    assert risk_concentration(weights.to_numpy(), covariance.to_numpy()) < 1e-6


def test_risk_contributions_sum_to_portfolio_volatility(synthetic_returns):
    """Euler's theorem: sigma_p is homogeneous of degree one in w."""
    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    weights = risk_parity_weights(covariance)
    contributions = risk_contributions(weights.to_numpy(), covariance.to_numpy())
    assert contributions.sum() == pytest.approx(
        portfolio_volatility(weights.to_numpy(), covariance.to_numpy())
    )


def test_risk_parity_gives_more_weight_to_lower_volatility_assets(synthetic_returns):
    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    weights = risk_parity_weights(covariance)
    volatility = np.sqrt(np.diag(covariance.to_numpy()))
    order_by_vol = np.argsort(volatility)
    assert list(np.argsort(-weights.to_numpy())) == list(order_by_vol)


def test_two_solvers_agree(synthetic_returns):
    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    newton = risk_parity_weights(covariance, method="newton")
    slsqp = risk_parity_weights(covariance, method="slsqp")
    assert (newton - slsqp).abs().max() < 1e-4


def test_covariance_estimators_are_positive_definite(synthetic_returns):
    for method in ("sample", "ewma", "shrinkage", "pca_denoised"):
        covariance = estimate_covariance(synthetic_returns, method, 252)
        assert is_positive_definite(covariance), f"{method} produced a non-PSD matrix"


def test_shrinkage_improves_conditioning(synthetic_returns):
    from src.portfolio.covariance import condition_number

    window = synthetic_returns.tail(120)  # few observations relative to assets
    sample = condition_number(estimate_covariance(window, "sample", 120))
    shrunk = condition_number(estimate_covariance(window, "shrinkage", 120))
    assert shrunk <= sample


def test_psd_repair_preserves_the_diagonal():
    broken = pd.DataFrame(
        [[1.0, 1.3, 0.2], [1.3, 1.0, 0.2], [0.2, 0.2, 1.0]],
        index=list("ABC"), columns=list("ABC"),
    )
    assert not is_positive_definite(broken)
    repaired = nearest_positive_definite(broken)
    assert is_positive_definite(repaired)
    assert np.diag(repaired.to_numpy()) == pytest.approx(np.diag(broken.to_numpy()))


def test_constraints_are_respected_after_projection(tickers):
    constraints = Constraints(min_weight=0.0, max_weight=0.30, group_limits={"grp": 0.8},
                              group_map={t: "grp" for t in tickers}, net_exposure=1.0)
    raw = pd.Series([0.9, 0.05, 0.02, 0.02, 0.01], index=tickers)
    projected = constraints.project(raw)
    assert projected.sum() == pytest.approx(1.0, abs=1e-8)
    assert projected.max() <= 0.30 + 1e-9
    assert projected.min() >= 0.0


def test_vectorised_projection_matches_the_row_version(tickers, rng):
    constraints = Constraints(min_weight=0.0, max_weight=0.30,
                              group_limits={"a": 0.5}, net_exposure=1.0,
                              group_map={t: ("a" if i < 3 else "b") for i, t in enumerate(tickers)})
    frame = pd.DataFrame(rng.random((40, len(tickers))), columns=tickers)
    frame = frame.div(frame.sum(axis=1), axis=0)
    vectorised = project_frame(frame, constraints)
    per_row = pd.DataFrame({i: constraints.project(row) for i, row in frame.iterrows()}).T
    assert np.allclose(vectorised.to_numpy(), per_row.to_numpy(), atol=1e-8)


def test_infeasible_constraints_are_detected(tickers):
    constraints = Constraints(min_weight=0.0, max_weight=0.05, group_limits={}, group_map={},
                              net_exposure=1.0)
    feasible, reason = constraints.is_feasible(tickers)
    assert not feasible
    assert "cannot reach" in reason


def test_turnover_limit_caps_the_trade(tickers):
    previous = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=tickers)
    target = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)
    limited = apply_turnover_limit(target, previous, 0.4)
    assert float((limited - previous).abs().sum()) == pytest.approx(0.4)


def test_minimum_variance_has_lower_variance_than_equal_weight(synthetic_returns):
    covariance = estimate_covariance(synthetic_returns, "sample", len(synthetic_returns))
    constraints = Constraints(min_weight=0.0, max_weight=1.0, group_limits={}, group_map={},
                              net_exposure=1.0)
    minimum = minimum_variance_weights(covariance, constraints)
    equal = np.full(covariance.shape[0], 1.0 / covariance.shape[0])
    assert minimum.expected_volatility <= portfolio_volatility(equal, covariance.to_numpy()) + 1e-9


def test_mean_variance_respects_bounds(synthetic_returns, tickers):
    covariance = estimate_covariance(synthetic_returns, "shrinkage", len(synthetic_returns))
    mu = synthetic_returns.mean() * 252
    constraints = Constraints(min_weight=0.0, max_weight=0.35, group_limits={}, group_map={},
                              net_exposure=1.0)
    result = mean_variance_weights(mu, covariance, 5.0, constraints)
    assert result.success
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.weights.max() <= 0.35 + 1e-6
    assert result.weights.min() >= -1e-9
    assert constraints.violations(result.weights) == {}
