"""Closed-form mean-variance portfolios: the algebra of the frontier (the hyperbola, two-fund separation, the capital market line) and agreement with the numerical optimiser when its
bounds are loosened to allow shorting, which is the claim that makes the closed forms a check on it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import closed_form as cf
from src.portfolio.constraints import Constraints
from src.portfolio.mean_variance import maximum_sharpe_weights, mean_variance_weights, minimum_variance_weights


def _market(n=5, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n * 0.04 + np.eye(n) * 0.01
    mu = pd.Series(rng.uniform(0.02, 0.12, n))
    names = [f"A{i}" for i in range(n)]
    return mu.set_axis(names), pd.DataFrame(cov, index=names, columns=names)


LOOSE = Constraints(min_weight=-20.0, max_weight=20.0, gross_leverage=None, net_exposure=1.0, long_only=False)


def test_the_global_minimum_variance_portfolio_is_the_nose_of_the_hyperbola_and_matches_the_optimiser():
    mu, cov = _market()
    g = cf.global_minimum_variance(cov, mu)
    k = cf.frontier_constants(mu, cov)
    assert g.weights.sum() == pytest.approx(1.0) and g.volatility ** 2 == pytest.approx(1.0 / k["A"]) and g.expected_return == pytest.approx(k["B"] / k["A"])
    numeric = minimum_variance_weights(cov, LOOSE)
    assert np.allclose(g.weights.to_numpy(), numeric.weights.to_numpy(), atol=1e-5)
    assert k["D"] > 0 and k["A"] > 0
    assert cf.frontier_variance(mu, cov, k["B"] / k["A"]) == pytest.approx(1.0 / k["A"])


def test_frontier_portfolios_lie_on_the_hyperbola_have_the_target_return_and_mix_two_funds():
    mu, cov = _market(seed=1)
    p1, p2 = cf.target_return_portfolio(mu, cov, 0.07), cf.target_return_portfolio(mu, cov, 0.11)
    for p, target in ((p1, 0.07), (p2, 0.11)):
        assert p.expected_return == pytest.approx(target) and p.weights.sum() == pytest.approx(1.0)
        assert p.volatility ** 2 == pytest.approx(float(cf.frontier_variance(mu, cov, target)))
    mid = cf.target_return_portfolio(mu, cov, 0.09)
    assert np.allclose(mid.weights.to_numpy(), 0.5 * p1.weights.to_numpy() + 0.5 * p2.weights.to_numpy()), "two-fund separation: a mix of two frontier portfolios is on the frontier"
    # no other fully invested portfolio of the same return has less variance
    rng = np.random.default_rng(3)
    S = cov.to_numpy()
    for _ in range(200):
        w = mid.weights.to_numpy() + rng.standard_normal(5) * 0.05
        w = w - (w.sum() - 1.0) / 5
        w = w - (w @ mu.to_numpy() - 0.09) * np.linalg.solve(S, mu.to_numpy()) / (mu.to_numpy() @ np.linalg.solve(S, mu.to_numpy()))
        w = w / w.sum() if abs(w.sum()) > 1e-9 else w
        if abs(w.sum() - 1.0) < 1e-9 and abs(w @ mu.to_numpy() - 0.09) < 1e-9:
            assert w @ S @ w >= mid.volatility ** 2 - 1e-12


def test_the_frontier_table_the_target_portfolio_and_the_optimiser_agree():
    mu, cov = _market(seed=2)
    table = cf.efficient_frontier(mu, cov, n_points=9)
    assert table["volatility"].is_monotonic_increasing and table["return"].iloc[0] == pytest.approx(cf.frontier_constants(mu, cov)["B"] / cf.frontier_constants(mu, cov)["A"])
    assert np.allclose(table["volatility"] ** 2, cf.frontier_variance(mu, cov, table["return"].to_numpy()))
    lower = cf.efficient_frontier(mu, cov, n_points=5, span=(0.0, 0.05))
    assert lower["volatility"].min() >= table["volatility"].min() - 1e-12, "the lower branch is above the minimum-variance level"
    numeric = mean_variance_weights(mu, cov, risk_aversion=4.0, constraints=LOOSE)
    exact = cf.utility_portfolio(mu, cov, 4.0)
    assert np.allclose(exact.weights.to_numpy(), numeric.weights.to_numpy(), atol=1e-4) and exact.weights.sum() == pytest.approx(1.0)
    on_frontier = cf.target_return_portfolio(mu, cov, exact.expected_return)
    assert np.allclose(on_frontier.weights.to_numpy(), exact.weights.to_numpy()), "an investor with this risk aversion holds a frontier portfolio"


def test_the_tangency_portfolio_has_the_highest_sharpe_ratio_and_the_capital_market_line_has_its_slope():
    mu, cov = _market(seed=4)
    rf = 0.01
    t = cf.tangency_portfolio(mu, cov, rf)
    assert t.weights.sum() == pytest.approx(1.0)
    best = t.sharpe(rf)
    line = cf.capital_market_line(mu, cov, rf)
    assert line["slope"] == pytest.approx(best) and line["tangency"].weights.round(12).equals(t.weights.round(12))
    for m in np.linspace(0.02, 0.12, 11):
        assert cf.target_return_portfolio(mu, cov, float(m)).sharpe(rf) <= best + 1e-12, "no frontier portfolio does better than the tangency one"
    numeric = maximum_sharpe_weights(mu, cov, rf, LOOSE)
    assert np.allclose(t.weights.to_numpy(), numeric.weights.to_numpy(), atol=1e-3)
    held = cf.utility_portfolio(mu, cov, 3.0, risk_free=rf)
    assert held.weights.to_numpy() == pytest.approx(held.weights.sum() * t.weights.to_numpy()), "with a risk-free asset every investor holds the same risky mix, only the amount differs"
    assert held.weights.sum() == pytest.approx((t.expected_return - rf) / (3.0 * t.volatility ** 2))


def test_the_weights_are_linear_in_the_expected_returns_which_is_why_noise_in_them_matters_so_much():
    mu, cov = _market(seed=5)
    p0 = cf.utility_portfolio(mu, cov, 5.0, risk_free=0.0).weights
    bump = pd.Series(0.0, index=mu.index)
    bump.iloc[0] = 0.01
    p1 = cf.utility_portfolio(mu + bump, cov, 5.0, risk_free=0.0).weights
    p2 = cf.utility_portfolio(mu + 2 * bump, cov, 5.0, risk_free=0.0).weights
    assert np.allclose((p2 - p1).to_numpy(), (p1 - p0).to_numpy()), "twice the change in a forecast, twice the change in the holding"
    assert float((p1 - p0).abs().sum()) > 0.05, "one point on one forecast moves the holdings by several per cent in total"


def test_bad_inputs_are_refused_with_a_reason():
    mu, cov = _market(seed=6)
    singular = cov.copy()
    singular.iloc[:, 1] = singular.iloc[:, 0]
    singular.iloc[1, :] = singular.iloc[0, :]
    with pytest.raises(ValueError, match="singular"):
        cf.global_minimum_variance(singular)
    with pytest.raises(ValueError, match="symmetric"):
        cf.global_minimum_variance(pd.DataFrame(np.triu(cov.to_numpy()), index=cov.index, columns=cov.columns))
    with pytest.raises(ValueError, match="finite"):
        cf.target_return_portfolio(mu.where(mu.index != "A0"), cov, 0.05)
    with pytest.raises(ValueError, match="risk aversion"):
        cf.utility_portfolio(mu, cov, 0.0)
    flat = pd.Series(0.05, index=mu.index)
    with pytest.raises(ValueError, match="proportional"):
        cf.target_return_portfolio(flat, cov, 0.05)
    g = cf.global_minimum_variance(cov, mu)
    assert g.condition > 1 and np.isfinite(g.sharpe(0.0))
    with pytest.raises(ValueError, match="infinity"):
        cf.tangency_portfolio(mu, cov, cf.frontier_constants(mu, cov)["B"] / cf.frontier_constants(mu, cov)["A"])
