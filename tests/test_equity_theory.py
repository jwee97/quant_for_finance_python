"""Portfolio theory and asset pricing: the frontier, the tangency portfolio and the capital market line, CAPM (standard and zero-beta) and APT, each against a closed form or a world with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from src.equity import theory as th


def market(n=6, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 3))
    cov = F @ F.T / 3 * 0.04 + np.diag(rng.uniform(0.01, 0.03, n))
    mu = rng.uniform(0.03, 0.12, n)
    names = [f"A{i}" for i in range(n)]
    return pd.Series(mu, index=names), pd.DataFrame(cov, index=names, columns=names)


# ------------------------------------------------------------------------------------------------------------------ the frontier
def test_the_frontier_constants_and_the_minimum_variance_portfolio():
    mu, cov = market()
    k = th.frontier_constants(mu, cov)
    one = np.ones(len(mu))
    assert k["A"] == pytest.approx(one @ np.linalg.inv(cov) @ one) and k["D"] == pytest.approx(k["A"] * k["C"] - k["B"] ** 2) and k["D"] > 0
    g = th.global_minimum_variance(mu, cov)
    assert g.weights.sum() == pytest.approx(1.0) and g.volatility ** 2 == pytest.approx(1.0 / k["A"])
    assert g.expected_return == pytest.approx(k["B"] / k["A"])                                   # the minimum-variance portfolio earns B / A
    rng = np.random.default_rng(1)
    for _ in range(300):
        w = rng.normal(size=len(mu))
        w /= w.sum()
        assert w @ cov.to_numpy() @ w >= g.volatility ** 2 - 1e-12


def test_a_frontier_portfolio_has_the_return_asked_for_and_the_least_variance_that_return_allows():
    mu, cov = market(7, 1)
    for target in (0.05, 0.08, 0.14):
        p = th.frontier_portfolio(mu, cov, target)
        assert p.weights.sum() == pytest.approx(1.0) and p.expected_return == pytest.approx(target)
        assert p.volatility ** 2 == pytest.approx(float(th.frontier_variance(mu, cov, target)))
        ref = minimize(lambda w: w @ cov.to_numpy() @ w, np.full(7, 1 / 7), jac=lambda w: 2 * cov.to_numpy() @ w, method="SLSQP", options={"ftol": 1e-15},
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}, {"type": "eq", "fun": lambda w: w @ mu.to_numpy() - target}])
        assert np.abs(ref.x - p.weights.to_numpy()).max() < 1e-5
    a, b = th.frontier_portfolio(mu, cov, 0.06), th.frontier_portfolio(mu, cov, 0.11)
    mix = th.frontier_weights_between(a, b, 0.3)
    r = float(mix @ mu)
    assert float(mix @ cov.to_numpy() @ mix) == pytest.approx(float(th.frontier_variance(mu, cov, r)))      # two funds span the frontier
    with pytest.raises(ValueError, match="no frontier"):
        th.frontier_portfolio(pd.Series(0.05, index=mu.index), cov, 0.05)
    with pytest.raises(ValueError, match="conformable"):
        th.global_minimum_variance(mu.iloc[:3], cov)


# ------------------------------------------------------------------------------------------------------------------ tangency and the capital market line
def test_the_tangency_portfolio_has_the_highest_sharpe_ratio_of_any_portfolio():
    mu, cov = market(8, 2)
    rf = 0.02
    t = th.tangency_portfolio(mu, cov, rf)
    raw = np.linalg.solve(cov.to_numpy(), mu.to_numpy() - rf)
    assert np.abs(t.weights.to_numpy() - raw / raw.sum()).max() < 1e-12 and t.weights.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    for _ in range(2000):
        w = rng.normal(size=8)
        w /= w.sum()
        s = (w @ mu.to_numpy() - rf) / np.sqrt(w @ cov.to_numpy() @ w)
        assert s <= t.sharpe + 1e-9
    ref = minimize(lambda w: -(w @ mu.to_numpy() - rf) / np.sqrt(w @ cov.to_numpy() @ w), np.full(8, 1 / 8), method="SLSQP", constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-15})
    assert -ref.fun == pytest.approx(t.sharpe, rel=1e-7)


def test_a_long_only_tangency_portfolio_never_shorts_and_is_never_better_than_the_free_one():
    mu, cov = market(10, 3)
    free, long_only = th.tangency_portfolio(mu, cov, 0.02), th.tangency_portfolio(mu, cov, 0.02, long_only=True)
    assert (long_only.weights >= -1e-9).all() and long_only.weights.sum() == pytest.approx(1.0) and long_only.sharpe <= free.sharpe + 1e-9
    rng = np.random.default_rng(3)
    for _ in range(2000):
        w = rng.dirichlet(np.ones(10))
        assert (w @ mu.to_numpy() - 0.02) / np.sqrt(w @ cov.to_numpy() @ w) <= long_only.sharpe + 1e-7
    with pytest.raises(ValueError, match="above the risk-free"):
        th.tangency_portfolio(mu, cov, 0.5, long_only=True)


def test_the_capital_market_line_joins_the_risk_free_rate_to_the_tangency_portfolio_and_bounds_everything_else():
    mu, cov = market(6, 4)
    rf = 0.02
    line = th.capital_market_line(mu, cov, rf, volatilities=[0.0, 0.1, 0.2])
    t = line["tangency"]
    assert line["intercept"] == rf and line["slope"] == pytest.approx(t.sharpe) and line["expected_return"][0] == rf
    assert rf + line["slope"] * t.volatility == pytest.approx(t.expected_return)
    rng = np.random.default_rng(4)
    for _ in range(500):
        w = rng.normal(size=6)
        w /= w.sum()
        ret, vol = w @ mu.to_numpy(), np.sqrt(w @ cov.to_numpy() @ w)
        assert ret <= rf + line["slope"] * vol + 1e-9                                                 # every risky portfolio is on or below the line
    for gamma in (2.0, 6.0, 20.0):
        mix = th.cml_allocation(t, rf, gamma)
        utility = lambda y: rf + y * (t.expected_return - rf) - 0.5 * gamma * (y * t.volatility) ** 2
        grid = np.linspace(-2, 12, 14001)
        assert mix["risky_share"] == pytest.approx(grid[np.argmax(utility(grid))], abs=2e-3)
        assert mix["cash"] == pytest.approx(1 - mix["risky_share"]) and mix["weights"].sum() == pytest.approx(mix["risky_share"])
        assert mix["expected_return"] == pytest.approx(rf + line["slope"] * mix["volatility"])         # the mix is on the line
    assert th.cml_allocation(t, rf, 1.0)["risky_share"] > 1.0 > th.cml_allocation(t, rf, 20.0)["risky_share"]      # a less risk-averse investor borrows
    with pytest.raises(ValueError):
        th.cml_allocation(t, rf, 0.0)


# ------------------------------------------------------------------------------------------------------------------ CAPM
def capm_world(seed=0, T=1200, n=40, premium=0.005, zero_beta=0.0, alpha_sd=0.0):
    """Returns r_i = a_i + beta_i m + e with a_i = zero_beta (1 - beta_i) so that E[r_i] = zero_beta + beta_i (E[m] - zero_beta): Black's line (the standard CAPM for zero_beta = 0)."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.3, 1.7, n)
    m = pd.Series(premium + 0.04 * rng.normal(size=T), name="mkt")
    a = zero_beta * (1 - beta) + alpha_sd * rng.normal(size=n)
    r = pd.DataFrame(a + np.outer(m, beta) + 0.03 * rng.normal(size=(T, n)), columns=[f"S{i}" for i in range(n)])
    return r, m, beta, a


def test_the_capm_regression_recovers_alpha_beta_and_r_squared():
    r, m, beta, a = capm_world(0, alpha_sd=0.002)
    out = th.capm_regression(r, m)
    assert np.abs(out["beta"].to_numpy() - beta).max() < 0.1 and np.abs(out["alpha"].to_numpy() - a).max() < 0.004
    ref = np.polyfit(m.to_numpy(), r["S3"].to_numpy(), 1)
    assert out.loc["S3", "beta"] == pytest.approx(ref[0]) and out.loc["S3", "alpha"] == pytest.approx(ref[1])
    pred = np.polyval(ref, m.to_numpy())
    assert out.loc["S3", "r2"] == pytest.approx(1 - ((r["S3"] - pred) ** 2).sum() / ((r["S3"] - r["S3"].mean()) ** 2).sum())
    assert (out["t_beta"] > 10).all() and th.capm_regression(r, m, periods_per_year=12)["alpha"].iloc[0] == pytest.approx(12 * out["alpha"].iloc[0])
    null = th.capm_regression(capm_world(1, alpha_sd=0.0)[0], capm_world(1)[1])
    assert abs(null["t_alpha"]).max() < 4 and (null["t_alpha"].abs() > 1.96).mean() < 0.2               # no alpha planted: few spurious ones
    plain = th.capm_regression(r, m, lags=0)
    assert not np.allclose(plain["t_beta"], out["t_beta"]) and np.allclose(plain["beta"], out["beta"])
    with pytest.raises(ValueError, match="ten"):
        th.capm_regression(r.iloc[:5], m.iloc[:5])
    assert th.capm_expected_returns(pd.Series([0.5, 1.5]), 0.02, 0.05).tolist() == pytest.approx([0.045, 0.095])


def test_the_pricing_line_has_a_zero_intercept_in_a_capm_world_and_a_positive_one_in_a_zero_beta_world():
    r, m, _, _ = capm_world(2, T=2400, n=60)
    capm = th.security_market_line(r, m)
    assert abs(capm.t_intercept) < 3 and capm.slope == pytest.approx(m.mean(), abs=0.003) and capm.r2 > 0.5
    zb = capm_world(3, T=2400, n=60, zero_beta=0.004)
    black = th.security_market_line(zb[0], zb[1])
    assert black.intercept == pytest.approx(0.004, abs=0.0015) and black.t_intercept > 3 and black.slope == pytest.approx(zb[1].mean() - 0.004, abs=0.003)
    forced = th.security_market_line(zb[0], zb[1], allow_zero_beta_rate=False)
    assert forced.intercept == 0.0 and np.isnan(forced.t_intercept) and forced.r2 < black.r2             # forcing the line through the origin fits worse


def test_the_zero_beta_portfolio_is_uncorrelated_with_the_benchmark_and_has_the_least_variance_that_allows():
    mu, cov = market(8, 5)
    bench = np.full(8, 1 / 8)
    z = th.zero_beta_portfolio(cov, bench, mu)
    S = cov.to_numpy()
    assert z.weights.sum() == pytest.approx(1.0) and float(z.weights.to_numpy() @ S @ bench) == pytest.approx(0.0, abs=1e-12)
    ref = minimize(lambda w: w @ S @ w, np.full(8, 1 / 8), jac=lambda w: 2 * S @ w, method="SLSQP", options={"ftol": 1e-16},
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}, {"type": "eq", "fun": lambda w: w @ S @ bench}])
    assert np.abs(ref.x - z.weights.to_numpy()).max() < 1e-5 and z.expected_return == pytest.approx(float(z.weights.to_numpy() @ mu.to_numpy()))


# ------------------------------------------------------------------------------------------------------------------ APT
def apt_world(seed=0, T=1500, n=40, mispriced=0, size=0.002):
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 1.0, (n, 2))
    lam = np.array([0.003, -0.001])
    f = pd.DataFrame(lam + 0.03 * rng.normal(size=(T, 2)), columns=["f1", "f2"])
    alpha = np.zeros(n)
    alpha[:mispriced] = size * rng.choice([-1, 1], mispriced)
    r = pd.DataFrame(alpha + f.to_numpy() @ B.T + 0.03 * rng.normal(size=(T, n)), columns=[f"S{i}" for i in range(n)])
    return r, f, B, lam, alpha


def test_the_apt_two_pass_recovers_the_premia_and_finds_no_mispricing_when_there_is_none():
    r, f, B, lam, _ = apt_world(0)
    out = th.apt_two_pass(r, f)
    assert np.abs(out["lambda"].to_numpy() - lam).max() < 0.002 and out["alpha_chi2"] < 2.5 * out["alpha_df"]
    assert np.abs(out["betas"].to_numpy() - B).max() < 0.15
    bad = th.apt_two_pass(*apt_world(1, mispriced=15, size=0.004)[:2])
    assert bad["alpha_chi2"] > 4 * bad["alpha_df"]                                                    # planted mispricing: the pricing errors are jointly far from zero
    expected = th.apt_expected_returns(out["betas"], out["lambda"], risk_free=0.001)
    assert expected.iloc[0] == pytest.approx(0.001 + out["betas"].iloc[0].to_numpy() @ out["lambda"].to_numpy())


def test_the_arbitrage_portfolio_has_no_net_investment_no_factor_exposure_and_the_alpha_asked_for():
    r, f, B, lam, alpha = apt_world(2, mispriced=12, size=0.003)
    alphas = pd.Series(alpha, index=r.columns)
    cov = pd.DataFrame(np.cov(r.to_numpy(), rowvar=False), index=r.columns, columns=r.columns)
    w = th.apt_arbitrage_portfolio(alphas, B, cov, target_alpha=0.01)
    assert w.sum() == pytest.approx(0.0, abs=1e-10) and np.abs(B.T @ w.to_numpy()).max() < 1e-9 and float(alphas @ w) == pytest.approx(0.01)
    S = cov.to_numpy()
    rng = np.random.default_rng(2)
    C = np.vstack([np.ones(len(w)), B.T, alpha])
    for _ in range(200):                                                                               # any other portfolio with the same exposures has more variance
        d = rng.normal(size=len(w))
        d -= C.T @ np.linalg.lstsq(C.T, d, rcond=None)[0]
        assert (w.to_numpy() + d) @ S @ (w.to_numpy() + d) >= w.to_numpy() @ S @ w.to_numpy() - 1e-12
    scaled = th.apt_arbitrage_portfolio(alphas, B, cov, gross=2.0)
    assert np.abs(scaled).sum() == pytest.approx(2.0)


def test_statistical_factors_span_the_planted_factors():
    r, f, B, _, _ = apt_world(3, T=3000)
    pcs, loadings = th.statistical_factors(r, 2)
    assert list(pcs.columns) == ["pc1", "pc2"] and loadings.shape == (40, 2)
    assert np.abs(loadings.to_numpy().T @ loadings.to_numpy() - np.eye(2)).max() < 1e-10
    X = np.column_stack([np.ones(len(pcs)), pcs.to_numpy()])
    for name in ("f1", "f2"):
        beta, *_ = np.linalg.lstsq(X, f[name].to_numpy(), rcond=None)
        resid = f[name].to_numpy() - X @ beta
        assert 1 - resid.var() / f[name].var() > 0.9                                                  # each planted factor is in the span of the two components
