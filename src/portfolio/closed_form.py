r"""Mean-variance portfolios in closed form (Markowitz 1952; Merton 1972).

With no constraint except that the weights sum to one, the problem "least variance for a given expected return" is a quadratic with linear constraints and has an exact solution. Four numbers
describe the whole frontier::

    A = 1' S^-1 1       B = 1' S^-1 mu       C = mu' S^-1 mu       D = A C - B^2        (S the covariance matrix, mu the expected returns)

The frontier of portfolios is the hyperbola ``var(m) = (A m^2 - 2 B m + C) / D`` in the return ``m``, with weights ``w(m) = S^-1 (lam 1 + gam mu)``, ``lam = (C - B m) / D``, ``gam = (A m - B) / D``:
every frontier portfolio is a mix of two others (two-fund separation) and the global minimum-variance portfolio ``S^-1 1 / A`` sits at the nose of the hyperbola, with return ``B / A`` and
variance ``1 / A``. With a risk-free asset the frontier becomes a straight line, the capital market line, through the *tangency* portfolio ``S^-1 (mu - rf 1) / (1' S^-1 (mu - rf 1))``
(the highest Sharpe ratio there is), and an investor with risk aversion ``g`` holds ``S^-1 (mu - rf 1) / g`` of risky assets in total.

These are exact and instant, and shorting is allowed: they are what the numerical optimisers in ``portfolio.mean_variance`` must reproduce when their bounds are loosened, which is how
they are checked; they also show at a glance how the weights respond to the inputs (the weights are linear in ``mu``, which is why they swing so much when ``mu`` is noisy).
Long-only and capped problems have no closed form: use ``portfolio.mean_variance``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Portfolio:
    """Weights with their expected return and volatility (and the condition number of the covariance matrix that produced them: above about 1e6 the weights mean little)."""

    weights: pd.Series
    expected_return: float
    volatility: float
    condition: float

    def sharpe(self, risk_free: float = 0.0) -> float:
        return float((self.expected_return - risk_free) / self.volatility) if self.volatility > 0 else float("nan")


def _inputs(mu, cov):
    cov = pd.DataFrame(cov)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("the covariance matrix must be square")
    assets = list(cov.columns)
    S = cov.to_numpy(dtype=float)
    if not np.allclose(S, S.T, atol=1e-10 * max(1.0, np.abs(S).max())):
        raise ValueError("the covariance matrix must be symmetric")
    m = None
    if mu is not None:
        m = (mu.reindex(assets) if isinstance(mu, pd.Series) else pd.Series(np.asarray(mu, dtype=float), index=assets)).to_numpy(dtype=float)
        if not np.all(np.isfinite(m)):
            raise ValueError("the expected returns must be finite and given for every asset")
    return assets, S, m


def _solve(S: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(S, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the covariance matrix is singular: it has no inverse (drop an asset that is a combination of the others)") from exc


def _portfolio(assets, S, mu, w) -> Portfolio:
    vol = float(np.sqrt(max(w @ S @ w, 0.0)))
    return Portfolio(pd.Series(w, index=assets), float(w @ mu) if mu is not None else float("nan"), vol, float(np.linalg.cond(S)))


def frontier_constants(mu, cov) -> dict:
    """``A, B, C, D`` of the frontier (see the module docstring)."""
    assets, S, m = _inputs(mu, cov)
    one = np.ones(len(assets))
    si1, sim = _solve(S, one), _solve(S, m)
    a, b, c = float(one @ si1), float(one @ sim), float(m @ sim)
    return {"A": a, "B": b, "C": c, "D": a * c - b * b}


def global_minimum_variance(cov, mu=None) -> Portfolio:
    """``S^-1 1 / A``: the least-risk fully invested portfolio, which needs no expected returns."""
    assets, S, m = _inputs(mu, cov)
    w = _solve(S, np.ones(len(assets)))
    return _portfolio(assets, S, m, w / w.sum())


def target_return_portfolio(mu, cov, target: float) -> Portfolio:
    """The fully invested portfolio of least variance with expected return ``target``."""
    assets, S, m = _inputs(mu, cov)
    k = frontier_constants(mu, cov)
    if k["D"] <= 1e-14 * max(abs(k["A"] * k["C"]), 1.0):
        raise ValueError("the expected returns are (nearly) proportional to each other: there is no frontier, only a point")
    lam, gam = (k["C"] - k["B"] * target) / k["D"], (k["A"] * target - k["B"]) / k["D"]
    w = _solve(S, lam * np.ones(len(assets)) + gam * m)
    return _portfolio(assets, S, m, w)


def tangency_portfolio(mu, cov, risk_free: float = 0.0) -> Portfolio:
    """The fully invested portfolio of the highest Sharpe ratio, ``S^-1 (mu - rf 1)`` scaled to sum to one (the weights are of the risky assets; it exists when the excess return of the
    global minimum-variance portfolio is positive, i.e. ``B / A > rf``, else the highest Sharpe ratio is that of its mirror image, a short position)."""
    assets, S, m = _inputs(mu, cov)
    raw = _solve(S, m - risk_free)
    total = raw.sum()
    if abs(total) < 1e-12:
        raise ValueError("the risk-free rate equals the return of the minimum-variance portfolio: the tangency portfolio is at infinity")
    return _portfolio(assets, S, m, raw / total)


def utility_portfolio(mu, cov, risk_aversion: float, risk_free: float | None = None) -> Portfolio:
    """The portfolio that maximises ``w'mu - (g / 2) w'Sw``. With a risk-free asset (``risk_free`` given) the weights of the risky assets are free (what is left of the wealth is lent or
    borrowed at ``risk_free``), ``w = S^-1 (mu - rf 1) / g``; otherwise they sum to one, ``w = S^-1 (mu - eta 1) / g`` with ``eta = (B - g) / A``."""
    if risk_aversion <= 0:
        raise ValueError("the risk aversion must be positive")
    assets, S, m = _inputs(mu, cov)
    if risk_free is not None:
        w = _solve(S, m - risk_free) / risk_aversion
    else:
        k = frontier_constants(mu, cov)
        w = _solve(S, m - (k["B"] - risk_aversion) / k["A"]) / risk_aversion
    return _portfolio(assets, S, m, w)


def efficient_frontier(mu, cov, n_points: int = 25, span: tuple[float, float] | None = None) -> pd.DataFrame:
    """Points on the frontier from the minimum-variance return up to ``span[1]`` (default: the largest expected return) and down to ``span[0]`` (the inefficient lower branch, default: the
    minimum-variance return): the target return, the volatility from the hyperbola and the weights. The volatility is computed from the weights, so a mismatch with the hyperbola
    would show an error in either."""
    assets, S, m = _inputs(mu, cov)
    k = frontier_constants(mu, cov)
    lo_default = k["B"] / k["A"]
    lo, hi = span if span is not None else (lo_default, float(m.max()))
    rows = []
    for target in np.linspace(lo, hi, n_points):
        p = target_return_portfolio(mu, cov, float(target))
        rows.append({"return": float(target), "volatility": p.volatility, **{a: float(w) for a, w in p.weights.items()}})
    return pd.DataFrame(rows)


def frontier_variance(mu, cov, target) -> np.ndarray:
    """The hyperbola ``(A m^2 - 2 B m + C) / D`` for the variance of the frontier portfolio of return ``target`` (array allowed)."""
    k = frontier_constants(mu, cov)
    t = np.asarray(target, dtype=float)
    return (k["A"] * t ** 2 - 2.0 * k["B"] * t + k["C"]) / k["D"]


def capital_market_line(mu, cov, risk_free: float) -> dict:
    """The line through the risk-free asset and the tangency portfolio: its slope is the maximum Sharpe ratio ``sqrt(H)``, ``H = (mu - rf 1)' S^-1 (mu - rf 1)``, a sum of squares of the
    excess-return forecasts weighted by how independent they are. Returns the slope and the tangency portfolio."""
    assets, S, m = _inputs(mu, cov)
    excess = m - risk_free
    h = float(excess @ _solve(S, excess))
    return {"slope": float(np.sqrt(h)), "tangency": tangency_portfolio(mu, cov, risk_free)}
