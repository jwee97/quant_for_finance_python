"""Risk parity -- model M2 (Ch. 19 §19.9, spec §27).

    sigma_p = sqrt(w' Sigma w)
    MRC_i   = d sigma_p / d w_i = (Sigma w)_i / sigma_p
    RC_i    = w_i * MRC_i,      sum_i RC_i = sigma_p   (Euler's theorem)

The target is ``RC_1 = ... = RC_N``, i.e. every asset contributes the same
share of portfolio risk. Unlike inverse volatility, this accounts for
correlations: two highly correlated assets are recognised as one bet.

Two solvers. The cyclical-coordinate-descent method of Spinu (2013) is the
default because it is fast and converges reliably for long-only books; SLSQP
is available for the constrained case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import Constraints
from .covariance import nearest_positive_definite, portfolio_volatility


def marginal_risk_contributions(weights, covariance) -> np.ndarray:
    """``MRC_i = (Sigma w)_i / sigma_p``."""
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    sigma = portfolio_volatility(w, cov)
    if sigma <= 1e-14:
        return np.zeros_like(w)
    return (cov @ w) / sigma


def risk_contributions(weights, covariance, normalise: bool = False) -> np.ndarray:
    """``RC_i = w_i * MRC_i``; these sum to the portfolio volatility."""
    w = np.asarray(weights, dtype=float).ravel()
    rc = w * marginal_risk_contributions(w, covariance)
    if normalise:
        total = rc.sum()
        return rc / total if abs(total) > 1e-14 else rc
    return rc


def risk_contribution_frame(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Per-asset weight, marginal contribution, contribution and share."""
    assets = list(weights.index)
    cov = covariance.loc[assets, assets]
    mrc = marginal_risk_contributions(weights.to_numpy(), cov.to_numpy())
    rc = weights.to_numpy() * mrc
    total = rc.sum()
    return pd.DataFrame(
        {
            "weight": weights.to_numpy(),
            "marginal_risk": mrc,
            "risk_contribution": rc,
            "risk_share": rc / total if abs(total) > 1e-14 else np.nan,
        },
        index=assets,
    )


def risk_concentration(weights, covariance) -> float:
    """Dispersion of risk shares. Zero for a perfect risk-parity solution."""
    rc = risk_contributions(weights, covariance, normalise=True)
    n = len(rc)
    return float(np.sqrt(np.sum((rc - 1.0 / n) ** 2)))


def _newton_risk_parity(cov: np.ndarray, budget: np.ndarray, max_iter: int = 500,
                        tolerance: float = 1e-10) -> np.ndarray:
    """Cyclical coordinate descent on the convex risk-parity formulation.

    Minimises ``0.5 w' Sigma w - sum_i b_i log(w_i)`` over ``w > 0``. The
    logarithmic barrier makes the problem strictly convex, so the solution is
    unique and the coordinate updates have a closed form: each step solves a
    scalar quadratic.
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        previous = w.copy()
        for i in range(n):
            # a x^2 + b x - budget_i = 0 with a = Sigma_ii, b = (Sigma w)_i - Sigma_ii w_i
            a = cov[i, i]
            b = float(cov[i] @ w) - cov[i, i] * w[i]
            if a <= 1e-18:
                continue
            w[i] = (-b + np.sqrt(b * b + 4.0 * a * budget[i])) / (2.0 * a)
        if np.max(np.abs(w - previous)) < tolerance:
            break
    total = w.sum()
    return w / total if total > 1e-14 else np.full(n, 1.0 / n)


def _slsqp_risk_parity(cov: np.ndarray, budget: np.ndarray, assets,
                       constraints: Constraints | None, max_iter: int = 400) -> np.ndarray:
    n = cov.shape[0]

    def objective(w):
        sigma = np.sqrt(max(w @ cov @ w, 1e-18))
        rc = w * (cov @ w) / sigma
        share = rc / max(rc.sum(), 1e-18)
        return float(np.sum((share - budget) ** 2))

    bounds = constraints.bounds(assets) if constraints else [(1e-6, 1.0)] * n
    bounds = [(max(lo, 1e-6), hi) for lo, hi in bounds]
    cons = constraints.scipy_constraints(assets) if constraints else [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    ]
    result = minimize(objective, np.full(n, 1.0 / n), method="SLSQP", bounds=bounds,
                      constraints=cons, options={"maxiter": max_iter, "ftol": 1e-12})
    return result.x if result.success else np.full(n, 1.0 / n)


def risk_parity_weights(covariance: pd.DataFrame, budget: pd.Series | None = None,
                        method: str = "newton", constraints: Constraints | None = None,
                        max_iter: int = 500, tolerance: float = 1e-10) -> pd.Series:
    """Weights whose risk contributions match ``budget`` (equal by default)."""
    assets = list(covariance.columns)
    cov = nearest_positive_definite(covariance).to_numpy(dtype=float)
    n = len(assets)
    if budget is None:
        target = np.full(n, 1.0 / n)
    else:
        target = budget.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        total = target.sum()
        target = target / total if total > 0 else np.full(n, 1.0 / n)

    if method == "newton":
        weights = _newton_risk_parity(cov, target, max_iter, tolerance)
        if constraints is not None:
            return constraints.project(pd.Series(weights, index=assets))
    elif method == "slsqp":
        weights = _slsqp_risk_parity(cov, target, assets, constraints, max_iter)
    else:
        raise ValueError(f"unknown risk-parity method '{method}'")
    return pd.Series(weights, index=assets)


def risk_parity_book(returns: pd.DataFrame, investable: pd.DataFrame, lookback: int = 252,
                     rebalance_index: pd.DatetimeIndex | None = None,
                     covariance_method: str = "shrinkage", method: str = "newton",
                     constraints: Constraints | None = None, min_assets: int = 5,
                     halflife: float = 60.0) -> pd.DataFrame:
    """Risk-parity weights through time, re-solved on each rebalance date."""
    from .covariance import estimate_covariance

    index = pd.DatetimeIndex(returns.index)
    dates = rebalance_index if rebalance_index is not None else index
    book = pd.DataFrame(np.nan, index=index, columns=returns.columns)

    for stamp in dates:
        if stamp not in index:
            continue
        position = index.get_loc(stamp)
        if position < lookback:
            continue
        window = returns.iloc[position - lookback:position]
        live = [c for c in returns.columns
                if bool(investable.loc[stamp, c]) and window[c].notna().sum() > lookback * 0.8]
        if len(live) < min_assets:
            continue
        try:
            cov = estimate_covariance(window[live].dropna(how="any"), covariance_method,
                                      lookback, halflife, annualise=True)
            weights = risk_parity_weights(cov, None, method, constraints)
        except (ValueError, np.linalg.LinAlgError):
            continue
        book.loc[stamp, live] = weights.reindex(live).to_numpy()
    return book.ffill().fillna(0.0)
