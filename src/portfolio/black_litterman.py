"""Black-Litterman -- Extension A / model M8 (Ch. 19 §19.5, spec §49).

The idea that solves the problem Stage 7 measures. Instead of feeding the
optimiser a noisy historical mean, start from the expected returns the market
itself implies, then tilt only where you have a view:

    Pi     = lambda * Sigma * w_mkt                 (reverse optimisation)
    E[R]   = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
             [(tau Sigma)^-1 Pi + P' Omega^-1 Q]    (posterior)

Because the prior is the equilibrium, an asset with no view keeps its
equilibrium weight. This is why Black-Litterman portfolios look sane while
raw mean-variance portfolios do not: the default answer is the market, and
views move you away from it in proportion to their confidence.

Here the views come from the momentum signal, which is the bridge the
specification asks for between alpha research and portfolio construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constraints import Constraints
from .covariance import nearest_positive_definite
from .mean_variance import OptimisationResult, mean_variance_weights


def implied_equilibrium_returns(covariance: pd.DataFrame, market_weights: pd.Series,
                                risk_aversion: float = 2.5) -> pd.Series:
    """``Pi = lambda Sigma w_mkt``: what the market must expect to hold this book."""
    assets = list(covariance.columns)
    cov = nearest_positive_definite(covariance).to_numpy(dtype=float)
    w = market_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    return pd.Series(risk_aversion * (cov @ w), index=assets, name="equilibrium_return")


def views_from_signal(signal: pd.Series, assets: list[str], spread: float = 0.03,
                      top_k: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a cross-sectional signal into relative views.

    Each view is "asset i outperforms the equal-weighted rest by q", stated
    only for the assets at the extremes of the signal. Relative views are used
    rather than absolute ones because a cross-sectional signal says nothing
    about the level of returns, only about the ordering -- and stating an
    absolute view you do not hold is how Black-Litterman gets misused.
    """
    scores = signal.reindex(assets).dropna()
    if scores.empty:
        return pd.DataFrame(columns=assets), pd.Series(dtype=float)

    standardised = (scores - scores.mean()) / (scores.std(ddof=1) or 1.0)
    ordered = standardised.sort_values()
    k = top_k or max(len(ordered) // 5, 1)
    selected = list(ordered.head(k).index) + list(ordered.tail(k).index)

    rows, q = [], []
    for asset in selected:
        row = pd.Series(0.0, index=assets)
        row[asset] = 1.0
        others = [a for a in assets if a != asset]
        row[others] = -1.0 / len(others)
        rows.append(row)
        q.append(float(standardised[asset]) * spread)
    return pd.DataFrame(rows, index=selected), pd.Series(q, index=selected)


def black_litterman_posterior(covariance: pd.DataFrame, equilibrium: pd.Series,
                              views: pd.DataFrame, view_returns: pd.Series,
                              tau: float = 0.05, view_confidence: float = 0.30) -> pd.Series:
    """Posterior expected returns.

    ``Omega`` is set proportional to the variance of each view portfolio
    (He-Litterman), scaled by ``view_confidence``: a view on a volatile
    combination is automatically held with less confidence than a view on a
    stable one, which is the behaviour you want and the reason not to set
    Omega by hand.
    """
    assets = list(covariance.columns)
    if views.empty:
        return equilibrium.reindex(assets)

    sigma = nearest_positive_definite(covariance).to_numpy(dtype=float)
    P = views.reindex(columns=assets).fillna(0.0).to_numpy(dtype=float)
    Q = view_returns.reindex(views.index).fillna(0.0).to_numpy(dtype=float)
    pi = equilibrium.reindex(assets).fillna(0.0).to_numpy(dtype=float)

    tau_sigma = tau * sigma
    omega = np.diag(np.diag(P @ tau_sigma @ P.T)) / max(view_confidence, 1e-6)
    omega = omega + np.eye(len(Q)) * 1e-12  # keep it invertible

    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(omega)
    precision = tau_sigma_inv + P.T @ omega_inv @ P
    mean = np.linalg.pinv(precision) @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    return pd.Series(mean, index=assets, name="posterior_return")


def black_litterman_weights(covariance: pd.DataFrame, market_weights: pd.Series,
                            signal: pd.Series | None = None, tau: float = 0.05,
                            risk_aversion: float = 2.5, view_confidence: float = 0.30,
                            view_spread: float = 0.03,
                            constraints: Constraints | None = None) -> OptimisationResult:
    """Full Black-Litterman pipeline: equilibrium -> views -> posterior -> MVO."""
    assets = list(covariance.columns)
    equilibrium = implied_equilibrium_returns(covariance, market_weights, risk_aversion)
    if signal is None:
        posterior = equilibrium
    else:
        views, view_returns = views_from_signal(signal, assets, view_spread)
        posterior = black_litterman_posterior(covariance, equilibrium, views, view_returns,
                                              tau, view_confidence)
    return mean_variance_weights(posterior, covariance, risk_aversion, constraints)
