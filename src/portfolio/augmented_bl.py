"""Black-Litterman with views on factors as well as on assets: the augmented universe.

A view like "value will beat growth by 2% a year" is a view on a *factor*, not on any one asset. Treat the factors as extra assets: stack ``z = (r, f)`` and write the joint covariance that the factor model
``r = B f + e`` implies,

    Sigma_zz = [[B F B' + D,  B F], [F B', F]]

with ``F`` the factor covariance and ``D`` the specific variances. The prior mean of the assets is the equilibrium return ``Pi``; the prior mean of the factors is the generalised-least-squares projection of
``Pi`` on the loadings, ``(B' Sigma^-1 B)^-1 B' Sigma^-1 Pi``, the factor returns that explain the equilibrium best. Views are rows over the stacked vector (a row may mix assets and factors), and the posterior is the usual
Black-Litterman one (He and Litterman: ``Omega`` proportional to the variance of each view) on ``z``. The asset part of the posterior mean is what the optimiser uses.

Two properties follow and are tested: with views only on assets this is the ordinary Black-Litterman posterior (marginalising a Gaussian), and a factor view moves each asset in proportion to its covariance with the
factor, so an asset with no loading on it does not move.

This is the augmented-universe construction. The book's own formulas for its augmented model were not available to check against, so this is documented as that construction and nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .covariance import nearest_positive_definite


def joint_covariance(loadings: pd.DataFrame, factor_cov: pd.DataFrame, specific_var: pd.Series) -> pd.DataFrame:
    """``Sigma_zz`` for ``z = (assets, factors)`` under ``r = B f + e``."""
    B = loadings.to_numpy(dtype=float)
    F = factor_cov.reindex(index=loadings.columns, columns=loadings.columns).to_numpy(dtype=float)
    D = np.diag(specific_var.reindex(loadings.index).to_numpy(dtype=float))
    top = np.hstack([B @ F @ B.T + D, B @ F])
    bottom = np.hstack([F @ B.T, F])
    names = list(loadings.index) + list(loadings.columns)
    return pd.DataFrame(np.vstack([top, bottom]), index=names, columns=names)


def factor_prior(loadings: pd.DataFrame, asset_cov: pd.DataFrame, equilibrium: pd.Series) -> pd.Series:
    """The factor returns that best explain the equilibrium asset returns, in the sense of generalised least squares."""
    B = loadings.to_numpy(dtype=float)
    Si = np.linalg.inv(nearest_positive_definite(asset_cov.reindex(index=loadings.index, columns=loadings.index)).to_numpy(dtype=float))
    pi = equilibrium.reindex(loadings.index).to_numpy(dtype=float)
    return pd.Series(np.linalg.solve(B.T @ Si @ B, B.T @ Si @ pi), index=loadings.columns)


def augmented_posterior(loadings: pd.DataFrame, factor_cov: pd.DataFrame, specific_var: pd.Series, equilibrium: pd.Series, views: pd.DataFrame, view_returns: pd.Series, tau: float = 0.05,
                        view_confidence: float = 0.30) -> dict:
    """The posterior mean of the assets and of the factors. ``views`` has one row per view and a column for every asset and factor (missing columns are zero); ``view_returns`` the return each view states.
    Returns ``assets`` and ``factors`` (the posterior means), the ``prior`` ones, and the covariance ``sigma`` of the augmented universe."""
    if views is None or views.empty:
        raise ValueError("give at least one view")
    sigma = joint_covariance(loadings, factor_cov, specific_var)
    names = list(sigma.columns)
    bad = [c for c in views.columns if c not in names]
    if bad:
        raise ValueError(f"views refer to unknown names: {', '.join(map(str, bad))}")
    pi_f = factor_prior(loadings, sigma.loc[loadings.index, loadings.index], equilibrium)
    pi = np.concatenate([equilibrium.reindex(loadings.index).to_numpy(dtype=float), pi_f.to_numpy(dtype=float)])
    P = views.reindex(columns=names).fillna(0.0).to_numpy(dtype=float)
    Q = view_returns.reindex(views.index).to_numpy(dtype=float)
    S = sigma.to_numpy(dtype=float)
    tS = tau * S
    omega = np.diag(np.diag(P @ tS @ P.T)) / max(view_confidence, 1e-6) + 1e-12 * np.eye(len(Q))
    post = pi + tS @ P.T @ np.linalg.solve(P @ tS @ P.T + omega, Q - P @ pi)                       # the posterior mean in covariance form: no inverse of the prior covariance is needed
    k = len(loadings.index)
    return {"assets": pd.Series(post[:k], index=loadings.index), "factors": pd.Series(post[k:], index=loadings.columns), "prior_assets": equilibrium.reindex(loadings.index), "prior_factors": pi_f, "sigma": sigma}
