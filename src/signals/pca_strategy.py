"""PCA-based statistical arbitrage -- Extension F (Ch. 22 §22.3.5-§22.3.7, spec §54).

The natural progression the specification describes::

    PCA for understanding risk   ->   PCA for generating alpha

The idea: regress each asset's returns on the first ``k`` principal
components estimated from a trailing window. The residual is the part of the
asset's move that its factor exposures do not explain. If residuals mean
revert, a cumulative residual far from zero is a tradable signal, and the
resulting book is factor neutral by construction -- it holds no net exposure
to the components it regressed out.

Everything is estimated on a trailing window and applied forward, so the
factor loadings at ``t`` never use returns after ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.pca import pca_decomposition


def rolling_residuals(returns: pd.DataFrame, n_components: int = 3, window: int = 252,
                      step: int = 21, min_assets: int = 5) -> pd.DataFrame:
    """Residual returns after removing the top ``k`` principal components.

    The decomposition is refit every ``step`` days on the trailing ``window``
    and then applied forward to the next block, which is what makes the
    residuals usable as a signal rather than an in-sample artefact.
    """
    data = returns.dropna(how="any")
    if data.shape[1] < min_assets or len(data) < window + step:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    residuals = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    for end in range(window, len(data), step):
        train = data.iloc[end - window:end]
        try:
            result = pca_decomposition(train, use_correlation=True)
        except (ValueError, np.linalg.LinAlgError):
            continue

        loadings = result.eigenvectors.iloc[:, :n_components].to_numpy()
        scale = train.std(ddof=1).replace(0.0, np.nan)
        block = data.iloc[end:end + step]
        standardised = (block - train.mean()) / scale

        # Project onto the factor space and take what is left over.
        factors = standardised.to_numpy() @ loadings
        explained = factors @ loadings.T
        residual = standardised.to_numpy() - explained
        residuals.iloc[end:end + step] = residual * scale.to_numpy()

    return residuals.reindex(returns.index)


def residual_zscore(residuals: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Z-score of the cumulative residual: the statistical-arbitrage signal.

    Cumulating over ``window`` days measures how far an asset has drifted from
    its factor-implied path; standardising makes the measure comparable across
    assets with different residual volatilities.
    """
    cumulative = residuals.rolling(window, min_periods=max(window // 2, 5)).sum()
    mean = cumulative.rolling(252, min_periods=60).mean()
    std = cumulative.rolling(252, min_periods=60).std(ddof=1)
    return (cumulative - mean) / std.replace(0.0, np.nan)


def pca_stat_arb_signal(returns: pd.DataFrame, n_components: int = 3, window: int = 252,
                        zscore_window: int = 21, step: int = 21) -> pd.DataFrame:
    """The tradable forecast: negative of the residual z-score.

    Negative because the hypothesis is reversion -- an asset that has run
    ahead of what its factor exposures justify is expected to give some of it
    back.
    """
    residuals = rolling_residuals(returns, n_components, window, step)
    return -residual_zscore(residuals, zscore_window)


def factor_exposure_check(weights: pd.Series, returns: pd.DataFrame,
                          n_components: int = 3) -> pd.Series:
    """Residual factor exposures of a book.

    The test of whether the construction did what it claims: a genuinely
    factor-neutral book should have exposures near zero to the components it
    removed.
    """
    data = returns.dropna(how="any")
    result = pca_decomposition(data, use_correlation=True)
    assets = [a for a in weights.index if a in result.eigenvectors.index]
    loadings = result.eigenvectors.loc[assets].iloc[:, :n_components]
    exposures = loadings.T @ weights.reindex(assets).fillna(0.0)
    return exposures.rename("factor_exposure")


def explained_variance_by_component(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """How many components explain the universe? (spec §54, Ch. 22 §22.3.7)"""
    result = pca_decomposition(returns.dropna(how="any").tail(window), use_correlation=True)
    return result.cumulative_variance
