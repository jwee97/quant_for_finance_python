"""Maximum-diversification portfolio (Choueifaty and Coignard 2008): maximise the ratio of the weighted average volatility to the portfolio volatility.

    max_w  (w' sigma) / sqrt(w' Sigma w)      s.t. the usual constraints

The ratio is one for a single asset or for perfectly correlated assets, and grows as the assets diversify one another. It needs no expected returns: it is the maximum-Sharpe
portfolio under the assumption that every asset has the same Sharpe ratio, which makes it a risk-based alternative to minimum variance that does not collapse into the lowest-volatility asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constraints import Constraints
from .covariance import nearest_positive_definite
from .mean_variance import OptimisationResult, _solve


def diversification_ratio(weights, covariance) -> float:
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
    return float(w @ np.sqrt(np.diag(cov)) / vol)


def maximum_diversification_weights(covariance: pd.DataFrame, constraints: Constraints | None = None, max_iter: int = 400) -> OptimisationResult:
    assets = list(covariance.columns)
    cov = nearest_positive_definite(covariance).to_numpy(dtype=float)
    sigma = np.sqrt(np.diag(cov))

    def objective(w):
        return -float(w @ sigma) / float(np.sqrt(max(w @ cov @ w, 1e-18)))

    def gradient(w):
        var = max(float(w @ cov @ w), 1e-18)
        vol = np.sqrt(var)
        return -(sigma / vol - (w @ sigma) * (cov @ w) / (vol * var))

    x, success, message, iterations = _solve(objective, gradient, assets, constraints, 1.0 / sigma / np.sum(1.0 / sigma), None, max_iter)
    return OptimisationResult(weights=pd.Series(x, index=assets), expected_return=0.0, expected_volatility=float(np.sqrt(x @ cov @ x)), objective=objective(x),
                              success=success, message=message, iterations=iterations)
