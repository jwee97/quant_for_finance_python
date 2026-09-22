"""Inverse volatility -- model M1 (Ch. 19 §19.9.2, spec §26).

    w_i = (1 / sigma_i) / sum_j (1 / sigma_j)

One step beyond equal weight, and it needs only the diagonal of the
covariance matrix. That is the whole point: variances are estimated far more
reliably than correlations, so this captures most of the risk-balancing
benefit while avoiding the part of the estimation problem that does the
damage. It equals true risk parity when all correlations are equal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.volatility import ewma_volatility, rolling_volatility
from .constraints import Constraints


def inverse_volatility_weights(volatility: pd.Series, floor: float = 1e-6) -> pd.Series:
    vol = volatility.astype(float).replace(0.0, np.nan).clip(lower=floor).dropna()
    if vol.empty:
        return pd.Series(dtype=float)
    inverse = 1.0 / vol
    return inverse / inverse.sum()


def inverse_vol_book(returns: pd.DataFrame, investable: pd.DataFrame, lookback: int = 63,
                     method: str = "ewma", halflife: float = 40.0,
                     constraints: Constraints | None = None) -> pd.DataFrame:
    """Inverse-volatility weights through time, using only trailing data."""
    vol = (ewma_volatility(returns, halflife) if method == "ewma"
           else rolling_volatility(returns, lookback))
    vol = vol.where(investable.astype(bool))
    inverse = 1.0 / vol.replace(0.0, np.nan)
    weights = inverse.div(inverse.sum(axis=1), axis=0).fillna(0.0)
    if constraints is not None:
        weights = weights.apply(lambda row: constraints.project(row) if row.sum() > 0 else row, axis=1)
    return weights
