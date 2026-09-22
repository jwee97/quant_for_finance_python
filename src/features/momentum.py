"""Momentum features (Ch. 22 §22.3.1, §22.3.9).

The raw quantity is total return over a lookback::

    M_{i,t}^{(k)} = P_{i,t} / P_{i,t-k} - 1

Three deliberate refinements, each answering a different question:

``skip_days``     skip the most recent day(s). Short-horizon reversal is a
                  distinct effect; leaving it inside a 12-month momentum
                  measure contaminates the signal and, worse, assumes you can
                  trade on the same close you measure.
``vol_scaled``    divide by realised volatility. Turns "which asset moved
                  most" into "which asset moved most per unit of risk", which
                  is the comparison that makes sense across a universe holding
                  both SHY (1.5% vol) and SLV (33% vol).
``ranked``        cross-sectional rank. Discards the magnitude entirely and
                  keeps only the ordering, which is robust to the fat tails
                  documented in the EDA stage.

All features are causal by construction: the value stamped at ``t`` uses
prices up to and including ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .volatility import ewma_volatility, rolling_volatility


def total_return_momentum(prices: pd.DataFrame, lookback: int = 126, skip: int = 0) -> pd.DataFrame:
    """``P_{t-skip} / P_{t-skip-lookback} - 1``."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    base = prices.shift(skip)
    return base / base.shift(lookback) - 1.0


def log_momentum(prices: pd.DataFrame, lookback: int = 126, skip: int = 0) -> pd.DataFrame:
    base = np.log(prices).shift(skip)
    return base - base.shift(lookback)


def volatility_scaled_momentum(prices: pd.DataFrame, returns: pd.DataFrame, lookback: int = 126,
                               skip: int = 0, vol_lookback: int = 63, vol_method: str = "rolling",
                               halflife: float = 40.0) -> pd.DataFrame:
    """Momentum per unit of risk: ``M / sigma``.

    The volatility is annualised and measured over its own window, so the
    ratio is roughly a t-statistic on the trend rather than a return.
    """
    raw = total_return_momentum(prices, lookback, skip)
    if vol_method == "ewma":
        vol = ewma_volatility(returns, halflife=halflife, annualise=True)
    else:
        vol = rolling_volatility(returns, window=vol_lookback, annualise=True)
    vol = vol.shift(skip).replace(0.0, np.nan)
    return raw / vol


def cross_sectional_rank(feature: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Rank across assets on each date, ignoring assets with no value."""
    ranked = feature.rank(axis=1, pct=pct, method="average")
    return ranked.where(feature.notna())


def ranked_momentum(prices: pd.DataFrame, lookback: int = 126, skip: int = 0,
                    centred: bool = True) -> pd.DataFrame:
    """Cross-sectional percentile rank of momentum, optionally centred on 0."""
    ranks = cross_sectional_rank(total_return_momentum(prices, lookback, skip))
    return (ranks - 0.5) * 2.0 if centred else ranks


def moving_average_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Classic trend filter, expressed as a continuous ratio rather than a
    binary signal so that it can be standardised like the other features."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    fast_ma = prices.rolling(fast, min_periods=max(fast // 2, 5)).mean()
    slow_ma = prices.rolling(slow, min_periods=max(slow // 2, 10)).mean()
    return fast_ma / slow_ma - 1.0


def time_series_momentum_sign(prices: pd.DataFrame, lookback: int = 252, skip: int = 0) -> pd.DataFrame:
    """Sign of trailing return: the canonical time-series (trend) signal."""
    return np.sign(total_return_momentum(prices, lookback, skip))


def momentum_family(prices: pd.DataFrame, returns: pd.DataFrame, lookbacks=(21, 63, 126, 252),
                    skip: int = 1, vol_lookback: int = 63) -> dict[str, pd.DataFrame]:
    """Every (variant, lookback) combination, keyed ``variant_lookback``.

    Built as a family on purpose: reporting one lookback that happens to work
    is parameter mining (Ch. 22 §22.2.4), so the whole surface is always
    available to the robustness stage.
    """
    out: dict[str, pd.DataFrame] = {}
    for lookback in lookbacks:
        out[f"raw_{lookback}"] = total_return_momentum(prices, lookback, skip)
        out[f"vol_scaled_{lookback}"] = volatility_scaled_momentum(
            prices, returns, lookback, skip, vol_lookback
        )
        out[f"ranked_{lookback}"] = ranked_momentum(prices, lookback, skip)
    return out
