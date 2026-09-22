"""Execution: turning target weights into a held book (spec §21).

The information timeline this module enforces::

    information available through t
              |
              v      signal computed from data <= t
        weights decided at t
              |
              v      signal_lag trading days
        book is HELD from t + lag
              |
              v
        return over (t+lag, t+lag+1] accrues to that book

``apply_execution_lag`` is the single place this shift happens. Any other
module that shifts weights in time is a bug waiting to be found by
``validation.leakage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.dates import rebalance_dates


def apply_execution_lag(weights: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Shift target weights forward by ``lag`` trading days.

    ``lag=1`` means a weight decided using data through the close of day t is
    held from the close of day t+1 onwards, so it earns the return of day
    t+2 first. That is deliberately conservative: it assumes you cannot
    trade the close you measured.
    """
    if lag < 0:
        raise ValueError("execution lag cannot be negative")
    if lag == 0:
        return weights.copy()
    return weights.shift(lag)


def restrict_to_rebalances(weights: pd.DataFrame, index: pd.DatetimeIndex,
                           frequency: str = "monthly") -> pd.DataFrame:
    """Keep target weights only on rebalance dates; hold in between.

    Returns a frame with targets on rebalance dates and NaN elsewhere, which
    ``drift_weights`` then fills according to how the book actually evolves.
    """
    marks = rebalance_dates(index, frequency)
    out = pd.DataFrame(np.nan, index=index, columns=weights.columns)
    common = marks.intersection(weights.index)
    out.loc[common] = weights.loc[common]
    return out


def drift_weights(targets: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Evolve the book between rebalances with realised asset returns.

    A portfolio set to 25/25/25/25 does not stay there: the winners grow. This
    is not a detail. Ignoring drift and re-imposing target weights every day
    silently assumes a daily rebalance and charges none of its turnover, which
    is one of the easiest ways to manufacture a costless strategy.

    ``targets`` carries target weights on rebalance dates and NaN elsewhere.
    """
    columns = targets.columns
    index = targets.index
    ret = returns.reindex(index).reindex(columns=columns).fillna(0.0).to_numpy(dtype=float)
    target_values = targets.to_numpy(dtype=float)

    held = np.full_like(target_values, np.nan)
    current = None
    for i in range(len(index)):
        row = target_values[i]
        if np.isfinite(row).any():
            current = np.nan_to_num(row, nan=0.0)
        if current is None:
            continue
        held[i] = current
        # Grow the book by the day's returns for the NEXT row's starting point.
        grown = current * (1.0 + ret[i] if i + 1 < len(index) else 1.0)
        total = np.abs(grown).sum()
        # Preserve gross exposure: drift changes relative weights, not leverage.
        gross = np.abs(current).sum()
        current = grown * (gross / total) if total > 1e-12 else grown
    return pd.DataFrame(held, index=index, columns=columns)


def build_held_weights(targets: pd.DataFrame, returns: pd.DataFrame, frequency: str = "monthly",
                       lag: int = 1, drift: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full execution pipeline: lag, restrict to rebalances, drift.

    Returns ``(held, traded)`` where ``held`` is the book in force on each
    date and ``traded`` is the weight change actually executed, which is what
    the cost model charges for.
    """
    lagged = apply_execution_lag(targets, lag)
    scheduled = restrict_to_rebalances(lagged, pd.DatetimeIndex(targets.index), frequency)

    if drift:
        held = drift_weights(scheduled, returns)
    else:
        held = scheduled.ffill()

    # Traded = target on a rebalance date minus what had drifted into the book.
    previous = held.shift(1)
    traded = pd.DataFrame(0.0, index=held.index, columns=held.columns)
    rebalanced = scheduled.notna().any(axis=1)
    traded.loc[rebalanced] = (held.loc[rebalanced] - previous.loc[rebalanced].fillna(0.0))
    first_valid = held.dropna(how="all").index
    if len(first_valid):
        traded.loc[first_valid[0]] = held.loc[first_valid[0]].fillna(0.0)
    return held, traded


def vol_target_scaling(weights: pd.DataFrame, returns: pd.DataFrame, target_vol: float = 0.10,
                       lookback: int = 63, max_leverage: float = 1.5,
                       periods_per_year: int = 252) -> tuple[pd.DataFrame, pd.Series]:
    """Scale the whole book so its forecast volatility matches a target.

    The forecast uses the covariance of returns up to and including ``t``
    only, applied to the weights held at ``t``; the scalar is then lagged by
    one day before use, so nothing here can see the volatility of the day it
    is sizing.
    """
    aligned = returns.reindex(weights.index).reindex(columns=weights.columns)
    portfolio = (weights.shift(1) * aligned).sum(axis=1)
    realised = portfolio.rolling(lookback, min_periods=max(lookback // 2, 20)).std(ddof=1) \
        * np.sqrt(periods_per_year)
    scalar = (target_vol / realised.replace(0.0, np.nan)).clip(upper=max_leverage).shift(1)
    scalar = scalar.fillna(1.0)
    return weights.mul(scalar, axis=0), scalar
