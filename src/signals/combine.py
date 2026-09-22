"""Combining signals and strategies (Ch. 22 §22.5, spec §38).

Two distinct levels of combination, kept deliberately separate:

**Signal level** -- blend forecasts into one forecast, then size once. Suitable
when the signals describe the same horizon and you want a single book.

**Strategy level** -- run each signal as its own strategy and blend the
*return streams*. This is the level at which the book's point about
diversification applies: a strategy with a lower Sharpe ratio can still
improve the total if its return stream is uncorrelated with the others, which
is a statement about returns, not about forecasts.

    sigma_p^2 = sum_i w_i^2 sigma_i^2 + 2 sum_{i<j} w_i w_j rho_ij sigma_i sigma_j

Three blending schemes are provided: equal weight, inverse volatility, and
IC-weighted (Ch. 20 §20.1.8, the Fundamental Law: allocate to skill).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine_signals(signals: dict[str, pd.DataFrame], weights: dict[str, float] | None = None,
                    standardise: bool = True) -> pd.DataFrame:
    """Blend forecast frames into one.

    Standardising each signal cross-sectionally before blending is not
    cosmetic: a raw momentum in return units and a z-score live on completely
    different scales, and summing them without standardising silently makes
    the larger-scaled one the whole signal.
    """
    if not signals:
        raise ValueError("no signals to combine")
    weights = weights or {name: 1.0 for name in signals}
    missing = set(signals) - set(weights)
    if missing:
        raise KeyError(f"no weight supplied for: {sorted(missing)}")

    total = sum(abs(w) for w in weights.values())
    if total <= 0:
        raise ValueError("signal weights sum to zero")

    combined = None
    for name, frame in signals.items():
        work = frame
        if standardise:
            centred = work.sub(work.mean(axis=1), axis=0)
            work = centred.div(work.std(axis=1, ddof=1).replace(0.0, np.nan), axis=0)
        contribution = work * (weights[name] / total)
        combined = contribution if combined is None else combined.add(contribution, fill_value=0.0)
    return combined


def signal_correlation(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average cross-sectional correlation between signal pairs.

    Computed per date and then averaged, which is the correct order: pooling
    all (asset, date) pairs into one correlation would mix cross-sectional
    and time-series variation.
    """
    names = list(signals)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            first, second = signals[a].align(signals[b], join="inner")
            daily = first.corrwith(second, axis=1)
            value = float(daily.mean())
            out.loc[a, b] = out.loc[b, a] = value
    return out


def inverse_volatility_weights(returns: pd.DataFrame, lookback: int = 252,
                               min_periods: int = 60) -> pd.DataFrame:
    """Time-varying inverse-volatility weights for a set of return streams."""
    vol = returns.rolling(lookback, min_periods=min_periods).std(ddof=1)
    inverse = 1.0 / vol.replace(0.0, np.nan)
    return inverse.div(inverse.sum(axis=1), axis=0)


def ic_weights(ic_series: dict[str, pd.Series], lookback: int = 252, floor: float = 0.0) -> pd.DataFrame:
    """Weights proportional to trailing information coefficient.

    The Fundamental Law of Active Management says skill and breadth drive the
    information ratio, so allocating across strategies in proportion to
    demonstrated IC is the natural extension. ``floor`` clips negative trailing
    IC to zero: a strategy currently showing no skill gets no capital rather
    than a short position, because a negative IC is far more likely to be noise
    than a reliable contrarian edge.
    """
    frame = pd.DataFrame(ic_series)
    rolling = frame.rolling(lookback, min_periods=max(lookback // 4, 20)).mean()
    clipped = rolling.clip(lower=floor)
    total = clipped.sum(axis=1).replace(0.0, np.nan)
    weights = clipped.div(total, axis=0)
    # Before any strategy has a trailing IC, fall back to equal weight rather
    # than holding nothing.
    return weights.fillna(1.0 / max(frame.shape[1], 1))


def combine_strategy_returns(returns: pd.DataFrame, method: str = "inverse_vol",
                             lookback: int = 252, ic_series: dict[str, pd.Series] | None = None,
                             lag: int = 1) -> tuple[pd.Series, pd.DataFrame]:
    """Blend strategy return streams; returns ``(combined, weights)``.

    Weights are lagged by ``lag`` days: the allocation held today may only use
    information available yesterday. Without the lag, an inverse-volatility
    blend quietly gets to see the volatility of the very day it is sizing.
    """
    if method == "equal":
        weights = pd.DataFrame(1.0 / returns.shape[1], index=returns.index, columns=returns.columns)
    elif method == "inverse_vol":
        weights = inverse_volatility_weights(returns, lookback)
    elif method == "ic_weighted":
        if not ic_series:
            raise ValueError("ic_weighted combination requires ic_series")
        weights = ic_weights(ic_series, lookback).reindex(returns.index).ffill()
    else:
        raise ValueError(f"unknown combination method '{method}'")

    weights = weights.shift(lag).reindex_like(returns)
    weights = weights.div(weights.sum(axis=1).replace(0.0, np.nan), axis=0)
    combined = (weights * returns).sum(axis=1, skipna=True)
    valid = weights.notna().any(axis=1) & returns.notna().any(axis=1)
    return combined.where(valid), weights


def diversification_table(returns: pd.DataFrame, combined: pd.Series | None = None,
                          periods_per_year: int = 252) -> pd.DataFrame:
    """Show what the combination bought, in Sharpe-ratio terms.

    The point the book makes in §22.5 is quantitative: the combined Sharpe
    ratio exceeds the weighted average of the individual Sharpe ratios by a
    factor that depends only on the correlation between the streams. This
    table puts that number on the page.
    """
    rows = {}
    for name in returns.columns:
        series = returns[name].dropna()
        vol = series.std(ddof=1) * np.sqrt(periods_per_year)
        rows[name] = {
            "ann_return": float(series.mean() * periods_per_year),
            "ann_vol": float(vol),
            "sharpe": float(series.mean() * periods_per_year / vol) if vol > 0 else np.nan,
        }
    table = pd.DataFrame(rows).T
    if combined is not None:
        series = combined.dropna()
        vol = series.std(ddof=1) * np.sqrt(periods_per_year)
        table.loc["combined"] = {
            "ann_return": float(series.mean() * periods_per_year),
            "ann_vol": float(vol),
            "sharpe": float(series.mean() * periods_per_year / vol) if vol > 0 else np.nan,
        }
        individual = table.drop(index="combined")["sharpe"]
        table.loc["combined", "avg_component_sharpe"] = float(individual.mean())
        table.loc["combined", "best_component_sharpe"] = float(individual.max())
        table.loc["combined", "diversification_gain"] = float(
            table.loc["combined", "sharpe"] - individual.mean()
        )
    return table
