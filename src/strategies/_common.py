"""Helpers shared by the strategy library. Every function here is causal: the value on date ``t`` uses data through ``t`` only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_vol(returns: pd.DataFrame, halflife: float = 40.0) -> pd.DataFrame:
    return np.sqrt((returns ** 2).ewm(halflife=halflife, adjust=False, min_periods=21).mean())


def atr_pct(data, n: int = 14) -> pd.DataFrame:
    """Average true range as a fraction of price. Falls back to absolute returns when the bundle has no high/low."""
    close = data.prices
    if data.high is None or data.low is None:
        return data.returns.abs().ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    prev = close.shift(1)
    tr = pd.concat([(data.high - data.low).stack(), (data.high - prev).abs().stack(), (data.low - prev).abs().stack()], axis=1).max(axis=1).unstack()
    return (tr.reindex_like(close).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / close)


def present(data, names) -> list[str]:
    return [n for n in names if n in data.assets]


def in_classes(data, *classes: str) -> list[str]:
    """Assets whose asset class or group is one of ``classes`` (names come from the universe configuration)."""
    return [a for a in data.assets if data.asset_class.get(a) in classes or data.group.get(a) in classes]


def expanding_z(x: pd.Series | pd.DataFrame, min_periods: int = 252, clip: float = 4.0):
    """Z-score against the expanding history through each date (the date itself included, as a feature known at the close is)."""
    z = (x - x.expanding(min_periods=min_periods).mean()) / x.expanding(min_periods=min_periods).std(ddof=1).replace(0.0, np.nan)
    return z.clip(-clip, clip)


def tanh_score(z, scale: float = 1.0):
    return np.tanh(z / scale)


def stateful(long_in: pd.DataFrame, long_out: pd.DataFrame, short_in: pd.DataFrame | None = None, short_out: pd.DataFrame | None = None) -> pd.DataFrame:
    """A position that is entered on one condition and held until another: +1, -1 or 0, per asset, causally (a forward state machine)."""
    li, lo = long_in.fillna(False).to_numpy(bool), long_out.fillna(False).to_numpy(bool)
    si = short_in.fillna(False).to_numpy(bool) if short_in is not None else np.zeros_like(li)
    so = short_out.fillna(False).to_numpy(bool) if short_out is not None else np.zeros_like(li)
    state = np.zeros(li.shape[1])
    out = np.zeros(li.shape)
    for t in range(li.shape[0]):
        long_exit = (state > 0) & lo[t]
        short_exit = (state < 0) & so[t]
        state = np.where(long_exit | short_exit, 0.0, state)
        state = np.where((state == 0) & li[t], 1.0, state)
        state = np.where((state == 0) & si[t], -1.0, state)
        out[t] = state
    return pd.DataFrame(out, index=long_in.index, columns=long_in.columns)


def only(frame: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Keep ``assets``; every other column is NaN (no forecast, no position)."""
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    keep = [a for a in assets if a in frame.columns]
    out[keep] = frame[keep]
    return out


def zero_except(index: pd.DatetimeIndex, columns, values: dict[str, pd.Series]) -> pd.DataFrame:
    """A score frame that is 0 everywhere except the named assets, which take the given series."""
    out = pd.DataFrame(0.0, index=index, columns=columns)
    for asset, series in values.items():
        if asset in out.columns:
            out[asset] = series.reindex(index)
    return out


def high_low(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily high and low, or the close for both when the bundle has none (so range indicators degrade to close-to-close ones)."""
    close = data.prices
    return (data.high if data.high is not None else close), (data.low if data.low is not None else close)


def wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's smoothing (an exponential average with alpha = 1/n), causal."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def month_end_flags(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each calendar month. The exchange calendar is public in advance, so this is not look-ahead."""
    idx = pd.DatetimeIndex(index)
    period = idx.to_period("M")
    return pd.Series(np.r_[period[1:] != period[:-1], False], index=idx)


def monthly_weights(weights_at_month_end: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hold month-end target weights until the next month-end (the weights are stamped on the decision day; the engine applies the lag)."""
    return weights_at_month_end.reindex(index).ffill()
