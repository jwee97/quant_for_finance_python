"""Mean-reversion features (Ch. 22 §22.3.1).

The core object is a standardised distance from a trailing mean::

    Z_{i,t} = (P_{i,t} - mu_{i,t}) / sigma_{i,t}

Two design choices worth stating:

*Log prices.* ``mu`` and ``sigma`` are computed on log prices so that the
z-score is scale invariant: a $30 ETF and a $400 ETF with identical percentage
dynamics produce the same z-score. On raw prices they would not.

*Sign convention.* These functions return the **z-score itself**, not a trade.
A mean-reversion hypothesis says the relationship with future returns is
negative; turning that into a position is the job of ``signals``. Keeping the
feature sign-neutral is what lets Stage 4 test the hypothesis instead of
assuming it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _basis(prices: pd.DataFrame, price_basis: str = "log") -> pd.DataFrame:
    if price_basis == "log":
        return np.log(prices)
    if price_basis == "raw":
        return prices
    raise ValueError("price_basis must be 'log' or 'raw'")


def price_zscore(prices: pd.DataFrame, window: int = 21, price_basis: str = "log",
                 min_periods: int | None = None) -> pd.DataFrame:
    """Distance from the trailing mean in trailing standard deviations."""
    basis = _basis(prices, price_basis)
    min_periods = min_periods or max(window // 2, 5)
    mean = basis.rolling(window, min_periods=min_periods).mean()
    std = basis.rolling(window, min_periods=min_periods).std(ddof=1)
    return (basis - mean) / std.replace(0.0, np.nan)


def return_zscore(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Standardised cumulative return over the window (a 'stretch' measure)."""
    cumulative = returns.rolling(window).sum()
    mean = cumulative.rolling(252, min_periods=60).mean()
    std = cumulative.rolling(252, min_periods=60).std(ddof=1)
    return (cumulative - mean) / std.replace(0.0, np.nan)


def short_term_reversal(returns: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Trailing short-horizon return. The reversal hypothesis says its
    relationship with the next period's return is negative."""
    return returns.rolling(window).sum()


def distance_from_high(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Drawdown from the rolling maximum. Negative, zero at a new high."""
    return prices / prices.rolling(window, min_periods=max(window // 4, 20)).max() - 1.0


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI, rescaled to ``[-1, 1]`` and centred at 0.

    Included because it is the best-known bounded reversion indicator and
    therefore a useful robustness check on the z-score: if the two disagree,
    the effect is an artefact of the measure, not the market.
    """
    change = prices.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    raw = 100.0 - 100.0 / (1.0 + rs)
    return (raw - 50.0) / 50.0


def bollinger_position(prices: pd.DataFrame, window: int = 21, n_std: float = 2.0) -> pd.DataFrame:
    """Position inside the Bollinger band: -1 at the lower band, +1 at the upper."""
    mean = prices.rolling(window, min_periods=max(window // 2, 5)).mean()
    std = prices.rolling(window, min_periods=max(window // 2, 5)).std(ddof=1)
    return (prices - mean) / (n_std * std.replace(0.0, np.nan))


def half_life_of_reversion(series: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life from the AR(1) regression

        dP_t = lambda (P_{t-1} - mu) dt + eps,   half-life = -log(2) / lambda

    A half-life longer than the sample, or a positive lambda, means the series
    is not mean reverting on this evidence.
    """
    y = series.dropna()
    if len(y) < 60:
        return float("nan")
    lagged = y.shift(1).dropna()
    delta = (y - y.shift(1)).dropna()
    common = lagged.index.intersection(delta.index)
    x = lagged.loc[common].to_numpy()
    dy = delta.loc[common].to_numpy()
    x_centred = x - x.mean()
    denominator = float(np.dot(x_centred, x_centred))
    if denominator <= 0:
        return float("nan")
    lam = float(np.dot(x_centred, dy - dy.mean()) / denominator)
    if lam >= 0:
        return float("inf")  # no reversion: shocks are not pulled back
    return float(-np.log(2.0) / lam)


def variance_ratio(series: pd.Series, lag: int = 5) -> float:
    """Lo-MacKinlay variance ratio.

    ``VR < 1`` means variance grows more slowly than linearly in time, i.e.
    mean reversion. ``VR > 1`` means trending. ``VR = 1`` is a random walk.
    This is the cleanest single statistic for deciding whether to bother
    building a reversion strategy at all.
    """
    y = series.dropna()
    if len(y) < lag * 10:
        return float("nan")
    single = y.diff().dropna()
    multi = y.diff(lag).dropna()
    var_single = float(single.var(ddof=1))
    var_multi = float(multi.var(ddof=1))
    if var_single <= 0:
        return float("nan")
    return var_multi / (lag * var_single)


def variance_ratio_table(prices: pd.DataFrame, lags=(2, 5, 10, 21, 63)) -> pd.DataFrame:
    """Variance ratios per asset across horizons, on log prices."""
    log_prices = np.log(prices)
    rows = {}
    for column in log_prices.columns:
        rows[column] = {f"vr_{lag}": variance_ratio(log_prices[column], lag) for lag in lags}
        rows[column]["half_life_zscore_21"] = half_life_of_reversion(
            price_zscore(prices[[column]], 21)[column]
        )
    return pd.DataFrame(rows).T


def mean_reversion_family(prices: pd.DataFrame, returns: pd.DataFrame,
                          lookbacks=(5, 10, 21, 63), price_basis: str = "log") -> dict[str, pd.DataFrame]:
    """Every (variant, lookback) combination for the reversion research."""
    out: dict[str, pd.DataFrame] = {}
    for lookback in lookbacks:
        z = price_zscore(prices, lookback, price_basis)
        out[f"zscore_{lookback}"] = z
        out[f"zscore_ranked_{lookback}"] = (z.rank(axis=1, pct=True).where(z.notna()) - 0.5) * 2.0
        out[f"reversal_{lookback}"] = short_term_reversal(returns, lookback)
    return out
