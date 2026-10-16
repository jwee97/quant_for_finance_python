"""Features of a price history that several forecasters share: volatility estimates, and the standardised description of "where the market is now" that the Gaussian process and the analogue forecaster learn from.

Everything at position ``t`` uses prices up to ``t`` and no later one. Returns are measured in units of the volatility of the day, so a move that is large for a calm market and ordinary for a stormy one is
described as it was (a 2% day is a shock at 0.5% daily volatility and nothing at 2%). That puts different periods, and different stocks, on one scale.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastError

LAMBDA = 0.94                                        # RiskMetrics decay
FEATURES = ("move over 5 days", "move over 21 days", "move over 63 days", "recent against long-run volatility", "distance from the 126-day high")
WARMUP = 127                                         # prices needed before the first feature (the 126-day high and the 126-day volatility)


def ewma_variance(r: np.ndarray, lam: float = LAMBDA, start: int = 30) -> np.ndarray:
    """EWMA variance after each return: ``v[i]`` uses ``r[0..i]`` (the first ``start`` returns seed it with their mean square)."""
    r = np.asarray(r, dtype=float)
    if len(r) < start:
        raise ForecastError("too few returns for an exponentially weighted variance")
    v = np.empty(len(r))
    cur = float(np.mean(r[:start] ** 2))
    for i, x in enumerate(r):
        cur = lam * cur + (1.0 - lam) * x * x
        v[i] = cur
    return v


def rolling_rms(r: np.ndarray, days: int) -> np.ndarray:
    """Root mean square of the last ``days`` returns after each return (NaN until there are that many)."""
    r = np.asarray(r, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(r * r)])
    out = np.full(len(r), np.nan)
    out[days - 1:] = np.sqrt((c[days:] - c[:-days]) / days)
    return out


def feature_table(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features and the volatility to scale outcomes by, for every price position ``t`` (rows before ``WARMUP`` are NaN).

    Columns (``FEATURES``): the move over 5, 21 and 63 days in units of the volatility over that many days; the log of the 10-day against the 126-day volatility (is the market calmer or stormier
    than usual?); the distance from the 126-day high in units of the volatility over 126 days (zero at a high, negative below it). Returns ``(X, sigma)`` where ``sigma[t]`` is the EWMA daily volatility
    after the return into day ``t``.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    X = np.full((n, len(FEATURES)), np.nan)
    sigma = np.full(n, np.nan)
    if n <= WARMUP:
        return X, sigma
    L = np.log(close)
    r = np.diff(L)
    sigma[1:] = np.sqrt(ewma_variance(r))
    s10, s126 = np.full(n, np.nan), np.full(n, np.nan)               # the volatility ending with the return into day t
    s10[1:], s126[1:] = rolling_rms(r, 10), rolling_rms(r, 126)
    for j, d in enumerate((5, 21, 63)):
        X[d:, j] = (L[d:] - L[:-d]) / (sigma[d:] * np.sqrt(d))
    X[:, 3] = np.log(s10 / s126)
    hi = np.full(n, np.nan)
    hi[126:] = np.lib.stride_tricks.sliding_window_view(L, 127).max(axis=1)
    X[:, 4] = (L - hi) / (s126 * np.sqrt(126))
    X[:WARMUP] = np.nan
    return X, sigma
