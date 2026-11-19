"""Turning a probability into a position (Lopez de Prado, chapter 10).

A classifier that says "this trade will pay with probability p" should bet more when p is higher and nothing when it is no better than chance. ``size_from_probability`` maps a probability to a size in [-1, 1]
through the test statistic ``z = (p - 1/n) / sqrt(p (1 - p))`` and the normal CDF, ``m = side x (2 Phi(z) - 1)``. ``average_active`` then averages the bets that are open on each day, so that many overlapping bets
do not pile up into one huge position and a position changes gradually, and ``discretize`` rounds sizes to a step to cut the turnover of tiny changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def size_from_probability(p, side=1.0, n_classes: int = 2, floor: float = 0.0) -> np.ndarray:
    """Bet size in [-1, 1] from the probability ``p`` that the bet is right. ``floor`` (default 0) is the smallest probability that takes a position; anything at or below ``1/n_classes`` gets size 0.
    ``p`` is clipped to (0, 1) so that a probability of exactly 0 or 1 gives size +-1 rather than a division by zero."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    z = (p - 1.0 / n_classes) / np.sqrt(p * (1.0 - p))
    m = 2.0 * norm.cdf(z) - 1.0
    m = np.where((p <= 1.0 / n_classes) | (p < floor), 0.0, m)
    return np.asarray(side, dtype=float) * m


def sigmoid_size(divergence, w: float = 1.0) -> np.ndarray:
    """Size from the divergence between a forecast and the current price, ``x / sqrt(w + x^2)``: 0 at no divergence, towards +-1 as it grows; ``w`` sets how fast."""
    x = np.asarray(divergence, dtype=float)
    return x / np.sqrt(w + x * x)


def discretize(m, step: float = 0.1) -> np.ndarray:
    """Round sizes to the nearest multiple of ``step`` (and keep them within [-1, 1])."""
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    return np.clip(np.round(np.asarray(m, dtype=float) / step) * step, -1.0, 1.0)


def average_active(sizes: pd.Series, pos0: np.ndarray, pos1: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """The average size of the bets open on each row of ``index``: a bet is open on rows ``pos0 + 1 .. pos1`` (the rows whose returns it earns). Rows with no open bet are 0."""
    n = len(index)
    s, e = np.asarray(pos0) + 1, np.asarray(pos1)
    tot, cnt = np.zeros(n + 1), np.zeros(n + 1)
    v = np.asarray(sizes, dtype=float)
    np.add.at(tot, np.clip(s, 0, n), v)
    np.add.at(tot, np.clip(e + 1, 0, n), -v)
    np.add.at(cnt, np.clip(s, 0, n), 1.0)
    np.add.at(cnt, np.clip(e + 1, 0, n), -1.0)
    t, c = np.cumsum(tot)[:n], np.cumsum(cnt)[:n]
    return pd.Series(np.divide(t, c, out=np.zeros(n), where=c > 0.5), index=index, name="position")


def turnover(position: pd.Series) -> float:
    """Average absolute daily change of the position (a measure of how much trading the sizing causes)."""
    return float(position.diff().abs().mean())
