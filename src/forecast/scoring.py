"""Scoring a predictive distribution against what happened.

A forecast that is a distribution is judged by proper scoring rules: rules an honest forecaster cannot beat by shading the answer.

``crps``       the continuous ranked probability score: the average distance of the draws from the outcome, less half the average distance of the draws from each other (Gneiting and Raftery 2007).
               It is in the units of the outcome (here a log return), is the absolute error when the forecast is a single number, and rewards a forecast for being both sharp and right. The default is
               the *fair* form (Ferro 2014): the plain ensemble formula is biased in favour of larger samples, which would make a method that was run with more draws look better.
``pinball``    the loss of one quantile: the 5% quantile is scored by how often, and how far, the outcome falls below it. The mean over many levels approximates the CRPS.
``coverage``   how often the outcome fell inside the central 50% and 90% of the draws, and how often it fell below the 5% quantile (the value-at-risk backtest of Kupiec).
``pit``        where the outcome fell among the draws, as a fraction. When the forecast is calibrated these are uniform on 0 to 1: a hump in the middle means the forecasts were too wide, a U means too
               narrow, a slope means a bias.

Forecasts for overlapping horizons (a 21-day return forecast every 10 days) share most of their outcome, so their losses are not independent. The tests here either thin the origins to one per horizon
(``stride``) or use a Newey-West variance (``lag``) so that a p-value does not pretend to more evidence than there is.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..validation.forecast_tests import _hac_mean_test, benjamini_hochberg, clark_west, diebold_mariano

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
COVERAGES = (0.5, 0.9)


def _rows(samples) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def crps(samples, y, fair: bool = True) -> np.ndarray:
    """CRPS of an ensemble for each outcome. ``samples``: ``(m, n)`` draws (or ``(n,)`` for one forecast); ``y``: ``(m,)`` outcomes. ``fair`` removes the finite-sample bias of the plain formula."""
    x = np.sort(_rows(samples), axis=1)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    m, n = x.shape
    if len(y) != m or n < 2:
        raise ValueError("one outcome per forecast and at least two draws")
    spread = (x * (2.0 * np.arange(n) - n + 1.0)).sum(axis=1)           # the sum of x_(j) - x_(i) over all pairs i < j
    return np.abs(x - y[:, None]).mean(axis=1) - spread / (n * (n - 1.0) if fair else float(n * n))


def pinball(samples, y, levels=LEVELS) -> np.ndarray:
    """Pinball loss of each quantile level, ``(m, len(levels))``: ``tau * (y - q)`` when the outcome is above the quantile, ``(1 - tau) * (q - y)`` when below."""
    x = _rows(samples)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    taus = np.asarray(levels, dtype=float)
    q = np.quantile(x, taus, axis=1).T
    d = y[:, None] - q
    return np.where(d >= 0, taus * d, (taus - 1.0) * d)


def interval_hits(samples, y, coverage: float = 0.9) -> np.ndarray:
    """Whether each outcome fell inside the central ``coverage`` interval of its draws."""
    x = _rows(samples)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    lo, hi = np.quantile(x, [(1.0 - coverage) / 2.0, 1.0 - (1.0 - coverage) / 2.0], axis=1)
    return (y >= lo) & (y <= hi)


def below_quantile(samples, y, level: float = 0.05) -> np.ndarray:
    """Whether each outcome fell below the ``level`` quantile of its draws (a value-at-risk exceedance when the level is small)."""
    x = _rows(samples)
    return np.atleast_1d(np.asarray(y, dtype=float)) < np.quantile(x, level, axis=1)


def pit(samples, y) -> np.ndarray:
    """The probability integral transform: the share of the draws below the outcome (a tie counts half)."""
    x = _rows(samples)
    y = np.atleast_1d(np.asarray(y, dtype=float))[:, None]
    return ((x < y).sum(axis=1) + 0.5 * (x == y).sum(axis=1)) / x.shape[1]


def pit_histogram(values, bins: int = 10) -> np.ndarray:
    """Share of the PIT values in each of ``bins`` equal slices of 0 to 1; a calibrated forecast gives ``1 / bins`` in each."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.full(bins, np.nan)
    return np.histogram(v, bins=bins, range=(0.0, 1.0))[0] / len(v)


# --------------------------------------------------------------------------------------------------------------------------------- comparing and testing
def stride_for(horizon: int, step: int) -> int:
    """Origins ``step`` days apart forecast overlapping stretches when the horizon is longer than the step; keeping one in ``stride`` makes the forecasts independent."""
    return max(1, int(math.ceil(horizon / max(1, step))))


def skill(model_loss, base_loss) -> float:
    """``1 - mean(model) / mean(baseline)``: the share by which the model's average loss is below the baseline's (negative when it is worse)."""
    a, b = np.asarray(model_loss, dtype=float), np.asarray(base_loss, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any() or b[ok].mean() <= 0:
        return float("nan")
    return float(1.0 - a[ok].mean() / b[ok].mean())


def better_than(model_loss, base_loss, lag: int = 0) -> dict:
    """Diebold-Mariano test, one-sided: is the model's mean loss below the baseline's? ``lag``: Newey-West lag (``stride - 1`` for overlapping forecasts). Returns the statistic and ``p`` (small when better)."""
    a, b = np.asarray(model_loss, dtype=float), np.asarray(base_loss, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    out = diebold_mariano(a[ok], b[ok], lag=lag, alternative="less") if ok.sum() >= 10 else {}
    return {"statistic": out.get("statistic", float("nan")), "p": out.get("p_value", float("nan")), "n": int(ok.sum())}


def binomial_p(hits, nominal: float, stride: int = 1) -> float:
    """Two-sided exact binomial p-value that ``hits`` (one in ``stride``) occur with probability ``nominal``: the coverage test of Kupiec when the hits are exceedances."""
    h = np.asarray(hits, dtype=float)[::max(1, stride)]
    h = h[np.isfinite(h)]
    if len(h) < 10:
        return float("nan")
    return float(stats.binomtest(int(h.sum()), len(h), nominal).pvalue)


def pit_p(values, stride: int = 1) -> float:
    """Kolmogorov-Smirnov p-value that the PIT values (one in ``stride``) are uniform; small means the forecasts were not calibrated."""
    v = np.asarray(values, dtype=float)[::max(1, stride)]
    v = v[np.isfinite(v)]
    return float(stats.kstest(v, "uniform").pvalue) if len(v) >= 10 else float("nan")


def mean_skill(y, mean_forecast, benchmark=None, lag: int = 0) -> dict:
    """The forecast of the average return against a benchmark forecast (zero by default; the standard benchmark of the literature is the historical average, Welch and Goyal 2008): Campbell-Thompson
    out-of-sample R-squared (positive: better than the benchmark) and the Clark-West p-value (the right test, because a model that estimates a mean the market does not have is mechanically worse than
    one that forecasts the benchmark)."""
    y, m = np.asarray(y, dtype=float), np.asarray(mean_forecast, dtype=float)
    b = np.zeros_like(y) if benchmark is None else np.asarray(benchmark, dtype=float)
    ok = np.isfinite(y) & np.isfinite(m) & np.isfinite(b)
    if ok.sum() < 20 or float(((y[ok] - b[ok]) ** 2).sum()) <= 0:
        return {"r2": float("nan"), "p": float("nan")}
    r2 = float(1.0 - ((y[ok] - m[ok]) ** 2).sum() / ((y[ok] - b[ok]) ** 2).sum())
    cw = clark_west(y[ok], b[ok], m[ok], lag=lag)
    return {"r2": r2, "p": float(cw.get("p_value", float("nan")))}


def direction_skill(y, p_up, lag: int = 0) -> dict:
    """How often the sign of the return was called right (up when ``p_up`` is above one half) against always saying up, which is right as often as the market rose. The comparison is a Diebold-Mariano test
    of the two 0-1 losses, one-sided."""
    y, p = np.asarray(y, dtype=float), np.asarray(p_up, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 20:
        return {"hit": float("nan"), "always_up": float("nan"), "edge": float("nan"), "p": float("nan")}
    up = y[ok] > 0
    hit = (p[ok] > 0.5) == up
    base = up
    out = diebold_mariano((~hit).astype(float), (~base).astype(float), lag=lag, alternative="less")
    return {"hit": float(hit.mean()), "always_up": float(base.mean()), "edge": float(hit.mean() - base.mean()), "p": float(out.get("p_value", float("nan")))}


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values: the chance of a false alarm among all the tests together, not for each alone."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


__all__ = ["LEVELS", "COVERAGES", "crps", "pinball", "interval_hits", "below_quantile", "pit", "pit_histogram", "stride_for", "skill", "better_than", "binomial_p", "pit_p", "mean_skill", "direction_skill",
           "holm_adjust", "benjamini_hochberg", "clark_west", "diebold_mariano", "_hac_mean_test"]
