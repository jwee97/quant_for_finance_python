"""Tests for comparing forecasts out of sample (Generation 2).

Three tools, each answering a different question:

``diebold_mariano``  Is forecast A's loss different from forecast B's? For
                     non-nested comparisons (two covariance estimators, two
                     volatility models). Uses a HAC variance because forecast
                     errors from overlapping horizons are serially correlated.

``clark_west``       Does a LARGER model beat the model nested inside it? The
                     standard Diebold-Mariano test is badly undersized here:
                     under the null that the extra regressors are useless, the
                     bigger model's out-of-sample MSPE is *mechanically* worse
                     because it estimates parameters that are truly zero, so a
                     naive test almost never rejects. Clark and West (2007)
                     subtract that noise term. This is the right test for
                     "does adding macro features help?".

``oos_r2``           Campbell-Thompson out-of-sample R-squared against a
                     benchmark forecast. Positive means the model beat it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _hac_mean_test(series: np.ndarray, lag: int) -> tuple[float, float]:
    """t-statistic of the mean of ``series`` with a Newey-West variance."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan")
    centred = x - x.mean()
    gamma0 = float(centred @ centred) / n
    variance = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        variance += 2.0 * weight * float(centred[k:] @ centred[:-k]) / n
    variance = max(variance, 1e-18)
    return float(x.mean()), float(x.mean() / np.sqrt(variance / n))


def diebold_mariano(loss_a: pd.Series | np.ndarray, loss_b: pd.Series | np.ndarray,
                    lag: int = 0, alternative: str = "two-sided") -> dict:
    """Diebold-Mariano test on a loss differential ``d_t = loss_a - loss_b``.

    A positive statistic means ``a`` has the larger loss, i.e. ``b`` is better.
    ``lag`` is the HAC truncation lag; use ``horizon - 1`` for overlapping
    multi-step forecasts.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    d = a - b
    mean, stat = _hac_mean_test(d, lag)
    if not np.isfinite(stat):
        return {}
    if alternative == "two-sided":
        p = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    elif alternative == "greater":          # a has larger loss than b
        p = 1.0 - stats.norm.cdf(stat)
    elif alternative == "less":
        p = stats.norm.cdf(stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    return {"mean_loss_difference": mean, "statistic": stat, "p_value": float(p),
            "n_obs": int(np.isfinite(d).sum())}


def oos_r2(actual: pd.Series | np.ndarray, benchmark: pd.Series | np.ndarray,
           forecast: pd.Series | np.ndarray) -> float:
    """``1 - SSE(model) / SSE(benchmark)``: positive when the model beats the benchmark."""
    y = np.asarray(actual, dtype=float)
    f0 = np.asarray(benchmark, dtype=float)
    f1 = np.asarray(forecast, dtype=float)
    valid = np.isfinite(y) & np.isfinite(f0) & np.isfinite(f1)
    denominator = float(((y[valid] - f0[valid]) ** 2).sum())
    if denominator <= 0:
        return float("nan")
    return float(1.0 - ((y[valid] - f1[valid]) ** 2).sum() / denominator)


def clark_west(actual: pd.Series | np.ndarray, restricted: pd.Series | np.ndarray,
               unrestricted: pd.Series | np.ndarray, lag: int = 0) -> dict:
    """Clark-West (2007) MSPE-adjusted test for nested forecasting models.

    H0: the restricted (small) model is the true one. H1: the unrestricted
    model has genuine predictive content. One-sided: the statistic is large and
    positive only when the bigger model really forecasts better.

        f_t = (y - f1)^2 - [ (y - f2)^2 - (f1 - f2)^2 ]

    The bracketed correction removes the estimation noise the larger model
    adds even when its extra variables are worthless.
    """
    y = np.asarray(actual, dtype=float)
    f1 = np.asarray(restricted, dtype=float)
    f2 = np.asarray(unrestricted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2)
    y, f1, f2 = y[valid], f1[valid], f2[valid]
    if len(y) < 20:
        return {}
    adjusted = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    mean, stat = _hac_mean_test(adjusted, lag)
    if not np.isfinite(stat):
        return {}
    return {
        "statistic": stat,
        "p_value": float(1.0 - stats.norm.cdf(stat)),
        "mspe_adjusted_difference": mean,
        "oos_r2": float(1.0 - ((y - f2) ** 2).sum() / ((y - f1) ** 2).sum()),
        "n_obs": int(len(y)),
    }


def benjamini_hochberg(p_values: pd.Series | np.ndarray, fdr: float = 0.10) -> np.ndarray:
    """Boolean mask of hypotheses rejected under Benjamini-Hochberg at level ``fdr``."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    thresholds = (np.arange(1, m + 1) / m) * fdr
    passing = p[order] <= thresholds
    cutoff = int(np.flatnonzero(passing).max()) if passing.any() else -1
    mask = np.zeros(m, dtype=bool)
    if cutoff >= 0:
        mask[order[:cutoff + 1]] = True
    return mask
