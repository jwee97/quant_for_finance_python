"""How plausible is a set of scenarios? Statistics that compare scenarios with the history they were made from.

Each is a number that should be close to zero (or one, where said) for scenarios that look like the data; a generator can pass some and fail others, which is the point of showing them together.

    marginal_ks        the average over assets of the Kolmogorov-Smirnov distance between the scenario and historical daily returns (0 = identical distributions)
    vol_error          the largest relative error of an asset's standard deviation
    corr_error         the largest absolute error of a pairwise correlation
    kurtosis_ratio     average kurtosis of scenarios over that of the history (1 = right; below 1 = tails too thin)
    tail_dep_error     the error of the lower-tail dependence (the chance both of a pair are in their worst 5% together), averaged over pairs
    clustering_error   the error of the first-order autocorrelation of absolute daily returns (volatility clustering), from within-scenario lags
    es_ratio           the 5% expected shortfall of the equal-weighted portfolio's daily return in scenarios over that in the history (1 = right)
    horizon_vol_ratio  the standard deviation of the equal-weighted portfolio's compounded return over the whole horizon, scenarios over history (needs the history to be cut into windows of that length)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _lower_tail(x: np.ndarray, q: float = 0.05) -> np.ndarray:
    n = x.shape[1]
    thr = np.quantile(x, q, axis=0)
    hit = (x <= thr).astype(float)
    both = hit.T @ hit / len(x)
    return both[np.triu_indices(n, 1)] / q if n > 1 else np.zeros(0)


def _acf1(paths: np.ndarray) -> float:
    a = np.abs(paths)
    x, y = a[:, :-1].reshape(-1, a.shape[2]), a[:, 1:].reshape(-1, a.shape[2])
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    return float(np.mean((xc * yc).sum(axis=0) / np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))))


def quality_report(real: pd.DataFrame | np.ndarray, scenarios: np.ndarray, weights=None) -> dict:
    """``real``: (T, N) daily returns; ``scenarios``: (n, horizon, N)."""
    R = np.asarray(real.dropna() if isinstance(real, pd.DataFrame) else real, dtype=float)
    S = np.asarray(scenarios, dtype=float)
    N = R.shape[1]
    flat = S.reshape(-1, N)
    w = np.full(N, 1.0 / N) if weights is None else np.asarray(weights, dtype=float)
    out = {"marginal_ks": float(np.mean([stats.ks_2samp(R[:, j], flat[:, j]).statistic for j in range(N)])),
           "vol_error": float(np.max(np.abs(flat.std(axis=0) / R.std(axis=0) - 1.0)))}
    if N > 1:
        out["corr_error"] = float(np.max(np.abs(np.corrcoef(R.T) - np.corrcoef(flat.T))))
        out["tail_dep_error"] = float(np.mean(np.abs(_lower_tail(R) - _lower_tail(flat))))
    out["kurtosis_ratio"] = float(np.mean(stats.kurtosis(flat, axis=0, fisher=False) / stats.kurtosis(R, axis=0, fisher=False)))
    if S.shape[1] > 1:
        out["clustering_error"] = abs(_acf1(S) - _acf1(R[None]))
    es = lambda x: -np.mean(np.sort(x)[:max(int(0.05 * len(x)), 1)])           # noqa: E731
    out["es_ratio"] = float(es(flat @ w) / es(R @ w))
    h = S.shape[1]
    if h > 1 and len(R) >= 4 * h:
        k = len(R) // h
        hist = np.prod(1.0 + (R[len(R) - k * h:] @ w).reshape(k, h), axis=1) - 1.0
        sim = np.prod(1.0 + S @ w, axis=1) - 1.0
        out["horizon_vol_ratio"] = float(sim.std() / hist.std())
    return out


def compare(real: pd.DataFrame, scenario_sets: dict, weights=None) -> pd.DataFrame:
    """One row per generator."""
    return pd.DataFrame({name: quality_report(real, S, weights) for name, S in scenario_sets.items()}).T
