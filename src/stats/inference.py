"""Statistical inference for research results: bootstrap intervals, multiple-testing corrections and Sharpe-ratio inference.

Backtests are searched, not planned: the best of many trials is biased upward. This module collects the corrections.

Family-wise error (the chance of ANY false discovery)
    ``p_adjust(method='bonferroni' | 'holm' | 'hochberg')`` and the bootstrap step-down of Romano & Wolf (2005), ``romano_wolf``, which uses the
    dependence between the strategies and is the most powerful of the four.
False-discovery rate (the share of discoveries that are false)
    ``p_adjust(method='bh' | 'by')`` (Benjamini-Hochberg, Benjamini-Yekutieli) and Storey's q-values, ``storey_qvalues``.
Sharpe-ratio inference
    ``sharpe_standard_error`` (iid, Mertens 2002 non-normal, Lo 2002 autocorrelation), ``probabilistic_sharpe_ratio``, ``min_track_record_length``,
    ``deflated_sharpe_ratio`` with the variance of the trial Sharpes estimated from the trials themselves (Bailey & Lopez de Prado 2014) and
    ``haircut_sharpe`` (Harvey & Liu 2015).
Bootstrap confidence intervals
    ``bootstrap_ci``: percentile, basic, BCa (bias-corrected and accelerated, Efron 1987) and studentised intervals, with iid or block resampling.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..probability.resampling import bootstrap_indices, optimal_block_length

EULER = 0.5772156649015329


# ------------------------------------------------------------------------------------------------------------ multiple testing
def p_adjust(p, method: str = "holm") -> np.ndarray:
    """Adjusted p-values, as in R's ``p.adjust``. ``bonferroni``, ``holm``, ``hochberg``, ``bh`` (= fdr_bh), ``by``. Reject where adjusted <= level."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    if method == "bonferroni":
        adj = np.minimum(ranked * m, 1.0)
    elif method == "holm":
        adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ranked), 1.0)
    elif method == "hochberg":
        adj = np.minimum(np.minimum.accumulate(((m - np.arange(m)) * ranked)[::-1])[::-1], 1.0)
    elif method in ("bh", "fdr_bh", "by", "fdr_by"):
        scale = 1.0 if method in ("bh", "fdr_bh") else float(np.sum(1.0 / np.arange(1, m + 1)))
        adj = np.minimum(np.minimum.accumulate((scale * m / np.arange(1, m + 1) * ranked)[::-1])[::-1], 1.0)
    else:
        raise ValueError("method must be bonferroni, holm, hochberg, bh or by")
    out = np.empty(m)
    out[order] = adj
    return out


def storey_qvalues(p, lam: float | None = None) -> dict:
    """Storey (2002) q-values: ``pi0`` estimated as the share of p-values above ``lam``, scaled, then made monotone. ``lam`` defaults to a smooth
    estimate over 0.05 to 0.95 (the cubic-spline rule is replaced by the median of the grid, which is stable for the few hundred tests of a research table)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    grid = np.arange(0.05, 0.96, 0.05)
    pi_grid = np.array([(p > g).mean() / (1.0 - g) for g in grid])
    pi0 = float(np.clip(np.median(pi_grid[-6:]) if lam is None else (p > lam).mean() / (1.0 - lam), 1.0 / m, 1.0))
    order = np.argsort(p)
    q = pi0 * m * p[order] / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return {"qvalues": out, "pi0": pi0}


def romano_wolf(returns: pd.DataFrame | np.ndarray, n_boot: int = 2000, block: float | None = None, seed: int = 0, kind: str = "stationary") -> pd.DataFrame:
    """Romano & Wolf (2005) step-down: family-wise-error-controlling p-values for ``H0_j: mean_j <= 0`` across the columns of ``returns``.

    The null distribution is the bootstrap distribution of the studentised statistics centred at the sample means, so the dependence among the strategies
    is used. Returns a table of the t-statistic, the unadjusted bootstrap p-value and the Romano-Wolf adjusted p-value.
    """
    cols = list(returns.columns) if isinstance(returns, pd.DataFrame) else [f"s{i}" for i in range(np.asarray(returns).shape[1])]
    R = np.asarray(returns, dtype=float)
    n, m = R.shape
    block = optimal_block_length(R.mean(axis=1), kind="stationary") if block is None else block
    mean, sd = R.mean(axis=0), R.std(axis=0, ddof=1) / np.sqrt(n)
    t = mean / sd
    rng = np.random.default_rng(seed)
    tb = np.empty((n_boot, m))
    for b in range(n_boot):
        s = R[bootstrap_indices(n, kind, block, rng)]
        tb[b] = (s.mean(axis=0) - mean) / (s.std(axis=0, ddof=1) / np.sqrt(n))
    raw = np.array([(tb[:, j] >= t[j]).mean() for j in range(m)])
    order = np.argsort(-t)
    adj = np.zeros(m)
    for rank, j in enumerate(order):
        remaining = order[rank:]
        adj[j] = (tb[:, remaining].max(axis=1) >= t[j]).mean()
    adj_sorted = np.maximum.accumulate(adj[order])
    adj[order] = adj_sorted
    return pd.DataFrame({"t": t, "p_raw": raw, "p_romano_wolf": np.minimum(adj, 1.0)}, index=cols)


# ------------------------------------------------------------------------------------------------------------------- Sharpe
def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    mu, sd = r.mean(), r.std(ddof=1)
    return mu, sd, float(sps.skew(r)), float(sps.kurtosis(r, fisher=False))


def sharpe_standard_error(returns, method: str = "mertens", lags: int | None = None) -> float:
    """Standard error of the PER-PERIOD Sharpe ratio. ``iid``: ``sqrt((1 + SR^2/2)/n)``; ``mertens``: adds skewness and kurtosis,
    ``sqrt((1 - g3 SR + (g4 - 1)/4 SR^2)/(n - 1))``; ``lo``: Lo (2002) autocorrelation adjustment, ``sqrt((1 + 2 sum_k (1 - k/(q+1)) rho_k) (1 + SR^2/2)/n)``."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    mu, sd, g3, g4 = _moments(r)
    sr = mu / sd
    if method == "iid":
        return float(np.sqrt((1.0 + 0.5 * sr ** 2) / n))
    if method == "mertens":
        return float(np.sqrt((1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (n - 1)))
    if method == "lo":
        q = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lags is None else lags
        c = r - mu
        rho = np.array([c[k:] @ c[:-k] / (c @ c) for k in range(1, q + 1)])
        factor = 1.0 + 2.0 * float(np.sum((1.0 - np.arange(1, q + 1) / (q + 1.0)) * rho))
        return float(np.sqrt(max(factor, 1e-12) * (1.0 + 0.5 * sr ** 2) / n))
    raise ValueError("method must be iid, mertens or lo")


def probabilistic_sharpe_ratio(returns, benchmark_sharpe: float = 0.0, annualisation: float = 1.0) -> float:
    """P(true Sharpe > benchmark) given the estimation error of the sample Sharpe including skew and kurtosis (Bailey & Lopez de Prado 2012).
    ``benchmark_sharpe`` is annualised if ``annualisation`` (periods per year) is given, per-period otherwise."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    mu, sd, *_ = _moments(r)
    sr = mu / sd
    bench = benchmark_sharpe / np.sqrt(annualisation)
    return float(sps.norm.cdf((sr - bench) / sharpe_standard_error(r, "mertens")))


def min_track_record_length(returns, benchmark_sharpe: float = 0.0, confidence: float = 0.95, annualisation: float = 1.0) -> float:
    """Number of observations needed for the sample Sharpe to be above ``benchmark_sharpe`` at ``confidence`` (Bailey & Lopez de Prado 2012)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    mu, sd, g3, g4 = _moments(r)
    sr = mu / sd
    bench = benchmark_sharpe / np.sqrt(annualisation)
    if sr <= bench:
        return float("inf")
    z = sps.norm.ppf(confidence)
    return float(1.0 + (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) * (z / (sr - bench)) ** 2)


def expected_max_sharpe(n_trials: int, trial_variance: float = 1.0) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates that all have true Sharpe 0 and variance ``trial_variance`` (per-period units):
    ``sqrt(V) ((1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e)))`` (Bailey & Lopez de Prado 2014)."""
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(trial_variance) * ((1.0 - EULER) * sps.norm.ppf(1.0 - 1.0 / n_trials) + EULER * sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))))


def deflated_sharpe_ratio(returns, trial_sharpes=None, n_trials: int | None = None) -> dict:
    """Deflated Sharpe ratio of the selected strategy's ``returns``.

    The benchmark is the expected maximum Sharpe among ``n_trials`` strategies of no skill, whose spread is the empirical variance of ``trial_sharpes``
    (per-period Sharpe ratios of ALL strategies tried) when given, otherwise the variance of the Sharpe estimator itself, ``1/n``. Returns the benchmark and
    the probability that the selected strategy's true Sharpe exceeds it.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    mu, sd, *_ = _moments(r)
    sr = mu / sd
    if trial_sharpes is not None:
        ts = np.asarray(trial_sharpes, dtype=float)
        n_trials = len(ts) if n_trials is None else n_trials
        var = float(np.var(ts, ddof=1))
    else:
        if n_trials is None:
            raise ValueError("give trial_sharpes or n_trials")
        var = 1.0 / len(r)
    bench = expected_max_sharpe(n_trials, var)
    prob = float(sps.norm.cdf((sr - bench) / sharpe_standard_error(r, "mertens")))
    return {"sharpe": float(sr), "benchmark": bench, "probability": prob, "n_trials": int(n_trials), "trial_variance": var}


def haircut_sharpe(sharpe: float, n_obs: int, n_tests: int, method: str = "holm", annualisation: float = 12.0, rho: float = 0.0) -> dict:
    """Harvey & Liu (2015) Sharpe-ratio haircut: the Sharpe that would give the SAME p-value after the multiple-testing correction.

    The observed annualised ``sharpe`` over ``n_obs`` periods gives a t-statistic ``SR sqrt(n / annualisation)``; its p-value is adjusted for
    ``n_tests`` (for the best strategy the Bonferroni, Holm and BH corrections coincide) and converted back. ``rho`` is the average correlation of the tests; it shrinks the effective number
    of independent tests to ``1 + (n_tests - 1)(1 - rho)``.
    """
    t = sharpe * np.sqrt(n_obs / annualisation)
    p = 2.0 * sps.norm.sf(abs(t))
    m = max(1.0, 1.0 + (n_tests - 1) * (1.0 - rho))
    if method not in ("bonferroni", "holm", "bh"):
        raise ValueError("method must be bonferroni, holm or bh")
    # for the best strategy of the family all three corrections reduce to the same single-test equivalent: the p-value times the effective number of tests
    # (Holm's first step, the Bonferroni bound, and BH's rank-1 threshold coincide); ``method`` is kept for the less extreme strategies of a ranked table
    adj = min(p * m, 1.0)
    t_adj = sps.norm.isf(adj / 2.0) if adj < 1 else 0.0
    sr_adj = t_adj * np.sqrt(annualisation / n_obs)
    return {"t_observed": float(t), "p_observed": float(p), "p_adjusted": float(adj), "sharpe_adjusted": float(sr_adj),
            "haircut": float(1.0 - sr_adj / sharpe) if sharpe else float("nan")}


# ------------------------------------------------------------------------------------------------------------------ bootstrap CI
def bootstrap_ci(data, statistic: Callable, alpha: float = 0.05, n_boot: int = 2000, method: str = "bca", resample: str = "iid", block: float | None = None,
                 seed: int = 0) -> dict:
    """Bootstrap confidence interval of ``statistic(data)``. ``method``: ``percentile``, ``basic``, ``bca`` or ``studentized`` (``statistic`` must then
    return ``(estimate, standard_error)``). ``resample``: ``iid``, ``moving``, ``circular`` or ``stationary`` (rows are time; ``block`` defaults to the
    Politis-White length). BCa uses a jackknife for the acceleration, so with block resampling it is approximate."""
    x = np.asarray(data, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    if resample != "iid" and block is None:
        block = optimal_block_length(x if x.ndim == 1 else x[:, 0], kind="stationary" if resample == "stationary" else "circular")
    draws = [x[bootstrap_indices(n, resample, block or 1, rng)] for _ in range(n_boot)]
    if method == "studentized":
        est, se = statistic(x)
        pairs = np.array([statistic(d) for d in draws])
        tstar = (pairs[:, 0] - est) / pairs[:, 1]
        lo, hi = np.quantile(tstar, [alpha / 2, 1 - alpha / 2])
        return {"estimate": float(est), "lower": float(est - hi * se), "upper": float(est - lo * se), "method": method}
    est = float(statistic(x))
    boot = np.array([statistic(d) for d in draws])
    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    elif method == "basic":
        q_lo, q_hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        lo, hi = 2 * est - q_hi, 2 * est - q_lo
    elif method == "bca":
        z0 = sps.norm.ppf(np.clip((boot < est).mean(), 1e-6, 1 - 1e-6))
        jack = np.array([statistic(np.delete(x, i, axis=0)) for i in range(n)]) if n <= 2000 else np.array([statistic(np.delete(x, i, axis=0)) for i in np.linspace(0, n - 1, 2000).astype(int)])
        d = jack.mean() - jack
        a = float((d ** 3).sum() / (6.0 * ((d ** 2).sum()) ** 1.5)) if (d ** 2).sum() > 0 else 0.0
        zs = sps.norm.ppf([alpha / 2, 1 - alpha / 2])
        adj = sps.norm.cdf(z0 + (z0 + zs) / (1.0 - a * (z0 + zs)))
        lo, hi = np.quantile(boot, adj)
    else:
        raise ValueError("method must be percentile, basic, bca or studentized")
    return {"estimate": est, "lower": float(lo), "upper": float(hi), "method": method, "bootstrap_se": float(boot.std(ddof=1))}
