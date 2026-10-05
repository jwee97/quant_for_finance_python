"""Tests that account for the size of a search (Generation 4, Priority 19).

Run enough rules and the best of them looks good by construction. Three tools
for asking what is left after that:

* **White's Reality Check**: bootstrap the maximum of the candidates' mean
  excess returns under the null that none has a positive expectation.
* **Hansen's Superior Predictive Ability (SPA) test**: the same, studentised
  and with a recentring that stops clearly bad candidates from inflating the
  null distribution (the Reality Check's known weakness: add enough
  terrible rules and nothing can ever be significant).
* **Probability of Backtest Overfitting** (Bailey, Borwein, Lopez de Prado and
  Zhu): across combinatorially symmetric train/test splits, how often does the
  in-sample winner land in the bottom half out of sample?

Inputs are matrices of daily excess returns, one column per candidate.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .robustness import stationary_bootstrap_indices


def bootstrap_means(matrix: np.ndarray, n_samples: int = 2000, mean_block: float = 21.0, seed: int = 7,
                    batch: int = 200) -> np.ndarray:
    """Stationary-bootstrap means of every column: an ``(n_samples, K)`` array.

    The same resampled day indices are used for all columns in a draw, which preserves the
    cross-correlation between candidates. Implemented as ``counts @ matrix / n`` so a draw costs one
    matrix-vector product rather than a gather of the whole matrix.
    """
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, k))
    for start in range(0, n_samples, batch):
        size = min(batch, n_samples - start)
        counts = np.zeros((size, n))
        for i in range(size):
            counts[i] = np.bincount(stationary_bootstrap_indices(n, mean_block, rng), minlength=n)
        out[start:start + size] = counts @ x / n
    return out


def white_reality_check(matrix: np.ndarray, n_samples: int = 2000, mean_block: float = 21.0, seed: int = 7,
                        boot: np.ndarray | None = None) -> dict:
    """p-value of H0: no candidate has a positive expected excess return (White, 2000)."""
    x = np.asarray(matrix, dtype=float)
    n = x.shape[0]
    mean = x.mean(axis=0)
    boot = bootstrap_means(x, n_samples, mean_block, seed) if boot is None else boot
    statistic = np.sqrt(n) * mean.max()
    null = np.sqrt(n) * (boot - mean).max(axis=1)
    return {"statistic": float(statistic), "p_value": float((null >= statistic).mean()), "best_index": int(mean.argmax()),
            "best_mean": float(mean.max())}


def hansen_spa(matrix: np.ndarray, n_samples: int = 2000, mean_block: float = 21.0, seed: int = 7,
               boot: np.ndarray | None = None) -> dict:
    """Hansen's SPA test (2005): lower, consistent and upper p-values.

    ``p_consistent`` is the one to report. The upper p-value equals the studentised Reality Check;
    the consistent one recentres only candidates that are clearly worse than the benchmark
    (``sqrt(n) * mean / sigma <= -sqrt(2 log log n)``).
    """
    x = np.asarray(matrix, dtype=float)
    n = x.shape[0]
    mean = x.mean(axis=0)
    boot = bootstrap_means(x, n_samples, mean_block, seed) if boot is None else boot
    sigma = np.sqrt(n) * boot.std(axis=0, ddof=1)
    sigma = np.where(sigma > 0, sigma, np.nan)
    t_k = np.sqrt(n) * mean / sigma
    statistic = max(0.0, float(np.nanmax(t_k)))
    threshold = np.sqrt(2.0 * np.log(np.log(n)))
    centred = np.sqrt(n) * (boot - mean) / sigma
    out = {"statistic": statistic, "best_index": int(np.nanargmax(t_k)), "best_t": float(np.nanmax(t_k))}
    # Hansen's three recentrings: min(mean, 0) (lower p), thresholded (consistent p), zero (upper p).
    for label, mu in (("lower", np.minimum(mean, 0.0)), ("consistent", np.where(t_k <= -threshold, mean, 0.0)),
                      ("upper", np.zeros_like(mean))):
        null = np.maximum(0.0, np.nanmax(centred + np.sqrt(n) * mu / sigma, axis=1))
        out[f"p_{label}"] = float((null >= statistic).mean())
    return out


def pbo_cscv(matrix: np.ndarray, blocks: int = 16, annualise: float = 252.0) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    The sample is cut into ``blocks`` contiguous blocks. For each of the ``C(blocks, blocks/2)`` ways of
    choosing half of them as in-sample, the in-sample Sharpe-best candidate is found and its out-of-sample
    relative rank ``w`` recorded; ``lambda = ln(w / (1 - w))``. PBO is the share of splits with
    ``lambda <= 0``, that is the in-sample winner at or below the out-of-sample median.
    """
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    if blocks % 2:
        raise ValueError("blocks must be even")
    size = n // blocks
    x = x[n - size * blocks:]                                     # drop the oldest remainder
    s1 = np.stack([x[i * size:(i + 1) * size].sum(axis=0) for i in range(blocks)])
    s2 = np.stack([(x[i * size:(i + 1) * size] ** 2).sum(axis=0) for i in range(blocks)])
    half = blocks // 2
    splits = list(combinations(range(blocks), half))
    flag = np.zeros((len(splits), blocks))
    for r, chosen in enumerate(splits):
        flag[r, list(chosen)] = 1.0
    m_is = half * size
    def sharpe(sum1, sum2, m):
        mean = sum1 / m
        var = (sum2 - m * mean ** 2) / (m - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, mean / np.sqrt(var) * np.sqrt(annualise), -np.inf)
    is_sharpe = sharpe(flag @ s1, flag @ s2, m_is)
    oos_sharpe = sharpe((1 - flag) @ s1, (1 - flag) @ s2, m_is)
    best = is_sharpe.argmax(axis=1)
    rows = np.arange(len(splits))
    best_oos = oos_sharpe[rows, best]
    rank = (oos_sharpe < best_oos[:, None]).sum(axis=1) + 1       # 1 = worst
    omega = rank / (k + 1.0)
    logit = np.log(omega / (1.0 - omega))
    best_is = is_sharpe[rows, best]
    slope = np.polyfit(best_is, best_oos, 1)[0] if np.ptp(best_is) > 0 else float("nan")
    return {"pbo": float((logit <= 0).mean()), "n_splits": len(splits), "logit": logit,
            "is_best_sharpe": best_is, "oos_of_best_sharpe": best_oos,
            "mean_oos_of_best": float(best_oos.mean()), "mean_is_of_best": float(best_is.mean()),
            "degradation_slope": float(slope), "share_oos_negative": float((best_oos < 0).mean())}
