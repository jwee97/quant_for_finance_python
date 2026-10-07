"""Resampling for dependent data: the block bootstraps, automatic block length, wild bootstrap and subsampling.

Financial returns are not i.i.d.: volatility clusters and many strategies have autocorrelated returns. An i.i.d. bootstrap destroys that
dependence and understates the sampling variability of anything that depends on it (Sharpe ratios, drawdowns, autocorrelations). The block bootstraps
keep short-range dependence by resampling whole blocks.

* ``iid_indices``: Efron's bootstrap.
* ``moving_block_indices`` (Kunsch 1989, Liu & Singh 1992): overlapping blocks of fixed length.
* ``circular_block_indices`` (Politis & Romano 1992): the same on a circle, which removes the end effect.
* ``stationary_indices`` (Politis & Romano 1994): blocks of geometric length, so the resampled series is itself stationary.
* ``optimal_block_length``: the Politis-White (2004) automatic rule with the Patton-Politis-White (2009) correction.
* ``wild_bootstrap_multipliers``: Rademacher, Mammen and Gaussian multipliers for regressions with heteroskedastic errors.
* ``subsample_statistics``: Politis-Romano subsampling, valid under weaker conditions than the bootstrap.

Every function takes a ``numpy.random.Generator`` or a seed, so every result is reproducible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def iid_indices(n: int, size: int | None = None, seed=0) -> np.ndarray:
    """``size`` indices drawn uniformly with replacement from ``range(n)``."""
    return _rng(seed).integers(0, n, size=n if size is None else size)


def moving_block_indices(n: int, block: int, seed=0) -> np.ndarray:
    """Indices for the moving-block bootstrap: ``ceil(n / block)`` blocks, each starting uniformly in ``0 .. n - block``, cut to length ``n``."""
    block = int(max(1, min(block, n)))
    rng = _rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def circular_block_indices(n: int, block: int, seed=0) -> np.ndarray:
    """Circular-block bootstrap: blocks may wrap around the end of the sample, so every observation is equally likely to be drawn."""
    block = int(max(1, min(block, n)))
    rng = _rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def stationary_indices(n: int, mean_block: float, seed=0) -> np.ndarray:
    """Stationary bootstrap: each position starts a new block with probability ``1 / mean_block`` and otherwise continues the previous one."""
    rng = _rng(seed)
    restart = rng.random(n) < 1.0 / max(mean_block, 1.0)
    restart[0] = True
    positions = np.flatnonzero(restart)
    block_id = np.cumsum(restart) - 1
    starts = rng.integers(0, n, size=len(positions))
    return (starts[block_id] + np.arange(n) - positions[block_id]) % n


_KINDS = {"iid": lambda n, b, s: iid_indices(n, seed=s), "moving": moving_block_indices, "circular": circular_block_indices,
          "stationary": stationary_indices}


def bootstrap_indices(n: int, kind: str = "stationary", block: float = 20.0, seed=0) -> np.ndarray:
    if kind not in _KINDS:
        raise ValueError(f"unknown bootstrap '{kind}'; choose from {sorted(_KINDS)}")
    return _KINDS[kind](n, block, seed)


def bootstrap_statistic(data, statistic: Callable, n_boot: int = 1000, kind: str = "stationary", block: float | None = None,
                        seed: int = 0) -> np.ndarray:
    """``statistic`` evaluated on ``n_boot`` resamples of the rows of ``data`` (a 1-d or 2-d array; rows are time).

    ``block`` defaults to the automatic Politis-White length of the first column.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if block is None:
        block = optimal_block_length(x if x.ndim == 1 else x[:, 0], kind="stationary" if kind == "stationary" else "circular")
    rng = np.random.default_rng(seed)
    return np.array([statistic(x[bootstrap_indices(n, kind, block, rng)]) for _ in range(n_boot)])


def optimal_block_length(x, kind: str = "stationary", max_lag: int | None = None, c: float = 2.0) -> float:
    """Politis & White (2004) automatic block length, with the Patton, Politis & White (2009) correction.

    ``kind='stationary'`` returns the expected block length for ``stationary_indices``; ``'circular'`` the fixed length for the block bootstraps.
    The estimate comes from the flat-top-kernel estimates of the long-run variance and of the sum of weighted autocovariances.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return float(max(1.0, n ** (1 / 3)))
    k_n = max(5, int(np.ceil(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n if max_lag is None else max_lag
    xc = x - x.mean()
    acov = np.array([xc[: n - k] @ xc[k:] / n for k in range(m_max + 1)])
    rho = acov / acov[0]
    # smallest m such that rho_k is insignificant for the next k_n lags (the flat-top truncation lag)
    band = c * np.sqrt(np.log10(n) / n)
    m = m_max
    for j in range(0, m_max - k_n + 1):
        if np.all(np.abs(rho[j + 1: j + 1 + k_n]) < band):
            m = max(j, 1)
            break
    M = min(2 * m, m_max)
    lam = np.clip(np.arange(-M, M + 1) / M, -1, 1)
    flat = np.where(np.abs(lam) <= 0.5, 1.0, 2.0 * (1.0 - np.abs(lam)))
    lags = np.abs(np.arange(-M, M + 1))
    g = float(np.sum(flat * acov[lags] * np.abs(lags)))
    sigma2 = float(np.sum(flat * acov[lags]))
    d = (4.0 / 3.0) * sigma2 ** 2 if kind == "circular" else 2.0 * sigma2 ** 2
    if d <= 0:
        return 1.0
    b = (2.0 * g ** 2 / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, min(3.0 * np.sqrt(n), n / 3.0)))


def wild_bootstrap_multipliers(n: int, kind: str = "rademacher", seed=0) -> np.ndarray:
    """Mean-zero, unit-variance multipliers: ``rademacher`` (+-1), ``mammen`` (matches skewness) or ``gaussian``."""
    rng = _rng(seed)
    if kind == "rademacher":
        return rng.choice([-1.0, 1.0], size=n)
    if kind == "mammen":
        p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        a, b = -(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2
        return np.where(rng.random(n) < p, a, b)
    if kind == "gaussian":
        return rng.standard_normal(n)
    raise ValueError("kind must be rademacher, mammen or gaussian")


def subsample_statistics(x, statistic: Callable, block: int, rate: float = 0.5) -> dict:
    """Politis-Romano subsampling: the statistic on every overlapping window of length ``block``.

    The root ``block**rate * (stat_b - stat_n)`` approximates the distribution of ``n**rate * (stat_n - theta)``, which gives confidence intervals
    when the bootstrap is not valid (heavy tails, non-smooth statistics).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    full = float(statistic(x))
    subs = np.array([statistic(x[i: i + block]) for i in range(n - block + 1)])
    roots = block ** rate * (subs - full)
    return {"statistic": full, "subsamples": subs, "roots": roots, "rate": rate, "n": n, "block": block}


def subsample_ci(result: dict, alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval from ``subsample_statistics``: ``stat - q_{1-a/2} / n^r, stat - q_{a/2} / n^r``."""
    roots, n, rate, stat = result["roots"], result["n"], result["rate"], result["statistic"]
    lo, hi = np.quantile(roots, [alpha / 2, 1 - alpha / 2])
    return float(stat - hi / n ** rate), float(stat - lo / n ** rate)
