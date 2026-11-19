"""Structural breaks and entropy (Lopez de Prado, chapters 17 and 18).

* ``sadf``: the supremum augmented Dickey-Fuller statistic (Phillips, Shi and Yu): for each end date, the largest ADF t-statistic over every start date that leaves at least ``min_window`` observations.
  A value above the critical value says the series has been *explosive* (a bubble) over some window ending then. Critical values depend on the sample and the window; ``sadf_critical`` simulates them under a random walk.
* ``chu_stinchcombe_white``: a supremum CUSUM test on the levels of a series, the largest standardised gain since any earlier date; it flags a change of level that a sum of differences would only show later.
* Entropy: how predictable a sequence of symbols is. ``plug_in_entropy`` (word frequencies), ``lempel_ziv`` (the share of new phrases when the sequence is parsed) and ``kontoyiannis_entropy`` (longest repeated
  match). They need the series *encoded* first (``encode_returns``): the information lost in the encoding is part of the answer.

None of these is a forecast by itself, and a statistic that is high in a sample is not evidence that it will be high next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _adf_t(y: np.ndarray, lags: int, trend: str) -> float:
    dy = np.diff(y)
    k = lags
    n = len(dy) - k
    if n < 8:
        return float("nan")
    yl = y[k:-1] if k else y[:-1]
    cols = [yl[:n]]
    for i in range(1, k + 1):
        cols.append(dy[k - i:len(dy) - i][:n])
    cols = np.column_stack(cols)
    if trend == "c":
        cols = np.column_stack([np.ones(n), cols])
    elif trend == "ct":
        cols = np.column_stack([np.ones(n), np.arange(n, dtype=float), cols])
    target = dy[k:][:n]
    gi = 1 if trend == "c" else 2 if trend == "ct" else 0
    try:
        xtx = cols.T @ cols
        beta = np.linalg.solve(xtx, cols.T @ target)
        resid = target - cols @ beta
        s2 = float(resid @ resid) / (n - cols.shape[1])
        with np.errstate(invalid="ignore"):
            se = np.sqrt(s2 * np.linalg.inv(xtx)[gi, gi])
        return float(beta[gi] / se) if se > 0 else float("nan")
    except np.linalg.LinAlgError:
        return float("nan")


def sadf(y: pd.Series, min_window: int = 40, lags: int = 1, trend: str = "c", max_starts: int = 30) -> pd.Series:
    """The SADF statistic at each date from ``min_window`` on. ``y`` is usually a log price. ``max_starts`` limits how many start dates are tried per end date (evenly spread), which bounds the cost at the price of
    a slightly smaller supremum. ``trend`` is "n", "c" (constant) or "ct"."""
    if trend not in ("n", "c", "ct"):
        raise ValueError("trend must be n, c or ct")
    v = y.astype(float).to_numpy()
    out = np.full(len(v), np.nan)
    for t in range(min_window, len(v)):
        last = t - min_window
        starts = np.unique(np.linspace(0, last, min(max_starts, last + 1)).astype(int))
        stats = [_adf_t(v[s:t + 1], lags, trend) for s in starts]
        stats = [s for s in stats if np.isfinite(s)]
        out[t] = max(stats) if stats else np.nan
    return pd.Series(out, index=y.index, name="sadf")


def sadf_critical(n: int, min_window: int = 40, lags: int = 1, trend: str = "c", reps: int = 100, quantile: float = 0.95, seed: int = 0, max_starts: int = 15) -> float:
    """The ``quantile`` of the largest SADF value over a random walk of ``n`` observations: what a series with no explosive stretch reaches. Simulated, so it varies with ``seed`` and ``reps``."""
    rng = np.random.default_rng(seed)
    peaks = []
    for _ in range(reps):
        walk = pd.Series(np.cumsum(rng.normal(0, 1, n)))
        peaks.append(np.nanmax(sadf(walk, min_window, lags, trend, max_starts).to_numpy()))
    return float(np.quantile(peaks, quantile))


def chu_stinchcombe_white(y: pd.Series, min_window: int = 20, b_alpha: float = 4.6) -> pd.DataFrame:
    """Supremum CUSUM on levels. At each date ``t`` and each earlier date ``n``: ``S = |y_t - y_n| / (sigma_t sqrt(t - n))``, with ``sigma_t`` the standard deviation of the one-step changes up to ``t``. Returns
    ``stat`` (the largest S) and ``critical`` (``sqrt(b_alpha + log(t - n))`` at the maximising ``n``; ``b_alpha = 4.6`` for 5%, from the paper): ``break_`` is where ``stat`` exceeds it."""
    v = y.astype(float).to_numpy()
    dy = np.diff(v, prepend=np.nan)
    stat, crit = np.full(len(v), np.nan), np.full(len(v), np.nan)
    for t in range(min_window, len(v)):
        sigma = np.nanstd(dy[1:t + 1], ddof=1)
        if not sigma > 0:
            continue
        n = np.arange(0, t - 1)
        s = np.abs(v[t] - v[n]) / (sigma * np.sqrt(t - n))
        j = int(np.argmax(s))
        stat[t] = s[j]
        crit[t] = np.sqrt(b_alpha + np.log(t - n[j]))
    out = pd.DataFrame({"stat": stat, "critical": crit}, index=y.index)
    out["break_"] = out["stat"] > out["critical"]
    return out


def encode_returns(returns: pd.Series, n_bins: int = 4, kind: str = "quantile") -> str:
    """Turn returns into a string of symbols: ``"sign"`` gives ``u``/``d``; ``"quantile"`` cuts them into ``n_bins`` equal-count bins (letters a, b, ...). The quantile cut uses the whole sample, so for a
    feature used in a model it must be fitted on the training data only."""
    r = returns.dropna().to_numpy(dtype=float)
    if kind == "sign":
        return "".join("u" if x > 0 else "d" for x in r)
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1)[1:-1])
    return "".join(chr(ord("a") + int(np.searchsorted(edges, x, side="right"))) for x in r)


def plug_in_entropy(msg: str, word: int = 1) -> float:
    """Shannon entropy per symbol, in bits, of the empirical distribution of words of length ``word`` (non-overlapping starts every symbol). Biased low for short messages and long words."""
    if len(msg) < word or word < 1:
        return float("nan")
    words = [msg[i:i + word] for i in range(len(msg) - word + 1)]
    _, counts = np.unique(words, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum() / word)


def lempel_ziv(msg: str) -> float:
    """The Lempel-Ziv (1976) complexity: the number of distinct phrases when the message is parsed left to right, divided by ``n / log2 n``. About 1 for a random sequence, lower for a repetitive (predictable) one."""
    n = len(msg)
    if n < 2:
        return float("nan")
    seen, i, count = set(), 0, 0
    while i < n:
        j = i + 1
        while j <= n and msg[i:j] in seen:
            j += 1
        seen.add(msg[i:j])
        count += 1
        i = j
    return float(count * np.log2(n) / n)


def kontoyiannis_entropy(msg: str, window: int | None = None) -> float:
    """Kontoyiannis' estimator: from each position, the length of the longest string that also occurred in the preceding ``window`` symbols (default: the whole past); the entropy rate is ``log2(window) / mean(length)``-like, in bits per symbol.
    Lower means more repeated structure."""
    n = len(msg)
    if n < 8:
        return float("nan")
    w = window or n // 2
    sums, cnt = 0.0, 0
    for i in range(w, n):
        past = msg[i - w:i]
        length = 1
        while i + length <= n and msg[i:i + length] in past:
            length += 1
        sums += (length) / np.log2(w)
        cnt += 1
    return float(1.0 / (sums / cnt)) if sums > 0 else float("nan")
