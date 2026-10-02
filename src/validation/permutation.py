"""Permutation tests for persistent labels (Generation 2).

A regime label is persistent: it stays on for weeks or months. Returns, and
especially volatility, are persistent too. Shuffling the label at random
destroys exactly the structure that makes two persistent series line up by
chance, so a naive permutation test is wildly oversized (it "finds" regime
effects in noise). Circularly shifting the label keeps its own persistence and
breaks only its alignment with the outcome, which is the null that matters:
"the regime has the same temporal texture, but is unrelated to returns".

Two statistics are computed for every shift, all shifts at once, as circular
cross-correlations through the FFT (five thousand shifts cost a few transforms):

* the in-regime mean minus the out-of-regime mean, and
* the same difference divided by its own Welch standard error (STUDENTISED).

The second is the primary. A regime defined by volatility is, by construction,
a set of days on which returns are more variable, so the plain difference of
means has a larger sampling error on the real flag than on a randomly shifted
one, and the shift distribution understates it: the test is anti-conservative
exactly where regimes matter most. Dividing by the local standard error removes
that mismatch (the size simulation in the tests shows the plain statistic
over-rejecting and the studentised one holding its level).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare(values, flag) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    f = np.asarray(flag, dtype=float)
    keep = np.isfinite(v) & np.isfinite(f)
    return v[keep], f[keep] > 0.5


def _all_shift_statistics(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, int, int]:
    """In-regime minus out-of-regime mean for every circular shift of the flag.

    ``stat[k]`` uses the flag shifted forward by ``k`` days; ``stat[0]`` is the
    observed statistic.
    """
    T = len(v)
    n_in = int(f.sum())
    n_out = T - n_in
    cross = np.fft.irfft(np.fft.rfft(v) * np.conj(np.fft.rfft(f.astype(float))), n=T)
    total = float(v.sum())
    return cross / n_in - (total - cross) / n_out, n_in, n_out


def _all_shift_t_statistics(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Welch t-statistic of the in/out difference for every circular shift of the flag."""
    T = len(v)
    n_in = int(f.sum())
    n_out = T - n_in
    flag_fft = np.conj(np.fft.rfft(f.astype(float)))
    c1 = np.fft.irfft(np.fft.rfft(v) * flag_fft, n=T)                 # sum of v over flagged days
    c2 = np.fft.irfft(np.fft.rfft(v * v) * flag_fft, n=T)             # sum of v^2 over flagged days
    s1, s2 = float(v.sum()), float((v * v).sum())
    mean_in, mean_out = c1 / n_in, (s1 - c1) / n_out
    var_in = np.maximum((c2 - n_in * mean_in ** 2) / (n_in - 1), 1e-30)
    var_out = np.maximum(((s2 - c2) - n_out * mean_out ** 2) / (n_out - 1), 1e-30)
    return (mean_in - mean_out) / np.sqrt(var_in / n_in + var_out / n_out)


def circular_shift_test(values, flag, n_shifts: int = 5000, min_shift: int = 63,
                        seed: int = 7) -> dict:
    """Two-sided circular-shift permutation test of a regime flag against an outcome.

    ``values`` is the outcome aligned with ``flag``. For a TRADABLE test align
    them with a lag: the flag at the close of t against the outcome on t+1.
    Shifts are drawn without replacement from ``min_shift .. T - min_shift`` so
    the shifted label is never nearly the original.

    ``p_value`` is from the studentised statistic (the primary);
    ``p_value_difference`` is from the plain difference of means, reported so
    the effect of studentising is visible.
    """
    v, f = _prepare(values, flag)
    T = len(v)
    n_in = int(f.sum())
    if n_in < 2 or T - n_in < 2 or T < 2 * min_shift + 10:
        return {}
    difference, n_in, n_out = _all_shift_statistics(v, f)
    t_stat = _all_shift_t_statistics(v, f)
    candidates = np.arange(min_shift, T - min_shift + 1)
    rng = np.random.default_rng(seed)
    shifts = candidates if len(candidates) <= n_shifts else rng.choice(candidates, size=n_shifts, replace=False)

    def p_of(stat: np.ndarray) -> float:
        extreme = int((np.abs(stat[shifts]) >= abs(stat[0]) - 1e-12).sum())
        return (1.0 + extreme) / (1.0 + len(shifts))

    return {
        "statistic": float(difference[0]),
        "t_statistic": float(t_stat[0]),
        "p_value": p_of(t_stat),
        "p_value_difference": p_of(difference),
        "n_in": n_in,
        "n_out": n_out,
        "mean_in": float(v[f].mean()),
        "mean_out": float(v[~f].mean()),
        "n_shifts": int(len(shifts)),
    }


def shuffle_test(values, flag, n_permutations: int = 2000, seed: int = 7) -> dict:
    """The naive i.i.d. permutation test, kept to show how oversized it is."""
    v, f = _prepare(values, flag)
    T = len(v)
    n_in = int(f.sum())
    if n_in < 2 or T - n_in < 2:
        return {}
    rng = np.random.default_rng(seed)

    def statistic(mask: np.ndarray) -> float:
        return float(v[mask].mean() - v[~mask].mean())

    observed = statistic(f)
    draws = np.array([statistic(rng.permutation(f)) for _ in range(n_permutations)])
    return {"statistic": observed,
            "p_value": (1.0 + int((np.abs(draws) >= abs(observed) - 1e-12).sum())) / (1.0 + n_permutations),
            "n_in": n_in, "n_out": T - n_in}


def lagged_pair(flag: pd.Series, outcome: pd.Series, lag: int = 1) -> tuple[pd.Series, pd.Series]:
    """Flag at the close of t aligned with the outcome on day t + ``lag``."""
    joined = pd.concat([flag.rename("flag"), outcome.shift(-lag).rename("outcome")], axis=1).dropna()
    return joined["flag"], joined["outcome"]
