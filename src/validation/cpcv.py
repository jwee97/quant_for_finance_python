"""Combinatorial purged cross-validation (López de Prado 2018): many backtest paths from one history, with purging and an embargo.

Cut the sample into ``n_groups`` consecutive groups and choose ``n_test_groups`` of them as the test set: there are ``C(n_groups, n_test_groups)`` such splits. Training uses the remaining groups, *purged* of any
observation whose forecast window (``horizon`` days) overlaps a test group and *embargoed* for ``embargo`` observations after each test group. Each group appears in ``C(n_groups - 1, n_test_groups - 1)`` of the
test sets, so out-of-sample predictions for it can be assembled into that many different complete paths through the history: ``phi = C(n_groups, n_test_groups) * n_test_groups / n_groups`` paths. The spread of
a statistic over those paths shows how much of a backtest's result is the particular order of events, which a single walk-forward path cannot say, and is the raw material for the probability of backtest overfitting
(:mod:`src.validation.multiple_testing`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


@dataclass
class CPCVSplit:
    train: np.ndarray
    test: np.ndarray
    test_groups: tuple


def group_bounds(n: int, n_groups: int) -> np.ndarray:
    return np.linspace(0, n, n_groups + 1).astype(int)


def cpcv_splits(n: int, n_groups: int = 6, n_test_groups: int = 2, horizon: int = 1, embargo: int = 0) -> list[CPCVSplit]:
    """All ``C(n_groups, n_test_groups)`` splits of ``range(n)``. ``horizon`` is the number of observations a label looks ahead: training observations within ``horizon - 1`` before a test group are purged."""
    if not 1 <= n_test_groups < n_groups or n_groups > n or horizon < 1 or embargo < 0:
        raise ValueError("1 <= n_test_groups < n_groups <= n, horizon >= 1, embargo >= 0")
    bounds = group_bounds(n, n_groups)
    out = []
    for chosen in combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(n, dtype=bool)
        drop = np.zeros(n, dtype=bool)
        for g in chosen:
            lo, hi = bounds[g], bounds[g + 1]
            test_mask[lo:hi] = True
            drop[max(lo - (horizon - 1), 0):min(hi + embargo, n)] = True
        out.append(CPCVSplit(np.flatnonzero(~drop), np.flatnonzero(test_mask), chosen))
    return out


def n_paths(n_groups: int, n_test_groups: int) -> int:
    return comb(n_groups - 1, n_test_groups - 1)


def assemble_paths(n: int, splits: list[CPCVSplit], predictions: list[np.ndarray], n_groups: int, n_test_groups: int) -> np.ndarray:
    """Out-of-sample paths. ``predictions[i]`` holds the predictions for ``splits[i].test`` (in that order, shape ``(len(test), ...)``). Every group is predicted in several splits; path ``p`` uses, for each group,
    the ``p``-th split that contains it, so each path covers the whole history once. Returns ``(n_paths, n, ...)``."""
    bounds = group_bounds(n, n_groups)
    by_group = {g: [] for g in range(n_groups)}
    for i, sp in enumerate(splits):
        for g in sp.test_groups:
            by_group[g].append(i)
    k = n_paths(n_groups, n_test_groups)
    first = np.asarray(predictions[0])
    paths = np.full((k, n) + first.shape[1:], np.nan)
    for g in range(n_groups):
        lo, hi = bounds[g], bounds[g + 1]
        for p, i in enumerate(by_group[g]):
            test = splits[i].test
            where = np.flatnonzero((test >= lo) & (test < hi))
            paths[p, lo:hi] = np.asarray(predictions[i])[where]
    return paths
