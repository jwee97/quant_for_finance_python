"""Purged and combinatorial purged cross-validation for event labels (Lopez de Prado, chapters 7 and 12).

``src/ops/cv.py`` purges with one fixed horizon in rows. Event labels end at different times (a barrier is touched early or late), so here each label is purged by its *own* span ``[pos0, pos1]``: a training
label is dropped if it overlaps a test label's span, and the ``embargo`` rows after the test block are dropped as well. The splits come in the same shape as ``ops.cv.cpcv_splits`` so ``ops.cv.cpcv_paths`` assembles the paths.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _blocks(n: int, k: int) -> list[np.ndarray]:
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[g], edges[g + 1]) for g in range(k)]


def purge_train(labels: pd.DataFrame, train: np.ndarray, test_blocks: list[np.ndarray], embargo: int = 0) -> np.ndarray:
    """Keep the training labels that neither overlap, nor sit within ``embargo`` rows after, any test block. A block's span runs from the first ``pos0`` to the last ``pos1`` among its labels."""
    p0, p1 = labels["pos0"].to_numpy(), labels["pos1"].to_numpy()
    keep = np.ones(len(train), dtype=bool)
    for b in test_blocks:
        if len(b) == 0:
            continue
        lo, hi = p0[b].min(), p1[b].max()
        keep &= (p1[train] < lo) | (p0[train] > hi + embargo)
    return train[keep]


def purged_folds(labels: pd.DataFrame, n_splits: int = 5, embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(train, test)`` positions for ``n_splits`` contiguous folds of the labels (in time order), with purging and embargo."""
    n = len(labels)
    out = []
    for blk in _blocks(n, n_splits):
        train = np.setdiff1d(np.arange(n), blk)
        out.append((purge_train(labels, train, [blk], embargo), blk))
    return out


def cpcv_event_splits(labels: pd.DataFrame, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 0) -> list[dict]:
    """All ``C(N, k)`` combinations of ``k`` test groups out of ``N``: each has ``test_groups``, ``test`` and the purged ``train`` positions."""
    n = len(labels)
    groups = _blocks(n, n_groups)
    out = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_blocks = [groups[g] for g in combo]
        train = np.setdiff1d(np.arange(n), np.concatenate(test_blocks))
        out.append({"test_groups": combo, "test": np.concatenate(test_blocks), "train": purge_train(labels, train, test_blocks, embargo)})
    return out


def leakage_check(labels: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> bool:
    """True if no training label's span overlaps a test label's span (the property purging must give)."""
    p0, p1 = labels["pos0"].to_numpy(), labels["pos1"].to_numpy()
    for t in test:
        if np.any((p0[train] <= p1[t]) & (p1[train] >= p0[t])):
            return False
    return True
