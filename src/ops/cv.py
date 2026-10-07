"""Cross-validation for financial time series: purged k-fold, embargo, and combinatorial purged cross-validation (CPCV).

Ordinary k-fold leaks: a label that looks ``horizon`` days ahead overlaps the neighbouring fold, so training rows contain information about the test period. Purging removes training rows whose label
window overlaps the test fold; the embargo additionally skips ``embargo`` rows after the test fold (serial correlation of features). CPCV (Lopez de Prado 2018, ch. 12) splits the sample into ``N`` ordered
groups, tests on every combination of ``k`` groups and trains on the rest, so every group is tested in several different training contexts and the test paths can be recombined into
``k/N * C(N, k)`` complete backtest paths: a DISTRIBUTION of out-of-sample performance rather than the single path walk-forward gives.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable

import numpy as np
import pandas as pd


def _purge(train: np.ndarray, test_lo: int, test_hi: int, horizon: int, embargo: int) -> np.ndarray:
    """Drop training rows whose label window [i, i + horizon] reaches into the test block [test_lo, test_hi) and the ``embargo`` rows after it."""
    keep = (train + horizon < test_lo) | (train >= test_hi + embargo)
    return train[keep]


def purged_kfold(n: int, n_splits: int = 5, horizon: int = 21, embargo: int = 0):
    """Yield ``(train_idx, test_idx)`` for ``n_splits`` contiguous folds with purging and embargo."""
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    all_idx = np.arange(n)
    for lo, hi in zip(edges[:-1], edges[1:]):
        test = all_idx[lo:hi]
        yield _purge(np.concatenate([all_idx[:lo], all_idx[hi:]]), lo, hi, horizon, embargo), test


def cpcv_splits(n: int, n_groups: int = 6, n_test_groups: int = 2, horizon: int = 21, embargo: int = 0) -> list[dict]:
    """All ``C(N, k)`` splits: each has ``test_groups``, the ``test`` indices (the union of those groups) and the purged ``train`` indices (purging is applied around EVERY test block)."""
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [np.arange(edges[g], edges[g + 1]) for g in range(n_groups)]
    out = []
    for combo in combinations(range(n_groups), n_test_groups):
        train = np.setdiff1d(np.arange(n), np.concatenate([groups[g] for g in combo]))
        for g in combo:
            train = _purge(train, groups[g][0], groups[g][-1] + 1, horizon, embargo)
        out.append({"test_groups": combo, "test": np.concatenate([groups[g] for g in combo]), "train": train})
    return out


def n_cpcv_paths(n_groups: int, n_test_groups: int) -> int:
    """``k / N * C(N, k)`` complete backtest paths."""
    return n_test_groups * comb(n_groups, n_test_groups) // n_groups


def cpcv_paths(splits: list[dict], n_groups: int) -> list[list[tuple[int, int]]]:
    """Assemble the backtest paths: path ``p`` uses, for each group ``g``, the ``p``-th split that tests ``g``. Returns, per path, the ``(split_index, group)`` pair for each group in order."""
    owners: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for i, s in enumerate(splits):
        for g in s["test_groups"]:
            owners[g].append(i)
    n_paths = min(len(v) for v in owners.values())
    return [[(owners[g][p], g) for g in range(n_groups)] for p in range(n_paths)]


def cpcv_evaluate(X: pd.DataFrame, y: pd.Series, fit_predict: Callable, n_groups: int = 6, n_test_groups: int = 2, horizon: int = 21, embargo: int = 5, periods_per_year: int = 252) -> dict:
    """Run a trained model through CPCV. ``fit_predict(X_train, y_train, X_test)`` returns a position (or forecast) for each test row; the strategy return of a row is ``position x y`` with
    ``y`` the realised return. The out-of-sample positions of the different splits are recombined into backtest paths, and each path's annualised Sharpe ratio is reported with the
    distribution across paths."""
    n = len(X)
    splits = cpcv_splits(n, n_groups, n_test_groups, horizon, embargo)
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    preds = []
    for s in splits:
        pos = np.asarray(fit_predict(X.iloc[s["train"]], y.iloc[s["train"]], X.iloc[s["test"]]), dtype=float)
        preds.append(pd.Series(pos, index=s["test"]))
    paths = cpcv_paths(splits, n_groups)
    rets = y.to_numpy()
    sharpes, path_returns = [], []
    for path in paths:
        pos = np.full(n, np.nan)
        for split_i, g in path:
            idx = np.arange(edges[g], edges[g + 1])
            pos[idx] = preds[split_i].loc[idx].to_numpy()
        r = pos * rets
        path_returns.append(pd.Series(r, index=X.index))
        sd = np.nanstd(r, ddof=1)
        sharpes.append(float(np.nanmean(r) / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"))
    sharpes = np.array(sharpes)
    return {"sharpes": sharpes, "mean": float(np.nanmean(sharpes)), "std": float(np.nanstd(sharpes, ddof=1)) if len(sharpes) > 1 else float("nan"),
            "prob_negative": float(np.mean(sharpes < 0)), "n_paths": len(paths), "n_splits": len(splits), "path_returns": path_returns}
