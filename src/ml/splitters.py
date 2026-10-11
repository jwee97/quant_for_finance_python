"""Time-series splitters: train/test partitions of the DATE axis in which no training label overlaps the information of a test row.

Everything here works on integer positions into a sorted array of unique dates; a dataset's rows are mapped to dates afterwards, so every asset on a date is on the same side of every split.

``purge`` is the number of days a label looks past its feature date (``lag + horizon``, see ``targets.label_end``). A training row at position ``i`` has a label that is complete at position
``i + purge``; it may be used for a prediction made at position ``j`` only if ``i + purge <= j``. ``embargo`` keeps out a further number of days (serial correlation in the features).

* ``walk_forward_splits``  refit every ``step`` days on everything before (expanding) or the last ``window`` days (rolling), test on the next ``step`` days: how a model is actually run
* ``purged_kfold_splits``  contiguous folds; training rows whose label reaches into the test fold, and the ``embargo`` days after it, are dropped
* ``embargoed_split``      one train block, an embargo, one test block
* ``cpcv_splits``          combinatorial purged cross-validation: every combination of ``k`` of ``N`` groups is a test set, so the out-of-sample paths can be recombined into a distribution of results
* ``segment_plan``         the development timeline: warm-up train, validation, test and forward segments, the last two reserved
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ops.cv import cpcv_splits as _cpcv, purged_kfold as _purged_kfold


@dataclass(frozen=True)
class Split:
    train: np.ndarray                 # positions into the date axis
    test: np.ndarray
    kind: str
    fold: int = 0
    groups: tuple = ()                # for CPCV: the groups tested

    def describe(self, dates: pd.DatetimeIndex | None = None) -> dict:
        out = {"fold": self.fold, "kind": self.kind, "n_train": int(len(self.train)), "n_test": int(len(self.test))}
        if dates is not None and len(self.train) and len(self.test):
            out.update({"train_start": str(dates[self.train[0]].date()), "train_end": str(dates[self.train[-1]].date()), "test_start": str(dates[self.test[0]].date()), "test_end": str(dates[self.test[-1]].date())})
        return out


def walk_forward_splits(n: int, first_test: int, step: int, purge: int, embargo: int = 0, window: str = "expanding", rolling_days: int = 1260, last_test: int | None = None) -> list[Split]:
    """Test blocks of ``step`` positions from ``first_test`` to ``last_test`` (default the end). Each trains on the positions complete by its first test position: up to ``first_test_of_block - purge - embargo``
    inclusive, from 0 (expanding) or the last ``rolling_days`` of them (rolling)."""
    if step < 1 or purge < 0 or embargo < 0:
        raise ValueError("step must be at least 1 and purge and embargo non-negative")
    if window not in ("expanding", "rolling"):
        raise ValueError("window must be expanding or rolling")
    end = n if last_test is None else min(n, last_test)
    out, fold = [], 0
    for lo in range(first_test, end, step):
        hi = min(lo + step, end)
        train_end = lo - purge - embargo + 1                                    # exclusive: position lo - purge - embargo is the last whose label is complete by lo
        if train_end < 30:
            raise ValueError(f"the first test block at position {lo} leaves only {max(train_end, 0)} training positions after the purge of {purge + embargo}")
        train_start = 0 if window == "expanding" else max(0, train_end - int(rolling_days))
        out.append(Split(np.arange(train_start, train_end), np.arange(lo, hi), "walk_forward", fold))
        fold += 1
    return out


def purged_kfold_splits(n: int, n_splits: int, purge: int, embargo: int = 0) -> list[Split]:
    """``n_splits`` contiguous test folds; the training rows are everything outside the fold except rows whose label window reaches into it and the ``embargo`` rows after it."""
    if n_splits < 2 or n_splits > n // 10:
        raise ValueError("n_splits must be at least 2 and leave at least 10 positions per fold")
    return [Split(tr, te, "purged_kfold", k) for k, (tr, te) in enumerate(_purged_kfold(n, n_splits, horizon=purge, embargo=embargo))]


def embargoed_split(n: int, test_fraction: float, purge: int, embargo: int) -> Split:
    """The first ``1 - test_fraction`` of the positions to train (less the purge and the embargo), the rest to test."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    cut = int(round(n * (1.0 - test_fraction)))
    return Split(np.arange(0, max(cut - purge - embargo + 1, 0)), np.arange(cut, n), "embargoed", 0)


def cpcv_splits(n: int, n_groups: int, n_test_groups: int, purge: int, embargo: int = 0) -> list[Split]:
    """All ``C(N, k)`` combinatorial purged splits (Lopez de Prado 2018, chapter 12)."""
    if not 2 <= n_test_groups < n_groups <= 12:
        raise ValueError("need 2 <= n_test_groups < n_groups <= 12")
    return [Split(s["train"], s["test"], "cpcv", i, tuple(int(g) for g in s["test_groups"])) for i, s in enumerate(_cpcv(n, n_groups, n_test_groups, horizon=purge, embargo=embargo))]


# ------------------------------------------------------------------------------------------------------------------ the development timeline
SEGMENTS = ("train", "validation", "test", "forward")


@dataclass(frozen=True)
class SegmentPlan:
    """Where the timeline is cut. ``train`` is the warm-up (the models are fit on it and nothing is predicted out of sample), ``validation`` is where models and settings are chosen, ``test`` and ``forward``
    are reserved: they are reported once after selection and never used to choose anything. ``bounds`` maps each segment to ``(first position, last position + 1)``."""

    dates: pd.DatetimeIndex
    bounds: dict

    def segment_of(self, pos: int) -> str:
        for name, (lo, hi) in self.bounds.items():
            if lo <= pos < hi:
                return name
        return "train" if pos < self.bounds["train"][1] else "forward"

    def labels(self) -> pd.Series:
        """The segment name of every date."""
        out = pd.Series("train", index=self.dates, dtype=object)
        for name, (lo, hi) in self.bounds.items():
            out.iloc[lo:hi] = name
        return out

    def first_prediction(self) -> int:
        return self.bounds["validation"][0]

    def to_dict(self) -> dict:
        return {k: {"start": str(self.dates[lo].date()), "end": str(self.dates[hi - 1].date()), "n_days": int(hi - lo)} for k, (lo, hi) in self.bounds.items() if hi > lo}


def segment_plan(dates: pd.DatetimeIndex, validation: float = 0.20, test: float = 0.15, forward: float = 0.15, min_train: int = 756) -> SegmentPlan:
    """Cut a sorted date axis into train / validation / test / forward from the END: the forward segment is the last ``forward`` share, the test segment the ``test`` share before it, the validation segment
    the ``validation`` share before that, and the rest is the warm-up. Refuses a timeline too short for ``min_train`` days of warm-up and at least 40 days in every non-empty segment."""
    n = len(dates)
    nf, nt, nv = int(round(n * forward)), int(round(n * test)), int(round(n * validation))
    train_end = n - nf - nt - nv
    for label, size in (("forward", nf), ("test", nt), ("validation", nv)):
        if 0 < size < 40:
            raise ValueError(f"the {label} segment would have only {size} days: use more data or a larger fraction")
    if train_end < min_train:
        raise ValueError(f"the warm-up training segment would have {train_end} days but split.min_train_days is {min_train}: the data has {n} dates, so reduce the validation/test/forward fractions or min_train_days")
    b = {"train": (0, train_end), "validation": (train_end, train_end + nv), "test": (train_end + nv, train_end + nv + nt), "forward": (train_end + nv + nt, n)}
    return SegmentPlan(pd.DatetimeIndex(dates), b)


def check_no_overlap(split: Split, purge: int, embargo: int = 0) -> bool:
    """True if no training label can see a test row's information: for sequential splits every training position is complete by the first test position (``i + purge + embargo <= first test``); for the
    others no training position lies in a test block, within ``purge`` positions before one, or within ``embargo`` positions after one."""
    if not len(split.train) or not len(split.test):
        return True
    if split.kind in ("walk_forward", "embargoed"):
        return bool(split.train.max() + purge + embargo <= split.test.min())
    if np.isin(split.train, split.test).any():
        return False
    breaks = np.flatnonzero(np.diff(split.test) > 1) + 1
    for block in np.split(split.test, breaks):
        if (((split.train + purge >= block[0]) & (split.train <= block[-1] + embargo))).any():
            return False
    return True
