"""Label intervals on the real calendar: which rows may train a model fitted at a given moment, and how much their labels overlap (P0-13, P0-17, P0-21).

A supervised label is a statement about the future of its row: for the platform's targets the row on day ``t`` (position ``p`` on the dataset's own date axis) is labelled by what happens from the close of day
``p + lag`` to the close of day ``p + lag + horizon``. Its **interval** is those two positions, its **availability** is the close of the end day. A row may train a model only if its label is available by the
moment the model is fitted, so the purge is "the label ended by the first decision the model will make" on the real calendar, not "``lag + horizon`` positions of whatever list of labelled dates happens to be at hand". Writing it
this way also covers irregular horizons and the longest of several heads.

Overlapping labels are dependent samples: rows one day apart share ``h - 1`` of their ``h`` daily returns (``return_days`` gives the interval to count). ``concurrency``, ``average_uniqueness`` and ``uniqueness_weights`` are the standard treatment (Lopez de Prado,
*Advances in Financial Machine Learning*, chapter 4): the number of labels alive on each day, the average of ``1 / concurrency`` over a label's own days, and weights proportional to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONVENTION = {"name": "close_to_close", "entry": "close of day t + lag", "exit": "close of day t + lag + horizon", "price": "adjusted close", "resolution": "date (session times arrive with the session calendars)"}


def label_ends(positions: np.ndarray, lag: int, horizon: int) -> np.ndarray:
    """Position on the date axis of the last day each label depends on. ``horizon`` is one number or one per row (labels of different lengths; for several heads, the longest of them)."""
    return np.asarray(positions, dtype=np.int64) + int(lag) + np.asarray(horizon, dtype=np.int64)


def label_frame(dates: pd.DatetimeIndex, positions: np.ndarray, lag: int, horizon: int) -> pd.DataFrame:
    """Per-row ``label_start_at``, ``label_end_at`` and ``label_available_at`` (the close of the end day; ``NaT`` when that day is not in the data yet), and the positions they come from."""
    start, end = np.asarray(positions, dtype=np.int64) + int(lag), label_ends(positions, lag, horizon)
    n = len(dates)

    def at(p):
        return pd.DatetimeIndex([dates[i] if 0 <= i < n else pd.NaT for i in p])

    return pd.DataFrame({"label_start_pos": start, "label_end_pos": end, "label_start_at": at(start), "label_end_at": at(end), "label_available_at": at(end)})


def eligible_for_fit(positions: np.ndarray, lag: int, horizon: int, decision_pos: int, embargo: int = 0) -> np.ndarray:
    """Rows whose label ended by the first decision the fitted model will make (``decision_pos``, a position on the date axis), less an embargo of whole days.

    The order of events within a closing timestamp is fixed: the closing prices print, labels whose last day is this close settle, the model is refitted if one is due, the day's features are computed, the
    decision is made. So a label that ends ON the decision day's close is available to a model that decides at that close, and one that ends a day later is not.
    """
    return label_ends(positions, lag, horizon) <= int(decision_pos) - int(embargo)


def return_days(positions: np.ndarray, lag: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """The first and last day of the RETURNS a label is built from (``p + lag + 1`` to ``p + lag + horizon``): what two labels share is returns, so this is the interval to count dependence on."""
    p = np.asarray(positions, dtype=np.int64)
    return p + int(lag) + 1, p + int(lag) + int(horizon)


def overlap(a_start: np.ndarray, a_end: np.ndarray, b_start: int, b_end: int) -> np.ndarray:
    """Which of the intervals ``[a_start, a_end]`` intersect ``[b_start, b_end]`` (both ends inclusive)."""
    return (np.asarray(a_start) <= b_end) & (np.asarray(a_end) >= b_start)


def concurrency(starts: np.ndarray, ends: np.ndarray, n_days: int) -> np.ndarray:
    """``c[d]`` = how many of the intervals ``[start, end]`` (inclusive) contain day ``d``."""
    delta = np.zeros(n_days + 1)
    s, e = np.clip(np.asarray(starts), 0, n_days), np.clip(np.asarray(ends) + 1, 0, n_days)
    np.add.at(delta, s, 1.0)
    np.add.at(delta, e, -1.0)
    return np.cumsum(delta)[:n_days]


def average_uniqueness(starts: np.ndarray, ends: np.ndarray, n_days: int) -> np.ndarray:
    """For each interval, the mean over its own days of ``1 / concurrency``: 1 for a label that overlaps nobody, about ``1 / horizon`` for labels one day apart."""
    c = concurrency(starts, ends, n_days)
    inv = np.divide(1.0, c, out=np.zeros_like(c), where=c > 0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    s, e = np.clip(np.asarray(starts), 0, n_days - 1), np.clip(np.asarray(ends), 0, n_days - 1)
    return (cum[e + 1] - cum[s]) / (e - s + 1)


def uniqueness_weights(starts: np.ndarray, ends: np.ndarray, n_days: int) -> np.ndarray:
    """Sample weights proportional to average uniqueness, scaled to mean one."""
    u = average_uniqueness(starts, ends, n_days)
    return u * len(u) / u.sum()


# ------------------------------------------------------------------------------------------------------------------ research diagnostics: purged K-fold and CPCV on the real calendar


@dataclass(frozen=True)
class IntervalSplit:
    """Row indices of one training/test split. ``chronological`` is False for the K-fold and CPCV splits: they train on blocks that come AFTER the ones they test, so their predictions are research diagnostics of
    how a model generalises over blocks, never a simulation of trading forward in time."""

    train: np.ndarray
    test: np.ndarray
    kind: str
    fold: int
    test_groups: tuple = ()
    chronological: bool = False


def _blocks(positions: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """The distinct decision positions, in order, cut into ``n_groups`` contiguous blocks of nearly equal numbers of dates."""
    unique = np.unique(np.asarray(positions, dtype=np.int64))
    if n_groups < 2 or n_groups > len(unique) // 10:
        raise ValueError("need at least 2 groups and at least 10 dates in each")
    return [b for b in np.array_split(unique, n_groups)]


def _spans(positions: np.ndarray, lag: int, horizon) -> tuple[np.ndarray, np.ndarray]:
    """Start and end day of each row's label: ``p + lag`` to ``p + lag + horizon``, with one horizon for all rows or one per row (irregular labels, such as ones that end at a barrier)."""
    pos = np.asarray(positions, dtype=np.int64)
    return pos + int(lag), pos + int(lag) + np.broadcast_to(np.asarray(horizon, dtype=np.int64), pos.shape)


def _excluded(starts: np.ndarray, ends: np.ndarray, test: np.ndarray, groups: list[np.ndarray], positions: np.ndarray, embargo: int) -> np.ndarray:
    """Rows to keep out of training for a test set: those whose label interval shares a day with any label of the test rows (found by marking the days the test labels cover and counting covered days inside each
    interval), and those whose label STARTS in the ``embargo`` days after the last label of a test group ends. Everything is on the real calendar, so a missing date moves nothing, and intervals may differ in length."""
    n_days = int(max(starts.max(), ends.max())) + 2
    covered = concurrency(starts[test], ends[test], n_days) > 0
    cum = np.concatenate([[0], np.cumsum(covered)])
    drop = (cum[ends + 1] - cum[starts]) > 0
    for g in groups:
        last = int(ends[np.isin(positions, g)].max())
        drop |= (starts > last) & (starts <= last + int(embargo))
    return drop


def purged_kfold_rows(positions: np.ndarray, horizon, n_splits: int = 5, embargo: int = 0, lag: int = 0) -> list[IntervalSplit]:
    """``n_splits`` contiguous test blocks of the rows' decision dates. Training rows are every row outside the block except those whose label overlaps one of the block's and those in the embargo after it. ``positions``
    are the rows' decision positions on the real calendar (one per row; dates with no row are simply absent), ``horizon`` the label length in days (one number, or one per row). A research diagnostic: see
    ``IntervalSplit.chronological``."""
    pos = np.asarray(positions, dtype=np.int64)
    starts, ends = _spans(pos, lag, horizon)
    out = []
    for k, group in enumerate(_blocks(pos, n_splits)):
        test = np.isin(pos, group)
        out.append(IntervalSplit(np.flatnonzero(~test & ~_excluded(starts, ends, test, [group], pos, embargo)), np.flatnonzero(test), "purged_kfold", k))
    return out


def cpcv_rows(positions: np.ndarray, horizon, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 0, lag: int = 0) -> list[IntervalSplit]:
    """Every choice of ``n_test_groups`` of ``n_groups`` blocks as the test set (combinatorial purged cross-validation, López de Prado 2018, ch. 12), purged and embargoed as in ``purged_kfold_rows``."""
    from itertools import combinations

    if not 2 <= n_test_groups < n_groups <= 12:
        raise ValueError("need 2 <= n_test_groups < n_groups <= 12")
    pos = np.asarray(positions, dtype=np.int64)
    starts, ends = _spans(pos, lag, horizon)
    blocks = _blocks(pos, n_groups)
    out = []
    for i, chosen in enumerate(combinations(range(n_groups), n_test_groups)):
        groups = [blocks[g] for g in chosen]
        test = np.isin(pos, np.concatenate(groups))
        out.append(IntervalSplit(np.flatnonzero(~test & ~_excluded(starts, ends, test, groups, pos, embargo)), np.flatnonzero(test), "cpcv", i, tuple(int(g) for g in chosen)))
    return out
