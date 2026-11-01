"""Label intervals on the real calendar: which rows may train a model fitted at a given moment, and how much their labels overlap (P0-13, P0-17, P0-21).

A supervised label is a statement about the future of its row: for the platform's targets the row on day ``t`` (position ``p`` on the dataset's own date axis) is labelled by what happens from the close of day
``p + lag`` to the close of day ``p + lag + horizon``. Its **interval** is those two positions, its **availability** is the close of the end day. A row may train a model only if its label is available by the
moment the model is fitted, so the purge is "the label ended by the first decision the model will make" on the real calendar, not "``lag + horizon`` positions of whatever list of labelled dates happens to be at hand". Writing it
this way also covers irregular horizons and the longest of several heads.

Overlapping labels are dependent samples: rows one day apart share ``h - 1`` of their ``h`` daily returns (``return_days`` gives the interval to count). ``concurrency``, ``average_uniqueness`` and ``uniqueness_weights`` are the standard treatment (Lopez de Prado,
*Advances in Financial Machine Learning*, chapter 4): the number of labels alive on each day, the average of ``1 / concurrency`` over a label's own days, and weights proportional to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONVENTION = {"name": "close_to_close", "entry": "close of day t + lag", "exit": "close of day t + lag + horizon", "price": "adjusted close", "resolution": "date (session times arrive with the session calendars)"}


def label_ends(positions: np.ndarray, lag: int, horizon: int) -> np.ndarray:
    """Position on the date axis of the last day each label depends on."""
    return np.asarray(positions, dtype=np.int64) + int(lag) + int(horizon)


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
