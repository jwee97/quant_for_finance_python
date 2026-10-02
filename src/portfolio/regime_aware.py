"""Regime-aware portfolio rules (Generation 2, Priority 1).

Each rule takes a base book that already exists and changes it as a function of
a regime probability known at the decision date. None re-optimises anything, and
each has at most one structural parameter, declared in config before any result:

``derisk_overlay``  RA1. Shift up to ``derisk_max`` of the book into cash as the
                    probability of the high-volatility state approaches one.
``blend_books``     RA2. A convex combination of two existing books, weighted by
                    that same probability. No free parameter.
``training_gate``   RA3. Switch a book off in regimes where it LOST money on
                    the training sample. The gate for each window uses only
                    returns that were already realised at that window's refit.

All three return target weights stamped at the decision date; the engine
applies the execution lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _probability(p: pd.Series, index: pd.Index) -> pd.Series:
    """A probability aligned to ``index``; undefined (before the first fit) means 'no information' = 0."""
    return p.reindex(index).fillna(0.0).clip(0.0, 1.0)


def derisk_overlay(base: pd.DataFrame, p_high: pd.Series, cash: str, derisk_max: float = 0.5) -> pd.DataFrame:
    """``(1 - d p) * base + d p * cash``, so gross exposure is unchanged and risk moves to cash."""
    if cash not in base.columns:
        raise KeyError(f"cash asset '{cash}' is not a column of the base book")
    p = _probability(p_high, base.index)
    shift = float(derisk_max) * p
    out = base.mul(1.0 - shift, axis=0)
    out[cash] = out[cash].fillna(0.0) + shift * base.sum(axis=1, min_count=1).fillna(0.0)
    return out.where(base.notna().any(axis=1), np.nan)


def blend_books(calm: pd.DataFrame, stressed: pd.DataFrame, p_high: pd.Series) -> pd.DataFrame:
    """``(1 - p) * calm + p * stressed`` on the dates both books exist."""
    index = calm.index.intersection(stressed.index)
    p = _probability(p_high, index)
    columns = calm.columns.union(stressed.columns)
    a = calm.reindex(index=index, columns=columns).fillna(0.0)
    b = stressed.reindex(index=index, columns=columns).fillna(0.0)
    both = calm.reindex(index).notna().any(axis=1) & stressed.reindex(index).notna().any(axis=1)
    out = a.mul(1.0 - p, axis=0) + b.mul(p, axis=0)
    return out.where(both, np.nan)


def gate_book(base: pd.DataFrame, gate: pd.Series) -> pd.DataFrame:
    """Multiply the book by a gate in [0, 1] known at each date (0 = flat)."""
    g = gate.reindex(base.index).fillna(1.0).clip(0.0, 1.0)
    return base.mul(g, axis=0)


def _state_gate(returns: np.ndarray, labels: np.ndarray, n_states: int, upto: int,
                min_days: int, horizon: int) -> tuple[np.ndarray, list[dict]]:
    """Open/closed flag per state from pairs (label on t, return on t + horizon) with t + horizon < upto."""
    t = np.arange(0, max(upto - horizon, 0))
    paired_label = labels[t]
    paired_return = returns[t + horizon]
    keep = np.isfinite(paired_return)
    opened = np.ones(n_states)
    rows = []
    for k in range(n_states):
        mask = keep & (paired_label == k)
        n_k = int(mask.sum())
        mean_k = float(paired_return[mask].mean()) if n_k else float("nan")
        if n_k >= min_days and mean_k <= 0.0:
            opened[k] = 0.0
        rows.append({"state": k, "n_days": n_k, "mean_daily_return": mean_k,
                     "gate_open": bool(opened[k])})
    return opened, rows


def training_gate(book_returns: pd.Series, windows: list[dict], index: pd.DatetimeIndex,
                  min_days: int = 60, horizon: int = 2) -> tuple[pd.Series, pd.DataFrame]:
    """A 0/1 gate from each window's TRAINING-sample in-regime mean return.

    For the window that starts at row ``r`` the training sample is rows
    ``0..r-1``. Pair the regime label on day t (the most likely state under that
    window's fit, filtered) with the book's return on day ``t + horizon``, the
    first day a weight decided on t can earn (the engine's one-day execution lag
    plus the day the book is held), keeping only pairs whose return was already
    realised before row ``r``. The gate is open (1) in a state when its training
    mean return is positive or when fewer than ``min_days`` pairs fall in it
    (no evidence to switch the book off), and closed (0) otherwise.

    Returns the gate over the out-of-sample rows (NaN before the first window)
    and a table of what each window learned.
    """
    returns = book_returns.reindex(index).to_numpy(dtype=float)
    gate = pd.Series(np.nan, index=index, name="gate")
    log = []
    for w in windows:
        start, end, alpha = w["start"], w["end"], w["alpha"]
        labels = alpha.argmax(axis=1)
        opened, rows = _state_gate(returns, labels, alpha.shape[1], start, min_days, horizon)
        log.extend({"refit": index[start], **row} for row in rows)
        gate.iloc[start:end] = opened[labels[start:end]]
    return gate, pd.DataFrame(log)


def full_sample_gate(book_returns: pd.Series, probabilities: pd.DataFrame, min_days: int = 60,
                     horizon: int = 2) -> tuple[pd.Series, pd.DataFrame]:
    """The LOOK-AHEAD control for ``training_gate``: one gate learned on the whole sample.

    Every day's gate is decided by what the book earned in that regime over the
    entire history, including days after it. Only ever used to show how much
    that overstates a strategy.
    """
    returns = book_returns.reindex(probabilities.index).to_numpy(dtype=float)
    labels = probabilities.to_numpy().argmax(axis=1)
    opened, rows = _state_gate(returns, labels, probabilities.shape[1], len(returns), min_days, horizon)
    return pd.Series(opened[labels], index=probabilities.index, name="gate"), pd.DataFrame(rows)
