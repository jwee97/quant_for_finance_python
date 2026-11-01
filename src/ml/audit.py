"""What each feature reads, how far back it reaches and whether anything after its date can matter; what a dataset holds and what it left out (P0-14, and the report half of P0-13/15/16).

``feature_dependencies`` measures instead of trusting. It perturbs ONE input panel at a time (prices, returns, volume, high, low, the unadjusted close, the macro panel, the market return), recomputes the
feature, and reads three facts off the difference: which panels the feature reads, how many trailing rows its value on the last date depends on, and whether replacing, blanking or freezing every row after a
cutoff (or leaving a hole that straddles it) changes any value on or before the cutoff. A feature that reads an input its spec does not declare, looks ahead through a backward fill, or needs more history than
it says is found by that test, not by someone reading the code.

``dataset_audit`` is the part of the same question that is about the DATA: the label convention and its cutoffs, missing and stale coverage per feature and per asset, the rows that were left out and why, the
point-in-time quality of the macro inputs and the session policy. It is stored with the dataset (``meta["audit"]``) and shown in the experiment report.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .features import FEATURES, Inputs, available

PANELS = ("prices", "returns", "volume", "high", "low", "raw_close", "macro", "market")
# The panels a feature may read, given the inputs its spec says it needs. Prices, returns and the market return are always allowed: every dataset has them.
ALLOWED = {"": {"prices", "returns", "market"}, "volume": {"volume"}, "ohlc": {"high", "low"}, "raw_close": {"raw_close"}, "macro": {"macro"}, "carry": {"macro"}, "iv": {"macro"}, "crypto": {"macro"}}
KINDS = ("noise", "missing", "stale", "straddle")
TOLERANCE = 1e-10


def allowed_reads(needs: tuple) -> set:
    out = set(ALLOWED[""])
    for n in needs:
        out |= ALLOWED.get(n, set())
    return out


# ------------------------------------------------------------------------------------------------------------------ perturbing an input
def _panel(x: Inputs, name: str):
    return getattr(x, name)


def _present(x: Inputs) -> list[str]:
    out = []
    for p in PANELS:
        v = _panel(x, p)
        if v is not None and len(v) > 0 and (not isinstance(v, pd.DataFrame) or v.shape[1] > 0):
            out.append(p)
    return out


def _noisy(frame, rows: np.ndarray, rng: np.random.Generator, additive: bool):
    """``frame`` with the selected rows pushed off: by a random factor in [0.8, 1.2] for levels (prices, volume ...), by noise in the scale of the data for returns and macro series."""
    out = frame.copy()
    if not rows.any():
        return out
    block = out.loc[rows]
    if additive:
        scale = np.nan_to_num(np.nanstd(frame.to_numpy(dtype=float), axis=0), nan=1.0)                # the spread of each column, so that a series near 2 is not pushed by one near 1e11
        out.loc[rows] = block.to_numpy(dtype=float) + rng.normal(0.0, 1.0, block.shape) * np.where(scale > 0, scale, 1.0)
    else:
        out.loc[rows] = block.to_numpy(dtype=float) * rng.uniform(0.8, 1.2, block.shape)
    return out


def _changed(x: Inputs, panels: list[str], rows: np.ndarray, kind: str, rng: np.random.Generator, hold: np.ndarray | None = None) -> Inputs:
    """``x`` with ``panels`` perturbed on ``rows`` (a boolean row mask) in the way ``kind`` names. ``hold`` is a row mask left missing in the base data too (the straddling hole)."""
    updates = {}
    for p in panels:
        frame = _panel(x, p)
        if frame is None:
            continue
        if kind == "noise":
            new = _noisy(frame, rows, rng, additive=p in ("returns", "macro", "market"))
        elif kind == "missing":
            new = frame.copy()
            new.loc[rows] = np.nan
        elif kind == "stale":
            new = frame.copy()
            k = max(int(np.flatnonzero(rows)[0]) - 1, 0) if rows.any() else 0
            held = frame.iloc[k]
            new.loc[rows] = np.broadcast_to(held.to_numpy(dtype=float), (int(rows.sum()), len(held))) if hasattr(held, "to_numpy") else float(held)
        else:
            raise ValueError(f"unknown perturbation '{kind}'")
        updates[p] = new
    return dataclasses.replace(x, **updates)


def _blank(frame, hole: np.ndarray):
    out = frame.copy()
    out.loc[hole] = np.nan
    return out


def _worst(a: pd.DataFrame, b: pd.DataFrame, rows: np.ndarray | slice) -> float:
    """The largest relative difference between two feature frames over ``rows``; a value missing in one and present in the other counts as 1."""
    left, right = a.iloc[rows].to_numpy(dtype=float), b.iloc[rows].to_numpy(dtype=float)
    nan_l, nan_r = np.isnan(left), np.isnan(right)
    if (nan_l != nan_r).any():
        return 1.0
    ok = ~nan_l
    if not ok.any():
        return 0.0
    return float(np.max(np.abs(left[ok] - right[ok]) / (1.0 + np.abs(left[ok]))))


# ------------------------------------------------------------------------------------------------------------------ the audit of one feature
def _frame(spec, x: Inputs) -> pd.DataFrame:
    return spec.fn(x).reindex(index=x.prices.index, columns=x.assets).replace([np.inf, -np.inf], np.nan)


def forward_changes(spec, x: Inputs, cutoff_pos: int, seed: int = 0) -> dict:
    """The largest change on or before the cutoff when everything after it is replaced (``noise``), blanked (``missing``), frozen at its last value (``stale``), or replaced while a hole straddles the cutoff (``straddle``)."""
    rng = np.random.default_rng(seed)
    n = len(x.prices.index)
    future = np.arange(n) > cutoff_pos
    prefix = ~future
    present = _present(x)
    out = {}
    base = _frame(spec, x)
    for kind in ("noise", "missing", "stale"):
        out[kind] = _worst(base, _frame(spec, _changed(x, present, future, kind, rng)), prefix)
    hole = (np.arange(n) >= cutoff_pos - 1) & (np.arange(n) <= cutoff_pos + 2)                    # a gap that begins inside the prefix and ends after it: a backward fill would reach across
    held = dataclasses.replace(x, **{p: _blank(_panel(x, p), hole) for p in present})
    out["straddle"] = _worst(_frame(spec, held), _frame(spec, _changed(held, present, future & ~hole, "noise", rng)), prefix)
    return out


def reads(spec, x: Inputs, base: pd.DataFrame, seed: int = 0) -> list[str]:
    """The input panels whose perturbation, anywhere in the history, changes the feature."""
    rng = np.random.default_rng(seed)
    everywhere = np.ones(len(x.prices.index), dtype=bool)
    found = []
    for p in _present(x):
        try:
            other = _frame(spec, _changed(x, [p], everywhere, "noise", rng))
        except Exception:                                                                            # noqa: BLE001 - a feature that cannot run without the panel certainly reads it
            found.append(p)
            continue
        if _worst(base, other, slice(None)) > TOLERANCE:
            found.append(p)
    return found


def window(spec, x: Inputs, base: pd.DataFrame, panels: list[str], seed: int = 0) -> int | None:
    """How many trailing rows the value on the last date depends on: one more than the largest distance back at which perturbing the older history still changes it. ``None`` if the feature has no value to measure."""
    n = len(x.prices.index)
    last = base.iloc[-1]
    if last.notna().sum() == 0:
        return None
    rng = np.random.default_rng(seed)

    def depends(distance: int) -> bool:
        older = np.arange(n) <= (n - 1 - distance)
        other = _frame(spec, _changed(x, panels, older, "noise", rng))
        return _worst(base, other, slice(n - 1, n)) > TOLERANCE

    if not depends(1):
        return 1
    lo, hi = 1, n - 1                                                                              # depends(lo) is true; find the largest distance for which it still is
    if depends(hi):
        return n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if depends(mid) else (lo, mid)
    return lo + 1


def feature_dependencies(x: Inputs, names: list[str] | None = None, cutoff_pos: int | None = None, seed: int = 0) -> dict:
    """The measured dependency table of the features the inputs support: ``{name: {reads, window, declared_window, declared_reads, forward}}``. ``forward`` is the largest change on or before the cutoff per
    perturbation kind and must be zero (within rounding) for every feature; ``skipped`` lists the features the data cannot compute, with the reason."""
    n = len(x.prices.index)
    cutoff = n - 60 if cutoff_pos is None else int(cutoff_pos)
    table, skipped = {}, {}
    for name in names or list(FEATURES):
        spec = FEATURES[name]
        why = available(spec, x)
        if why is not None:
            skipped[name] = why
            continue
        base = _frame(spec, x)
        if base.notna().sum().sum() == 0:
            skipped[name] = "no valid values"
            continue
        r = reads(spec, x, base, seed)
        table[name] = {"group": spec.group, "reads": sorted(r), "declared_reads": sorted(allowed_reads(spec.needs)), "window": window(spec, x, base, r, seed) if r else None,
                       "declared_window": int(spec.lookback), "memory": spec.memory, "forward": forward_changes(spec, x, cutoff, seed)}
    return {"cutoff_pos": cutoff, "n_rows": n, "features": table, "skipped": skipped}


def violations(report: dict, slack: float = 1.5) -> list[str]:
    """What the measurement says is wrong, as sentences: a feature that changes before the cutoff, reads a panel it does not declare, does not read one it needs, or needs more than ``slack`` times the history it declares
    (a feature whose memory decays, such as an exponential average, is listed by ``long_memory`` instead, not here)."""
    out = []
    for name, row in report["features"].items():
        bad = {k: v for k, v in row["forward"].items() if v > TOLERANCE}
        if bad:
            out.append(f"{name} changes before the cutoff when later data change ({', '.join(f'{k}: {v:.2g}' for k, v in bad.items())}): it looks ahead")
        extra = set(row["reads"]) - set(row["declared_reads"])
        if extra:
            out.append(f"{name} reads {sorted(extra)} but its spec does not declare it")
        if row["memory"] == "finite" and row["window"] is not None and row["window"] > slack * row["declared_window"]:
            out.append(f"{name} depends on {row['window']} trailing rows but declares {row['declared_window']}")
        if row["memory"] == "recursive" and row["window"] is not None and row["window"] <= row["declared_window"]:
            out.append(f"{name} is declared recursive but depends on only {row['window']} trailing rows")
    return out


def long_memory(report: dict) -> list[str]:
    """The features whose value on the last date still depends on a row more than ``declared lookback`` back although they are declared finite: none should be, so this is the list to look at when ``violations`` is
    not empty. Exponential filters are declared ``recursive`` and are not listed."""
    return sorted(n for n, row in report["features"].items() if row["memory"] == "finite" and row["window"] is not None and row["window"] > row["declared_window"])


# ------------------------------------------------------------------------------------------------------------------ the audit of one dataset
def stale_runs(prices: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """Per asset: the share of observed prices that repeat the previous one exactly, and the longest run of such repeats. A price that does not move for ``min_run`` sessions in a row is a mark carried over
    a closure or a stale quote, not a traded price."""
    rows = {}
    for a in prices.columns:
        p = prices[a].dropna()
        same = (p.diff() == 0.0).to_numpy()
        longest = run = 0
        for s in same:
            run = run + 1 if s else 0
            longest = max(longest, run)
        rows[a] = {"repeat_share": float(same.mean()) if len(same) else float("nan"), "longest_repeat_run": int(longest), "stale": bool(longest >= min_run)}
    return pd.DataFrame(rows).T


def dataset_audit(prices: pd.DataFrame, features: dict, kept: np.ndarray, causes: dict, label: dict, pit: dict | None = None, sessions: dict | None = None) -> dict:
    """The data half of the audit. ``features`` is ``{name: (group, wide frame)}`` before cross-sectional scaling, ``kept`` the row mask the dataset kept, ``causes`` the reason counts of the rows it left out."""
    n_cells = int(kept.size)
    rows = {}
    for name, (group, frame) in features.items():
        spec = FEATURES.get(name)
        rows[name] = {"group": group, "missing_share": float(frame.isna().to_numpy().mean()) if frame.size else float("nan"), "declared_window": int(spec.lookback) if spec else None,
                      "inputs": sorted(allowed_reads(spec.needs)) if spec else ["derived"]}
    stale = stale_runs(prices)
    return {"label": label, "cells": n_cells, "rows_kept": int(kept.sum()), "excluded": {k: int(v) for k, v in causes.items()}, "features": rows,
            "stale": {"assets": [str(a) for a in stale.index[stale["stale"].astype(bool)]], "longest_run": int(stale["longest_repeat_run"].max()) if len(stale) else 0, "rule": "a price that repeats the previous one for 5 or more sessions in a row"},
            "pit": pit or {}, "sessions": sessions or {}, "imputation": "none in the feature layer: values the data lack stay missing; each model learns its fill values (median) from its own eligible training rows", "backward_fill": "none"}
