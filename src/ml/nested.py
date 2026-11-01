"""Nested chronological selection (spec v3, Release 5, P1-15).

Choosing among models, feature sets or hyper-parameters on the same period that is then reported is selection bias. The nested procedure here keeps the choice inside the development window and walks
forward in time:

* at each predeclared ``cutoff`` the candidates are scored on the *inner window*: the most recent ``inner_days`` decision dates whose labels had **already ended** by the cutoff (``label_end <= cutoff``).
  Nothing that happened after the cutoff, and no label still being formed, enters the choice;
* the candidate chosen at a cutoff is then applied, unchanged, to the *outer window* that follows it, up to the next cutoff. The chained outer predictions are the only out-of-sample record of the
  procedure, and its result is a property of the *selection rule*, not of the best candidate in hindsight;
* ties (within ``tol`` of the best score) go to the candidate declared first, the predeclared simplicity order;
* nothing beyond ``until`` (the end of the development window) is read: the protected test and forward segments are evaluated only under a frozen plan (``frozen_eval``), never by this procedure.

Every step records its cutoff, the inner and outer windows, the number and a hash of the rows on each side, the scores of every candidate and the latest label end among the inner rows. ``audit`` recomputes
those facts independently from the label intervals and fails on any violation, so the record can be re-checked without trusting the function that wrote it.

A candidate's own predictions are assumed to be out-of-sample already (they come from ``walk_forward_train``, whose per-fold ``fit_cutoff`` the audit can also check): the nesting here is of the
SELECTION, on top of walk-forward fitting.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import pandas as pd

from . import evaluation as ev

VERSION = "nested-v1"


def _hash_dates(dates) -> str:
    d = pd.DatetimeIndex(dates)
    return hashlib.sha256(np.ascontiguousarray(d.asi8.astype(np.int64)).tobytes()).hexdigest()[:16] if len(d) else "empty"


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(index).as_unit("ns").asi8


def select_as_of(candidates: Mapping[str, pd.DataFrame], target: pd.DataFrame, label_end: pd.Series, cutoffs, *, inner_days: int = 252, min_inner: int = 60, tol: float = 0.0,
                 until=None, score=None) -> dict:
    """Walk through ``cutoffs`` choosing among ``candidates`` (name -> wide scores, date x asset) by their mean ``score`` (rank IC against ``target`` by default) on the inner window, and chain the chosen
    candidate's scores over the following outer window. ``label_end`` maps each decision date to the date its label ends (``Dataset.label_intervals()['label_end_at']``)."""
    names = list(candidates)
    if not names:
        raise ValueError("no candidates")
    cutoffs = [pd.Timestamp(c) for c in cutoffs]
    if cutoffs != sorted(cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be strictly increasing")
    until = pd.Timestamp(until) if until is not None else None
    dates = pd.DatetimeIndex(sorted(set().union(*[set(c.index) for c in candidates.values()])))
    ics = {n: (score or ev.rank_ic_wide)(candidates[n].reindex(index=dates), target.reindex(index=dates, columns=candidates[n].columns)) for n in names}
    ends = label_end.reindex(dates)
    chain = pd.DataFrame(np.nan, index=dates, columns=next(iter(candidates.values())).columns)
    steps = []
    bounds = cutoffs + [None]
    for k, c in enumerate(cutoffs):
        nxt = bounds[k + 1]
        if until is not None and c >= until:
            steps.append({"step": k, "cutoff": str(c.date()), "state": "skipped", "reason": f"the cutoff is at or beyond the end of the development window ({until.date()})"})
            continue
        mature = dates[(ends.notna() & (ends <= c)).to_numpy()]
        inner = mature[-inner_days:]
        usable = [d for d in inner if all(np.isfinite(ics[n].get(d, np.nan)) for n in names)]
        if len(usable) < min_inner:
            steps.append({"step": k, "cutoff": str(c.date()), "state": "insufficient", "n_inner": len(usable), "reason": f"{len(usable)} mature decision dates with a score for every candidate: fewer than {min_inner}"})
            continue
        scores = {n: float(np.nanmean([ics[n][d] for d in usable])) for n in names}
        best = max(scores.values())
        chosen = next(n for n in names if scores[n] >= best - tol)                      # the first-declared candidate within tol of the best
        end = nxt if nxt is not None else (until if until is not None else dates[-1])
        if until is not None and end > until:
            end = until
        outer = dates[(dates > c) & (dates <= end)]
        chain.loc[outer] = candidates[chosen].reindex(index=outer, columns=chain.columns).to_numpy()
        steps.append({"step": k, "cutoff": str(c.date()), "state": "selected", "chosen": chosen, "scores": scores, "tol": tol, "n_inner": len(usable), "inner_first": str(usable[0].date()), "inner_last": str(usable[-1].date()),
                      "inner_hash": _hash_dates(usable), "max_inner_label_end": str(pd.Timestamp(ends.loc[usable].max()).date()), "n_outer": int(len(outer)),
                      "outer_first": str(outer[0].date()) if len(outer) else None, "outer_last": str(outer[-1].date()) if len(outer) else None, "outer_hash": _hash_dates(outer),
                      "inner_dates": [str(d.date()) for d in usable], "outer_dates": [str(d.date()) for d in outer]})
    n_sel = sum(1 for s in steps if s["state"] == "selected")
    return {"version": VERSION, "candidates": names, "inner_days": inner_days, "min_inner": min_inner, "tol": tol, "until": str(until.date()) if until is not None else None, "steps": steps,
            "chain": chain, "eligibility": "applied" if n_sel else "insufficient", "reason": "" if n_sel else "no cutoff had enough mature decision dates to choose on",
            "n_selected": n_sel, "choices": [s.get("chosen") for s in steps if s["state"] == "selected"],
            "notes": ["the choice at each cutoff uses only decision dates whose labels had ended by the cutoff", "the chained outer predictions are the out-of-sample record of the selection rule; the best candidate in hindsight is not"]}


def audit(result: dict, label_end: pd.Series, fold_records: Mapping[str, list] | None = None) -> dict:
    """Re-check a nested-selection record from the label intervals alone. Returns ``{"ok": bool, "violations": [...]}``. Checks, per selected step: every inner date's label ended on or before the cutoff;
    the inner window lies before the outer window; the outer window starts after the cutoff; windows are disjoint and ordered; the recorded hashes match the recorded dates; nothing exceeds ``until``;
    and, when ``fold_records`` (candidate -> the ``folds`` of its ``TrainResult``) are given, that each fold's fit cutoff precedes the first decision it predicted."""
    bad: list[str] = []
    prev_outer_last = None
    until = pd.Timestamp(result["until"]) if result.get("until") else None
    for s in result["steps"]:
        if s["state"] != "selected":
            continue
        c = pd.Timestamp(s["cutoff"])
        inner, outer = pd.DatetimeIndex(s["inner_dates"]), pd.DatetimeIndex(s["outer_dates"])
        ends = pd.DatetimeIndex(label_end.reindex(inner))
        if ends.isna().any() or (ends > c).any():
            bad.append(f"step {s['step']}: an inner row's label ends after the cutoff {c.date()} (latest {ends.max()})")
        if len(outer) and outer.min() <= c:
            bad.append(f"step {s['step']}: the outer window starts on or before the cutoff")
        if len(outer) and inner.max() >= outer.min():
            bad.append(f"step {s['step']}: the inner and outer windows overlap")
        if prev_outer_last is not None and len(outer) and outer.min() <= prev_outer_last:
            bad.append(f"step {s['step']}: the outer window overlaps the previous one")
        if len(outer):
            prev_outer_last = outer.max()
        if _hash_dates(inner) != s["inner_hash"] or _hash_dates(outer) != s["outer_hash"]:
            bad.append(f"step {s['step']}: a recorded hash does not match the recorded dates")
        if until is not None and len(outer) and outer.max() > until:
            bad.append(f"step {s['step']}: the outer window runs beyond the development window ({until.date()})")
        for name, folds in (fold_records or {}).items():
            for f in folds:
                if f.get("skipped") or not f.get("fit_cutoff") or not f.get("test_start"):
                    continue
                if pd.Timestamp(f["fit_cutoff"]) > pd.Timestamp(f["test_start"]):
                    bad.append(f"candidate {name}: a fold was fitted at {f['fit_cutoff']}, after the first decision it predicted ({f['test_start']})")
    return {"ok": not bad, "violations": bad, "steps_checked": sum(1 for s in result["steps"] if s["state"] == "selected")}
