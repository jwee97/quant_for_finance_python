"""Drift monitoring on stable references, with coverage, sample counts and alert lineage (spec v3, Release 8, P3-01). Optional: nothing runs unless it is asked for.

Two different questions are kept apart.

* **Input drift.** Has the distribution of a feature moved away from the one the model was fitted on? The **reference window is fixed**: the ``reference_days`` dates ending at the model's fit cutoff, not
  a window that slides with the data and so forgets what the model learned. The recent window is the ``recent_days`` dates up to ``as_of``. Per feature the report gives the PSI (a conventional,
  configurable *heuristic*: 0.10 to watch and 0.25 to alert are not significance levels), a Kolmogorov-Smirnov statistic with a p-value computed from an **effective sample size** that allows for serial
  dependence (daily features are autocorrelated, so the nominal sample size would overstate the evidence), the p-values adjusted across features (Benjamini-Hochberg or Holm), the number of observations in
  each window and the share of cells that had a value. A feature with too few observations or too little coverage is ``insufficient``: never ``ok``.
* **Predictive-performance drift.** Has the signal stopped working? Only **settled** outcomes count: a decision whose label has not ended by ``as_of`` is not in either window. The recent mean rank IC is
  compared with the reference one by a Newey-West interval on the difference, with at least ``horizon - 1`` lags because consecutive labels overlap.

A drift result is an **alert, not an action**. ``lineage`` links each alert to the previous look at the same subject (when it first appeared, how many looks in a row), because a rule that is checked every
day will fire by chance some day, and the report says how many looks and features were tested.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import inference as inf
from . import monitoring as mon

VERSION = "drift-v1"
STATES = ("ok", "watch", "alert", "insufficient")


@dataclass(frozen=True)
class DriftSpec:
    reference_days: int = 504
    recent_days: int = 63
    min_reference: int = 250                 # observations in the reference window
    min_recent: int = 40                     # observations in the recent window
    min_coverage: float = 0.8                # share of cells with a value, in each window
    psi_watch: float = 0.10                  # heuristics, not significance levels
    psi_alert: float = 0.25
    ks_level: float = 0.01                   # level for the adjusted KS p-value
    correction: str = "bh"                   # bh | holm | none
    performance_min_dates: int = 30          # settled dates with an IC in each window
    level: float = 0.05

    def __post_init__(self):
        if min(self.reference_days, self.recent_days) < 5 or min(self.min_reference, self.min_recent) < 5:
            raise ValueError("the windows (in dates) and the minimum observation counts (cells) are at least 5")
        if not 0 <= self.psi_watch <= self.psi_alert or not 0 < self.ks_level < 1 or not 0 < self.level < 1 or not 0 <= self.min_coverage <= 1:
            raise ValueError("psi_watch <= psi_alert, levels in (0, 1) and min_coverage in [0, 1]")
        if self.correction not in ("bh", "holm", "none"):
            raise ValueError("correction must be bh, holm or none")

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def psi(reference, recent, bins: int = 10) -> float:
    """Population stability index of ``recent`` against ``reference``: the existing ``monitoring.psi`` (deciles of the reference as bin edges, empty bins floored at half an observation, NaN with fewer than 50
    and 20 finite values), with one rule added. A constant reference has no deciles to bin by; ``monitoring.psi`` reads it as 0.0 whatever the recent values are, which would call a feature that left its
    only value 'stable'. Here it is 0.0 only if the recent values are that same constant, and infinite (a different distribution) otherwise."""
    e, a = np.asarray(reference, dtype=float), np.asarray(recent, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) >= 50 and len(a) >= 20 and len(np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))) < 3:
        return 0.0 if bool(np.all(a == e[0])) else float("inf")
    return float(mon.psi(e, a, bins))


def effective_n(x) -> float:
    """The number of independent observations a serially dependent series is worth: ``n (1 - r) / (1 + r)`` with ``r`` the lag-1 autocorrelation (clipped to [0, 0.95]; negative dependence is not credited)."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or np.std(v) == 0:
        return float(len(v))
    r = float(np.clip(np.corrcoef(v[:-1], v[1:])[0, 1], 0.0, 0.95)) if len(v) > 3 else 0.0
    return float(len(v) * (1.0 - r) / (1.0 + r))


def panel_effective_n(W: pd.DataFrame) -> float:
    """The effective number of observations in a date x asset window: the finite cells, discounted by the mean lag-1 autocorrelation of each asset's series in time (so the stacking order of a panel does not
    hide the persistence). Dependence between assets on the same date is not modelled: the p-value built on this stays optimistic, and the report says so."""
    n = int(np.isfinite(W.to_numpy(dtype=float)).sum())
    rs = []
    for c in W.columns:
        v = W[c].to_numpy(dtype=float)
        a, b = v[:-1], v[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 10 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            rs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    r = float(np.clip(np.mean(rs), 0.0, 0.95)) if rs else 0.0
    return float(n * (1.0 - r) / (1.0 + r))


def ks_effective(reference, recent, n_eff: tuple | None = None) -> dict:
    """Two-sample Kolmogorov-Smirnov statistic with the p-value of the asymptotic distribution evaluated at the effective sample sizes of the two series (given, or from their own lag-1 autocorrelation)."""
    e, a = np.asarray(reference, dtype=float), np.asarray(recent, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < 5 or len(a) < 5:
        return {"statistic": None, "p_value": None, "n_eff": None}
    d = float(stats.ks_2samp(e, a).statistic)
    ne, na = n_eff if n_eff is not None else (effective_n(e), effective_n(a))
    en = np.sqrt(ne * na / (ne + na))
    p = float(stats.kstwobign.sf((en + 0.12 + 0.11 / max(en, 1e-9)) * d))                   # the small-sample correction of Stephens
    return {"statistic": d, "p_value": min(max(p, 0.0), 1.0), "n_eff": {"reference": ne, "recent": na}}


def adjust(pvalues: list, method: str) -> list:
    """Benjamini-Hochberg (the default: it controls the false discovery rate under positive dependence), Holm, or none; missing p-values stay missing."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and np.isfinite(p)]
    out = list(pvalues)
    if not idx or method == "none":
        return out
    p = np.array([pvalues[i] for i in idx], dtype=float)
    m = len(p)
    order = np.argsort(p)
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    else:
        adj = np.minimum.accumulate((m / (np.arange(m) + 1.0) * p[order])[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    res = np.empty(m)
    res[order] = adj
    for i, v in zip(idx, res):
        out[i] = float(v)
    return out


def reference_dates(dates: pd.DatetimeIndex, fit_cutoff, days: int) -> pd.DatetimeIndex:
    """The ``days`` dates ending at ``fit_cutoff`` (inclusive): the window the model was fitted on, which does not move when new data arrive."""
    d = pd.DatetimeIndex(dates).sort_values().unique()
    end = d.searchsorted(pd.Timestamp(fit_cutoff), side="right")
    return d[max(0, end - int(days)): end]


def feature_drift(X: pd.DataFrame, spec: DriftSpec, *, fit_cutoff, as_of=None) -> dict:
    """Input drift of every column of ``X`` (a date x asset panel stacked as rows with the date first in the index) against the fixed reference window ending at ``fit_cutoff``."""
    dates = X.index.get_level_values(0)
    uniq = pd.DatetimeIndex(dates.unique()).sort_values()
    as_of = pd.Timestamp(as_of) if as_of is not None else uniq[-1]
    ref = reference_dates(uniq, fit_cutoff, spec.reference_days)
    upto = uniq[uniq <= as_of]
    rec = upto[max(0, len(upto) - spec.recent_days):]
    overlap = ref.intersection(rec)
    rows, notes = [], []
    if len(overlap):
        notes.append(f"{len(overlap)} dates are in both windows: the recent window reaches back into the reference period, so drift is understated")
    n_ref_cells, n_rec_cells = len(ref) * max(1, X.index.droplevel(0).nunique()), len(rec) * max(1, X.index.droplevel(0).nunique())
    r_mask, c_mask = dates.isin(ref), dates.isin(rec)
    for col in X.columns:
        e, a = X.loc[r_mask, col].to_numpy(dtype=float), X.loc[c_mask, col].to_numpy(dtype=float)
        wide = X[col].unstack("asset") if X.index.nlevels == 2 else None
        ne, na = int(np.isfinite(e).sum()), int(np.isfinite(a).sum())
        cov_e, cov_a = ne / n_ref_cells if n_ref_cells else 0.0, na / n_rec_cells if n_rec_cells else 0.0
        row = {"feature": col, "n_reference": ne, "n_recent": na, "coverage_reference": cov_e, "coverage_recent": cov_a, "psi": None, "ks": None, "ks_p_value": None, "n_eff": None}
        if ne < spec.min_reference or na < spec.min_recent or cov_e < spec.min_coverage or cov_a < spec.min_coverage:
            reasons = [r for r, bad in (("too few reference observations", ne < spec.min_reference), ("too few recent observations", na < spec.min_recent),
                                        ("reference coverage below the floor", cov_e < spec.min_coverage), ("recent coverage below the floor", cov_a < spec.min_coverage)) if bad]
            rows.append({**row, "state": "insufficient", "reason": "; ".join(reasons)})
            continue
        eff = None if wide is None else (panel_effective_n(wide.reindex(ref)), panel_effective_n(wide.reindex(rec)))
        ks = ks_effective(e, a, eff)
        rows.append({**row, "psi": psi(e, a), "ks": ks["statistic"], "ks_p_value": ks["p_value"], "n_eff": ks["n_eff"]})
    adj = adjust([r["ks_p_value"] for r in rows], spec.correction)
    for r, q in zip(rows, adj):
        if r.get("state") == "insufficient":
            continue
        r["ks_p_adjusted"] = q
        p = r["psi"]
        shifted = q is not None and q < spec.ks_level
        if p is not None and np.isfinite(p) and p >= spec.psi_alert and shifted:
            r["state"], r["reason"] = "alert", f"PSI {p:.3f} is above the alert heuristic {spec.psi_alert} and the adjusted KS p-value {q:.3g} is below {spec.ks_level}"
        elif (p is not None and np.isfinite(p) and p >= spec.psi_watch) or shifted:
            r["state"], r["reason"] = "watch", f"PSI {p:.3f}; adjusted KS p-value {q:.3g}: one of the two heuristics is crossed, not both"
        else:
            r["state"], r["reason"] = "ok", "neither heuristic is crossed"
    counts = {s: sum(1 for r in rows if r["state"] == s) for s in STATES}
    return {"version": VERSION, "reference": {"first": str(ref[0].date()) if len(ref) else None, "last": str(ref[-1].date()) if len(ref) else None, "dates": int(len(ref)), "fixed_at": str(pd.Timestamp(fit_cutoff).date())},
            "recent": {"first": str(rec[0].date()) if len(rec) else None, "last": str(rec[-1].date()) if len(rec) else None, "dates": int(len(rec))}, "features": sorted(rows, key=lambda r: -(r["psi"] or 0.0)),
            "counts": counts, "tests": {"features": len(rows), "multiplicity": spec.correction, "p_value_basis": "asymptotic Kolmogorov distribution at effective sample sizes (the mean lag-1 autocorrelation of each asset's series); "
                                                                                                "dependence between assets on a date is not modelled, so the p-values stay optimistic"},
            "thresholds": {"psi_watch": spec.psi_watch, "psi_alert": spec.psi_alert, "ks_level": spec.ks_level, "note": "the PSI thresholds are conventional heuristics that can be configured; they are not significance levels"},
            "notes": notes}


def settled_mask(label_end: pd.Series, as_of) -> pd.Series:
    """True for the decisions whose label had ended by ``as_of``: the only ones that may inform a performance statistic."""
    return pd.Series(pd.to_datetime(label_end).to_numpy() <= pd.Timestamp(as_of).to_datetime64(), index=label_end.index)


def performance_drift(ic: pd.Series, label_end: pd.Series, spec: DriftSpec, *, horizon: int, fit_cutoff, as_of) -> dict:
    """Has the rank IC fallen since the model was fitted? ``ic`` is the daily rank IC by decision date, ``label_end`` the date each decision's label ended. Only decisions settled by ``as_of`` are used; the
    reference window is the settled decisions up to the fit cutoff, the recent window the last ``recent_days`` settled dates. The difference carries a Newey-West interval."""
    ic = ic.dropna()
    le = label_end.reindex(ic.index)
    settled = ic[settled_mask(le, as_of).to_numpy() & le.notna().to_numpy()]
    out = {"version": VERSION, "as_of": str(pd.Timestamp(as_of).date()), "settled_dates": int(len(settled)), "unsettled_dates_excluded": int(len(ic) - len(settled)), "horizon": int(horizon)}
    ref = settled[settled.index <= pd.Timestamp(fit_cutoff)].iloc[-spec.reference_days:]
    rec = settled[settled.index > pd.Timestamp(fit_cutoff)].iloc[-spec.recent_days:]
    out.update(reference_dates=int(len(ref)), recent_dates=int(len(rec)))
    if len(ref) < spec.performance_min_dates or len(rec) < spec.performance_min_dates:
        return {**out, "state": "insufficient", "reason": f"fewer than {spec.performance_min_dates} settled dates in the {'reference' if len(ref) < spec.performance_min_dates else 'recent'} window"}
    rm, cm = float(ref.mean()), float(rec.mean())
    res = inf.hac_mean(rec - rm, horizon=int(horizon), level=1 - spec.level, label="recent mean rank IC minus the reference mean")              # recent IC against the reference IC, treated as fixed
    out.update(reference_ic=rm, recent_ic=cm, difference=res.as_dict())
    d = res.as_dict()
    if d["eligibility"] != "applied" or d.get("interval") is None:
        return {**out, "state": "insufficient", "reason": "the difference could not be estimated: " + str(d.get("reason") or d["eligibility"])}
    lo, hi = d["interval"]
    if hi < 0:
        return {**out, "state": "alert", "reason": f"the recent mean IC {cm:.4f} is below the reference {rm:.4f} and the interval of the difference ({lo:+.4f} to {hi:+.4f}) excludes zero"}
    if cm < rm:
        return {**out, "state": "watch", "reason": f"the recent mean IC {cm:.4f} is below the reference {rm:.4f}, but the interval ({lo:+.4f} to {hi:+.4f}) includes zero"}
    return {**out, "state": "ok", "reason": f"the recent mean IC {cm:.4f} is not below the reference {rm:.4f}"}


def lineage(registry, subject_id: str, kind: str, current: dict) -> dict:
    """Link this look to the previous look at the same subject: which features were alerting then, which are alerting now, since when, and how many looks in a row. The history is the append-only table."""
    from . import diagnostics as dx

    prev = dx.latest(registry, subject_id, kind)
    now = {r["feature"] for r in current.get("features", []) if r["state"] == "alert"}
    if prev is None:
        return {"previous_record": None, "alerting_now": sorted(now), "new": sorted(now), "persisting": [], "cleared": [], "consecutive_looks": {f: 1 for f in sorted(now)}, "looks_so_far": 1}
    before = {r["feature"] for r in prev["content"].get("features", []) if r["state"] == "alert"}
    streak = (prev["content"].get("lineage") or {}).get("consecutive_looks", {})
    return {"previous_record": prev["seq"], "previous_at": prev["created"], "alerting_now": sorted(now), "new": sorted(now - before), "persisting": sorted(now & before), "cleared": sorted(before - now),
            "consecutive_looks": {f: int(streak.get(f, 1)) + 1 if f in before else 1 for f in sorted(now)}, "looks_so_far": int((prev["content"].get("lineage") or {}).get("looks_so_far", 1)) + 1}


def report(registry, exp_id: str, *, spec: DriftSpec | None = None, as_of=None) -> dict:
    """The drift report of a saved experiment: input drift of its features against the window its latest model was fitted on, and the drift of its settled predictive performance on the validation segment.
    Features are not outcomes, so input drift may look at dates beyond the validation segment; performance reads settled development outcomes only."""
    from . import diagnostics as dx
    from . import evaluation as ev
    from . import experiments as xp
    from .backtest import label_days
    from .targets import forward_return

    spec = spec or DriftSpec()
    cfg, ds, train, _ = xp.load_run(registry, exp_id)
    folds = [f for f in train.folds if not f.get("skipped") and f.get("fit_cutoff")]
    if not folds:
        return {"version": VERSION, "experiment_id": exp_id, "eligibility": "insufficient", "reason": "the experiment records no fold with a fit cutoff (it was saved before fold lineage was kept)"}
    fit_cutoff = pd.Timestamp(folds[-1]["fit_cutoff"])
    when = pd.Timestamp(as_of) if as_of is not None else ds.dates[-1]
    out = {"version": VERSION, "experiment_id": exp_id, "spec": asdict(spec), "spec_hash": spec.hash(), "as_of": str(when.date()), "model": cfg["model"]["kind"], "eligibility": "applied"}
    out.update(inputs=feature_drift(ds.X, spec, fit_cutoff=fit_cutoff, as_of=when))
    scores = train.scores_wide(ds)
    fwd = forward_return(ds.prices, int(ds.horizon), int(ds.lag)).reindex(index=ds.dates, columns=ds.assets)
    seg = label_days(scores.index, train.plan.labels()) == "validation"                            # development outcomes only
    ic = ev.rank_ic_wide(scores[seg.to_numpy()], fwd.loc[scores.index[seg.to_numpy()]]).dropna()
    pos = ds.dates.get_indexer(ic.index)
    label_end = pd.Series([ds.dates[min(p + int(ds.lag) + int(ds.horizon), len(ds.dates) - 1)] for p in pos], index=ic.index)
    out["performance"] = performance_drift(ic, label_end, spec, horizon=int(ds.horizon), fit_cutoff=fit_cutoff, as_of=when)
    out["performance"]["scope"] = "settled decisions on the validation segment: a development statistic, not live monitoring"
    out["lineage"] = lineage(registry, exp_id, "drift_report", out["inputs"])
    out["note"] = "an alert is a proposal to look, not an action: nothing here retrains, rolls back or promotes anything"
    return dx._jsonable(out)                                                                       # noqa: SLF001
