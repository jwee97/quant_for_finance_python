"""Monitoring of calibration, the prediction distribution, uncertainty and model disagreement, each on its own (spec v3, Release 8, P3-11). Optional; nothing runs unless it is asked for.

Four things can go wrong with a forecaster that reports its own uncertainty, and they are different failures that need different responses, so each has its own check, state and threshold:

* **Calibration.** Does the interval the model states (say 80%) contain the settled outcome that often? Only outcomes that have settled count.
* **Prediction distribution.** Has the distribution of the point forecasts moved away from the one in the reference window (PSI and the dependence-aware KS of ``drift``)?
* **Uncertainty.** Has the level of the stated uncertainty (the predictive standard deviation) risen or fallen against the reference?
* **Disagreement.** Has the spread between the ensemble's members changed? With fewer than ``min_members`` members present the disagreement is not defined.

**Missing is not zero.** A missing uncertainty estimate (no distribution, a NaN standard deviation, a member that did not report) is ``unavailable`` or ``degraded`` with its reason, serialised as ``null``,
and it can never make a check look healthy: a monitor whose every check is unavailable is itself ``unavailable``, not ``ok``. What happens when a member is missing is part of the policy, declared up
front: ``degrade`` (continue on the members that are present, and say so) or ``alert``.

**Escalation.** A check that alerts is an ``alert``; the level it is reported at rises with the number of consecutive looks at which it alerted (``info`` -> ``warning`` -> ``critical``), because a rule
looked at every day fires by chance some day and persistence is the evidence that it is not chance. Nothing here acts: an alert is for a person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import drift as dr

VERSION = "uncertainty-monitor-v1"
CHECKS = ("calibration", "distribution", "uncertainty", "disagreement")
STATES = ("ok", "watch", "alert", "degraded", "unavailable", "not_applicable")
LEVELS = ("info", "warning", "critical")


@dataclass(frozen=True)
class MonitorPolicy:
    interval_level: float = 0.8              # the nominal coverage of the stated interval
    coverage_tolerance: float = 0.10         # an absolute gap that is watched; twice this alerts
    uncertainty_ratio_watch: float = 1.5     # recent mean predictive std over the reference, either way
    uncertainty_ratio_alert: float = 2.0
    disagreement_ratio_watch: float = 1.5
    disagreement_ratio_alert: float = 2.0
    min_settled: int = 100                   # settled forecasts needed to judge calibration
    min_obs: int = 100                       # observations needed in each window for a distributional check
    min_members: int = 2                     # fewer members than this and disagreement is undefined
    on_missing_member: str = "degrade"       # degrade | alert
    escalate_after: int = 2                  # consecutive alerting looks per level

    def __post_init__(self):
        if not 0 < self.interval_level < 1 or self.coverage_tolerance <= 0 or self.uncertainty_ratio_watch <= 1 or self.uncertainty_ratio_alert < self.uncertainty_ratio_watch:
            raise ValueError("interval_level in (0, 1), coverage_tolerance > 0 and 1 < watch <= alert for the ratios")
        if self.disagreement_ratio_watch <= 1 or self.disagreement_ratio_alert < self.disagreement_ratio_watch or self.min_members < 2 or self.escalate_after < 1:
            raise ValueError("disagreement ratios 1 < watch <= alert, at least two members and escalate_after >= 1")
        if self.on_missing_member not in ("degrade", "alert"):
            raise ValueError("on_missing_member is degrade or alert")


def _unavailable(check: str, reason: str, state: str = "unavailable", **extra) -> dict:
    return {"check": check, "state": state, "value": None, "reference": None, "reason": reason, **extra}


def calibration(lower, upper, outcome, settled, policy: MonitorPolicy) -> dict:
    """Realised coverage of the stated interval on the settled outcomes, against its nominal level. A missing bound or outcome is not a miss and not a hit: it is left out and counted."""
    lo, hi, y, ok_settled = (np.asarray(v, dtype=float) for v in (lower, upper, outcome, settled))
    usable = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y) & (ok_settled > 0)
    n_missing = int((~(np.isfinite(lo) & np.isfinite(hi))).sum())
    if usable.sum() < policy.min_settled:
        return _unavailable("calibration", f"{int(usable.sum())} settled forecasts with an interval; {policy.min_settled} are needed" + (f" ({n_missing} forecasts had no interval)" if n_missing else ""),
                            n=int(usable.sum()), missing_interval=n_missing)
    hit = (y[usable] >= lo[usable]) & (y[usable] <= hi[usable])
    cov = float(hit.mean())
    se = float(np.sqrt(policy.interval_level * (1 - policy.interval_level) / usable.sum()))
    gap = cov - policy.interval_level
    state = "alert" if abs(gap) > 2 * policy.coverage_tolerance else "watch" if abs(gap) > policy.coverage_tolerance else "ok"
    side = "too narrow: the interval misses more often than it says" if gap < 0 else "too wide: the interval is more cautious than it says"
    return {"check": "calibration", "state": state, "value": cov, "reference": policy.interval_level, "n": int(usable.sum()), "missing_interval": n_missing, "binomial_se": se,
            "reason": f"realised coverage {cov:.3f} against {policy.interval_level:.2f}" + ("" if state == "ok" else f": {side}")}


def level_check(check: str, recent, reference, watch: float, alert: float, policy: MonitorPolicy, what: str) -> dict:
    """The mean of a non-negative quantity (predictive standard deviation, member spread) in the recent window against the reference window, as a ratio; the ratio can be too high or too low."""
    r, e = np.asarray(recent, dtype=float), np.asarray(reference, dtype=float)
    n_missing = int((~np.isfinite(r)).sum() + (~np.isfinite(e)).sum())
    r, e = r[np.isfinite(r)], e[np.isfinite(e)]
    if len(r) < policy.min_obs or len(e) < policy.min_obs:
        return _unavailable(check, f"{len(r)} recent and {len(e)} reference observations of {what}; {policy.min_obs} are needed in each" + (f" ({n_missing} were missing)" if n_missing else ""), n_recent=len(r), n_reference=len(e),
                            missing=n_missing)
    me, mr = float(e.mean()), float(r.mean())
    if me <= 0:
        return _unavailable(check, f"the reference mean of {what} is not positive, so a ratio is undefined", n_recent=len(r), n_reference=len(e))
    ratio = mr / me
    worst = max(ratio, 1.0 / ratio) if ratio > 0 else float("inf")
    state = "alert" if worst >= alert else "watch" if worst >= watch else "ok"
    return {"check": check, "state": state, "value": ratio, "reference": 1.0, "recent_mean": mr, "reference_mean": me, "n_recent": len(r), "n_reference": len(e), "missing": n_missing,
            "reason": f"the mean {what} is {ratio:.2f} times the reference" + ("" if state == "ok" else (" (higher: the model is less sure)" if ratio > 1 else " (lower: the model is more sure than it was)"))}


def distribution_check(recent, reference, policy: MonitorPolicy, spec: dr.DriftSpec | None = None) -> dict:
    """Has the distribution of the point forecasts moved? PSI (a heuristic) and the effective-sample-size KS, both required for an alert, one for a watch."""
    spec = spec or dr.DriftSpec(reference_days=max(5, policy.min_obs), recent_days=max(5, policy.min_obs), min_reference=policy.min_obs, min_recent=policy.min_obs)
    r, e = np.asarray(recent, dtype=float), np.asarray(reference, dtype=float)
    r, e = r[np.isfinite(r)], e[np.isfinite(e)]
    if len(r) < policy.min_obs or len(e) < policy.min_obs:
        return _unavailable("distribution", f"{len(r)} recent and {len(e)} reference forecasts; {policy.min_obs} are needed in each", n_recent=len(r), n_reference=len(e))
    p, ks = dr.psi(e, r), dr.ks_effective(e, r)
    shifted = ks["p_value"] is not None and ks["p_value"] < spec.ks_level
    state = "alert" if p >= spec.psi_alert and shifted else "watch" if p >= spec.psi_watch or shifted else "ok"
    return {"check": "distribution", "state": state, "value": float(p), "reference": 0.0, "ks_p_value": ks["p_value"], "n_recent": len(r), "n_reference": len(e),
            "reason": f"PSI {p:.3f} (heuristic: watch {spec.psi_watch}, alert {spec.psi_alert}); KS p-value {ks['p_value']:.3g} at effective sizes" if ks["p_value"] is not None else f"PSI {p:.3f}"}


def disagreement(member_scores: pd.DataFrame | None, policy: MonitorPolicy, split: int) -> dict:
    """The cross-member standard deviation of the forecasts, recent rows (those after ``split``) against the earlier ones. ``member_scores`` has one column per member; a column that is missing or empty is a
    missing member, handled as the policy says; with fewer than ``min_members`` present the disagreement is undefined (``unavailable``), never zero."""
    if member_scores is None or member_scores.shape[1] == 0:
        return _unavailable("disagreement", "the model has no members to disagree: it is a single model", state="not_applicable")
    present = [c for c in member_scores.columns if member_scores[c].notna().sum() >= policy.min_obs]
    missing = [c for c in member_scores.columns if c not in present]
    if len(present) < policy.min_members:
        return _unavailable("disagreement", f"only {len(present)} member(s) reported (missing: {missing or 'none'}); at least {policy.min_members} are needed for a spread", missing_members=missing)
    spread = member_scores[present].std(axis=1, ddof=1).to_numpy()
    out = level_check("disagreement", spread[split:], spread[:split], policy.disagreement_ratio_watch, policy.disagreement_ratio_alert, policy, "member spread")
    out["members"] = present
    out["missing_members"] = missing
    if missing and out["state"] in ("ok", "watch", "unavailable"):
        out["state"] = "alert" if policy.on_missing_member == "alert" else ("degraded" if out["state"] != "unavailable" else "unavailable")
        out["reason"] += f"; member(s) {missing} did not report, and the policy is '{policy.on_missing_member}'"
    return out


def escalate(state: str, consecutive: int, policy: MonitorPolicy) -> str | None:
    """The level an alert is reported at: info at first, warning after ``escalate_after`` consecutive alerting looks, critical after twice that. Only an alert has a level."""
    if state != "alert":
        return None
    return LEVELS[min(len(LEVELS) - 1, max(0, (consecutive - 1) // policy.escalate_after))]


def overall(checks: list[dict]) -> dict:
    """The monitor's own state: the worst state among the checks that could be computed; if none could, it is ``unavailable`` (never ``ok``); a degraded or partly unavailable monitor is ``degraded``."""
    order = {"ok": 0, "watch": 1, "degraded": 2, "alert": 3}
    live = [c for c in checks if c["state"] in order]
    gaps = [c for c in checks if c["state"] == "unavailable"]
    if not live:
        return {"state": "unavailable", "reason": "no check could be computed: absence of evidence is not evidence of health"}
    worst = max(live, key=lambda c: order[c["state"]])["state"]
    if gaps and worst in ("ok", "watch"):
        return {"state": "degraded", "reason": f"{len(gaps)} check(s) could not be computed ({', '.join(c['check'] for c in gaps)}); the others are {worst}"}
    return {"state": worst, "reason": f"the worst of the checks that could be computed is {worst}"}


def monitor(frame: pd.DataFrame, outcome: pd.Series, label_end: pd.Series, *, as_of, reference_end, policy: MonitorPolicy | None = None, members: pd.DataFrame | None = None, members_missing_reason: str | None = None, previous: dict | None = None) -> dict:
    """The four checks on a table of forecasts. ``frame`` has the date as the first index level and columns ``mean`` and, when the model reports them, ``std``, ``lower`` and ``upper``; ``outcome`` and
    ``label_end`` are aligned with it. Rows are split at ``reference_end`` (the reference window ends there) and only outcomes settled by ``as_of`` inform calibration."""
    policy = policy or MonitorPolicy()
    dates = pd.DatetimeIndex(frame.index.get_level_values(0))
    ref = (dates <= pd.Timestamp(reference_end))
    rec = (dates > pd.Timestamp(reference_end)) & (dates <= pd.Timestamp(as_of))
    settled = (pd.to_datetime(label_end.reindex(frame.index)).to_numpy() <= pd.Timestamp(as_of).to_datetime64()) & rec
    col = lambda c: frame[c].to_numpy(dtype=float) if c in frame.columns else np.full(len(frame), np.nan)          # noqa: E731
    checks = [calibration(col("lower")[rec], col("upper")[rec], outcome.reindex(frame.index).to_numpy(dtype=float)[rec], settled[rec].astype(float), policy)]
    if "lower" not in frame.columns or "upper" not in frame.columns:
        checks[0] = _unavailable("calibration", "the model states no interval: calibration cannot be monitored", state="not_applicable")
    checks.append(distribution_check(col("mean")[rec], col("mean")[ref], policy))
    if "std" not in frame.columns:
        checks.append(_unavailable("uncertainty", "the model states no predictive standard deviation", state="not_applicable"))
    else:
        checks.append(level_check("uncertainty", col("std")[rec], col("std")[ref], policy.uncertainty_ratio_watch, policy.uncertainty_ratio_alert, policy, "predictive standard deviation"))
    mem = None if members is None else members.reindex(frame.index)
    if mem is not None:
        checks.append(disagreement(mem.iloc[np.flatnonzero(ref | rec)], policy, int(ref.sum())))
    elif members_missing_reason:                                                                                  # an ensemble whose members' forecasts were not kept is missing them: not a single model
        checks.append(_unavailable("disagreement", members_missing_reason))
    else:
        checks.append(disagreement(None, policy, 0))
    streak = (previous or {}).get("streak", {})
    for c in checks:
        c["consecutive_alerts"] = int(streak.get(c["check"], 0)) + 1 if c["state"] == "alert" else 0
        c["level"] = escalate(c["state"], c["consecutive_alerts"], policy)
    live = [c for c in checks if c["state"] != "not_applicable"]
    return {"version": VERSION, "policy": asdict(policy), "as_of": str(pd.Timestamp(as_of).date()), "reference_end": str(pd.Timestamp(reference_end).date()), "checks": checks,
            "overall": overall(live) if live else {"state": "not_applicable", "reason": "the model has nothing to monitor here"}, "streak": {c["check"]: c["consecutive_alerts"] for c in checks},
            "note": "each check stands alone; an unavailable check is reported as null with its reason and is never read as zero uncertainty or as health"}
