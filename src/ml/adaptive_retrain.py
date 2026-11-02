"""Adaptive retraining: scheduled, diagnostic-triggered and constrained challenger-search retraining of a saved experiment (spec v3, Release 8 follow-up, P3-02).

``retrain`` holds the procedure (a trigger becomes a proposal or is suppressed with its reason, a person approves, a failure suspends, a recovery lifts it, a rollback goes through governance). This
module supplies what that procedure was missing: **what triggers a retraining**, **what a retraining trains**, and **how the result is judged**.

* **Triggers.** A *schedule* (monthly by default, which is what a daily strategy's 21-session refit is), or one of three diagnostics read from records that already exist: a *drift alert* on a feature (P3-01),
  a *performance alert* (the settled rank IC fell, P3-01) and a *calibration alert* (the stated interval no longer covers what it says, P3-11). A trigger is verified against the data or the stored record, never
  asserted by the caller, and it is only a proposal: nothing is trained until a named person approves it.
* **A constrained challenger search.** The incumbent's own configuration is the baseline of a frozen comparison card (family ``retraining``). The challengers change only how the *same* model, features, target,
  position rule and costs are trained: the **training window** (expanding, or rolling over a configurable number of days), the **refit schedule** and, if the policy lists them, nearby **hyper-parameter**
  values. Anything else is refused by the card, the number of challengers is capped, and the Holm adjustment counts every one of them.
* **Identical out-of-sample observations.** The incumbent is re-run beside its challengers on the same data and the same out-of-sample blocks (the card refuses an arm whose test blocks differ) and every
  difference is paired on the days both have; the evidence records that no day was dropped from either side. All of it is development data (the validation segment): the protected segments are never read.
* **Judgement.** Eligible arms go through the constrained selection of ``model_selection`` (pooled validation net Sharpe, eligibility gates, Pareto, then the simplest arm that is not meaningfully worse). A
  challenger is *recommended* only if the paired comparison shows it distinguishably better than the incumbent (Holm-adjusted, interval above zero, at least the policy's minimum effect) **and** it is the arm the
  selection chose. A drift alert is why the search ran; it is not evidence for the challenger, and a better historical Sharpe on data the incumbent was fitted to counts for nothing.
* **Nothing is promoted.** The outcome is a recommendation with its evidence. Versions of the incumbent and of every challenger are registered with governance (their content digests), the evidence is stored
  append-only with the policy hash, and a promotion still needs the gates and a recorded human approval; a rollback to the earlier version stays available.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import comparison as cmp_
from . import diagnostics as dx
from . import model_selection as ms
from . import retrain as rt

VERSION = "adaptive-retrain-v1"
EVIDENCE_KIND = "retrain_evidence"
FREQUENCIES = ("monthly", "weekly", "quarterly", "sessions")
MODES = ("expanding", "rolling")
MONTHLY_SESSIONS = 21                                   # trading sessions in a month: the refit interval of a daily strategy that retrains monthly
MAX_CHALLENGERS = 12
PERIOD_NAME = {"weekly": "week", "monthly": "month", "quarterly": "quarter"}


class AdaptiveError(rt.RetrainError):
    """A retraining request the adaptive procedure refuses."""


@dataclass(frozen=True)
class RetrainSchedule:
    """When a scheduled retraining is due. A calendar rule ("a new month has begun since the last fit") or a count of sessions."""

    frequency: str = "monthly"                           # monthly | weekly | quarterly | sessions
    every_sessions: int = MONTHLY_SESSIONS               # used when frequency is "sessions"

    def __post_init__(self):
        if self.frequency not in FREQUENCIES or int(self.every_sessions) < 1:
            raise AdaptiveError(f"the schedule is one of {list(FREQUENCIES)} and every_sessions is at least 1")


@dataclass(frozen=True)
class AdaptivePolicy:
    """Every threshold of the adaptive procedure, with a documented default; the policy and its hash are stored with each look."""

    name: str = "adaptive-monthly"
    schedule: RetrainSchedule = field(default_factory=RetrainSchedule)
    triggers: tuple = ("schedule", "drift_alert", "performance_alert", "calibration_alert", "manual")
    cooldown_days: float = 21.0                          # days between two proposals
    min_new_rows: int = 10                               # dated rows since the last fit that a proposal needs (a monthly schedule fires on the first session of a month)
    min_train_days: int = 504                            # history needed to retrain at all
    max_failures: int = 3                                # consecutive failed retrainings before the subject is suspended
    calibration_min_consecutive: int = 1                 # looks in a row at which the calibration check alerted
    windows: tuple = (("expanding", None), ("rolling", 1260))   # the training windows tried: (mode, rolling days); the incumbent's own is left out
    refit_every: tuple = (MONTHLY_SESSIONS,)             # sessions between refits tried (21 = monthly for daily strategies)
    param_grid: dict = field(default_factory=dict)       # nearby hyper-parameter values to try, one parameter at a time: {"alpha": [1.0, 10.0, 100.0]}
    max_challengers: int = 6                             # a hard cap on the challengers (the Holm adjustment counts all of them)
    min_effect: float = 0.0                              # the smallest Sharpe improvement that counts, in annualised Sharpe units
    level: float = 0.05                                  # the Holm-adjusted level of the paired comparison
    n_boot: int = 1000
    seed: int = 0
    selection: dict = field(default_factory=dict)        # a ``model_selection.SelectionPolicy`` mapping for the choice among the incumbent and its challengers

    def __post_init__(self):
        bad = [t for t in self.triggers if t not in rt.TRIGGERS]
        if bad or not self.triggers or not self.name:
            raise AdaptiveError(f"a policy has a name and triggers drawn from {list(rt.TRIGGERS)}")
        if self.cooldown_days < 0 or self.min_new_rows < 1 or self.min_train_days < 1 or self.max_failures < 1 or self.calibration_min_consecutive < 1:
            raise AdaptiveError("cooldown_days >= 0 and min_new_rows, min_train_days, max_failures and calibration_min_consecutive >= 1")
        for w in self.windows:
            mode, days = (tuple(w) + (None,))[:2] if isinstance(w, (list, tuple)) else (w, None)
            if mode not in MODES or (mode == "rolling" and not (isinstance(days, (int, float)) and days >= 60)):
                raise AdaptiveError(f"a window is ('expanding', None) or ('rolling', days >= 60), not {w!r}")
        if not self.windows and not self.param_grid and not self.refit_every:
            raise AdaptiveError("a policy needs at least one window, refit interval or hyper-parameter to try")
        if any(int(r) < 1 for r in self.refit_every) or not 1 <= int(self.max_challengers) <= MAX_CHALLENGERS or not 0.0 < float(self.level) < 1.0 or float(self.min_effect) < 0:
            raise AdaptiveError(f"refit intervals >= 1, 1 <= max_challengers <= {MAX_CHALLENGERS}, level in (0, 1) and min_effect >= 0")
        if int(self.n_boot) < 200:
            raise AdaptiveError("n_boot >= 200")
        for k, v in self.param_grid.items():
            if not isinstance(v, (list, tuple)) or not v:
                raise AdaptiveError(f"param_grid['{k}'] is a non-empty list of values")
        ms.policy_from(self.selection)                    # a selection policy that does not parse is refused now, not after the compute

    def hash(self) -> str:
        import json

        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:16]

    def retrain_policy(self) -> rt.RetrainPolicy:
        """The procedure's policy (triggers, cooldown, data floors, failures). Autonomous retraining stays off: a person approves every proposal."""
        return rt.RetrainPolicy(name=self.name, triggers=tuple(self.triggers), cooldown_days=self.cooldown_days, min_new_rows=self.min_new_rows, min_train_days=self.min_train_days, autonomous=False,
                                require_nested_validation=True, max_failures=self.max_failures)


def policy_from(raw: dict | None) -> AdaptivePolicy:
    """A policy from a plain mapping (the API, the command line): unknown keys are refused and ``autonomous`` cannot be enabled."""
    raw = dict(raw or {})
    if raw.pop("autonomous", False):
        raise AdaptiveError("autonomous retraining cannot be enabled here: a retraining is proposed by a trigger and started only after a person approves it")
    if raw.pop("require_nested_validation", True) is not True:
        raise AdaptiveError("a challenger is always compared with the incumbent on identical out-of-sample observations: the nested validation cannot be switched off here")
    allowed = set(AdaptivePolicy.__dataclass_fields__)
    extra = sorted(set(raw) - allowed)
    if extra:
        raise AdaptiveError(f"a retraining policy has {sorted(allowed)}, not {extra}")
    if "schedule" in raw and not isinstance(raw["schedule"], RetrainSchedule):
        sched = dict(raw["schedule"] or {})
        bad = sorted(set(sched) - set(RetrainSchedule.__dataclass_fields__))
        if bad:
            raise AdaptiveError(f"a schedule has {sorted(RetrainSchedule.__dataclass_fields__)}, not {bad}")
        raw["schedule"] = RetrainSchedule(**sched)
    for k in ("triggers", "refit_every"):
        if k in raw:
            raw[k] = tuple(raw[k])
    if "windows" in raw:
        raw["windows"] = tuple((w[0], w[1] if len(w) > 1 else None) if isinstance(w, (list, tuple)) else (w, None) for w in raw["windows"])
    return AdaptivePolicy(**raw)


# ------------------------------------------------------------------------------------------------------------------------------ the schedule
def schedule_due(dates, last_fit, as_of, schedule: RetrainSchedule | None = None) -> dict:
    """Is a scheduled retraining due at ``as_of``, given the date the current model was fitted? A calendar schedule is due when ``as_of`` falls in a later week, month or quarter than the last fit; a
    session schedule when that many sessions have passed. ``next_due`` is the first session of the next period (or the date of the next session count)."""
    schedule = schedule or RetrainSchedule()
    d = pd.DatetimeIndex(dates).sort_values().unique()
    last, now = pd.Timestamp(last_fit), pd.Timestamp(as_of)
    base = {"frequency": schedule.frequency, "last_fit": str(last.date()), "as_of": str(now.date())}
    if schedule.frequency == "sessions":
        after = d[d > last]
        since = int((after <= now).sum())
        due = since >= schedule.every_sessions
        nxt = after[schedule.every_sessions - 1] if len(after) >= schedule.every_sessions else None
        return {**base, "due": bool(due), "sessions_since_last_fit": since, "every_sessions": int(schedule.every_sessions), "next_due": None if nxt is None else str(pd.Timestamp(nxt).date()),
                "rule": f"due once {schedule.every_sessions} sessions have passed since the last fit"}
    code = {"weekly": "W", "monthly": "M", "quarterly": "Q"}[schedule.frequency]
    p_last, p_now = pd.Period(last, code), pd.Period(now, code)
    elapsed = int((p_now - p_last).n)
    start = (p_last + 1).start_time
    later = d[d >= start]
    return {**base, "due": bool(elapsed >= 1), "periods_elapsed": elapsed, "next_due": str((pd.Timestamp(later[0]) if len(later) else start).date()),
            "rule": f"due when a new {PERIOD_NAME[schedule.frequency]} has begun since the last fit (the calendar {PERIOD_NAME[schedule.frequency]} of the last fit's cutoff is over)"}


# ------------------------------------------------------------------------------------------------------------------------------ the diagnostic triggers
def diagnostic(registry, exp_id: str, kind: str, *, calibration_min_consecutive: int = 1) -> dict:
    """What the recorded diagnostics say about one trigger kind: ``{"kind", "fired", "state", "reason", "detail"}``. Nothing is computed here: a diagnostic that was never run is ``not_computed``, one that
    cannot exist for this model (no stated interval) is ``not_applicable``, and neither is read as a clean bill of health."""
    out = {"kind": kind, "fired": False, "state": "not_computed", "detail": {}}
    if kind in ("drift_alert", "performance_alert"):
        rec = dx.latest(registry, exp_id, "drift_report")
        if rec is None:
            return {**out, "reason": f"a {kind} trigger needs a recorded drift report that shows an alert: none has been computed for this experiment"}
        c = rec["content"]
        if c.get("eligibility") not in (None, "applied"):
            return {**out, "state": "insufficient", "reason": f"the latest drift report could not be computed ({c.get('reason') or c.get('eligibility')})", "detail": {"drift_record": rec["seq"]}}
        if kind == "drift_alert":
            alerting = sorted(r["feature"] for r in ((c.get("inputs") or {}).get("features") or []) if r.get("state") == "alert")
            if not alerting:
                return {**out, "state": "ok", "reason": "the latest drift report records no alerting feature, so there is no drift alert to act on", "detail": {"drift_record": rec["seq"]}}
            return {**out, "fired": True, "state": "alert", "reason": f"{len(alerting)} feature(s) alert in drift report {rec['seq']}", "detail": {"drift_record": rec["seq"], "alerting_features": alerting}}
        perf = c.get("performance") or {}
        if perf.get("state") != "alert":
            return {**out, "state": perf.get("state") or "ok", "reason": "the latest drift report records no performance alert, so there is none to act on", "detail": {"drift_record": rec["seq"]}}
        return {**out, "fired": True, "state": "alert", "reason": perf.get("reason") or "the settled rank IC fell", "detail": {"drift_record": rec["seq"], "performance": {k: perf.get(k) for k in ("state", "reason")}}}
    if kind == "calibration_alert":
        rec = dx.latest(registry, exp_id, "uncertainty_monitor")
        if rec is None:
            return {**out, "reason": "a calibration_alert trigger needs a recorded uncertainty-monitor look that shows a calibration alert: none has been computed for this experiment"}
        c = rec["content"]
        if c.get("eligibility") == "not_applicable":
            return {**out, "state": "not_applicable", "reason": c.get("reason") or "this model states no uncertainty, so its calibration cannot be monitored", "detail": {"monitor_record": rec["seq"]}}
        cal = next((x for x in c.get("checks") or [] if x.get("check") == "calibration"), None)
        if cal is None or cal.get("state") in ("unavailable", "not_applicable", None):
            return {**out, "state": (cal or {}).get("state") or "unavailable", "reason": (cal or {}).get("reason") or "the calibration check could not be computed in the latest monitor look",
                    "detail": {"monitor_record": rec["seq"]}}
        n = int(cal.get("consecutive_alerts") or 0)
        if cal["state"] != "alert" or n < int(calibration_min_consecutive):
            why = (f"the calibration check alerted in {n} look(s) in a row; the policy needs {calibration_min_consecutive}" if cal["state"] == "alert" else "the latest monitor look records no calibration alert, so there is none to act on")
            return {**out, "state": cal["state"], "reason": why, "detail": {"monitor_record": rec["seq"], "consecutive_alerts": n}}
        return {**out, "fired": True, "state": "alert", "reason": cal.get("reason") or "the stated interval no longer covers what it says",
                "detail": {"monitor_record": rec["seq"], "consecutive_alerts": n, "level": cal.get("level"), "value": cal.get("value"), "nominal": cal.get("reference")}}
    return {**out, "reason": f"{kind} has no diagnostic"}


def run_context(registry, exp_id: str) -> dict:
    """What the schedule and the data floors need from a saved experiment: its calendar, the date its latest model was fitted, the rows since then and the days of history before it."""
    from . import experiments as xp

    _, ds, train, _ = xp.load_run(registry, exp_id)
    folds = [f for f in train.folds if not f.get("skipped") and f.get("fit_cutoff")]
    if not folds:
        raise AdaptiveError("the experiment records no fold with a fit cutoff, so the data since its last fit cannot be counted")
    cutoff = pd.Timestamp(folds[-1]["fit_cutoff"])
    return {"dates": ds.dates, "last_fit": cutoff, "rows_since_fit": int((ds.dates > cutoff).sum()), "train_days": int((ds.dates <= cutoff).sum()), "data_end": ds.dates[-1]}


def evaluate_triggers(registry, exp_id: str, policy: AdaptivePolicy | None = None, *, as_of=None, refresh: bool = False, actor: str = "system") -> dict:
    """Every trigger of the policy, evaluated now: whether it fired and why. ``refresh`` runs the drift report and the uncertainty monitor first (each look is recorded, as always); without it the
    latest stored looks are read. Nothing is proposed or trained here."""
    from . import advanced as adv

    policy = policy or AdaptivePolicy()
    ctx = run_context(registry, exp_id)
    when = pd.Timestamp(as_of) if as_of is not None else ctx["data_end"]
    refreshed = {}
    if refresh:
        if any(k in policy.triggers for k in ("drift_alert", "performance_alert")):
            refreshed["drift_record"] = adv.run_drift(registry, exp_id, as_of=when, actor=actor).get("record_seq")
        if "calibration_alert" in policy.triggers:
            mon = adv.run_monitor(registry, exp_id, as_of=when, actor=actor)
            refreshed["monitor_record"] = mon.get("record_seq")
    rows = []
    for kind in rt.TRIGGERS:
        if kind == "schedule":
            due = schedule_due(ctx["dates"], ctx["last_fit"], when, policy.schedule)
            rows.append({"kind": kind, "fired": bool(due["due"]), "state": "due" if due["due"] else "not_due", "reason": due["rule"] + (f"; next due {due['next_due']}" if not due["due"] and due.get("next_due") else ""), "detail": due})
        elif kind == "manual":
            rows.append({"kind": kind, "fired": False, "state": "on_request", "reason": "a person may always ask for a retraining; it is a proposal like the others", "detail": {}})
        else:
            rows.append(diagnostic(registry, exp_id, kind, calibration_min_consecutive=policy.calibration_min_consecutive))
    for r in rows:
        r["enabled"] = r["kind"] in policy.triggers
    return {"version": VERSION, "subject": exp_id, "as_of": str(when.date()), "last_fit": str(ctx["last_fit"].date()), "rows_since_fit": ctx["rows_since_fit"], "train_days": ctx["train_days"],
            "triggers": rows, "fired": [r["kind"] for r in rows if r["fired"] and r["enabled"]], "refreshed": refreshed, "policy": policy.name, "policy_hash": policy.hash(),
            "note": "a trigger is a proposal to look: nothing is trained until a person approves it, and nothing is promoted by retraining"}


def propose(registry, exp_id: str, *, kind: str, policy: AdaptivePolicy | None = None, actor: str = "system", as_of=None, now: float | None = None, rpolicy: rt.RetrainPolicy | None = None) -> dict:
    """Verify a trigger and record it as a proposal (or as suppressed, with its reason). A schedule trigger must be due; a diagnostic trigger must be backed by the recorded alert; a manual one is accepted as
    declared. The caller cannot assert a drift, performance, calibration or schedule trigger that the data and the records do not show."""
    policy = policy or AdaptivePolicy()
    if kind not in rt.TRIGGERS:
        raise rt.RetrainError(f"a trigger is one of {list(rt.TRIGGERS)}")
    ctx = run_context(registry, exp_id)
    detail: dict = {}
    if kind == "schedule":
        due = schedule_due(ctx["dates"], ctx["last_fit"], pd.Timestamp(as_of) if as_of is not None else ctx["data_end"], policy.schedule)
        if not due["due"]:
            raise rt.RetrainError(f"the {policy.schedule.frequency} schedule is not due: the last fit was {due['last_fit']}, the data run to {due['as_of']} and the next retraining is due {due.get('next_due') or 'later'}")
        detail = {"schedule": due}
    elif kind != "manual":
        d = diagnostic(registry, exp_id, kind, calibration_min_consecutive=policy.calibration_min_consecutive)
        if not d["fired"]:
            raise rt.RetrainError(d["reason"])
        detail = d["detail"]
    return rt.trigger(registry, exp_id, rpolicy or policy.retrain_policy(), kind=kind, detail=detail, rows_since_fit=ctx["rows_since_fit"], train_days=ctx["train_days"], actor=actor, now=now)


# ------------------------------------------------------------------------------------------------------------------------------ the constrained challenger search
def _window_of(cfg: dict) -> tuple:
    sp = cfg.get("split") or {}
    return str(sp.get("window", "expanding")), (int(sp.get("rolling_days", 1260)) if sp.get("window") == "rolling" else None), sp.get("refit_every")


def challenger_changes(cfg: dict, policy: AdaptivePolicy) -> tuple[list[dict], dict]:
    """The challengers as changes laid over the incumbent's configuration, and what was left out. Windows and refit intervals are crossed; hyper-parameter values are tried one at a time; a change equal to the
    incumbent's own settings is left out; the list is cut at ``max_challengers`` in that order, and the cut is reported."""
    from .models import default_refit

    base_mode, base_days, base_refit = _window_of(cfg)
    base_refit = base_refit or default_refit((cfg.get("model") or {}).get("kind", ""))
    params = (cfg.get("model") or {}).get("params") or {}
    out, skipped = [], []
    for w in policy.windows:
        mode, days = (tuple(w) + (None,))[:2]
        for r in policy.refit_every:
            same_window = mode == base_mode and (mode == "expanding" or int(days) == (base_days or 1260))
            if same_window and int(r) == int(base_refit):
                skipped.append({"window": mode, "rolling_days": days, "refit_every": int(r), "reason": "the incumbent's own settings"})
                continue
            split = {"window": mode, "refit_every": int(r), **({"rolling_days": int(days)} if mode == "rolling" else {})}
            label = f"{mode}{'' if mode == 'expanding' else f' {int(days)}d'}, refit {int(r)}d"
            out.append({"name": f"retrain: {label}"[:80], "split": split})
    for key, values in policy.param_grid.items():
        for v in values:
            if params.get(key) == v:
                skipped.append({"param": key, "value": v, "reason": "the incumbent's own value"})
                continue
            out.append({"name": f"retrain: {key}={v}"[:80], "model": {"params": {key: v}}})
    cut = max(0, len(out) - int(policy.max_challengers))
    return out[: int(policy.max_challengers)], {"left_out": skipped, "cut_by_max_challengers": cut}


def _digest(index) -> str:
    return hashlib.sha256(np.ascontiguousarray(pd.DatetimeIndex(index).values.astype("datetime64[ns]").astype("int64")).tobytes()).hexdigest()[:16]


def _register(registry, exp_id: str, actor: str) -> dict | None:
    """A strategy version (content digest of the experiment's settings, data and model artifact) in governance, so the incumbent and every challenger can be returned to by name."""
    from . import governance as gv

    try:
        v = gv.register_version(registry, exp_id, actor=actor)
        return {"version_id": v["version_id"], "created": v["created"]}
    except Exception as exc:                                                                         # noqa: BLE001  (a version that cannot be registered is reported, not fatal to the evidence)
        return {"version_id": None, "error": f"{type(exc).__name__}: {exc}"}


def search(registry, exp_id: str, policy: AdaptivePolicy | None = None, *, proposal: int | None = None, bundle=None, config=None, builder=None, progress=None, actor: str = "system") -> dict:
    """Run the constrained challenger search for a saved experiment and judge the result. Development data only; nothing is promoted. Returns the decision and the evidence (also stored,
    append-only, as kind ``retrain_evidence``)."""
    policy = policy or AdaptivePolicy()
    progress = progress or (lambda msg: None)
    rep = registry.report(exp_id)
    cfg = rep["config"]
    from . import research_controls as rc

    study = rc.study_of(cfg)
    changes, left = challenger_changes(cfg, policy)
    if not changes:
        raise AdaptiveError("the policy leaves no challenger: every window, refit interval and hyper-parameter value equals the incumbent's own")
    label = ", ".join(c["name"].removeprefix("retrain: ") for c in changes)
    hypothesis = (f"[retraining of {exp_id}{'' if proposal is None else f', proposal {proposal}'}] A challenger trained with one of ({label}) is better than the incumbent on the same out-of-sample observations, "
                  "by more than the policy's minimum effect and with the Holm-adjusted evidence it declares")
    card = cmp_.new_card(hypothesis, "retraining", cfg, changes, study_id=study, one_change_description="how the same model is trained (its window, refit schedule or nearby hyper-parameters); the features, target, position rule and costs are unchanged",
                         primary_metric="net_sharpe", min_effect=float(policy.min_effect), level=float(policy.level), n_boot=int(policy.n_boot), seed=int(policy.seed))
    cmp_.freeze(registry, card, actor=actor)
    out = cmp_.run(registry, card["card_id"], bundle=bundle, config=config, builder=builder, progress=progress)
    results = out["results"] or {}
    exps = (out.get("evidence_artifacts") or {}).get("experiments") or {}
    base_id = exps.get("baseline")
    if base_id is None:
        raise AdaptiveError("the incumbent could not be re-run on the current data: " + "; ".join(str(f.get("detail")) for f in out.get("negative_findings", [])[:2]))
    arms = {a["name"]: a for a in results.get("candidates", [])}
    pol = ms.policy_from(policy.selection)
    cands, by_exp = [], {}
    for name, eid in [("incumbent", base_id)] + [(c["name"], exps.get(c["name"])) for c in card["candidates"]]:
        if eid is None:
            continue
        try:
            c = ms.candidate_from_experiment(registry, eid, audit=True)                              # the folds' facts are re-derived from the rows they recorded
        except Exception as exc:                                                                     # noqa: BLE001
            c = {"id": eid, "kind": None, "params": {}, "error": f"{type(exc).__name__}: {exc}"}
        if name != "incumbent" and arms.get(name, {}).get("status") != "compared":
            c = {"id": eid, "kind": c.get("kind"), "params": c.get("params", {}), "error": f"not comparable with the incumbent: {arms.get(name, {}).get('reason') or arms.get(name, {}).get('status')}"}
        c["label"] = name
        by_exp[eid] = name
        cands.append(c)
    sel = ms.select(cands, pol)
    sel = ms.store_result(registry, sel, pol, study_id=study, actor=actor) if cands else sel
    base_cand = next((c for c in cands if c["id"] == base_id), None)
    base_days = None if base_cand is None or base_cand.get("returns") is None else _digest(base_cand["returns"].index)
    rows = []
    for c in card["candidates"]:
        a, eid = arms.get(c["name"], {}), exps.get(c["name"])
        cand = next((x for x in cands if x["id"] == eid), None)
        sharpe = ((a.get("metrics") or {}).get("net_sharpe") or {})
        diff = sharpe.get("difference") or {}
        dropped = sharpe.get("dropped")
        days = None if cand is None or cand.get("returns") is None else _digest(cand["returns"].index)
        identical = bool(a.get("status") == "compared" and dropped is not None and not any(dropped.values()) and days is not None and days == base_days)
        rows.append({"name": c["name"], "experiment_id": eid, "changed": c["changed"], "status": a.get("status"), "reason": a.get("reason"), "verdict": a.get("verdict"), "verdict_reason": a.get("reason") if a.get("verdict") else None,
                     "adjusted_p_value": a.get("adjusted_p_value"), "net_sharpe": {"incumbent": sharpe.get("baseline"), "challenger": sharpe.get("candidate"), "difference": diff, "aligned": sharpe.get("aligned"),
                                                                                  "dropped": dropped}, "ic": ((a.get("metrics") or {}).get("ic") or {}).get("difference"),
                     "identical_oos": {"identical": identical, "oos_days_hash": days, "incumbent_oos_days_hash": base_days, "dropped": dropped},
                     "version": _register(registry, eid, actor) if eid else None})
    decision = decide(rows, sel)
    reproduced = _reproduced(registry, exp_id, base_cand)
    evidence = {"version": VERSION, "subject": exp_id, "proposal": proposal, "policy": asdict(policy), "policy_hash": policy.hash(), "study_id": study, "card_id": card["card_id"],
                "scope": {"mode": "development", "segments": ["validation"], "protected_segments_read": False, "note": "the incumbent and every challenger were run with stop_after=validation on the same data"},
                "incumbent": {"experiment_id": exp_id, "rerun_experiment_id": base_id, "version": _register(registry, exp_id, actor), "rerun_version": _register(registry, base_id, actor), "reproduced": reproduced},
                "window": {"incumbent": {"mode": _window_of(cfg)[0], "rolling_days": _window_of(cfg)[1], "refit_every": _window_of(cfg)[2]}, "tried": [list(w) for w in policy.windows], "refit_every": list(policy.refit_every)},
                "search_space": {"challengers": [c["name"] for c in card["candidates"]], **left, "multiplicity": f"Holm over {len(card['candidates'])} challenger(s)"},
                "challengers": rows, "selection": {"selection_id": sel.get("selection_id"), "selected": sel.get("selected"), "highest_ranked": sel.get("highest_ranked"), "counts": sel.get("counts"),
                                                   "ranking": sel.get("ranking"), "pareto_front": (sel.get("pareto") or {}).get("front"), "labels": by_exp, "simplicity_claim": ((sel.get("simplicity") or {}).get("claim") or {}).get("kind"),
                                                   "changed_by_simplicity": sel.get("changed_by_simplicity")},
                "decision": decision, "promoted": False,
                "promotion": "nothing was promoted: a challenger becomes the model in force only through the gates (staging needs gate B; production needs A, B and C and a recorded human approval). "
                             "Drift, performance and calibration alerts say why the search ran; they are not evidence for the challenger.",
                "rollback": "the incumbent's strategy version stays registered: rt.rollback / POST /api/ml/retraining/action {action: rollback} returns to it by name",
                "negative_findings": out.get("negative_findings") or []}
    evidence["record_seq"] = dx.store(registry, EVIDENCE_KIND, exp_id, dx._jsonable(evidence), policy_hash=policy.hash(), study_id=study, actor=actor)                      # noqa: SLF001
    return dx._jsonable(evidence)                                                                                                                                       # noqa: SLF001


def decide(rows: list[dict], sel: dict) -> dict:
    """The decision of a search from its arms and the constrained selection over them. A challenger is recommended only if (1) the constrained selection chose it, (2) the paired comparison shows it
    distinguishably better than the incumbent (Holm-adjusted, interval above zero, at least the minimum effect: the card's verdict) and (3) it was compared on identical out-of-sample observations. A
    drift alert, a better score on data the incumbent was fitted to or a lower complexity is not one of the conditions."""
    chosen = sel.get("selected")
    pick = next((r for r in rows if r["experiment_id"] == chosen), None)
    if pick is None:
        return {"decision": "keep_incumbent", "challenger": None, "reason": ("no arm was eligible under the selection policy" if sel.get("selection_status") != "selected" else
                "the constrained selection prefers the incumbent: it is the simplest arm that is not meaningfully worse than the highest-ranked one")}
    if pick["verdict"] != "supported_on_development":
        return {"decision": "keep_incumbent", "challenger": pick["name"], "reason": f"'{pick['name']}' was selected by the constrained selection but is not distinguishably better than the incumbent on the same out-of-sample "
                                                                                    f"observations ({pick['verdict']}): the incumbent stays"}
    if not pick["identical_oos"]["identical"]:
        return {"decision": "keep_incumbent", "challenger": pick["name"], "reason": f"'{pick['name']}' was not compared with the incumbent on identical out-of-sample observations (days dropped: {pick['identical_oos']['dropped']})"}
    return {"decision": "recommend_challenger", "challenger": pick["name"], "experiment_id": pick["experiment_id"],
            "reason": f"'{pick['name']}' is distinguishably better than the incumbent on identical out-of-sample observations (Holm-adjusted p = {pick['adjusted_p_value']:.3g}) and is the arm the constrained selection chose"}


def _reproduced(registry, exp_id: str, rerun: dict | None) -> dict:
    """Did re-running the incumbent's configuration on the current data give the incumbent's own validation returns? (If the data have been extended it need not.)"""
    if rerun is None or rerun.get("returns") is None:
        return {"checked": False, "reason": "the incumbent could not be re-run"}
    try:
        saved = ms.candidate_from_experiment(registry, exp_id)["returns"]
    except Exception as exc:                                                                         # noqa: BLE001
        return {"checked": False, "reason": f"the incumbent's own returns could not be read ({type(exc).__name__})"}
    both = pd.concat([saved, rerun["returns"]], axis=1, join="inner").dropna()
    if len(both) < 20:
        return {"checked": True, "reproduced": False, "common_days": int(len(both)), "reason": "the re-run and the saved incumbent share fewer than 20 validation days (the data or the segments have changed)"}
    diff = float(np.abs(both.iloc[:, 0] - both.iloc[:, 1]).max())
    return {"checked": True, "reproduced": bool(diff <= 1e-9 and len(saved) == len(rerun["returns"])), "common_days": int(len(both)), "max_abs_difference": diff,
            "note": "the incumbent is re-run beside its challengers so that all of them see the same data; when the data have not changed the re-run reproduces the saved incumbent"}


# ------------------------------------------------------------------------------------------------------------------------------ through the retraining procedure
def make_trainer(registry, policy: AdaptivePolicy, holder: dict, *, proposal: int | None = None, bundle=None, config=None, builder=None, progress=None, actor: str = "system"):
    """The trainer ``retrain.start`` wants: it runs the constrained search, keeps the evidence in ``holder`` and reports the recommended challenger (or that there is none) as a development result."""
    def trainer(subject: str) -> dict:
        ev = search(registry, subject, policy, proposal=proposal, bundle=bundle, config=config, builder=builder, progress=progress, actor=actor)
        holder["evidence"] = ev
        ref = {"record_seq": ev["record_seq"], "card_id": ev["card_id"], "kind": EVIDENCE_KIND}
        d = ev["decision"]
        if d["decision"] != "recommend_challenger":
            return {"no_challenger": True, "holdout_status": "development", "reason": d["reason"], "evidence_ref": ref}
        return {"challenger_id": d["experiment_id"], "holdout_status": "development", "evidence_ref": ref}
    return trainer


def make_validator(holder: dict):
    """The nested validation ``retrain.start`` wants, read from the evidence of the search: the paired net-Sharpe difference of the chosen challenger against the incumbent, on identical observations."""
    def validator(incumbent_id: str, challenger_id: str) -> dict:
        ev = holder.get("evidence") or {}
        row = next((r for r in ev.get("challengers", []) if r.get("experiment_id") == challenger_id), None)
        if row is None:
            return {"eligibility": "not_comparable", "reason": "the challenger is not one of the arms of the search"}
        d = (row["net_sharpe"] or {}).get("difference") or {}
        if d.get("eligibility") != "applied":
            return {"eligibility": d.get("eligibility", "insufficient"), "reason": d.get("reason")}
        return {"eligibility": "applied", "difference": d, "segment": "validation", "identical_oos": row["identical_oos"], "verdict": row["verdict"]}
    return validator


def start(registry, subject: str, proposal: int, policy: AdaptivePolicy | None = None, *, bundle=None, config=None, builder=None, progress=None, actor: str, now: float | None = None) -> dict:
    """Run the retraining of an open, approved proposal: the constrained search, then the procedure's own checks (development mode, a nested validation, failure counting). Returns the procedure's outcome
    with the evidence; the outcome is never a promotion."""
    policy = policy or AdaptivePolicy()
    holder: dict = {}
    out = rt.start(registry, subject, proposal, policy.retrain_policy(), incumbent_id=subject, trainer=make_trainer(registry, policy, holder, proposal=proposal, bundle=bundle, config=config, builder=builder,
                                                                                                                     progress=progress, actor=actor), validator=make_validator(holder), actor=actor, now=now)
    return {**out, "evidence": holder.get("evidence")}


# ------------------------------------------------------------------------------------------------------------------------------ versions and evidence
def evidence_records(registry, subject: str) -> list[dict]:
    """Every stored evidence record of a subject, oldest first, with its sequence number and time."""
    import json

    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = c.execute("SELECT seq, content, created FROM analysis_records WHERE kind=? AND subject_id=? ORDER BY seq", (EVIDENCE_KIND, subject)).fetchall()
    return [{**json.loads(r["content"]), "record_seq": r["seq"], "stored_at": r["created"]} for r in rows]


def latest_evidence(registry, subject: str) -> dict | None:
    rows = evidence_records(registry, subject)
    return rows[-1] if rows else None


def evidence_history(registry, subject: str) -> list[dict]:
    return [{"record_seq": c["record_seq"], "stored_at": c["stored_at"], "proposal": c.get("proposal"), "decision": (c.get("decision") or {}).get("decision"), "challenger": (c.get("decision") or {}).get("challenger"),
             "card_id": c.get("card_id"), "policy_hash": c.get("policy_hash")} for c in evidence_records(registry, subject)]


def versions(registry, subject: str) -> dict:
    """The strategy versions in the lineage of a retraining: the incumbent's own and, for every search, the re-run incumbent's and each challenger's, with the state each is in (active, halted, rolled back)."""
    from . import governance as gv

    seen, rows = set(), []

    def add(role, name, exp_id, v):
        vid = (v or {}).get("version_id")
        if vid and vid not in seen:
            seen.add(vid)
            try:
                st = gv.state(registry, vid)["state"]
            except gv.GovernanceError:
                st = "unknown"
            rows.append({"role": role, "name": name, "experiment_id": exp_id, "version_id": vid, "state": st})

    for ev in evidence_records(registry, subject):
        add("incumbent", "incumbent", ev["incumbent"]["experiment_id"], ev["incumbent"].get("version"))
        add("incumbent re-run", "incumbent re-run", ev["incumbent"]["rerun_experiment_id"], ev["incumbent"].get("rerun_version"))
        for r in ev.get("challengers", []):
            add("challenger", r["name"], r.get("experiment_id"), r.get("version"))
    return {"subject": subject, "versions": rows, "note": "each version is a content digest of the experiment's settings, data and model artifact; a rollback names two of them and keeps both"}


def rollback(registry, subject: str, from_version: str, to_version: str, *, reason: str, actor: str) -> dict:
    """Return to an earlier version of this retraining's lineage. Both versions must belong to it; governance keeps both and the reason."""
    known = {v["version_id"] for v in versions(registry, subject)["versions"]}
    for vid in (from_version, to_version):
        if vid not in known:
            raise rt.RetrainError(f"{vid} is not a version in the retraining lineage of {subject}: {sorted(known) or 'none has been registered yet'}")
    return rt.rollback(registry, subject, from_version, to_version, reason=reason, actor=actor)


def monthly_default() -> dict:
    """The configuration overrides of the default retraining for a daily strategy: refit every 21 sessions (monthly) on an expanding window. The other window is an explicit setting of ``split``."""
    return {"split": {"refit_every": MONTHLY_SESSIONS, "window": "expanding"}}

