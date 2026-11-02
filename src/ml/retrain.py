"""Retraining triggers, proposals, validation and rollback (spec v3, Release 8, P3-02). Optional, and autonomous retraining is off unless a policy turns it on.

A drift alert, a performance alert, a schedule or a person may *trigger* a retraining. A trigger is not a retraining: it is evaluated against a declared policy and either becomes a **proposal** or is
**suppressed with its reason**, and both are recorded (the history is the append-only ``analysis_records`` table, kind ``retrain_event``):

* a trigger the policy does not enable is suppressed; a subject suspended after repeated failures is suppressed until a person documents a recovery;
* a second trigger while a proposal is open is a **duplicate** and is linked to it, not queued again;
* a new proposal within the **cooldown** of the last one is suppressed, and so is one with too little new data or too short a training history;
* with ``autonomous=False`` (the default) a proposal needs a named person's approval before anything is trained.

A retraining runs in **development mode**: the trainer must report that the protected segments were not read (the registry's own holdout status is consulted when the challenger is in it), or the result is
rejected as having consumed them. The challenger then faces the incumbent in a **nested validation** (a paired comparison on the validation days both have); a challenger demonstrably worse is rejected. A
completed retraining registers a challenger **as a candidate and never promotes it**: stage changes go through the gates and, for production, a recorded human approval. A failure leaves the incumbent in
force, counts toward a suspension, and is recovered only by a documented decision. Rollback delegates to the governance module, which keeps the version history.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

from . import diagnostics as dx

VERSION = "retrain-v1"
KIND = "retrain_event"
TRIGGERS = ("drift_alert", "performance_alert", "calibration_alert", "schedule", "manual")
TERMINAL = ("completed", "failed", "rejected")


class RetrainError(ValueError):
    """A request the retraining procedure refuses."""


@dataclass(frozen=True)
class RetrainPolicy:
    name: str
    triggers: tuple = ("drift_alert", "performance_alert", "manual")
    cooldown_days: float = 21.0
    min_new_rows: int = 63
    min_train_days: int = 504
    autonomous: bool = False
    require_nested_validation: bool = True
    max_failures: int = 3

    def __post_init__(self):
        if not self.name or any(t not in TRIGGERS for t in self.triggers) or not self.triggers:
            raise RetrainError(f"a policy has a name and triggers drawn from {list(TRIGGERS)}")
        if self.cooldown_days < 0 or self.min_new_rows < 1 or self.min_train_days < 1 or self.max_failures < 1:
            raise RetrainError("cooldown_days >= 0, min_new_rows >= 1, min_train_days >= 1 and max_failures >= 1")

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:16]


def _now(now: float | None) -> float:
    return time.time() if now is None else float(now)


def _log(registry, subject: str, event: str, actor: str, now: float | None, **content) -> int:
    return dx.store(registry, KIND, subject, {"event": event, "at_epoch": _now(now), "actor": actor, "version": VERSION, **content}, actor=actor)


def events(registry, subject: str) -> list[dict]:
    """Every retraining event of a subject, oldest first."""
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = c.execute("SELECT seq, content, created FROM analysis_records WHERE kind=? AND subject_id=? ORDER BY seq", (KIND, subject)).fetchall()
    return [{"seq": r["seq"], "created": r["created"], **json.loads(r["content"])} for r in rows]


def status(registry, subject: str) -> dict:
    """The state of a subject's retraining history: the open proposal, the last proposal's time, consecutive failures and whether it is suspended."""
    open_p, last_p, fails, suspended, last_done = None, None, 0, False, None
    for e in events(registry, subject):
        ev = e["event"]
        if ev == "proposal":
            open_p, last_p = e["seq"], e["at_epoch"]
        elif ev in TERMINAL and open_p is not None and e.get("proposal") == open_p:
            open_p = None
            if ev == "failed":
                fails += 1
            elif ev == "completed":
                fails, last_done = 0, e["at_epoch"]
        elif ev == "suspended":
            suspended = True
        elif ev == "recovered":
            suspended, fails = False, 0
    return {"subject": subject, "open_proposal": open_p, "last_proposal_epoch": last_p, "consecutive_failures": fails, "suspended": suspended, "last_completed_epoch": last_done}


def trigger(registry, subject: str, policy: RetrainPolicy, *, kind: str, detail: dict | None = None, rows_since_fit: int, train_days: int, actor: str = "system", now: float | None = None) -> dict:
    """Record a trigger and decide whether it becomes a proposal. Every trigger leaves a row, proposal or not."""
    if kind not in TRIGGERS:
        raise RetrainError(f"a trigger is one of {list(TRIGGERS)}")
    st = status(registry, subject)
    t = _now(now)
    base = {"trigger": kind, "trigger_detail": detail or {}, "policy": policy.name, "policy_hash": policy.hash()}
    reason = None
    if kind not in policy.triggers:
        reason = f"the policy '{policy.name}' does not enable '{kind}' triggers"
    elif st["suspended"]:
        reason = f"suspended after {st['consecutive_failures']} consecutive failed retrainings: a person must document a recovery first"
    elif st["open_proposal"] is not None:
        reason = f"duplicate: proposal {st['open_proposal']} is still open"
    elif st["last_proposal_epoch"] is not None and (t - st["last_proposal_epoch"]) < policy.cooldown_days * 86400:
        reason = f"cooldown: the last proposal was {(t - st['last_proposal_epoch']) / 86400:.1f} days ago and the policy waits {policy.cooldown_days:g}"
    elif rows_since_fit < policy.min_new_rows:
        reason = f"only {rows_since_fit} new dated rows since the last fit; the policy needs {policy.min_new_rows}"
    elif train_days < policy.min_train_days:
        reason = f"{train_days} days of history; the policy needs {policy.min_train_days} to retrain at all"
    if reason is not None:
        seq = _log(registry, subject, "suppressed", actor, now, reason=reason, duplicate_of=st["open_proposal"] if reason.startswith("duplicate") else None, **base)
        return {"outcome": "suppressed", "reason": reason, "event": seq, "proposal": None}
    seq = _log(registry, subject, "proposal", actor, now, requires_approval=not policy.autonomous, rows_since_fit=int(rows_since_fit), train_days=int(train_days), **base)
    return {"outcome": "proposed", "proposal": seq, "requires_approval": not policy.autonomous, "reason": "an alert is a proposal: nothing is trained until the policy allows it" if not policy.autonomous else "autonomous retraining is enabled by this policy"}


def approve(registry, subject: str, proposal: int, *, actor: str, note: str, now: float | None = None) -> dict:
    if not actor or not str(note).strip():
        raise RetrainError("an approval names who approves and why")
    if status(registry, subject)["open_proposal"] != proposal:
        raise RetrainError(f"proposal {proposal} is not the open proposal of {subject}")
    seq = _log(registry, subject, "approved", actor, now, proposal=proposal, note=str(note).strip())
    return {"outcome": "approved", "event": seq}


def reject(registry, subject: str, proposal: int, *, actor: str, note: str, now: float | None = None) -> dict:
    if not actor or not str(note).strip():
        raise RetrainError("a rejection names who rejects and why")
    if status(registry, subject)["open_proposal"] != proposal:
        raise RetrainError(f"proposal {proposal} is not the open proposal of {subject}")
    seq = _log(registry, subject, "rejected", actor, now, proposal=proposal, reason="declined by a person: " + str(note).strip())
    return {"outcome": "rejected", "event": seq}


def _holdout_status(registry, result: dict) -> str | None:
    cid = result.get("challenger_id")
    try:
        rep = registry.report(cid) if cid else None
    except Exception:                                                                            # the challenger is not in this registry: take what the trainer reports
        rep = None
    return ((rep or {}).get("holdout") or {}).get("status") or result.get("holdout_status")


def start(registry, subject: str, proposal: int, policy: RetrainPolicy, *, incumbent_id: str, trainer, validator=None, actor: str = "system", now: float | None = None) -> dict:
    """Run the retraining of an open proposal. ``trainer(subject)`` returns ``{"challenger_id", "holdout_status"}``; ``validator(incumbent_id, challenger_id)`` returns the nested validation
    (``{"eligibility", "difference": {"interval": [lo, hi]}}``). The challenger is never promoted."""
    st = status(registry, subject)
    if st["open_proposal"] != proposal:
        raise RetrainError(f"proposal {proposal} is not the open proposal of {subject}")
    approved = any(e["event"] == "approved" and e.get("proposal") == proposal for e in events(registry, subject))
    if not policy.autonomous and not approved:
        raise RetrainError("autonomous retraining is off: a person must approve the proposal first")
    _log(registry, subject, "started", actor, now, proposal=proposal, incumbent=incumbent_id)
    try:
        result = trainer(subject)
        hs = _holdout_status(registry, result)
        if hs != "development":
            raise RetrainError(f"the retraining did not stay in development mode (holdout status: {hs or 'not recorded'}): a retraining cannot read the protected segments")
        if result.get("no_challenger"):                                                          # the search ran and nothing was distinguishably better: the incumbent stays in force
            seq = _log(registry, subject, "rejected", actor, now, proposal=proposal, reason=result.get("reason") or "no challenger was distinguishably better than the incumbent", evidence=result.get("evidence_ref"))
            return {"outcome": "rejected", "event": seq, "reason": result.get("reason") or "no challenger was distinguishably better than the incumbent: the incumbent stays in force", "promoted": False,
                    "evidence_ref": result.get("evidence_ref")}
        challenger = result["challenger_id"]
        verdict = None
        if policy.require_nested_validation:
            if validator is None:
                raise RetrainError("the policy requires a nested validation and none was given")
            verdict = validator(incumbent_id, challenger)
            d = (verdict or {}).get("difference") or {}
            if verdict.get("eligibility") != "applied" or d.get("interval") is None:
                raise RetrainError("the nested validation could not be estimated: " + str(verdict.get("reason") or verdict.get("eligibility")))
            if d["interval"][1] < 0:
                seq = _log(registry, subject, "rejected", actor, now, proposal=proposal, challenger=challenger, reason="the challenger is demonstrably worse than the incumbent on the validation days both have", validation=verdict)
                return {"outcome": "rejected", "event": seq, "reason": "the challenger is worse than the incumbent: the incumbent stays in force", "promoted": False}
    except Exception as exc:                                                                     # a failed retraining leaves the incumbent in force
        fails = st["consecutive_failures"] + 1
        seq = _log(registry, subject, "failed", actor, now, proposal=proposal, error=f"{type(exc).__name__}: {exc}", consecutive_failures=fails)
        out = {"outcome": "failed", "event": seq, "error": str(exc), "incumbent_kept": True, "consecutive_failures": fails}
        if fails >= policy.max_failures:
            _log(registry, subject, "suspended", actor, now, reason=f"{fails} consecutive failed retrainings", proposal=proposal)
            out["suspended"] = True
        return out
    seq = _log(registry, subject, "completed", actor, now, proposal=proposal, challenger=challenger, incumbent=incumbent_id, validation=verdict, stage="candidate", promoted=False, evidence=result.get("evidence_ref"),
               promotion="a challenger is registered as a candidate; promotion goes through the gates and, for production, a recorded human approval")
    return {"outcome": "completed", "event": seq, "challenger_id": challenger, "promoted": False, "validation": verdict, "evidence_ref": result.get("evidence_ref")}


def recover(registry, subject: str, *, actor: str, note: str, now: float | None = None) -> dict:
    """Lift a suspension. It needs a named person and a written note: the failures stay in the history."""
    if not actor or not str(note).strip():
        raise RetrainError("a recovery names who decided it and what was found")
    if not status(registry, subject)["suspended"]:
        raise RetrainError(f"{subject} is not suspended")
    return {"outcome": "recovered", "event": _log(registry, subject, "recovered", actor, now, note=str(note).strip())}


def rollback(registry, subject: str, from_version: str, to_version: str, *, reason: str, actor: str, now: float | None = None) -> dict:
    """Return to an earlier strategy version through the governance module (which keeps both versions and the reason) and record the event."""
    from . import governance as gv

    if not str(reason).strip() or not actor:
        raise RetrainError("a rollback names who decided it and why")
    res = gv.rollback(registry, from_version, to_version, reason, actor)
    seq = _log(registry, subject, "rollback", actor, now, from_version=from_version, to_version=to_version, reason=str(reason).strip())
    return {"outcome": "rolled_back", "event": seq, "governance": res}


def nested_validation(registry, incumbent_id: str, challenger_id: str, **kw) -> dict:
    """The default validator: the paired comparison of two saved experiments on the validation days both have (``comparison.compare_saved``), read through its net-Sharpe difference."""
    from . import comparison as cmp_

    out = cmp_.compare_saved(registry, incumbent_id, challenger_id, **kw)
    if out.get("eligibility") != "applied":
        return {"eligibility": out.get("eligibility", "not_comparable"), "reason": out.get("reason")}
    return {"eligibility": "applied", "difference": out["metrics"]["net_sharpe"]["difference"], "ic": out["metrics"]["ic"]["difference"], "segment": out["segment"]}
