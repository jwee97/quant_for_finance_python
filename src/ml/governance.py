"""Governance of a deployed strategy (spec v3, Release 5, P1-25).

What is deployed is not a trained model. It is a *strategy version*: the fitted artifact together with the signal rules, the execution settings, the dataset and feature definitions and the risk limits that
turn its scores into a book. Changing any of them changes the identity, so evidence gathered for one version (prospective commitments, a monitoring history, a human approval) never silently carries over
to another.

* ``register_version`` stores the immutable identity of an experiment under a given set of risk limits (idempotent; the same inputs give the same ``sv-...`` id);
* ``RiskLimits`` are documented numbers (gross, net, position size, turnover, drawdown, daily loss, staleness) with a content hash; ``commit_forecast`` is the service function every caller (API, CLI,
  forward test) uses to put a forecast in the ledger, and it refuses a book that breaches the limits or a version that is halted or rolled back - recording the refusal;
* ``monitor`` reads drift, the live information coefficient and the realised returns against the limits, appends a ``monitoring_check`` event and halts the version on a hard breach;
* ``halt`` / ``resume`` / ``rollback`` and ``document_recovery`` keep the operating procedures in an append-only audit (``governance_events``);
* ``approve_deployment`` records the separate human decision, tied to the version it approved.

Nothing here places an order, promotes a model, or decides that a strategy is good: Gate C (``r5_gates``) reads this record and a person decides.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import prospective as pr
from . import provenance as pv

VERSION = "governance-v1"
LIMITS = {
    "max_gross": ("float", 0.0, 20.0, "gross exposure: the sum of absolute weights"),
    "max_net_abs": ("float", 0.0, 20.0, "absolute net exposure: the absolute sum of weights"),
    "max_position": ("float", 0.0, 5.0, "the largest absolute weight of one position"),
    "max_turnover": ("float", 0.0, 40.0, "the sum of absolute weight changes of one rebalance"),
    "max_drawdown": ("float", 0.0, 1.0, "the strategy is halted when the drawdown of the monitored returns is deeper than this"),
    "max_daily_loss": ("float", 0.0, 1.0, "the strategy is halted when a one-day loss is larger than this"),
    "max_staleness_days": ("int", 1, 365, "new commitments are refused and the strategy is flagged when the latest decision is older than this many calendar days"),
}
PSI_ALERT = 0.25
STATES = ("active", "halted", "rolled_back")


class GovernanceError(ValueError):
    pass


class LimitBreach(GovernanceError):
    def __init__(self, breaches: list[dict]):
        self.breaches = breaches
        super().__init__("; ".join(b["message"] for b in breaches))


class StrategyHalted(GovernanceError):
    pass


def stamp(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() if now is None else float(now)))


# ------------------------------------------------------------------------------------------------------------------ risk limits
@dataclass(frozen=True)
class RiskLimits:
    """Documented limits. A limit left at ``None`` is not set (and is said not to be set); there are no defaults."""

    limits: dict = field(default_factory=dict)
    name: str = "risk-limits"
    note: str = ""

    def __post_init__(self):
        for k, v in self.limits.items():
            if k not in LIMITS:
                raise GovernanceError(f"unknown risk limit '{k}'; the limits are {sorted(LIMITS)}")
            if v is None:
                continue
            kind, lo, hi = LIMITS[k][0], LIMITS[k][1], LIMITS[k][2]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= float(v) <= hi) or (kind == "int" and float(v) != int(v)):
                raise GovernanceError(f"{k} must be a{'n integer' if kind == 'int' else ' number'} in [{lo}, {hi}]")

    def set(self) -> dict:
        return {k: v for k, v in sorted(self.limits.items()) if v is not None}

    def content(self) -> dict:
        return {"schema": VERSION, "name": self.name, "note": self.note, "limits": self.set()}

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.content(), sort_keys=True).encode()).hexdigest()[:16]


def limits_from_content(content: dict) -> RiskLimits:
    return RiskLimits(dict(content.get("limits") or {}), content.get("name", "risk-limits"), content.get("note", ""))


def check_weights(limits: RiskLimits, weights: dict, previous: dict | None = None) -> list[dict]:
    """The limits a book breaches (an empty list when it breaches none). ``previous`` is the book it replaces, for the turnover limit."""
    lim = limits.set()
    w = {k: float(v) for k, v in (weights or {}).items()}
    out: list[dict] = []

    def breach(name, value, bound):
        out.append({"limit": name, "value": float(value), "bound": float(bound), "message": f"{name}: {value:.4g} exceeds the limit {bound:g}"})

    gross, net = sum(abs(v) for v in w.values()), sum(w.values())
    if "max_gross" in lim and gross > lim["max_gross"] + 1e-9:
        breach("max_gross", gross, lim["max_gross"])
    if "max_net_abs" in lim and abs(net) > lim["max_net_abs"] + 1e-9:
        breach("max_net_abs", abs(net), lim["max_net_abs"])
    if "max_position" in lim and w and max(abs(v) for v in w.values()) > lim["max_position"] + 1e-9:
        breach("max_position", max(abs(v) for v in w.values()), lim["max_position"])
    if "max_turnover" in lim:
        prev = {k: float(v) for k, v in (previous or {}).items()}
        turn = sum(abs(w.get(k, 0.0) - prev.get(k, 0.0)) for k in set(w) | set(prev))
        if turn > lim["max_turnover"] + 1e-9:
            breach("max_turnover", turn, lim["max_turnover"])
    return out


def check_returns(limits: RiskLimits, returns: pd.Series) -> list[dict]:
    """Drawdown and one-day-loss breaches of a daily return series (net returns of the strategy's actual book)."""
    lim = limits.set()
    r = pd.Series(returns, dtype=float).dropna()
    out: list[dict] = []
    if r.empty:
        return out
    if "max_daily_loss" in lim and float(-r.min()) > lim["max_daily_loss"] + 1e-12:
        out.append({"limit": "max_daily_loss", "value": float(-r.min()), "bound": lim["max_daily_loss"], "message": f"max_daily_loss: a one-day loss of {-r.min():.2%} on {r.idxmin()} exceeds {lim['max_daily_loss']:.2%}"})
    if "max_drawdown" in lim:
        eq = (1.0 + r).cumprod()
        dd = float((1.0 - eq / eq.cummax().clip(lower=1.0)).max())
        if dd > lim["max_drawdown"] + 1e-12:
            out.append({"limit": "max_drawdown", "value": dd, "bound": lim["max_drawdown"], "message": f"max_drawdown: a drawdown of {dd:.2%} exceeds {lim['max_drawdown']:.2%}"})
    return out


# ------------------------------------------------------------------------------------------------------------------ the audit
def record(registry, kind: str, subject_id: str, detail: dict | None, actor: str, now: float | None = None) -> int:
    with registry.transaction() as c:
        cur = c.execute("INSERT INTO governance_events (kind, subject_id, detail, actor, at) VALUES (?,?,?,?,?)",
                        (kind, subject_id, json.dumps(detail or {}, sort_keys=True, default=str), str(actor or "system"), stamp(now)))
    return int(cur.lastrowid)


def events(registry, subject_id: str | None = None, kind: str | None = None, limit: int = 200) -> list[dict]:
    q, args = "SELECT * FROM governance_events", []
    cond = []
    if subject_id:
        cond.append("subject_id=?")
        args.append(subject_id)
    if kind:
        cond.append("kind=?")
        args.append(kind)
    q += (" WHERE " + " AND ".join(cond) if cond else "") + " ORDER BY seq DESC LIMIT ?"
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute(q, args + [int(limit)])]
    for r in rows:
        r["detail"] = json.loads(r["detail"] or "{}")
    return rows


# ------------------------------------------------------------------------------------------------------------------ strategy identity
def _model_artifact(registry, experiment_id: str) -> tuple[str | None, str | None]:
    from .gates import _ancestors                                                                # noqa: PLC0415

    for owner in [experiment_id] + _ancestors(registry, experiment_id):
        path = Path(registry.root) / "experiments" / owner / "model.pkl"
        if path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()[:32], owner
    return None, None


def identity(registry, experiment_id: str, risk_limits_hash: str | None = None) -> dict:
    """The content that defines a strategy version. The platform's code hash is NOT part of it (it changes with every release and would restart the evidence); it is recorded in each commitment."""
    report = registry.report(experiment_id)
    cfg = report.get("config") or {}
    section = lambda k: pv.digest(cfg[k], 16) if cfg.get(k) is not None else None            # noqa: E731
    model_sha, owner = _model_artifact(registry, experiment_id)
    prov = pv.load(registry, experiment_id) or {}
    return {"schema": VERSION, "experiment_id": experiment_id, "model_artifact": {"sha256": model_sha, "saved_with": owner},
            "settings": {"dataset": section("dataset"), "model": section("model"), "split": section("split"), "signal": section("signal"), "backtest": section("backtest"),
                         "execution": section("execution") or "vectorized_legacy"},
            "universe": (cfg.get("dataset") or {}).get("universe"), "data": ((prov.get("data") or {}).get("data_hash")), "risk_limits": risk_limits_hash}


def version_id_of(content: dict) -> str:
    return "sv-" + pv.digest(content, 16)


def register_limits(registry, limits: RiskLimits, actor: str) -> str:
    if not str(actor or "").strip():
        raise GovernanceError("risk limits are registered by a named person")
    h = limits.content_hash()
    with registry.transaction() as c:
        if c.execute("SELECT 1 FROM risk_limits WHERE limits_hash=?", (h,)).fetchone() is None:
            c.execute("INSERT INTO risk_limits (limits_hash, content, created, actor) VALUES (?,?,?,?)", (h, json.dumps(limits.content(), sort_keys=True), stamp(), str(actor).strip()))
    return h


def get_limits(registry, limits_hash: str | None) -> RiskLimits | None:
    if not limits_hash:
        return None
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT content FROM risk_limits WHERE limits_hash=?", (limits_hash,)).fetchone()
    return limits_from_content(json.loads(row["content"])) if row else None


def register_version(registry, experiment_id: str, *, actor: str, risk_limits: RiskLimits | None = None, model_id: str | None = None) -> dict:
    """Store the identity of ``experiment_id`` under ``risk_limits`` (or under none) and return it. Idempotent. A change to the signal, execution, dataset, model or risk settings is a new version."""
    if not str(actor or "").strip():
        raise GovernanceError("a strategy version records who registered it")
    lh = register_limits(registry, risk_limits, actor) if risk_limits is not None else None
    content = identity(registry, experiment_id, lh)
    vid = version_id_of(content)
    with registry.transaction() as c:
        row = c.execute("SELECT version_id FROM strategy_versions WHERE version_id=?", (vid,)).fetchone()
        created = row is None
        if created:
            c.execute("INSERT INTO strategy_versions (version_id, experiment_id, model_id, content, risk_limits_hash, created, actor) VALUES (?,?,?,?,?,?,?)",
                      (vid, experiment_id, model_id, json.dumps(content, sort_keys=True), lh, stamp(), str(actor).strip()))
    if created:
        record(registry, "version_registered", vid, {"experiment_id": experiment_id, "risk_limits": lh, "settings": content["settings"]}, actor)
    return {"version_id": vid, "created": created, "content": content, "risk_limits_hash": lh}


def get_version(registry, version_id: str) -> dict:
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT * FROM strategy_versions WHERE version_id=?", (version_id,)).fetchone()
    if row is None:
        raise GovernanceError(f"unknown strategy version '{version_id}'")
    d = dict(row)
    d["content"] = json.loads(d["content"])
    return d


def versions_of(registry, experiment_id: str) -> list[dict]:
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute("SELECT * FROM strategy_versions WHERE experiment_id=? ORDER BY rowid", (experiment_id,))]
    for r in rows:
        r["content"] = json.loads(r["content"])
    return rows


def current_version(registry, experiment_id: str) -> dict | None:
    v = versions_of(registry, experiment_id)
    return v[-1] if v else None


# ------------------------------------------------------------------------------------------------------------------ operating state
def state(registry, version_id: str) -> dict:
    """``active``, ``halted`` or ``rolled_back``, from the audit (the last halt, resume or rollback event decides)."""
    get_version(registry, version_id)
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT seq, kind, detail, actor, at FROM governance_events WHERE subject_id=? AND kind IN ('halted','resumed','rolled_back') ORDER BY seq DESC LIMIT 1", (version_id,)).fetchone()
    if row is None or row["kind"] == "resumed":
        return {"state": "active", "since": None if row is None else row["at"], "reason": ""}
    detail = json.loads(row["detail"] or "{}")
    return {"state": "halted" if row["kind"] == "halted" else "rolled_back", "since": row["at"], "reason": detail.get("reason", ""), "actor": row["actor"], "event_seq": row["seq"]}


def halt(registry, version_id: str, reason: str, actor: str, now: float | None = None) -> dict:
    if not str(reason or "").strip() or not str(actor or "").strip():
        raise GovernanceError("a halt records who halted the strategy and why")
    if state(registry, version_id)["state"] == "rolled_back":
        raise GovernanceError("this version was rolled back: it is out of service and is not halted again")
    seq = record(registry, "halted", version_id, {"reason": reason.strip()}, actor, now)
    return {"version_id": version_id, "state": "halted", "event_seq": seq}


def resume(registry, version_id: str, note: str, actor: str) -> dict:
    """Return a halted version to service. A rolled-back version cannot be resumed: register a new version."""
    if not str(note or "").strip() or not str(actor or "").strip():
        raise GovernanceError("resuming records who resumed the strategy and why")
    st = state(registry, version_id)["state"]
    if st == "rolled_back":
        raise GovernanceError("a rolled-back version cannot be resumed: register a new version (its evidence starts again)")
    if st == "active":
        return {"version_id": version_id, "state": "active", "unchanged": True}
    seq = record(registry, "resumed", version_id, {"note": note.strip()}, actor)
    return {"version_id": version_id, "state": "active", "event_seq": seq}


def rollback(registry, from_version: str, to_version: str, reason: str, actor: str) -> dict:
    """Take ``from_version`` out of service for good and name the version that takes over. The target must exist and be active; nothing is deleted."""
    if not str(reason or "").strip() or not str(actor or "").strip():
        raise GovernanceError("a rollback records who rolled back and why")
    if from_version == to_version:
        raise GovernanceError("a version cannot be rolled back to itself")
    get_version(registry, from_version)
    if state(registry, to_version)["state"] != "active":
        raise GovernanceError(f"{to_version} is not active: roll back to a version that is in service")
    seq = record(registry, "rolled_back", from_version, {"to": to_version, "reason": reason.strip()}, actor)
    return {"from_version": from_version, "to_version": to_version, "state": "rolled_back", "event_seq": seq}


def document_recovery(registry, version_id: str, procedure: str, actor: str) -> int:
    """The written procedure for recovering this version (what to halt, whom to tell, which version to return to, how to reconcile). Gate C needs one."""
    if len(str(procedure or "").strip()) < 20 or not str(actor or "").strip():
        raise GovernanceError("a recovery procedure is written text of some substance, recorded with the person who wrote it")
    get_version(registry, version_id)
    return record(registry, "recovery_plan", version_id, {"procedure": procedure.strip()}, actor)


# ------------------------------------------------------------------------------------------------------------------ the service function for forecasts
def commit_forecast(registry, version_id: str, *, decision_ts, universe_version: str, horizon: int, forecasts: dict, weights: dict | None = None, previous_weights: dict | None = None, actor: str,
                    now: float | None = None, **ledger) -> dict:
    """Put a forecast (and the book it implies) in the prospective ledger. Refused, with the refusal recorded, when the version is halted or rolled back or the book breaches a documented limit. The API, the CLI
    and the forward test all come through here, so the rules cannot differ by entry point."""
    v = get_version(registry, version_id)
    st = state(registry, version_id)
    if st["state"] != "active":
        record(registry, "commitment_refused", version_id, {"reason": f"the version is {st['state']}: {st['reason']}", "decision_ts": str(decision_ts)}, actor, now)
        raise StrategyHalted(f"{version_id} is {st['state']} ({st['reason'] or 'no reason recorded'}): new commitments are refused")
    limits = get_limits(registry, v["risk_limits_hash"])
    if limits is not None:
        bad = check_weights(limits, weights or {}, previous_weights) if weights is not None else []
        age = limits.set().get("max_staleness_days")
        if age is not None and ledger.get("data_cutoff") is not None:
            lag_days = (pd.Timestamp(decision_ts) - pd.Timestamp(ledger["data_cutoff"])).days
            if lag_days > age:
                bad.append({"limit": "max_staleness_days", "value": float(lag_days), "bound": float(age), "message": f"max_staleness_days: the data is {lag_days} days older than the decision (limit {age})"})
        if bad:
            record(registry, "limit_breach_refused", version_id, {"breaches": bad, "decision_ts": str(decision_ts)}, actor, now)
            raise LimitBreach(bad)
    return pr.commit(registry, strategy_version=version_id, decision_ts=decision_ts, universe_version=universe_version, horizon=horizon, forecasts=forecasts, weights=weights, actor=actor, now=now, **ledger)


# ------------------------------------------------------------------------------------------------------------------ monitoring
def monitor(registry, version_id: str, *, drift: dict | None = None, live_ic: dict | None = None, returns: pd.Series | None = None, last_decision=None, actor: str = "system",
            now: float | None = None, halt_on_breach: bool = True) -> dict:
    """One monitoring check of a version: feature and score drift (PSI), the live rank IC against the backtest's, the realised returns against the drawdown and daily-loss limits and the age of the latest
    decision against the staleness limit. The check is appended to the audit with its inputs; a hard breach of a documented limit halts the version (unless ``halt_on_breach`` is False). Alerts that are not
    limit breaches (drift, a decaying IC) are reported and do not halt: they are for a person."""
    v = get_version(registry, version_id)
    limits = get_limits(registry, v["risk_limits_hash"])
    alerts: list[str] = []
    breaches: list[dict] = []
    if drift:
        if drift.get("n_alert"):
            alerts.append(f"{drift['n_alert']} feature(s) have drifted (PSI above {PSI_ALERT})")
        if drift.get("prediction_psi") is not None and np.isfinite(drift["prediction_psi"]) and drift["prediction_psi"] > PSI_ALERT:
            alerts.append(f"the distribution of the scores has shifted (PSI {drift['prediction_psi']:.2f})")
    if live_ic and live_ic.get("z_vs_backtest") is not None and live_ic["z_vs_backtest"] < -2.0:
        alerts.append(f"the live rank IC is {abs(live_ic['z_vs_backtest']):.1f} standard errors below the backtest's")
    if limits is not None:
        if returns is not None:
            breaches += check_returns(limits, returns)
        age_limit = limits.set().get("max_staleness_days")
        if age_limit is not None and last_decision is not None:
            age = (pd.Timestamp(time.time() if now is None else now, unit="s").normalize() - pd.Timestamp(last_decision).normalize()).days
            if age > age_limit:
                breaches.append({"limit": "max_staleness_days", "value": float(age), "bound": float(age_limit), "message": f"max_staleness_days: the latest decision is {age} days old (limit {age_limit})"})
    status = "breach" if breaches else "alert" if alerts else "ok"
    detail = {"status": status, "alerts": alerts, "breaches": breaches, "inputs": {"drift": bool(drift), "live_ic": bool(live_ic), "returns": None if returns is None else int(pd.Series(returns).dropna().shape[0]),
                                                                                  "last_decision": None if last_decision is None else str(last_decision)},
              "limits": None if limits is None else limits.content_hash()}
    seq = record(registry, "monitoring_check", version_id, detail, actor, now)
    halted = False
    if breaches and halt_on_breach and state(registry, version_id)["state"] == "active":
        halt(registry, version_id, "; ".join(b["message"] for b in breaches), "monitoring", now)
        halted = True
    return {"version_id": version_id, "status": status, "alerts": alerts, "breaches": breaches, "event_seq": seq, "halted": halted, "limits_set": limits is not None}


def last_monitoring(registry, version_id: str) -> dict | None:
    rows = events(registry, version_id, "monitoring_check", limit=1)
    return rows[0] if rows else None


# ------------------------------------------------------------------------------------------------------------------ the separate human decision
def approve_deployment(registry, version_id: str, *, actor: str, note: str, stage: str = "production", gate_seqs: dict | None = None) -> dict:
    """Record the human decision to deploy this version. It is tied to the version's identity: a later change to the signal, execution or risk settings is a different version and this approval does not
    cover it. It does not move a model (``Registry.promote`` does, and re-checks the gates); it is the person's name and reason."""
    if not str(actor or "").strip() or not str(note or "").strip():
        raise GovernanceError("a deployment approval records the person and the reason")
    v = get_version(registry, version_id)
    seq = record(registry, "deployment_approved", version_id, {"stage": stage, "note": note.strip(), "gate_seqs": gate_seqs or {}, "identity": pv.digest(v["content"], 16)}, actor)
    return {"version_id": version_id, "stage": stage, "event_seq": seq}


def approvals(registry, version_id: str) -> list[dict]:
    return events(registry, version_id, "deployment_approved")


def summary(registry, version_id: str) -> dict:
    v = get_version(registry, version_id)
    limits = get_limits(registry, v["risk_limits_hash"])
    plans = events(registry, version_id, "recovery_plan", limit=1)
    return {"version": v, "state": state(registry, version_id), "risk_limits": None if limits is None else limits.content(), "last_monitoring": last_monitoring(registry, version_id),
            "recovery_plan": plans[0] if plans else None, "approvals": approvals(registry, version_id), "events": events(registry, version_id, limit=50)}
