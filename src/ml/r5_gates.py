"""The evidence providers that Release 5 adds to Gates B and C (spec v3, 8.1, 8.3, 8.4 and P1-25).

Each provider reads PERSISTED evidence and the adopted validation policy, and returns gate items (``ok`` / ``failed`` / ``missing`` / ``stale`` / ``blocked`` / ``not_applicable``):

* ``statistical_items`` (Gate B) - the stored statistical diagnostics read against the adopted policy. No adopted policy is ``blocked`` (the platform implies no threshold); diagnostics computed under a
  different policy are ``stale``; a criterion the evidence does not satisfy is ``failed`` - a correctly implemented losing strategy fails Gate B and still passes Gate A;
* ``prospective_items`` (Gate C) - committed and settled *prospective* forecasts of the current strategy version against the policy's minimums; replays never count;
* ``operational_items`` (Gate C) - the strategy version's identity, documented risk limits, a written recovery procedure, an in-service state and a monitoring check that is no older than the policy allows.

Nothing here can be set by a request: the items are computed from rows that the service functions wrote.
"""

from __future__ import annotations

import time

import pandas as pd

from . import diagnostics as dx
from . import governance as gv
from . import policy as pol
from . import prospective as pr


def _gate_item(item_id, label, status, reason="", evidence=None):
    from .gates import Item                                                                       # noqa: PLC0415 - gates imports this module lazily; the reverse import happens at call time

    return Item(item_id, label, status, reason, evidence)


def _no_policy(item_id: str, label: str):
    return [_gate_item(item_id, label, "blocked", "no validation policy has been adopted: the platform implies no Sharpe, deflated-Sharpe, PBO, p-value or paper-trading-duration threshold. A named person adopts one "
                                                    "before the evaluation (POST /api/ml/validation-policy); an illustrative example is provided and is not a recommendation")]


def _converted(items: list[dict]):
    return [_gate_item(i["id"], i["label"], i["status"], i["reason"], None if i.get("value") is None else str(i["value"])) for i in items]


RANK = {"failed": 5, "blocked": 4, "stale": 3, "missing": 2, "ok": 0, "not_applicable": 0}


def _machinery(registry, requirements: list[str], item_id: str, label: str):
    """The ml-verify evidence for the procedures a gate relies on, as one item with the worst status among them: a number from an unverified procedure is not evidence."""
    from .gates import _verification_items                                                        # noqa: PLC0415

    items = _verification_items(registry, "", requirements)
    worst = max(items, key=lambda i: RANK[i.status])
    refs = ",".join(sorted(i.evidence or i.id for i in items))
    if RANK[worst.status] == 0:
        return _gate_item(item_id, label, "ok", "verified: " + ", ".join(requirements), refs)
    return _gate_item(item_id, label, worst.status, f"{worst.label}: {worst.reason}", refs)


def statistical_items(registry, experiment_id: str, report: dict) -> list:
    adopted = pol.adopted(registry)
    if adopted is None:
        return _no_policy("statistical_diagnostics", "uncertainty, benchmarks, robustness and capacity")
    policy, adoption = adopted
    ph = policy.content_hash()
    head = _gate_item("policy", "adopted validation policy", "ok", f"{policy.name} {policy.version} adopted by {adoption['actor']} on {adoption['at']}" + (" - an ILLUSTRATIVE example, not a recommendation" if policy.illustrative else ""), ph)
    rec = dx.latest(registry, experiment_id, "diagnostics")
    if rec is None:
        return [head, _gate_item("statistical_diagnostics", "statistical diagnostics", "missing", "no diagnostics have been computed for this experiment: POST /api/ml/statistical-diagnostics/run (or quant ml-diagnostics)")]
    if rec["policy_hash"] != ph:
        return [head, _gate_item("statistical_diagnostics", "statistical diagnostics", "stale", f"the stored diagnostics were computed under policy {rec['policy_hash'] or 'none'}; the adopted policy is {ph}: compute them again", rec["content_hash"])]
    items = _converted(dx.evaluate_policy(policy, rec["content"]))
    for i in items:
        i.evidence = f"{rec['content_hash']}:{i.evidence}"
    reqs = ["P1-16", "P1-17", "P1-18", "P1-19"] + (["P1-20"] if policy.required("neighborhood_min_share_positive") else []) + (["P1-21"] if policy.required("delay_min_ratio") or policy.required("capacity_min_net_ratio") else []) \
        + (["P1-22"] if policy.required("calibration_max_ece") else [])
    out = [head, _machinery(registry, reqs, "statistical_machinery", "verified statistical procedures")] + items
    if (report.get("config") or {}).get("model", {}).get("kind") in _rl_kinds() or _kind(registry, experiment_id) in _rl_kinds():
        out.append(_seed_item(registry, experiment_id))
    return out


def _rl_kinds():
    from .gates import RL_KINDS                                                                   # noqa: PLC0415

    return RL_KINDS


def _kind(registry, experiment_id: str) -> str:
    try:
        return registry.get_experiment(experiment_id).get("model_kind") or ""
    except Exception:                                                                             # noqa: BLE001
        return ""


def _seed_item(registry, experiment_id: str):
    """A reinforcement-learning result needs evaluation over predeclared seeds (P1-23): the record must exist with at least the minimum number of seeds, and the policy variability is shown beside the
    time-series uncertainty. The best seed is never the evidence."""
    from . import multiseed as ms                                                                 # noqa: PLC0415

    rec = dx.latest(registry, experiment_id, "multiseed")
    if rec is None:
        return _gate_item("dx:seeds", "evaluation over predeclared seeds", "missing", "a reinforcement-learning result is judged over several predeclared seeds, not one run: POST /api/ml/multiseed/run (or quant ml-multiseed)")
    c = rec["content"]
    if c.get("eligibility") != "applied" or len(c["declaration"]["seeds"]) < ms.MIN_SEEDS:
        return _gate_item("dx:seeds", "evaluation over predeclared seeds", "missing", c.get("reason") or f"fewer than {ms.MIN_SEEDS} seeds", rec["content_hash"])
    s = c["summary"]
    return _gate_item("dx:seeds", "evaluation over predeclared seeds", "ok", f"{len(c['declaration']['seeds'])} seeds; median {c['metric']} {s['median']:.2f} (range {s['min']:.2f} to {s['max']:.2f}); {c.get('flag') or ''}".strip(), rec["content_hash"])


def prospective_items(registry, experiment_id: str, report: dict) -> list:
    adopted = pol.adopted(registry)
    if adopted is None:
        return _no_policy("prospective_evidence", "prospective evidence")
    policy, _ = adopted
    v = gv.current_version(registry, experiment_id)
    if v is None:
        return [_gate_item("prospective_evidence", "prospective evidence", "missing", "no strategy version is registered for this experiment, so no prospective forecast can be attributed to it: register one (POST /api/ml/governance/version) "
                                                                                       "and run the forward test")]
    counts = pr.counts(registry, v["version_id"])
    out = _converted(dx.prospective_items(policy, counts))
    for i in out:
        i.evidence = f"{v['version_id']}:{i.evidence}"
    return [_machinery(registry, ["P1-24"], "ledger_machinery", "verified prospective ledger")] + out


def operational_items(registry, experiment_id: str, report: dict) -> list:
    adopted = pol.adopted(registry)
    v = gv.current_version(registry, experiment_id)
    if v is None:
        return [_gate_item("operational_policy", "monitoring, reconciliation, recovery and rollback", "missing",
                           "no strategy version is registered: the version's identity must include the signal, execution and risk settings before operating controls can be assessed (POST /api/ml/governance/version)")]
    vid = v["version_id"]
    out = [_machinery(registry, ["P1-25"], "governance_machinery", "verified governance controls"),
           _gate_item("strategy_version", "strategy version identity", "ok", f"{vid}: model artifact, dataset, signal, execution and risk settings are part of the identity", vid)]
    limits = gv.get_limits(registry, v["risk_limits_hash"])
    out.append(_gate_item("risk_limits", "documented risk limits", "ok" if limits is not None and limits.set() else "missing",
                          "" if limits is not None and limits.set() else "this version has no documented risk limits: register a version with limits (limits are part of the identity, so this restarts the prospective evidence)",
                          v["risk_limits_hash"]))
    st = gv.state(registry, vid)
    out.append(_gate_item("operating_state", "strategy in service", "ok" if st["state"] == "active" else "failed" if st["state"] == "halted" else "blocked",
                          "" if st["state"] == "active" else f"the version is {st['state']} since {st['since']}: {st['reason']}", st.get("since")))
    plans = gv.events(registry, vid, "recovery_plan", limit=1)
    out.append(_gate_item("recovery_plan", "written recovery and rollback procedure", "ok" if plans else "missing", "" if plans else "no recovery procedure is recorded for this version (POST /api/ml/governance/recovery)",
                          str(plans[0]["seq"]) if plans else None))
    mon = gv.last_monitoring(registry, vid)
    if mon is None:
        out.append(_gate_item("monitoring", "model and data monitoring", "missing", "no monitoring check has been run for this version (the forward test step runs one; POST /api/ml/governance/monitor)"))
    else:
        max_age = None if adopted is None else adopted[0].get("monitoring_max_age_days")
        age = (pd.Timestamp(time.time(), unit="s").normalize() - pd.Timestamp(mon["at"]).normalize()).days
        if mon["detail"].get("status") in ("breach",):
            out.append(_gate_item("monitoring", "model and data monitoring", "failed", "the latest monitoring check found a breach of a documented limit: " + "; ".join(b["message"] for b in mon["detail"].get("breaches", [])), str(mon["seq"])))
        elif mon["detail"].get("status") == "alert":
            out.append(_gate_item("monitoring", "model and data monitoring", "failed", "the latest monitoring check raised alerts: " + "; ".join(mon["detail"].get("alerts", [])), str(mon["seq"])))
        elif max_age is not None and age > max_age:
            out.append(_gate_item("monitoring", "model and data monitoring", "stale", f"the latest monitoring check is {age} days old; the policy asks for one at most {max_age} days old", str(mon["seq"])))
        else:
            out.append(_gate_item("monitoring", "model and data monitoring", "ok", f"latest check {mon['at']} found no alert" + ("" if max_age is not None else "; the policy sets no maximum age"), str(mon["seq"])))
    return out
