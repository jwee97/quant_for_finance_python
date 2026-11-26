"""What each research gate means, what is holding it, and what to do next, derived from the gate evaluation (``gates.evaluate``) and never able to change it.

``gates.evaluate`` says, item by item, whether the evidence is ``ok``, ``failed``, ``blocked``, ``stale`` or ``missing``. This module adds the part a person needs to act:

* a plain statement of the gate's purpose, what passing means and what it does NOT mean;
* the **kind** of obstacle, because the remedy differs: ``missing_evidence`` (nothing was produced yet), ``execution_failure`` (the check could not run or ended abnormally), ``validation_failure`` (it ran and the
  result did not meet the requirement), ``stale_evidence`` (it was good, the thing it checked has changed), ``policy_restriction`` (a rule or a missing decision blocks it), ``contaminated_evidence`` (the protected
  outcomes were read outside a frozen plan: this cannot be undone), ``insufficient_evidence`` (it ran but is too small to conclude);
* the primary next action as DATA (an endpoint and a body the page may call, or a pointer to a workflow), never a command line;
* what cannot be completed immediately and why.

Nothing here marks a gate passed. A contaminated holdout stays contaminated: the guidance for it is to plan a new look on outcomes nobody has read, which may not exist yet.
"""

from __future__ import annotations

import time

from . import verification

KINDS = ("missing_evidence", "execution_failure", "validation_failure", "stale_evidence", "policy_restriction", "contaminated_evidence", "insufficient_evidence")

GATE_TEXT = {
    "A": {"title": "Algorithm Verified",
          "purpose": "Is the algorithm implemented the way its documentation says it is?",
          "evidence": ["mathematical identities and deterministic tests", "invariants and numerical benchmarks", "architecture and output contracts", "serialization and train/inference parity", "regression tests"],
          "passing_means": "the code does what its specification says, on the tests the repository registers for this model family, against the current source.",
          "not_implied": "Nothing about profit. A correct algorithm can still lose money; passing Gate A says only that the arithmetic and the contracts hold."},
    "B": {"title": "Strategy Validated",
          "purpose": "Is there credible out-of-sample evidence for this strategy, under a plan fixed before the outcomes were seen?",
          "evidence": ["a frozen plan and a holdout nobody had read", "chronological validation", "realistic transaction costs", "statistical significance and its intervals", "multiple-testing control", "benchmark comparison", "robustness checks"],
          "passing_means": "one look at untouched outcomes, taken under a frozen plan, supports the result by the criteria of the adopted validation policy.",
          "not_implied": "That the strategy will work in the future, or that it is cheap to run. A holdout that was read outside a plan stays contaminated: it is never reclassified as untouched."},
    "C": {"title": "Operationally Ready",
          "purpose": "Is there enough prospective evidence and are the safeguards in place to run this strategy for its intended use?",
          "evidence": ["settled prospective (paper-trading) forecasts", "documented risk and exposure limits", "monitoring and drift checks that are current", "data freshness", "execution assumptions", "failure handling and rollback", "operational controls"],
          "passing_means": "the strategy version has a track record made after it was committed, and the controls to stop and recover it exist.",
          "not_implied": "Permission to trade automatically. Production is a separate, recorded human decision."},
}
WHY_NOT_NOW = {
    "contaminated_evidence": "The outcomes this experiment was evaluated on were read outside a frozen plan. That cannot be undone; a new look needs outcomes nobody has read, which means data that did not exist (or was not touched) when the earlier looks were taken.",
    "prospective": "Prospective evidence is made only by waiting: forecasts are committed before their outcomes exist and settle as time passes. It cannot be produced by running a computation.",
}


def classify(item: dict, verification_row: dict | None = None) -> str | None:
    """The kind of obstacle an item represents (``None`` for an item that is met)."""
    status, reason, iid = item["status"], (item.get("reason") or "").lower(), item["id"]
    if status in ("ok", "not_applicable"):
        return None
    if status == "stale":
        return "stale_evidence"
    if iid == "frozen_plan":
        return "contaminated_evidence" if status == "blocked" and ("outside a frozen plan" in reason or "not untouched" in reason or "re-use" in reason) else "missing_evidence" if status == "missing" else "validation_failure" if status == "failed" else "policy_restriction"
    if iid.startswith("verification:"):
        if status == "missing":
            return "missing_evidence"
        if status == "blocked":
            return "execution_failure"                                                                  # tests were skipped (a missing package): what they check was not verified here
        if status == "failed":
            details = " ".join(d.get("message", "") for d in ((verification_row or {}).get("failure_details") or []))
            return "execution_failure" if any(k in (reason + " " + details.lower()) for k in ("timed out", "not in this deployment", "exited with status", "no test was collected")) else "validation_failure"
    if "policy" in reason and status in ("blocked", "missing"):
        return "policy_restriction"
    if status == "missing":
        return "missing_evidence"
    if status == "failed":
        return "validation_failure"
    if "too few" in reason or "insufficient" in reason or "sufficient" in reason:
        return "insufficient_evidence"
    return "policy_restriction"


def _age_text(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    return f"{int(seconds)} s ago" if seconds < 90 else f"{int(seconds / 60)} min ago" if seconds < 5400 else f"{seconds / 3600:.1f} h ago" if seconds < 172800 else f"{int(seconds / 86400)} days ago"


def annotate(res: dict, registry, experiment_id: str) -> dict:
    """``res`` (from ``gates.evaluate``/``decide``) with, on each gate, ``title``, ``purpose``, ``passing_means``, ``not_implied``, ``display_status``, ``blocker`` (kind and text of the worst obstacle), ``obstacles``
    (every non-met item with its kind), ``evidence_summary``, ``last_verified`` and ``next_action``. The gate's own ``status`` is untouched. Returns ``res``."""
    cur = {r["requirement_id"]: r for r in verification.current(registry)}
    live = dict(verification.LIVE)
    pre = verification.preflight()
    for name, g in res["gates"].items():
        text = GATE_TEXT[name]
        items = g["items"]
        for i in items:
            req = i["id"].split(":", 1)[1] if i["id"].startswith("verification:") else None
            row = cur.get(req) if req else None
            i["kind"] = classify(i, row)
            if req:
                i["requirement_id"] = req
                i["verification"] = None if row is None else {k: row.get(k) for k in ("status", "stale", "created", "counts", "seconds", "source_fingerprint", "age_seconds", "tests", "failure_details", "check_id")}
                i["action"] = {"id": "run_requirement", "label": f"Run {req}", "endpoint": "/api/ml/verification/run", "body": {"requirements": [req]}} if i["status"] != "ok" and pre["can_run"] else None
        bad = [i for i in items if i["kind"]]
        met = len(items) - len(bad)
        order = {"contaminated_evidence": 0, "validation_failure": 1, "execution_failure": 2, "policy_restriction": 3, "stale_evidence": 4, "insufficient_evidence": 5, "missing_evidence": 6}
        bad.sort(key=lambda i: order[i["kind"]])
        running = bool(live.get("running")) and name == "A" and any(i.get("requirement_id") in (live.get("requirements") or []) for i in items)
        display = "running" if running else ("stale" if g["status"] == "pending" and g["stale"] and not any(i["kind"] in ("missing_evidence",) for i in items) else g["status"])
        g.update(title=text["title"], purpose=text["purpose"], passing_means=text["passing_means"], not_implied=text["not_implied"], evaluates=text["evidence"], display_status=display,
                 obstacles=[{"item": i["id"], "label": i["label"], "kind": i["kind"], "reason": i["reason"]} for i in bad],
                 blocker=None if not bad and not g["dependencies"] else {"kind": bad[0]["kind"] if bad else "policy_restriction", "text": (bad[0]["label"] + ": " + bad[0]["reason"]) if bad else g["reasons"][0] if g["reasons"] else ""},
                 evidence_summary=f"{met} of {len(items)} checks met" + (f"; depends on Gate {' and '.join(g['dependencies'])}" if g["dependencies"] else ""))
        g["last_verified"] = _last_verified(name, items, cur)
        g["next_action"] = _next_action(name, g, bad, pre, res, registry, experiment_id)
    res["verification_runner"] = {"can_run": pre["can_run"], "blockers": pre["blockers"], "live": live if live.get("running") else None}
    return res


def _last_verified(name: str, items: list[dict], cur: dict) -> dict | None:
    if name != "A":
        return None
    rows = [cur[i["requirement_id"]] for i in items if i.get("requirement_id") in cur and cur[i["requirement_id"]]["created"]]
    if not rows:
        return None
    newest = max(rows, key=lambda r: r["created"])
    return {"at": newest["created"], "ago": _age_text(newest.get("age_seconds")), "requirement": newest["requirement_id"]}


def _next_action(name: str, g: dict, bad: list[dict], pre: dict, res: dict, registry, experiment_id: str) -> dict:
    if name == "A":
        todo = [i["requirement_id"] for i in g["items"] if i.get("requirement_id") and i["status"] != "ok"]
        if g["display_status"] == "running":
            return {"id": "wait", "label": "Verification is running", "detail": "Watch the progress below."}
        if todo and not pre["can_run"]:
            b = pre["blockers"][0]
            return {"id": "fix_environment", "label": "Resolve what stops verification from running", "detail": f"{b['message']}. {b['fix']}.", "blockers": pre["blockers"]}
        if todo:
            return {"id": "run_gate_a", "label": f"Run Gate A verification ({len(todo)} requirement{'s' if len(todo) != 1 else ''})", "endpoint": "/api/ml/verification/run", "body": {"requirements": todo},
                    "detail": "Runs only the registered tests for this model family, on the server."}
        other = [i for i in g["items"] if i["status"] != "ok"]
        if other:
            return {"id": "explain", "label": "Resolve the remaining items", "detail": f"{other[0]['label']}: {other[0]['reason']}"}
        return {"id": "none", "label": "Nothing to do", "detail": "Every Gate A check is current and passed."}
    if name == "B":
        if any(i["kind"] == "contaminated_evidence" for i in bad):
            return {"id": "plan_new_look", "label": "Plan a new frozen evaluation", "opens": "plan", "endpoint": "/api/ml/plan-readiness", "detail": WHY_NOT_NOW["contaminated_evidence"]}
        if g["dependencies"] and res["gates"]["A"]["status"] != "passed":
            return {"id": "gate_a_first", "label": "Finish Gate A first", "detail": "Gate B requires Gate A."}
        if bad:
            return {"id": "explain", "label": "Produce the missing evidence", "detail": f"{bad[0]['label']}: {bad[0]['reason']}"}
        return {"id": "none", "label": "Nothing to do", "detail": "Every Gate B check is met."}
    if any(i["kind"] == "policy_restriction" and "policy" in i["reason"].lower() for i in bad):
        return {"id": "adopt_policy", "label": "Adopt a validation policy", "opens": "policy", "detail": "Gate C cannot be judged until a person adopts the thresholds it is judged by."}
    if g["dependencies"] and any(res["gates"][d]["status"] != "passed" for d in g["dependencies"]):
        return {"id": "earlier_gates_first", "label": "Finish Gates A and B first", "detail": "Gate C builds on them."}
    return {"id": "prospective_setup", "label": "Set up prospective monitoring", "opens": "prospective", "endpoint": "/api/ml/prospective-readiness", "detail": WHY_NOT_NOW["prospective"]}


# ------------------------------------------------------------------------------------------------------------------ workflows behind the actions
def plan_readiness(registry, experiment_id: str) -> dict:
    """What a new frozen plan for this experiment's history would need, from the ledger: which stretches of dates are already consumed (by an evaluation or by an exposure), the first date a new look could
    protect, and the minimum amount of evidence the adopted policy asks for. Reads the registry only; creates nothing."""
    import pandas as pd

    from . import policy as pol

    exp = registry.get_experiment(experiment_id)
    consumed, assets = [], set()
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute("SELECT * FROM holdout_access_log WHERE range_start IS NOT NULL ORDER BY seq")]
        plans = [dict(r) for r in c.execute("SELECT plan_id, study_id, range_start, range_end, untouched FROM holdout_reservations")]
    for r in rows:
        if r["access_kind"] == "exposure" or (r["access_kind"] in ("evaluate", "evaluate_reuse") and r["status"] == "claimed"):
            consumed.append({"kind": r["access_kind"], "plan_id": r["plan_id"], "start": r["range_start"], "end": r["range_end"], "at": r["at"], "purpose": r["purpose"]})
    last = max((pd.Timestamp(x["end"]) for x in consumed), default=None)
    adopted = pol.adopted(registry)
    need = None if adopted is None else {k: adopted[0].content().get("criteria", adopted[0].content()).get(k) for k in ("min_segment_days", "prospective_min_days", "prospective_min_settled")}
    today = pd.Timestamp.today().normalize()
    first = None if last is None else (last + pd.Timedelta(days=1))
    unseen = None if first is None else max(int((today - first).days) + 1, 0)
    steps = ["Collect market data for dates after the last consumed date (the data sources keep updating; this is the only way to obtain outcomes nobody has read).",
             "Write the plan before looking: the candidates, their parameters, the metrics, the multiple-testing treatment and the decision rule (POST /api/ml/evaluation-plan).",
             "Reserve the unseen dates and freeze the plan (POST /api/ml/evaluation-plan/freeze). Reserving dates that overlap an earlier look is recorded, and the plan is then not 'untouched'.",
             "Evaluate once (POST /api/ml/evaluate). A second evaluation of the same plan is recorded as a re-use and cannot pass."]
    return {"experiment_id": experiment_id, "model_kind": exp.get("model_kind"), "consumed": consumed[-20:], "n_consumed": len(consumed), "reservations": plans[-20:], "consumed_through": None if last is None else str(last.date()),
            "first_untouched_date": None if first is None else str(first.date()), "calendar_days_since": unseen, "minimum_evidence": need, "policy_adopted": adopted is not None, "today": str(today.date()),
            "can_start_now": bool(unseen) and (need is None or need.get("min_segment_days") is None or unseen >= 1),
            "note": "Calendar days since the last consumed date are an upper bound on the unseen trading days; the data actually available decides.", "steps": steps, "cannot_complete_now": WHY_NOT_NOW["contaminated_evidence"]}


def prospective_readiness(registry, experiment_id: str) -> dict:
    """What Gate C needs and how far this experiment is from it: the strategy version, the documented limits, the recovery plan, the monitoring check, the prospective counts against the policy minimums."""
    from . import governance as gv, policy as pol, prospective as pr

    adopted = pol.adopted(registry)
    v = gv.current_version(registry, experiment_id)
    counts = None if v is None else pr.counts(registry, v["version_id"])
    need = None if adopted is None else {k: adopted[0].content().get("criteria", adopted[0].content()).get(k) for k in ("prospective_min_days", "prospective_min_settled", "monitoring_max_age_days")}
    steps = [{"step": "Adopt a validation policy", "done": adopted is not None, "how": "a person adopts the thresholds (the platform implies none)"},
             {"step": "Register a strategy version", "done": v is not None, "how": "POST /api/ml/governance/version: model, dataset, signal, execution and risk settings become its identity"},
             {"step": "Document risk limits", "done": bool(v and gv.get_limits(registry, v["risk_limits_hash"]) is not None and gv.get_limits(registry, v["risk_limits_hash"]).set()), "how": "part of the version; changing them restarts the evidence"},
             {"step": "Write a recovery and rollback procedure", "done": bool(v and gv.events(registry, v["version_id"], "recovery_plan", limit=1)), "how": "POST /api/ml/governance/recovery"},
             {"step": "Run the forward test on new data each period", "done": bool(counts and counts.get("settled")), "how": "POST /api/ml/forward/step: forecasts are committed before outcomes exist and settle later"},
             {"step": "Keep monitoring current", "done": bool(v and gv.last_monitoring(registry, v["version_id"])), "how": "POST /api/ml/governance/monitor"}]
    return {"experiment_id": experiment_id, "version": None if v is None else v["version_id"], "counts": counts, "minimum_evidence": need, "steps": steps, "policy_adopted": adopted is not None,
            "cannot_complete_now": WHY_NOT_NOW["prospective"] + (" The evidence minimums are set by the adopted policy." if adopted else " No policy is adopted, so no minimum is defined yet.")}
