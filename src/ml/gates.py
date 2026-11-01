"""Three independent promotion gates, computed on the server from persisted evidence (spec v3, 3.3 and the Release 3 exit conditions).

* **A — Algorithm Verified**: the numerical, causal, architecture, serialization, output-contract and train/inference checks that apply to this model family (the ``ml-verify`` evidence, current for the source tree it
  hashed), a provenance record of this experiment for the code that is now checked out, a study lineage, an output contract and a loadable inference bundle. It says the algorithm is implemented as specified. It says nothing about profit.
* **B — Strategy Validated**: Gate A, and a frozen evaluation plan whose one look at an untouched holdout produced this experiment, selection-bias controls that were applicable and met, null-control evidence at a sufficient
  replication count, and the realistic-execution, uncertainty, benchmark, robustness and capacity evidence of the later releases. Where a kind of evidence has no provider yet the gate is ``blocked`` and says so.
* **C — Operationally Ready**: Gates A and B, prospective evidence, and the operating controls (monitoring, reconciliation, recovery, rollback) under a predefined policy. This programme never grants it by itself, and it
  never places an order: it only makes a strategy eligible for a separate human decision.

A gate is ``passed``, ``failed``, ``pending`` (evidence not yet produced, or stale) or ``blocked`` (it cannot pass until something that this subject cannot change does: an inapplicable required procedure, no provider,
a failed ancestor). ``stale`` marks a decision whose evidence set no longer matches the current one (the code, the data or the policy changed): a stale gate is shown as pending, never green. Decisions are
appended to ``gate_decisions`` with the policy version and the hashes of every item, and nothing in a request can set one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .provenance import digest, source_tree_hash
from .registry import Registry, RegistryError

POLICY_VERSION = "gates-v1"
GATES = ("A", "B", "C")
ITEM_STATUS = ("ok", "failed", "missing", "stale", "blocked", "not_applicable")

RL_KINDS = {"ppo", "sac", "recurrent_ppo", "rl_ppo", "rl_sac", "rl_recurrent_ppo"}
RANKER_KINDS = {"lambdamart", "xgb_ranker", "lgbm_ranker"}
GRAPH_KINDS = {"gcn", "gat"}
DEEP_KINDS = {"lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft", "ae", "vae", "gcn", "gat"}
REGIME_KINDS = {"hmm", "gaussian_hmm", "markov_switching", "regime_ridge"}
COMMON_REQUIREMENTS = ["P0-09", "P0-11", "P0-12"] + [f"P0-{n:02d}" for n in range(13, 23)]                  # the output contracts, serialization and the causal/data/inference pipeline apply to every model


class PromotionBlocked(RegistryError):
    """A stage change the persisted evidence does not support. The message says which gate and why."""


@dataclass
class Item:
    id: str
    label: str
    status: str
    reason: str = ""
    evidence: str | None = None                       # a hash or a stable reference of the evidence behind it

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "status": self.status, "reason": self.reason, "evidence": self.evidence}


def family_requirements(kind: str) -> list[str]:
    """The ``ml-verify`` requirements that apply to a model kind: the shared pipeline ones, plus those of its family."""
    req = list(COMMON_REQUIREMENTS)
    if kind in RL_KINDS:
        req += ["P0-01", "P0-02", "P0-03", "P0-04", "P0-30"]
    if kind in RANKER_KINDS:
        req += ["P0-05"]
    if kind in GRAPH_KINDS:
        req += ["P0-06", "P0-07"]
    if kind in DEEP_KINDS:
        req += ["P0-08"]
    if kind in REGIME_KINDS:
        req += ["P0-10"]
    return sorted(set(req))


# ------------------------------------------------------------------------------------------------------------------ evidence providers
def _verification_items(registry: Registry, kind: str) -> list[Item]:
    from . import verification

    cur = {r["requirement_id"]: r for r in verification.current(registry)}
    items = []
    for req in family_requirements(kind):
        if req not in cur:
            items.append(Item(f"verification:{req}", f"requirement {req}", "missing", "this requirement has no verification registered in this checkout"))
            continue
        r = cur[req]
        ref = f"{req}:{r['check_id']}:{r['artifact_hash']}"
        if r["status"] == "pass" and r["stale"]:
            items.append(Item(f"verification:{req}", f"requirement {req}", "stale", "the source it checked has changed since the check passed: run ml-verify again", ref))
        elif r["status"] == "pass":
            items.append(Item(f"verification:{req}", f"requirement {req}", "ok", r["reason"], ref))
        elif r["status"] == "fail":
            items.append(Item(f"verification:{req}", f"requirement {req}", "failed", r["reason"], ref))
        elif r["status"] == "blocked":
            items.append(Item(f"verification:{req}", f"requirement {req}", "blocked", r["reason"], ref))
        else:
            items.append(Item(f"verification:{req}", f"requirement {req}", "missing", "never run" if r["reason"] == "never run" else r["reason"], ref))
    return items


def _provenance_item(registry: Registry, experiment_id: str) -> Item:
    from . import provenance

    rec = provenance.load(registry, experiment_id)
    if rec is None:
        return Item("provenance", "provenance record", "missing", "this experiment has no provenance record (it was saved before they existed, or was never recorded)")
    now = source_tree_hash()["sha256"]
    if rec["code"]["sha256"] != now:
        return Item("provenance", "provenance record", "stale", "the source tree has changed since this experiment ran: its recorded code identity is not the checked-out code", rec["identity"])
    return Item("provenance", "provenance record", "ok", f"identity {rec['identity'][:12]}", rec["identity"])


def _lineage_item(registry: Registry, experiment_id: str) -> Item:
    from . import research_controls as rc

    key = rc.op_key_of_experiment(registry, experiment_id)
    if key is None:
        return Item("lineage", "study lineage", "missing", "this experiment is not in any study's lineage (it ran before lineage was recorded)")
    return Item("lineage", "study lineage", "ok", f"operation {key}", key)


def _contract_item(report: dict) -> Item:
    contract = report.get("output_contract")
    if not contract:
        return Item("output_contract", "output contract", "missing", "the report has no output contract")
    return Item("output_contract", "output contract", "ok", str(contract.get("output_kind", "")), digest(contract, 16))


def _bundle_item(registry: Registry, experiment_id: str) -> Item:
    """The experiment's inference bundle, or, for one that re-uses a parent's fitted model (a re-backtest, a meta-labelled run), the nearest ancestor's: the bundle belongs to the model, not to the position rule."""
    for owner in [experiment_id] + _ancestors(registry, experiment_id):
        path = Path(registry.root) / "experiments" / owner / "model.qfb"
        if not path.exists():
            continue
        try:
            from .bundle import InferenceBundle

            header = InferenceBundle.load(path).header
        except Exception as exc:                                                                  # noqa: BLE001 - a damaged file is a failed item, not a crash
            return Item("inference_bundle", "inference bundle", "failed", f"{type(exc).__name__}: {exc}")
        return Item("inference_bundle", "inference bundle", "ok", f"schema {header['schema']['hash']}" + ("" if owner == experiment_id else f" (the fitted model of {owner})"), header["schema"]["hash"])
    return Item("inference_bundle", "inference bundle", "missing", "no model.qfb beside this experiment or any ancestor")


def _ancestors(registry: Registry, experiment_id: str) -> list[str]:
    out, seen, cur = [], {experiment_id}, experiment_id
    while True:
        try:
            parent = registry.get_experiment(cur).get("parent_id")
        except RegistryError:
            break
        if not parent or parent in seen:
            break
        out.append(parent)
        seen.add(parent)
        cur = parent
    return out


def _holdout_item(registry: Registry, report: dict) -> Item:
    h = report.get("holdout") or {}
    status = h.get("status")
    if status is None:
        return Item("frozen_plan", "frozen plan and untouched holdout", "missing", "this experiment records no holdout status (it ran before the access ledger existed)")
    if status == "development":
        return Item("frozen_plan", "frozen plan and untouched holdout", "missing", "a development run: its protected segments were not read, so there is no evaluation to count")
    if status != "evaluated_under_plan":
        return Item("frozen_plan", "frozen plan and untouched holdout", "blocked", "its test and forward outcomes were read outside a frozen plan (exploratory): it cannot be validated on them")
    from . import research_controls as rc

    try:
        st = rc.plan_state(registry, h["plan_id"], brief=True)
    except rc.PlanError as exc:
        return Item("frozen_plan", "frozen plan and untouched holdout", "failed", str(exc))
    with registry._conn() as c:                                                                  # noqa: SLF001
        claim = c.execute("SELECT seq, access_kind FROM holdout_access_log WHERE idempotency_key=? AND status='claimed'", (h.get("idempotency_key"),)).fetchone()
    if claim is None:
        return Item("frozen_plan", "frozen plan and untouched holdout", "failed", "the ledger has no evaluation event for this run")
    if not (st["state"] in ("consumed", "frozen") and st["reservation"]):
        return Item("frozen_plan", "frozen plan and untouched holdout", "failed", f"the plan is {st['state']}")
    if claim["access_kind"] == "evaluate_reuse" or not st["reservation"]["untouched"]:
        return Item("frozen_plan", "frozen plan and untouched holdout", "blocked", "the holdout was not untouched (earlier looks at the same history, or a re-use): this evaluation is not an unbiased claim", st["content_hash"])
    return Item("frozen_plan", "frozen plan and untouched holdout", "ok", f"plan {h['plan_id']} evaluated once (event {claim['seq']})", f"{st['content_hash']}:{claim['seq']}")


def _selection_item(registry: Registry, experiment_id: str, report: dict) -> Item:
    from . import research_controls as rc

    study = rc.study_of_experiment(registry, experiment_id) or (report.get("config") or {}).get("study")
    if not study:
        return Item("selection", "selection-bias controls", "missing", "no study is recorded for this experiment")
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute("SELECT * FROM selection_evidence WHERE study_id=? AND (subject IS NULL OR subject=?) ORDER BY id", (study, experiment_id))]
    if not rows:
        return Item("selection", "selection-bias controls", "missing", "no selection test has been run for this study: run POST /api/ml/statistical-diagnostics")
    latest = {}
    for r in rows:
        latest[r["method"]] = r
    refs = ",".join(f"{m}:{r['id']}" for m, r in sorted(latest.items()))
    applied = [r for r in latest.values() if r["status"] == "applied"]
    statistic = {r["method"]: json.loads(r["statistic"] or "{}") for r in applied}
    if not applied:
        why = "; ".join(f"{r['method']}: {r['status']} ({(r['reason'] or '')[:80]})" for r in latest.values())
        return Item("selection", "selection-bias controls", "blocked", f"no selection procedure was applicable to this study, and a procedure that was not applied does not pass: {why}", refs)
    failing = [m for m, s in statistic.items() if s.get("passes") is False]
    if failing:
        return Item("selection", "selection-bias controls", "failed", f"the selection control(s) {', '.join(failing)} did not support the result", refs)
    if not any(s.get("passes") is True for s in statistic.values()):
        return Item("selection", "selection-bias controls", "missing", "the applied procedures state no pass criterion", refs)
    return Item("selection", "selection-bias controls", "ok", "applied: " + ", ".join(sorted(statistic)), refs)


def _null_item(registry: Registry, experiment_id: str) -> Item:
    from . import research_controls as rc

    study = rc.study_of_experiment(registry, experiment_id)
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute("SELECT * FROM null_studies WHERE state='done' AND study_id=? ORDER BY rowid DESC", (study,))] if study else []
    if not rows:
        return Item("null_controls", "null-control false-promotion rate", "missing", "no completed null study exists for this experiment's study: POST /api/ml/null-controls/run")
    res = json.loads(rows[0]["result"] or "{}")
    ref = f"{rows[0]['null_id']}:{rows[0]['spec_hash']}"
    if not res.get("sufficient"):
        return Item("null_controls", "null-control false-promotion rate", "blocked", f"the latest null study ({rows[0]['null_id']}) has too few replications to control the error rate: it validates the plumbing only. {res.get('verdict', '')}", ref)
    if res.get("controlled") is False:
        return Item("null_controls", "null-control false-promotion rate", "failed", res.get("verdict", "the false-promotion rate exceeds the predeclared tolerance"), ref)
    return Item("null_controls", "null-control false-promotion rate", "ok", res.get("verdict", ""), ref)


def _no_provider(item_id: str, label: str, release: str, what: str):
    def provider(registry: Registry, experiment_id: str, report: dict) -> Item:
        return Item(item_id, label, "blocked", f"{what} has no provider in this checkout ({release} of the programme implements it): the gate cannot pass without it")
    return provider


@dataclass(frozen=True)
class GatePolicy:
    """What each gate requires. ``b_providers`` and ``c_providers`` map an item id to a function ``(registry, experiment_id, report) -> Item``; later releases replace the placeholders."""

    version: str = POLICY_VERSION
    a_requires_provenance: bool = True
    b_providers: dict = field(default_factory=lambda: {
        "realistic_net_oos": _no_provider("realistic_net_oos", "realistic net out-of-sample evidence", "Release 4", "the executable-trading and reconciled-accounting evidence"),
        "statistical_diagnostics": _no_provider("statistical_diagnostics", "uncertainty, benchmarks, robustness and capacity", "Release 5", "the statistical-validity diagnostics"),
    })
    c_providers: dict = field(default_factory=lambda: {
        "prospective_evidence": _no_provider("prospective_evidence", "prospective evidence", "Release 5", "committed, settled prospective forecasts"),
        "operational_policy": _no_provider("operational_policy", "monitoring, reconciliation, recovery and rollback", "Release 8", "the operating controls"),
    })

    def with_providers(self, b: dict | None = None, c: dict | None = None, version: str | None = None) -> "GatePolicy":
        return replace(self, b_providers={**self.b_providers, **(b or {})}, c_providers={**self.c_providers, **(c or {})}, version=version or self.version)


DEFAULT_POLICY = GatePolicy()


# ------------------------------------------------------------------------------------------------------------------ evaluation
def _status_of(items: list[Item], dependencies: list[tuple[str, dict]] = ()) -> tuple[str, bool]:
    """``(gate status, stale)`` from the items: ``failed`` beats ``blocked`` beats ``pending``; stale or missing items leave the gate ``pending``; only all-ok (or justified not-applicable) passes."""
    st = [i.status for i in items] + [("ok" if d["status"] == "passed" and not d["stale"] else "failed" if d["status"] == "failed" else "blocked" if d["status"] == "blocked" else "missing") for _, d in dependencies]
    stale = "stale" in st
    if "failed" in st:
        return "failed", stale
    if "blocked" in st:
        return "blocked", stale
    if "missing" in st or "stale" in st:
        return "pending", stale
    return "passed", False


def evaluate(registry: Registry, experiment_id: str, policy: GatePolicy | None = None) -> dict:
    """Compute the three gates for an experiment from the persisted evidence. Pure with respect to the registry: nothing is written (``decide`` persists)."""
    policy = policy or DEFAULT_POLICY
    exp = registry.get_experiment(experiment_id)
    try:
        report = registry.report(experiment_id)
    except Exception:                                                                             # noqa: BLE001 - an experiment whose report cannot be read has no evidence
        report = {}
    kind = exp.get("model_kind") or ""
    a_items = _verification_items(registry, kind)
    if policy.a_requires_provenance:
        a_items += [_provenance_item(registry, experiment_id), _lineage_item(registry, experiment_id), _contract_item(report), _bundle_item(registry, experiment_id)]
    if exp.get("status") != "done":
        a_items.append(Item("experiment_status", "experiment finished", "failed", f"the experiment's status is {exp.get('status')}"))
    ancestors = _ancestors(registry, experiment_id)
    inherited = []
    for anc in ancestors:
        try:
            a_anc = [i for i in _verification_items(registry, registry.get_experiment(anc).get("model_kind") or "") if i.status == "failed"]
        except RegistryError:
            continue
        if a_anc:
            inherited.append(Item(f"ancestor:{anc}", f"ancestor {anc}", "blocked", f"an ancestor experiment has failed algorithm checks ({a_anc[0].id}): a critical failure blocks its descendants"))
    a_items += inherited[:1]
    gate_a = _gate("A", a_items, [], policy)
    b_items = [_holdout_item(registry, report), _selection_item(registry, experiment_id, report), _null_item(registry, experiment_id)] + [p(registry, experiment_id, report) for p in policy.b_providers.values()]
    gate_b = _gate("B", b_items, [("A", gate_a)], policy)
    c_items = [p(registry, experiment_id, report) for p in policy.c_providers.values()]
    gate_c = _gate("C", c_items, [("A", gate_a), ("B", gate_b)], policy)
    return {"experiment_id": experiment_id, "model_kind": kind, "policy_version": policy.version, "gates": {"A": gate_a, "B": gate_b, "C": gate_c},
            "note": "Gate A says the algorithm is implemented as specified; it says nothing about profit. A strategy can fail Gate B and a correct algorithm still passes Gate A."}


def _gate(name: str, items: list[Item], deps: list[tuple[str, dict]], policy: GatePolicy) -> dict:
    status, stale = _status_of(items, deps)
    evidence = sorted(f"{i.id}={i.status}:{i.evidence}" for i in items) + sorted(f"gate{n}={d['status']}:{d['evidence_hash']}" for n, d in deps)
    reasons = [f"{i.label}: {i.reason}" for i in items if i.status != "ok" and i.reason] + [f"requires Gate {n}, which is {d['status']}" for n, d in deps if d["status"] != "passed" or d["stale"]]
    return {"gate": name, "status": status, "stale": stale, "policy_version": policy.version, "items": [i.as_dict() for i in items], "reasons": reasons, "evidence": evidence,
            "evidence_hash": digest(evidence, 24), "dependencies": [n for n, _ in deps]}


def decide(registry: Registry, experiment_id: str, actor: str = "system", policy: GatePolicy | None = None) -> dict:
    """Evaluate and append the decisions. A gate whose status, stale flag, policy and evidence set equal the last persisted decision is not written again. Each decision keeps its predecessor's sequence number.
    Returns the evaluation with ``seq`` and ``previous_seq`` for every gate and ``changed``: whether anything differs from the last stored decision (for a stale gate, ``previous`` says what was decided before)."""
    res = evaluate(registry, experiment_id, policy or DEFAULT_POLICY)
    with registry.transaction() as c:
        for name in GATES:
            g = res["gates"][name]
            last = c.execute("SELECT * FROM gate_decisions WHERE subject_id=? AND gate=? ORDER BY seq DESC LIMIT 1", (experiment_id, name)).fetchone()
            g["previous"] = None if last is None else {"status": last["status"], "seq": last["seq"], "at": last["at"], "stale": bool(last["stale"])}
            if last is not None and last["policy_version"] == g["policy_version"] and last["status"] == g["status"] and bool(last["stale"]) == g["stale"] and json.loads(last["evidence"] or "{}").get("hash") == g["evidence_hash"]:
                g["seq"], g["previous_seq"], g["new"] = last["seq"], last["previous_seq"], False
                continue
            cur = c.execute("INSERT INTO gate_decisions (subject_id, gate, policy_version, status, stale, evidence, reasons, actor, previous_seq, at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (experiment_id, name, g["policy_version"], g["status"], int(g["stale"]), json.dumps({"hash": g["evidence_hash"], "items": g["evidence"]}), json.dumps(g["reasons"]), actor,
                             None if last is None else last["seq"], time.strftime("%Y-%m-%d %H:%M:%S")))
            g["seq"], g["previous_seq"], g["new"] = int(cur.lastrowid), None if last is None else last["seq"], True
    res["gate_seqs"] = {n: res["gates"][n]["seq"] for n in GATES}
    res["changed"] = any(res["gates"][n]["new"] for n in GATES)
    return res


def history(registry: Registry, experiment_id: str, limit: int = 100) -> list[dict]:
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute("SELECT * FROM gate_decisions WHERE subject_id=? ORDER BY seq DESC LIMIT ?", (experiment_id, int(limit)))]
    for r in rows:
        r["evidence"], r["reasons"] = json.loads(r["evidence"] or "{}"), json.loads(r["reasons"] or "[]")
    return rows


def stage_history(registry: Registry, model_id: str) -> list[dict]:
    with registry._conn() as c:                                                                  # noqa: SLF001
        return [dict(r) for r in c.execute("SELECT * FROM stage_changes WHERE model_id=? ORDER BY seq", (model_id,))]


NEEDS = {"staging": ("A", "B"), "production": ("A", "B", "C")}


def authorize_stage(registry: Registry, model_id: str, experiment_id: str, stage: str, approval: dict | None = None, policy: GatePolicy | None = None) -> dict:
    """Allow or refuse a stage change from the persisted evidence. ``none``, ``candidate`` and ``archived`` need none (a model leaving use, or being registered, is not a claim). ``staging`` needs Gates A and B;
    ``production`` needs A, B and C and a recorded human approval (``{"actor": ..., "note": ...}``): the approval is a decision of a person, never inferred from the gates."""
    if stage not in NEEDS:
        return {"allowed": True, "stage": stage, "gate_seqs": {}, "reason": "this stage makes no claim that needs evidence"}
    res = decide(registry, experiment_id, "promotion", policy or DEFAULT_POLICY)
    bad = [(n, res["gates"][n]) for n in NEEDS[stage] if res["gates"][n]["status"] != "passed" or res["gates"][n]["stale"]]
    if bad:
        detail = " | ".join(f"Gate {n} is {g['status']}{' (stale)' if g['stale'] else ''}: {(g['reasons'] or ['no reason recorded'])[0]}" for n, g in bad)
        raise PromotionBlocked(f"{model_id} cannot move to {stage}: {detail}")
    if stage == "production" and not (approval and str(approval.get("actor", "")).strip() and str(approval.get("note", "")).strip()):
        raise PromotionBlocked(f"{model_id} passes the gates, but production is a human decision: record an approval with an actor and a note")
    return {"allowed": True, "stage": stage, "gate_seqs": res["gate_seqs"], "policy_version": res["policy_version"]}
