"""Research controls: study lineage, frozen evaluation plans, protected-holdout reservations and the access ledger (spec v3, P0-23, P0-25, P0-28, and the shared infrastructure of P1-26).

Four things live here, all persisted in the experiment registry and all append-only where history matters.

**Study lineage.** Every way of trying something on the data (a manual experiment, an API or command-line one, a feature search, a hyperparameter trial, a model choice, a signal or portfolio variation, an ensemble, a
meta-labelling run, a re-backtest) is an *operation*. ``begin`` writes it before the work starts (``registered``, then ``running``), and it ends as ``done``, ``failed`` or ``cancelled``; failures and cancellations are
looks too, so they stay in the count. An operation that only re-runs an already counted one (the finalisation of a search trial, the evaluation of a frozen plan) has ``counts=0`` and a ``parent_key``: the same try is
never counted twice. A study is by default the universe the data comes from, not the cache key of a panel, so changing the feature set, the target, the date range or the name of an experiment does not start a new
count; ``study`` in the configuration names one explicitly. Deleting an experiment leaves a tombstone and does not lower any count.

**Evaluation plans.** A plan fixes, before any protected outcome is read, what will be evaluated and how: the baseline and the candidates (their full merged configurations), the signal, execution and retraining rules, the
metrics, the uncertainty method, the selection treatment and the decision policy. ``create_plan`` stores it (a plan is immutable: a changed plan is a new plan), ``reserve`` binds it to the exact outcomes it protects
(assets, dates, and a hash of the realised returns) and reports any overlap with an earlier reservation, evaluation or exposure of the same history, whatever it was called; ``freeze_plan`` is the explicit transition after
which the plan can be evaluated.

**The access ledger.** ``begin_evaluation`` is the only way to an authorised look at a reserved holdout. It runs inside one write transaction: a retry with the same idempotency key returns the same result and never a second
look; a different key on a consumed plan is refused unless the caller says it is a deliberate re-use, which is then recorded as one and is never described as untouched. A run that reads the test and forward segments
outside a plan (``open_outcome_access``) is refused while a reservation that has not been evaluated covers that history; elsewhere it is recorded as an exposure of it (after a plan's evaluation the history is spent, and
reading it again is exactly that), and the experiment's report says ``exposed_exploratory``. The ledger is
what ``reserve`` consults, so renaming a study or rebuilding the data under another key does not make the history untouched again.

Scope of the protection: these controls govern the application's research workflow (the API, the command line and the Python functions that go through ``run_experiment`` and the evaluation service). They do not stop
the owner of a local checkout from editing the SQLite file or calling lower-level model code; stronger isolation needs process and storage permissions.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
import time
import uuid
from concurrent.futures import CancelledError
from dataclasses import dataclass, field

import pandas as pd

from .provenance import canon, digest

KINDS = ("experiment", "feature_search", "feature_choice", "hpo_trial", "model_choice", "signal_variation", "portfolio_variation", "ensemble", "meta_label", "rebacktest", "finalize", "evaluation", "null_replication")
NON_COUNTING = ("finalize", "evaluation", "feature_choice", "null_replication")             # they re-run, materialise or conclude a try that is counted elsewhere
STATES = ("registered", "running", "done", "failed", "cancelled")
TERMINAL = ("done", "failed", "cancelled")
ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
PLAN_STATES = ("draft", "reserved", "frozen", "consumed", "void")
PLAN_VERSION = 1


class ControlError(ValueError):
    """A request the research controls refuse; the message says what to do instead."""


class DuplicateOperation(ControlError):
    """The same operation key was registered twice (a retry): it must be resumed, not counted again."""


class HoldoutAccessError(ControlError):
    """Protected outcomes were asked for without an authorised evaluation context."""


class PlanError(ControlError):
    """An evaluation plan that cannot be created, reserved, frozen or evaluated as asked."""


class ResearchCancelled(Exception):
    """Raised by a job that was asked to stop while an operation was running."""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _j(obj) -> str:
    return json.dumps(obj, default=str, sort_keys=True)


# ------------------------------------------------------------------------------------------------------------------ studies
def default_study_id(cfg: dict) -> str:
    """The study of an experiment that names none: the universe its data comes from. A named universe is its own study; a list of tickers is identified by the sorted list."""
    universe = cfg["dataset"]["universe"]
    if isinstance(universe, str):
        return f"u-{universe}"[:80]
    return "u-" + hashlib.sha256(json.dumps(sorted(str(t).upper() for t in universe)).encode()).hexdigest()[:10]


def study_of(cfg: dict, explicit: str | None = None) -> str:
    sid = str(explicit or cfg.get("study") or default_study_id(cfg))
    if not ID.match(sid):
        raise ControlError("a study name is 1 to 80 letters, digits, '.', '_', ':' or '-'")
    return sid


def ensure_study(registry, study_id: str, name: str | None = None, hypothesis: str | None = None, parent_study: str | None = None, **fields) -> dict:
    """Create the study if it does not exist (never rewrites an existing one) and return its row."""
    if not ID.match(study_id):
        raise ControlError("a study name is 1 to 80 letters, digits, '.', '_', ':' or '-'")
    with registry._lock, registry._conn() as c:                                                   # noqa: SLF001
        c.execute("INSERT OR IGNORE INTO research_studies (study_id, name, hypothesis, parent_study, baseline_hash, candidate_family, selection_rule, metrics, manifest_hash, plan_hash, budget, status, created) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (study_id, name or study_id, hypothesis, parent_study, fields.get("baseline_hash"), fields.get("candidate_family"), fields.get("selection_rule"),
                                                          _j(fields.get("metrics")) if fields.get("metrics") is not None else None, fields.get("manifest_hash"), None,
                                                          _j(fields.get("budget")) if fields.get("budget") is not None else None, "open", _now()))
        row = c.execute("SELECT * FROM research_studies WHERE study_id=?", (study_id,)).fetchone()
    return dict(row)


def link_study(registry, child: str, parent: str) -> None:
    """Record that ``child`` continues ``parent`` (a renamed or split study): the child's lineage view includes the parent's operations. Set once; a parent is never replaced."""
    ensure_study(registry, parent)
    ensure_study(registry, child)
    with registry._lock, registry._conn() as c:                                                   # noqa: SLF001
        row = c.execute("SELECT parent_study FROM research_studies WHERE study_id=?", (child,)).fetchone()
        if row["parent_study"] and row["parent_study"] != parent:
            raise ControlError(f"study {child} already continues {row['parent_study']}; a parent is not replaced")
        c.execute("UPDATE research_studies SET parent_study=? WHERE study_id=? AND parent_study IS NULL", (parent, child))


def study_family(registry, study_id: str) -> list[str]:
    """The study and every ancestor (the studies it continues), oldest last."""
    out, seen = [], set()
    with registry._conn() as c:                                                                  # noqa: SLF001
        sid = study_id
        while sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
            row = c.execute("SELECT parent_study FROM research_studies WHERE study_id=?", (sid,)).fetchone()
            sid = row["parent_study"] if row else None
    return out


# ------------------------------------------------------------------------------------------------------------------ operations
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("research_active_operation", default=None)
_CANCEL = (KeyboardInterrupt, SystemExit, CancelledError, ResearchCancelled, GeneratorExit)


@dataclass
class Operation:
    registry: object
    op_key: str
    study_id: str
    kind: str
    counts: bool = True
    parent_key: str | None = None
    adopt: bool = False
    config_hash: str | None = None
    feature_hash: str | None = None
    dataset_key: str | None = None
    data_identity: str | None = None
    seed: int | None = None
    experiment_id: str | None = None
    state: str = "registered"
    finish_detail: dict = field(default_factory=dict)
    _token: object = field(default=None, repr=False)

    def _append(self, state: str, detail: dict | None = None, experiment_id: str | None = None) -> None:
        if experiment_id:
            self.experiment_id = experiment_id
        with self.registry._lock, self.registry._conn() as c:                                      # noqa: SLF001
            c.execute("INSERT INTO lineage_events (op_key, study_id, parent_key, kind, state, counts, config_hash, feature_hash, dataset_key, data_identity, seed, experiment_id, detail, at) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (self.op_key, self.study_id, self.parent_key, self.kind, state, int(self.counts), self.config_hash, self.feature_hash, self.dataset_key,
                                                                self.data_identity, self.seed, self.experiment_id, _j(detail or {}), _now()))
        self.state = state

    def start(self) -> "Operation":
        if self.state == "registered":
            self._append("running")
        return self

    def done(self, experiment_id: str | None = None, **detail) -> None:
        if self.state not in TERMINAL:
            self._append("done", detail, experiment_id)

    def fail(self, error: str, **detail) -> None:
        if self.state not in TERMINAL:
            self._append("failed", {"error": str(error)[:500], **detail})

    def cancel(self, reason: str = "cancelled") -> None:
        if self.state not in TERMINAL:
            self._append("cancelled", {"reason": str(reason)[:200]})

    def finish_with(self, exc: BaseException) -> None:
        if isinstance(exc, _CANCEL):
            self.cancel(type(exc).__name__)
        else:
            self.fail(f"{type(exc).__name__}: {exc}")

    def __enter__(self) -> "Operation":
        self._token = _ACTIVE.set(self)
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ACTIVE.reset(self._token)
        if exc is not None:
            self.finish_with(exc)
        else:
            self.done(**self.finish_detail)
        return False


def active() -> Operation | None:
    return _ACTIVE.get()


def begin(registry, kind: str, study_id: str, *, op_key: str | None = None, parent_key: str | None = None, counts: bool = True, adopt: bool = False, config_hash: str | None = None,
          feature_hash: str | None = None, dataset_key: str | None = None, data_identity: str | None = None, seed: int | None = None, detail: dict | None = None, resume: bool = False,
          start: bool = True) -> Operation:
    """Register an operation before it runs. ``op_key`` makes the registration idempotent: a retry with a key already registered is refused (``DuplicateOperation``) unless ``resume`` is set and the earlier
    attempt never finished, in which case it continues the same operation and is not counted again."""
    if kind not in KINDS:
        raise ControlError(f"an operation kind is one of {', '.join(KINDS)}")
    if not ID.match(study_id):
        raise ControlError("a study name is 1 to 80 letters, digits, '.', '_', ':' or '-'")
    key = op_key or f"op-{uuid.uuid4().hex[:16]}"
    parent = parent_key or (active().op_key if active() is not None else None)
    ensure_study(registry, study_id)
    with registry.transaction() as c:
        last = c.execute("SELECT state, counts, kind FROM lineage_events WHERE op_key=? ORDER BY seq DESC LIMIT 1", (key,)).fetchone()
        if last is not None:
            if not resume or last["state"] in TERMINAL:
                raise DuplicateOperation(f"operation {key} is already registered ({last['state']}); a retry must resume it, not count it again")
            counts = bool(last["counts"])
        else:
            c.execute("INSERT INTO lineage_events (op_key, study_id, parent_key, kind, state, counts, config_hash, feature_hash, dataset_key, data_identity, seed, experiment_id, detail, at) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (key, study_id, parent, kind, "registered", int(counts), config_hash, feature_hash, dataset_key, data_identity, seed, None, _j(detail or {}), _now()))
    op = Operation(registry, key, study_id, kind, counts, parent, adopt, config_hash, feature_hash, dataset_key, data_identity, seed, state="registered")
    if start:
        op.start()
    return op


def experiment_op(registry, cfg: dict, kind: str = "experiment", *, parent_experiment: str | None = None, counts: bool = True, study: str | None = None, experiment_id: str | None = None,
                  detail: dict | None = None, feature_hash: str | None = None, data_identity: str | None = None, adopt_parent: bool = True):
    """The operation of one experiment run. Inside an operation that ADOPTS its experiments (a search trial, a finalisation) the experiment is that operation: nothing is registered twice. Without a registry
    there is nothing to record and ``None`` is returned. Use it as ``with experiment_op(...) as op:`` (``op`` may be ``None``)."""
    from .config import config_hash, dataset_key

    outer = active()
    if outer is not None and outer.adopt and adopt_parent:
        return _Adopted(outer, experiment_id)
    if registry is None:
        return _Adopted(None, experiment_id)
    parent_key = op_key_of_experiment(registry, parent_experiment) if parent_experiment else None
    op = begin(registry, kind, study_of(cfg, study), parent_key=parent_key, counts=counts, config_hash=config_hash(cfg), feature_hash=feature_hash, dataset_key=dataset_key(cfg), data_identity=data_identity,
               seed=int(cfg.get("seed", 0)), detail=detail, start=False)
    op.experiment_id = experiment_id
    return op


class _Adopted:
    """The experiment run inside an enclosing operation that adopts it (or inside none, with no registry): it records the experiment id on the enclosing operation and leaves finishing it to the enclosing code."""

    def __init__(self, op: Operation | None, experiment_id: str | None):
        self.op, self.experiment_id = op, experiment_id

    def __enter__(self):
        if self.op is not None and self.experiment_id and self.experiment_id != "unsaved":
            self.op.experiment_id = self.experiment_id
        return self.op

    def __exit__(self, *exc) -> bool:
        return False


def op_key_of_experiment(registry, experiment_id: str) -> str | None:
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT op_key FROM lineage_events WHERE experiment_id=? ORDER BY seq DESC LIMIT 1", (experiment_id,)).fetchone()
    return row["op_key"] if row else None


def study_of_experiment(registry, experiment_id: str) -> str | None:
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT study_id FROM lineage_events WHERE experiment_id=? ORDER BY seq DESC LIMIT 1", (experiment_id,)).fetchone()
    return row["study_id"] if row else None


def _latest(registry, study_ids: list[str]) -> list[dict]:
    q = ",".join("?" * len(study_ids))
    with registry._conn() as c:                                                                  # noqa: SLF001
        first = {r["op_key"]: dict(r) for r in c.execute(f"SELECT * FROM lineage_events WHERE study_id IN ({q}) AND state='registered' ORDER BY seq", study_ids)}
        last = {}
        for r in c.execute(f"SELECT * FROM lineage_events WHERE study_id IN ({q}) ORDER BY seq", study_ids):
            last[r["op_key"]] = dict(r)
    out = []
    for key, f in first.items():
        l = last[key]
        out.append({"op_key": key, "study_id": f["study_id"], "parent_key": f["parent_key"], "kind": f["kind"], "state": l["state"], "counts": bool(f["counts"]), "config_hash": f["config_hash"],
                    "feature_hash": f["feature_hash"], "dataset_key": f["dataset_key"], "data_identity": f["data_identity"], "seed": f["seed"], "experiment_id": l["experiment_id"] or f["experiment_id"],
                    "registered": f["at"], "finished": l["at"] if l["state"] in TERMINAL else None, "detail": {**json.loads(f["detail"] or "{}"), **json.loads(l["detail"] or "{}")}})
    return out


def raw_looks(registry, study_id: str, include_ancestors: bool = True) -> int:
    """How many tries the study has recorded: operations that count, whatever their end (a failed or cancelled try is a try). This is a count of looks, not an estimate of independent trials: related trials
    are fewer than that in effect (``Study.effective_trials`` estimates it from the returns)."""
    ids = study_family(registry, study_id) if include_ancestors else [study_id]
    return sum(1 for o in _latest(registry, ids) if o["counts"])


def lineage(registry, study_id: str | None = None, experiment_id: str | None = None) -> dict:
    """The study's record: its operations with their latest state, the counts by kind and state, the distinct feature sets and configurations tried, the plans, the holdout accesses and the tombstones of deleted
    experiments. Given an ``experiment_id``, the study it belongs to and the chain of parents of that experiment."""
    if study_id is None and experiment_id is not None:
        study_id = study_of_experiment(registry, experiment_id)
    if study_id is None:
        raise ControlError("give a study or an experiment")
    family = study_family(registry, study_id)
    ops = _latest(registry, family)
    counted = [o for o in ops if o["counts"]]
    by_kind, by_state = {}, {}
    for o in counted:
        by_kind[o["kind"]] = by_kind.get(o["kind"], 0) + 1
        by_state[o["state"]] = by_state.get(o["state"], 0) + 1
    with registry._conn() as c:                                                                  # noqa: SLF001
        study = c.execute("SELECT * FROM research_studies WHERE study_id=?", (study_id,)).fetchone()
        tombs = [dict(r) for r in c.execute(f"SELECT * FROM experiment_tombstones WHERE study_id IN ({','.join('?' * len(family))}) ORDER BY deleted_at", family)]
        plans = [dict(r) for r in c.execute(f"SELECT plan_id, study_id, content_hash, created FROM evaluation_plans WHERE study_id IN ({','.join('?' * len(family))}) ORDER BY created", family)]
        sel = [dict(r) for r in c.execute(f"SELECT id, study_id, subject, method, status, reason, created FROM selection_evidence WHERE study_id IN ({','.join('?' * len(family))}) ORDER BY id", family)]
    chain = []
    if experiment_id:
        key = op_key_of_experiment(registry, experiment_id)
        index = {o["op_key"]: o for o in ops}
        while key and key in index and key not in [c["op_key"] for c in chain]:
            chain.append(index[key])
            key = index[key]["parent_key"]
    return {"study": dict(study) if study else {"study_id": study_id}, "studies": family, "operations": sorted(ops, key=lambda o: o["registered"]), "raw_looks": len(counted),
            "by_kind": by_kind, "by_state": by_state, "failed": by_state.get("failed", 0), "cancelled": by_state.get("cancelled", 0),
            "feature_sets": sorted({o["feature_hash"] for o in ops if o["feature_hash"]}), "configurations": sorted({o["config_hash"] for o in ops if o["config_hash"]}),
            "plans": [dict(p, **plan_state(registry, p["plan_id"], brief=True)) for p in plans], "holdout_access": access_log(registry, study_ids=family), "tombstones": tombs, "selection_evidence": sel,
            "parents": chain, "note": "Raw look counts are counts of recorded tries, failed and cancelled ones included. They are not estimates of independent trials; those are reported with the selection tests."}


# ------------------------------------------------------------------------------------------------------------------ feature search (P0-28)
def register_feature_search(registry, study_id: str, variants: list[dict], criterion: str, *, selected: list[str] | None = None, rule: dict | None = None, op_key: str | None = None,
                            adjustment: dict | None = None) -> dict:
    """Record an adaptive feature search: every variant tried, rejected ones included, as one counted operation each, and the choice made with the rule and the adjustment evidence that justify it.

    A variant is ``{"features": [...], "score": float | None, "accepted": bool, "reason": str}``. The adjustment is what ``selection.adjust_screen`` returned (or an explicit statement that no valid adjustment
    is available). Nothing here changes which features are used; it makes the search part of the study's history so that the next try is deflated by it, and so that the choice can be reproduced."""
    ensure_study(registry, study_id)
    keys, group = [], op_key or f"fs-{uuid.uuid4().hex[:12]}"
    for i, v in enumerate(variants):
        names = sorted(str(f) for f in v.get("features", []))
        op = begin(registry, "feature_search", study_id, op_key=f"{group}-{i}", counts=True, feature_hash=digest(names, 16),
                   detail={"criterion": criterion, "features": names, "score": v.get("score"), "accepted": bool(v.get("accepted")), "reason": v.get("reason", "")})
        op.done()
        keys.append(op.op_key)
    choice = begin(registry, "feature_choice", study_id, op_key=f"{group}-choice", counts=False, parent_key=keys[0] if keys else None,
                   feature_hash=digest(sorted(selected or []), 16), detail={"selected": sorted(selected or []), "rule": rule or {}, "adjustment": adjustment or {"status": "unavailable", "reason": "none recorded"},
                                                                          "variants": keys})
    choice.done()
    return {"group": group, "variants": keys, "choice": choice.op_key, "n_tried": len(variants), "n_rejected": sum(1 for v in variants if not v.get("accepted")),
            "feature_hash": digest(sorted(selected or []), 16)}


def reproduce_feature_choice(registry, choice_key: str, scores: dict[str, float], rule_fn) -> dict:
    """Re-derive a recorded choice from the recorded rule: ``rule_fn(scores, rule)`` must return the selected features. Whether it reproduces the recorded set is the evidence that the choice follows from the rule
    and the scores, not from anything else."""
    with registry._conn() as c:                                                                  # noqa: SLF001
        done = c.execute("SELECT detail FROM lineage_events WHERE op_key=? AND state='done' ORDER BY seq DESC LIMIT 1", (choice_key,)).fetchone()
        first = c.execute("SELECT detail FROM lineage_events WHERE op_key=? AND state='registered'", (choice_key,)).fetchone()
    if done is None or first is None:
        raise ControlError(f"no recorded feature choice {choice_key}")
    detail = {**json.loads(first["detail"] or "{}"), **json.loads(done["detail"] or "{}")}
    again = sorted(rule_fn(scores, detail.get("rule") or {}))
    return {"reproduced": again == sorted(detail["selected"]), "recorded": sorted(detail["selected"]), "again": again, "adjustment": detail.get("adjustment")}


# ------------------------------------------------------------------------------------------------------------------ evaluation plans
def _plan_events(c, plan_id: str) -> list[dict]:
    return [dict(r) for r in c.execute("SELECT * FROM plan_events WHERE plan_id=? ORDER BY seq", (plan_id,))]


def _state_of(events: list[dict]) -> str:
    state = "draft"
    for e in events:
        if e["state"] in ("reserved", "frozen", "consumed", "void"):
            state = e["state"]
    return state


def create_plan(registry, study_id: str, candidates: list[dict], baseline: dict | None = None, *, metrics: list[str] | None = None, uncertainty_method: str = "not_specified",
                selection_adjustment: str = "none", decision_policy: dict | None = None, retraining_rule: dict | None = None, notes: str = "", actor: str = "api") -> dict:
    """Store an evaluation plan (a draft). ``candidates`` are experiment overrides; each is merged into its full configuration and hashed, so the plan fixes exactly what will run. Several candidates share one
    evaluation and need a declared ``selection_adjustment`` (the evidence for each is consumed by the look at all of them)."""
    from .config import config_hash, experiment_config

    if not candidates:
        raise PlanError("a plan needs at least one candidate")
    if len(candidates) > 1 and str(selection_adjustment or "none") == "none":
        raise PlanError("several candidates share one look at the protected outcomes: declare the selection treatment (for example 'bonferroni', 'holm', 'reality_check', 'spa' or 'dsr_all_trials')")
    cfgs = []
    for i, over in enumerate(candidates):
        cfg = experiment_config({**over, "study": study_id})
        cfgs.append({"name": cfg["name"], "config_hash": config_hash(cfg), "config": cfg})
    base = None
    if baseline is not None:
        bcfg = experiment_config({**baseline, "study": study_id})
        base = {"name": bcfg["name"], "config_hash": config_hash(bcfg), "config": bcfg}
    first = cfgs[0]["config"]
    content = {"version": PLAN_VERSION, "study_id": study_id, "baseline": base, "candidates": cfgs,
               "feature_selection": [{"candidate": c["name"], "feature_groups": c["config"]["dataset"].get("feature_groups"), "features": c["config"]["dataset"].get("features")} for c in cfgs],
               "hyperparameters": [{"candidate": c["name"], "model": c["config"]["model"]} for c in cfgs], "signal_rule": [c["config"]["signal"] for c in cfgs],
               "execution": [{"candidate": c["name"], "backtest": c["config"]["backtest"]} for c in cfgs], "retraining_rule": retraining_rule or first["split"],
               "metrics": list(metrics or ["sharpe", "ic", "max_drawdown"]), "uncertainty_method": uncertainty_method, "selection_adjustment": selection_adjustment,
               "evaluation_horizon": {"segments": ["test", "forward"], "universe": first["dataset"]["universe"], "start": first["dataset"].get("start"), "end": first["dataset"].get("end"),
                                      "split": first["split"]},
               "decision_policy": decision_policy or {"rule": "report the pre-declared metrics with their intervals; no candidate is promoted by this look alone"}, "notes": notes}
    chash = digest(content, 64)
    plan_id = f"plan-{chash[:12]}"
    ensure_study(registry, study_id)
    with registry.transaction() as c:
        c.execute("INSERT OR IGNORE INTO evaluation_plans (plan_id, study_id, content, content_hash, created) VALUES (?,?,?,?,?)", (plan_id, study_id, canon(content), chash, _now()))
        if not _plan_events(c, plan_id):
            c.execute("INSERT INTO plan_events (plan_id, state, actor, detail, at) VALUES (?,?,?,?,?)", (plan_id, "draft", actor, _j({"content_hash": chash}), _now()))
    return plan_state(registry, plan_id)


def plan_state(registry, plan_id: str, brief: bool = False) -> dict:
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT * FROM evaluation_plans WHERE plan_id=?", (plan_id,)).fetchone()
        if row is None:
            raise PlanError(f"unknown evaluation plan '{plan_id}'")
        events = _plan_events(c, plan_id)
        res = c.execute("SELECT * FROM holdout_reservations WHERE plan_id=? ORDER BY created LIMIT 1", (plan_id,)).fetchone()
        evals = c.execute("SELECT COUNT(*) FROM holdout_access_log WHERE plan_id=? AND access_kind IN ('evaluate','evaluate_reuse') AND status='claimed'", (plan_id,)).fetchone()[0]
    state = _state_of(events)
    out = {"plan_id": plan_id, "state": state, "study_id": row["study_id"], "content_hash": row["content_hash"], "created": row["created"], "events": [{k: e[k] for k in ("state", "actor", "at")} for e in events],
           "reservation": None if res is None else {"reservation_id": res["reservation_id"], "assets": json.loads(res["assets"]), "range": [res["range_start"], res["range_end"]], "untouched": bool(res["untouched"]),
                                                  "overlaps": json.loads(res["overlaps"] or "[]")}, "evaluations": int(evals)}
    if not brief:
        out["content"] = json.loads(row["content"])
    return out


def _overlaps(c, identity: dict, exclude_plan: str | None = None) -> list[dict]:
    """Earlier reservations, evaluations and exposures that share an asset and a stretch of dates with ``identity``, or have the same outcome content."""
    lo, hi = pd.Timestamp(identity["start"]), pd.Timestamp(identity["end"])
    mine = set(identity["assets"])
    found = []
    for r in c.execute("SELECT * FROM holdout_reservations"):
        if exclude_plan and r["plan_id"] == exclude_plan:
            continue
        shared = sorted(mine & set(json.loads(r["assets"])))
        same_content = bool(identity.get("outcome_hash")) and r["outcome_hash"] == identity.get("outcome_hash")
        if (shared and pd.Timestamp(r["range_start"]) <= hi and pd.Timestamp(r["range_end"]) >= lo) or same_content:
            consumed = c.execute("SELECT COUNT(*) FROM holdout_access_log WHERE plan_id=? AND access_kind IN ('evaluate','evaluate_reuse') AND status='claimed'", (r["plan_id"],)).fetchone()[0] > 0
            found.append({"kind": "reservation", "reservation_id": r["reservation_id"], "plan_id": r["plan_id"], "study_id": r["study_id"], "shared_assets": shared, "range": [r["range_start"], r["range_end"]],
                          "consumed": bool(consumed), "same_outcome_content": same_content})
    for a in c.execute("SELECT * FROM holdout_access_log WHERE access_kind='exposure' AND range_start IS NOT NULL"):
        shared = sorted(mine & set(json.loads(a["assets"] or "[]")))
        if shared and pd.Timestamp(a["range_start"]) <= hi and pd.Timestamp(a["range_end"]) >= lo:
            found.append({"kind": "exposure", "seq": a["seq"], "plan_id": a["plan_id"], "study_id": a["study_id"], "shared_assets": shared, "range": [a["range_start"], a["range_end"]], "consumed": True,
                          "same_outcome_content": False, "purpose": a["purpose"], "result_ref": a["result_ref"]})
    return found


def reserve(registry, plan_id: str, identity: dict, actor: str = "api") -> dict:
    """Bind a plan to the outcomes it protects: ``identity`` is ``{"assets": [...], "start", "end", "outcome_hash"}`` (``identity_of``). Any overlap with another reservation, an evaluation or an exposure is recorded
    with the reservation; ``untouched`` is False when any of them was consumed. Reserving the same plan again returns the first reservation."""
    for k in ("assets", "start", "end"):
        if not identity.get(k):
            raise PlanError(f"a reservation needs the assets and the dates it protects ({k} is missing)")
    with registry.transaction() as c:
        row = c.execute("SELECT study_id FROM evaluation_plans WHERE plan_id=?", (plan_id,)).fetchone()
        if row is None:
            raise PlanError(f"unknown evaluation plan '{plan_id}'")
        have = c.execute("SELECT reservation_id FROM holdout_reservations WHERE plan_id=?", (plan_id,)).fetchone()
        if have is None:
            overlaps = _overlaps(c, identity, exclude_plan=plan_id)
            untouched = not any(o["consumed"] for o in overlaps)
            rid = "hold-" + hashlib.sha256(canon({"plan": plan_id, "assets": sorted(identity["assets"]), "start": identity["start"], "end": identity["end"]}).encode()).hexdigest()[:12]
            c.execute("INSERT INTO holdout_reservations (reservation_id, plan_id, study_id, assets, range_start, range_end, outcome_hash, untouched, overlaps, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (rid, plan_id, row["study_id"], _j(sorted(identity["assets"])), str(identity["start"]), str(identity["end"]), identity.get("outcome_hash"), int(untouched), _j(overlaps), _now()))
            c.execute("INSERT INTO plan_events (plan_id, state, actor, detail, at) VALUES (?,?,?,?,?)", (plan_id, "reserved", actor, _j({"reservation_id": rid, "untouched": untouched, "n_overlaps": len(overlaps)}), _now()))
            c.execute("INSERT INTO holdout_access_log (at, reservation_id, plan_id, study_id, access_kind, purpose, actor, entry_point, status, assets, range_start, range_end, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (_now(), rid, plan_id, row["study_id"], "reserve", "reservation", actor, "service", "recorded", _j(sorted(identity["assets"])), str(identity["start"]), str(identity["end"]),
                       _j({"untouched": untouched, "overlaps": overlaps})))
    return plan_state(registry, plan_id)


def freeze_plan(registry, plan_id: str, actor: str = "api") -> dict:
    """The explicit, persisted transition after which the plan may be evaluated. Refused unless the plan has a reservation. Freezing a frozen plan is a no-op."""
    with registry.transaction() as c:
        events = _plan_events(c, plan_id)
        if not events:
            raise PlanError(f"unknown evaluation plan '{plan_id}'")
        state = _state_of(events)
        if state in ("frozen", "consumed"):
            return plan_state(registry, plan_id)
        if state == "void":
            raise PlanError("a void plan cannot be frozen")
        if state != "reserved":
            raise PlanError("reserve the plan's holdout before freezing it: a plan with no reservation protects nothing")
        row = c.execute("SELECT content_hash FROM evaluation_plans WHERE plan_id=?", (plan_id,)).fetchone()
        c.execute("INSERT INTO plan_events (plan_id, state, actor, detail, at) VALUES (?,?,?,?,?)", (plan_id, "frozen", actor, _j({"content_hash": row["content_hash"]}), _now()))
    return plan_state(registry, plan_id)


def void_plan(registry, plan_id: str, reason: str, actor: str = "api") -> dict:
    """Withdraw a plan that has not been evaluated (its reservation stays in the ledger: the history it named is not released)."""
    with registry.transaction() as c:
        state = _state_of(_plan_events(c, plan_id))
        if state == "consumed":
            raise PlanError("a consumed plan cannot be voided: its look has happened")
        c.execute("INSERT INTO plan_events (plan_id, state, actor, detail, at) VALUES (?,?,?,?,?)", (plan_id, "void", actor, _j({"reason": reason}), _now()))
    return plan_state(registry, plan_id)


def identity_of(ds, segment_plan) -> dict:
    """The outcomes a dataset's reserved segments (test and forward) would reveal: the assets, the first and last date, and a hash of the realised returns over them."""
    from .provenance import outcome_hash

    dates = segment_plan.dates
    first, last = dates[segment_plan.bounds["test"][0]], dates[-1]
    return {"assets": sorted(ds.assets), "start": str(first.date()), "end": str(last.date()), "outcome_hash": outcome_hash(ds, first, last)}


# ------------------------------------------------------------------------------------------------------------------ the access ledger
@dataclass(frozen=True)
class EvaluationContext:
    """Authority for one evaluation event: the plan, its reservation and the idempotency key the ledger recorded. Only ``begin_evaluation`` creates one."""

    plan_id: str
    reservation_id: str
    key: str
    actor: str
    entry_point: str
    reuse: bool = False


def begin_evaluation(registry, plan_id: str, idempotency_key: str, *, actor: str = "api", entry_point: str = "service", purpose: str = "final_evaluation", allow_reuse: bool = False,
                     resume: bool = False) -> dict:
    """Claim the one look at a frozen plan's holdout. Returns ``{"status": "claimed", "context": EvaluationContext}``, or ``{"status": "replayed", "result": ...}`` when this key already completed (a retry gets the
    same answer and no second look), or ``{"status": "in_progress"}`` when it was claimed and has not finished (``resume=True`` takes it over after a crash). Raises ``PlanError`` when the plan is not frozen or
    was already consumed under another key without ``allow_reuse``, and when another plan's consumed holdout overlaps this one's."""
    if not ID.match(str(idempotency_key)):
        raise PlanError("an idempotency key is 1 to 80 letters, digits, '.', '_', ':' or '-'")
    with registry.transaction() as c:
        events = _plan_events(c, plan_id)
        if not events:
            raise PlanError(f"unknown evaluation plan '{plan_id}'")
        res = c.execute("SELECT * FROM holdout_reservations WHERE plan_id=?", (plan_id,)).fetchone()
        rows = [dict(r) for r in c.execute("SELECT * FROM holdout_access_log WHERE idempotency_key=? ORDER BY seq", (idempotency_key,))]
        if rows:
            if rows[0]["plan_id"] != plan_id:
                raise PlanError(f"the idempotency key {idempotency_key} belongs to another plan")
            done = [r for r in rows if r["status"] == "completed"]
            if done:
                return {"status": "replayed", "result": json.loads(done[-1]["detail"] or "{}"), "result_ref": done[-1]["result_ref"], "plan_id": plan_id}
            if any(r["status"] == "failed" for r in rows) and not resume:
                return {"status": "failed", "detail": json.loads([r for r in rows if r["status"] == "failed"][-1]["detail"] or "{}"), "plan_id": plan_id}
            if not resume:
                return {"status": "in_progress", "plan_id": plan_id, "since": rows[0]["at"]}
            n = sum(1 for r in rows if r["status"].startswith("resumed")) + 1
            c.execute("INSERT INTO holdout_access_log (at, reservation_id, plan_id, study_id, access_kind, purpose, idempotency_key, actor, entry_point, status, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (_now(), rows[0]["reservation_id"], plan_id, rows[0]["study_id"], rows[0]["access_kind"], purpose, idempotency_key, actor, entry_point, f"resumed-{n}", "{}"))
            return {"status": "claimed", "context": EvaluationContext(plan_id, rows[0]["reservation_id"], idempotency_key, actor, entry_point, rows[0]["access_kind"] == "evaluate_reuse"), "resumed": True}
        state = _state_of(events)
        if state != "frozen" and state != "consumed":
            raise PlanError(f"the plan is {state}, not frozen: freeze it (after reserving its holdout) before evaluating")
        if res is None:
            raise PlanError("the plan has no reservation")
        identity = {"assets": json.loads(res["assets"]), "start": res["range_start"], "end": res["range_end"], "outcome_hash": res["outcome_hash"]}
        taken = [o for o in _overlaps(c, identity, exclude_plan=plan_id) if o["consumed"]]
        consumed_here = state == "consumed"
        reuse = bool(consumed_here or taken)
        if reuse and not allow_reuse:
            why = ("this plan's holdout was already evaluated" if consumed_here else "earlier looks at this history overlap it: " + "; ".join(f"{o['kind']} {o.get('reservation_id') or o.get('seq')}" for o in taken[:3]))
            raise PlanError(f"{why}. A deliberate new evaluation is a new access event and is never untouched: pass allow_reuse to record it as a re-use")
        kind = "evaluate_reuse" if reuse else "evaluate"
        c.execute("INSERT INTO holdout_access_log (at, reservation_id, plan_id, study_id, access_kind, purpose, idempotency_key, actor, entry_point, status, assets, range_start, range_end, detail) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (_now(), res["reservation_id"], plan_id, res["study_id"], kind, purpose, idempotency_key, actor, entry_point, "claimed", res["assets"], res["range_start"],
                                                           res["range_end"], _j({"reuse": reuse, "overlaps": taken})))
        c.execute("INSERT INTO plan_events (plan_id, state, actor, detail, at) VALUES (?,?,?,?,?)", (plan_id, "consumed", actor, _j({"key": idempotency_key, "access": kind}), _now()))
    return {"status": "claimed", "context": EvaluationContext(plan_id, res["reservation_id"], idempotency_key, actor, entry_point, reuse)}


def complete_evaluation(registry, ctx: EvaluationContext, result_ref: str, result: dict | None = None) -> None:
    """Record the end of an evaluation event with what it produced. The key now returns this result to any retry."""
    with registry.transaction() as c:
        first = c.execute("SELECT * FROM holdout_access_log WHERE idempotency_key=? ORDER BY seq LIMIT 1", (ctx.key,)).fetchone()
        if first is None:
            raise PlanError("this evaluation was never claimed")
        c.execute("INSERT OR IGNORE INTO holdout_access_log (at, reservation_id, plan_id, study_id, access_kind, purpose, idempotency_key, actor, entry_point, status, result_ref, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (_now(), ctx.reservation_id, ctx.plan_id, first["study_id"], first["access_kind"], first["purpose"], ctx.key, ctx.actor, ctx.entry_point, "completed", result_ref, _j(result or {})))


def fail_evaluation(registry, ctx: EvaluationContext, error: str) -> None:
    """Record that an evaluation stopped without finishing. The holdout stays consumed: outcomes may already have been read, and nothing proves they were not."""
    with registry.transaction() as c:
        first = c.execute("SELECT * FROM holdout_access_log WHERE idempotency_key=? ORDER BY seq LIMIT 1", (ctx.key,)).fetchone()
        if first is not None:
            c.execute("INSERT OR IGNORE INTO holdout_access_log (at, reservation_id, plan_id, study_id, access_kind, purpose, idempotency_key, actor, entry_point, status, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (_now(), ctx.reservation_id, ctx.plan_id, first["study_id"], first["access_kind"], first["purpose"], ctx.key, ctx.actor, ctx.entry_point, "failed", _j({"error": str(error)[:500]})))


def access_log(registry, study_ids: list[str] | None = None, plan_id: str | None = None, limit: int = 500) -> list[dict]:
    q, args = "SELECT * FROM holdout_access_log WHERE 1=1", []
    if study_ids:
        q += f" AND study_id IN ({','.join('?' * len(study_ids))})"
        args += list(study_ids)
    if plan_id:
        q += " AND plan_id=?"
        args.append(plan_id)
    q += " ORDER BY seq DESC LIMIT ?"
    args.append(int(limit))
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute(q, args)]
    for r in rows:
        r["detail"] = json.loads(r["detail"] or "{}")
        r["assets"] = json.loads(r["assets"]) if r["assets"] else None
    return rows


def open_outcome_access(registry, cfg: dict, identity: dict, evaluation: EvaluationContext | None, *, entry_point: str, purpose: str, experiment_id: str) -> dict:
    """Called before a run predicts the test and forward segments. With an authorised ``evaluation`` for a candidate of its frozen plan the access is the plan's evaluation. Without one, a run on history that a
    reservation covers is refused; on history nothing reserved it is recorded as an exposure and labelled exploratory. Returns the ``holdout`` block of the experiment's report."""
    from .config import config_hash

    chash = config_hash(cfg)
    with registry.transaction() as c:
        if evaluation is not None:
            events = _plan_events(c, evaluation.plan_id)
            claim = c.execute("SELECT * FROM holdout_access_log WHERE idempotency_key=? AND status='claimed'", (evaluation.key,)).fetchone()
            row = c.execute("SELECT content FROM evaluation_plans WHERE plan_id=?", (evaluation.plan_id,)).fetchone()
            if not events or claim is None or row is None or _state_of(events) not in ("consumed", "frozen"):
                raise HoldoutAccessError("the evaluation context is not backed by a claimed evaluation in the ledger")
            content = json.loads(row["content"])
            allowed = {x["config_hash"] for x in content["candidates"]} | ({content["baseline"]["config_hash"]} if content.get("baseline") else set())
            if chash not in allowed:
                raise HoldoutAccessError(f"configuration {chash} is not a candidate or the baseline of the frozen plan {evaluation.plan_id}: a plan authorises only what it froze")
            res = c.execute("SELECT * FROM holdout_reservations WHERE reservation_id=?", (evaluation.reservation_id,)).fetchone()
            if (res is None or set(identity["assets"]) - set(json.loads(res["assets"])) or pd.Timestamp(identity["start"]) < pd.Timestamp(res["range_start"])
                    or pd.Timestamp(identity["end"]) > pd.Timestamp(res["range_end"])):
                raise HoldoutAccessError("the data of this run is not the history the plan's reservation protects")
            return {"status": "evaluated_under_plan", "plan_id": evaluation.plan_id, "reservation_id": evaluation.reservation_id, "idempotency_key": evaluation.key, "reuse": bool(evaluation.reuse),
                    "untouched": bool(res["untouched"]) and not evaluation.reuse, "note": "the test and forward segments were predicted once, as the frozen plan's evaluation"}
        covering = _overlaps(c, identity)
        held = [o for o in covering if o["kind"] == "reservation" and not o["consumed"]]            # a reservation protects until its evaluation has happened; after that the history is spent, and reading it again is recorded
        if held:
            raise HoldoutAccessError(f"this run would read outcomes that reservation {held[0]['reservation_id']} (plan {held[0]['plan_id']}) protects. Run a development experiment (it stops at the end of the "
                                     "validation segment) or evaluate the frozen plan with POST /api/ml/evaluate")
        c.execute("INSERT INTO holdout_access_log (at, plan_id, study_id, access_kind, purpose, actor, entry_point, status, result_ref, assets, range_start, range_end, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (_now(), None, study_of(cfg), "exposure", purpose, "run", entry_point, "exposed", experiment_id, _j(sorted(identity["assets"])), identity["start"], identity["end"],
                   _j({"config_hash": chash, "earlier_exposures": len([o for o in covering if o["kind"] == "exposure"])})))
    return {"status": "exposed_exploratory", "untouched": False,
            "note": "the test and forward segments were read outside a frozen evaluation plan: this run is exploratory, its test and forward numbers are not an untouched holdout, and it cannot support the Strategy Validated gate"}


TOUCHED = ("evaluated_under_plan", "exposed_exploratory", "reanalysis_of_exposed_outcomes")


def inherit_holdout(registry, parent_ids: list[str], child_id: str, purpose: str, entry_point: str = "service") -> dict:
    """The ``holdout`` block of an experiment made from saved ones (a re-backtest, a meta-labelled run): it inherits what its parents read. If any parent's report shows protected outcomes were read, this is a
    re-analysis of them: the ledger gets a ``reanalysis`` row, the block says so and is never untouched. A parent that stopped at the validation segment passes ``development`` on. A parent saved before the
    ledger existed passes on ``inherited_unknown``: unknown is not treated as untouched."""
    blocks = []
    for pid in parent_ids:
        try:
            blocks.append((pid, registry.report(pid).get("holdout") or {}))
        except Exception:                                                                           # noqa: BLE001 - an unreadable parent is unknown, not clean
            blocks.append((pid, {}))
    touched = [(pid, b) for pid, b in blocks if b.get("status") in TOUCHED]
    parents = {pid: b.get("status", "inherited_unknown") for pid, b in blocks}
    if touched:
        pid, b = touched[0]
        ident = b.get("identity") or {}
        study = study_of_experiment(registry, pid)
        with registry._lock, registry._conn() as c:                                                 # noqa: SLF001
            c.execute("INSERT INTO holdout_access_log (at, reservation_id, plan_id, study_id, access_kind, purpose, actor, entry_point, status, result_ref, assets, range_start, range_end, detail) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (_now(), b.get("reservation_id"), b.get("plan_id"), study, "reanalysis", purpose, "run", entry_point, "recorded", child_id,
                                                               _j(ident.get("assets")) if ident else None, ident.get("start"), ident.get("end"), _j({"parents": parents})))
        return {"status": "reanalysis_of_exposed_outcomes", "untouched": False, "parents": parents, "plan_id": b.get("plan_id"), "reservation_id": b.get("reservation_id"), "identity": ident or None,
                "note": "built from experiments whose test and forward outcomes were already read: this is another look at them, not an untouched evaluation"}
    if blocks and all(b.get("status") == "development" for _, b in blocks):
        return {"status": "development", "parents": parents, "note": "built from development runs: no protected outcome was read"}
    return {"status": "inherited_unknown", "parents": parents, "untouched": False, "note": "a parent was saved before the access ledger existed; what it read is unknown and is not treated as untouched"}


def holdout_labels(registry, experiment_ids: list[str]) -> dict:
    """What each experiment's protected outcomes were, from the ledger and the lineage rather than the report files: ``evaluated_under_plan``, ``exposed_exploratory``,
    ``reanalysis_of_exposed_outcomes`` or ``development_or_unrecorded``."""
    if not experiment_ids:
        return {}
    q = ",".join("?" * len(experiment_ids))
    out = {i: "development_or_unrecorded" for i in experiment_ids}
    with registry._conn() as c:                                                                  # noqa: SLF001
        for r in c.execute(f"SELECT result_ref, access_kind FROM holdout_access_log WHERE result_ref IN ({q}) AND access_kind IN ('exposure','reanalysis')", list(experiment_ids)):
            out[r["result_ref"]] = "exposed_exploratory" if r["access_kind"] == "exposure" else "reanalysis_of_exposed_outcomes"
        for r in c.execute(f"SELECT DISTINCT experiment_id FROM lineage_events WHERE kind='evaluation' AND state='done' AND experiment_id IN ({q})", list(experiment_ids)):
            out[r["experiment_id"]] = "evaluated_under_plan"
    return out


def guard_development(train, segment_plan) -> None:
    """The invariant behind the development mode: nothing predicted by a run that stops at the end of the validation segment lies in the test or forward segments. A violation is a defect of the
    training code, not a user error, and stops the run."""
    dates = pd.DatetimeIndex(train.predictions.index.get_level_values(0).unique())
    cut = segment_plan.dates[segment_plan.bounds["validation"][1] - 1]
    if len(dates) and dates.max() > cut:
        raise HoldoutAccessError(f"a development run produced predictions for {dates.max().date()}, after the end of the validation segment ({cut.date()}): protected outcomes were reached")


def holdout_status(registry, study_id: str | None = None, plan_id: str | None = None) -> dict:
    """The plans, their reservations (with overlaps and whether they are untouched), the evaluations and the access history for a study or a plan."""
    with registry._conn() as c:                                                                  # noqa: SLF001
        ids = [r["plan_id"] for r in c.execute("SELECT plan_id FROM evaluation_plans" + (" WHERE study_id=?" if study_id else "") + " ORDER BY created", ([study_id] if study_id else []))]
    if plan_id:
        ids = [plan_id]
    plans = [plan_state(registry, p, brief=True) for p in ids]
    with registry._conn() as c:                                                                  # noqa: SLF001
        exposures = [dict(r) for r in c.execute("SELECT seq, at, study_id, purpose, result_ref, assets, range_start, range_end FROM holdout_access_log WHERE access_kind='exposure' ORDER BY seq DESC LIMIT 200")]
    return {"plans": plans, "exposures": exposures, "accesses": access_log(registry, [study_id] if study_id else None, plan_id),
            "reasons": ["reserved: the plan has bound its holdout and nothing may read it but the plan's evaluation", "consumed: the plan's evaluation has happened; a new evaluation is a re-use and is never untouched",
                        "exposure: a run read these outcomes without a plan: the history is exploratory"]}
