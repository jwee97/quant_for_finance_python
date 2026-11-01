"""The one place a frozen evaluation plan is run (spec v3, 6.2): reserve, freeze, evaluate once, record.

``plan_and_reserve`` creates a plan from candidate configurations, builds the data to learn exactly which assets and dates its holdout would reveal, reserves that history in the ledger (reporting any overlap with
earlier looks) and, when asked, freezes it. ``evaluate_plan`` claims the single evaluation event (``research_controls.begin_evaluation``: idempotent, serialised, refused for a consumed plan unless re-use is declared),
runs every candidate and the baseline under that authority, and records the result under the idempotency key: a retry with the same key returns the same experiments and reads nothing again.

``compat_finalize`` is what a request of the old shape (``finalize: true`` on a tuning or a search) is routed through. It does not bypass the plan: it creates and freezes one from the best validation candidates, says
so in the response together with the replacement (``POST /api/ml/evaluation-plan`` then ``/api/ml/evaluate``), and evaluates it once.
"""

from __future__ import annotations

import hashlib
import json

from . import research_controls as rc
from .config import config_hash, experiment_config
from .provenance import outcome_hash
from .experiments import ExperimentError, prepare, run_experiment
from .training import plan_for

DEPRECATION = ("finalize=true is deprecated. It now freezes an evaluation plan from the best validation candidates, reserves the test and forward history and evaluates it once, recording all of it. To control "
               "that yourself: POST /api/ml/evaluation-plan (create and freeze) and POST /api/ml/evaluate (one evaluation event, idempotent).")


def holdout_identity(cfg: dict, bundle=None, config=None, builder=None, dataset=None, info=None):
    """``(identity, dataset, info)``: the outcomes a configuration's test and forward segments would reveal. This builds the dataset (without looking at any outcome beyond what building it needs)."""
    ds, info = prepare(cfg, bundle, config, builder, dataset, info)
    try:
        plan, _ = plan_for(ds, cfg["split"], cfg["dataset"]["target"])
    except ValueError as exc:
        raise ExperimentError(str(exc)) from None
    return rc.identity_of(ds, plan), ds, info


def plan_and_reserve(registry, study_id: str, candidates: list[dict], baseline: dict | None = None, *, freeze: bool = False, actor: str = "api", bundle=None, config=None, builder=None, dataset=None,
                     info=None, **plan_fields) -> dict:
    """Create the plan, reserve its holdout (the history the first candidate's configuration reveals; every candidate must reveal the same) and optionally freeze it."""
    plan = rc.create_plan(registry, study_id, candidates, baseline, actor=actor, **plan_fields)
    content = plan["content"]
    configs = ([content["baseline"]] if content.get("baseline") else []) + content["candidates"]
    identities, first_ds = [], None
    for c in configs:
        ident, ds, info_c = holdout_identity(c["config"], bundle, config, builder, dataset if first_ds is None else None, info if first_ds is None else None)
        first_ds = first_ds if first_ds is not None else ds
        identities.append(ident)
    if len({tuple(i["assets"]) for i in identities}) > 1:
        raise rc.PlanError("the candidates would reveal the outcomes of different assets: one plan evaluates one holdout. Create one plan per data set.")
    # Candidates that differ only in how far ahead they look (a tuned horizon) start their test segments a few days apart: the plan protects the union of what they would reveal.
    start, end = min(i["start"] for i in identities), max(i["end"] for i in identities)
    identity = {"assets": identities[0]["assets"], "start": start, "end": end, "outcome_hash": identities[0]["outcome_hash"] if len({(i["start"], i["end"]) for i in identities}) == 1 else outcome_hash(first_ds, start, end)}
    identities = [identity]
    state = rc.reserve(registry, plan["plan_id"], identities[0], actor)
    if freeze:
        state = rc.freeze_plan(registry, plan["plan_id"], actor)
    return state


def evaluate_plan(registry, plan_id: str, *, idempotency_key: str, actor: str = "api", entry_point: str = "service", allow_reuse: bool = False, resume: bool = False, bundle=None, config=None,
                  builder=None, progress=None, robustness: bool | None = False, parent_ids: dict | None = None) -> dict:
    """Evaluate the frozen plan once. Returns ``{"status": "evaluated", "experiments": [...], ...}`` or, for a retry, the stored result (``replayed``), ``in_progress`` or ``failed``."""
    claim = rc.begin_evaluation(registry, plan_id, idempotency_key, actor=actor, entry_point=entry_point, allow_reuse=allow_reuse, resume=resume)
    if claim["status"] != "claimed":
        return claim
    ctx = claim["context"]
    plan = rc.plan_state(registry, plan_id)
    content = plan["content"]
    roles = ([("baseline", content["baseline"])] if content.get("baseline") else []) + [("candidate", c) for c in content["candidates"]]
    results, cache, live = [], {}, []
    progress = progress or (lambda msg: None)
    try:
        for i, (role, cand) in enumerate(roles):
            cfg = cand["config"]
            key = hashlib.sha256(json.dumps(cfg["dataset"], sort_keys=True, default=str).encode()).hexdigest()[:12]
            if key not in cache:
                cache[key] = prepare(experiment_config(cfg), bundle, config, builder, None, None, progress)
            ds, info = cache[key]
            progress(f"evaluating {role} {i + 1} of {len(roles)}: {cand['name']}")
            with rc.begin(registry, "evaluation", content["study_id"], op_key=f"{idempotency_key}-{cand['config_hash']}", counts=False, adopt=True, config_hash=cand["config_hash"], seed=int(cfg.get("seed", 0)),
                          detail={"plan_id": plan_id, "role": role, "idempotency_key": idempotency_key}, resume=bool(claim.get("resumed"))) as op:
                exp = run_experiment({**cfg, "name": cfg["name"]}, config=config, registry=registry, dataset=ds, info=info, robustness=robustness, evaluation=ctx, entry_point=entry_point,
                                     parent_id=(parent_ids or {}).get(cand["config_hash"]), study=content["study_id"])
                op.experiment_id = exp.id
            live.append(exp)
            results.append({"role": role, "name": cand["name"], "config_hash": cand["config_hash"], "experiment_id": exp.id, "holdout": exp.report["holdout"],
                            "segments": exp.report["segments"]["stats"], "sharpe": exp.report["metrics"]["oos"].get("sharpe")})
        rc.complete_evaluation(registry, ctx, ",".join(r["experiment_id"] for r in results), {"experiments": results})
    except BaseException as exc:                                                                      # the holdout stays consumed: outcomes may have been read before the failure
        rc.fail_evaluation(registry, ctx, f"{type(exc).__name__}: {exc}")
        raise
    return {"status": "evaluated", "plan_id": plan_id, "idempotency_key": idempotency_key, "reuse": ctx.reuse, "experiments": results,
            "untouched": all(r["holdout"].get("untouched") for r in results) if results else False, "ledger": rc.access_log(registry, plan_id=plan_id, limit=20), "_experiments": live}


def compat_finalize(registry, study_id: str, base_overrides: dict, params_list: list[dict], *, names: list[str] | None = None, selection_adjustment: str = "dsr_all_trials", idempotency_key: str,
                    actor: str = "api", entry_point: str = "api", bundle=None, config=None, builder=None, progress=None, robustness: bool | None = False, dataset=None, info=None) -> dict:
    """The deprecated ``finalize=true`` path: freeze a plan from the chosen parameter sets, reserve, evaluate once, record. A repeat with the same key returns the stored result."""
    from .optimization import apply_params

    cands = []
    for i, params in enumerate(params_list):
        cfg = apply_params(experiment_config(base_overrides), params)
        if names and i < len(names) and names[i]:
            cfg["name"] = str(names[i])[:80]
        cands.append(cfg)
    chosen = cands[:1] if len(cands) == 1 else cands
    adjustment = "none" if len(chosen) == 1 else selection_adjustment
    state = plan_and_reserve(registry, study_id, chosen, None, freeze=True, actor=actor, bundle=bundle, config=config, builder=builder, dataset=dataset, info=info, selection_adjustment=adjustment,
                             uncertainty_method="pending (Release 5)", notes="created by the deprecated finalize=true shortcut from the best validation candidates")
    # The old request never promised an untouched holdout. Where the history was already read (an earlier direct run, an earlier evaluation of the same plan) the evaluation goes ahead as a RECORDED RE-USE: the
    # ledger says so, the experiments' reports say it was not untouched, and Gate B will not count it.
    out = evaluate_plan(registry, state["plan_id"], idempotency_key=idempotency_key, actor=actor, entry_point=entry_point, allow_reuse=True, bundle=bundle, config=config, builder=builder, progress=progress,
                        robustness=robustness)
    out["deprecated"] = DEPRECATION
    out["plan"] = {"plan_id": state["plan_id"], "content_hash": state["content_hash"], "reservation": state["reservation"]}
    return out


__all__ = ["DEPRECATION", "compat_finalize", "config_hash", "evaluate_plan", "holdout_identity", "plan_and_reserve"]
