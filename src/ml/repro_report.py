"""Reproducible reports from persisted artifact hashes and policy versions (spec v3, Release 8, P3-12). Optional: a report exists only when someone builds one.

A report is a **pure function of a snapshot**. ``snapshot`` is the only step that reads the live registry: it copies, at one moment, what the report needs (the experiment's stored report, the hash of each
artifact file, the latest stored diagnostics of each kind with its sequence number and content hash, the three gates as computed from the evidence, the validation policy that was adopted, the version of
every procedure). ``render`` turns that snapshot into the report and does nothing else, so the same snapshot always gives the same bytes. The report is stored **with its snapshot** in the append-only
``analysis_records`` table under its own hash, and:

* **A historical report never changes.** Building another report later (the model was refit, a newer diagnostic was recorded, an artifact was replaced) makes a *new* report with a new id; the old one is
  returned exactly as stored. ``reproduce`` re-renders a stored report from its own snapshot and compares the hashes: it proves the content derives from the recorded inputs and from nothing else.
* **Staleness is said, not hidden.** ``verify_inputs`` compares the recorded input hashes with the registry as it is now (artifact files, the latest record of each diagnostic kind, the configuration
  hash) and lists what has changed since the report was built.
* **Nothing is invented.** A diagnostic that was never computed is listed as ``not computed``; a figure on simulated data carries that label beside it; the gate reasons, the execution mode's limits, the
  report's own warnings and the unresolved items all appear in ``limitations``, with the same wording the dashboard shows.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from . import diagnostics as dx

VERSION = "report-v1"
KIND = "reproducible_report"
DIAGNOSTIC_KINDS = ("statistical_diagnostics", "alpha_diagnostics", "portfolio_diagnostics", "delay_report", "drift_report", "uncertainty_monitor", "multiseed")
SECTIONS = ("header", "performance", "uncertainty_selection", "coverage", "costs_capacity", "attribution", "lineage", "gates", "limitations")


class ReportError(ValueError):
    """A request the reporting procedure refuses."""


def _canon(x) -> str:
    return json.dumps(dx._jsonable(x), sort_keys=True, separators=(",", ":"), allow_nan=False)             # noqa: SLF001


def digest(x, n: int = 16) -> str:
    return hashlib.sha256(_canon(x).encode()).hexdigest()[:n]


def _file_hash(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def snapshot(registry, exp_id: str, *, now: float | None = None) -> dict:
    """Everything a report of this experiment is built from, copied at one moment. The only function that reads the live registry."""
    from . import gates
    from . import policy as pol

    rec = registry.get_experiment(exp_id)
    rep = registry.report(exp_id)
    arts = [{"name": a["name"], "kind": a["kind"], "bytes": a["bytes"], "sha256": _file_hash(a["path"])} for a in rec.get("artifacts", [])]
    diags = {}
    for kind in DIAGNOSTIC_KINDS:
        r = dx.latest(registry, exp_id, kind)
        diags[kind] = None if r is None else {"seq": r["seq"], "content_hash": r["content_hash"], "created": r["created"], "policy_hash": r["policy_hash"], "content": r["content"]}
    adopted = pol.adopted(registry)
    ev = gates.evaluate(registry, exp_id)
    return dx._jsonable({"experiment_id": exp_id, "experiment": {k: rec.get(k) for k in ("name", "model_kind", "target", "horizon", "signal_method", "source", "universe", "status", "created", "config_hash", "dataset_key")},  # noqa: SLF001
                         "report": {k: rep.get(k) for k in ("config", "config_hash", "dataset", "segments", "metrics", "ic", "overfit", "importance", "trials", "warnings", "execution", "holdout", "lineage", "audit",
                                                            "robustness", "reproducibility", "provenance")},
                         "artifacts": arts, "diagnostics": diags, "gates": ev, "policy": {"validation_policy": None if adopted is None else adopted[0].hash(), "gate_policy": ev.get("policy_version")},
                         "versions": {"report": VERSION, "diagnostic_kinds": list(DIAGNOSTIC_KINDS)}, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() if now is None else now))})


def _simulated(source) -> bool:
    return any(w in str(source or "").lower() for w in ("synth", "simulat", "fake", "toy"))


def render(snap: dict) -> dict:
    """The report, as a pure function of the snapshot."""
    exp, rep, diags, ev = snap["experiment"], snap["report"], snap["diagnostics"], snap["gates"]
    simulated = _simulated(exp.get("source")) or _simulated((rep.get("dataset") or {}).get("name"))
    label = "SIMULATED DATA: these figures describe a simulation, not a market" if simulated else None
    gates = {n: {"status": g["status"], "stale": g.get("stale", False), "reasons": g.get("reasons", []), "items": [{"id": i["id"], "label": i["label"], "status": i["status"], "reason": i["reason"]} for i in g.get("items", [])]}
             for n, g in ev["gates"].items()}
    order = sorted(diags)                                                                              # stored JSON has sorted keys: the report must not depend on the order a mapping was built in
    absent = [k for k in order if diags[k] is None]
    present = {k: {"seq": diags[k]["seq"], "content_hash": diags[k]["content_hash"], "created": diags[k]["created"], "policy_hash": diags[k]["policy_hash"]} for k in order if diags[k] is not None}
    limits = []
    limits += [f"report warning: {w}" for w in rep.get("warnings") or []]
    for n, g in gates.items():
        if g["status"] != "passed":
            limits += [f"Gate {n} is {g['status']}: {r}" for r in (g["reasons"] or [f"no reason recorded ({g['status']})"])]
    ex = rep.get("execution") or {}
    limits += [f"execution ({ex.get('mode', 'not recorded')}): cannot support {c}" for c in ex.get("cannot_support") or []]
    limits += [f"not modelled: {c}" for c in ex.get("not_modelled") or []]
    ho = rep.get("holdout") or {}
    limits.append(f"holdout status: {ho.get('status', 'not recorded')}" + (f" ({ho['note']})" if ho.get("note") else ""))
    limits += [f"{k}: not computed" for k in absent]
    for k in order:
        if diags[k] is not None:
            limits += [f"{k}: {c}" for c in (diags[k]["content"].get("caveats") or [])[:4]]
    stat = (diags.get("statistical_diagnostics") or {}).get("content")
    alpha = (diags.get("alpha_diagnostics") or {}).get("content")
    out = {
        "report_version": VERSION, "simulated_data_label": label,
        "header": {"experiment_id": snap["experiment_id"], "name": exp.get("name"), "model": exp.get("model_kind"), "target": exp.get("target"), "horizon": exp.get("horizon"), "signal": exp.get("signal_method"),
                   "data_source": exp.get("source"), "universe": exp.get("universe"), "config_hash": exp.get("config_hash"), "dataset_key": exp.get("dataset_key"), "status": exp.get("status"),
                   "created": exp.get("created"), "snapshot_captured_at": snap["captured_at"], "policy_versions": snap["policy"], "artifact_hashes": {a["name"]: a["sha256"] for a in snap["artifacts"]}},
        "performance": {"segments": (rep.get("segments") or {}).get("stats"), "oos": (rep.get("metrics") or {}).get("oos"), "benchmark": (rep.get("metrics") or {}).get("benchmark"),
                        "ic": {"all": (rep.get("ic") or {}).get("all"), "by_segment": (rep.get("ic") or {}).get("by_segment")}, "label": label or "net of the declared costs, by segment"},
        "uncertainty_selection": {"trials": rep.get("trials"), "overfit": {k: (rep.get("overfit") or {}).get(k) for k in ("generalization_score", "deflated_sharpe_probability", "n_trials", "psr_test_forward")},
                                  "statistical_diagnostics": None if stat is None else {"eligibility": stat.get("eligibility"), "record": present["statistical_diagnostics"]}, "diagnostics_present": present,
                                  "diagnostics_not_computed": absent},
        "coverage": {"dataset": {k: (rep.get("dataset") or {}).get(k) for k in ("n_rows", "n_dates", "n_assets", "n_features", "start", "end", "missing_share")},
                     "scored_coverage": None if alpha is None else alpha.get("coverage"), "audit": rep.get("audit")},
        "costs_capacity": {"execution_mode": ex.get("mode"), "assumptions": ex.get("assumptions"), "price_convention": ex.get("price_convention"), "capacity": ex.get("capacity", "not established"),
                           "delay_report": None if diags.get("delay_report") is None else present["delay_report"], "robustness": rep.get("robustness")},
        "attribution": {"importance": None if not rep.get("importance") else {"table": (rep["importance"].get("table") or [])[:15], "rank_stability": rep["importance"].get("rank_stability")},
                        "note": "feature importance describes the model, not the market; it is not a causal attribution"},
        "lineage": {"lineage": rep.get("lineage"), "provenance": rep.get("provenance"), "reproducibility": rep.get("reproducibility"), "holdout": rep.get("holdout")},
        "gates": {"policy_version": ev.get("policy_version"), "gates": gates, "note": ev.get("note")},
        "limitations": limits,
        "reading": "implementation defects, strategy underperformance, missing data and insufficient prospective evidence are separate things: the gates keep them apart and this report repeats them without merging them",
    }
    return dx._jsonable(out)                                                                           # noqa: SLF001


def build(registry, exp_id: str, *, actor: str, now: float | None = None) -> dict:
    """Snapshot, render, hash and store (append-only). The report id is the hash of its content: building again from unchanged inputs returns the same report, and any change makes a new one."""
    if not actor:
        raise ReportError("a report records who asked for it")
    snap = snapshot(registry, exp_id, now=now)
    content = render(snap)
    stable = {k: v for k, v in snap.items() if k != "captured_at"}                                  # the capture time is not an input: two builds of the same evidence are the same report
    report_id = "rp-" + digest({"inputs": stable, "version": VERSION}, 14)
    existing = _stored(registry, report_id)
    if existing is not None:
        return {"report_id": report_id, "status": "unchanged", "report_hash": existing["report_hash"]}
    rec = {"report_id": report_id, "experiment_id": exp_id, "report_hash": digest(content), "version": VERSION, "snapshot": snap, "content": content,
           "inputs": {"artifacts": {a["name"]: a["sha256"] for a in snap["artifacts"]}, "diagnostics": {k: (None if v is None else {"seq": v["seq"], "content_hash": v["content_hash"]}) for k, v in snap["diagnostics"].items()},
                      "config_hash": snap["experiment"].get("config_hash"), "policy": snap["policy"]}}
    dx.store(registry, KIND, report_id, rec, actor=actor)
    return {"report_id": report_id, "status": "built", "report_hash": rec["report_hash"]}


def _stored(registry, report_id: str) -> dict | None:
    r = dx.latest(registry, report_id, KIND)
    return None if r is None else r["content"]


def get(registry, report_id: str) -> dict:
    """A stored report exactly as it was built: nothing is recomputed."""
    rec = _stored(registry, report_id)
    if rec is None:
        raise ReportError(f"unknown report '{report_id}'")
    return rec


def list_reports(registry, exp_id: str) -> list[dict]:
    """The reports built for an experiment, newest first. A report holds a whole snapshot, so only the rows that name the experiment are read in full (the stored JSON has sorted keys and a fixed layout)."""
    needle = f'"experiment_id": {json.dumps(str(exp_id))}'
    with registry._conn() as c:                                                                      # noqa: SLF001
        rows = c.execute("SELECT subject_id, seq, created, content FROM analysis_records WHERE kind=? AND instr(content, ?) > 0 ORDER BY seq DESC", (KIND, needle)).fetchall()
    out = []
    for r in rows:
        body = json.loads(r["content"])
        if body.get("experiment_id") == exp_id:
            out.append({"report_id": r["subject_id"], "built": r["created"], "report_hash": body["report_hash"], "captured_at": body["snapshot"]["captured_at"]})
    return out


def reproduce(registry, report_id: str) -> dict:
    """Render the stored report again from its own stored snapshot and compare the hashes: the content comes from the recorded inputs and from nothing else."""
    rec = get(registry, report_id)
    again = digest(render(copy.deepcopy(rec["snapshot"])))
    return {"report_id": report_id, "reproduced": again == rec["report_hash"], "stored_hash": rec["report_hash"], "recomputed_hash": again}


def verify_inputs(registry, report_id: str) -> dict:
    """What has changed in the registry since the report was built: artifact files, the latest record of each diagnostic kind, the configuration, the adopted policy. The report itself is not touched."""
    from . import policy as pol

    rec = get(registry, report_id)
    inp, changed, same = rec["inputs"], [], []
    exp = registry.get_experiment(rec["experiment_id"])
    now_art = {a["name"]: _file_hash(a["path"]) for a in exp.get("artifacts", [])}
    for name, h in inp["artifacts"].items():
        (same if now_art.get(name) == h else changed).append({"input": f"artifact:{name}", **({} if now_art.get(name) == h else {"was": h, "now": now_art.get(name)})})
    for kind, was in inp["diagnostics"].items():
        r = dx.latest(registry, rec["experiment_id"], kind)
        now = None if r is None else {"seq": r["seq"], "content_hash": r["content_hash"]}
        (same if now == was else changed).append({"input": f"diagnostic:{kind}", **({} if now == was else {"was": was, "now": now})})
    cfg_now = exp.get("config_hash")
    (same if cfg_now == inp["config_hash"] else changed).append({"input": "config_hash", **({} if cfg_now == inp["config_hash"] else {"was": inp["config_hash"], "now": cfg_now})})
    adopted = pol.adopted(registry)
    p_now = None if adopted is None else adopted[0].hash()
    (same if p_now == inp["policy"]["validation_policy"] else changed).append({"input": "validation_policy", **({} if p_now == inp["policy"]["validation_policy"] else {"was": inp["policy"]["validation_policy"], "now": p_now})})
    return {"report_id": report_id, "changed": changed, "unchanged": [s["input"] for s in same], "stale": bool(changed),
            "note": "the stored report is unchanged by any of this: a newer report would be a new report with a new id"}


def to_markdown(content: dict) -> str:
    """A plain-text rendering of a stored report's content, with the evidence label and the limitations first."""
    h = content["header"]
    lines = [f"# Report on {h.get('name') or h['experiment_id']}", ""]
    if content.get("simulated_data_label"):
        lines += [f"**{content['simulated_data_label']}**", ""]
    lines += [f"* experiment `{h['experiment_id']}`, model {h.get('model')}, target {h.get('target')}, data source {h.get('data_source')}", f"* configuration `{h.get('config_hash')}`, snapshot taken {h['snapshot_captured_at']}, "
              f"report `{content.get('report_version')}`", f"* policies: {json.dumps(h['policy_versions'], sort_keys=True)}", "", "## Limitations and unresolved items", ""]
    lines += [f"* {x}" for x in content["limitations"]] or ["* none recorded"]
    lines += ["", "## Gates", ""]
    for n, g in content["gates"]["gates"].items():
        lines.append(f"* Gate {n}: **{g['status']}**" + ("".join(f"\n    * {r}" for r in g["reasons"])))
    perf = content["performance"]
    lines += ["", "## Performance (" + str(perf["label"]) + ")", "", "```", json.dumps({"segments": perf["segments"], "oos": perf["oos"]}, indent=1, sort_keys=True)[:4000], "```", "",
              "## Uncertainty and selection", "", f"* diagnostics present: {', '.join(content['uncertainty_selection']['diagnostics_present']) or 'none'}", f"* not computed: {', '.join(content['uncertainty_selection']['diagnostics_not_computed']) or 'none'}",
              "", "## Costs and capacity", "", f"* execution mode: {content['costs_capacity']['execution_mode']}", "", "## Reading", "", content["reading"], ""]
    return "\n".join(lines)
