"""Persisted verification evidence for the model-correctness requirements (spec v3, Releases 1 and 2).

A requirement is verified by the test files that check it. ``run_verification`` runs them (pytest in a subprocess, so a crash cannot take the app down), reads the JUnit report, and records one row per
requirement in the registry: the status, the counts, the environment, a hash of the source files the requirement concerns, and the hash and path of the report that proves it. The status rules are strict:

* ``pass``: every selected test ran and passed (none failed, errored or was skipped);
* ``fail``: at least one failed or errored;
* ``blocked``: nothing failed but some tests were skipped (a missing optional package, for instance): a skipped family has not been verified;
* ``not_run``: the test file is missing or collected nothing.

A row is ``stale`` when the source files it hashed have changed since: its green is not shown as current. Nothing in a request can set a status; only a run does.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

from .registry import Registry

ROOT = Path(__file__).resolve().parents[2]
CHECK_VERSION = 1
SOURCES = {
    "rl": ["src/ml/rl", "src/ml/signals.py"], "ranker": ["src/ml/boosting.py"], "graph": ["src/ml/graph.py"], "arch": ["src/ml/deep"], "contracts": ["src/ml/contracts.py", "src/ml/boosting.py", "src/ml/regimes.py"],
    "hmm": ["src/ml/regimes.py"], "serialization": ["src/ml/models.py", "src/ml/deep/model.py", "src/ml/graph.py", "src/ml/rl/agent.py", "src/ml/ensemble_models.py"],
    # Release 2: causal data and training integrity
    "intervals": ["src/ml/intervals.py", "src/ml/training.py", "src/ml/splitters.py", "src/ml/datasets.py", "src/ml/experiments.py", "src/ml/cli.py"],
    "features": ["src/ml/features.py", "src/ml/audit.py", "src/ml/datasets.py", "src/ml/ensembles.py", "src/ml/backtest.py", "src/framework/data.py", "src/assets/bundles.py"],
    "pit": ["src/ml/pit.py", "src/ml/sessions.py", "src/ml/datasets.py", "src/framework/data.py"],
    "causal": ["src/ml/causal.py", "src/ml/training.py", "src/ml/regimes.py", "src/ml/graph.py", "src/ml/rl/agent.py", "src/ml/deep/model.py", "src/ml/ensemble_models.py", "src/ml/models.py"],
    "lineage": ["src/ml/ensemble_models.py", "src/ml/ensembles.py", "src/ml/training.py"],
    "sequences": ["src/ml/deep", "src/ml/graph.py", "src/ml/rl/agent.py", "src/ml/regimes.py", "src/ml/ensemble_models.py", "src/ml/causal.py"],
    "dependence": ["src/ml/intervals.py", "src/ml/training.py", "src/ml/models.py", "src/ml/datasets.py"],
    "bundle": ["src/ml/bundle.py", "src/ml/models.py", "src/ml/causal.py", "src/ml/monitoring.py", "src/ml/experiments.py"],
}
REQUIREMENTS = {                                                # requirement id -> (test files, source group)
    "P0-01": (["tests/test_ml_rl_accounting_reference.py"], "rl"),
    "P0-02": (["tests/test_ml_rl_accounting_reference.py", "tests/test_ml_rl.py"], "rl"),
    "P0-03": (["tests/test_ml_rl_accounting_reference.py"], "rl"),
    "P0-04": (["tests/test_ml_rl_accounting_reference.py"], "rl"),
    "P0-05": (["tests/test_ml_ranker_groups.py"], "ranker"),
    "P0-06": (["tests/test_ml_graph_masks.py"], "graph"),
    "P0-07": (["tests/test_ml_graph_masks.py"], "graph"),
    "P0-08": (["tests/test_ml_architecture_contracts.py"], "arch"),
    "P0-09": (["tests/test_ml_output_contracts.py"], "contracts"),
    "P0-10": (["tests/test_ml_hmm_reference.py"], "hmm"),
    "P0-11": (["tests/test_ml_output_contracts.py"], "contracts"),
    "P0-12": (["tests/test_ml_serialization_matrix.py"], "serialization"),
    "P0-30": (["tests/test_ml_rl_accounting_reference.py"], "rl"),
    "P0-13": (["tests/test_ml_interval_splits.py", "tests/test_ml_dependence.py", "tests/test_ml_product_surfaces.py"], "intervals"),
    "P0-14": (["tests/test_ml_feature_audit.py"], "features"),
    "P0-15": (["tests/test_ml_pit_sessions.py"], "pit"),
    "P0-16": (["tests/test_ml_pit_sessions.py"], "pit"),
    "P0-17": (["tests/test_ml_interval_splits.py"], "intervals"),
    "P0-18": (["tests/test_ml_training_isolation.py", "tests/test_ml_causal_views.py"], "causal"),
    "P0-19": (["tests/test_ml_oof_lineage.py"], "lineage"),
    "P0-20": (["tests/test_ml_inference_parity.py"], "sequences"),
    "P0-21": (["tests/test_ml_dependence.py", "tests/test_ml_interval_splits.py"], "dependence"),
    "P0-22": (["tests/test_ml_inference_bundle.py", "tests/test_ml_product_surfaces.py"], "bundle"),
}


class VerificationError(ValueError):
    """A request for a check that does not exist."""


def source_hash(group: str) -> str:
    """SHA-256 over the paths and bytes of the source files a requirement concerns (a directory contributes every .py file in it)."""
    h = hashlib.sha256()
    for entry in SOURCES[group]:
        path = ROOT / entry
        for f in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            if f.exists():
                h.update(str(f.relative_to(ROOT)).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def environment() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm", "xgboost", "statsmodels", "hmmlearn", "gymnasium", "stable-baselines3", "pytest"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _run_file(path: str, report: Path, timeout: int) -> dict:
    cmd = [sys.executable, "-m", "pytest", path, "-q", "-p", "no:cacheprovider", "-W", "ignore", "-o", "junit_family=xunit2", f"--junitxml={report}", "-p", "no:xdist"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return {"file": path, "passed": 0, "failed": 0, "skipped": 0, "errors": 1, "seconds": float(timeout), "exit_code": -1, "failures": ["timed out"], "skips": []}
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    failures, skips = [], []
    if report.exists():
        for case in ET.parse(report).getroot().iter("testcase"):
            name = case.get("name", "")
            if case.find("failure") is not None:
                counts["failed"] += 1
                failures.append(name)
            elif case.find("error") is not None:
                counts["errors"] += 1
                failures.append(name)
            elif case.find("skipped") is not None:
                counts["skipped"] += 1
                skips.append(f"{name}: {case.find('skipped').get('message', '')}"[:160])
            else:
                counts["passed"] += 1
    return {"file": path, **counts, "seconds": round(time.perf_counter() - t0, 1), "exit_code": code, "failures": failures[:10], "skips": skips[:10]}


def status_of(results: list[dict]) -> tuple[str, str]:
    """The status and the reason from the results of the files that check a requirement."""
    if not results or sum(r["passed"] + r["failed"] + r["skipped"] + r["errors"] for r in results) == 0:
        return "not_run", "no test was collected"
    bad = [f"{r['file']}: {', '.join(r['failures'][:3])}" for r in results if r["failed"] or r["errors"]]
    if bad:
        return "fail", "; ".join(bad)
    skipped = [s for r in results for s in r["skips"] if ": NOT_APPLICABLE" not in s]                  # a skip that says NOT_APPLICABLE (a package that cannot coexist with this environment) is a justified gap, counted apart
    n_na = sum(r["skipped"] for r in results) - len(skipped)
    if skipped:
        return "blocked", f"{len(skipped)} tests were skipped, so what they check was not verified here: {skipped[0]}"
    return "pass", f"{sum(r['passed'] for r in results)} tests passed" + (f"; {n_na} not applicable in this environment (see the skip reasons)" if n_na else "")


def run_verification(registry: Registry, requirements: list[str] | None = None, progress=None, timeout: int = 900) -> list[dict]:
    """Run the tests behind the requirements (all of them by default), record one check per requirement, and return the records."""
    progress = progress or (lambda msg: None)
    wanted = list(requirements or REQUIREMENTS)
    unknown = [r for r in wanted if r not in REQUIREMENTS]
    if unknown:
        raise VerificationError(f"unknown requirement(s) {unknown}; known: {sorted(REQUIREMENTS)}")
    files = sorted({f for r in wanted for f in REQUIREMENTS[r][0]})
    out_dir = registry.root / "verification"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = {}
    for i, f in enumerate(files):
        progress(f"running {f} ({i + 1} of {len(files)})")
        results[f] = _run_file(f, out_dir / f"{stamp}-{Path(f).stem}.xml", timeout) if (ROOT / f).exists() else {"file": f, "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "seconds": 0.0, "exit_code": 4,
                                                                                                         "failures": [], "skips": []}
    env, records = environment(), []
    for req in wanted:
        paths, group = REQUIREMENTS[req]
        mine = [results[p] for p in paths]
        status, reason = status_of(mine)
        report = out_dir / f"{stamp}-{Path(paths[0]).stem}.xml"
        digest = hashlib.sha256(report.read_bytes()).hexdigest() if report.exists() else None
        row = registry.log_check(req, "tests:" + "+".join(Path(p).stem for p in paths), status, CHECK_VERSION, source_hash(group), {"exact": "see the tests: each carries its own tolerances"}, {"files": mine}, env, digest,
                                 str(report) if report.exists() else None, reason)
        records.append({"id": row, "requirement_id": req, "status": status, "reason": reason, "files": mine})
    return records


def current(registry: Registry, requirement_id: str | None = None) -> list[dict]:
    """The latest check of each requirement with whether it is stale (the source it hashed has changed since) and the list of requirements that have never been checked."""
    latest = {}
    for row in registry.list_checks(requirement_id, limit=2000):
        latest.setdefault(row["requirement_id"], row)
    out = []
    for req, (_, group) in REQUIREMENTS.items():
        if requirement_id and req != requirement_id:
            continue
        row = latest.get(req)
        out.append({"requirement_id": req, "status": row["status"] if row else "not_run", "stale": bool(row) and row["subject_hash"] != source_hash(group), "reason": row["reason"] if row else "never run",
                    "created": row["created"] if row else None, "check_id": row["check_id"] if row else None, "environment": row["environment"] if row else None, "artifact_hash": row["artifact_hash"] if row else None,
                    "files": (row["observed"] or {}).get("files") if row else None})
    return out


def as_json(records: list[dict]) -> str:
    return json.dumps(records, indent=2, default=str)
