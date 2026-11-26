"""Interruption and recovery in FRESH processes, for every deep-training loop of the platform.

For each scenario three processes run: a reference that trains to the end, a worker that is killed with SIGKILL after N finished units (no exception, no cleanup), and a NEW process that restores the newest checkpoint
and finishes. The resumed run must end exactly where the reference ends (the loss history, the outputs and every parameter), its logs must be contiguous with no duplicated or missing unit, and it must say at which
recovery level it continued. Scenario definitions: ``tests/fresh_process_scenarios.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from tests.fresh_process_scenarios import SCENARIOS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KILL_AFTER = {"rl_ppo": 2}
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}


def run(name: str, folder: Path, mode: str, out: Path, kill_after: int = 0) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(folder), mode, str(out)] + (["--kill-after", str(kill_after)] if kill_after else [])
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def close(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, **TOLERANCE)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_killed_worker_resumes_in_a_new_process_and_ends_where_the_uninterrupted_run_ends(name, tmp_path):
    ref_dir, run_dir = tmp_path / "ref", tmp_path / "run"
    ref = run(name, ref_dir, "reference", tmp_path / "ref.json")
    assert ref.returncode == 0, ref.stderr[-2000:]
    dead = run(name, run_dir, "interrupted", tmp_path / "dead.json", KILL_AFTER.get(name, 3))
    assert dead.returncode == -9, f"the worker was not killed before it finished (exit {dead.returncode}); the scenario is too short: {dead.stderr[-800:]}"
    assert not (tmp_path / "dead.json").exists()
    before = {f: len([r for r in rows(f) if r.get("type") == "epoch"]) for f in run_dir.rglob("metrics.jsonl")}
    assert before and all(n >= KILL_AFTER.get(name, 3) for n in before.values()) and list(run_dir.rglob("last.pt")), "no committed checkpoint survived the kill"
    resumed = run(name, run_dir, "resume", tmp_path / "res.json")
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    a, b = json.loads((tmp_path / "ref.json").read_text()), json.loads((tmp_path / "res.json").read_text())
    worst = 0.0
    for key in ("loss", "train_loss", "outputs", "params"):
        if a.get(key) is not None and (len(a[key]) or key in b):
            worst = max(worst, close(a[key], b[key]))
    for key, v in (a.get("extra") or {}).items():
        if isinstance(v, (list, float)) and np.all(np.isfinite(np.asarray(v, dtype=float))):
            worst = max(worst, close(v, b["extra"][key]))
    # the logs of the resumed run: one row per unit, in order, nothing duplicated, nothing missing; the gradient summaries the same
    for f in run_dir.rglob("metrics.jsonl"):
        epochs = [r["epoch"] for r in rows(f) if r.get("type") == "epoch"]
        assert epochs == list(range(1, len(epochs) + 1)), (f, epochs)
        resume_rows = [r for r in rows(f) if r.get("type") == "resume"]
        assert resume_rows and resume_rows[-1]["level"] == "exact", resume_rows
        assert rows(f)[-1]["type"] == "finish"
        g = f.with_name("gradients.jsonl")
        if g.exists():
            by_opt: dict = {}
            for r in rows(g):
                by_opt.setdefault(r["optimizer"], []).append(r["epoch"])
            for opt, eps in by_opt.items():
                assert eps == list(range(len(eps))), (g, opt, eps)
    level = [r for f in run_dir.rglob("metrics.jsonl") for r in rows(f) if r.get("type") == "resume"][-1]
    print(f"{name}: max |reference - resumed| = {worst:.2e}")
    record = os.environ.get("QUANT_RECORD_RECOVERY")
    if record:                                                                                      # one line per scenario, for docs/generated/fresh_process_results.jsonl
        with open(record, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"scenario": name, "killed_after_units": min(before.values()), "resumed_from_unit": level["epoch"], "recovery_level": level["level"], "units_total": max(len([r for r in rows(f) if r.get("type") == "epoch"]) for f in run_dir.rglob("metrics.jsonl")),
                                 "max_abs_difference": worst, "reference_seconds": a.get("seconds"), "resumed_seconds": b.get("seconds"), "worker_exit": dead.returncode}) + "\n")
