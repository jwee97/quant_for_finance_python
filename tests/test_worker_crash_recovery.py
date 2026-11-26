"""A worker crash at the application level: an app trains a deep network as a job in its own process, the process is killed (SIGKILL), a NEW app is started on the same data folder, shows the run as interrupted, and the
user resumes it through the API: the training continues from its checkpoint and the job completes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.ml.deep import runs                                                  # noqa: E402
from tests.app_roots import isolated_root                                     # noqa: E402
from tests.app_worker_scenario import build                                   # noqa: E402
from src.utils.config import load_config                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wait_for(cond, timeout=240.0):
    end = time.time() + timeout
    while time.time() < end:
        got = cond()
        if got:
            return got
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_a_killed_app_leaves_a_resumable_run_and_a_new_app_resumes_it(tmp_path):
    root = isolated_root(tmp_path / "root", load_config())
    child = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "tests.app_worker_scenario", str(root)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        run_dir = runs.base(root) / "crash-run"

        def epochs():
            files = list(run_dir.rglob("metrics.jsonl"))
            return sum(1 for f in files for line in f.read_text().splitlines() if '"type": "epoch"' in line) if files else 0
        wait_for(lambda: epochs() >= 6)
        os.kill(child.pid, signal.SIGKILL)
    finally:
        child.kill()
        child.wait()
    fit = next(run_dir.rglob("metrics.jsonl")).parent
    before = [json.loads(ln) for ln in (fit / "metrics.jsonl").read_text().splitlines()]
    tail = [r for r in before if r["type"] != "gradient_event"]                           # a warning is appended right after its epoch's row, so the kill can land between the two
    assert tail[-1]["type"] == "epoch" and (fit / "last.pt").exists(), "a killed process leaves no end marker, only the committed checkpoints"
    old = time.time() - 3600
    for f in run_dir.rglob("*"):
        if f.is_file():
            os.utime(f, (old, old))                                                       # an hour has passed: no sign of life since
    app = build(root)                                                                     # a NEW application on the same folder
    listing = app.extra_get["/api/training/runs"]({})["runs"]
    row = next(r for r in listing if r["id"] == "crash-run")
    assert row["status"] == "orphaned" and row["resumable"], row                         # shown as dead and resumable; nothing was resumed on its own
    assert before == [json.loads(ln) for ln in (fit / "metrics.jsonl").read_text().splitlines()]
    req = run_dir / "request.json"
    saved = json.loads(req.read_text())
    n_before = len([r for r in before if r["type"] == "epoch"])
    total = n_before + 3
    saved["body"]["model"]["params"]["epochs"] = total                                        # the epochs are an execution setting: shorten them so that the test ends
    req.write_text(json.dumps(saved))
    resp = app.extra_post["/api/training/resume"]({"id": "crash-run", "mode": "resume"})
    job = wait_for(lambda: (j := app.job(resp["job"]))["status"] in ("done", "error") and j)
    assert job["status"] == "done", job.get("error")
    after = [json.loads(ln) for ln in (fit / "metrics.jsonl").read_text().splitlines()]
    resumes = [r for r in after if r["type"] == "resume"]
    assert resumes and resumes[-1]["level"] == "exact" and resumes[-1]["epoch"] >= 6 and resumes[-1]["epoch"] == n_before
    epochs_logged = [r["epoch"] for r in after if r["type"] == "epoch"]
    assert epochs_logged == list(range(1, total + 1)) and after[-1]["type"] == "finish"
    g = [json.loads(ln)["epoch"] for ln in (fit / "gradients.jsonl").read_text().splitlines()]
    assert g == list(range(total))
