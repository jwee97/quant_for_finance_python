"""The operating-system level recovery checks of ``scripts/verify_windows_recovery.py`` as tests. They run on whatever platform runs the suite and only prove things about THAT platform: on Linux they are Linux
results, not Windows evidence (the script's own report records the platform). The Task Scheduler check exists only on Windows and is skipped elsewhere. The training and browser checks of the script are covered by
``tests/test_fresh_process_*.py``, ``tests/test_epoch_commit_protocol.py`` and ``tests/test_webapp_training_recovery_browser.py``; run the script itself for the whole list."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("torch")

import scripts.verify_windows_recovery as v   # noqa: E402


@pytest.mark.parametrize("name", ["spawn_worker_kill", "atomic_replace_contention", "sqlite_lock_after_kill", "training_kill_resume"])
def test_the_platform_check_passes(name, tmp_path):
    res = v.CHECKS[name](tmp_path)
    assert res["ok"], res["detail"]


@pytest.mark.skipif(sys.platform != "win32", reason="Task Scheduler exists only on Windows")
def test_task_scheduler_resumes_a_terminated_run(tmp_path):
    res = v.check_task_scheduler_restart(tmp_path)
    assert res["ok"], res["detail"]


def test_the_report_never_claims_windows_verification_from_another_platform(tmp_path):
    out = tmp_path / "r.json"
    assert v.main(["--only", "atomic_replace_contention", "--report", str(out)]) == 0
    import json

    report = json.loads(out.read_text())
    assert report["platform"] == sys.platform and report["summary"]["verified_on_windows"] is False
