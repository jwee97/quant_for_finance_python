"""Faults a real deployment meets, injected into real worker processes: the same run stopped and resumed over and over; a kill in the middle of writing a checkpoint; a disk with no space for a checkpoint or for a log
line (simulated ``ENOSPC``, and the kernel's own file-size limit); and the continuity of the gradient diagnostics across all of it.

Everything is compared with a run that was never interrupted: the losses, the weights and the gradient summaries must be the same, because nothing a fault does may change what is trained, only what is written down.
Both training loops are used: ``fit_network`` (the N-BEATS scenario) and ``TrainingSession`` (a forecast GRU)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import commit, runs                              # noqa: E402
from tests.fresh_process_scenarios import was_killed               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UNITS = {"nbeats": 8, "forecast_gru": 10}
NAMES = sorted(UNITS)


def run(name, folder, mode, out, *extra, fault=""):
    env = {**os.environ, "QUANT_FAULT": fault} if fault else {k: v for k, v in os.environ.items() if k != "QUANT_FAULT"}
    cmd = [sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(folder), mode, str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def fit(folder):
    return next(Path(folder).rglob("metrics.jsonl")).parent


@pytest.fixture(scope="module", params=NAMES)
def reference(request, tmp_path_factory):
    name = request.param
    base = tmp_path_factory.mktemp(f"ref-{name}")
    r = run(name, base / "ref", "reference", base / "ref.json")
    assert r.returncode == 0, r.stderr[-1500:]
    return name, json.loads((base / "ref.json").read_text()), fit(base / "ref")


def assert_same_training(got, ref):
    np.testing.assert_allclose(got["loss"], ref["loss"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["train_loss"], ref["train_loss"], rtol=1e-6, atol=1e-8)
    for key in ("params", "outputs"):
        if key in ref:
            np.testing.assert_allclose(got[key], ref[key], rtol=1e-6, atol=1e-7)


def gradient_series(folder):
    """The numbers the gradient monitor wrote for each unit: (optimizer, unit) -> its global statistics."""
    return {(g["optimizer"], g["epoch"]): g["global"] for g in lines(folder / "gradients.jsonl")}


def assert_same_gradients(folder, ref_folder, units):
    a, b = gradient_series(ref_folder), gradient_series(folder)
    assert sorted(b) == sorted(a) and sorted(e for _, e in b) == list(range(units)), "one gradient summary per unit, none missing, none twice"
    for key in a:
        for stat, value in a[key].items():
            if isinstance(value, (int, float)):
                assert b[key][stat] == pytest.approx(value, rel=1e-6, abs=1e-9), (key, stat)             # the diagnostics are continuous across the interruptions


def assert_clean_log(folder, units):
    text = (folder / "metrics.jsonl").read_text()
    log = [json.loads(x) for x in text.splitlines()]                                                    # every line is a whole record
    epochs = [r["epoch"] for r in log if r["type"] == "epoch"]
    assert epochs == list(range(1, units + 1)) and log[-1]["type"] == "finish", epochs
    return log


# ---------------------------------------------------------------------------------------------------------------------------------------- repeated stop and resume
def test_a_run_stopped_and_resumed_three_times_ends_where_the_uninterrupted_run_ends(reference, tmp_path):
    name, ref, ref_folder = reference
    units = UNITS[name]
    d = tmp_path / "run"
    assert was_killed(run(name, d, "interrupted", tmp_path / "x.json", "--kill-after", "2").returncode)
    for k, upto in enumerate((4, 6), 1):
        r = run(name, d, "interrupted_resume", tmp_path / f"x{k}.json", "--kill-after", str(upto))
        assert was_killed(r.returncode), (k, r.returncode, r.stderr[-600:])
    final = run(name, d, "resume", tmp_path / "res.json")
    assert final.returncode == 0, final.stderr[-1500:]
    assert_same_training(json.loads((tmp_path / "res.json").read_text()), ref)
    folder = fit(d)
    log = assert_clean_log(folder, units)
    resumes = [r for r in log if r["type"] == "resume"]
    assert len(resumes) == 3 and all(r["level"] == "exact" for r in resumes), resumes
    assert [r["epoch"] for r in resumes] == sorted(r["epoch"] for r in resumes), "each resume starts at or after the previous one"
    assert resumes[-1]["epoch"] >= 4
    for lost in (r for r in log if r["type"] == "recomputed_work"):
        assert lost["units"] and all(u > 0 for u in lost["units"]) and lost["message"]
    assert_same_gradients(folder, ref_folder, units)
    done = commit.read_commit(folder)
    assert done["state"] == "applied" and done["committed_unit"] == units
    assert not list(folder.glob("*.tmp*")) and not (folder / "inflight.json").exists()


# ---------------------------------------------------------------------------------------------------------------------------------------- killed while writing a checkpoint
@pytest.mark.parametrize("stage", ["mid_write", "after_tmp"])
def test_a_kill_while_a_checkpoint_is_being_written_loses_only_that_unit(stage, reference, tmp_path):
    name, ref, ref_folder = reference
    units = UNITS[name]
    d = tmp_path / "run"
    dead = run(name, d, "interrupted", tmp_path / "x.json", "--kill-at", f"commit:{stage}:3")
    assert was_killed(dead.returncode), dead.stderr[-500:]
    folder = fit(d)
    left = sorted(p.name for p in folder.glob("last.next.pt*"))
    assert any(".tmp" in n for n in left), f"the killed write leaves its temporary file: {left}"
    assert int(torch.load(folder / "last.pt", weights_only=False)["epoch"]) + 1 == 2, "the last committed checkpoint is the previous unit's, untouched"
    res = run(name, d, "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    assert_same_training(json.loads((tmp_path / "res.json").read_text()), ref)
    log = assert_clean_log(folder, units)
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact" and resume["epoch"] == 2
    lost = [r for r in log if r["type"] == "recomputed_work"]
    assert [u for r in lost for u in r["units"]] == [3], lost
    assert any(r["type"] == "recovery_note" and "never finished" in r["message"] for r in log), "the leftover temporary file was removed and the removal reported"
    assert not list(folder.glob("*.tmp*")) and not (folder / "last.next.pt").exists()
    assert_same_gradients(folder, ref_folder, units)


# ---------------------------------------------------------------------------------------------------------------------------------------- no space left
def test_checkpoints_that_cannot_be_written_do_not_stop_training_and_a_later_kill_recomputes_exactly_those_units(reference, tmp_path):
    name, ref, ref_folder = reference
    units = UNITS[name]
    d = tmp_path / "run"
    dead = run(name, d, "interrupted", tmp_path / "x.json", "--kill-after", "4", fault="enospc_ckpt:3-4")
    assert was_killed(dead.returncode), dead.stderr[-500:]
    folder = fit(d)
    before = lines(folder / "metrics.jsonl")
    epochs = [r for r in before if r["type"] == "epoch"]
    assert [(r["epoch"], r["committed"]) for r in epochs] == [(1, True), (2, True), (3, False), (4, False)], "the units whose checkpoint failed are marked provisional"
    assert [r["epoch"] for r in epochs if r["checkpoint"] and "error" in r["checkpoint"]] == [3, 4] and all("No space left" in r["checkpoint"]["error"] for r in epochs[2:])
    assert {r["epoch"] for r in before if r["type"] == "checkpoint_failed"} <= {3, 4}, "the warning row follows the unit's row, so a kill between them may lose the warning (the row itself carries the error)"
    cm = commit.read_commit(folder)
    assert cm["committed_unit"] == 2 and cm["state"] == "applied", "commit.json never claims a unit whose checkpoint was not written"
    summary = runs.summarise(folder)
    assert summary["committed_epoch"] == 2 and summary["epoch"] == 4 and summary["uncommitted_epochs"] == 2, summary
    assert not list(folder.glob("*.tmp*")), "a failed write leaves no partial file behind"
    res = run(name, d, "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    assert_same_training(json.loads((tmp_path / "res.json").read_text()), ref)
    log = assert_clean_log(folder, units)
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact" and resume["epoch"] == 2
    lost = [r for r in log if r["type"] == "recomputed_work"]
    assert len(lost) == 1 and lost[0]["logged_not_committed"] == [3, 4] and "never committed" in lost[0]["message"], lost
    assert lost[0]["units"] in ([3, 4], [3, 4, 5]), "units 3 and 4 were logged without a checkpoint; unit 5 as well when the kill came after it had started"
    assert_same_gradients(folder, ref_folder, units)


def test_a_disk_that_is_full_for_a_while_does_not_change_what_is_trained(reference, tmp_path):
    name, ref, _ = reference
    d = tmp_path / "run"
    r = run(name, d, "reference", tmp_path / "res.json", fault="enospc_ckpt:3-5")
    assert r.returncode == 0, r.stderr[-1500:]
    assert_same_training(json.loads((tmp_path / "res.json").read_text()), ref)
    log = lines(fit(d) / "metrics.jsonl")
    assert [r["epoch"] for r in log if r["type"] == "checkpoint_failed"] == [3, 4, 5]
    committed = {r["epoch"]: r["committed"] for r in log if r["type"] == "epoch"}
    assert [e for e, c in committed.items() if not c] == [3, 4, 5] and committed[6] and committed[UNITS[name]]
    assert commit.read_commit(fit(d))["committed_unit"] == UNITS[name], "once space is back the next checkpoint commits everything up to it"


def test_a_log_line_lost_to_a_full_disk_is_a_gap_in_the_monitor_never_a_failure_of_training(reference, tmp_path):
    name, ref, _ = reference
    d = tmp_path / "run"
    r = run(name, d, "reference", tmp_path / "res.json", fault="enospc_log:3-3")
    assert r.returncode == 0, r.stderr[-1500:]
    assert_same_training(json.loads((tmp_path / "res.json").read_text()), ref)
    log = lines(fit(d) / "metrics.jsonl")
    assert [r["epoch"] for r in log if r["type"] == "epoch"] == [e for e in range(1, UNITS[name] + 1) if e != 3], "the row of unit 3 is missing (documented limit: lost log lines are not rebuilt)"
    assert log[-1]["type"] == "finish" and int(torch.load(fit(d) / "last.pt", weights_only=False)["epoch"]) + 1 == UNITS[name], "the checkpoints, which hold the full histories, are intact"


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE is POSIX")
def test_the_kernels_own_file_size_limit_fails_every_checkpoint_write_and_training_still_finishes(tmp_path):
    name = "nbeats"
    ref = run(name, tmp_path / "ref", "reference", tmp_path / "ref.json")
    assert ref.returncode == 0, ref.stderr[-800:]
    r = run(name, tmp_path / "run", "reference", tmp_path / "res.json", fault="fsize:100000")
    assert r.returncode == 0, r.stderr[-1500:]
    assert_same_training(json.loads((tmp_path / "res.json").read_text()), json.loads((tmp_path / "ref.json").read_text()))
    folder = fit(tmp_path / "run")
    log = lines(folder / "metrics.jsonl")
    failed = [r for r in log if r["type"] == "checkpoint_failed"]
    assert len(failed) == UNITS[name] and all(r.get("error") for r in failed), "every checkpoint write was refused by the operating system and reported"
    assert not (folder / "last.pt").exists() and not list(folder.glob("*.tmp*")), "no partial checkpoint and no temporary file"
    assert all(not r["committed"] for r in log if r["type"] == "epoch") and log[-1]["type"] == "finish"
    summary = runs.summarise(folder)
    assert summary["status"] == "completed" and not summary["resumable"], "a run with no checkpoint is never offered for resume"
