"""The completion protocol of a training unit (``src/ml/deep/commit.py``): a process is killed (SIGKILL, no cleanup) at each step between "the unit was computed" and "the unit is announced", and a NEW process resumes.

Whatever the step, the resumed run must: finish with the weights and losses of a run that was never interrupted; hold exactly one metrics row per unit, in order, with no torn line; hold exactly one gradient
summary per unit; never treat a unit the checkpoint does not contain as finished; restore the record of a unit the checkpoint does contain but the logs missed; and say which units were recomputed. Both loops
are covered: ``fit_network`` (the N-BEATS scenario) and ``TrainingSession`` (a forecast GRU)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import commit, runs                                   # noqa: E402
from tests.fresh_process_scenarios import was_killed                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UNITS = {"nbeats": 8, "forecast_gru": 10}
N = 3                                                                  # the unit whose completion is interrupted


def run(name, folder, mode, out, *extra):
    cmd = [sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(folder), mode, str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def rows(path):
    return [json.loads(x) for x in lines(path) if x.strip()]


def fit(folder):
    return next(Path(folder).rglob("metrics.jsonl")).parent


def blob(folder, name):
    return torch.load(io.BytesIO((fit(folder) / name).read_bytes()), map_location="cpu", weights_only=False)


@pytest.fixture(scope="module", params=sorted(UNITS))
def reference(request, tmp_path_factory):
    name = request.param
    base = tmp_path_factory.mktemp(f"ref-{name}")
    out = base / "ref.json"
    r = run(name, base / "ref", "reference", out)
    assert r.returncode == 0, r.stderr[-1500:]
    return name, json.loads(out.read_text())


# stage -> (a unit-N row is restored from the checkpoint, the units that must be reported as recomputed, the state commit.json is left in)
STAGES = {
    "before_checkpoint": dict(recovered_row=False, recomputed=[N], state="applied"),
    "mid_rename": dict(recovered_row=True, recomputed=[], state="applied"),
    "before_meta": dict(recovered_row=True, recomputed=[], state="applied"),
    "after_meta": dict(recovered_row=True, recomputed=[], state="committed"),
    "after_gradient": dict(recovered_row=True, recomputed=[], state="committed"),
    "torn_row": dict(recovered_row=True, recomputed=[], state="committed"),
    "after_row": dict(recovered_row=False, recomputed=[], state="committed"),
    "after_applied": dict(recovered_row=False, recomputed=[], state="applied"),
}


@pytest.mark.parametrize("stage", list(STAGES))
def test_a_kill_at_every_step_of_a_units_completion_resumes_to_the_uninterrupted_result(stage, reference, tmp_path):
    name, ref = reference
    units = UNITS[name]
    expect = STAGES[stage]
    dead = run(name, tmp_path / "run", "interrupted", tmp_path / "dead.json", "--kill-at", f"commit:{stage}:{N}")
    assert was_killed(dead.returncode) and not (tmp_path / "dead.json").exists(), (dead.returncode, dead.stderr[-800:])
    folder = fit(tmp_path / "run")

    # ---- what the monitor and recovery can see right after the kill ---------------------------------------------------------------------------------------------------------------------
    killed_rows = rows(folder / "metrics.jsonl") if stage != "torn_row" else [json.loads(x) for x in lines(folder / "metrics.jsonl")[:-1]]
    if stage == "torn_row":
        last = (folder / "metrics.jsonl").read_text().splitlines()[-1]
        with pytest.raises(ValueError):
            json.loads(last)                                            # the last line really is torn
    if stage == "mid_rename":
        assert not (folder / "last.pt").exists() and (folder / "last.next.pt").exists(), "the commit was interrupted between its two renames"
    durable = max((int(blob(tmp_path / "run", f)["epoch"]) + 1 for f in ("last.pt", "last.next.pt") if (folder / f).exists()), default=0)
    assert durable == (N - 1 if stage == "before_checkpoint" else N), (stage, durable)
    summary = runs.summarise(folder)
    assert summary["committed_epoch"] is None or summary["committed_epoch"] <= durable, "the monitor never claims more than the checkpoint holds"
    logged = [r["epoch"] for r in killed_rows if r.get("type") == "epoch"]
    assert max(logged, default=0) <= durable, "no epoch row exists for a unit the checkpoint does not hold (the row follows the commit)"
    cm = commit.read_commit(folder)
    assert cm is not None and cm["committed_unit"] <= durable

    # ---- the resume ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    res = run(name, tmp_path / "run", "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    got = json.loads((tmp_path / "res.json").read_text())
    np.testing.assert_allclose(got["loss"], ref["loss"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["train_loss"], ref["train_loss"], rtol=1e-6, atol=1e-8)
    for key in ("params", "outputs"):
        if key in ref:
            np.testing.assert_allclose(got[key], ref[key], rtol=1e-6, atol=1e-7)

    text = (folder / "metrics.jsonl").read_text()
    log = [json.loads(x) for x in text.splitlines()]                                          # every line parses: the torn one was repaired before anything was appended
    assert text.endswith("\n")
    epochs = [r for r in log if r["type"] == "epoch"]
    assert [r["epoch"] for r in epochs] == list(range(1, units + 1)) and log[-1]["type"] == "finish", "one row per unit, in order, run closed"
    assert all("committed" in r for r in epochs)
    recovered = [r for r in epochs if r.get("recovered")]
    assert [r["epoch"] for r in recovered] == ([N] if expect["recovered_row"] else []) and all(r["committed"] for r in recovered)
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact" and resume["epoch"] == (N - 1 if stage == "before_checkpoint" else N), resume
    lost = [r for r in log if r["type"] == "recomputed_work"]
    assert [u for r in lost for u in r["units"]] == expect["recomputed"], lost
    if lost:
        assert lost[0]["in_flight"] == N and f"{N}" in lost[0]["message"]
    if stage == "mid_rename":
        assert any(r["type"] == "recovery_note" and "interrupted checkpoint commit" in r["message"] for r in log)
        assert resume["file"] == "last.pt"
    grads = [json.loads(x) for x in lines(folder / "gradients.jsonl")]
    assert [g["epoch"] for g in grads] == list(range(units)), "exactly one gradient summary per unit"
    assert sum(1 for g in grads if g.get("recovered")) == (1 if stage in ("mid_rename", "before_meta", "after_meta") else 0)          # the summary is written before the row, so a kill after it leaves the summary in place

    # ---- the completion record at the end ---------------------------------------------------------------------------------------------------------------------------------------------
    done = commit.read_commit(folder)
    assert done["state"] == "applied" and done["committed_unit"] == done["applied_unit"] == units
    final = runs.summarise(folder)
    assert final["status"] == "completed" and final["epoch"] == units and final["committed_epoch"] == units and final["uncommitted_epochs"] is None
    assert not list(folder.glob("*.tmp*")) and not (folder / "last.next.pt").exists() and not (folder / "inflight.json").exists()


def test_an_epoch_that_was_logged_but_never_checkpointed_is_provisional_and_reported_as_recomputed(tmp_path):
    """With a checkpoint every fourth epoch the rows of epochs 5 to 6 exist when the process dies after epoch 6 but the checkpoint holds epoch 4: the resume runs them again and says so."""
    folder = tmp_path / "run" / "fit"
    folder.mkdir(parents=True)
    # a log in which epochs 5 and 6 were seen but the newest checkpoint is epoch 4
    log = [{"type": "start", "at": 1.0, "epochs": 8}] + [{"type": "epoch", "epoch": e, "epochs": 8, "at": float(e), "committed": e == 4} for e in range(1, 7)]
    log.insert(6, {"type": "gradient_event", "epoch": 3, "kind": "spike", "at": 3.5})                           # unit 4 (0-based 3): kept
    log.append({"type": "gradient_event", "epoch": 4, "kind": "spike", "at": 5.5})                                # unit 5 (0-based 4): dropped
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    (folder / "gradients.jsonl").write_text("".join(json.dumps({"optimizer": "optimizer", "epoch": e}) + "\n" for e in range(6)))
    commit.write_commit(folder, committed=4, applied=4, state="applied")
    report = runs.reconcile_logs(folder, 4, {"row": log[4], "grad": [{"optimizer": "optimizer", "epoch": 3}], "events": []})
    assert report["dropped_units"] == [5, 6] and report["recovered_row"] is False
    kept = rows(folder / "metrics.jsonl")
    assert [r["epoch"] for r in kept if r["type"] == "epoch"] == [1, 2, 3, 4]
    assert [r["at"] for r in kept if r["type"] == "gradient_event"] == [3.5], "the warning of unit 4 stays, the warning of unit 5 goes with its unit"
    assert [g["epoch"] for g in rows(folder / "gradients.jsonl")] == [0, 1, 2, 3]
    work = commit.recompute_report("epoch", 4, report, {"unit": 5, "phase": None})
    assert work["units"] == [5, 6] and work["in_flight"] == 6 and work["logged_not_committed"] == [5]
    assert "epoch 6 had started" in work["message"] and "epoch 5 was logged but never committed" in work["message"]


def test_a_half_written_last_line_is_cut_before_the_next_row_is_appended(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"type": "epoch", "epoch": 1}) + "\n" + '{"type": "epoch", "epo')
    assert commit.repair_tail(p) > 0
    commit.append_line(p, {"type": "epoch", "epoch": 2})
    assert [json.loads(x)["epoch"] for x in p.read_text().splitlines()] == [1, 2]
    assert commit.repair_tail(p) == 0                                                     # a clean file is left alone
    only_torn = tmp_path / "t.jsonl"
    only_torn.write_text('{"a": ')
    assert commit.repair_tail(only_torn) == 6 and only_torn.read_text() == ""


def test_commit_json_is_replaced_atomically_and_a_reader_never_sees_a_partial_file(tmp_path):
    commit.write_commit(tmp_path, committed=1, applied=0, state="committed")
    stop = time.time() + 1.5
    seen = set()

    def writer():
        k = 1
        while time.time() < stop:
            commit.write_commit(tmp_path, committed=k, applied=k, state="applied")
            k += 1
    import threading

    th = threading.Thread(target=writer)
    th.start()
    while time.time() < stop:
        doc = commit.read_commit(tmp_path)
        assert doc is not None and doc["committed_unit"] >= doc["applied_unit"]
        seen.add(doc["committed_unit"])
    th.join()
    assert len(seen) > 5


def test_the_durable_steps_of_a_unit_happen_in_the_documented_order(tmp_path, monkeypatch):
    """checkpoint (the commit point) -> commit.json "committed" -> gradient summary -> metrics row -> commit.json "applied" -> job progress. Recovery never reads job progress; it only ever lags the commit."""
    from src.ml import live
    from src.ml.deep import checkpoint, trainer
    from tests.test_training_recovery import cfg, make

    trace: list[str] = []
    real_save, real_commit, real_log, real_grad, real_epoch = checkpoint.CheckpointManager.save, trainer.write_commit, trainer._log, trainer.append_summary, live.epoch

    def save(self, state, best=False):
        out = real_save(self, state, best)
        trace.append(f"checkpoint:{int(state['epoch']) + 1}")
        return out

    def write(folder, *, committed, applied, state, **kw):
        trace.append(f"commit.json:{state}:{committed}")
        return real_commit(folder, committed=committed, applied=applied, state=state, **kw)

    def log(folder, row):
        if row.get("type") == "epoch":
            trace.append(f"row:{row['epoch']}:{'committed' if row['committed'] else 'provisional'}")
        return real_log(folder, row)

    def grad(folder, row):
        trace.append(f"gradient:{int(row['epoch']) + 1}")
        return real_grad(folder, row)

    def progress(epoch, *a, **k):
        trace.append(f"progress:{epoch + 1}")
        return real_epoch(epoch, *a, **k)

    monkeypatch.setattr(checkpoint.CheckpointManager, "save", save)
    monkeypatch.setattr(trainer, "write_commit", write)
    monkeypatch.setattr(trainer, "_log", log)
    monkeypatch.setattr(trainer, "append_summary", grad)
    monkeypatch.setattr(live, "epoch", progress)
    batch, loss, net, n_tr, n_va = make("nbeats")
    trainer.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path / "run", epochs=2, grad_level="standard"))
    body = trace
    first = body[:6]
    assert first == ["checkpoint:1", "commit.json:committed:1", "gradient:1", "row:1:committed", "commit.json:applied:1", "progress:1"], body
    assert body[6:12] == ["checkpoint:2", "commit.json:committed:2", "gradient:2", "row:2:committed", "commit.json:applied:2", "progress:2"], body
