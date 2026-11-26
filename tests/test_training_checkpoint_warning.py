"""A checkpoint that could not be written is a warning the monitor must carry: the run's summary distinguishes the latest completed epoch from the latest epoch a restart would resume from (the durable one)."""

from __future__ import annotations

import json

from src.ml.deep import runs


def write(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def epoch(n, committed, **kw):
    return {"type": "epoch", "epoch": n, "epochs": 8, "val_loss": 1.0 / n, "seconds": 1.0, "at": 1e9 + n, "committed": committed, "best_epoch": n, **kw}


def test_failed_checkpoint_writes_are_summarised_with_the_latest_completed_and_the_latest_durable_epoch(tmp_path):
    f = tmp_path / "fit"
    rows = [{"type": "start", "epochs": 8, "at": 1e9}, epoch(1, True), epoch(2, True), epoch(3, False), {"type": "checkpoint_failed", "epoch": 3, "at": 1e9 + 3, "error": "[Errno 28] No space left on device"},
            epoch(4, False), {"type": "checkpoint_failed", "epoch": 4, "at": 1e9 + 4, "error": "[Errno 28] No space left on device"}]
    write(f, rows)
    (f / "commit.json").write_text(json.dumps({"committed_unit": 2, "state": "applied"}))
    s = runs.summarise(f, now=1e9 + 5)
    cf = s["checkpoint_failures"]
    assert cf["count"] == 2 and cf["units"] == [3, 4] and cf["latest_completed"] == 4 and cf["latest_durable"] == 2 and cf["unsafe_units"] == 2 and cf["durable"] is False
    assert "No space left" in cf["last_error"]


def test_a_run_whose_checkpoints_never_failed_has_no_warning_and_a_failed_run_that_later_caught_up_is_durable(tmp_path):
    clean_run = tmp_path / "a"
    write(clean_run, [{"type": "start", "epochs": 3, "at": 1e9}, epoch(1, True), epoch(2, True)])
    assert runs.summarise(clean_run, now=1e9 + 3)["checkpoint_failures"] is None
    done = tmp_path / "b"
    write(done, [{"type": "start", "epochs": 3, "at": 1e9}, epoch(1, True), epoch(2, False), {"type": "checkpoint_failed", "epoch": 2, "at": 1e9 + 2, "error": "full"}, epoch(3, True),
                 {"type": "finish", "stopped_by": "completed", "at": 1e9 + 4}])
    (done / "commit.json").write_text(json.dumps({"committed_unit": 3, "state": "applied"}))
    cf = runs.summarise(done, now=1e9 + 5)["checkpoint_failures"]
    assert cf["count"] == 1 and cf["latest_durable"] == 3 and cf["unsafe_units"] == 0 and cf["durable"] is True


def test_a_run_that_never_wrote_any_checkpoint_has_every_epoch_at_risk(tmp_path):
    f = tmp_path / "c"
    write(f, [{"type": "start", "epochs": 8, "at": 1e9}, epoch(1, False), {"type": "checkpoint_failed", "epoch": 1, "at": 1e9 + 1, "error": "Permission denied"}, epoch(2, False),
              {"type": "checkpoint_failed", "epoch": 2, "at": 1e9 + 2, "error": "Permission denied"}])
    cf = runs.summarise(f, now=1e9 + 3)["checkpoint_failures"]
    assert cf["latest_durable"] == 0 and cf["unsafe_units"] == 2 and "Permission denied" in cf["last_error"]
