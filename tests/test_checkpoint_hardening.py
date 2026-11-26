"""The checkpoint files themselves, attacked one way at a time (no training loop needed): every kind of damage a crash, a full disk or a bad copy can leave, and what the reader does with each; interrupted writes and the
leftovers they leave; and how many files a run keeps, when, and what is never deleted.

Damage matrix (the newest ``last.pt`` is damaged, an older ``last.prev.pt`` is good unless stated): truncated, empty, one flipped byte (checksum mismatch), a sidecar checksum that disagrees, a missing sidecar with
an intact payload, garbage that is not a checkpoint, a payload that loads but lacks the model, a deleted ``last.pt``. In every case the reader must return the newest checkpoint that verifies, never raise, and name
what it skipped; when nothing verifies it must say so (restart) or fall back to ``best.pt`` as a weights-only warm start."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import checkpoint as ck, runs            # noqa: E402
from src.ml.deep.checkpoint import CheckpointManager, assess   # noqa: E402
from src.ml.deep.context import CheckpointIncompatible     # noqa: E402


def state(epoch: int, size: int = 64) -> dict:
    return {"model": {"w": torch.full((size,), float(epoch))}, "optimizer": {}, "scheduler": {}, "rng": {}, "history": {}, "epoch": epoch, "config_fp": "c", "data_fp": "d", "device": "cpu"}


def two(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(state(1), best=True)
    mgr.save(state(2), best=False)
    return mgr


# ---------------------------------------------------------------------------------------------------------------------------------------- damage
def truncate(p): p.write_bytes(p.read_bytes()[:40])
def empty(p): p.write_bytes(b"")
def flip(p):
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    p.write_bytes(bytes(raw))
def bad_sidecar(p): p.with_name(p.name + ".sha256").write_text("0" * 64)
def garbage(p): p.write_bytes(os.urandom(2048))
def no_model(p):
    buf = io.BytesIO()
    torch.save({"epoch": 2, "note": "not a checkpoint"}, buf)
    p.write_bytes(buf.getvalue())
    p.with_name(p.name + ".sha256").unlink(missing_ok=True)
def deleted(p): p.unlink()


@pytest.mark.parametrize("damage", [truncate, empty, flip, bad_sidecar, garbage, no_model, deleted], ids=lambda f: f.__name__)
def test_a_damaged_newest_checkpoint_falls_back_to_the_previous_one_and_names_what_it_skipped(damage, tmp_path):
    mgr = two(tmp_path)
    damage(tmp_path / "last.pt")
    blob, which, problems = mgr.latest_valid()
    assert blob is not None and int(blob["epoch"]) == 1 and which == "last.prev.pt"
    if damage is not deleted:
        assert any("last.pt" in p for p in problems), problems
    verdict = assess(blob, which, "c", "d", "cpu")
    assert verdict["level"] == "epoch_boundary" and any("damaged" in r for r in verdict["reasons"]), "never reported as exact"


def test_an_intact_payload_without_a_sidecar_is_still_read(tmp_path):
    mgr = two(tmp_path)
    (tmp_path / "last.pt.sha256").unlink()
    blob, which, problems = mgr.latest_valid()
    assert int(blob["epoch"]) == 2 and which == "last.pt" and not problems


def test_when_both_generations_are_damaged_only_best_survives_as_a_weights_only_warm_start(tmp_path):
    mgr = two(tmp_path)
    flip(tmp_path / "last.pt")
    truncate(tmp_path / "last.prev.pt")
    blob, which, problems = mgr.latest_valid()
    assert which == "best.pt" and len(problems) == 2
    verdict = assess(blob, which, "c", "d", "cpu")
    assert verdict["level"] == "warm_start" and "weights restored" in verdict["reasons"][0]


def test_when_nothing_verifies_the_reader_says_restart(tmp_path):
    mgr = two(tmp_path)
    for f in ("last.pt", "last.prev.pt", "best.pt"):
        truncate(tmp_path / f)
    blob, which, problems = mgr.latest_valid()
    assert blob is None and which == "" and len(problems) == 3
    assert assess(blob, which, "c", "d", "cpu")["level"] == "restart_required"
    assert CheckpointManager(tmp_path / "nowhere").latest_valid() == (None, "", [])


def test_a_checkpoint_of_another_model_or_data_is_refused_not_repaired(tmp_path):
    mgr = two(tmp_path)
    blob, which, _ = mgr.latest_valid()
    with pytest.raises(CheckpointIncompatible):
        assess(blob, which, "another", "d", "cpu")
    with pytest.raises(CheckpointIncompatible):
        assess(blob, which, "c", "another", "cpu")
    assert assess(blob, which, "c", "another", "cpu", allow_warm_start=True)["level"] == "warm_start", "a changed data window may be continued as a warm start when the user asks"
    with pytest.raises(CheckpointIncompatible):
        assess(blob, which, "another", "d", "cpu", allow_warm_start=True)                            # a different MODEL is never continued
    assert not issubclass(CheckpointIncompatible, ValueError), "a walk-forward must not be able to swallow the refusal as a skipped fit"


# ---------------------------------------------------------------------------------------------------------------------------------------- interrupted writes
def test_an_interrupted_commit_between_the_renames_is_completed_not_lost(tmp_path):
    mgr = two(tmp_path)
    mgr.save(state(3))                                                        # 3 is committed: last=3, prev=2
    last, prev, nxt = tmp_path / "last.pt", tmp_path / "last.prev.pt", tmp_path / "last.next.pt"
    mgr._atomic(nxt, state(4))                                                # a 4th checkpoint staged...
    os.replace(last, prev)                                                    # ...and the process dies after the first rename, before the second
    assert not last.exists()
    blob, which, _ = mgr.latest_valid()
    assert int(blob["epoch"]) == 4 and which == "last.next.pt", "the staged file is complete and newest: it is read, not ignored"
    notes = mgr.repair()
    assert any("interrupted checkpoint commit (epoch 5)" in n for n in notes)
    blob, which, _ = mgr.latest_valid()
    assert int(blob["epoch"]) == 4 and which == "last.pt" and not nxt.exists()
    assert int(torch.load(prev, weights_only=False)["epoch"]) == 3, "the generation that was current stays as the fallback"


def test_a_staged_file_that_is_older_than_last_is_discarded(tmp_path):
    mgr = two(tmp_path)
    mgr._atomic(tmp_path / "last.next.pt", state(1))
    assert int(mgr.latest_valid()[0]["epoch"]) == 2
    mgr.repair()
    assert not (tmp_path / "last.next.pt").exists() and int(mgr.latest_valid()[0]["epoch"]) == 2


def test_a_half_written_temporary_file_is_never_read_and_is_removed(tmp_path):
    mgr = two(tmp_path)
    (tmp_path / "last.next.pt.tmp4242").write_bytes(b"PK\x03\x04 half a checkpoint")
    (tmp_path / "best.pt.tmp4242").write_bytes(b"x" * 10)
    assert int(mgr.latest_valid()[0]["epoch"]) == 2
    notes = mgr.repair()
    assert sorted(p.name for p in tmp_path.glob("*.tmp*")) == [] and len(notes) == 2


def test_a_damaged_staged_file_is_removed_and_last_is_kept(tmp_path):
    mgr = two(tmp_path)
    (tmp_path / "last.next.pt").write_bytes(b"not a checkpoint")
    assert int(mgr.latest_valid()[0]["epoch"]) == 2
    assert any("damaged staged" in n for n in mgr.repair()) and not (tmp_path / "last.next.pt").exists()


def test_a_write_that_fails_leaves_no_partial_file_and_does_not_touch_the_existing_checkpoint(tmp_path, monkeypatch):
    mgr = two(tmp_path)
    before = (tmp_path / "last.pt").read_bytes()
    import errno

    real = os.fsync

    def full(fd):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(os, "fsync", full)
    with pytest.raises(OSError):
        mgr.save(state(3))
    monkeypatch.setattr(os, "fsync", real)
    assert (tmp_path / "last.pt").read_bytes() == before and int(mgr.latest_valid()[0]["epoch"]) == 2
    assert list(tmp_path.glob("*.tmp*")) == [] and not (tmp_path / "last.next.pt").exists()


# ---------------------------------------------------------------------------------------------------------------------------------------- retention
def test_a_run_keeps_two_generations_best_and_only_the_milestones_asked_for(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_milestones=3)
    for e in range(1, 10):
        mgr.save(state(e), best=e in (2, 5))
        files = sorted(p.name for p in tmp_path.glob("*.pt"))
        assert {"last.pt"} <= set(files) and set(files) <= {"last.pt", "last.prev.pt", "best.pt", "epoch-3.pt", "epoch-6.pt", "epoch-9.pt"}
    assert int(torch.load(tmp_path / "best.pt", weights_only=False)["epoch"]) == 5                  # the checkpoint written when the last improvement happened
    assert int(torch.load(tmp_path / "last.prev.pt", weights_only=False)["epoch"]) == 8
    assert sorted(p.name for p in tmp_path.glob("epoch-*.pt")) == ["epoch-3.pt", "epoch-6.pt", "epoch-9.pt"]
    assert list(tmp_path.glob("*.tmp*")) == []


def test_the_size_cap_removes_the_oldest_milestones_first_and_never_last_prev_or_best(tmp_path):
    one = len(_blob_bytes(state(1, 4096)))
    mgr = CheckpointManager(tmp_path, keep_milestones=1, max_bytes=int(one * 5.5))
    for e in range(1, 12):
        mgr.save(state(e, 4096), best=e == 1)
    names = {p.name for p in tmp_path.glob("*.pt")}
    assert {"last.pt", "last.prev.pt", "best.pt"} <= names, "the cap never removes what recovery needs"
    milestones = sorted(int(n[6:-3]) for n in names if n.startswith("epoch-"))
    assert milestones and milestones == sorted(milestones) and milestones[-1] == 12 and len(milestones) < 11, milestones      # the newest milestones survive, the oldest went first
    assert sum(p.stat().st_size for p in tmp_path.glob("*.pt")) <= one * 6


def _blob_bytes(s):
    b = io.BytesIO()
    torch.save(s, b)
    return b.getvalue()


def test_a_finished_run_is_trimmed_but_an_interrupted_one_is_never_touched(tmp_path):
    import json

    for name, last_row in (("done", {"type": "finish", "stopped_by": "completed", "at": 3.0}), ("broken", {"type": "interrupted", "epoch": 2, "at": 3.0})):
        fit = tmp_path / name / "fit"
        fit.mkdir(parents=True)
        mgr = CheckpointManager(fit, keep_milestones=1)
        for e in range(1, 4):
            mgr.save(state(e), best=e == 1)
        rows = [{"type": "start", "at": 1.0, "epochs": 3}, {"type": "epoch", "epoch": 2, "epochs": 3, "at": 2.0, "seconds": 1.0, "best_epoch": 1, "val_loss": 1.0}, last_row]
        (fit / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert runs.trim_completed(tmp_path / "done") > 0 and runs.trim_completed(tmp_path / "broken") == 0
    assert sorted(p.name for p in (tmp_path / "done" / "fit").glob("*.pt")) == ["best.pt", "last.pt"]
    assert {"last.pt", "last.prev.pt"} <= {p.name for p in (tmp_path / "broken" / "fit").glob("*.pt")}, "an interrupted run keeps every fallback it has"
