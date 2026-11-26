"""A checkpoint is a pickle: reading one must never run code the installation did not write.

The sidecar hash beside a checkpoint authenticates nothing (whoever can replace the file can replace the hash), so a file is read with ``weights_only=True`` and only an HMAC under this machine's secret allows the
unrestricted load."""

from __future__ import annotations

import io
import os
import pickle

import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import checkpoint as ck, integrity  # noqa: E402


class Marker:
    """An object whose unpickling runs code: it writes a file named in ``PATH``."""

    def __reduce__(self):
        return (os.system, (f"echo owned > {os.environ['QL_MARKER']}",))


def good_state(epoch=3):
    return {"model": {"w": torch.ones(2)}, "optimizer": {}, "epoch": epoch, "rng": {"torch": "00"}, "config_fp": "c", "data_fp": "d"}


def forge(folder, name="last.pt", payload=None):
    """What an attacker with write access to the run folder can do: a malicious pickle with a matching sha256 sidecar."""
    buf = io.BytesIO()
    pickle.dump(payload, buf)
    raw = buf.getvalue()
    (folder / name).write_bytes(raw)
    (folder / (name + ".sha256")).write_text(ck._checksum(raw))


def test_a_forged_checkpoint_with_a_matching_hash_is_not_executed_and_is_reported(tmp_path, monkeypatch):
    marker = tmp_path / "owned.txt"
    monkeypatch.setenv("QL_MARKER", str(marker))
    folder = tmp_path / "run"
    folder.mkdir()
    torch.save({"model": {}, "epoch": 0}, folder / "unused.pt")
    forge(folder, "last.pt", {"model": Marker(), "epoch": 5})
    mgr = ck.CheckpointManager(folder)
    state, which, problems = mgr.latest_valid()
    assert state is None and not marker.exists(), "nothing ran"
    assert any("not signed by this installation" in p for p in problems), problems


def test_a_forged_file_cannot_hide_behind_a_good_older_checkpoint(tmp_path, monkeypatch):
    marker = tmp_path / "owned.txt"
    monkeypatch.setenv("QL_MARKER", str(marker))
    folder = tmp_path / "run"
    mgr = ck.CheckpointManager(folder)
    mgr.save(good_state(1))
    mgr.save(good_state(2))
    forge(folder, "last.pt", {"model": Marker(), "epoch": 9})
    state, which, problems = mgr.latest_valid()
    assert not marker.exists() and which == "last.prev.pt" and int(state["epoch"]) == 1
    assert any("not signed" in p for p in problems)


def test_checkpoints_written_here_load_without_the_unrestricted_loader(tmp_path, monkeypatch):
    folder = tmp_path / "run"
    mgr = ck.CheckpointManager(folder)
    mgr.save(good_state(4), best=True)
    assert (folder / "last.pt.hmac").exists() and (folder / "best.pt.hmac").exists()
    real = torch.load
    calls = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: (calls.append(k.get("weights_only")), real(*a, **k))[1])
    state, which, problems = ck.CheckpointManager(folder).latest_valid()
    assert which == "last.pt" and int(state["epoch"]) == 4 and problems == []
    assert calls and all(c is True for c in calls), "plain data never needs the unrestricted loader"


def test_the_signatures_follow_the_files_through_every_rotation(tmp_path):
    folder = tmp_path / "run"
    mgr = ck.CheckpointManager(folder)
    for e in range(1, 5):
        mgr.save(good_state(e))
    for name in ("last.pt", "last.prev.pt"):
        raw = (folder / name).read_bytes()
        assert integrity.verify(raw, (folder / (name + ".hmac")).read_text()), name
    assert not list(folder.glob("last.next.pt*"))


def test_a_signed_file_that_needs_the_unrestricted_loader_still_loads_and_the_same_bytes_unsigned_do_not(tmp_path):
    folder = tmp_path / "run"
    mgr = ck.CheckpointManager(folder)
    state = {**good_state(2), "extra": Custom()}                      # a class instance: not plain data
    mgr._atomic(folder / "last.pt", state)
    loaded, which, problems = ck.CheckpointManager(folder).latest_valid()
    assert loaded is not None and isinstance(loaded["extra"], Custom)
    (folder / "last.pt.hmac").unlink()
    again = ck.CheckpointManager(folder)
    assert again.latest_valid()[0] is None and again.unsafe


class Custom:
    pass


def test_the_key_is_stable_per_machine_and_a_signature_from_another_key_does_not_verify(tmp_path, monkeypatch):
    monkeypatch.setenv(integrity.ENV, "aa" * 32)
    sig = integrity.sign(b"data")
    assert integrity.verify(b"data", sig) and not integrity.verify(b"datb", sig)
    monkeypatch.setenv(integrity.ENV, "bb" * 32)
    assert not integrity.verify(b"data", sig)
