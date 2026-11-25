"""Durable training checkpoints: atomic writes, a fallback when the newest file is damaged, a compatibility check before a run continues, and an honest statement of how exactly it can continue.

A checkpoint holds everything the loop needs to carry on from a finished epoch: the network, the optimizer, the scheduler, the random-number states (PyTorch, CUDA, NumPy, the shuffling generator), the metric history, the best
score and weights' epoch, the early-stopping counter, the fingerprints of the configuration and the data, and the versions it was written with. It is written to a temporary file in the same folder, flushed and synchronised,
read back and verified (a checksum of the payload), and only then does it replace ``last.pt``; the previous ``last.pt`` is kept as ``last.prev.pt`` until the next one is committed, so an interrupted write can never destroy the
only recoverable state.

Recovery levels (``RecoveryLevel``), reported as they are and never inflated:

``exact``            model, optimizer, scheduler, every RNG, shuffling generator, history and early-stopping state were restored on the same software and device: the continued run follows the uninterrupted trajectory under
                     deterministic kernels. NOT claimed across hardware, library versions or nondeterministic kernels.
``epoch_boundary``   training continues from the last finished epoch but something needed for bit-for-bit continuation was missing or differs (a different device or library version, no RNG state).
``warm_start``       only the weights could be restored (no usable optimizer state): training continues from the saved weights with fresh optimizer state.
``restart_required`` nothing safe can be restored (no valid checkpoint, or the model or data changed).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .context import CheckpointIncompatible  # noqa: F401  (re-exported: defined beside TrainingHalted so that the web layer can name it without importing PyTorch)

SCHEMA = 2
RECOVERY = ("exact", "epoch_boundary", "warm_start", "restart_required")


def fingerprint(obj) -> str:
    """A short stable hash of a JSON-able description (architecture, hyperparameters, dataset window)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def software_versions() -> dict:
    import sys

    return {"torch": str(torch.__version__), "numpy": str(np.__version__), "python": sys.version.split()[0], "cuda": None if torch.version.cuda is None else str(torch.version.cuda)}


def rng_states(gen: np.random.Generator | None) -> dict:
    return {"torch": torch.get_rng_state().numpy().tobytes().hex(), "cuda": [s.numpy().tobytes().hex() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [], "numpy_global": _np_state(),
            "generator": None if gen is None else json.loads(json.dumps(gen.bit_generator.state, default=_jsonable))}


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    return str(x)


def _np_state() -> list:
    s = np.random.get_state()
    return [s[0], s[1].tolist(), int(s[2]), int(s[3]), float(s[4])]


def restore_rng(state: dict, gen: np.random.Generator | None) -> None:
    torch.set_rng_state(torch.tensor(list(bytes.fromhex(state["torch"])), dtype=torch.uint8))
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor(list(bytes.fromhex(s)), dtype=torch.uint8) for s in state["cuda"]])
    g = state["numpy_global"]
    np.random.set_state((g[0], np.array(g[1], dtype=np.uint32), g[2], g[3], g[4]))
    if gen is not None and state.get("generator"):
        gen.bit_generator.state = state["generator"]


def _checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class CheckpointManager:
    """The checkpoint folder of one training run. ``keep_milestones`` keeps ``epoch-<n>.pt`` every that many epochs (0: none); ``keep_last`` is the number of ``last`` generations kept (at least 2: the previous one is the fallback)."""

    def __init__(self, folder, keep_milestones: int = 0, keep_last: int = 2, max_bytes: int | None = None):
        self.dir = Path(folder)
        self.keep_milestones, self.keep_last, self.max_bytes = int(keep_milestones), max(int(keep_last), 2), max_bytes

    # ------------------------------------------------------------------------------------------------ writing
    def _atomic(self, path: Path, state: dict) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save(state, buf)
        data = buf.getvalue()
        digest = _checksum(data)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        check = torch.load(io.BytesIO(tmp.read_bytes()), map_location="cpu", weights_only=False)                     # read back what was written before it replaces anything
        if _checksum(tmp.read_bytes()) != digest or check.get("epoch") != state.get("epoch"):
            tmp.unlink(missing_ok=True)
            raise OSError("the checkpoint did not verify after writing; the previous one was kept")
        os.replace(tmp, path)
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(digest)
        return {"path": str(path), "bytes": len(data), "sha256": digest}

    def save(self, state: dict, best: bool = False) -> dict:
        """Commit ``state`` as the newest ``last.pt`` (the previous one becomes ``last.prev.pt``), and as ``best.pt`` when ``best``. Milestones and the disk cap are applied after the commit."""
        state = {**state, "schema": SCHEMA, "saved_at": time.time(), "versions": software_versions()}
        last, prev = self.dir / "last.pt", self.dir / "last.prev.pt"
        info = self._atomic(self.dir / "last.next.pt", state)
        if last.exists():
            os.replace(last, prev)
            if last.with_name("last.pt.sha256").exists():
                os.replace(last.with_name("last.pt.sha256"), prev.with_name("last.prev.pt.sha256"))
        os.replace(self.dir / "last.next.pt", last)
        (self.dir / "last.next.pt.sha256").replace(last.with_name("last.pt.sha256"))
        if best:
            self._atomic(self.dir / "best.pt", state)
        if self.keep_milestones and (int(state["epoch"]) + 1) % self.keep_milestones == 0:
            self._atomic(self.dir / f"epoch-{int(state['epoch']) + 1}.pt", state)
        self._cap()
        return {**info, "epoch": int(state["epoch"]), "path": str(last)}

    def _cap(self) -> None:
        if not self.max_bytes:
            return
        extras = sorted(self.dir.glob("epoch-*.pt"), key=lambda p: p.stat().st_mtime)
        while extras and sum(f.stat().st_size for f in self.dir.glob("*.pt")) > self.max_bytes:
            extras.pop(0).unlink(missing_ok=True)

    # ------------------------------------------------------------------------------------------------ reading
    def _read(self, path: Path) -> dict | None:
        try:
            raw = path.read_bytes()
            side = path.with_name(path.name + ".sha256")
            if side.exists() and side.read_text().strip() != _checksum(raw):
                return None
            blob = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            return blob if isinstance(blob, dict) and "model" in blob and "epoch" in blob else None
        except Exception:                                                                                           # noqa: BLE001 - a damaged file is a result: fall back to the previous one
            return None

    def latest_valid(self) -> tuple[dict | None, str, list[str]]:
        """``(state, which file, problems)``: the newest checkpoint that reads and verifies (``last``, else ``last.prev``, else ``best`` which carries weights only for a warm start)."""
        problems = []
        for name in ("last.pt", "last.prev.pt", "best.pt"):
            p = self.dir / name
            if not p.exists():
                continue
            blob = self._read(p)
            if blob is not None:
                return blob, name, problems
            problems.append(f"{name} is damaged and was skipped")
        return None, "", problems

    def listing(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.pt")) if self.dir.is_dir() else []:
            blob = self._read(p)
            out.append({"file": p.name, "bytes": p.stat().st_size, "valid": blob is not None, "epoch": None if blob is None else int(blob["epoch"]) + 1, "val_loss": None if blob is None else blob.get("val_loss"), "saved_at": None if blob is None else blob.get("saved_at")})
        return out


def assess(blob: dict | None, which: str, config_fp: str, data_fp: str, device: str, allow_warm_start: bool = False) -> dict:
    """Whether the saved state may continue this run and at what recovery level. Raises :class:`CheckpointIncompatible` when the model or the data changed (unless ``allow_warm_start``, which then restores weights only)."""
    if blob is None:
        return {"level": "restart_required", "reasons": ["no valid checkpoint"], "state": None}
    reasons = []
    if blob.get("config_fp") != config_fp:
        reasons.append("the architecture or hyperparameters differ from the saved run")
    if blob.get("data_fp") != data_fp:
        reasons.append("the training data window or features differ from the saved run")
    if blob.get("schema") != SCHEMA:
        reasons.append(f"checkpoint schema {blob.get('schema')} is not {SCHEMA}")
    if reasons:
        if allow_warm_start and blob.get("config_fp") == config_fp:
            return {"level": "warm_start", "reasons": reasons, "state": blob}
        raise CheckpointIncompatible(reasons)
    notes = []
    v = blob.get("versions", {})
    now = software_versions()
    if v.get("torch") != now["torch"] or v.get("numpy") != now["numpy"]:
        notes.append("library versions differ from the run that wrote the checkpoint")
    if blob.get("device") != device:
        notes.append(f"the device changed ({blob.get('device')} to {device})")
    if which == "best.pt":
        return {"level": "warm_start", "reasons": ["only best.pt could be read: weights restored, optimizer, scheduler and counters start fresh"], "state": blob}
    complete = all(blob.get(k) is not None for k in ("optimizer", "scheduler", "rng", "history"))
    if not complete:
        notes.append("the checkpoint lacks optimizer, scheduler, RNG or history state")
    if which != "last.pt":
        notes.append(f"the newest checkpoint was damaged: continuing from {which}")
    return {"level": "exact" if (complete and not notes) else "epoch_boundary", "reasons": notes, "state": blob}
