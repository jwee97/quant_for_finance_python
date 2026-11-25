"""The training runs on disk: what the Training Monitor lists, reads and controls.

A run is a folder under ``data/user/training/<run id>/`` holding ``request.json`` (the request that started it, so it can be resumed) and one sub-folder per fit (a walk-forward refits on each window) with the
checkpoints, ``metrics.jsonl`` (one row per epoch plus start, resume, interruption and finish markers) and an optional ``control.json`` written by the monitor. Everything the monitor shows is read from these files,
never from memory: reloading the page, restarting the app or opening the run from a second browser shows the same history.

A run's status is DERIVED from its last row and the age of its file: ``running`` (a recent row and no end marker), ``completed``, ``stopped`` (user stop, pause or early stop are distinguished), ``interrupted`` (an
interruption marker) or ``orphaned`` (no end marker and no activity: the process died without saying so, e.g. an out-of-memory kill).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

ID = re.compile(r"^[A-Za-z0-9_.-]{3,80}$")
FIT = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
STALE_FLOOR = 180.0                      # seconds without a new row before a run with no end marker counts as orphaned
MAX_ROWS = 4000


def base(root) -> Path:
    return Path(root) / "data" / "user" / "training"


def read_rows(folder: Path) -> list[dict]:
    p = folder / "metrics.jsonl"
    rows = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue                                                                      # a half-written last line is skipped, not fatal
    except OSError:
        pass
    return rows[-MAX_ROWS:]


def status_of(rows: list[dict], mtime: float, now: float | None = None) -> str:
    now = now or time.time()
    if not rows:
        return "empty"
    last = rows[-1]
    t = last.get("type")
    if t == "finish":
        return {"completed": "completed", "early_stopping": "stopped_early", "user_stop": "stopped", "user_pause": "paused"}.get(last.get("stopped_by", "completed"), "completed")
    if t == "interrupted":
        return "interrupted"
    seconds = [r["seconds"] for r in rows if r.get("type") == "epoch" and r.get("seconds")]
    limit = max(STALE_FLOOR, 6 * float(np.median(seconds))) if seconds else STALE_FLOOR
    return "running" if now - mtime <= limit else "orphaned"


def summarise(folder: Path, now: float | None = None) -> dict:
    rows = read_rows(folder)
    ep = [r for r in rows if r.get("type") == "epoch"]
    start = next((r for r in reversed(rows) if r.get("type") == "start"), {})
    mtime = (folder / "metrics.jsonl").stat().st_mtime if (folder / "metrics.jsonl").exists() else 0.0
    status = status_of(rows, mtime, now)
    done, total = (ep[-1]["epoch"] if ep else 0), int(start.get("epochs") or (ep[-1]["epochs"] if ep else 0))
    mean = float(np.mean([r["seconds"] for r in ep[-5:]])) if ep else None
    best = min((r["val_loss"] for r in ep if r.get("val_loss") is not None), default=None)
    events = [r for r in rows if r.get("type") not in ("epoch", "start", "finish")]
    resumable = status in ("interrupted", "orphaned", "paused", "stopped") and (folder / "last.pt").exists()
    return {"fit": folder.name, "status": status, "epoch": done, "epochs": total, "best_val": best, "best_epoch": ep[-1]["best_epoch"] if ep else None, "last_at": ep[-1]["at"] if ep else start.get("at"),
            "started_at": start.get("at"), "eta_seconds": None if mean is None or status != "running" else max(total - done, 0) * mean, "elapsed_seconds": float(sum(r["seconds"] for r in ep)) if ep else 0.0,
            "n_params": start.get("n_params"), "device": start.get("device"), "events": events[-20:], "resumable": resumable, "mtime": mtime}


def fit_folders(run: Path) -> list[Path]:
    if (run / "metrics.jsonl").exists():
        return [run]
    return sorted((p for p in run.glob("*") if p.is_dir() and (p / "metrics.jsonl").exists()), key=lambda p: p.stat().st_mtime)


def list_runs(root, now: float | None = None) -> list[dict]:
    b = base(root)
    out = []
    for run in sorted(b.glob("*")) if b.is_dir() else []:
        if not run.is_dir() or run.name.startswith("_") or not ID.match(run.name):
            continue
        fits = [summarise(f, now) for f in fit_folders(run)]
        if not fits:
            continue
        req = {}
        try:
            req = json.loads((run / "request.json").read_text())
        except (OSError, ValueError):
            pass
        newest = max(fits, key=lambda f: f["mtime"])
        out.append({"id": run.name, "label": req.get("label") or run.name, "kind": (req.get("body", {}).get("model") or {}).get("kind"), "n_fits": len(fits), "status": newest["status"], "epoch": newest["epoch"], "epochs": newest["epochs"],
                    "best_val": newest["best_val"], "last_at": newest["last_at"], "resumable": any(f["resumable"] for f in fits), "request_saved": bool(req), "mtime": newest["mtime"], "disk_bytes": _bytes(run)})
    return sorted(out, key=lambda r: -r["mtime"])


def _bytes(p: Path) -> int:
    return int(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()))


def detail(root, run_id: str, fit: str | None = None, since: int = 0, now: float | None = None) -> dict:
    if not ID.match(run_id):
        raise ValueError("invalid run id")
    run = base(root) / run_id
    if not run.is_dir():
        raise KeyError(run_id)
    folders = fit_folders(run)
    if not folders:
        raise KeyError(run_id)
    if fit:
        if not FIT.match(fit):
            raise ValueError("invalid fit name")
        folder = next((f for f in folders if f.name == fit), None)
        if folder is None:
            raise KeyError(fit)
    else:
        folder = folders[-1]
    rows = read_rows(folder)
    ep = [r for r in rows if r.get("type") == "epoch"]
    s = summarise(folder, now)
    keep = ep[int(since):] if since else ep
    series = {"epoch": [r["epoch"] for r in keep], "train_loss": [r["train_loss"] for r in keep], "val_loss": [r["val_loss"] for r in keep], "lr": [r["lr"] for r in keep], "grad_norm": [r["grad_norm"] for r in keep],
              "seconds": [r["seconds"] for r in keep], "rss_mb": [r.get("process_rss_mb") for r in keep], "peak_mb": [r.get("peak_mb") for r in keep], "best_val": [r["best_val"] for r in keep]}
    from .checkpoint import CheckpointManager                                             # PyTorch is imported only when a run's checkpoints are listed

    mgr = CheckpointManager(folder)
    memory = next((e for e in reversed(rows) if e.get("type") == "interrupted" and re.search(r"MemoryError|ResourceLimit|out of memory", str(e.get("reason", "")), re.I)), None)
    advice = None
    if memory is not None and s["status"] in ("interrupted", "orphaned"):
        from . import profiles

        batch = int((((_request(run) or {}).get("body") or {}).get("model") or {}).get("params", {}).get("batch_size", 0) or 0)
        advice = {"reason": str(memory.get("reason")), **(profiles.suggest_smaller(batch) if batch else {"note": "lower the batch size or the lookback, or choose the conservative profile"})}
    return {"id": run_id, "fit": folder.name, "fits": [f.name for f in folders], "summary": s, "memory_advice": advice, "series": series, "n_epochs_total": len(ep), "events": [r for r in rows if r.get("type") not in ("epoch",)], "checkpoints": mgr.listing(),
            "request": _request(run), "control": _control_state(folder)}


def _request(run: Path) -> dict | None:
    try:
        return json.loads((run / "request.json").read_text())
    except (OSError, ValueError):
        return None


def _control_state(folder: Path) -> str:
    for p in (folder / "control.json", folder.parent / "control.json"):
        try:
            return str(json.loads(p.read_text()).get("action", ""))
        except (OSError, ValueError):
            continue
    return ""


def control(root, run_id: str, action: str, fit: str | None = None) -> dict:
    """Ask a running training to ``stop`` or ``pause`` after its current epoch (both commit a checkpoint first, then end the whole experiment; they differ only in what the monitor calls them), or ``clear`` the
    request. The file is the RUN's (``control.json`` beside ``request.json``) so that it also reaches the fits that have not started yet. Cooperative: the loop reads it at the end of each epoch."""
    if action not in ("stop", "pause", "clear"):
        raise ValueError("action must be stop, pause or clear")
    if not ID.match(run_id):
        raise ValueError("invalid run id")
    run = base(root) / run_id
    if not (run / "request.json").exists() and not fit_folders(run):
        raise KeyError(run_id)
    d = {"fit": fit or ""}
    p = run / "control.json"
    if action == "clear":
        p.unlink(missing_ok=True)
        for f in fit_folders(run):
            (f / "control.json").unlink(missing_ok=True)
    else:
        p.write_text(json.dumps({"action": action, "at": time.time()}))
    return {"id": run_id, "fit": d["fit"], "control": "" if action == "clear" else action, "note": "read by the training loop at the end of its current epoch"}


def control_clear(run: Path) -> None:
    (run / "control.json").unlink(missing_ok=True)
    for f in fit_folders(run):
        (f / "control.json").unlink(missing_ok=True)


def archive(root, run_id: str) -> dict:
    if not ID.match(run_id):
        raise ValueError("invalid run id")
    src = base(root) / run_id
    if not src.is_dir():
        raise KeyError(run_id)
    dst = base(root) / "_archive"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / f"{run_id}-{int(time.time())}"
    shutil.move(str(src), str(target))
    return {"id": run_id, "archived_to": target.name}


def disk_usage(root) -> dict:
    b = base(root)
    total = _bytes(b) if b.is_dir() else 0
    try:
        free = shutil.disk_usage(str(b if b.exists() else Path(root))).free
    except OSError:
        free = None
    return {"training_bytes": total, "disk_free_bytes": free}
