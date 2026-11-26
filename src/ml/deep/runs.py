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
    from .commit import read_commit

    cm = read_commit(folder)
    committed = None if cm is None else int(cm.get("committed_unit", 0))                         # what the checkpoint holds, as the training process itself recorded it
    if committed is None and ep and any(r.get("committed") for r in ep):
        committed = max(int(r["epoch"]) for r in ep if r.get("committed"))
    pending = None if committed is None or status in ("completed", "stopped_early") else max(done - committed, 0)
    failed = [r for r in rows if r.get("type") == "checkpoint_failed"]                               # every failed write, not only the newest events kept in ``events``
    unsafe = max(done - (committed or 0), 0) if failed and status not in ("completed", "stopped_early") else 0
    failures = None if not failed else {"count": len(failed), "units": [int(r.get("epoch", 0)) for r in failed][-12:], "last_error": str(failed[-1].get("error", ""))[:300], "latest_completed": done, "latest_durable": committed or 0,
                                        "unsafe_units": unsafe, "durable": unsafe == 0 and status in ("completed", "stopped_early")}
    resumable = status in ("interrupted", "orphaned", "paused", "stopped") and ((folder / "last.pt").exists() or (folder / "last.next.pt").exists())
    return {"fit": folder.name, "status": status, "epoch": done, "epochs": total, "best_val": best, "best_epoch": ep[-1]["best_epoch"] if ep else None, "last_at": ep[-1]["at"] if ep else start.get("at"),
            "started_at": start.get("at"), "eta_seconds": None if mean is None or status != "running" else max(total - done, 0) * mean, "elapsed_seconds": float(sum(r["seconds"] for r in ep)) if ep else 0.0,
            "n_params": start.get("n_params"), "device": start.get("device"), "unit": start.get("unit") or "epoch", "kind": start.get("kind"), "phase": ep[-1].get("phase") if ep else None, "ceiling": start.get("ceiling"), "committed_epoch": committed, "uncommitted_epochs": pending, "checkpoint_failures": failures, "commit_state": None if cm is None else cm.get("state"), "events": events[-20:], "resumable": resumable, "mtime": mtime}


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
        out.append({"id": run.name, "label": req.get("label") or run.name, "kind": (req.get("body", {}).get("model") or {}).get("kind") or newest.get("kind"), "n_fits": len(fits), "status": newest["status"], "epoch": newest["epoch"], "epochs": newest["epochs"],
                    "best_val": newest["best_val"], "last_at": newest["last_at"], "unit": newest.get("unit"), "phase": newest.get("phase"), "resumable": any(f["resumable"] for f in fits), "request_saved": bool(req), "mtime": newest["mtime"], "disk_bytes": _bytes(run)})
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
    series = {"epoch": [r["epoch"] for r in keep], "train_loss": [r.get("train_loss") for r in keep], "val_loss": [r.get("val_loss") for r in keep], "lr": [r.get("lr") for r in keep], "grad_norm": [r.get("grad_norm") for r in keep],
              "seconds": [r.get("seconds") for r in keep], "rss_mb": [r.get("process_rss_mb") for r in keep], "peak_mb": [r.get("peak_mb") for r in keep], "best_val": [r.get("best_val") for r in keep],
              "cpu_pct": [r.get("cpu_pct") for r in keep], "system_used_mb": [r.get("system_used_mb") for r in keep], "gpu_mb": [r.get("gpu_alloc_mb") for r in keep]}
    series["phase"] = [r.get("phase") for r in keep]
    extra = sorted({k for r in keep for k in (r.get("metrics") or {})})
    series["metrics"] = {k: [(r.get("metrics") or {}).get(k) for r in keep] for k in extra}                 # model-specific curves: reconstruction and KL, critic gap, rewards ...
    from .checkpoint import CheckpointManager                                             # PyTorch is imported only when a run's checkpoints are listed

    mgr = CheckpointManager(folder)
    memory = next((e for e in reversed(rows) if e.get("type") == "interrupted" and re.search(r"MemoryError|ResourceLimit|out of memory", str(e.get("reason", "")), re.I)), None)
    advice = None
    if memory is not None and s["status"] in ("interrupted", "orphaned"):
        from . import profiles

        batch = int((((_request(run) or {}).get("body") or {}).get("model") or {}).get("params", {}).get("batch_size", 0) or 0)
        advice = {"reason": str(memory.get("reason")), **(profiles.suggest_smaller(batch) if batch else {"note": "lower the batch size or the lookback, or choose the conservative profile"})}
    return {"id": run_id, "fit": folder.name, "fits": [f.name for f in folders], "summary": s, "memory_advice": advice, "gradient_events": [r for r in rows if r.get("type") == "gradient_event"][-200:], "diagnostics": _diagnostics(folder), "series": series, "n_epochs_total": len(ep), "events": [r for r in rows if r.get("type") not in ("epoch", "gradient_event")], "checkpoints": mgr.listing(),
            "request": _request(run), "control": _control_state(folder)}


def _diagnostics(folder: Path) -> dict | None:
    try:
        return json.loads((folder / "diagnostics.json").read_text())
    except (OSError, ValueError):
        return None


def gradients(root, run_id: str, fit: str | None = None, optimizer: str | None = None, flow_epoch: int | None = None, hist_epochs: list[int] | None = None, max_layers: int = 40) -> dict:
    """The gradient diagnostics of one fit, read from ``gradients.jsonl``: the per-epoch series of the global statistics, the layer table of one epoch (the gradient-flow chart), the per-layer series of the ratios and the
    histograms of chosen epochs. ``flow_epoch`` and ``hist_epochs`` are 1-based epoch numbers."""
    d = detail(root, run_id, fit)
    folder = base(root) / run_id / d["fit"] if d["fit"] != run_id else base(root) / run_id
    if not (folder / "metrics.jsonl").exists():
        folder = base(root) / run_id
    rows = read_rows_all(folder / "gradients.jsonl") or []
    opts = list(dict.fromkeys(r.get("optimizer", "optimizer") for r in rows))
    out = {"id": run_id, "fit": d["fit"], "optimizers": opts, "optimizer": None, "epochs": [], "level": (d.get("diagnostics") or {}).get("level", "off"), "interval": (d.get("diagnostics") or {}).get("interval") or "auto", "family": (d.get("diagnostics") or {}).get("family", "default"),
           "series": {}, "layers": [], "flow": None, "layer_series": {}, "histograms": {}, "note": None}
    if not rows:
        out["note"] = "no gradient summary has been written: the monitoring level was off, or no epoch has finished since it was on"
        return out
    name = optimizer if optimizer in opts else opts[0]
    mine = [r for r in rows if r.get("optimizer", "optimizer") == name]
    out["optimizer"], out["level"] = name, mine[-1].get("level", out["level"])
    out["epochs"] = [r["epoch"] for r in mine]

    def col(f):
        return [f(r) for r in mine]
    g = lambda r, k: (r.get("global") or {}).get(k)                                                    # noqa: E731
    s = lambda r, k: ((r.get("global") or {}).get("sampled") or {}).get(k)                              # noqa: E731
    obs = lambda r: max((g(r, "observed_steps") or 0), 1)                                               # noqa: E731
    out["series"] = {"epoch": [e + 1 for e in out["epochs"]], "l2_median": col(lambda r: g(r, "l2_median")), "l2_mean": col(lambda r: g(r, "l2_mean")), "l2_max": col(lambda r: g(r, "l2_max")), "l2_p90": col(lambda r: g(r, "l2_p90")),
                     "l2_p99": col(lambda r: g(r, "l2_p99")), "l2_cv": col(lambda r: g(r, "l2_cv")), "clip_frequency": col(lambda r: g(r, "clip_frequency")), "clip_value": col(lambda r: g(r, "clip_value")), "post_clip": col(lambda r: g(r, "post_clip_l2_mean")),
                     "nonfinite_steps": col(lambda r: g(r, "nonfinite_steps") or 0), "skipped_steps": col(lambda r: g(r, "skipped_steps") or 0), "skipped_share": col(lambda r: (g(r, "skipped_steps") or 0) / max(g(r, "optimizer_steps") or 1, 1)),
                     "missing_gradients": col(lambda r: g(r, "missing_gradients") or 0), "linf": col(lambda r: s(r, "linf")), "rms": col(lambda r: s(r, "rms")), "mean_abs": col(lambda r: s(r, "mean_abs")), "std": col(lambda r: s(r, "std")),
                     "zero_frac": col(lambda r: s(r, "zero_frac")), "near_zero_frac": col(lambda r: s(r, "near_zero_frac")), "p50": col(lambda r: s(r, "p50")), "p90": col(lambda r: s(r, "p90")), "p95": col(lambda r: s(r, "p95")), "p99": col(lambda r: s(r, "p99"))}
    names = list(dict.fromkeys(l["name"] for r in mine for l in r.get("layers", [])))
    out["layers"] = names
    fe = (int(flow_epoch) - 1) if flow_epoch else out["epochs"][-1]
    pick = next((r for r in mine if r["epoch"] == fe), mine[-1])
    out["flow"] = {"epoch": pick["epoch"], "rows": pick.get("layers", [])}
    keep = names if len(names) <= max_layers else names[:max_layers]
    for nme in keep:
        ser = {"rms": [], "g_w": [], "update_ratio": []}
        for r in mine:
            lay = next((l for l in r.get("layers", []) if l["name"] == nme), None)
            for k in ser:
                ser[k].append(None if lay is None else lay.get(k))
        out["layer_series"][nme] = ser
    have = {r["epoch"]: r["histogram"] for r in mine if r.get("histogram")}
    want = [int(e) - 1 for e in (hist_epochs or [])] or sorted(have)[:1] + sorted(have)[-1:]
    out["histograms"] = {str(e): have[e] for e in dict.fromkeys(want) if e in have}
    out["histogram_epochs"] = sorted(have)
    return out


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


def reconcile_logs(folder, done: int, last: dict | None) -> dict:
    """Make the run's logs agree with the checkpoint that is being resumed (``done`` finished units): see ``commit.reconcile`` for the protocol. Returns what was changed and what is recomputed."""
    from .commit import reconcile

    return reconcile(folder, done, last)


def read_rows_all(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:                                                                    # a half-written last line after a crash
            continue
    return rows


def _rewrite(path: Path, rows: list[dict]) -> None:
    from .commit import atomic_write_text

    atomic_write_text(path, "".join(json.dumps(r, default=float) + "\n" for r in rows))


def control_clear(run: Path) -> None:
    (run / "control.json").unlink(missing_ok=True)
    for f in fit_folders(run):
        (f / "control.json").unlink(missing_ok=True)


def trim_completed(run: Path) -> int:
    """Disk retention for a run that ended normally: the previous-generation checkpoint and the milestones are only fallbacks for a run that may still be interrupted, so they are removed; ``last.pt`` and ``best.pt``
    stay (a finished run can be extended or warm-started). Returns the bytes freed. Interrupted, paused and stopped runs are never touched."""
    freed = 0
    for fit in fit_folders(run):
        if summarise(fit)["status"] not in ("completed", "stopped_early"):
            continue
        for f in list(fit.glob("last.prev.pt*")) + list(fit.glob("epoch-*.pt*")):
            try:
                freed += f.stat().st_size
                f.unlink()
            except OSError:
                pass
    return freed


def prune(root, keep_days: float = 30.0, dry_run: bool = False) -> dict:
    """Remove whole runs that ended normally (completed or stopped early) and have not been touched for ``keep_days`` days, and archived runs older than twice that. A run that can still be resumed is never removed."""
    now, removed, freed = time.time(), [], 0
    cutoff = keep_days * 86400.0
    for r in list_runs(root):
        if r["status"] in ("completed", "stopped_early") and now - r["mtime"] > cutoff:
            removed.append(r["id"])
            freed += r["disk_bytes"]
            if not dry_run:
                shutil.rmtree(base(root) / r["id"], ignore_errors=True)
    arch = base(root) / "_archive"
    for d in sorted(arch.glob("*")) if arch.is_dir() else []:
        try:
            if d.is_dir() and now - d.stat().st_mtime > 2 * cutoff:
                removed.append("_archive/" + d.name)
                freed += _bytes(d)
                if not dry_run:
                    shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    return {"removed": removed, "freed_bytes": int(freed), "dry_run": bool(dry_run), "keep_days": keep_days}


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
