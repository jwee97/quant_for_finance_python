"""One training loop for every network: device choice, chronological early stopping, checkpoints, gradient clipping, and a record of what the run cost.

The caller supplies the data as two things, ``batch_fn(split, rows)`` (tensors for those rows of the ``"train"`` or ``"val"`` split, already on the device) and ``loss_fn(net, batch)`` (a scalar). The
loop shuffles the training rows each epoch, clips the gradient norm, steps AdamW, halves the learning rate when the validation loss stalls, keeps the best weights and stops after ``patience`` epochs
without an improvement. The validation rows are the caller's choice and must come AFTER the training rows in time (``DeepModel`` cuts the tail of the training window, with a purge gap).

``groups`` (optional) names the group of each row of each split, in practice its day. A batch is then made of WHOLE groups, shuffled as groups, which is what a loss that compares the assets of one day
with each other needs (a daily correlation, a pairwise ranking). Without it the rows are shuffled one by one, as before.
"""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import numpy as np
import torch

from .. import live, resources
from .context import TrainingHalted
from .gradients import GradientMonitor, append_summary, resolve_level
from .runs import reconcile_logs
from .checkpoint import CheckpointManager, assess, restore_rng, rng_states


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0
    num_threads: int = 0
    checkpoint_dir: str | None = None
    resume: bool = False
    min_delta: float = 0.0
    checkpoint_every: int = 1            # epochs between ``last.pt`` writes (the best-validation checkpoint is written whenever the score improves)
    keep_milestones: int = 0             # keep ``epoch-<n>.pt`` every this many epochs (0: none)
    max_checkpoint_bytes: int = 0        # drop the oldest milestones above this size (0: no cap)
    config_fp: str = ""                  # fingerprints of the architecture/hyperparameters and of the training data: a resume is refused when they differ
    data_fp: str = ""
    halt_raises: bool = False            # a stop or pause from the Training monitor raises TrainingHalted (after the checkpoint) so that the experiment around the fit ends too
    allow_warm_start: bool = False       # when the fingerprints differ, restore the weights only (explicitly a new experiment from saved weights)
    accumulation_steps: int = 1          # gradient accumulation: the optimizer steps every this many batches
    amp: bool = False                    # mixed precision on CUDA only
    grad_level: str = "off"              # gradient diagnostics: off, lightweight, standard, detailed, or auto (lightweight on Replit, standard elsewhere, off without a run folder)
    grad_interval: int = 0               # optimizer steps between full samples (0: chosen from the level)
    grad_family: str = "default"         # model family for the warning thresholds (default, transformer, recurrent, gan, rl)


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    epoch_seconds: list = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False
    device: str = "cpu"
    device_note: str = ""
    n_params: int = 0
    seconds: float = 0.0
    peak_memory_mb: float | None = None
    events: list = field(default_factory=list)          # interruptions, resumes, user stops: {type, epoch, at, ...}
    recovery: dict | None = None                        # {level, reasons, resumed_from_epoch, file} when the run continued from a checkpoint
    stopped_by: str = ""                                # "" | early_stopping | user_stop | user_pause

    def to_dict(self) -> dict:
        d = asdict(self)
        d["epochs_run"] = len(self.train_loss)
        return d


def resolve_device(preference: str = "auto") -> tuple[torch.device, str]:
    """``(device, note)``: ``auto`` picks CUDA when PyTorch can see a GPU; asking for CUDA without one falls back to the CPU and says so."""
    pref = (preference or "auto").lower()
    if pref not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if pref == "cpu":
        return torch.device("cpu"), ""
    if torch.cuda.is_available():
        return torch.device("cuda"), ""
    return torch.device("cpu"), "no GPU is visible to PyTorch, so the CPU was used" if pref == "cuda" else ""


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(rss / 2 ** 20 if sys.platform == "darwin" else rss / 1024.0)                  # bytes on macOS, kilobytes on Linux: the process's peak, not the run's own
    except (ImportError, ValueError):
        return None


def save_checkpoint(path: Path, net, opt, epoch: int, best: float, val: float, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "epoch": int(epoch), "best_val": float(best), "val_loss": float(val), "config": asdict(cfg)}, path)


def load_checkpoint(path: Path, net, opt=None, device="cpu") -> dict:
    blob = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(blob["model"])
    if opt is not None and "optimizer" in blob:
        opt.load_state_dict(blob["optimizer"])
    return blob


def batches(n: int, size: int, rng: np.random.Generator | None, groups: np.ndarray | None = None) -> list[np.ndarray]:
    """The rows of each batch. Without ``groups``: ``size`` rows at a time, shuffled when a generator is given. With them: whole groups, added until the batch holds at least ``size`` rows."""
    if groups is None:
        order = rng.permutation(n) if rng is not None else np.arange(n)
        return [order[i:i + size] for i in range(0, n, size)]
    ids, inverse = np.unique(np.asarray(groups)[:n], return_inverse=True)
    by_group = np.argsort(inverse, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inverse, minlength=len(ids)))])
    out, current, held = [], [], 0
    for g in (rng.permutation(len(ids)) if rng is not None else np.arange(len(ids))):
        current.append(by_group[bounds[g]:bounds[g + 1]])
        held += len(current[-1])
        if held >= size:
            out.append(np.concatenate(current))
            current, held = [], 0
    if current:
        out.append(np.concatenate(current))
    return out


def _control(folder: Path | None) -> str:
    """``stop`` or ``pause`` when the Training Monitor asked for it (``control.json`` in the run's folder), else an empty string. Honoured at the end of the current epoch, after its checkpoint."""
    if folder is None:
        return ""
    for path in (folder / "control.json", folder.parent / "control.json" if (folder.parent / "request.json").exists() else None):          # the fit's own file, then the run's: a stop ends the whole experiment, not one of its fits
        if path is None:
            continue
        try:
            return str(json.loads(path.read_text()).get("action", ""))
        except (OSError, ValueError):
            continue
    return ""


def _resource_row(clock: list) -> dict:
    """CPU, memory and (when present) GPU use since the previous call: what the Training monitor's resource panel shows."""
    now = [time.process_time(), time.perf_counter()]
    cpu = 100.0 * (now[0] - clock[0]) / max(now[1] - clock[1], 1e-9)
    clock[:] = now
    row = {"system_used_mb": resources.memory_used_mb(), "limit_mb": resources.memory_limit_mb(), "cpu_pct": round(cpu, 1)}
    if torch.cuda.is_available():
        row["gpu_alloc_mb"] = torch.cuda.memory_allocated() / 1048576.0
    return row


def _log(folder: Path | None, row: dict) -> None:
    """Append one row to the run's ``metrics.jsonl``: the Training Monitor reads this file, so a reloaded page or a restarted app still has the whole history."""
    if folder is None:
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
    except OSError:
        pass


def fit_network(net: torch.nn.Module, batch_fn, loss_fn, n_train: int, n_val: int, cfg: TrainConfig, groups: dict | None = None, aux_fn=None) -> History:
    """Train ``net`` in place and return its history; the net ends up with its best validation weights. ``batch_fn`` must move its tensors to ``net``'s device (``net`` is already there).

    With ``cfg.checkpoint_dir`` every finished epoch is committed as an atomic checkpoint (see ``checkpoint``) and a metrics row is appended to ``metrics.jsonl``; with ``cfg.resume`` the run continues from the newest valid
    checkpoint if its configuration and data fingerprints match, restoring the optimizer, scheduler, RNG states, history and early-stopping counter; ``hist.recovery`` says at which level (exact, epoch_boundary, warm_start)."""
    t0 = time.perf_counter()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = next(net.parameters()).device
    hist = History(device=device.type, n_params=int(sum(p.numel() for p in net.parameters())))
    if n_train < 1:
        raise ValueError("no training rows")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, cfg.patience // 2))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
    folder = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    mgr = CheckpointManager(folder, cfg.keep_milestones, 2, cfg.max_checkpoint_bytes or None) if folder is not None else None
    level = resolve_level(cfg.grad_level, resources.on_replit(), folder is not None) if cfg.grad_level != "off" else "off"
    mon = GradientMonitor("optimizer", dict(net.named_parameters()), level, interval=cfg.grad_interval, family=cfg.grad_family, clip=cfg.grad_clip if cfg.grad_clip > 0 else None, accumulation=max(int(cfg.accumulation_steps), 1)) if level != "off" else None
    start, best, best_state, bad, total_seconds = 0, float("inf"), deepcopy(net.state_dict()), 0, 0.0
    if mgr is not None and cfg.resume:
        blob, which, problems = mgr.latest_valid()
        verdict = assess(blob, which, cfg.config_fp, cfg.data_fp, device.type, cfg.allow_warm_start)          # raises CheckpointIncompatible when the model or the data changed
        if verdict["state"] is not None:
            blob = verdict["state"]
            net.load_state_dict(blob["model"])
            level = verdict["level"]
            if level != "warm_start":
                opt.load_state_dict(blob["optimizer"]) if blob.get("optimizer") is not None else None
                if blob.get("scheduler") is not None:
                    sched.load_state_dict(blob["scheduler"])
                if blob.get("scaler") is not None and scaler.is_enabled():
                    scaler.load_state_dict(blob["scaler"])
                if blob.get("rng") is not None and level == "exact":
                    restore_rng(blob["rng"], rng)
                h = blob.get("history") or {}
                for k in ("train_loss", "val_loss", "lr", "grad_norm", "epoch_seconds"):
                    getattr(hist, k).extend(h.get(k, []))
                hist.best_epoch = int(h.get("best_epoch", 0))
                hist.events.extend(h.get("events", []))
                best, bad, total_seconds = float(blob.get("best_val", best)), int(blob.get("bad", 0)), float(h.get("seconds", 0.0))
                start = int(blob["epoch"]) + 1
                if mon is not None:
                    mon.load_state_dict((blob.get("grad") or {}).get("optimizer"))
                if folder is not None:
                    reconcile_logs(folder, start, blob.get("last"))
                bb = mgr._read(folder / "best.pt")
                best_state = deepcopy(bb["model"]) if bb is not None else deepcopy(net.state_dict())
            else:
                best_state = deepcopy(net.state_dict())
            hist.recovery = {"level": level, "reasons": verdict["reasons"] + problems, "resumed_from_epoch": int(blob["epoch"]) + 1 if level != "warm_start" else 0, "file": which}
            ev = {"type": "resume", "epoch": start, "at": time.time(), "level": level, "reasons": hist.recovery["reasons"], "file": which}
            hist.events.append(ev)
            _log(folder, ev)
        elif problems:
            hist.events.append({"type": "no_valid_checkpoint", "at": time.time(), "problems": problems})
    previous_threads = torch.get_num_threads()
    if cfg.num_threads > 0:
        torch.set_num_threads(int(cfg.num_threads))
    elif previous_threads > resources.default_threads():                                  # a container shows the host's CPUs: more threads than the quota allows fight each other and slow everything
        torch.set_num_threads(resources.default_threads())
        cfg = dataclasses_replace(cfg, num_threads=resources.default_threads())
    live.start(epochs=cfg.epochs, patience=cfg.patience, device=hist.device, n_params=hist.n_params, n_train=int(n_train), n_val=int(n_val), batch_size=cfg.batch_size)
    _log(folder, {"type": "start", "at": time.time(), "epochs": cfg.epochs, "patience": cfg.patience, "device": hist.device, "n_params": hist.n_params, "n_train": int(n_train), "n_val": int(n_val), "batch_size": cfg.batch_size,
                  "config_fp": cfg.config_fp, "data_fp": cfg.data_fp, "resumed_from": start, "diagnostics": {"level": level, "interval": cfg.grad_interval or "auto", "family": cfg.grad_family, "accumulation_steps": max(int(cfg.accumulation_steps), 1), "amp": bool(cfg.amp), "clip": cfg.grad_clip},
                  "unit": "epoch", "optimizers": ["optimizer"], "modules": ["net"]})
    if folder is not None and mon is not None:
        try:
            (folder / "diagnostics.json").write_text(json.dumps({"level": level, "interval": cfg.grad_interval, "family": cfg.grad_family, "replit": resources.on_replit(), "kind": "fit_network"}))
        except OSError:
            pass
    finished = False
    clock = [time.process_time(), time.perf_counter()]
    warned_at = -99
    accum = max(int(cfg.accumulation_steps), 1)
    try:
        if start > 0 and bad >= cfg.patience:                                                       # the saved run had already stopped early: resuming it must not train one more epoch than the original did
            hist.stopped_early, hist.stopped_by = True, "early_stopping"
        for epoch in range(start, start if hist.stopped_early else cfg.epochs):
            e0 = time.perf_counter()
            net.train()
            total, seen, norms = 0.0, 0, []
            plan = batches(n_train, cfg.batch_size, rng, None if groups is None else groups["train"])
            aux_sum: dict = {}
            opt_step = 0
            if mon is not None:
                mon.begin_epoch(epoch, -(-len(plan) // accum), float(opt.param_groups[0]["lr"]))
            opt.zero_grad(set_to_none=True)
            for bi, rows in enumerate(plan):
                with torch.autocast(device_type=device.type, enabled=bool(cfg.amp and device.type == "cuda")):
                    loss = loss_fn(net, batch_fn("train", rows))
                if not torch.isfinite(loss):
                    _log(folder, {"type": "nonfinite_loss", "epoch": epoch, "at": time.time()})
                    raise ValueError("training diverged (the loss is not finite); lower the learning rate")
                scaler.scale(loss / accum).backward()
                if aux_fn is not None:
                    for k_, v_ in aux_fn().items():
                        aux_sum[k_] = aux_sum.get(k_, 0.0) + v_.detach().double() * len(rows)                   # kept on the device until the epoch ends
                if (bi + 1) % accum == 0 or bi + 1 == len(plan):
                    scaler.unscale_(opt)                                                                 # the gradient below is the true one, whatever the mixed-precision scale was
                    if mon is not None:
                        mon.observe(opt_step)                                                            # the completed (accumulated), unscaled, unclipped gradient
                    norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))))
                    if mon is not None:
                        mon.after_clip(opt_step)
                        mon.before_step(opt_step)
                    scale_before = scaler.get_scale() if scaler.is_enabled() else None
                    scaler.step(opt)
                    scaler.update()
                    if mon is not None:
                        mon.after_step(opt_step)
                        if scale_before is not None and scaler.get_scale() < scale_before:               # the scaler found a non-finite gradient and skipped this update
                            mon.note_skipped(opt_step)
                    opt_step += 1
                    opt.zero_grad(set_to_none=True)
                total += float(loss.detach()) * len(rows)
                seen += len(rows)
                live.batch(bi + 1, len(plan), total / max(seen, 1), epoch)                       # how far through the epoch; also where a cancelled run stops
            train_loss = total / max(seen, 1)
            val_loss = _evaluate(net, batch_fn, loss_fn, n_val, cfg.batch_size * 2, None if groups is None else groups["val"]) if n_val else train_loss
            hist.train_loss.append(train_loss)
            hist.val_loss.append(val_loss)
            hist.lr.append(float(opt.param_groups[0]["lr"]))
            hist.grad_norm.append(float(np.mean(norms)) if norms else 0.0)
            hist.epoch_seconds.append(time.perf_counter() - e0)
            sched.step(val_loss)
            improved = val_loss < best - cfg.min_delta
            if improved:
                best, best_state, bad = val_loss, deepcopy(net.state_dict()), 0
                hist.best_epoch = epoch
            else:
                bad += 1
            grad_events: list = []
            if mon is not None:
                gsum, grad_events = mon.end_epoch(epoch, lr=hist.lr[-1], train_loss=train_loss, val_loss=val_loss if n_val else None)
                if gsum is not None:
                    append_summary(folder, gsum)
                hist.events.extend(grad_events)
            action = _control(folder)
            if action in ("stop", "pause"):                                                          # recorded BEFORE the checkpoint so that the saved history carries it
                hist.stopped_by = "user_stop" if action == "stop" else "user_pause"
                hist.events.append({"type": hist.stopped_by, "epoch": epoch + 1, "at": time.time()})
            last_epoch = epoch + 1 >= cfg.epochs or bad >= cfg.patience or action in ("stop", "pause")
            row = {"type": "epoch", "epoch": epoch + 1, "epochs": cfg.epochs, "train_loss": train_loss, "val_loss": val_loss if n_val else None, "lr": hist.lr[-1], "grad_norm": hist.grad_norm[-1], "seconds": hist.epoch_seconds[-1],
                   "best_val": best, "best_epoch": hist.best_epoch + 1, "bad": bad, "patience": cfg.patience, "improved": bool(improved), "at": time.time(), "checkpoint": None, "peak_mb": peak_memory_mb(device),
                   "process_rss_mb": resources.rss_mb(), **_resource_row(clock), "metrics": {k_: float(v_) / max(seen, 1) for k_, v_ in aux_sum.items()}}
            if mgr is not None and (improved or last_epoch or (epoch + 1) % max(int(cfg.checkpoint_every), 1) == 0):
                hist.seconds = total_seconds + time.perf_counter() - t0
                state = {"model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict() if scaler.is_enabled() else None, "epoch": int(epoch), "global_step": int(epoch + 1) * len(plan),
                         "best_val": float(best), "val_loss": float(val_loss), "bad": int(bad), "rng": rng_states(rng), "history": {**{k: list(getattr(hist, k)) for k in ("train_loss", "val_loss", "lr", "grad_norm", "epoch_seconds")},
                         "best_epoch": hist.best_epoch, "events": list(hist.events), "seconds": hist.seconds}, "config_fp": cfg.config_fp, "data_fp": cfg.data_fp, "device": hist.device, "config": asdict(cfg), "grad": {"optimizer": mon.state_dict()} if mon is not None else {}, "last": {"row": row, "grad": [gsum] if mon is not None and gsum else [], "events": list(grad_events)}}
                try:
                    info = mgr.save(state, best=improved)
                    ck = {"epoch": epoch + 1, "bytes": info["bytes"], "path": info["path"], "best": bool(improved)}
                except OSError as exc:                                                              # a full disk must not end the training: say so and carry on without a new checkpoint
                    ck = {"epoch": epoch + 1, "error": str(exc)}
                    hist.events.append({"type": "checkpoint_failed", "epoch": epoch + 1, "at": time.time(), "error": str(exc)})
            else:
                ck = None
            lim_, used_ = row.get("limit_mb"), row.get("system_used_mb")
            if lim_ and used_ and used_ / lim_ > 0.85 and epoch - warned_at >= 5:                         # the machine is nearly out of memory: say so before it kills the run
                warned_at = epoch
                warn = {"type": "resource_warning", "kind": "memory", "epoch": epoch + 1, "at": time.time(), "used_mb": used_, "limit_mb": lim_, "message": f"the machine is using {used_:,.0f} of {lim_:,.0f} MB"}
                grad_events.append(warn)
                hist.events.append(warn)
            row["checkpoint"] = ck
            _log(folder, row)
            for ev_ in grad_events:                                                                     # warnings the gradient monitor raised: persisted next to the metrics, never acted on
                _log(folder, ev_)
            live.epoch(epoch, train_loss, val_loss if n_val else None, lr=hist.lr[-1], grad=hist.grad_norm[-1], seconds=hist.epoch_seconds[-1], best=best, best_epoch=hist.best_epoch, bad=bad, improved=improved,
                       epochs=cfg.epochs, patience=cfg.patience, peak_mb=peak_memory_mb(device))
            if action in ("stop", "pause"):
                _log(folder, hist.events[-1])
                break
            if bad >= cfg.patience:
                hist.stopped_early, hist.stopped_by = True, "early_stopping"
                _log(folder, {"type": "early_stopping", "epoch": epoch + 1, "at": time.time()})
                break
        finished = True
    except BaseException as exc:                                                                    # a cancel, a resource stop or a crash: leave a marker; the last finished epoch is already on disk
        _log(folder, {"type": "interrupted", "epoch": len(hist.train_loss), "at": time.time(), "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
        raise
    finally:
        if cfg.num_threads > 0:
            torch.set_num_threads(previous_threads)
    net.load_state_dict(best_state)
    net.eval()
    hist.seconds = total_seconds + time.perf_counter() - t0
    hist.peak_memory_mb = peak_memory_mb(device)
    _log(folder, {"type": "finish", "at": time.time(), "epochs_run": len(hist.train_loss), "best_epoch": hist.best_epoch + 1, "best_val": None if not np.isfinite(best) else float(best), "stopped_by": hist.stopped_by or "completed", "seconds": hist.seconds})
    live.finish(best_epoch=hist.best_epoch + 1, best_val=None if not np.isfinite(best) else float(best), stopped_early=hist.stopped_early, final_train=hist.train_loss[-1] if hist.train_loss else None, peak_mb=hist.peak_memory_mb)
    if cfg.halt_raises and hist.stopped_by in ("user_stop", "user_pause"):
        raise TrainingHalted("stop" if hist.stopped_by == "user_stop" else "pause", len(hist.train_loss))
    return hist


@torch.no_grad()
def _evaluate(net, batch_fn, loss_fn, n_val: int, chunk: int, groups: np.ndarray | None = None) -> float:
    net.eval()
    total, seen = 0.0, 0
    for rows in batches(n_val, chunk, None, groups):
        total += float(loss_fn(net, batch_fn("val", rows))) * len(rows)
        seen += len(rows)
    return total / max(seen, 1)
