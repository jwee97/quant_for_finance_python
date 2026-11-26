"""One training lifecycle for every iterative PyTorch loop of the platform: restore, report, diagnose, checkpoint, stop, fail.

``fit_network`` (the shared trainer of the feature-based networks) has this logic built in. The other loops (forecast and graph networks, autoencoders, GANs, diffusion, deep hedging, physics-informed networks,
the reinforcement-learning agents) each have their own mathematics and keep it: they call a :class:`TrainingSession` at the same few points instead of re-implementing checkpoints.

    sess = TrainingSession("wgan_gp", config={...}, data=fingerprint, modules={"G": G, "D": D}, optimizers={"g": g_opt, "d": d_opt}, rngs={"numpy": rng},
                           capture=lambda: {"gap": gap_list}, apply=lambda s: gap_list.__init__(s["gap"]), units=n_blocks)
    with sess:
        resumed = sess.restore()                      # None for a fresh start; otherwise what was restored and how exactly
        for unit in range(resumed.start if resumed else 0, n_blocks):
            sess.begin_unit(unit, steps_per_unit)
            ...                                       # the loop's own mathematics, calling mon.observe(step) / mon.after_step(step) after each optimizer step
            if sess.end_unit(unit, {"train_loss": loss, "val_loss": None, "gap": gap}) == "stop":
                break
        sess.finish()

Everything lives in the run's folder (``<run>/<kind>-<fingerprint>/``): the checkpoints (``checkpoint.py``: atomic, verified, a previous one kept), ``metrics.jsonl`` (one row per unit plus markers and events: what
the Training monitor reads), ``gradients.jsonl`` (compact per-unit gradient summaries) and ``diagnostics.json`` (the monitoring profile). With no run active (no job context and no ``folder``) the session does
nothing and costs nothing, so a loop behaves exactly as before.

A checkpoint holds every module, optimizer, scheduler and scaler state, the global and named random-number generators, the loop's own state (``capture`` / ``apply``: histories, best weights, counters,
replay buffers, EMA, schedules), the gradient monitors' detector state and the fingerprints of the configuration and the data. ``ceiling`` states honestly what the loop can promise: ``exact`` only when everything
that influences the next unit is restored; a loop whose environment or data stream cannot be restored passes ``epoch_boundary``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import resources
from . import context as train_context
from .checkpoint import CheckpointManager, assess, fingerprint, restore_rng, rng_states
from .context import TrainingHalted
from .gradients import GradientMonitor, append_summary, resolve_level
from .runs import reconcile_logs
from .trainer import _control, _log


def data_fingerprint(*arrays) -> str:
    """A cheap, stable fingerprint of the training arrays: shapes, dtypes, sums and a strided sample of each. Two different datasets of the same shape practically never collide; an identical one always matches."""
    parts = []
    for a in arrays:
        if a is None:
            parts.append(None)
            continue
        a = np.asarray(a)
        flat = a.reshape(-1)
        sample = flat[:: max(1, flat.size // 2000)]
        fin = sample[np.isfinite(sample)] if np.issubdtype(sample.dtype, np.floating) else sample
        parts.append([list(a.shape), str(a.dtype), float(np.nansum(a.astype(np.float64))) if a.size else 0.0, float(np.sum(np.abs(fin.astype(np.float64)))) if fin.size else 0.0, [round(float(x), 9) for x in fin[:64]]])
    return fingerprint(parts)


def trainer_settings(params: dict, kind: str, config, data, family: str = "default") -> dict:
    """Keyword arguments for ``TrainConfig`` that connect a ``fit_network`` call to the active run: its checkpoint folder (``<run>/<kind>-<fingerprint>``), resume and warm-start flags, the fingerprints, the checkpoint
    cadence and the gradient-monitoring level. With no run active nothing is checkpointed and the monitoring stays off unless the model's own ``grad_monitor`` parameter asks for it."""
    ctx = train_context.current()
    config_fp, data_fp = fingerprint(config), fingerprint(data)
    folder = train_context.loop_folder(kind, config_fp, data_fp)
    level = str(params.get("grad_monitor") or "auto")
    if level == "auto" and ctx and ctx.get("grad_level") not in (None, "auto"):
        level = ctx["grad_level"]
    return {"checkpoint_dir": str(folder) if folder else None, "resume": bool(ctx and ctx["resume"]), "allow_warm_start": bool(ctx and ctx.get("warm")), "halt_raises": bool(ctx), "config_fp": config_fp, "data_fp": data_fp,
            "checkpoint_every": int(params.get("checkpoint_every", 1) or 1), "keep_milestones": int(params.get("keep_milestones", 0) or 0), "grad_level": level if (folder or level != "auto") else "off",
            "grad_interval": int(params.get("grad_interval", 0) or 0), "grad_family": family}


class Callback:
    """Optional lifecycle hooks; every method may be left out. They only observe: a callback must not change the training."""

    def on_start(self, session, info: dict): ...
    def on_restore(self, session, info: dict): ...
    def on_unit_start(self, session, unit: int): ...
    def on_unit_end(self, session, unit: int, row: dict): ...
    def on_checkpoint(self, session, unit: int, info: dict): ...
    def on_event(self, session, event: dict): ...
    def on_resource_warning(self, session, warning: dict): ...
    def on_stop(self, session, action: str): ...
    def on_failure(self, session, error: BaseException): ...
    def on_finish(self, session, info: dict): ...


class NullMonitor:
    """The monitor of a loop with diagnostics off: the same calls, no work."""

    level = "off"

    def __getattr__(self, name):
        if name in ("begin_epoch", "observe", "after_clip", "before_step", "after_step", "note_skipped"):
            return lambda *a, **k: None
        raise AttributeError(name)

    def sample_due(self, step: int) -> bool:
        return False

    def end_epoch(self, *a, **k):
        return None, []

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        pass


class Resumed:
    def __init__(self, start: int, level: str, reasons: list[str], file: str, state: dict):
        self.start, self.level, self.reasons, self.file, self.state = start, level, reasons, file, state

    def as_dict(self) -> dict:
        return {"start": self.start, "level": self.level, "reasons": self.reasons, "file": self.file}


def _gen_state(gen) -> dict:
    if isinstance(gen, np.random.Generator):
        return {"type": "numpy", "state": json.loads(json.dumps(gen.bit_generator.state, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x) if isinstance(x, np.integer) else str(x)))}
    if isinstance(gen, torch.Generator):
        return {"type": "torch", "state": gen.get_state().numpy().tobytes().hex()}
    if isinstance(gen, np.random.RandomState):
        s = gen.get_state()
        return {"type": "randomstate", "state": [s[0], s[1].tolist(), int(s[2]), int(s[3]), float(s[4])]}
    raise TypeError(f"cannot save the state of a {type(gen).__name__}")


def _gen_restore(gen, st: dict) -> None:
    if st["type"] == "numpy":
        gen.bit_generator.state = st["state"]
    elif st["type"] == "torch":
        gen.set_state(torch.tensor(list(bytes.fromhex(st["state"])), dtype=torch.uint8))
    else:
        g = st["state"]
        gen.set_state((g[0], np.array(g[1], dtype=np.uint32), g[2], g[3], g[4]))


class TrainingSession:
    def __init__(self, kind: str, *, config: dict, data, modules: dict, optimizers: dict | None = None, schedulers: dict | None = None, scaler=None, rngs: dict | None = None, capture=None, apply=None,
                 units: int, unit_name: str = "epoch", checkpoint_every: int = 1, family: str = "default", grad_level: str | None = None, grad_interval: int = 0, clip: float | None = None, accumulation: int = 1,
                 ceiling: str = "exact", device: str = "cpu", folder=None, resume: bool | None = None, callbacks: list | None = None, patience: int | None = None, keep_milestones: int = 0, steps_per_unit: int | None = None, n_params: int | None = None, lr: float | None = None):
        self.kind, self.modules, self.optimizers, self.schedulers, self.scaler = kind, dict(modules), dict(optimizers or {}), dict(schedulers or {}), scaler
        self.rngs, self.capture, self.apply, self.units, self.unit_name = dict(rngs or {}), capture, apply, int(units), unit_name
        self.checkpoint_every, self.family, self.ceiling, self.device, self.patience = max(int(checkpoint_every), 1), family, ceiling, device, patience
        self.config_fp, self.data_fp = fingerprint(config), fingerprint(data)
        ctx = train_context.current()
        self.ctx = ctx
        self.dir = Path(folder) if folder is not None else train_context.loop_folder(kind, self.config_fp, self.data_fp)
        self.resume_requested = bool(resume) if resume is not None else bool(ctx and ctx["resume"])
        self.allow_warm = bool(ctx and ctx.get("warm"))
        self.halt_raises = bool(ctx)
        self.callbacks = list(callbacks or [])
        self.mgr = CheckpointManager(self.dir, keep_milestones, 2) if self.dir is not None else None
        requested = grad_level if grad_level is not None else ((ctx or {}).get("grad_level") or "auto")
        self.grad_level = resolve_level(requested, resources.on_replit(), self.dir is not None)
        self.grad_interval, self.clip, self.accumulation = int(grad_interval), clip, int(accumulation)
        self.monitors: dict[str, GradientMonitor] = {}
        self.steps_per_unit = steps_per_unit
        self._n_params_given, self._lr_given = n_params, lr
        self.history_rows = 0
        self.halted: str | None = None
        self.resumed: Resumed | None = None
        self.best_val = float("inf")
        self.best_unit = -1
        self._t_unit = 0.0
        self._t_start = time.perf_counter()
        self._seconds_before = 0.0
        self._cpu_last = (time.process_time(), time.perf_counter())
        self._warned = -99
        self._started = False
        self._phase = None

    # ------------------------------------------------------------------------------------------------------ plumbing
    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _emit(self, hook: str, *args) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if fn is not None:
                try:
                    fn(self, *args)
                except Exception:                                                                 # noqa: BLE001 - a callback only observes
                    pass

    def log(self, row: dict) -> None:
        _log(self.dir, row)

    def monitor(self, name: str, params, *, clip: float | None = None, accumulation: int | None = None, family: str | None = None):
        """The gradient monitor of one optimizer (``params`` is ``{name: Parameter}`` or a module). A loop with several optimizers asks once per optimizer."""
        if self.grad_level == "off":
            return NullMonitor()
        if isinstance(params, torch.nn.Module):
            params = dict(params.named_parameters())
        mon = GradientMonitor(name, params, self.grad_level, interval=self.grad_interval, family=family or self.family, clip=self.clip if clip is None else clip, accumulation=accumulation or self.accumulation)
        self.monitors[name] = mon
        return mon

    # ------------------------------------------------------------------------------------------------------ start and restore
    def restore(self) -> Resumed | None:
        """Continue from the newest valid checkpoint when a resume was requested; ``None`` (a fresh start) otherwise. Raises ``CheckpointIncompatible`` when the saved run is not this one."""
        info = {"kind": self.kind, "units": self.units, "unit_name": self.unit_name, "grad_level": self.grad_level, "config_fp": self.config_fp, "data_fp": self.data_fp, "resume": self.resume_requested}
        self._emit("on_start", info)
        out = None
        if self.mgr is not None and self.resume_requested:
            blob, which, problems = self.mgr.latest_valid()
            verdict = assess(blob, which, self.config_fp, self.data_fp, self.device, self.allow_warm)
            if verdict["state"] is not None:
                blob, level, reasons = verdict["state"], verdict["level"], verdict["reasons"] + problems
                for n, m in self.modules.items():
                    m.load_state_dict(blob["model"][n])
                if level != "warm_start":
                    for n, o in self.optimizers.items():
                        o.load_state_dict(blob["optimizer"][n])
                    for n, s in self.schedulers.items():
                        s.load_state_dict(blob["scheduler"][n])
                    if self.scaler is not None and blob.get("scaler"):
                        self.scaler.load_state_dict(blob["scaler"])
                    if level == "exact":
                        restore_rng(blob["rng"], None)
                        for n, g in self.rngs.items():
                            if n in blob["rng"].get("gens", {}):
                                _gen_restore(g, blob["rng"]["gens"][n])
                    if self.apply is not None:
                        self.apply(blob["history"]["extra"])
                    for n, mon in self.monitors.items():
                        mon.load_state_dict((blob.get("grad") or {}).get(n))
                    self._seconds_before = float(blob["history"].get("seconds", 0.0))
                    self.best_val, self.best_unit = float(blob["history"].get("best_val", float("inf"))), int(blob["history"].get("best_unit", -1))
                    start = int(blob["epoch"]) + 1
                    if self.dir is not None:
                        reconcile_logs(self.dir, start, blob.get("last"))
                else:
                    start = 0
                if level == "exact" and self.ceiling != "exact":
                    level, reasons = "epoch_boundary", reasons + [f"this loop cannot restore everything that shapes the next {self.unit_name} (its environment or data stream): the continuation is approximate"]
                out = self.resumed = Resumed(start, level, reasons, which, blob)
                lost = self._read_inflight(start)
                if lost:
                    self.log({"type": "recomputed_work", "epoch": start, "at": time.time(), "unit": lost["unit"] + 1, "phase": lost.get("phase"),
                              "message": f"{self.unit_name} {lost['unit'] + 1}{' (' + lost['phase'] + ')' if lost.get('phase') else ''} had started but was not committed when the process ended: it is recomputed from the last checkpoint"})
                ev = {"type": "resume", "epoch": start, "at": time.time(), "level": level, "reasons": reasons, "file": which}
                self.log(ev)
                self._emit("on_restore", out.as_dict())
            elif problems:
                self.log({"type": "no_valid_checkpoint", "at": time.time(), "problems": problems})
        return out

    def _read_inflight(self, start: int) -> dict | None:
        if self.dir is None:
            return None
        path = self.dir / "inflight.json"
        try:
            note = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        path.unlink(missing_ok=True)
        return note if int(note.get("unit", -1)) >= start else None

    def _ensure_started(self) -> None:
        if self._started or self.dir is None:
            self._started = True
            return
        self._started = True
        n_params = self._n_params_given if self._n_params_given is not None else int(sum(p.numel() for m in self.modules.values() for p in m.parameters()))
        self.log({"type": "start", "at": time.time(), "epochs": self.units, "unit": self.unit_name, "kind": self.kind, "patience": self.patience, "device": self.device, "n_params": n_params, "batch_size": None,
                  "config_fp": self.config_fp, "data_fp": self.data_fp, "resumed": self.resumed is not None, "ceiling": self.ceiling, "diagnostics": {"level": self.grad_level, "interval": self.grad_interval or "auto", "family": self.family},
                  "optimizers": list(self.optimizers), "modules": list(self.modules)})
        try:
            (self.dir / "diagnostics.json").write_text(json.dumps({"level": self.grad_level, "interval": self.grad_interval, "family": self.family, "replit": resources.on_replit(), "kind": self.kind}))
            req = self.dir.parent / "request.json"
            if self.ctx and self.ctx.get("request") and not req.exists():
                req.write_text(json.dumps(self.ctx["request"], default=str))
        except OSError:
            pass

    # ------------------------------------------------------------------------------------------------------ units
    def begin_unit(self, unit: int, steps: int | None = None, phase: str | None = None) -> None:
        self._ensure_started()
        self._t_unit = time.perf_counter()
        self._phase = phase
        self._mark_inflight(unit, phase)
        for mon in self.monitors.values():
            mon.begin_epoch(unit, steps or self.steps_per_unit, self._lr())
        self._emit("on_unit_start", unit)

    def _mark_inflight(self, unit: int, phase: str | None) -> None:
        """A durable note that ``unit`` has started and is not committed. A hard kill leaves it behind, and the next resume reports the work after the last checkpoint as recomputed."""
        if self.dir is None:
            return
        try:
            (self.dir / "inflight.json").write_text(json.dumps({"unit": int(unit), "phase": phase, "at": time.time()}))
        except OSError:
            pass

    def _clear_inflight(self) -> None:
        if self.dir is not None:
            try:
                (self.dir / "inflight.json").unlink(missing_ok=True)
            except OSError:
                pass

    def _lr(self):
        if self._lr_given is not None:
            return float(self._lr_given)
        if self._phase in self.optimizers:                                                         # a run with phases (Adam, then L-BFGS) reports the learning rate of the optimizer that is working
            return float(self.optimizers[self._phase].param_groups[0]["lr"])
        for o in self.optimizers.values():
            return float(o.param_groups[0]["lr"])
        return None

    def _resource_row(self) -> dict:
        now = (time.process_time(), time.perf_counter())
        wall = max(now[1] - self._cpu_last[1], 1e-9)
        cpu = 100.0 * (now[0] - self._cpu_last[0]) / wall
        self._cpu_last = now
        row = {"process_rss_mb": resources.rss_mb(), "system_used_mb": resources.memory_used_mb(), "limit_mb": resources.memory_limit_mb(), "cpu_pct": round(cpu, 1)}
        if torch.cuda.is_available():
            row["gpu_alloc_mb"] = torch.cuda.memory_allocated() / 1048576.0
            row["peak_mb"] = torch.cuda.max_memory_allocated() / 1048576.0
        return row

    def end_unit(self, unit: int, metrics: dict, improved: bool | None = None, last: bool = False, phase: str | None = None) -> str:
        """Close a unit: summarise the gradients, save a checkpoint when due, write the metrics row, check the monitor's stop and pause requests. Returns ``stop`` when the loop must break."""
        self._ensure_started()
        seconds = time.perf_counter() - self._t_unit
        train, val = metrics.get("train_loss"), metrics.get("val_loss")
        if val is not None and np.isfinite(val) and val < self.best_val:
            self.best_val, self.best_unit = float(val), unit
            improved = True if improved is None else improved
        events: list[dict] = []
        grad_norm = None
        summaries: list[dict] = []
        for name, mon in self.monitors.items():
            summary, ev = mon.end_epoch(unit, lr=self._lr(), train_loss=train, val_loss=val, action="continued")
            if summary is not None:
                summaries.append(summary)
                append_summary(self.dir, summary)
                grad_norm = grad_norm if grad_norm is not None else summary["global"].get("l2_median")
            events += ev
        action = _control(self.dir) if self.dir is not None else ""
        if action in ("stop", "pause"):
            self.halted = "user_stop" if action == "stop" else "user_pause"
            events.append({"type": self.halted, "epoch": unit + 1, "at": time.time()})
        res = self._resource_row()
        warn = None
        lim, used = res.get("limit_mb"), res.get("system_used_mb")
        if lim and used and used / lim > 0.85 and unit - self._warned >= 5:
            self._warned = unit
            warn = {"type": "resource_warning", "kind": "memory", "epoch": unit + 1, "at": time.time(), "used_mb": used, "limit_mb": lim, "message": f"the machine is using {used:,.0f} of {lim:,.0f} MB"}
            events.append(warn)
            self._emit("on_resource_warning", warn)
        row = {"type": "epoch", "epoch": unit + 1, "epochs": self.units, "unit": self.unit_name, "train_loss": train, "val_loss": val, "lr": self._lr(), "grad_norm": grad_norm, "seconds": seconds,
               "best_val": None if not np.isfinite(self.best_val) else self.best_val, "best_epoch": self.best_unit + 1, "bad": None, "patience": self.patience, "improved": bool(improved), "at": time.time(), "checkpoint": None,
               "metrics": {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in metrics.items() if k not in ("train_loss", "val_loss")}, "phase": phase or getattr(self, "_phase", None), **res}
        done = last or unit + 1 >= self.units or bool(self.halted)
        due = (unit + 1) % self.checkpoint_every == 0
        ck = None
        if self.mgr is not None and (due or improved or done):
            try:
                info = self.mgr.save(self._state(unit, metrics, {"row": row, "grad": summaries, "events": events}), best=bool(improved))
                ck = {"epoch": unit + 1, "bytes": info["bytes"], "path": info["path"], "best": bool(improved)}
                self._emit("on_checkpoint", unit, ck)
            except OSError as exc:                                                                # a full disk must not end the training
                ck = {"epoch": unit + 1, "error": str(exc)}
                events.append({"type": "checkpoint_failed", "epoch": unit + 1, "at": time.time(), "error": str(exc)})
        row["checkpoint"] = ck
        self.log(row)
        for ev in events:
            self.log(ev)
            self._emit("on_event", ev)
        self._emit("on_unit_end", unit, row)
        self.history_rows += 1
        self._clear_inflight()
        if self.halted:
            self._emit("on_stop", self.halted)
            return "stop"
        return "continue"

    def _state(self, unit: int, metrics: dict, last: dict | None = None) -> dict:
        gens = {n: _gen_state(g) for n, g in self.rngs.items()}
        rng = rng_states(None)
        rng["gens"] = gens
        return {"model": {n: m.state_dict() for n, m in self.modules.items()}, "optimizer": {n: o.state_dict() for n, o in self.optimizers.items()}, "scheduler": {n: s.state_dict() for n, s in self.schedulers.items()},
                "scaler": self.scaler.state_dict() if self.scaler is not None and self.scaler.is_enabled() else None, "epoch": int(unit), "rng": rng,
                "history": {"extra": self.capture() if self.capture is not None else {}, "seconds": self._seconds_before + (time.perf_counter() - self._t_start), "best_val": self.best_val, "best_unit": self.best_unit},
                "val_loss": metrics.get("val_loss"), "config_fp": self.config_fp, "data_fp": self.data_fp, "device": self.device, "grad": {n: m.state_dict() for n, m in self.monitors.items()}, "kind": self.kind, "last": last}

    # ------------------------------------------------------------------------------------------------------ end
    def finish(self, **summary) -> None:
        """Write the closing row. Raises ``TrainingHalted`` when the user stopped or paused the run from the monitor (inside a job), so that whatever surrounds the loop ends too."""
        self._ensure_started()
        stopped = self.halted or summary.pop("stopped_by", None) or "completed"
        self.log({"type": "finish", "at": time.time(), "epochs_run": self.history_rows, "best_epoch": self.best_unit + 1, "best_val": None if not np.isfinite(self.best_val) else self.best_val, "stopped_by": stopped,
                  "seconds": self._seconds_before + (time.perf_counter() - self._t_start), **summary})
        self._emit("on_finish", {"stopped_by": stopped})
        if self.halt_raises and self.halted:
            raise TrainingHalted("stop" if self.halted == "user_stop" else "pause", self.history_rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and not isinstance(exc, TrainingHalted) and self.dir is not None:
            self.log({"type": "interrupted", "epoch": self.history_rows, "at": time.time(), "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
            self._emit("on_failure", exc)
        return False
