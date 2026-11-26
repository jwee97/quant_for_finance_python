"""Gradient diagnostics for any PyTorch training loop: where the gradient is, how large it is, whether it is finite, how far each layer moves, and warnings when the optimisation looks unhealthy.

A :class:`GradientMonitor` watches one optimizer's parameters. The loop calls it at four points of an optimizer step and nothing else changes in the loop:

    loss.backward()                     # (with AMP: scaler.unscale_(opt) first, so that the measured gradient is the true one)
    mon.observe(step)                   # the COMPLETED gradient (after accumulation), BEFORE clipping
    clip_grad_norm_(...)
    mon.after_clip(step)                # the gradient the optimizer will use, after clipping
    mon.before_step(step); opt.step(); mon.after_step(step)     # the update actually applied (only on sampled steps)

``mon.end_epoch(...)`` closes a reporting unit: it returns one compact summary (written to ``gradients.jsonl``) and the events the unit raised.

What is measured (``level``):

``off``          nothing.
``lightweight``  every ``interval`` steps (default 10): the global L2 norm of the unclipped gradient (every step if clipping is on: it is the number the clipper computes anyway), non-finite detection, clip
                 frequency, the count of trainable parameters with no gradient.
``standard``     lightweight, plus at about four sampled steps per epoch: global mean/median/RMS/standard deviation, signed minimum and maximum, L-infinity norm, P50, P90, P95, P99 of |g|, zero and near-zero
                 fractions; per layer: L2, RMS, mean |g|, max |g|, near-zero fraction, parameter norm, gradient-to-weight ratio and the measured update-to-weight ratio.
``detailed``     standard at about eight samples per epoch, plus per-layer medians and a histogram of log10 |g| (zeros and non-finite values counted apart).

Design rules that make the numbers mean something:

* The gradient is measured at the optimizer step, after all accumulation steps and (with mixed precision) after unscaling; microbatch gradients are not mixed in.
* ``None`` (no gradient reached the parameter) is not the same as exactly zero: they are counted apart. Parameters with ``requires_grad=False`` are frozen on purpose and are excluded from every denominator.
* Sparse gradients are measured over their stored values; the implicit zeros are counted in the zero fraction.
* Non-finite entries are counted and excluded from the other statistics (never replaced by a small number).
* Nothing keeps a tensor: each statistic is reduced to a Python number on the sampled step, one device-to-host transfer per sample.
* The monitor only reads gradients and parameters. It never changes them, never sets a hyperparameter, and a warning never stops or alters the training.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

LEVELS = ("off", "lightweight", "standard", "detailed")
PERCENTILES = (50, 90, 95, 99)
HIST_LO, HIST_HI, HIST_BINS = -12.0, 4.0, 40                       # log10 |g| from 1e-12 to 1e4
SAMPLES_PER_EPOCH = {"standard": 4, "detailed": 8}
PERCENTILE_CAP = 200_000                                           # elements sorted for the global percentiles; more are subsampled with a fixed stride and the result says so
MAX_LAYERS = 64
SEVERITY = ("info", "warning", "critical")

THRESHOLDS = {
    "persist": 3,               # consecutive reporting units a condition must hold before it is a warning (one bad epoch is not a diagnosis)
    "explode_clip_freq": 0.5,   # share of optimizer steps whose gradient was clipped
    "explode_growth": 100.0,    # the unit's median global norm against the median of the first units
    "explode_abs": 1e4,         # an absolute global norm no healthy small network reaches
    "spike_factor": 10.0,       # one step's global norm against the median of the unit (a spike is recorded, never called explosion)
    "vanish_abs": 1e-7,         # a layer's RMS gradient below this ...
    "vanish_rel": 1e-3,         # ... AND below this share of the largest layer's RMS
    "depth_decay": 1e-3,        # first layer's RMS over the last layer's RMS, across the depth of the model
    "dead_zero_frac": 0.999,    # share of zero entries ...
    "update_min": 1e-9,         # update-to-weight ratio below this means the layer is not moving
    "update_max": 0.1,          # above this a step moves the layer by a tenth of its norm: usually a learning-rate problem
    "cv": 2.0,                  # coefficient of variation of the step norms within a unit
    "loss_spike": 0.5,          # a unit's training loss more than this share above the best so far
    "overfit_units": 3,         # units in a row of falling training loss and rising validation loss
    "shift_decades": 2.0,       # change of the median log10 |g| between consecutive units
    "cooldown": 10,             # a warning is not repeated for this many units
}
FAMILY_OVERRIDES = {            # model-aware: what is normal differs
    "transformer": {"explode_clip_freq": 0.8, "update_max": 0.2},      # warm-up and clipping at 1.0 are standard practice for attention models
    "recurrent": {"explode_clip_freq": 0.7},                           # recurrent nets are clipped routinely
    "gan": {"vanish_rel": 1e-4, "cv": 4.0},                            # adversarial gradients oscillate by design
    "rl": {"cv": 4.0, "loss_spike": 2.0},                              # policy-gradient losses are noisy
}


def thresholds_for(family: str = "default", override: dict | None = None) -> dict:
    return {**THRESHOLDS, **FAMILY_OVERRIDES.get(family, {}), **(override or {})}


def resolve_level(level: str | None, on_replit: bool, has_folder: bool) -> str:
    """``auto`` is lightweight on Replit, standard elsewhere, and off when there is nowhere to persist the result."""
    level = (level or "auto").lower()
    if level == "auto":
        if not has_folder:
            return "off"
        return "lightweight" if on_replit else "standard"
    if level not in LEVELS:
        raise ValueError(f"gradient monitoring level must be auto or one of {LEVELS}")
    return level


def layer_key(param_name: str, depth: int = 0) -> str:
    """The module a parameter belongs to (``blocks.0.fc.weight`` -> ``blocks.0.fc``); ``depth`` keeps only that many leading parts (used when a model has too many layers to list)."""
    parts = param_name.split(".")
    parts = parts[:-1] if len(parts) > 1 else parts
    return ".".join(parts[:depth]) if depth else ".".join(parts)


def component_of(layer: str) -> str:
    return layer.split(".")[0]


def _f(x):
    """A finite float for JSON, or None."""
    x = float(x)
    return x if math.isfinite(x) else None


class GradientMonitor:
    """See the module docstring. ``params`` is ``{name: parameter}`` (``module.named_parameters()``, optionally prefixed per module); ``name`` labels this optimizer in the output."""

    def __init__(self, name: str, params: dict[str, torch.nn.Parameter], level: str = "standard", *, interval: int = 0, near_zero: float = 1e-8, family: str = "default", thresholds: dict | None = None,
                 clip: float | None = None, accumulation: int = 1, max_layers: int = MAX_LAYERS):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        self.name, self.level, self.near_zero, self.family, self.clip, self.accumulation = name, level, float(near_zero), family, clip, max(int(accumulation), 1)
        self.T = thresholds_for(family, thresholds)
        self.params = dict(params)
        self.frozen = sorted(n for n, p in self.params.items() if not p.requires_grad)
        self.active = {n: p for n, p in self.params.items() if p.requires_grad}
        keys = {n: layer_key(n) for n in self.active}
        depth = 0
        while len(set(keys.values())) > max_layers and depth < 6:                         # too many layers to list: merge into their parents
            depth = max((len(k.split(".")) for k in keys.values())) - 1 if depth == 0 else depth - 1
            keys = {n: layer_key(n, depth) for n in self.active}
            if depth <= 1:
                break
        self.layer_of = keys
        self.layers: list[str] = list(dict.fromkeys(keys.values()))                         # order of first appearance = depth order
        self.interval = int(interval)
        self._epoch, self._steps_per_epoch, self._lr = 0, None, None
        self._reset_unit()
        self.history: deque = deque(maxlen=24)                                            # compact summaries of recent units, for the persistence rules
        self.events_total = 0
        self._last_event: dict[str, int] = {}
        self._first_norms: list[float] = []
        self._best_train = float("inf")

    # ------------------------------------------------------------------------------------------------------ one reporting unit
    def _reset_unit(self) -> None:
        self._norms: list[torch.Tensor] = []            # 0-d device tensors, one per observed step; reduced once at the end of the unit
        self._norm_steps: list[int] = []
        self._post: list[float] = []
        self._samples: list[dict] = []
        self._layer_acc: dict[str, dict] = {}
        self._hist = np.zeros(HIST_BINS, dtype=np.int64)
        self._zeros_hist, self._nonfinite_hist = 0, 0
        self._opt_steps = 0
        self._skipped = 0
        self._pre_snapshot: dict[str, torch.Tensor] | None = None
        self._update: dict[str, list] = {}
        self._missing_names: set[str] = set()
        self._percentile_sampled = False

    def begin_epoch(self, epoch: int, steps_per_epoch: int | None = None, lr: float | None = None) -> None:
        self._epoch, self._steps_per_epoch, self._lr = int(epoch), steps_per_epoch, lr
        self._reset_unit()

    def _every(self) -> int:
        if self.interval > 0:
            return self.interval
        if self.level == "lightweight":
            return 10
        n = self._steps_per_epoch or 100
        return max(1, n // SAMPLES_PER_EPOCH.get(self.level, 8))

    def sample_due(self, step: int) -> bool:
        return self.level in ("standard", "detailed") and step % self._every() == 0

    def norm_due(self, step: int) -> bool:
        return self.level != "off" and (self.clip is not None or step % (self.interval if self.interval > 0 else (10 if self.level == "lightweight" else 1)) == 0 or self.sample_due(step))

    # ------------------------------------------------------------------------------------------------------ the four calls
    def _grads(self) -> list[tuple[str, torch.Tensor]]:
        return [(n, p.grad) for n, p in self.active.items() if p.grad is not None]

    @torch.no_grad()
    def observe(self, step: int) -> None:
        """The completed, unscaled, unclipped gradient of this optimizer step."""
        if self.level == "off":
            return
        self._opt_steps += 1
        grads = self._grads()
        if len(grads) < len(self.active):
            self._missing_names.update(n for n, p in self.active.items() if p.grad is None)
        if not self.norm_due(step):
            return
        dense = [g if not g.is_sparse else g.coalesce().values() for _, g in grads]
        if dense:
            norms = torch.stack([torch.linalg.vector_norm(g.float()) for g in dense])
            self._norms.append(torch.linalg.vector_norm(norms.double()))                   # nothing leaves the device here
        else:
            self._norms.append(torch.zeros((), dtype=torch.float64))
        self._norm_steps.append(step)
        if self.sample_due(step):
            self._sample(step, grads)
            self._take_snapshot()

    @torch.no_grad()
    def after_clip(self, step: int) -> None:
        if self.level in ("standard", "detailed") and self.sample_due(step):
            grads = [g if not g.is_sparse else g.coalesce().values() for _, g in self._grads()]
            if grads:
                total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]).double())
                self._post.append(float(total))

    @torch.no_grad()
    def _take_snapshot(self) -> None:
        self._pre_snapshot = {n: p.detach().clone() for n, p in self.active.items()}

    @torch.no_grad()
    def before_step(self, step: int) -> None:
        pass                                                                              # the snapshot was taken in ``observe``: the parameters do not change between them

    @torch.no_grad()
    def after_step(self, step: int) -> None:
        """The parameter update the optimizer really applied (a step the scaler skipped moves nothing and is measured as such)."""
        snap, self._pre_snapshot = self._pre_snapshot, None
        if snap is None:
            return
        per: dict[str, list] = {}
        for n, p in self.active.items():
            if n not in snap:
                continue
            d = torch.linalg.vector_norm((p.detach() - snap[n]).float())
            w = torch.linalg.vector_norm(snap[n].float())
            per.setdefault(self.layer_of[n], []).append(torch.stack([d.double() ** 2, w.double() ** 2]))
        keys = list(per)
        if not keys:
            return
        vals = torch.stack([torch.stack(per[k]).sum(0) for k in keys]).cpu().numpy()      # one transfer
        for k, (d2, w2) in zip(keys, vals):
            self._update.setdefault(k, []).append(math.sqrt(d2) / math.sqrt(w2) if w2 > 0 else float("nan"))

    def note_skipped(self, step: int) -> None:
        """A mixed-precision scaler skipped the step because of a non-finite gradient: the weights did not move."""
        self._skipped += 1

    # ------------------------------------------------------------------------------------------------------ one sample
    @torch.no_grad()
    def _sample(self, step: int, grads: list[tuple[str, torch.Tensor]]) -> None:
        thr = self.near_zero
        gmap = dict(grads)
        rows, names, numel_dense = [], [], []
        flat_for_pct = []
        total_n = sum(int(g.numel()) for _, g in grads)
        stride = max(1, math.ceil(total_n / PERCENTILE_CAP))
        for n, g in grads:
            dense_n = int(g.numel())
            gv = g.coalesce().values() if g.is_sparse else g
            gv = gv.detach().float()
            fin = torch.isfinite(gv)
            bad = (~fin).sum()
            gz = torch.where(fin, gv, torch.zeros_like(gv))
            a = gz.abs()
            nz = int(gv.numel())
            vec = torch.stack([(gz.double() ** 2).sum(), a.double().sum(), gz.double().sum(), a.max() if nz else torch.zeros((), device=gv.device), gz.min() if nz else torch.zeros((), device=gv.device),
                               gz.max() if nz else torch.zeros((), device=gv.device), ((gv == 0) & fin).sum().double(), ((a < thr) & fin).sum().double(), bad.double(),
                               torch.linalg.vector_norm(self.active[n].detach().float()).double(),
                               torch.as_tensor(float(dense_n - nz), dtype=torch.float64, device=gv.device)])                  # implicit zeros of a sparse gradient
            rows.append(vec.double())
            names.append(n)
            numel_dense.append(dense_n)
            if self.level in ("standard", "detailed"):
                flat_for_pct.append(a.flatten()[fin.flatten()][::stride] if bool(bad) else a.flatten()[::stride])
        table = torch.stack(rows).cpu().numpy()                                           # THE device-to-host transfer of this sample
        n_all = np.array(numel_dense, dtype=np.float64)
        sumsq, sumabs, sumg = table[:, 0], table[:, 1], table[:, 2]
        n_fin = np.maximum(n_all - table[:, 8], 1.0)
        g_n = float(n_fin.sum())
        glob = {"step": int(step), "l2": math.sqrt(float(sumsq.sum())), "linf": float(table[:, 3].max()), "mean_abs": float(sumabs.sum() / g_n), "rms": math.sqrt(float(sumsq.sum() / g_n)),
                "std": math.sqrt(max(float(sumsq.sum() / g_n) - float(sumg.sum() / g_n) ** 2, 0.0)), "min": float(table[:, 4].min()), "max": float(table[:, 5].max()),
                "zero_frac": float((table[:, 6].sum() + table[:, 10].sum()) / n_all.sum()), "near_zero_frac": float((table[:, 7].sum() + table[:, 10].sum()) / n_all.sum()),
                "nonfinite": int(table[:, 8].sum()), "missing": len(self.active) - len(grads), "n": int(n_all.sum())}
        if flat_for_pct:
            allabs = torch.cat(flat_for_pct)
            self._percentile_sampled = self._percentile_sampled or stride > 1
            if allabs.numel():
                srt = torch.sort(allabs).values
                for q in PERCENTILES:
                    glob[f"p{q}"] = float(srt[min(int(round(q / 100.0 * (srt.numel() - 1))), srt.numel() - 1)])
                glob["median"] = glob["p50"]
                if self.level == "detailed":
                    nzv = srt[srt > 0]
                    self._zeros_hist += int((srt == 0).sum())
                    if nzv.numel():
                        lg = torch.log10(nzv).clamp(HIST_LO, HIST_HI - 1e-9)
                        self._hist += torch.histc(lg, bins=HIST_BINS, min=HIST_LO, max=HIST_HI).cpu().numpy().astype(np.int64)
            self._nonfinite_hist += glob["nonfinite"]
        self._samples.append(glob)
        # per layer
        by_layer: dict[str, list[int]] = {}
        for i, n in enumerate(names):
            by_layer.setdefault(self.layer_of[n], []).append(i)
        for layer, idx in by_layer.items():
            sq, ab, nn_ = float(sumsq[idx].sum()), float(sumabs[idx].sum()), float(n_fin[idx].sum())
            wnorm = math.sqrt(float((table[idx, 9] ** 2).sum()))
            l2 = math.sqrt(sq)
            row = {"l2": l2, "rms": math.sqrt(sq / nn_) if nn_ else 0.0, "mean_abs": ab / nn_ if nn_ else 0.0, "max_abs": float(table[idx, 3].max()), "zero_frac": float((table[idx, 6].sum() + table[idx, 10].sum()) / n_all[idx].sum()),
                   "near_zero_frac": float((table[idx, 7].sum() + table[idx, 10].sum()) / n_all[idx].sum()), "nonfinite": int(table[idx, 8].sum()), "param_norm": wnorm, "g_w": l2 / wnorm if wnorm > 0 else float("nan"),
                   "n_params": int(n_all[idx].sum())}
            if self.level == "detailed":
                vals = torch.cat([(gmap[names[i]].coalesce().values() if gmap[names[i]].is_sparse else gmap[names[i]]).detach().float().abs().flatten() for i in idx])
                row["median_abs"] = float(vals.median()) if vals.numel() else 0.0
            acc = self._layer_acc.setdefault(layer, {"n": 0, "sums": {}, "max": {}})
            acc["n"] += 1
            for k, v in row.items():
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    continue
                acc["sums"][k] = acc["sums"].get(k, 0.0) + v
                acc["max"][k] = max(acc["max"].get(k, -math.inf), v)

    # ------------------------------------------------------------------------------------------------------ closing the unit
    def end_epoch(self, epoch: int, *, lr: float | None = None, train_loss: float | None = None, val_loss: float | None = None, action: str = "continued") -> tuple[dict | None, list[dict]]:
        """``(summary, events)`` for the unit just finished; ``summary`` is compact JSON-able data, ``events`` the warnings it raised."""
        if self.level == "off":
            return None, []
        norms = np.array(torch.stack(self._norms).cpu().numpy() if self._norms else [], dtype=np.float64)
        steps = list(self._norm_steps)
        finite = np.isfinite(norms)
        fn = norms[finite]
        glob: dict = {"observed_steps": int(len(norms)), "optimizer_steps": int(self._opt_steps), "skipped_steps": int(self._skipped), "nonfinite_steps": int((~finite).sum()),
                      "l2_median": _f(np.median(fn)) if fn.size else None, "l2_mean": _f(fn.mean()) if fn.size else None, "l2_max": _f(fn.max()) if fn.size else None, "l2_min": _f(fn.min()) if fn.size else None,
                      "l2_std": _f(fn.std()) if fn.size else None, "l2_cv": _f(fn.std() / fn.mean()) if fn.size > 1 and fn.mean() > 0 else None,
                      "missing_gradients": int(len(self._missing_names)), "frozen_params": len(self.frozen), "trainable_params": len(self.active)}
        pcts = {f"l2_p{q}": _f(np.percentile(fn, q)) for q in PERCENTILES} if fn.size else {}
        glob.update(pcts)
        if self.clip:
            glob["clip_value"] = float(self.clip)
            glob["clip_frequency"] = float((norms[finite] > self.clip).mean()) if fn.size else None
            glob["clipped_steps"] = int((fn > self.clip).sum()) + int((~finite).sum())
            if self._post:
                glob["post_clip_l2_mean"] = _f(np.mean(self._post))
        if self._samples:
            keys = ("linf", "mean_abs", "rms", "std", "min", "max", "zero_frac", "near_zero_frac", "p50", "p90", "p95", "p99", "median")
            glob["sampled"] = {k: _f(np.mean([s[k] for s in self._samples if k in s])) for k in keys if any(k in s for s in self._samples)}
            glob["sampled"]["min"] = _f(min(s["min"] for s in self._samples))
            glob["sampled"]["max"] = _f(max(s["max"] for s in self._samples))
            glob["sampled"]["linf"] = _f(max(s["linf"] for s in self._samples))
            glob["sampled"]["nonfinite"] = int(sum(s["nonfinite"] for s in self._samples))
            glob["sampled"]["n_samples"] = len(self._samples)
            glob["sampled"]["percentiles_subsampled"] = bool(self._percentile_sampled)
        layers = []
        for layer in self.layers:
            acc = self._layer_acc.get(layer)
            if not acc:
                continue
            n = acc["n"]
            row = {"name": layer, "component": component_of(layer)}
            for k, s in acc["sums"].items():
                row[k] = _f(s / n) if k not in ("nonfinite", "n_params") else int(acc["max"][k])
            row["max_abs"] = _f(acc["max"].get("max_abs"))
            ups = [u for u in self._update.get(layer, []) if math.isfinite(u)]
            if ups:
                row["update_ratio"] = _f(np.mean(ups))
            layers.append(row)
        summary = {"type": "gradient", "optimizer": self.name, "epoch": int(epoch), "level": self.level, "lr": lr, "train_loss": train_loss, "val_loss": val_loss, "global": glob, "layers": layers}
        if self.level == "detailed" and self._hist.sum() + self._zeros_hist + self._nonfinite_hist > 0:
            summary["histogram"] = {"lo": HIST_LO, "hi": HIST_HI, "bins": self._hist.tolist(), "zeros": int(self._zeros_hist), "nonfinite": int(self._nonfinite_hist), "note": "log10 |g|; values outside the range are counted in the end bins"}
        events = self._detect(epoch, norms, steps, glob, layers, lr, train_loss, val_loss, action)
        entry = ({"epoch": int(epoch), "l2_median": glob["l2_median"], "clip_frequency": glob.get("clip_frequency"), "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
                             "layers": {r["name"]: {"rms": r.get("rms"), "zero_frac": r.get("zero_frac"), "l2": r.get("l2"), "update_ratio": r.get("update_ratio")} for r in layers},
                  "median_log": _f(math.log10(glob["sampled"]["median"])) if glob.get("sampled", {}).get("median") else None, "cv": glob.get("l2_cv")})
        entry.update(getattr(self, "_flags_now", {}))                                      # the conditions this unit met: the persistence rules count them across units
        self.history.append(entry)
        if glob["l2_median"] is not None and len(self._first_norms) < self.T["persist"]:
            self._first_norms.append(glob["l2_median"])
        if train_loss is not None and math.isfinite(train_loss):
            self._best_train = min(self._best_train, train_loss)
        summary["events"] = len(events)
        return summary, events

    # ------------------------------------------------------------------------------------------------------ pathologies
    def _run(self, key: str, k: int = 0) -> int:
        """How many consecutive recent units (ending now) satisfied the condition recorded under ``key``."""
        n = 0
        for h in reversed(self.history):
            if h.get(key):
                n += 1
            else:
                break
        return n

    def _emit(self, out: list, kind: str, severity: str, epoch: int, step: int | None, layers: list[str], metrics: dict, thresholds: dict, lr, action: str, message: str) -> None:
        last = self._last_event.get(kind + "|" + ",".join(layers[:3]))
        if last is not None and epoch - last < self.T["cooldown"] and severity != "critical":
            return
        self._last_event[kind + "|" + ",".join(layers[:3])] = epoch
        self.events_total += 1
        out.append({"type": "gradient_event", "kind": kind, "severity": severity, "optimizer": self.name, "epoch": int(epoch), "step": step, "layers": layers[:12], "metrics": metrics, "thresholds": thresholds, "lr": lr,
                    "action": action, "message": message, "at": __import__("time").time()})

    def _detect(self, epoch: int, norms: np.ndarray, steps: list[int], glob: dict, layers: list[dict], lr, train_loss, val_loss, action: str) -> list[dict]:
        T, out = self.T, []
        flags: dict[str, bool] = {}
        fin = np.isfinite(norms)
        # non-finite: immediate, and critical: the step's update is garbage unless a scaler skipped it
        if (~fin).any() or (glob.get("sampled", {}).get("nonfinite", 0) or 0) > 0:
            first = steps[int(np.flatnonzero(~fin)[0])] if (~fin).any() else None
            skipped = self._skipped >= int((~fin).sum()) > 0
            self._emit(out, "nonfinite_gradient", "critical", epoch, first, [], {"nonfinite_steps": int((~fin).sum()), "skipped_by_scaler": int(self._skipped)}, {}, lr,
                       "skipped_step" if skipped else action, "non-finite values in the gradient" + (" (the mixed-precision scaler skipped those steps)" if skipped else ": the optimizer step used them unless the loop guards it"))
        fn = norms[fin]
        if fn.size >= 5:                                                                  # a spike is recorded, never named an explosion on its own
            med = float(np.median(fn))
            hot = np.flatnonzero(fn > T["spike_factor"] * max(med, 1e-30))
            if hot.size:
                self._emit(out, "gradient_spike", "info", epoch, steps[int(np.flatnonzero(fin)[hot[0]])], [], {"norm": float(fn[hot[0]]), "unit_median": med, "spikes": int(hot.size)}, {"spike_factor": T["spike_factor"]}, lr, action,
                           f"{hot.size} step(s) with a gradient norm over {T['spike_factor']:g} times the unit's median")
        med_norm = glob.get("l2_median")
        # exploding
        clipf = glob.get("clip_frequency")
        flags["clip_hot"] = clipf is not None and clipf >= T["explode_clip_freq"]
        base = float(np.median(self._first_norms)) if self._first_norms else None
        flags["grew"] = bool(med_norm is not None and base and len(self._first_norms) >= T["persist"] and base > 0 and med_norm > T["explode_growth"] * base) or bool(med_norm is not None and med_norm > T["explode_abs"])
        # persistent conditions are counted over the recorded history plus this unit
        def streak(key: str, now: bool) -> int:
            return (self._run(key, 0) + 1) if now else 0

        cur = {"clip_hot": flags["clip_hot"], "grew": flags["grew"]}
        rms = {r["name"]: r.get("rms") for r in layers if r.get("rms") is not None}
        top = max(rms.values()) if rms else None
        weak = [n for n, v in rms.items() if top and v < T["vanish_abs"] and v < T["vanish_rel"] * top] if len(rms) >= 2 else ([n for n, v in rms.items() if v < T["vanish_abs"]] if rms else [])
        dead = [r["name"] for r in layers if r.get("l2") == 0.0 and (r.get("zero_frac") or 0) >= T["dead_zero_frac"]]
        stuck = [r["name"] for r in layers if r.get("update_ratio") is not None and r["update_ratio"] < T["update_min"] and (r.get("l2") or 0) > 0 and (train_loss or 0) > 0]
        big = [r["name"] for r in layers if r.get("update_ratio") is not None and r["update_ratio"] > T["update_max"]]
        ordered = [rms[r["name"]] for r in layers if r["name"] in rms and rms[r["name"]] > 0]
        decay = bool(len(ordered) >= 3 and ordered[-1] > 0 and ordered[0] / ordered[-1] < T["depth_decay"])
        cur.update({"weak": bool(weak), "dead": bool(dead), "stuck": bool(stuck), "big": bool(big), "decay": decay, "cv": bool(glob.get("l2_cv") is not None and glob["l2_cv"] > T["cv"])})
        cur["loss_spike"] = bool(train_loss is not None and math.isfinite(self._best_train) and train_loss > self._best_train * (1 + T["loss_spike"]) and self._best_train > 0)
        tl = [h["train_loss"] for h in list(self.history)[-(T["overfit_units"]):]] + [train_loss]
        vl = [h["val_loss"] for h in list(self.history)[-(T["overfit_units"]):]] + [val_loss]
        cur["overfit"] = bool(all(x is not None for x in tl + vl) and len(tl) > T["overfit_units"] and all(tl[i + 1] < tl[i] for i in range(len(tl) - 1)) and all(vl[i + 1] > vl[i] for i in range(len(vl) - 1)))
        persist = T["persist"]

        def s(key: str) -> int:
            return streak(key, cur[key])
        lrchange = bool(self.history and self.history[-1].get("lr") is not None and lr is not None and abs(self.history[-1]["lr"] - lr) > 1e-12)
        if s("clip_hot") >= persist:
            self._emit(out, "repeated_clipping", "warning", epoch, None, [], {"clip_frequency": clipf, "clip_value": glob.get("clip_value"), "units": s("clip_hot")}, {"explode_clip_freq": T["explode_clip_freq"], "persist": persist}, lr, action,
                       f"{clipf:.0%} of the optimizer steps were clipped for {s('clip_hot')} units in a row: the clip value is setting the step, not the optimizer")
        if s("grew") >= persist or (cur["grew"] and (glob.get("nonfinite_steps") or 0) > 0):
            self._emit(out, "exploding_gradient", "critical" if (glob.get("nonfinite_steps") or 0) else "warning", epoch, None, [], {"median_norm": med_norm, "reference": base, "units": s("grew")},
                       {"explode_growth": T["explode_growth"], "explode_abs": T["explode_abs"], "persist": persist}, lr, action, "the median global gradient norm stays far above where training began")
        if s("weak") >= persist:
            self._emit(out, "vanishing_gradient", "warning", epoch, None, weak, {n: rms[n] for n in weak[:8]}, {"vanish_abs": T["vanish_abs"], "vanish_rel": T["vanish_rel"], "persist": persist}, lr, action,
                       "layers whose RMS gradient is negligible, absolutely and next to the strongest layer, for several units in a row")
        if s("decay") >= persist:
            self._emit(out, "gradient_decay_with_depth", "warning", epoch, None, [layers[0]["name"], layers[-1]["name"]], {"first_over_last": ordered[0] / ordered[-1]}, {"depth_decay": T["depth_decay"], "persist": persist}, lr, action,
                       "the first layer's gradient is a thousandth of the last layer's or less")
        if s("dead") >= persist:
            self._emit(out, "dead_layer", "warning", epoch, None, dead, {"zero_frac": {r["name"]: r["zero_frac"] for r in layers if r["name"] in dead}}, {"dead_zero_frac": T["dead_zero_frac"], "persist": persist}, lr, action,
                       "trainable layers that received exactly zero gradient at every sampled step")
        if s("stuck") >= persist:
            self._emit(out, "no_parameter_update", "warning", epoch, None, stuck, {n: next(r["update_ratio"] for r in layers if r["name"] == n) for n in stuck[:8]}, {"update_min": T["update_min"], "persist": persist}, lr, action,
                       "layers with a gradient whose weights barely move (update-to-weight ratio near zero)")
        if cur["big"]:
            self._emit(out, "large_update_ratio", "warning", epoch, None, big, {n: next(r["update_ratio"] for r in layers if r["name"] == n) for n in big[:8]}, {"update_max": T["update_max"]}, lr, action,
                       "a step moves these layers by more than a tenth of their norm" + (" (the learning rate changed in this unit)" if lrchange else ""))
        if s("cv") >= persist:
            self._emit(out, "unstable_optimization", "warning", epoch, None, [], {"cv": glob.get("l2_cv"), "lr_changed": lrchange}, {"cv": T["cv"], "persist": persist}, lr, action,
                       "the gradient norm swings wildly from step to step" + (" and the learning rate just changed" if lrchange else ""))
        if cur["loss_spike"]:
            self._emit(out, "loss_spike", "warning", epoch, None, [], {"train_loss": train_loss, "best": self._best_train}, {"loss_spike": T["loss_spike"]}, lr, action, "the training loss jumped well above its best")
        if cur["overfit"]:
            self._emit(out, "potential_overfitting", "info", epoch, None, [], {"train_loss": tl[-1], "val_loss": vl[-1]}, {"overfit_units": T["overfit_units"]}, lr, action, "training loss keeps falling while validation loss keeps rising")
        mlog, prev = (glob.get("sampled") or {}).get("median"), self.history[-1].get("median_log") if self.history else None
        if mlog and prev is not None and math.log10(mlog) is not None and abs(math.log10(mlog) - prev) > T["shift_decades"]:
            self._emit(out, "gradient_distribution_shift", "info", epoch, None, [], {"median_log10": math.log10(mlog), "previous": prev}, {"shift_decades": T["shift_decades"]}, lr, action,
                       "the typical gradient magnitude moved by orders of magnitude between units")
        self._flags_now = cur
        return out

    # ------------------------------------------------------------------------------------------------------ persistence of the detector
    def state_dict(self) -> dict:
        return {"history": list(self.history), "first_norms": list(self._first_norms), "best_train": None if not math.isfinite(self._best_train) else self._best_train, "events_total": self.events_total, "last_event": dict(self._last_event),
                "level": self.level}

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.history = deque(state.get("history", []), maxlen=24)
        self._first_norms = list(state.get("first_norms", []))
        self._best_train = float("inf") if state.get("best_train") is None else float(state["best_train"])
        self.events_total = int(state.get("events_total", 0))
        self._last_event = dict(state.get("last_event", {}))


GRADIENT_FILE_CAP = 4_000_000          # bytes kept in gradients.jsonl; beyond it older units are thinned (every other one), never the newest


def append_summary(folder, row: dict) -> None:
    """Append one unit's compact gradient summary to ``gradients.jsonl``; the file is bounded (older units are thinned, the newest 40 kept whole)."""
    if folder is None:
        return
    path = Path(folder) / "gradients.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if path.stat().st_size > GRADIENT_FILE_CAP:
            lines = path.read_text(encoding="utf-8").splitlines()
            keep = [ln for i, ln in enumerate(lines) if i >= len(lines) - 40 or i % 2 == 0]
            path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    except OSError:
        pass
