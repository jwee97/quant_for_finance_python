"""One training loop for every network: device choice, chronological early stopping, checkpoints, gradient clipping, and a record of what the run cost.

The caller supplies the data as two things, ``batch_fn(split, rows)`` (tensors for those rows of the ``"train"`` or ``"val"`` split, already on the device) and ``loss_fn(net, batch)`` (a scalar). The
loop shuffles the training rows each epoch, clips the gradient norm, steps AdamW, halves the learning rate when the validation loss stalls, keeps the best weights and stops after ``patience`` epochs
without an improvement. The validation rows are the caller's choice and must come AFTER the training rows in time (``DeepModel`` cuts the tail of the training window, with a purge gap).
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


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


def fit_network(net: torch.nn.Module, batch_fn, loss_fn, n_train: int, n_val: int, cfg: TrainConfig) -> History:
    """Train ``net`` in place and return its history; the net ends up with its best validation weights. ``batch_fn`` must move its tensors to ``net``'s device (``net`` is already there)."""
    t0 = time.perf_counter()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    hist = History(device=next(net.parameters()).device.type, n_params=int(sum(p.numel() for p in net.parameters())))
    if n_train < 1:
        raise ValueError("no training rows")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, cfg.patience // 2))
    folder = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    start, best, best_state, bad = 0, float("inf"), deepcopy(net.state_dict()), 0
    if folder is not None and cfg.resume and (folder / "last.pt").exists():
        blob = load_checkpoint(folder / "last.pt", net, opt, next(net.parameters()).device)
        start, best, best_state = blob["epoch"] + 1, blob["best_val"], deepcopy(net.state_dict())
    previous_threads = torch.get_num_threads()
    if cfg.num_threads > 0:
        torch.set_num_threads(int(cfg.num_threads))
    try:
        for epoch in range(start, cfg.epochs):
            e0 = time.perf_counter()
            net.train()
            order = rng.permutation(n_train)
            total, seen, norms = 0.0, 0, []
            for i in range(0, n_train, cfg.batch_size):
                rows = order[i:i + cfg.batch_size]
                loss = loss_fn(net, batch_fn("train", rows))
                if not torch.isfinite(loss):
                    raise ValueError("training diverged (the loss is not finite); lower the learning rate")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))))
                opt.step()
                total += float(loss.detach()) * len(rows)
                seen += len(rows)
            train_loss = total / max(seen, 1)
            val_loss = _evaluate(net, batch_fn, loss_fn, n_val, cfg.batch_size * 2) if n_val else train_loss
            hist.train_loss.append(train_loss)
            hist.val_loss.append(val_loss)
            hist.lr.append(float(opt.param_groups[0]["lr"]))
            hist.grad_norm.append(float(np.mean(norms)) if norms else 0.0)
            hist.epoch_seconds.append(time.perf_counter() - e0)
            sched.step(val_loss)
            if val_loss < best - cfg.min_delta:
                best, best_state, bad = val_loss, deepcopy(net.state_dict()), 0
                hist.best_epoch = epoch
                if folder is not None:
                    save_checkpoint(folder / "best.pt", net, opt, epoch, best, val_loss, cfg)
            else:
                bad += 1
            if folder is not None:
                save_checkpoint(folder / "last.pt", net, opt, epoch, best, val_loss, cfg)
            if bad >= cfg.patience:
                hist.stopped_early = True
                break
    finally:
        if cfg.num_threads > 0:
            torch.set_num_threads(previous_threads)
    net.load_state_dict(best_state)
    net.eval()
    hist.seconds = time.perf_counter() - t0
    hist.peak_memory_mb = peak_memory_mb(next(net.parameters()).device)
    return hist


@torch.no_grad()
def _evaluate(net, batch_fn, loss_fn, n_val: int, chunk: int) -> float:
    net.eval()
    total, seen = 0.0, 0
    for i in range(0, n_val, chunk):
        rows = np.arange(i, min(i + chunk, n_val))
        total += float(loss_fn(net, batch_fn("val", rows))) * len(rows)
        seen += len(rows)
    return total / max(seen, 1)
