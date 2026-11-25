"""Measure what checkpointing costs: epoch time with and without it, the size of a checkpoint, the time to write and to read one, and process memory, for N-BEATS and N-HiTS at three sizes on this machine.

    python scripts/benchmark_training.py [--out docs/generated/training_benchmark.json]

Synthetic data, CPU, a few epochs: it says how the checkpointing behaves on the machine it runs on, not how fast a real experiment is."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from src.ml import resources
from src.ml.deep import checkpoint as ck, trainer as tn
from src.ml.deep.nets import build_net

SIZES = {"small": (16, 2, 2), "medium": (64, 3, 3), "large": (256, 4, 4)}                  # width, blocks, layers
LOOK, FEAT, N_TRAIN, N_VAL, EPOCHS = 20, 30, 4000, 1000, 4


def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_TRAIN + N_VAL, LOOK, FEAT)).astype(np.float32)
    y = (X[:, -1, 0] + 0.1 * rng.normal(size=len(X))).astype(np.float32)
    split = {"train": np.arange(N_TRAIN), "val": np.arange(N_TRAIN, N_TRAIN + N_VAL)}
    return (lambda name, rows: (torch.from_numpy(X[split[name][rows]]), torch.from_numpy(y[split[name][rows]]))), (lambda net, b: ((net(b[0]).reshape(len(b[1]), -1)[:, 0] - b[1]) ** 2).mean())


def run(arch: str, size: str, folder: str | None) -> dict:
    width, blocks, layers = SIZES[size]
    torch.manual_seed(0)
    net = build_net(arch, FEAT, LOOK, 1, {"width": width, "blocks": blocks, "layers": layers, "pool_sizes": "4,2,1", "dropout": 0.1})
    batch, loss = data()
    cfg = tn.TrainConfig(epochs=EPOCHS, batch_size=256, learning_rate=0.01, patience=50, checkpoint_dir=folder, config_fp="b", data_fp="b", seed=1)
    t0 = time.perf_counter()
    h = tn.fit_network(net, batch, loss, N_TRAIN, N_VAL, cfg)
    total = time.perf_counter() - t0
    return {"seconds": total, "epoch_seconds": float(np.mean(h.epoch_seconds)), "params": int(sum(p.numel() for p in net.parameters())), "rss_mb": resources.rss_mb()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/generated/training_benchmark.json")
    args = ap.parse_args()
    torch.set_num_threads(max(1, resources.default_threads()))
    rows = []
    for arch in ("nbeats", "nhits"):
        for size in SIZES:
            run(arch, size, None)                                                              # warm-up: the first run of a process pays one-off start-up costs
            plain = run(arch, size, None)
            with tempfile.TemporaryDirectory() as tmp:
                saved = run(arch, size, tmp)
                mgr = ck.CheckpointManager(tmp)
                t0 = time.perf_counter()
                blob, which, _ = mgr.latest_valid()
                read = time.perf_counter() - t0
                size_bytes = (Path(tmp) / "last.pt").stat().st_size
                with tempfile.TemporaryDirectory() as tmp2:                                         # the cost of committing one checkpoint (serialise, fsync, read back and verify, replace), measured directly
                    writer, times = ck.CheckpointManager(tmp2), []
                    for k in range(5):
                        t1 = time.perf_counter()
                        writer.save({**blob, "epoch": k}, best=True)
                        times.append(time.perf_counter() - t1)
                rows.append({"arch": arch, "size": size, "params": plain["params"], "epoch_seconds_no_checkpoint": round(plain["epoch_seconds"], 4), "epoch_seconds_with_checkpoint": round(saved["epoch_seconds"], 4),
                             "commit_seconds_mean": round(float(np.mean(times)), 4), "commit_seconds_max": round(float(np.max(times)), 4), "checkpoint_mb": round(size_bytes / 1e6, 3), "read_and_verify_seconds": round(read, 4),
                             "restored_epoch": int(blob["epoch"]) + 1 if blob else None, "process_rss_mb": round(saved["rss_mb"], 1)})
                print(rows[-1])
    out = {"machine": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(), "limits": resources.machine()},
           "setup": {"train_rows": N_TRAIN, "val_rows": N_VAL, "lookback": LOOK, "features": FEAT, "epochs": EPOCHS, "batch_size": 256, "device": "cpu"}, "rows": rows}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
