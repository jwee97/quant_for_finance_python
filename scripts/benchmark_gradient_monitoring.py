"""What gradient monitoring costs: time per epoch and peak process memory at each level (off, lightweight, standard, detailed) for N-BEATS-style networks of three sizes, each level in a fresh process.

    python scripts/benchmark_gradient_monitoring.py [--out docs/generated/gradient_monitoring_benchmark.json] [--threads 1]

Synthetic data, CPU. It says how the monitor behaves on the machine it runs on (pass --threads 1 to imitate a single-core Replit container); it is not a claim about any other machine."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

SIZES = {"small": (16, 2, 2), "medium": (64, 3, 3), "large": (256, 4, 4)}       # width, blocks, layers
LEVELS = ("off", "lightweight", "standard", "detailed")
LOOK, FEAT, N_TRAIN, N_VAL, EPOCHS, BATCH = 20, 30, 20000, 2000, 4, 128


def child(level: str, size: str, threads: int) -> None:
    import numpy as np
    import torch

    from src.ml import resources
    from src.ml.deep import trainer as tn
    from src.ml.deep.nets import build_net

    torch.set_num_threads(threads)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_TRAIN + N_VAL, LOOK, FEAT)).astype(np.float32)
    y = (X[:, -1, 0] + 0.1 * rng.normal(size=len(X))).astype(np.float32)
    split = {"train": np.arange(N_TRAIN), "val": np.arange(N_TRAIN, N_TRAIN + N_VAL)}
    batch = lambda name, rows: (torch.from_numpy(X[split[name][rows]]), torch.from_numpy(y[split[name][rows]]))          # noqa: E731
    loss = lambda net, b: ((net(b[0]).reshape(len(b[1]), -1)[:, 0] - b[1]) ** 2).mean()                                  # noqa: E731
    width, blocks, layers = SIZES[size]
    torch.manual_seed(0)
    net = build_net("nbeats", FEAT, LOOK, 1, {"width": width, "blocks": blocks, "layers": layers, "pool_sizes": "4,2,1", "dropout": 0.1})
    peak = [resources.rss_mb() or 0.0]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], resources.rss_mb() or 0.0)
            time.sleep(0.01)
    threading.Thread(target=watch, daemon=True).start()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg = tn.TrainConfig(epochs=EPOCHS, batch_size=BATCH, learning_rate=0.01, patience=50, checkpoint_dir=tmp, config_fp="b", data_fp="b", seed=1, grad_level=level, grad_clip=1.0)
        t0 = time.perf_counter()
        h = tn.fit_network(net, batch, loss, N_TRAIN, N_VAL, cfg)
        total = time.perf_counter() - t0
        size_kb = sum(f.stat().st_size for f in Path(tmp).glob("gradients.jsonl")) / 1024
    stop.set()
    print(json.dumps({"epoch_seconds": sorted(h.epoch_seconds)[len(h.epoch_seconds) // 2], "total_seconds": total, "peak_rss_mb": peak[0], "gradient_log_kb": size_kb, "params": int(sum(p.numel() for p in net.parameters()))}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/generated/gradient_monitoring_benchmark.json")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--child", nargs=2)
    args = ap.parse_args()
    if args.child:
        child(args.child[0], args.child[1], args.threads)
        return
    rows = []
    for size in SIZES:
        base = None
        for level in LEVELS:
            runs = []
            for _ in range(3):                                                                    # the median of three fresh processes
                out = subprocess.run([sys.executable, "-W", "ignore", __file__, "--child", level, size, "--threads", str(args.threads)], capture_output=True, text=True, check=True)
                runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
            runs.sort(key=lambda r: r["epoch_seconds"])
            mid = runs[1]
            base = mid if level == "off" else base
            row = {"size": size, "level": level, "params": mid["params"], "epoch_seconds": round(mid["epoch_seconds"], 4), "overhead_vs_off_pct": round(100.0 * (mid["epoch_seconds"] / base["epoch_seconds"] - 1.0), 1),
                   "peak_rss_mb": round(mid["peak_rss_mb"], 1), "extra_rss_mb_vs_off": round(mid["peak_rss_mb"] - base["peak_rss_mb"], 1), "gradient_log_kb": round(mid["gradient_log_kb"], 1)}
            rows.append(row)
            print(row, flush=True)
    result = {"machine": {"platform": platform.platform(), "python": platform.python_version(), "threads": args.threads}, "setup": {"train_rows": N_TRAIN, "val_rows": N_VAL, "batch": BATCH, "epochs": EPOCHS, "lookback": LOOK, "features": FEAT,
              "clip": 1.0}, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
