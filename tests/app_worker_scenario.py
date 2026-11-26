"""A whole app in its own process, training a deep network as a job, for the worker-crash test: ``python -m tests.app_worker_scenario <root>``. It runs until it is killed."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.utils.config import load_config
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch

PARAMS = {"epochs": 400, "max_rows": 1500, "width": 16, "blocks": 2, "layers": 2, "batch_size": 128, "lookback": 8, "patience": 400, "num_threads": 1, "grad_monitor": "standard"}
BODY = {"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "nbeats", "params": PARAMS}, "mode": "development", "run_id": "crash-run"}


def build(root: Path) -> App:
    config = load_config()
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    return App(config, builder=UniverseBuilder(config, TickerStore(root / "prices", fetch=_fetch), default_loader=lambda c: bundle), root=root)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    app = build(root)
    app.with_request("/api/ml/train", app.extra_post["/api/ml/train"], dict(BODY))
    while True:
        time.sleep(1)
