"""An app server in its own process, for the browser-reconnection check: ``python -m tests.app_server_scenario <root> <port>`` serves the Training monitor on 127.0.0.1:<port> until it is killed."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch

if __name__ == "__main__":
    root, port = Path(sys.argv[1]), int(sys.argv[2])
    config = load_config()
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(root / "prices", fetch=_fetch), default_loader=lambda c: bundle), root=root)
    run = server.start(app, port=port)
    print("serving", run.token, flush=True)
    while True:
        time.sleep(1)
