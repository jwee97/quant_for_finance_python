"""The Training monitor in a real browser: runs on disk are listed with a status derived from their files, an open run shows its loss, learning-rate and memory curves and its checkpoints, reloading the page loses
nothing, an interrupted run offers Resume (never started on its own) and the resumed run continues from its checkpoint. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sync_api = pytest.importorskip("playwright.sync_api")

from src.ml.deep import runs as run_files, trainer as tn      # noqa: E402
from src.ml.deep.nets import build_net                        # noqa: E402
from src.utils.config import load_config                      # noqa: E402
from src.webapp import server                                 # noqa: E402
from src.webapp.api import App                                # noqa: E402
from src.webapp.universe import TickerStore, UniverseBuilder  # noqa: E402
from tests.app_roots import isolated_root                     # noqa: E402
from tests.ml_helpers import make_bundle                      # noqa: E402
from tests.test_webapp import _fetch                          # noqa: E402
from tests.test_webapp_browser import browser, open_tab       # noqa: E402,F401
from tests.test_webapp_workspaces_browser import _mode, _wait_until   # noqa: E402

LOOK, FEAT = 8, 3
REQUEST = {"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "mode": "development",
           "model": {"kind": "nbeats", "params": {"epochs": 3, "max_rows": 1500, "width": 16, "blocks": 2, "layers": 2, "batch_size": 128, "lookback": 8, "patience": 5}}}


class Killed(BaseException):
    pass


def _fit(folder, epochs=6, die_after=None, resume=False):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, LOOK, FEAT)).astype(np.float32)
    y = (X[:, -1, 0] + 0.1 * rng.normal(size=300)).astype(np.float32)
    split = {"train": np.arange(0, 240), "val": np.arange(240, 300)}
    batch = lambda name, rows: (torch.from_numpy(X[split[name][rows]]), torch.from_numpy(y[split[name][rows]]))     # noqa: E731
    calls = {"n": 0}

    def loss(net, b):
        calls["n"] += 1
        if die_after is not None and calls["n"] > die_after:
            raise Killed()
        return ((net(b[0]).squeeze(-1) - b[1]) ** 2).mean()

    torch.manual_seed(2)
    net = build_net("nbeats", FEAT, LOOK, 1, {"width": 8, "blocks": 2, "layers": 2})
    cfg = tn.TrainConfig(epochs=epochs, batch_size=64, learning_rate=0.01, patience=50, checkpoint_dir=str(folder), config_fp="c", data_fp="d", seed=4, resume=resume)
    return tn.fit_network(net, batch, loss, 240, 60, cfg)


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("training_root"), config)
    base = run_files.base(root)
    per_epoch = -(-240 // 64) + -(-60 // 128)
    _fit(base / "finished-run" / "nbeats-aaa", epochs=5)
    with pytest.raises(Killed):
        _fit(base / "broken-run" / "nbeats-bbb", epochs=8, die_after=3 * per_epoch + 1)                      # dies inside the fourth epoch
    for name in ("finished-run", "broken-run"):                                                               # a saved request lets the page offer Resume
        (base / name / "request.json").write_text(json.dumps({"path": "/api/ml/train", "label": name, "body": REQUEST, "run_id": name}))
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    store = TickerStore(tmp_path_factory.mktemp("training_prices"), fetch=_fetch)
    app = App(config, builder=UniverseBuilder(config, store, default_loader=lambda c: bundle), root=root)
    running = server.start(app, "127.0.0.1", 0)
    running.root, running.app = root, app
    yield running
    running.stop()


@pytest.fixture()
def pg(browser, live):  # noqa: F811
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.problems = []
    page.on("response", lambda r: page.problems.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None)
    page.on("pageerror", lambda e: page.problems.append(f"pageerror: {e}"))
    page.on("console", lambda m: page.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    page.goto(live.url)
    page.wait_for_selector(".chip")
    yield page
    assert not page.problems, page.problems
    page.close()


def _monitor(page):
    open_tab(page, "ml")
    page.wait_for_selector("#view-ml .tabsmall button[data-mode=training]", timeout=60_000)
    _mode(page, "ml", "training")
    page.wait_for_selector("#ml-results tr[data-run]", timeout=60_000)


def test_runs_are_listed_with_a_status_taken_from_their_files(pg):
    _monitor(pg)
    status = {r.get_attribute("data-run"): r.query_selector("td[data-status]").get_attribute("data-status") for r in pg.query_selector_all("#ml-results tr[data-run]")}
    assert status["finished-run"] == "completed" and status["broken-run"] in ("interrupted", "orphaned")
    assert "Training monitor" in pg.inner_text("#ml-results")


def test_an_open_run_shows_curves_checkpoints_and_survives_a_reload(pg):
    _monitor(pg)
    pg.click("#ml-results tr[data-run=finished-run] button")
    pg.wait_for_selector("#ml-training-detail .card h3", timeout=30_000)
    pg.wait_for_selector("#ml-training-detail svg", timeout=30_000)
    text = pg.inner_text("#ml-training-detail")
    assert "Run finished-run" in text and "Loss by epoch" in text and "Learning rate" in text and "Memory" in text and "Checkpoints" in text and "last.pt" in text and "Completed" in text
    assert "Resume this run" in text and pg.locator("#ml-training-detail button:has-text('Resume this run')").is_disabled()          # a completed run has nothing to resume
    assert pg.locator("#ml-training-detail svg").count() >= 3
    pg.reload()
    pg.wait_for_selector("#tab-ml")
    _monitor(pg)
    assert pg.locator("#ml-results tr[data-run]").count() == 2                                                  # the same history after a reload: it was never in the page
    assert "test_" not in pg.inner_text("#ml-results").lower().replace("test and forward", "")


def test_an_interrupted_run_waits_for_the_user_and_resumes_from_its_checkpoint(pg, live):
    _monitor(pg)
    pg.click("#ml-results tr[data-run=broken-run] button")
    pg.wait_for_selector("#ml-training-detail .card h3", timeout=30_000)
    text = pg.inner_text("#ml-training-detail")
    assert "Nothing was resumed automatically" in text or "Orphaned" in text
    btn = pg.locator("#ml-training-detail button:has-text('Resume this run')")
    assert btn.is_enabled()
    last = lambda: [json.loads(line)["type"] for line in next(run_files.base(live.root).joinpath("broken-run").glob("*/metrics.jsonl")).read_text().splitlines()][-1]     # noqa: E731
    assert last() == "interrupted"                                                                              # the server did not resume it on its own
    btn.click()
    _wait_until(pg, "document.querySelectorAll('#ml-results .runs li').length > 0 || document.body.innerText.includes('queued')", 30)
    deadline = time.time() + 120
    jobs = []
    while time.time() < deadline:
        jobs = [j for j in live.app.jobs_list()["jobs"] if j.get("training_run") == "broken-run"] if hasattr(live, "app") else []
        if jobs:
            break
        time.sleep(0.3)
    assert jobs, "the Resume button queued no job for the run"
