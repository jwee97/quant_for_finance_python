"""The Training monitor showing RECOVERED runs of the newly supported families in a real browser: a physics-informed network killed inside an L-BFGS step and resumed (two phases, the recomputed work recorded), a forecast
network killed and resumed (epochs), and a Stable-Baselines3 PPO agent killed and resumed (timesteps and episode rewards; skipped as dependency-blocked without an SB3 Python). The runs are produced by genuine
subprocesses that are killed with SIGKILL, then resumed by new subprocesses, on the app's own data folder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("playwright.sync_api")

from src.ml.deep import runs as run_files                     # noqa: E402
from src.utils.config import load_config                       # noqa: E402
from src.webapp import server                                  # noqa: E402
from src.webapp.api import App                                 # noqa: E402
from src.webapp.universe import TickerStore, UniverseBuilder   # noqa: E402
from tests.app_roots import isolated_root                      # noqa: E402
from tests.ml_helpers import make_bundle                       # noqa: E402
from tests.test_webapp import _fetch                           # noqa: E402
from tests.test_webapp_browser import browser, open_tab        # noqa: E402,F401
from tests.test_webapp_training_browser import _monitor        # noqa: E402
from tests.test_fresh_process_sb3 import PY as SB3_PY          # noqa: E402

from tests.fresh_process_scenarios import was_killed   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scenario(py, name, folder, mode, *extra):
    out = Path(str(folder) + f".{mode}.json")
    return subprocess.run([py, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(folder), mode, str(out), *extra], cwd=ROOT, capture_output=True, text=True, timeout=900)


def killed_then_resumed(py, name, folder, *kill):
    dead = scenario(py, name, folder, "interrupted", *kill)
    assert was_killed(dead.returncode), dead.stderr[-600:]
    res = scenario(py, name, folder, "resume")
    assert res.returncode == 0, res.stderr[-1000:]


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("recovery_root"), config)
    base = run_files.base(root)
    killed_then_resumed(sys.executable, "pinn", base / "pinn-run", "--kill-at", "closure:12")                       # killed inside an L-BFGS closure evaluation
    killed_then_resumed(sys.executable, "forecast_lstm", base / "lstm-run", "--kill-after", "3")
    if SB3_PY:
        killed_then_resumed(SB3_PY, "sb3_ppo", base / "ppo-run", "--kill-after", "3")
    import os, time
    dead = subprocess.run([sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", "forecast_lstm", str(base / "orphan-run"), "interrupted", str(base / "orphan.json"), "--kill-after", "4"],
                          cwd=ROOT, capture_output=True, text=True, timeout=900, env={**os.environ, "QUANT_FAULT": "enospc_ckpt:3-4"})
    assert was_killed(dead.returncode), dead.stderr[-600:]                                                     # checkpoints of epochs 3 and 4 could not be written: two epochs are logged but not committed
    for f in (base / "orphan-run").rglob("*"):
        if f.is_file():
            os.utime(f, (time.time() - 7200,) * 2)                                                              # nothing has written to it for two hours: orphaned
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    store = TickerStore(tmp_path_factory.mktemp("recovery_prices"), fetch=_fetch)
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


def open_run(pg, run):
    _monitor(pg)
    pg.click(f"#ml-results tr[data-run={run}] button")
    pg.wait_for_selector("#ml-training-detail .trtabs button", timeout=30_000)
    pg.wait_for_selector("#ml-training-view svg", timeout=30_000)


def test_a_recovered_pinn_shows_adam_and_lbfgs_as_two_phases_and_the_recomputed_work(pg):
    _monitor(pg)
    row = pg.inner_text("#ml-results tr[data-run=pinn-run]")
    assert "Completed" in row and "pinn" in row and "lbfgs" in row                                          # progress shows the phase the run ended in, not just a count
    open_run(pg, "pinn-run")
    head = pg.inner_text("#ml-training-detail")
    assert "Phase" in head and "L-BFGS" in head and "Resumed after" in head and "Exact" in head
    view = pg.inner_text("#ml-training-view")
    assert "Loss by optimisation phase" in view and "Adam loss" in view and "L-BFGS loss" in view
    pg.click("#ml-training-detail .trtabs button[data-trview=events]")
    pg.wait_for_selector("#ml-training-view .card")
    events = pg.inner_text("#ml-training-view")
    assert "recomputed work" in events and "L-BFGS" in events.replace("lbfgs", "L-BFGS") and "level exact" in events
    pg.click("#ml-training-detail .trtabs button[data-trview=gradient]")                                        # two optimizers, each with its own diagnostics
    pg.wait_for_selector("#ml-training-view .seg button")
    assert {b.inner_text() for b in pg.query_selector_all("#ml-training-view .seg button")} >= {"adam", "lbfgs"}


def test_a_recovered_forecast_network_continues_its_epochs(pg):
    open_run(pg, "lstm-run")
    head = pg.inner_text("#ml-training-detail")
    assert "Epoch" in head and "10 / 10" in head and "Resumed after epoch 3" in head or "Resumed after epoch 4" in head
    assert "Completed" in head or "Early stopping" in head
    assert "Loss by epoch" in pg.inner_text("#ml-training-view")


@pytest.mark.skipif(SB3_PY is None, reason="dependency-blocked: no Python with stable-baselines3, sb3-contrib and gymnasium")
def test_a_recovered_sb3_agent_is_shown_in_timesteps_and_rewards(pg):
    open_run(pg, "ppo-run")
    head = pg.inner_text("#ml-training-detail")
    assert "Timestep block" in head and "10 / 10" in head and "Timesteps" in head and "1,280" in head and "mean episode reward" in head and "Resumed after" in head
    view = pg.inner_text("#ml-training-view")
    assert "Loss by timesteps" in view and "mean reward" in view
    pg.click("#ml-training-detail .trtabs button[data-trview=events]")
    pg.wait_for_selector("#ml-training-view .card")
    text = pg.inner_text("#ml-training-view")
    assert "diagnostics note" in text and "recovery limits" in text


def test_an_interrupted_run_shows_what_is_committed_and_what_would_be_recomputed(pg):
    open_run(pg, "orphan-run")
    head = pg.inner_text("#ml-training-detail")
    assert "Orphaned" in head and "Committed" in head, head
    assert "2 epochs logged after it will be recomputed on resume" in head, head
    assert pg.query_selector("#ml-training-detail [data-resume], #ml-training-detail button")                    # and it can be resumed


def test_a_run_whose_checkpoints_could_not_be_written_says_so_prominently_and_names_the_latest_durable_epoch(pg):
    open_run(pg, "orphan-run")
    warn = pg.locator("#ml-training-detail [data-role=checkpoint-warning]")
    assert warn.count() == 1 and warn.get_attribute("role") == "alert"
    text = warn.inner_text()
    assert "Checkpoints could not be written" in text and "No space left" in text, text
    assert "latest completed epoch is 4" in text and "restart continues from, is 2" in text and "2 epochs of work will be redone" in text, text
