"""The research-record page in a real browser shows the execution mode of the newest experiment: a realistic re-run with its reconciliation, itemised costs and capacity, and a vectorized result with what it
cannot support. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_assets_ml import finished
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _mode, _page, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge"}, "explain": False}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("exec_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("exec_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    parent = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "vectorized parent", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, parent
    running.stop()


def _open_record(browser, running):  # noqa: F811
    pg = _page(browser, running)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "controls")
    return pg


def test_the_record_page_shows_a_vectorized_result_and_then_a_realistic_rerun_with_its_numbers(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open_record(browser, running)
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Execution mode: vectorized legacy')", timeout=120.0)
    text = pg.locator("#ml-results").text_content()
    assert "order lifecycle and partial fills" in text and "capacity" in text
    pg.close()
    child = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "execution": {"mode": "realistic_cash_equity", "capital": 750_000}}))
    pg = _open_record(browser, running)
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Execution mode: realistic cash equity')", timeout=120.0)
    text = pg.locator("#ml-results").text_content()
    assert child["id"][-6:] in text and "Reconciled" in text and "Itemised costs" in text and "Share of the target filled" in text and "applied once" in text
    assert "impact coefficient is an assumption" in text, "an uncalibrated cost is said to be an assumption on the page"
    assert not pg.problems, pg.problems
    pg.close()
