"""The ML Strategy Explorer in a real browser: a decision-policy run is listed, its price chart carries trade markers with reasons, the decision log filters and searches, the round trips and executions are
tables, and a date can be replayed. Skipped when Playwright or Chromium is missing."""

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
from tests.test_webapp_workspaces_browser import _page, _wait_until
from tests.test_ml_strategy_api import BASE

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("strategy_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("strategy_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    run = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "policy run", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, run
    running.stop()


def test_the_explorer_shows_markers_the_log_the_trades_and_a_replay(browser, live):  # noqa: F811
    running, app, run = live
    pg = _page(browser, running)
    pg.click("#tab-strategy")
    pg.wait_for_selector("#strategy [data-role=strategy-tiles] .tile", timeout=120000)
    text = pg.locator("#strategy").inner_text()
    assert "ML Strategy Explorer" in text and "Prediction:" in text and "Decision:" in text and "Execution:" in text, "the three layers are named on the page"
    assert "Round trips" in text and "Win rate" in text and "Profit factor" in text and "Turnover / year" in text
    assert pg.locator("#strategy .card [data-role=marker-legend]").count() >= 1
    assert pg.locator("#strategy svg.chart .event").count() > 0, "the trades are drawn on the price"
    assert pg.locator("#strategy svg.chart .event title").first.text_content().strip(), "each marker says why"
    # the decision log: filter by action, then search the reasons
    rows_all = pg.locator("[data-role=decision-log] tbody tr").count()
    assert rows_all > 0
    pg.select_option("[data-role=decision-log] select[data-key=action]", "BUY")
    _wait_until(pg, "[...document.querySelectorAll('[data-role=decision-log] tbody tr td:nth-child(3)')].length > 0 && [...document.querySelectorAll('[data-role=decision-log] tbody tr td:nth-child(3)')].every(td => td.textContent === 'BUY')", timeout=30.0)
    pg.select_option("[data-role=decision-log] select[data-key=action]", "")
    pg.fill("[data-role=decision-log] input[data-key=q]", "top")
    _wait_until(pg, "[...document.querySelectorAll('[data-role=decision-log] tbody tr td:last-child')].length > 0 && [...document.querySelectorAll('[data-role=decision-log] tbody tr td:last-child')].every(td => td.textContent.toLowerCase().includes('top'))", timeout=30.0)
    # round trips, then executed trades
    assert "Closed because" in pg.locator("[data-role=trades]").inner_text()
    pg.click("[data-role=trades] .seg button:nth-child(2)")
    pg.wait_for_selector("[data-role=trades] th:has-text('Decided')", timeout=30000)
    assert "Executed" in pg.locator("[data-role=trades]").inner_text()
    # replay a decision date: only what was known, the trades that followed apart
    pg.click("[data-role=replay] button:has-text('Replay')")
    pg.wait_for_selector("[data-role=replay] table", timeout=30000)
    rp = pg.locator("[data-role=replay]").inner_text()
    assert "Known on" in rp and "Book going in" in rp and "Executed afterwards" in rp
    # the builder: running a new strategy from the form adds a second run to the picker
    pg.click("[data-role=strategy-builder] summary")
    pg.select_option("[data-role=strategy-builder] select[data-key=model]", "lasso")
    pg.fill("[data-role=strategy-builder] input[data-key=min_train_days]", "504")
    pg.click("[data-role=strategy-builder] button:text-is('Run the strategy')")
    _wait_until(pg, "document.querySelectorAll('[data-role=strategy-picker] option').length === 2", timeout=300.0)
    assert not pg.problems, pg.problems
    pg.close()
