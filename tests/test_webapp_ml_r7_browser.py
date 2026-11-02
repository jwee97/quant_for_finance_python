"""The research record in a real browser shows Release 7: the delay report (computed once, kept, read back by a fresh page) and the new comparison families, which are chosen, frozen, run once and listed with
their verdict, while a combination is offered only once components have been compared on their own. Skipped when Playwright or Chromium is missing."""

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
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge"}, "explain": False}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("r7_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("r7_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    exp = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "r7-record", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, exp
    running.stop()


def _open(browser, running):  # noqa: F811
    pg = _page(browser, running)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "controls")
    return pg


def _has(pg, text, timeout=120.0):
    _wait_until(pg, f"document.querySelector('#ml-results').textContent.includes({text!r})", timeout=timeout)


def test_the_delay_report_is_computed_once_and_read_back_by_a_fresh_page(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running)
    _has(pg, "Delay report")
    assert "have not been computed for this experiment" in pg.locator("#ml-results").text_content()
    pg.get_by_role("button", name="Compute the delay report").click()
    _has(pg, "Break-even cost (bps)")
    text = pg.locator("#ml-results").text_content()
    for needle in ("stale book", "Calendar days (mean)", "IC against undelayed", "Share of the book filled", "declaration", "The experiment declared"):
        assert needle in text, needle
    assert not pg.problems, pg.problems
    pg.close()
    pg = _open(browser, running)
    _has(pg, "Break-even cost (bps)")
    assert not pg.problems, pg.problems
    pg.close()


def test_a_rebalancing_card_is_chosen_run_and_listed_and_a_combination_waits_for_its_components(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running)
    _has(pg, "Controlled comparisons against this experiment")
    pg.select_option("label:has-text('Family') select", "combination")
    _has(pg, "Run at least two single-change cards first")
    pg.select_option("label:has-text('Family') select", "rebalancing")
    _has(pg, "No-trade region, margin 1 x the incremental cost")
    pg.get_by_label("No-trade region, margin 3 x the incremental cost").check()
    pg.fill("label:has-text('Hypothesis') input", "a no-trade region saves more cost than it gives up")
    pg.fill("label:has-text('The one change') input", "the rebalance rule")
    pg.get_by_role("button", name="Freeze the card and run it").click()
    _wait_until(pg, "/(supported on development|inconclusive|worse)/.test(document.querySelector('#ml-results').textContent) && document.querySelector('#ml-results').textContent.includes('cmp-')", timeout=300.0)
    text = pg.locator("#ml-results").text_content()
    assert "Rebalancing" in text and "a no-trade region saves more cost than it gives up" in text
    assert not pg.problems, pg.problems
    pg.close()
