"""The Machine Learning Lab's report shows the buys and sells of a run that uses an ordinary position rule (not only decision_policy), and the Strategy Explorer and Model Comparison let the person choose the
universe. In a real browser; skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.ml_helpers import make_bundle
from tests.test_ml_strategy_api import BASE
from tests.test_webapp import _fetch
from tests.test_webapp_assets_ml import finished
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _page, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("lab_trades_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("lab_trades_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    run = finished(app, app.extra_post["/api/ml/train"]({"dataset": BASE["dataset"], "split": BASE["split"], "model": BASE["model"], "explain": False, "name": "lab run", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, run
    running.stop()


def test_the_labs_report_draws_the_buys_and_sells_of_an_ordinary_position_rule(browser, live):  # noqa: F811
    running, app, run = live
    pg = _page(browser, running)
    pg.evaluate("id => localStorage.setItem('ql.ml.cur', JSON.stringify(id))", run["id"])
    pg.reload()
    pg.wait_for_selector(".chip")
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    pg.click("#ml button:has-text('Experiments')")
    pg.locator("#ml button:has-text('Open')").first.click()
    pg.wait_for_selector("[data-role=lab-trades] svg.chart .event", timeout=60000)
    card = pg.locator("[data-role=lab-trades]")
    text = card.inner_text()
    assert "Buys and sells" in text and "keeps no decision log" in text and "Turnover / year" in text and "Cost drag / year" in text
    assert "Win rate" not in text, "a rule that holds every asset has no round trips to win or lose, so those tiles are not shown"
    assert card.locator("svg.chart .event").count() > 0, "every trade is a marker on the price"
    assert card.locator("svg.chart .event title").first.text_content().strip(), "each marker says why"
    card.locator("select[data-key=asset]").select_option(index=1)
    _wait_until(pg, "document.querySelectorAll('[data-role=lab-trades] svg.chart').length >= 2", timeout=30.0)
    card.locator("button:has-text('Open in the ML Strategy Explorer')").click()
    pg.wait_for_selector("#strategy [data-role=strategy-tiles] .tile", timeout=60000)
    assert pg.url.endswith(f"#/strategy?id={run['id']}") and not pg.problems, pg.problems
    pg.close()


def test_the_strategy_explorer_and_the_comparison_offer_every_universe(browser, live):  # noqa: F811
    running, _, _ = live
    pg = _page(browser, running)
    for tab, form in (("strategy", "[data-role=strategy-builder]"), ("modelcompare", "[data-role=compare-form]")):
        pg.click(f"#tab-{tab}")
        pg.wait_for_selector(form)
        if tab == "strategy":
            pg.locator("[data-role=strategy-builder] summary").click()
        values = pg.locator(f"{form} select[data-key=universe] option").evaluate_all("els => els.map(e => e.value)")
        assert values == ["etf", "crypto", "futures", "custom"], values
        assert pg.locator(f"{form} input[data-key=tickers]").count() == 1
    pg.close()
