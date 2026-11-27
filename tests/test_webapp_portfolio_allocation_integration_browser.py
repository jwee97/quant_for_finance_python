"""The Portfolio allocation view with currency and research integration in a real browser: a two-currency universe (an offline fake provider supplies the exchange rate), the assumptions the page states about the hedge, the
real historical weights and uncertainty it draws, the benchmark comparison, and saving the finished study to a research session and reading it back there. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.ml_helpers import make_bundle
from tests.test_ml_fx_integration import make_fetch
from tests.test_webapp_assets_ml import finished
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_portfolio_allocation_browser import _open_section, _results_text, _wait_run
from tests.test_webapp_workspaces_browser import _mode, _page, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")
PARENT = {"dataset": {"universe": ["NEWAAA", "NEWTOYO"], "feature_groups": ["price", "momentum", "volatility"], "horizon": 5, "target": "fwd_ret_5", "min_history": 60},
          "split": {"min_train_days": 300, "refit_every": 126, "validation_fraction": 0.2, "test_fraction": 0.2, "forward_fraction": 0.1}, "model": {"kind": "ridge"},
          "signal": {"method": "proportional", "holding_period": 1}, "backtest": {"rebalance": "weekly", "target_vol": None}, "explain": False}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("alloc_fx_root"), config)
    builder = UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("alloc_fx_prices"), fetch=make_fetch()), default_loader=lambda c: make_bundle(n_assets=2, n=80, macro=False))
    app = App(config, builder=builder, root=root)
    parent = finished(app, app.extra_post["/api/ml/train"]({**PARENT, "name": "us and japan", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, parent
    running.stop()


def _open(browser, running):  # noqa: F811
    pg = _page(browser, running)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "allocation")
    pg.wait_for_selector("#ml select[data-key=alAllocator]", timeout=60000)
    return pg


def test_the_page_states_the_currency_assumptions_draws_real_weights_and_saves_the_study(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.select_option("#ml select[data-key=alAllocator]", "inverse_vol")
    _open_section(pg, "9. Currency")
    pg.select_option("#ml select[data-key=alFxBase]", "USD")
    _open_section(pg, "9. Currency")                                                              # choosing a base redraws the form: the section closes again
    pg.fill("#ml input[data-key=alFxCcy]", "NEWAAA:USD, NEWTOYO:JPY")
    pg.fill("#ml input[data-key=alFxHedge]", "50")
    pg.click("#ml .runbar button.primary")
    _wait_run(pg, "Inverse volatility")
    text = _results_text(pg)
    for needle in ("Currency and FX (base USD)", "Not exchange-accurate", "forward points and financing rates are not available", "Forecast conversion", "Currency conversion cost", "Hedge P&L / cost", "Usable days", "Currency exposure through time",
                   "Weights through time", "Turnover and cost of each decision", "Risk contribution at the latest decision", "Forecast uncertainty (sd)", "Horizon, holding and decisions"):
        assert needle in text, needle
    assert "exchange-accurate hedging" not in text.replace("Not exchange-accurate", ""), "the page never claims exchange-accurate hedging"
    assert pg.locator("#ml-results [data-allocation-fx]").count() == 1
    rows = pg.locator("#ml-results [data-allocation-fx] table").first.locator("tbody tr")
    assert rows.count() >= 1 and "JPY" in " ".join(pg.locator("#ml-results [data-allocation-fx] table").first.locator("tbody tr td:first-child").all_inner_texts())
    pg.click("#ml-results button[data-role=save-allocation-study]")
    pg.wait_for_selector("dialog[data-role=save-study]")
    pg.fill("dialog[data-role=save-study] input[data-key=new_session]", "Allocation with currency")
    pg.fill("dialog[data-role=save-study] input[data-key=title]", "Inverse volatility, USD base")
    pg.click("dialog[data-role=save-study] button[data-role=save-confirm]")
    _wait_until(pg, "document.querySelector('dialog[data-role=save-study]') === null", timeout=60.0)
    sessions = app.extra_get["/api/research/sessions"]({})["sessions"]
    mine = next(s for s in sessions if s["name"] == "Allocation with currency")
    saved = app.extra_get["/api/research/session"]({"id": mine["id"]})["session"]
    assert len(saved["artifacts"]) == 1
    art = app.extra_get["/api/research/artifact"]({"id": saved["artifacts"][0]})
    art = art.get("artifact", art)
    assert art["title"] == "Inverse volatility, USD base" and art["kind"] == "ml_experiment" and art["config"]["currency"] == "USD"
    assert art["validation"]["fx"]["exchange_accurate"] is False, "the saved study keeps the statement that the hedge is not exchange-accurate"
    assert not pg.problems, pg.problems
    pg.close()
