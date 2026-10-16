"""The Forecast tab in a real browser: the methods are listed by family with the random walk fixed as the yardstick, a forecast runs and its result reads in words (a verdict, the fan of the price with a choice of
method, what each method says now, the scoreboard, the calibration), a table stands behind every chart, the forecast can be downloaded, a reloaded page finds the run again, Stop ends one early, and an app older than
the tab says so. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from src.forecast import METHODS
from src.forecast.base import Forecast, Forecaster
from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_browser import _page_with, browser  # noqa: F401  (the module's Chromium fixture)

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live_forecast(tmp_path_factory):
    config = load_config()
    root = tmp_path_factory.mktemp("forecast_root")
    (root / "data").mkdir()
    (root / "data" / "raw").symlink_to(config.root / "data" / "raw", target_is_directory=True)
    (root / "docs").symlink_to(config.root / "docs", target_is_directory=True)
    bundle = make_bundle(n_assets=6, n=1400, seed=3, alpha=0.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("forecast_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    running = server.start(app, "127.0.0.1", 0)
    running.app = app
    yield running
    running.stop()


def _page(browser, live, width=1400):  # noqa: F811
    pg = browser.new_page(viewport={"width": width, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live.url)
    pg.wait_for_selector(".chip")
    return pg


def _open(page):
    page.click("#tab-forecast")
    page.wait_for_selector("#forecast .panel select[data-role=forecast-ticker]")
    page.wait_for_selector("#forecast .checks input[type=checkbox]")


def _only(page, ids):
    """Tick exactly these methods (the random walk is always on)."""
    page.evaluate("""(ids) => { for (const c of document.querySelectorAll('#forecast .checks input[type=checkbox]')) { const want = ids.includes(c.dataset.value); if (c.checked !== want) c.click(); } }""", ids)


def _run(page, ids=("riskmetrics", "garch", "hull_white"), horizon="21"):
    _open(page)
    _only(page, list(ids))
    page.select_option("#forecast input[data-key=horizon], #forecast select[data-key=horizon]", horizon)
    page.click("[data-role=forecast-effort] button[data-value=quick]")
    page.click("#forecast .runbar button.btn.primary")
    page.wait_for_selector("#forecast-results .verdict-card", timeout=240_000)


def test_the_form_lists_the_methods_by_family_with_the_yardstick_fixed_and_the_presets_tick_the_right_ones(browser, live_forecast):  # noqa: F811
    page = _page(browser, live_forecast)
    try:
        _open(page)
        assert page.locator("[data-role=forecast-ticker] option").count() == 6 and page.locator("[data-role=forecast-ticker]").input_value() == "A00"
        families = page.locator("#forecast .checks > span.hint").all_inner_texts()
        assert families[:4] == ["Monte Carlo", "Volatility", "Markov", "Gaussian process"] and "Combination" in families
        fixed = page.locator("#forecast label.f.inline input[disabled]")
        assert fixed.count() == 1 and fixed.is_checked()
        assert page.locator("#forecast .checks input[type=checkbox]").count() == len(METHODS) - 1 + 1       # every method but the yardstick, and the blend
        ticked = lambda: page.evaluate("[...document.querySelectorAll('#forecast .checks input[type=checkbox]')].filter((c) => c.checked).map((c) => c.dataset.value)")   # noqa: E731
        assert "garch" in ticked() and "gp_price" not in ticked() and "random_walk" not in ticked()
        page.click("#forecast button:has-text('All of them')")
        assert len(ticked()) == len(METHODS)                                       # all methods but the yardstick (which has no box) plus the blend
        page.click("#forecast button:has-text('Fast ones only')")
        assert "hmm" not in ticked() and "riskmetrics" in ticked() and "gp_returns" not in ticked()
        page.click("#forecast button:has-text('Recommended')")
        assert "hmm" in ticked() and "gp_price" not in ticked()
        assert "methods ticked besides the random walk" in page.inner_text("[data-role=forecast-summary]")
        assert not page.problems, page.problems
    finally:
        page.close()


def test_a_run_with_no_method_ticked_is_refused_with_the_reason_and_the_button_is_off(browser, live_forecast):  # noqa: F811
    page = _page(browser, live_forecast)
    try:
        _open(page)
        _only(page, [])
        assert page.is_disabled("#forecast .runbar button.btn.primary")
        assert "Tick at least one method besides the random walk" in page.inner_text("#forecast .runbar .guard")
        _only(page, ["garch"])
        assert page.is_enabled("#forecast .runbar button.btn.primary")
    finally:
        page.close()


def test_a_forecast_reads_in_words_draws_the_fan_and_leaves_a_table_behind_every_chart(browser, live_forecast):  # noqa: F811
    page = _page(browser, live_forecast)
    try:
        _run(page)
        verdict = page.locator("#forecast-results .verdict-card")
        assert verdict.get_attribute("data-level") in ("good", "mixed", "bad") and verdict.locator("li").count() >= 5
        assert "Tested on" in verdict.inner_text() and "A forecast is not a strategy" in verdict.inner_text()
        assert "Forecast for A00" in page.inner_text("#forecast-results .head h2") and "21 trading days ahead" in page.inner_text("#forecast-results .head .sub")

        fan = page.locator("[data-role=forecast-fan]")
        assert fan.locator("svg.chart").count() == 1 and fan.locator("svg.chart path[fill-opacity]").count() >= 2, "the fan is drawn as bands"
        pills = fan.locator(".tabsmall button")
        assert pills.count() == 4 and "Random walk" in pills.all_inner_texts() and "GJR-GARCH" in pills.all_inner_texts()
        before = fan.locator("p.note").inner_text()
        fan.locator(".tabsmall button[data-value=riskmetrics]").click()
        assert fan.locator(".tabsmall button[data-value=riskmetrics]").get_attribute("aria-pressed") == "true" and fan.locator("p.note").inner_text() != before
        assert "RiskMetrics" in fan.locator("p.note").inner_text() or "Exponentially weighted" in fan.locator("p.note").inner_text()
        fan.locator(".seg button:has-text('Table')").click()
        rows = fan.locator("table.data tbody tr")
        assert rows.count() >= 5 and fan.locator("table.data th").all_inner_texts()[:2] == ["Day", "Date"]
        fan.locator(".seg button:has-text('Chart')").click()

        now = page.locator("[data-role=forecast-now] table.data")
        assert now.locator("tbody tr").count() == 4 and "Random walk" in now.inner_text() and "yardstick" in now.inner_text()
        board = page.locator("[data-role=forecast-scoreboard]")
        assert board.locator("svg.chart").count() == 1 and "Skill" in board.inner_text()
        board.locator(".seg button:has-text('Table')").click()
        assert board.locator("table.data tbody tr").count() == 3 and "Skill at 21 days" in board.inner_text()
        cal = page.locator("[data-role=forecast-calibration]")
        assert cal.locator("svg.chart").count() == 1 and cal.locator(".tile").count() == 4 and "90% range held the outcome" in cal.inner_text()
        cal.locator(".tabsmall button[data-value=garch]").click()
        assert cal.locator(".tabsmall button[data-value=garch]").get_attribute("aria-pressed") == "true"
        assert "What each method assumes" in page.inner_text("#forecast-results") and "What the methods learned" in page.inner_text("#forecast-results")
        assert page.locator("#forecast-results details.how").count() >= 4
        assert not page.problems, page.problems
    finally:
        page.close()


def test_the_forecast_can_be_downloaded_as_a_csv_of_the_fans(browser, live_forecast):  # noqa: F811
    page = _page(browser, live_forecast)
    try:
        _run(page, ids=("garch", "hull_white"), horizon="5")
        with page.expect_download() as info:
            page.click("button[data-action=download-forecast]")
        text = open(info.value.path(), encoding="utf-8").read().splitlines()
        assert text[0] == "method,step,date,q05,q25,median,q75,q95" and len(text) == 1 + 3 * 5          # the yardstick and two methods, five days each
        first = text[1].split(",")
        assert first[0] == "random_walk" and first[1] == "1" and float(first[3]) < float(first[5]) < float(first[7])
    finally:
        page.close()


def test_a_reloaded_page_finds_the_run_again_and_opens_it(browser, live_forecast):  # noqa: F811
    page = _page(browser, live_forecast)
    try:
        _run(page, ids=("riskmetrics",), horizon="5")
        page.reload()
        page.wait_for_selector("body[data-ready]")                                        # the page comes back on the tab it was on, so the Backtest chips are hidden
        page.click("#tab-forecast")
        page.wait_for_selector("#forecast-results .runs .runlist li[data-status=done]")
        assert "Recent runs" in page.inner_text("#forecast-results .runs")
        page.click("#forecast-results .runs li[data-status=done] button[data-action=open-run]")
        page.wait_for_selector("#forecast-results .verdict-card")
        assert "5 trading days ahead" in page.inner_text("#forecast-results .head .sub")
        assert not page.problems, page.problems
    finally:
        page.close()


class Gated(Forecaster):
    name, short, label, min_history = "gated", "Gated", "Gated", 300
    gate = threading.Event()
    reached = threading.Event()
    calls = 0

    def fit(self, h, horizons):
        return {}

    def predict(self, state, h, horizons, n, rng):
        type(self).calls += 1
        if type(self).calls == 34:
            type(self).reached.set()
            assert type(self).gate.wait(60)
        return Forecast(np.cumsum(0.01 * rng.standard_normal((n, max(horizons))), axis=1))


def test_stop_on_a_running_row_ends_the_run_early_and_the_page_says_how_many_forecasts_it_has(browser, live_forecast, monkeypatch):  # noqa: F811
    Gated.gate, Gated.reached, Gated.calls = threading.Event(), threading.Event(), 0
    monkeypatch.setitem(METHODS, "gated", Gated())
    page = _page(browser, live_forecast)
    try:
        job = live_forecast.app.extra_post["/api/forecast/run"]({"ticker": "A00", "methods": ["gated"], "effort": "quick"})["job"]
        assert Gated.reached.wait(120)
        page.click("#tab-forecast")
        row = f"#forecast-results .runs li[data-job='{job}']"
        page.wait_for_selector(f"{row}[data-status=running] button[data-action=stop-run]")
        page.click(f"{row}[data-status=running] button[data-action=stop-run]")
        Gated.gate.set()
        page.wait_for_selector(f"{row}[data-status=done] button[data-action=open-run]", timeout=120_000)
        page.click(f"{row}[data-status=done] button[data-action=open-run]")
        page.wait_for_selector("#forecast-results .verdict-card")
        assert "Stopped early" in page.inner_text("#forecast-results") and live_forecast.app.job(job)["result"]["stopped"] is True
    finally:
        Gated.gate.set()
        page.close()


def test_an_app_older_than_the_tab_says_so_and_the_other_tabs_still_work(browser, live_forecast):  # noqa: F811
    page = _page_with(browser, live_forecast, rewrite=lambda catalog: catalog.update(api_version=8))
    try:
        page.wait_for_selector(".chip")
        page.click("#tab-forecast")
        page.wait_for_selector("#forecast .card.empty")
        assert "needs a newer app" in page.inner_text("#forecast") and "version 8" in page.inner_text("#forecast") and "needs 9" in page.inner_text("#forecast")
        page.click("#tab-tune")
        page.wait_for_selector("#tune .panel")
    finally:
        page.close()
