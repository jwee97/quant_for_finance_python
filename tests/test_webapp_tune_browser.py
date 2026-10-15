"""The Tune tab in a real browser: the settings of a strategy appear with their ranges and help, a search runs and its result reads in words (a verdict, the settings, the two windows, the charts), the winner
can be sent to the Backtest tab, a reloaded page finds the search again, Stop ends one early, and an app older than the tab says so. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import threading
import time

import pytest

from src.utils.config import load_config
from src.webapp import server, tune_api
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_tune_api import FakePipeline
from tests.test_webapp import _fetch
from tests.test_webapp_browser import _page_with, browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _wait_until

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live_tune(tmp_path_factory):
    config = load_config()
    root = tmp_path_factory.mktemp("tune_root")
    (root / "data").mkdir()
    (root / "data" / "raw").symlink_to(config.root / "data" / "raw", target_is_directory=True)
    (root / "docs").symlink_to(config.root / "docs", target_is_directory=True)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("tune_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
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


def _open_tune(page, strategy="kalman_trend"):
    page.click("#tab-tune")
    page.wait_for_selector(".tuneset")
    if strategy:
        page.select_option("[data-role=tune-strategy]", strategy)
        page.wait_for_selector(f".tuneset[data-name={'level_noise' if strategy == 'kalman_trend' else 'lookback'}]")


def _ensure_search(app):
    """A finished search in the job list, made through the API when an earlier test has not left one (the tests can be run one at a time)."""
    if any(j["kind"] == "tune" and j["status"] == "done" for j in app.jobs_list()["jobs"]):
        return
    body = {"strategy": "kalman_trend", "tune": [{"name": "level_noise", "low": 0.002, "high": 0.5, "scale": "log"}, {"name": "slope_noise", "low": 1e-7, "high": 1e-3, "scale": "log"}],
            "allocator": "sleeves", "effort": "custom", "trials": 6, "seconds": 120}
    job = app.extra_post["/api/tune/run"](body)["job"]
    end = time.time() + 180
    while app.job(job)["status"] in ("queued", "running") and time.time() < end:
        time.sleep(0.05)
    assert app.job(job)["status"] == "done"


def _small_search(page, trials="6"):
    page.click("[data-role=tune-effort] button[data-value=custom]")
    page.fill("[data-role=tune-custom] input[data-key=trials]", trials)
    page.fill("[data-role=tune-custom] input[data-key=seconds]", "120")
    page.click("#tune .runbar button.btn.primary")


def test_the_tab_lists_a_strategys_settings_with_help_ranges_and_a_reason_when_it_cannot_run(browser, live_tune):  # noqa: F811
    page = _page(browser, live_tune)
    try:
        page.click("#tab-tune")
        page.wait_for_selector(".tuneset")
        assert page.locator("[data-role=tune-strategy]").input_value() == "dual_momentum", "it starts from the strategy the Backtest tab has"
        assert {page.locator(".tuneset").nth(i).get_attribute("data-name") for i in range(page.locator(".tuneset").count())} == {"lookback", "skip", "top_k"}
        assert "8 tickers" in page.inner_text("[data-role=tune-universe]") and "Backtest tab" in page.inner_text("[data-role=tune-universe]")
        page.select_option("[data-role=tune-strategy]", "kalman_trend")
        page.wait_for_selector(".tuneset[data-name=level_noise]")
        names = [page.locator(".tuneset").nth(i).get_attribute("data-name") for i in range(page.locator(".tuneset").count())]
        assert names == ["level_noise", "slope_noise", "vol_halflife", "clip", "scale"]
        ticked = [page.locator(f".tuneset[data-name={n}] input[data-role=tune-tick]").is_checked() for n in names]
        assert ticked == [True, True, True, False, True], "clip only acts through its product with scale, so it starts unticked"
        row = page.locator(".tuneset[data-name=level_noise]")
        assert row.locator("[data-field=now]").input_value() == "0.02" and row.locator("[data-field=from]").input_value() == "0.002" and row.locator("[data-field=to]").input_value() == "0.5"
        assert "Higher" in row.inner_text() and row.locator(".help").is_visible() and not page.locator(".tuneset[data-name=clip] .help").is_visible()
        assert page.locator(".tuneset[data-name=clip] [data-field=from]").count() == 0 and "Keep at" in page.inner_text(".tuneset[data-name=clip]")
        assert page.inner_text("[data-role=tune-summary]").startswith("4 of 5 settings ticked")
        page.click(".tuneset[data-name=clip] input[data-role=tune-tick]")
        assert page.locator(".tuneset[data-name=clip] [data-field=from]").input_value() == "1" and page.inner_text("[data-role=tune-summary]").startswith("5 of 5")

        page.fill(".tuneset[data-name=level_noise] [data-field=from]", "0.9")                                   # the lowest value above the highest
        assert "the lowest value to try must be below the highest" in page.inner_text("#tune .runbar .guard") and page.locator("#tune .runbar button.btn.primary").is_disabled()
        page.fill(".tuneset[data-name=level_noise] [data-field=from]", "0.002")
        for name in names:
            page.uncheck(f".tuneset[data-name={name}] input[data-role=tune-tick]")
        assert page.inner_text("#tune .runbar .guard") == "Tick at least one setting to tune." and page.locator("#tune .runbar button.btn.primary").is_disabled()
        page.click("#tune button:has-text('Back to the suggested settings')")
        page.wait_for_selector(".tuneset[data-name=level_noise]")
        assert page.locator(".tuneset[data-name=level_noise] input[data-role=tune-tick]").is_checked() and page.locator("#tune .runbar .guard").is_hidden()

        page.select_option("[data-role=tune-strategy]", "momentum_13612w")                                       # a strategy that compares tickers, with one setting
        page.wait_for_selector(".tuneset[data-name=month]")
        page.select_option("[data-role=tune-strategy]", "basis_momentum")                                        # nothing to tune
        page.wait_for_selector("#tune .runbar .guard:not([hidden])")
        assert "no numeric or yes/no settings" in page.inner_text("#tune .runbar .guard") and page.locator(".tuneset").count() == 0
        assert not page.problems, page.problems
    finally:
        page.close()


def test_a_search_runs_the_result_reads_in_words_and_the_winner_goes_to_the_backtest_tab(browser, live_tune):  # noqa: F811
    page = _page(browser, live_tune)
    try:
        _open_tune(page)
        _small_search(page)
        page.wait_for_selector("[data-role=tune-progress]")
        assert "Looking for good settings" in page.inner_text("#tune-results") and page.locator("button[data-action=stop-tune]").count() == 1
        page.wait_for_selector(".verdict-card", timeout=240_000)
        card = page.locator(".verdict-card")
        assert card.get_attribute("data-level") in ("good", "mixed", "bad", "keep") and card.locator("h2").inner_text() and card.locator(".vicon").inner_text() in ("✓", "!", "✗", "=")
        reasons = card.locator("li").all_inner_texts()
        assert any("untouched" in r or "nothing tried did better" in r.lower() for r in reasons) and any("tries" in r and "luck" in r for r in reasons)
        text = page.inner_text("#tune-results")
        for heading in ("Best settings for kalman_trend", "The settings", "On the days the search could see", "On the days it never saw: the honest test", "Growth of 1: tuned against starting settings",
                        "Every try, in order", "The best tries", "How much to trust it", "Read before relying on it"):
            assert heading in text, heading
        assert "6 settings tried" in text and "level noise" in text and "Sharpe ratio" in text and "Swings a year" in text and "Holding every ticker equally" in text
        assert page.locator("#tune-results .card.compare").count() == 2 and page.locator("#tune-results .card.compare.honest").count() == 1
        assert page.locator("#tune-results svg.chart").count() >= 6, "the growth curve, one chart for each of the four ticked settings, and every try"
        assert page.locator("#tune-results .tile", has_text="Settings tried").count() == 1
        assert page.inner_text("#trials") == "6 trials on this universe", "the tries count in the Backtest tab's trial counter"
        assert not page.locator("#tune-results .runs .card").count() and page.locator("#tune-results .runs li[data-status=done]").count() >= 1

        page.click("#tune-results .chartcard:has-text('Growth of 1') button:has-text('Table')")
        assert "Starting settings" in page.inner_text("#tune-results .chartcard:has-text('Growth of 1') table")
        page.click("#theme")                                                                              # a redraw (theme, width) keeps the result and the runs card
        _wait_until(page, "document.querySelectorAll('#tune-results .verdict-card').length === 1 && document.querySelectorAll('#tune-results .runs li').length >= 1")
        page.click("#theme")

        best = float(page.locator("#tune-results .card:has(h3:text-is('The settings')) tbody tr").first.locator("td").nth(2).inner_text().replace("⚑", "").strip())
        page.click("button[data-action=use-settings]")
        assert page.locator("#tab-backtest").get_attribute("aria-selected") == "true"
        assert page.locator("#s-body select").first.input_value() == "kalman_trend"
        level = float(page.locator("#s-body .params input").first.input_value())
        assert level == pytest.approx(best, rel=0.01), "the Backtest tab has the level noise the search found (the table shows it to three figures)"
        assert page.locator("#p-body select").first.input_value() == "sleeves", "the Backtest tab chose the same allocation the search used"
        assert not page.problems, page.problems
    finally:
        page.close()


def test_the_result_fits_a_phone_and_nothing_scrolls_sideways(browser, live_tune):  # noqa: F811
    _ensure_search(live_tune.app)
    page = _page(browser, live_tune, width=420)
    try:
        _open_tune(page, "kalman_trend")
        page.wait_for_selector("#tune-results .runs li[data-status=done]")
        page.click("#tune-results .runs button[data-action=open-run]")
        page.wait_for_selector(".verdict-card")
        assert page.evaluate("document.documentElement.scrollWidth") <= 420
        for selector in (".verdict-card", "#tune-results .card.compare", "#tune-results svg.chart"):
            box = page.locator(selector).first.bounding_box()
            assert box["x"] >= 0 and box["x"] + box["width"] <= 421, selector
        assert not page.problems, page.problems
    finally:
        page.close()


def test_a_reloaded_page_finds_the_search_again_opens_it_and_can_clear_it(browser, live_tune):  # noqa: F811
    app = live_tune.app
    _ensure_search(app)
    page = _page(browser, live_tune)
    try:
        page.click("#tab-tune")
        page.wait_for_selector("#tune-results .runs li[data-status=done]")
        assert "Recent runs" in page.inner_text("#tune-results .runs") and "kalman_trend on 8 tickers" in page.inner_text("#tune-results .runs li")
        page.click("#tune-results .runs button[data-action=open-run]")
        page.wait_for_selector(".verdict-card")
        assert "Best settings for kalman_trend" in page.inner_text("#tune-results")
        rows = page.locator("#tune-results .runs li").count()
        page.click("#tune-results .runs button[data-action=clear-run]")
        _wait_until(page, f"document.querySelectorAll('#tune-results .runs li').length === {rows - 1}")
        if rows - 1 >= 2:
            page.click("#tune-results .runs button[data-action=clear-ended]")
            _wait_until(page, "document.querySelectorAll('#tune-results .runs li').length === 0")
        assert not [j for j in app.jobs_list()["jobs"] if j["kind"] == "tune" and j["status"] != "done"], "the search was not lost"
        assert not page.problems, page.problems
    finally:
        page.close()


def test_stop_ends_a_search_early_and_the_result_says_so(browser, live_tune, monkeypatch):  # noqa: F811
    monkeypatch.setattr(tune_api, "Pipeline", FakePipeline)
    FakePipeline.runs, FakePipeline.specs, FakePipeline.gate = 0, [], threading.Event()
    page = _page(browser, live_tune)
    try:
        _open_tune(page)
        page.click("[data-role=tune-effort] button[data-value=standard]")
        page.click("#tune .runbar button.btn.primary")
        page.wait_for_selector("button[data-action=stop-tune]")
        end = time.time() + 60
        while FakePipeline.runs < 4 and time.time() < end:
            time.sleep(0.05)
        assert FakePipeline.runs >= 4
        _wait_until(page, "document.querySelector('[data-role=tune-progress]').textContent.includes('try ')", 15.0)           # the page asks again every 0.7 s
        page.click("button[data-action=stop-tune]")
        assert "Stopping after this try" in page.inner_text("button[data-action=stop-tune]") and page.locator("button[data-action=stop-tune]").is_disabled()
        FakePipeline.gate.set()
        page.wait_for_selector(".verdict-card", timeout=60_000)
        assert "You stopped it early" in page.inner_text("#tune-results")
        assert page.locator("#tune-results .tile", has_text="Settings tried").inner_text().split("\n")[1] in {"4", "5"}
        assert not page.problems, page.problems
    finally:
        FakePipeline.gate.set()
        page.close()


def test_an_app_older_than_the_tab_says_so_and_the_other_tabs_still_work(browser, live_tune):  # noqa: F811
    page = _page_with(browser, live_tune, rewrite=lambda catalog: catalog.update(api_version=7))
    try:
        page.wait_for_selector(".chip")
        page.click("#tab-tune")
        page.wait_for_selector("#tune .card.empty")
        assert "needs a newer app" in page.inner_text("#tune") and "version 7" in page.inner_text("#tune") and "needs 8" in page.inner_text("#tune")
        page.click("#tab-backtest")
        assert page.locator("#panel details").count() >= 3
    finally:
        page.close()
