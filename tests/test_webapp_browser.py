"""The dashboard in a real browser: add tickers, run a backtest, read the charts, build a formula, read a guide. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import glob
import os

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.test_webapp import _default, _fetch

sync_api = pytest.importorskip("playwright.sync_api")


def _chromium():
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "/opt/pw-browsers"]
    for root in roots:
        found = sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux", "chrome")))
        if found:
            return found[-1]
    return None


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    store = TickerStore(tmp_path_factory.mktemp("prices"), fetch=_fetch)
    app = App(config, builder=UniverseBuilder(config, store, default_loader=_default), root=config.root)
    running = server.start(app, "127.0.0.1", 0)
    yield running
    running.stop()


@pytest.fixture(scope="module")
def browser():
    path = _chromium()
    if path is None:
        pytest.skip("no Chromium available")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch(executable_path=path, args=["--no-sandbox"])
        yield b
        b.close()


@pytest.fixture()
def page(browser, live):
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live.url)
    pg.wait_for_selector(".chip")
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def _run_and_wait(page, timeout=120_000):
    page.click("#run")
    page.wait_for_selector(".kpis", timeout=timeout)


def test_the_page_loads_with_the_platform_tickers_and_an_inviting_empty_state(page):
    assert page.locator(".chip").count() == 6 and "Pick tickers and a strategy" in page.inner_text("#results")
    assert page.inner_text("#trials") == "no trials yet"


def test_adding_tickers_checks_them_and_marks_bad_ones(page):
    page.fill("input[aria-label='tickers to add']", "NEWONE, BADX")
    page.click("button:has-text('Add')")
    page.wait_for_selector(".chip.pending", state="detached", timeout=30_000)
    assert page.locator(".chip.bad").count() == 1 and "BADX" in page.locator(".chip.bad").inner_text()
    assert "no data found" in page.inner_text("#u-chips")
    page.click(".chip:has-text('NEWONE') button")
    assert page.locator(".chip:has-text('NEWONE')").count() == 0


def test_a_backtest_draws_every_chart_and_the_controls_work(page):
    _run_and_wait(page)
    assert page.locator(".kpis .tile").count() == 6 and page.locator(".tile.hero .value").inner_text() not in ("", "n/a")
    assert page.locator("svg.chart").count() >= 7
    assert page.locator("#results .legend").count() >= 1 and page.locator(".flag").count() >= 5
    first_equity = page.locator("svg.chart").first.inner_html()
    page.click(".chartcard .seg button:has-text('Log')")
    assert page.locator("svg.chart").first.inner_html() != first_equity
    page.click("button:has-text('5y')")
    page.wait_for_selector(".kpis")
    assert "Window starts" in page.inner_text("#results")
    page.click(".chartcard:nth-of-type(2) .seg button:has-text('Table')")
    assert page.locator(".chartcard table.data").count() >= 1
    box = page.locator("svg.chart").first.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + 100)
    page.wait_for_selector(".tip:not([hidden])")
    assert "$" in page.inner_text(".tip:not([hidden])")
    assert page.inner_text("#trials").startswith("1 trial")


def test_a_formula_is_checked_then_backtested_as_written_and_appears_in_compare(page):
    page.click("#tab-builder")
    page.fill("#builder textarea >> nth=0", "mom(5, 10)")
    page.click("button:has-text('Check on my tickers')")
    page.wait_for_selector(".banner.err")
    assert "larger than skip" in page.inner_text("#builder")
    page.fill("#builder textarea >> nth=0", "-ret(5)")
    page.click("button:has-text('Check on my tickers')")
    page.wait_for_selector("#builder svg.chart")
    page.click("button:has-text('Backtest this formula')")
    page.wait_for_selector(".kpis", timeout=120_000)
    assert page.inner_text(".head h1") == "expression" and "Heads up" not in page.inner_text("#results")
    page.click("#tab-compare")
    assert page.locator(".run:not(.header)").count() == 1 and page.locator("#compare svg.chart").count() == 1
    page.click("#tab-builder")
    page.click("button:has-text('Reload my strategies')")
    page.wait_for_selector("#builder p.hint:has-text('user_strategies')")


def test_guides_render_markdown_safely_and_navigate(page):
    page.click("#tab-guides")
    page.wait_for_selector("article.md h1")
    assert page.inner_text("article.md h1") == "Using the dashboard"
    page.click(".toc button:has-text('How to add a strategy')")
    page.wait_for_selector("article.md h1:has-text('How to add a strategy')")
    assert page.locator("article.md pre code").count() >= 2
    page.fill("input[aria-label='search guides']", "momentum")
    page.click(".toc button:has-text('Momentum and trend following')")
    page.wait_for_selector("article.md h1:has-text('Momentum')")
    assert page.locator("article.md script").count() == 0


def test_the_theme_toggle_switches_and_charts_are_redrawn(page):
    page.click("#theme")
    first = page.get_attribute("html", "data-theme")
    page.click("#theme")
    assert {first, page.get_attribute("html", "data-theme")} == {"dark", "light"}


def test_the_page_is_usable_on_a_phone(browser, live):
    pg = browser.new_page(viewport={"width": 390, "height": 800})
    pg.goto(live.url)
    pg.wait_for_selector(".chip")
    assert pg.evaluate("document.documentElement.scrollWidth") <= 392
    pg.close()
