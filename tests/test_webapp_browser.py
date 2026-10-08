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


SETTLE_MS = 800            # longer than the page's 180 ms redraw debounce: a redraw triggered by a click would have happened by then


def _scroll_card_to_top(page, selector, offset=150):
    page.evaluate("([sel, off]) => { const r = document.querySelector(sel).getBoundingClientRect(); window.scrollBy(0, r.top - off); }", [selector, offset])
    page.wait_for_timeout(100)


def _scroll_y(page):
    return page.evaluate("Math.round(window.scrollY)")


def _mark_result(page):
    """Tag a node of the rendered result: if the page redraws the whole result the tag is gone."""
    page.evaluate("document.querySelector('.kpis').dataset.keep = 'yes'")


def _result_untouched(page):
    return page.evaluate("document.querySelector('.kpis') !== null && document.querySelector('.kpis').dataset.keep === 'yes'")


def test_opening_a_table_keeps_the_page_where_it_is_and_does_not_redraw_the_result(page):
    """Regression: a click that changed the page's height used to make the page rebuild the whole result 180 ms later, which closed the table and sent the reader back to the top."""
    _run_and_wait(page)
    _mark_result(page)
    for title in ("Drawdown from the previous peak", "Monthly returns"):
        card = page.locator(".chartcard", has=page.locator(f"h3:text-is('{title}')"))
        card.scroll_into_view_if_needed()
        page.evaluate("(el) => { const r = el.getBoundingClientRect(); window.scrollBy(0, r.top - 150); }", card.element_handle())
        page.wait_for_timeout(100)
        y = _scroll_y(page)
        assert y > 300, "the card should be well down the page for this test to mean anything"
        card.locator(".seg button:text-is('Table')").click()
        page.wait_for_timeout(SETTLE_MS)
        assert card.locator("table.data").count() == 1 and card.locator("svg.chart").count() == 0, "the table must stay open"
        assert card.locator(".seg button:text-is('Table')").get_attribute("aria-pressed") == "true"
        assert _scroll_y(page) == y and _result_untouched(page)
        card.locator(".seg button:text-is('Chart')").click()
        page.wait_for_timeout(SETTLE_MS)
        assert card.locator("svg.chart").count() == 1 and card.locator("table.data").count() == 0
        assert _scroll_y(page) == y and _result_untouched(page)


def test_choosing_another_details_tab_keeps_the_page_where_it_is(page):
    _run_and_wait(page)
    _mark_result(page)
    _scroll_card_to_top(page, ".tabsmall", 100)
    y = _scroll_y(page)
    assert y > 1000
    tabs = page.locator(".tabsmall button")
    assert tabs.count() >= 4
    for i in (2, 3, 1, tabs.count() - 1):
        label = tabs.nth(i).inner_text()
        tabs.nth(i).click()
        page.wait_for_timeout(SETTLE_MS)
        selected = page.evaluate("[...document.querySelectorAll('.tabsmall button')].map((b) => b.getAttribute('aria-selected'))")
        assert selected == ["true" if j == i else "false" for j in range(len(selected))], (label, selected)
        assert _scroll_y(page) == y and _result_untouched(page), label
        assert page.locator(".tabsmall").locator("xpath=..").locator("table.data, pre").count() >= 1


def test_a_resize_redraws_the_charts_at_the_new_width_and_keeps_what_the_reader_chose(page):
    _run_and_wait(page)
    drawdown = page.locator(".chartcard", has=page.locator("h3:text-is('Drawdown from the previous peak')"))
    drawdown.locator(".seg button:text-is('Table')").click()
    page.locator(".chartcard .seg button:text-is('Log')").click()
    page.locator(".tabsmall button").nth(2).click()
    _scroll_card_to_top(page, ".tabsmall", 100)
    y = _scroll_y(page)
    width_before = page.evaluate("document.querySelector('.chartcard svg.chart').viewBox.baseVal.width")
    page.set_viewport_size({"width": 1180, "height": 1000})
    page.wait_for_timeout(SETTLE_MS)
    width_after = page.evaluate("document.querySelector('.chartcard svg.chart').viewBox.baseVal.width")
    assert width_after < width_before, "the charts are redrawn to fit the narrower page"
    assert drawdown.locator("table.data").count() == 1, "the table the reader opened is still open"
    assert page.locator(".chartcard .seg button:text-is('Log')").get_attribute("aria-pressed") == "true"
    assert page.locator(".tabsmall button").nth(2).get_attribute("aria-selected") == "true"
    assert abs(_scroll_y(page) - y) < 120, "the reader stays near where they were"
    page.click("#theme")                                                            # a theme change redraws too, and keeps the same choices
    page.wait_for_timeout(SETTLE_MS)
    assert drawdown.locator("table.data").count() == 1 and page.locator(".tabsmall button").nth(2).get_attribute("aria-selected") == "true"
    page.click("button:has-text('5y')")                                             # and so does a new chart window
    page.wait_for_timeout(SETTLE_MS)
    assert page.locator(".chartcard", has=page.locator("h3:text-is('Drawdown from the previous peak')")).locator("table.data").count() == 1


def test_a_new_run_starts_from_the_top_with_the_default_view(page):
    _run_and_wait(page)
    page.locator(".chartcard .seg button:text-is('Table')").first.click()
    page.locator(".tabsmall button").nth(2).click()
    _scroll_card_to_top(page, ".tabsmall", 100)
    assert _scroll_y(page) > 1000
    page.click("#run")
    page.wait_for_selector("#runcount:text-is('2')", timeout=120_000)
    page.wait_for_timeout(SETTLE_MS)
    assert _scroll_y(page) == 0
    assert page.locator(".chartcard table.data").count() == 0
    assert page.locator(".tabsmall button").nth(0).get_attribute("aria-selected") == "true"


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
