"""The dashboard in a real browser: add tickers, run a backtest, read the charts, build a formula, read a guide. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import glob
import os

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.providers import ITickProvider, RateLimiter
from src.webapp.universe import TickerStore, UniverseBuilder, itick_source
from tests.fake_itick import KEY, FakeClock, FakeITick
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


def _app_with_itick(tmp, key):
    config = load_config()
    fake, clock = FakeITick(), FakeClock()
    provider = ITickProvider(key=key, base_url="https://api-free.itick.org", limiter=RateLimiter(5, clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep)
    builder = UniverseBuilder(config, TickerStore(tmp / "yahoo", fetch=_fetch), default_loader=_default, sources={"itick": itick_source(tmp, provider, tmp / "itick")})
    return App(config, builder=builder, root=config.root), fake


@pytest.fixture(scope="module")
def live_itick(tmp_path_factory):
    app, fake = _app_with_itick(tmp_path_factory.mktemp("with_key"), KEY)
    running = server.start(app, "127.0.0.1", 0)
    running.fake = fake
    yield running
    running.stop()


@pytest.fixture(scope="module")
def live_no_key(tmp_path_factory):
    app, fake = _app_with_itick(tmp_path_factory.mktemp("without_key"), "")
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


# ------------------------------------------------------------------------------------------------- an app that is older than the page
def _page_with(browser, live, rewrite=None, missing=None):
    """The dashboard opened with the server's answers changed on the way: ``rewrite(catalog)`` edits the catalogue (to stand in for an app that is older than the page, or one that needs a
    restart) and ``missing`` is a path the server pretends not to know (HTTP 404, as an older app answers a request it has never heard of)."""
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    if rewrite:
        def edit(route):
            response = route.fetch()
            data = response.json()
            rewrite(data)
            route.fulfill(response=response, json=data)

        pg.route("**/api/catalog", edit)
    if missing:
        pg.route(f"**{missing}", lambda route: route.fulfill(status=404, json={"error": "not found"}))
    pg.goto(live.url)
    return pg


def test_an_app_older_than_the_page_says_so_instead_of_leaving_empty_lists(browser, live):
    """The case that happened on Replit: the files were updated while the app kept running, so the new page met a catalogue without data sources."""
    def old(catalog):
        for key in ("sources", "default_source", "api_version", "restart_needed"):
            catalog.pop(key, None)

    pg = _page_with(browser, live, rewrite=old)
    try:
        pg.wait_for_selector("#notice .banner.err")
        text = pg.inner_text("#notice")
        assert "The app is older than this page" in text and "Stop the app and press Run again" in text and "reports version 1" in text
        assert pg.locator(".layout").is_hidden() and pg.locator("#run").count() == 0
        assert pg.locator("#notice button:has-text('Reload this page')").count() == 1
        assert not pg.problems, pg.problems
    finally:
        pg.close()


def test_the_first_release_with_data_sources_sent_no_version_and_is_still_accepted(browser, live_itick):
    pg = _page_with(browser, live_itick, rewrite=lambda catalog: catalog.pop("api_version", None))
    try:
        pg.wait_for_selector(".chip")
        assert pg.locator("#notice .banner").count() == 0 and pg.locator("#u-source-select option").count() == 2 and pg.locator("#run").is_enabled()
        assert not pg.problems, pg.problems
    finally:
        pg.close()


def test_a_restart_notice_is_shown_when_the_code_changed_after_the_app_started_and_the_page_still_works(browser, live_itick):
    pg = _page_with(browser, live_itick, rewrite=lambda catalog: catalog.update(restart_needed=True))
    try:
        pg.wait_for_selector(".chip")
        assert "updated after it started" in pg.inner_text("#notice") and pg.locator("#notice .banner.err").count() == 0
        assert pg.locator("#u-source-select option").count() == 2 and pg.locator("#s-body select").first.locator("option").count() > 10 and pg.locator("#run").is_enabled()
        assert not pg.problems, pg.problems
    finally:
        pg.close()


def test_a_part_that_cannot_be_drawn_is_named_and_the_rest_of_the_page_still_works(browser, live):
    pg = _page_with(browser, live, rewrite=lambda catalog: catalog.update(sources=[None]))         # the data source menu cannot make sense of this
    try:
        pg.wait_for_selector("#notice .banner.err")
        assert "the data source menu" in pg.inner_text("#notice") and "Reload the page" in pg.inner_text("#notice")
        assert pg.locator("#s-body select").first.locator("option").count() > 10 and pg.locator("#p-body select").count() >= 1 and pg.locator(".chip").count() == 6
        assert pg.locator("#run").is_enabled()
        assert not pg.problems, pg.problems
    finally:
        pg.close()


def test_a_missing_ticker_count_check_is_explained_and_not_reported_as_not_found(browser, live):
    pg = _page_with(browser, live, missing="/api/requirements")
    try:
        pg.wait_for_selector("#guard:not([hidden])")
        guard = pg.inner_text("#guard")
        assert "older than the page" in guard and "Stop the app and press Run again" in guard and guard.strip().lower() != "not found"
        assert pg.locator("#run").is_disabled()
    finally:
        pg.close()


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
    assert abs(_scroll_y(page) - y) < 250, "the reader stays near where they were (a narrower page re-flows the rows of tiles above, so not to the pixel; the bug sent them to 0)"
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


# ------------------------------------------------------------------------------------------------------ one ticker, earnings, trades, data sources
def _only_tickers(page, text):
    page.click("button:text-is('Clear')")
    page.fill("input[aria-label='tickers to add']", text)
    page.click("button:text-is('Add')")
    page.wait_for_selector(".chip.pending", state="detached", timeout=30_000)


def _choose_strategy(page, name, then=None):
    """Pick a strategy; the page then asks the server what the choice needs (after a short pause), so wait for the Run button to settle into the state the test expects."""
    page.locator("#s-body select").first.select_option(name)
    page.wait_for_selector(then, timeout=15_000) if then else page.wait_for_timeout(800)


def test_a_strategy_that_ranks_tickers_cannot_be_run_on_one_ticker_and_one_that_trades_each_ticker_can(page):
    _only_tickers(page, "AAA")
    assert page.locator("#u-count").inner_text() == "1 ticker"
    _choose_strategy(page, "momentum", "#run:disabled")
    assert page.locator("#run").is_disabled() and page.locator("#guard").is_visible()
    assert "ranks the tickers against each other" in page.inner_text("#guard") and "at least 2 tickers and you have 1" in page.inner_text("#guard")
    assert "(needs 2+ tickers)" in page.locator("#s-body select").first.locator("option[value='momentum']").inner_text()
    page.fill("input[aria-label='tickers to add']", "BBB")
    page.click("button:text-is('Add')")
    page.wait_for_selector("#run:not(:disabled)", timeout=15_000)
    assert not page.locator("#run").is_disabled() and not page.locator("#guard").is_visible()                 # a second ticker is enough for a ranking
    page.click(".chip:has-text('BBB') button")
    page.wait_for_selector("#run:disabled", timeout=15_000)
    assert page.locator("#run").is_disabled()
    _choose_strategy(page, "ma_crossover", "#run:not(:disabled)")
    assert not page.locator("#run").is_disabled() and not page.locator("#guard").is_visible()
    assert "Works on each ticker alone" in page.inner_text("#s-body") and "Trades tab" in page.inner_text("#s-body")
    _run_and_wait(page)
    assert "1 ticker ·" in page.inner_text(".head .sub") and "buy and hold" in page.inner_text(".kpis")
    assert page.locator(".kpis .tile").count() == 6 and page.locator(".earnrow .tile").count() == 6


def test_the_earnings_and_trades_tabs_show_dollars_round_trips_and_buys_and_sells(page):
    _only_tickers(page, "AAA")
    _choose_strategy(page, "ma_crossover")
    _run_and_wait(page)
    tabs = page.locator(".tabsmall button")
    assert tabs.nth(0).inner_text() == "Earnings" and tabs.nth(1).inner_text() == "Trades" and tabs.nth(0).get_attribute("aria-selected") == "true"
    details = page.locator(".tabsmall").locator("xpath=..")
    earnings = details.inner_text().lower()                                         # headings are shown in capitals by the style sheet
    for text in ("Starting capital", "$100,000", "Net profit", "Profitable months", "Trading costs paid", "Deepest fall from a peak", "Profit by calendar year", "Profit by month"):
        assert text.lower() in earnings, text
    assert details.locator("svg.chart").count() == 1 and details.locator("table.data").count() >= 6
    tabs.nth(1).click()
    page.wait_for_timeout(300)
    trades = details.inner_text().lower()
    for text in ("A round trip is one stretch", "Win rate", "Profit factor", "Average days held", "Buys / sells", "Latest buys and sells", "long", "short"):
        assert text.lower() in trades, text
    assert details.locator("table.data").count() >= 5
    assert "Account value from $100,000" in page.inner_text("#results") and page.locator(".earnrow").inner_text().count("$") >= 4
    page.fill("input[aria-label='tickers to add']", "")


def test_the_starting_capital_changes_the_dollar_figures(page):
    page.locator("#p-body").locator("xpath=ancestor::details").locator("summary").click()
    page.fill("#p-body input[placeholder^='blank = the AUM']", "50000")
    _only_tickers(page, "AAA")
    _choose_strategy(page, "ma_crossover")
    _run_and_wait(page)
    assert "$50,000" in page.inner_text(".earnrow") and "Account value from $50,000" in page.inner_text("#results")
    page.fill("#p-body input[placeholder^='blank = the AUM']", "")


def test_a_wrong_symbol_is_refused_in_the_page_without_spoiling_the_others(page):
    page.click("button:text-is('Clear')")
    page.fill("input[aria-label='tickers to add']", "AAA, NO$PE, BBB")
    page.click("button:text-is('Add')")
    page.wait_for_selector(".chip.pending", state="detached", timeout=30_000)
    assert page.locator(".chip.bad").count() == 1 and "NO$PE" in page.locator(".chip.bad").inner_text()
    assert page.locator(".chip:not(.bad)").count() == 2 and "not a valid ticker" in page.inner_text("#u-chips")


def test_the_source_menu_offers_yahoo_alone_when_the_app_has_no_other_source(page):
    assert page.locator("#u-source-select option").all_inner_texts() == ["Yahoo Finance"]
    assert "No key needed" in page.inner_text("#u-source")


@pytest.fixture()
def page_itick(browser, live_itick):
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live_itick.url)
    pg.wait_for_selector(".chip")
    yield pg
    assert not pg.problems, pg.problems
    assert KEY not in pg.content() and KEY not in pg.evaluate("JSON.stringify(Object.assign({}, localStorage))")        # the key is nowhere in the page, not even in what it remembers
    pg.close()


def test_choosing_itick_downloads_when_you_run_waits_for_the_limit_and_never_shows_the_key(page_itick, live_itick):
    pg, fake = page_itick, live_itick.fake
    assert pg.locator("#u-source-select option").all_inner_texts() == ["Yahoo Finance", "iTick"]
    pg.select_option("#u-source-select", "itick")
    assert "5 calls a minute" in pg.inner_text("#u-source") and "ITICK_API_KEY" in pg.inner_text("#u-source")
    assert pg.locator("#u-source .setup").count() == 0 and pg.locator("#u-source button:has-text('Test the iTick connection')").count() == 1      # a key exists: no set-up guide, a test button
    pg.click("button:text-is('Clear')")
    pg.fill("input[aria-label='tickers to add']", "NVDA, MSFT")
    pg.click("button:text-is('Add')")
    pg.wait_for_selector(".chip.pending", state="detached", timeout=30_000)
    assert fake.calls == [], "adding a ticker must not spend any of the rate-limited calls"
    assert "↓" in pg.inner_text("#u-chips") and "downloaded when you run" in pg.inner_text("#u-chips") and "calls" in pg.inner_text("#u-chips")
    pg.click("text=Test the iTick connection")
    pg.wait_for_selector("#u-source-result:has-text('iTick works')", timeout=30_000)
    assert len(fake.calls) == 1
    _choose_strategy(pg, "ma_crossover")
    _run_and_wait(pg)
    assert "· iTick ·" in pg.inner_text(".head .sub") and "2 tickers" in pg.inner_text(".head .sub")
    pg.wait_for_selector("#u-chips .chip:has-text('✓')")
    assert "↓" not in pg.inner_text("#u-chips") and len(fake.calls) > 4
    assert set(fake.tokens) == {KEY}


@pytest.fixture()
def page_no_key(browser, live_no_key, monkeypatch):
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live_no_key.url)
    pg.wait_for_selector(".chip")
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def test_the_setup_guide_copies_the_secret_name_and_its_reload_button_keeps_iTick_selected(browser, live_no_key, monkeypatch):
    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], viewport={"width": 1400, "height": 1000})
    pg = context.new_page()
    problems = []
    pg.on("pageerror", lambda e: problems.append(str(e)))
    try:
        pg.goto(live_no_key.url)
        pg.wait_for_selector(".chip")
        pg.select_option("#u-source-select", "itick")
        pg.click("#u-source button:has-text('Copy the secret name')")
        pg.wait_for_selector(".toast")
        assert "copied" in pg.inner_text(".toast") and pg.evaluate("navigator.clipboard.readText()") == "ITICK_API_KEY"
        with pg.expect_navigation():
            pg.click("#u-source button:has-text('I added it and restarted: reload')")
        pg.wait_for_selector("#u-source .setup")
        assert pg.locator("#u-source-select").input_value() == "itick"                                               # the choice survives the reload
        assert not problems, problems
    finally:
        context.close()


def test_without_a_key_the_page_says_how_to_set_one_up_and_refuses_the_download(page_no_key, monkeypatch):
    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    pg = page_no_key
    assert pg.locator("#u-source-select option").all_inner_texts() == ["Yahoo Finance", "iTick (not set up)"]
    pg.select_option("#u-source-select", "itick")
    text = pg.inner_text("#u-source")
    assert "ITICK_API_KEY" in text and "Secrets" in text and "never sent to this page" in text
    assert pg.locator("#u-source button:has-text('Test the iTick connection')").count() == 0                       # the test button waits for a key (the steps may still mention it)
    steps = pg.locator("#u-source .setup ol.steps li")
    assert steps.count() == 4 and pg.locator("#u-source .setup code").inner_text() == "ITICK_API_KEY"
    assert "Stop" in steps.nth(2).inner_text() and "Run" in steps.nth(2).inner_text() and "reload" in steps.nth(2).inner_text()
    assert pg.locator("#u-source .setup button").all_inner_texts() == ["Copy the secret name", "I added it and restarted: reload"]
    assert pg.locator("#u-source input").count() == 0 and pg.locator("#u-source textarea").count() == 0       # there is nowhere on the page to type a key
    pg.click("button:text-is('Clear')")
    pg.fill("input[aria-label='tickers to add']", "NVDA")
    pg.click("button:text-is('Add')")
    pg.wait_for_selector(".chip.pending", state="detached", timeout=30_000)
    assert pg.locator(".chip.bad").count() == 1 and "not set up" in pg.inner_text("#u-chips")
    pg.select_option("#u-source-select", "yahoo")                                  # Yahoo does not need it: the same ticker is checked again against the other source
    pg.wait_for_selector(".chip.pending", state="detached", timeout=30_000)
