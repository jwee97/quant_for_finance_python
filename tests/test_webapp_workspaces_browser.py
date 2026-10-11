"""The Options, Futures, Crypto and Machine learning tabs in a real browser: they open on a read-only view with the source of every number printed, their forms refuse what cannot run, the machine-learning
tab trains a model, checks it, explains it and follows it forward, and every view fits a phone. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import time

import pytest

from src.assets.free_data.http import Http
from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.free_data_helpers import FakeFeeds, cboe_payload
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_browser import _page_with, browser  # noqa: F401  (``browser`` is the module's Chromium fixture)
from tests.test_webapp_data import CBOE, YAHOO, _crypto_routes, yahoo_route

sync_api = pytest.importorskip("playwright.sync_api")

SOURCES = {"Platform data", "Your file", "Downloaded: free data", "Simulated: a model", "Mixed sources"}


@pytest.fixture(scope="module")
def live_ws(tmp_path_factory):
    """An app on synthetic prices with a planted signal. Its registry lives in a temporary directory; the repository's data (Deribit, your files) and guides are reached through links, read only."""
    config = load_config()
    root = tmp_path_factory.mktemp("ws_root")
    for folder in ("data", "docs"):
        (root / folder).symlink_to(config.root / folder, target_is_directory=True)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    store = TickerStore(tmp_path_factory.mktemp("ws_prices"), fetch=_fetch)
    running = server.start(App(config, builder=UniverseBuilder(config, store, default_loader=lambda c: bundle), root=root), "127.0.0.1", 0)
    yield running
    running.stop()


@pytest.fixture()
def ws(browser, live_ws):  # noqa: F811
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live_ws.url)
    pg.wait_for_selector(".chip")
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def _mode(page, tab, mode):
    page.click(f"#view-{tab} .tabsmall button[data-mode={mode}]")


def _open_all(page, tab):
    page.evaluate(f"document.querySelectorAll('#{tab} details').forEach(d => d.open = true)")


def _run(page, tab):
    page.click(f"#{tab} .runbar button.primary")


def _wait_until(page, expression, timeout=60.0):
    """Poll a JavaScript expression until it is true. (``wait_for_function`` evaluates a string, which the page's content security policy refuses.)"""
    end = time.time() + timeout
    while time.time() < end:
        if page.evaluate(expression):
            return
        time.sleep(0.2)
    raise AssertionError(f"still false after {timeout:.0f} s: {expression}")


# ------------------------------------------------------------------------------------------------------ options, futures and crypto
@pytest.mark.parametrize("tab, ready, expected", [("options", "#options-results .tile", {"Simulated: a model", "Mixed sources"}), ("futures", "#futures-results .tile", {"Simulated: a model", "Mixed sources"}),
                                                  ("crypto", "#crypto-results .tile", {"Mixed sources"})])
def test_each_asset_tab_opens_on_a_read_only_view_that_says_where_its_numbers_come_from(ws, tab, ready, expected):
    ws.click(f"#tab-{tab}")
    ws.wait_for_selector(ready, timeout=90_000)
    badge = ws.locator(f"#{tab}-results .prov .src").first.inner_text()
    assert badge in SOURCES and badge in expected, "the repository holds no option quotes or futures prices, so those pages must not claim to be data"
    assert ws.locator(f"#{tab}-results .banner.err").count() == 0 and "Read the guide" in ws.inner_text(f"#{tab}-results")
    assert "Simulated" in ws.inner_text(f"#{tab}-results"), "the simulated part is named in words, not only badged"


def test_the_option_chain_shows_both_sides_and_the_strategy_builder_draws_a_payoff(ws):
    ws.click("#tab-options")
    ws.wait_for_selector("#options-results .tile", timeout=90_000)
    _mode(ws, "options", "chain")
    ws.wait_for_selector("#options-results table.chain tbody tr", timeout=60_000)
    assert ws.locator("#options-results table.chain tbody tr").count() > 8 and ws.locator("#options-results table.chain tbody tr.atm").count() == 1
    assert "CALLS" in ws.inner_text("#options-results table.chain thead").upper() and "PUTS" in ws.inner_text("#options-results table.chain thead").upper()
    _mode(ws, "options", "strategy")
    _run(ws, "options")
    ws.wait_for_selector("#options-results .charts", timeout=60_000)
    assert "Break-even" in ws.inner_text("#options-results") and ws.locator("#options-results svg.chart").count() >= 1 and ws.locator("#options-results .banner.err").count() == 0


def test_the_futures_margin_view_prices_an_example_book_and_warns(ws):
    ws.click("#tab-futures")
    ws.wait_for_selector("#futures-results .tile", timeout=90_000)
    _mode(ws, "futures", "margin")
    ws.click("#futures button:has-text('Load an example')")
    ws.wait_for_selector("#futures .leg")
    _run(ws, "futures")
    ws.wait_for_selector("#futures-results .charts", timeout=60_000)
    assert "Margin used" in ws.inner_text("#futures-results") and ws.locator("#futures-results .banner.err").count() == 0


def test_the_crypto_views_keep_real_and_simulated_apart_and_count_what_survives_the_fees(ws):
    ws.click("#tab-crypto")
    ws.wait_for_selector("#crypto-results .tile", timeout=90_000)
    _mode(ws, "crypto", "funding")
    ws.wait_for_selector("#crypto-results table.data", timeout=60_000)
    text = ws.inner_text("#crypto-results")
    assert "DERIBIT" in text and "Simulated: a model" in text and ws.locator("#crypto-results svg.chart").count() >= 1
    _mode(ws, "crypto", "cross")
    ws.wait_for_selector("#crypto-results .earnrow", timeout=60_000)
    assert "Pairs that pay" in ws.inner_text("#crypto-results") and "illustrative" in ws.inner_text("#crypto-results")


@pytest.mark.parametrize("tab, ready, title", [("options", "#options-results .tile", "Reading an option chain"), ("futures", "#futures-results .tile", "Reading a futures curve"), ("crypto", "#crypto-results .tile", "Crypto funding, basis"),
                                               ("ml", "#ml-results .card", "The machine-learning research pipeline")])
def test_each_new_tab_opens_its_own_guide(ws, tab, ready, title):
    ws.click(f"#tab-{tab}")
    ws.wait_for_selector(ready, timeout=90_000)
    ws.click(f"#{tab}-results button:text-is('Read the guide')")
    ws.wait_for_selector("article.md h1", timeout=30_000)
    assert title in ws.inner_text("article.md h1")


# ------------------------------------------------------------------------------------------------------ free data
@pytest.fixture(scope="module")
def live_free(tmp_path_factory):
    """An app whose downloads go to a temporary folder and whose network is a table of answers in the shape of the live feeds, so that no test reaches the internet. The repository's prices and Deribit
    data are linked in, read only."""
    config = load_config()
    root = tmp_path_factory.mktemp("free_root")
    (root / "data").mkdir()
    (root / "data" / "raw").symlink_to(config.root / "data" / "raw", target_is_directory=True)
    (root / "docs").symlink_to(config.root / "docs", target_is_directory=True)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    store = TickerStore(tmp_path_factory.mktemp("free_prices"), fetch=_fetch)
    app = App(config, builder=UniverseBuilder(config, store, default_loader=lambda c: bundle), root=root)
    feeds = FakeFeeds({CBOE + "SPY.json": cboe_payload("SPY", spot=500.0, vol=0.2), YAHOO + "CL": yahoo_route(90.0, days=120), **_crypto_routes("SOL")})
    app.data.http = Http(feeds, sleep=lambda s: None, pause=0.0, retries=0)
    app.data.yfinance = type("OfflineYahoo", (), {"Ticker": staticmethod(lambda symbol: (_ for _ in ()).throw(RuntimeError("offline")))})
    running = server.start(app, "127.0.0.1", 0)
    running.app, running.feeds = app, feeds
    yield running
    running.stop()


@pytest.fixture()
def free(browser, live_free):  # noqa: F811
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live_free.url)
    pg.wait_for_selector(".chip")
    pg.live = live_free
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def _badge(page, tab):
    page.wait_for_selector(f"#{tab}-results .prov .src.downloaded, #{tab}-results .prov .src.mixed, #{tab}-results .prov .src.simulated", timeout=90_000)
    return page.locator(f"#{tab}-results .prov .src").first.inner_text()


def _download(page, tab):
    """Press the card's button and wait until the progress line is gone and the result is shown."""
    page.click(f"#{tab} details.freedata button[data-action=download]")
    page.wait_for_selector(f"#{tab} details.freedata .fdres", timeout=90_000)
    _wait_until(page, f"!document.querySelector('#{tab} details.freedata .fdlive .progress')")


def test_the_options_tab_downloads_a_real_chain_analyses_it_as_downloaded_data_and_forgets_it_when_asked(free):
    page = free
    page.click("#tab-options")
    page.wait_for_selector("#options-results .tile", timeout=90_000)
    card = page.locator("#options details.freedata")
    page.wait_for_selector("#options details.freedata button[data-action=download]")
    assert card.count() == 1 and card.evaluate("d => d.open"), "with nothing downloaded the card is open: the way to real data is the first thing seen"
    assert "No account and no key" in card.inner_text() and "Nothing is downloaded yet" in card.inner_text()
    assert page.locator("#options select[data-key=underlying] option[value^='dl:']").count() == 0
    assert _badge(page, "options") in {"Simulated: a model", "Mixed sources"}
    card.locator("input[data-key=extra]").fill("zzzz")                                          # a ticker without listed options, to see a failure explained
    _download(page, "options")
    ok, bad = page.locator("#options .fdres li:not(.fail)"), page.locator("#options .fdres li.fail")
    assert ok.count() == 1 and "SPY" in ok.first.inner_text() and bad.count() == 1 and "ZZZZ" in bad.first.inner_text() and "no option chain" in bad.first.inner_text()
    page.wait_for_selector("#options .fdlist li[data-name=SPY]")
    assert "Cboe delayed quotes" in page.inner_text("#options .fdlist li[data-name=SPY]") and "downloaded just now" in page.inner_text("#options .fdlist li[data-name=SPY]")
    select = page.locator("#options select[data-key=underlying]")
    assert select.input_value() == "dl:SPY" and "real chain" in select.locator("option:checked").inner_text(), "the downloaded chain is chosen for you, and the menu says it is real"
    assert _badge(page, "options") in {"Downloaded: free data", "Mixed sources"}
    assert "Cboe delayed quotes" in page.locator("#options-results").text_content() and page.locator("#options-results .banner.err").count() == 0
    saved = page.live.app.data.store.path("options", "SPY")
    assert saved.is_file() and "downloads" in saved.parts and not (page.live.app.root / "data" / "user" / "options").exists(), "the file is saved where the page says, not in the folder for your own files"
    # update asks the feed again; delete asks twice and takes the chain out of the menu
    before = len(page.live.feeds.calls)
    page.click("#options .fdlist li[data-name=SPY] button[data-action=update]")
    _wait_until(page, "!!document.querySelector('#options details.freedata .fdres')")
    _wait_until(page, "!document.querySelector('#options details.freedata .fdlive .progress')")
    assert len(page.live.feeds.calls) > before
    item = page.locator("#options .fdlist li[data-name=SPY]")
    item.locator("button[data-action=delete]").click()
    assert "Really delete?" in item.inner_text()
    item.locator("button[data-action=delete]").click()
    page.wait_for_selector("#options .fdlist li[data-name=SPY]", state="detached", timeout=30_000)
    assert page.locator("#options select[data-key=underlying] option[value^='dl:']").count() == 0 and "Nothing is downloaded yet" in page.inner_text("#options details.freedata")
    assert _badge(page, "options") in {"Simulated: a model", "Mixed sources"} and page.locator("#options .fdres").count() == 0


def test_the_futures_tab_downloads_a_curve_chooses_it_and_says_what_the_feed_lacks(free):
    page = free
    page.click("#tab-futures")
    page.wait_for_selector("#futures-results .tile", timeout=90_000)
    card = page.locator("#futures details.freedata")
    page.wait_for_selector("#futures details.freedata button[data-action=download]")
    assert card.evaluate("d => d.open") and card.locator(".picks button[aria-pressed=true]").count() == 3, "three roots are ticked to begin with"
    for root in ("ES", "GC"):                                                                   # the table of answers has Yahoo's crude oil only
        card.locator(f".picks button[data-value={root}]").click()
        assert card.locator(f".picks button[data-value={root}]").get_attribute("aria-pressed") == "false"
    card.locator("select[data-key=contracts]").select_option("4")
    assert "takes about" in card.locator("[data-role=estimate]").inner_text()
    _download(page, "futures")
    assert page.locator("#futures .fdres li:not(.fail)").count() == 1 and page.locator("#futures .fdres li.fail").count() == 0
    page.wait_for_selector("#futures .fdlist li[data-name=CL]")
    select = page.locator("#futures select[data-key=root]")
    assert select.input_value() == "CL" and "(downloaded)" in select.locator("option:checked").inner_text()
    assert _badge(page, "futures") in {"Downloaded: free data", "Mixed sources"}
    text = page.locator("#futures-results").text_content()
    assert "Yahoo Finance" in text and "computed from the exchange" in text and page.locator("#futures-results .banner.err").count() == 0
    assert page.locator("#futures-results svg.chart").count() >= 1


def test_the_crypto_tab_downloads_a_coin_and_reads_its_real_books_and_funding(free):
    page = free
    page.click("#tab-crypto")
    page.wait_for_selector("#crypto-results .tile", timeout=90_000)
    card = page.locator("#crypto details.freedata")
    page.wait_for_selector("#crypto details.freedata button[data-action=download]")
    assert card.evaluate("d => d.open") and card.locator("input[data-value]").count() == 3 and card.locator("input[data-value]:checked").count() == 3
    card.locator("select[data-key=days]").select_option("260")
    _download(page, "crypto")
    assert page.locator("#crypto .fdres li:not(.fail)").count() == 3 and page.locator("#crypto .fdres li.fail").count() == 0
    assert [t.strip() for t in page.locator("#crypto .fdres li b").all_inner_texts()] == ["✓ SOL · daily history", "✓ SOL · order books", "✓ SOL · OKX funding"]
    page.wait_for_selector("#crypto .fdlist li", timeout=30_000)
    assert page.locator("#crypto .fdlist ul.dl li").count() == 3
    select = page.locator("#crypto select[data-key=symbol]")
    assert select.input_value() == "SOL" and "downloaded" in select.locator("option:checked").inner_text()
    assert _badge(page, "crypto") in {"Downloaded: free data", "Mixed sources"}
    assert "downloaded" in page.locator("#crypto-results .earnrow").first.inner_text().lower() or page.locator("#crypto-results .earnrow .tile").count() >= 5
    _mode(page, "crypto", "cross")
    page.wait_for_selector("#crypto-results table.data", timeout=60_000)
    assert "Real quotes at one moment" in page.inner_text("#crypto-results") and "spot with spot" in page.inner_text("#crypto-results")
    _mode(page, "crypto", "liquidity")
    page.wait_for_selector("#crypto-results svg.chart", timeout=60_000)
    text = page.inner_text("#crypto-results")
    assert "Real order books taken" in text and "OKX is quoted in USDT" in text and "n/a" in text, "books too thin for $1 million say n/a instead of a made-up cost"
    _mode(page, "crypto", "funding")
    page.wait_for_selector("#crypto-results table.data", timeout=60_000)
    assert "downloaded" in page.locator("#crypto-results").text_content().lower()


def test_an_app_without_the_data_routes_explains_the_card_instead_of_leaving_it_empty(browser, live_ws):  # noqa: F811
    pg = _page_with(browser, live_ws, missing="/api/data/status")
    try:
        pg.wait_for_selector(".chip")
        pg.click("#tab-options")
        pg.wait_for_selector("#options details.freedata", timeout=90_000)
        pg.wait_for_selector("#options details.freedata :text('does not offer free data downloads')", timeout=30_000)
        assert "Stop the app and press Run again" in pg.inner_text("#options details.freedata")
        pg.click("#tab-backtest")
        assert pg.locator(".chip").count() >= 1
    finally:
        pg.close()


def test_an_app_one_version_behind_keeps_the_machine_learning_tab_and_asks_the_asset_tabs_for_a_restart(browser, live_ws):  # noqa: F811
    """The Free data cards arrived with API version 5: an app at 4 still serves the Machine learning tab, but the three asset tabs need the newer one."""
    pg = _page_with(browser, live_ws, rewrite=lambda catalog: catalog.update(api_version=4))
    try:
        pg.wait_for_selector(".chip")
        for tab in ("options", "futures", "crypto"):
            pg.click(f"#tab-{tab}")
            pg.wait_for_selector(f"#{tab} .card.empty")
            assert "needs a newer app" in pg.inner_text(f"#{tab}") and "this tab needs 5" in pg.inner_text(f"#{tab}")
        pg.click("#tab-ml")
        pg.wait_for_selector("#ml .panel")
        assert "needs a newer app" not in pg.inner_text("#ml") and not pg.problems, pg.problems
    finally:
        pg.close()


# ------------------------------------------------------------------------------------------------------ machine learning
def test_a_form_that_cannot_run_is_not_sent(ws):
    ws.click("#tab-ml")
    ws.wait_for_selector("#ml .panel")
    _mode(ws, "ml", "train")
    _open_all(ws, "ml")
    boxes = ws.locator("#ml details:has(summary:text-is('3. Data')) input[data-value]")
    assert boxes.count() >= 7 and ws.locator("#ml .runbar button.primary").is_enabled()
    for i in range(boxes.count()):
        boxes.nth(i).uncheck()
    assert ws.locator("#ml .runbar button.primary").is_disabled() and "at least one group of features" in ws.inner_text("#ml .runbar .guard")
    boxes.nth(0).check()
    assert ws.locator("#ml .runbar button.primary").is_enabled()
    ws.fill("#ml input[data-key=valFrac]", "40")
    ws.fill("#ml input[data-key=testFrac]", "40")
    assert ws.locator("#ml .runbar button.primary").is_disabled() and "at least 30%" in ws.inner_text("#ml .runbar .guard")
    _mode(ws, "ml", "compare")
    assert ws.locator("#ml .runbar button.primary").is_disabled() and "between two and eight" in ws.inner_text("#ml .runbar .guard")


def test_the_machine_learning_tab_trains_a_model_checks_it_explains_it_and_follows_it_forward(ws):
    ws.click("#tab-ml")
    ws.wait_for_selector("#ml-results h2:text-is('No experiments yet')", timeout=60_000)        # a fresh registry says so and offers the next step
    ws.click("#ml-results button:text-is('Train a model')")
    ws.wait_for_selector("#ml .runbar button.primary:has-text('Train and backtest')")
    _open_all(ws, "ml")
    ws.fill("#ml input[data-key=name]", "browser ridge")
    ws.fill("#ml input[data-key=minTrain]", "500")
    ws.fill("#ml input[data-key=refit]", "63")
    _run(ws, "ml")
    ws.wait_for_selector("#ml-results .charts", timeout=240_000)
    out = ws.inner_text("#ml-results")
    assert "browser ridge" in out and "Train (in-sample)" in out and "Generalization" in out and "Deflated Sharpe" in out
    equity = ws.locator("#ml-results .chartcard", has=ws.locator("h3:text-is('Equity curve by segment')"))
    assert all(label in equity.text_content() for label in ("Train (in-sample)", "Validation", "Test", "Forward")), "the four sections of the equity curve are named on the chart"
    assert ws.locator("#ml-results .earnrow").first.locator(".tile").count() == 4 and "What the model leans on" in out
    # the leaderboard has it, with the number of looks at the data
    _mode(ws, "ml", "experiments")
    ws.wait_for_selector("#ml-results table.lb tbody tr", timeout=60_000)
    assert ws.locator("#ml-results table.lb tbody tr").count() == 1 and "browser ridge" in ws.inner_text("#ml-results table.lb")
    # overfitting: stress tests run on the saved predictions and the page moves to the stress-tested copy
    _mode(ws, "ml", "overfit")
    ws.wait_for_selector("#ml-results button:text-is('Run the stress tests')", timeout=60_000)
    assert "Generalization" in ws.inner_text("#ml-results") and "Meta-label" in ws.inner_text("#ml-results")
    ws.click("#ml-results button:text-is('Run the stress tests')")
    ws.wait_for_selector("#ml-results table.data .verdict", timeout=240_000)
    assert "child of" in ws.inner_text("#ml-results") and ws.locator("#ml-results .verdict.pass, #ml-results .verdict.warn, #ml-results .verdict.fail").count() >= 5
    ws.wait_for_selector("#ml select[data-key=exp]", timeout=60_000)
    _wait_until(ws, "(() => { const s = document.querySelector('#ml select[data-key=exp]'); return !!s && s.options[s.selectedIndex].text.includes('rebacktest'); })()")      # the form follows the experiment on show
    # explain and forward test offer only an experiment with a model of its own
    _mode(ws, "ml", "explain")
    ws.wait_for_selector("#ml select[data-key=exp]", timeout=60_000)
    options = ws.locator("#ml select[data-key=exp] option")
    assert options.count() == 1 and "rebacktest" not in options.first.inner_text() and "browser ridge" in options.first.inner_text()
    _run(ws, "ml")
    ws.wait_for_selector("#ml-results h3:text-is('From the average score to this score')", timeout=120_000)
    assert "Shapley" in ws.inner_text("#ml-results")
    _mode(ws, "ml", "forward")
    ws.wait_for_selector("#ml-results .earnrow", timeout=60_000)
    _run(ws, "ml")
    ws.wait_for_selector("#ml-results h3:has-text('Signal for')", timeout=120_000)
    assert "Paper only" in ws.inner_text("#ml-results") and "Recorded signals" in ws.inner_text("#ml-results")
    # two experiments can be compared; a child can be deleted after a second click
    _mode(ws, "ml", "experiments")
    ws.wait_for_selector("#ml-results table.lb tbody tr", timeout=60_000)
    assert ws.locator("#ml-results table.lb tbody tr").count() == 2 and "child of" in ws.inner_text("#ml-results table.lb")
    assert ws.locator("#ml-results button:text-is('Compare the selected')").is_disabled()
    for box in ws.locator("#ml-results table.lb tbody input[type=checkbox]").all():
        box.check()
    ws.click("#ml-results button:text-is('Compare the selected')")
    ws.wait_for_selector("#ml-results h1:has-text('Comparing 2 experiments')", timeout=60_000)
    assert ws.locator("#ml-results svg.chart").count() >= 1 and "Settings and results" in ws.inner_text("#ml-results")
    _mode(ws, "ml", "experiments")
    ws.wait_for_selector("#ml-results table.lb tbody tr")
    child = ws.locator("#ml-results table.lb tbody tr", has_text="child of")
    child.locator("button:text-is('Delete')").click()
    assert "Click again" in child.inner_text()
    child.locator("button").nth(1).click()
    ws.wait_for_selector("#ml-results table.lb tbody tr:nth-child(2)", state="detached", timeout=30_000)
    assert ws.locator("#ml-results table.lb tbody tr").count() == 1


def test_a_request_the_server_refuses_is_explained_on_the_page(ws):
    ws.click("#tab-ml")
    ws.wait_for_selector("#ml .panel")
    _mode(ws, "ml", "train")
    _open_all(ws, "ml")
    ws.fill("#ml input[data-key=minTrain]", "3000")
    _run(ws, "ml")
    ws.wait_for_selector("#ml-results .banner.err", timeout=60_000)
    assert "warm-up" in ws.inner_text("#ml-results") or "train" in ws.inner_text("#ml-results").lower()
    ws.problems[:] = [p for p in ws.problems if "status of 400" not in p]


# ------------------------------------------------------------------------------------------------------ versions and phones
def test_an_app_that_predates_the_new_tabs_says_so_on_each_of_them(browser, live_ws):  # noqa: F811
    pg = _page_with(browser, live_ws, rewrite=lambda catalog: catalog.update(api_version=3))
    try:
        pg.wait_for_selector(".chip")
        for tab in ("options", "futures", "crypto", "ml"):
            pg.click(f"#tab-{tab}")
            pg.wait_for_selector(f"#{tab} .card.empty")
            assert "needs a newer app" in pg.inner_text(f"#{tab}") and "Stop the app and press Run again" in pg.inner_text(f"#{tab}")
        pg.click("#tab-backtest")
        assert pg.locator(".chip").count() >= 1, "the Backtest tab keeps working with an app that is one version behind"
        assert not pg.problems, pg.problems
    finally:
        pg.close()


@pytest.mark.parametrize("tab, ready", [("options", "#options-results .tile"), ("futures", "#futures-results .tile"), ("crypto", "#crypto-results .tile"), ("ml", "#ml-results .card")])
def test_the_new_tabs_fit_a_phone(browser, live_ws, tab, ready):  # noqa: F811
    pg = browser.new_page(viewport={"width": 390, "height": 800})
    try:
        pg.goto(live_ws.url)
        pg.wait_for_selector(".chip")
        pg.click(f"#tab-{tab}")
        pg.wait_for_selector(ready, timeout=90_000)
        assert pg.evaluate("document.documentElement.scrollWidth") <= 392
        modes = [b.get_attribute("data-mode") for b in pg.query_selector_all(f"#view-{tab} .tabsmall button")]
        assert len(modes) >= 5
        for mode in modes:
            _mode(pg, tab, mode)
            pg.wait_for_selector(f"#{tab}-results .card, #{tab}-results .tile, #{tab}-results .banner", timeout=90_000)
            assert pg.evaluate("document.documentElement.scrollWidth") <= 392, mode
    finally:
        pg.close()
