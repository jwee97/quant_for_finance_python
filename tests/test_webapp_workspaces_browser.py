"""The Options, Futures, Crypto and Machine learning tabs in a real browser: they open on a read-only view with the source of every number printed, their forms refuse what cannot run, the machine-learning
tab trains a model, checks it, explains it and follows it forward, and every view fits a phone. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import time

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_browser import _page_with, browser  # noqa: F401  (``browser`` is the module's Chromium fixture)

sync_api = pytest.importorskip("playwright.sync_api")

SOURCES = {"Platform data", "Your file", "Simulated: a model", "Mixed sources"}


@pytest.fixture(scope="module")
def live_ws(tmp_path_factory):
    """An app on synthetic prices with a planted signal. Its registry lives in a temporary directory; the repository's data (Deribit, your files) is reached through a link, read only."""
    config = load_config()
    root = tmp_path_factory.mktemp("ws_root")
    (root / "data").symlink_to(config.root / "data", target_is_directory=True)
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
