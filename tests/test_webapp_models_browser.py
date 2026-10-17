"""The Options tab's three model views in a real browser: the fit (a job that ends in a ranking, the smiles and implied densities beside the quotes, and the parameters), the model prices (each product under
every model, the exercise boundary of an American option, a barrier the form refuses) and the hedging cost; every control named, every chart with a table behind it, and each view fitting a phone. Skipped when
Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from tests.test_webapp_browser import browser  # noqa: F401  (``browser`` is the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _mode, _page, _run, _start, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")

FAST = ("merton", "heston")                                                                    # the two quickest fits that differ most: jumps and stochastic volatility


@pytest.fixture(scope="module")
def live_models(tmp_path_factory):
    running = _start(tmp_path_factory, "models")
    yield running
    running.stop()


@pytest.fixture()
def mp(browser, live_models):  # noqa: F811
    pg = _page(browser, live_models)
    pg.click("#tab-options")
    pg.wait_for_selector("#options-results .tile", timeout=90_000)
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def _choose_models(page, wanted=FAST):
    """Tick exactly ``wanted`` in the model list of the open form."""
    for box in page.query_selector_all("#options .checks input[type=checkbox]"):
        want = box.get_attribute("data-value") in wanted
        if box.is_checked() != want:
            box.set_checked(want)


def _done(page, timeout=300_000):
    page.wait_for_selector("#options-results .charts", timeout=timeout)
    _wait_until(page, "!document.querySelector('#options .runbar .spinner')", 60)


def _titles(page):
    return [t.strip() for t in page.eval_on_selector_all("#options-results .card h3, #options-results .chartcard h3", "els => els.map(e => e.textContent)")]


def test_the_fit_ranks_the_models_draws_the_smiles_and_says_what_it_fitted(mp):
    _mode(mp, "options", "fit")
    mp.wait_for_selector("#options-results .card.empty")
    assert "Fit the pricing models" in mp.inner_text("#options-results")
    assert sorted(b.get_attribute("data-value") for b in mp.query_selector_all("#options .checks input[type=checkbox]")) == ["heston", "kou", "merton", "nig", "vg"]
    _choose_models(mp)
    _run(mp, "options")
    _done(mp)
    text = mp.inner_text("#options-results")
    assert mp.locator("#options-results .prov .src").first.inner_text() in {"Simulated: a model", "Mixed sources"} and "fitted to another model" in text, "a fit to a simulated chain must say that it is a model of a model"
    assert "Best model" in text and "Flat volatility" in text and "Heston" in text and "Merton" in text
    assert mp.locator("#options-results svg.chart").count() >= 4 and mp.locator("#options-results .banner.err").count() == 0
    titles = " | ".join(_titles(mp))
    assert "How far each model is from the quotes" in titles and "What the quotes imply for the price" in titles and "The chance of a fall at least this big" in titles and "The fitted parameters" in titles
    pills = mp.locator("#options-results .filters .tabsmall button")
    assert pills.count() >= 2 and mp.locator("#options-results .filters .tabsmall button[aria-pressed=true]").count() == 1
    before = mp.inner_text("#options-results")
    mp.locator("#options-results .filters .tabsmall button[aria-pressed=false]").first.click()
    _wait_until(mp, "document.querySelectorAll('#options-results .filters .tabsmall button[aria-pressed=true]').length === 1 && document.querySelector('#options-results .chartcard h3') !== null", 30)
    assert mp.inner_text("#options-results") != before, "another expiry draws another smile"
    assert mp.locator("#options-results .chartcard").count() >= 4 and "Read the guide" in text


def test_every_chart_of_the_fit_has_a_table_behind_it_and_every_control_a_name(mp):
    _mode(mp, "options", "fit")
    _choose_models(mp, ("heston",))
    _run(mp, "options")
    _done(mp)
    cards = mp.locator("#options-results .chartcard")
    assert cards.count() >= 4
    for i in range(cards.count()):
        assert cards.nth(i).locator("button").count() >= 2, i                                    # the Chart and Table switch of the card
    unnamed = mp.evaluate("""[...document.querySelectorAll('#options input:not([type=hidden]), #options select, #options textarea')].filter((e) => e.offsetParent !== null)
        .filter((e) => !(e.getAttribute('aria-label') || e.getAttribute('aria-labelledby') || e.title || (e.labels && e.labels.length))).map((e) => e.tagName + ':' + (e.dataset.key || e.id || e.className))""")
    assert unnamed == []
    levels = mp.evaluate("[...document.querySelectorAll('#view-options h1,#view-options h2,#view-options h3,#view-options h4')].filter((e) => e.offsetParent !== null).map((e) => Number(e.tagName[1]))")
    assert all(b - a <= 1 for a, b in zip(levels, levels[1:])), levels


def test_the_model_prices_show_the_spread_between_models_and_an_american_option_its_boundary(mp):
    _mode(mp, "options", "price")
    mp.wait_for_selector("#options-results .card.empty")
    mp.select_option("#options select[data-key=pProduct]", "american")
    mp.wait_for_selector("#options select[data-key=pRight]")
    assert mp.input_value("#options select[data-key=pRight]") == "put", "choosing an American option switches to the put it is usually asked for"
    _choose_models(mp)
    _run(mp, "options")
    _done(mp)
    text = mp.inner_text("#options-results")
    assert "Black-Scholes" in text and "Range across models" in text and "Early-exercise premium" in text and "Where early exercise starts paying" in text
    assert "second state variable" in text, "Heston cannot price early exercise here and the table says why, rather than leaving a blank"
    assert mp.locator("#options-results .banner.err").count() == 0 and mp.locator("#options-results svg.chart").count() >= 2
    mp.select_option("#options select[data-key=pProduct]", "digital")
    _run(mp, "options")
    _wait_until(mp, "document.querySelector('#options-results h2') && document.querySelector('#options-results h2').textContent.includes('Digital')", 120)
    _done(mp)
    assert "How each model differs from Black-Scholes by strike" in " | ".join(_titles(mp)) and mp.locator("#options-results table").count() >= 1


def test_a_barrier_on_the_wrong_side_of_the_price_is_not_sent_and_a_kind_brings_its_usual_level(mp):
    _mode(mp, "options", "price")
    mp.select_option("#options select[data-key=pProduct]", "barrier")
    mp.wait_for_selector("#options input[data-key=pBarrier]")
    assert mp.input_value("#options input[data-key=pBarrier]") == "90"
    mp.fill("#options input[data-key=pBarrier]", "105")
    assert mp.locator("#options .runbar button.primary").is_disabled() and "A down barrier must be below the price" in mp.inner_text("#options .runbar .guard")
    mp.select_option("#options select[data-key=pBarrierKind]", "up-out")
    assert mp.input_value("#options input[data-key=pBarrier]") == "110", "a barrier kind brings its usual level"
    assert mp.locator("#options .runbar button.primary").is_enabled()
    _choose_models(mp, ("merton",))
    _run(mp, "options")
    _done(mp)
    assert "monitored" in mp.inner_text("#options-results") and "Range across models" in mp.inner_text("#options-results")


def test_the_hedging_view_prices_the_cost_and_draws_the_band_and_the_outcomes(mp):
    _mode(mp, "options", "hedge")
    mp.wait_for_selector("#options-results .card.empty")
    mp.fill("#options input[data-key=hPaths]", "2000")
    mp.fill("#options input[data-key=hCost]", "10")
    _run(mp, "options")
    _done(mp)
    text = mp.inner_text("#options-results")
    for phrase in ("Black-Scholes price", "Seller's price with costs", "No-trade band", "Utility-based ask and bid", "What rebalancing more often costs", "The no-trade band around the delta", "Profit and loss of each hedging rule"):
        assert phrase in text, phrase
    assert mp.locator("#options-results svg.chart").count() >= 3 and mp.locator("#options-results .banner.err").count() == 0
    assert mp.locator("#options-results table").count() >= 2, "the three rules and the utility-based prices are tables"
    mp.fill("#options input[data-key=hCost]", "0")
    _run(mp, "options")
    _wait_until(mp, "document.querySelector('#options-results').innerText.includes('Leland number 0.00')", 120)


@pytest.mark.parametrize("mode", ["fit", "price", "hedge"])
def test_the_model_views_fit_a_phone(browser, live_models, mode):  # noqa: F811
    pg = browser.new_page(viewport={"width": 390, "height": 800})
    try:
        pg.goto(live_models.url)
        pg.wait_for_selector(".chip")
        pg.click("#tab-options")
        pg.wait_for_selector("#options-results .tile", timeout=90_000)
        _mode(pg, "options", mode)
        pg.wait_for_selector("#options-results .card.empty")
        if mode == "fit":
            _choose_models(pg, ("heston",))
        elif mode == "hedge":
            pg.fill("#options input[data-key=hPaths]", "1000")
        else:
            _choose_models(pg, ("merton",))
        _run(pg, "options")
        _done(pg)
        assert pg.evaluate("document.documentElement.scrollWidth") <= 392, mode
        assert pg.evaluate("[...document.querySelectorAll('#options-results svg.chart')].every((s) => s.getBoundingClientRect().right <= 392)"), mode
    finally:
        pg.close()
