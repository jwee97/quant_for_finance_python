"""iTick in the Free data cards of the Futures, Crypto and Options tabs, in a real browser: choosing the source, searching the product list, downloading through a fake iTick service, the History view of a
futures product, the steps that turn iTick on when there is no key, and a phone. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.assets.free_data import yahoo_futures as yf
from src.assets.free_data.http import Http
from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.providers import ITickProvider, RateLimiter
from src.webapp.universe import TickerStore, UniverseBuilder, itick_source
from tests.fake_itick import KEY, FakeClock, FakeITick
from tests import free_data_helpers as fh
from tests.free_data_helpers import FakeFeeds, cboe_payload
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_data import CBOE, YAHOO, _crypto_routes, yahoo_route
from tests.test_webapp_workspaces_browser import _mode, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")


def _serve(tmp_path_factory, key):
    config = load_config()
    root = tmp_path_factory.mktemp("itick_root")
    (root / "data").mkdir()
    (root / "data" / "raw").symlink_to(config.root / "data" / "raw", target_is_directory=True)
    (root / "docs").symlink_to(config.root / "docs", target_is_directory=True)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    clock, fake = FakeClock(), FakeITick(stamp="evening_prior", bars=1600)
    provider = ITickProvider(key=key, root=root, base_url="https://api-free.itick.org", limiter=RateLimiter(5, clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep)
    sources = {"itick": itick_source(root, provider, root / "itick")}
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("itick_prices"), fetch=_fetch), default_loader=lambda c: bundle, sources=sources), root=root)
    feeds = FakeFeeds({CBOE + "SPY.json": cboe_payload("SPY", spot=500.0, vol=0.2), YAHOO + "CL": yahoo_route(90.0, days=120), **_crypto_routes("SOL")})
    app.data.http = Http(feeds, sleep=lambda s: None, pause=0.0, retries=0)
    running = server.start(app, "127.0.0.1", 0)
    running.app, running.fake, running.feeds = app, fake, feeds
    return running


@pytest.fixture(scope="module")
def live_itick(tmp_path_factory):
    running = _serve(tmp_path_factory, KEY)
    yield running
    running.stop()


@pytest.fixture(scope="module")
def live_nokey(tmp_path_factory, monkeypatch_module):
    running = _serve(tmp_path_factory, None)
    yield running
    running.stop()


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.delenv("ITICK_API_KEY", raising=False)
    yield mp
    mp.undo()


def _page(browser, live, width=1400, height=1000):  # noqa: F811
    pg = browser.new_page(viewport={"width": width, "height": height})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live.url)
    pg.wait_for_selector(".chip")
    return pg


@pytest.fixture()
def itick(browser, live_itick):  # noqa: F811
    pg = _page(browser, live_itick)
    pg.live = live_itick
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def _source(page, tab, value):
    page.click(f"#{tab} [data-role=source] button[data-value={value}]")


def _finish(page, tab):
    page.click(f"#{tab} details.freedata button[data-action=download]")
    page.wait_for_selector(f"#{tab} details.freedata .fdres", timeout=90_000)
    _wait_until(page, f"!document.querySelector('#{tab} details.freedata .fdlive .progress')")


def _first_hit(page, tab):
    return page.locator(f"#{tab} .finder .hits button.hit b").first.inner_text().replace("✓", "").strip()


def test_the_futures_tab_searches_itick_downloads_products_and_reads_their_history(itick):
    page = itick
    page.click("#tab-futures")
    page.wait_for_selector("#futures-results .tile", timeout=90_000)
    card = page.locator("#futures details.freedata")
    page.wait_for_selector("#futures details.freedata button[data-action=download]")
    assert card.evaluate("d => d.open") and card.locator("[data-role=source] button[aria-pressed=true]").inner_text().startswith("Yahoo")
    groups = card.locator(".grouplabel").all_text_contents()                                   # inner_text would give the upper case the style draws
    assert groups[:3] == ["Energy", "Metals", "Grains"] and "Crypto" in groups[-1], "the 44 roots are grouped, not one long row"
    assert card.locator(".picks button").count() >= 46 and card.locator("[data-role=finder-search]").count() == 0

    _source(page, "futures", "itick")
    page.wait_for_selector("#futures .finder .hits button.hit")
    assert card.locator("[data-role=source] button[aria-pressed=true]").inner_text().startswith("iTick") and card.locator(".picks button").count() == 1, "only the chosen product is left of the chips"
    assert "price history to read in the History view" in card.inner_text() and "Your iTick key is set" in card.inner_text(), "the key is there, so no setup steps"
    assert _first_hit(page, "futures") == "US:ES" and "products people look for" in card.inner_text(), "with nothing typed the list is the products people look for"
    assert "US:NQ" in page.locator("#futures .finder .chosen button").inner_text(), "one product is chosen to begin with"

    page.fill("#futures [data-role=finder-search]", "crude")
    _wait_until(page, "(document.querySelector('#futures .finder .hits button.hit b') || {textContent: ''}).textContent.indexOf('US:CL') >= 0")
    page.click("#futures .finder .hits button[data-value='US:CL']")
    page.select_option("#futures [data-role=finder-region]", "HK")
    page.fill("#futures [data-role=finder-search]", "hang seng")
    _wait_until(page, "(document.querySelector('#futures .finder .hits button.hit b') || {textContent: ''}).textContent.indexOf('HK:HSI') >= 0")
    assert page.locator("#futures .finder .hits button.hit").count() >= 3 and all(t.strip().startswith("HK:") for t in page.locator("#futures .finder .hits button.hit b").all_inner_texts())
    page.click("#futures .finder .hits button[data-value='HK:HSI']")
    assert [b.get_attribute("data-value") for b in page.locator("#futures .finder .chosen button").all()] == ["US:NQ", "US:CL", "HK:HSI"]
    page.select_option("#futures [data-role=finder-region]", "")
    page.fill("#futures [data-role=finder-search]", "zzzzqq")
    page.wait_for_selector("#futures .finder button:text('Try US:ZZZZQQ anyway')", timeout=20_000)
    assert "No product in the list matches" in page.inner_text("#futures .finder")
    page.click("#futures .finder .chosen button[data-value='US:CL']")                          # a chosen product is removed by clicking it
    assert [b.get_attribute("data-value") for b in page.locator("#futures .finder .chosen button").all()] == ["US:NQ", "HK:HSI"]
    card.locator("select[data-key=years]").select_option("3")
    assert "takes about" in card.locator("[data-role=estimate]").inner_text()

    _finish(page, "futures")
    ok = page.locator("#futures .fdres li:not(.fail) b").all_inner_texts()
    assert [t.split(" · ")[0].replace("✓", "").strip() for t in ok] == ["US:NQ", "HK:HSI"] and all("price history" in t for t in ok)
    assert page.locator("#futures .fdres li.fail").count() == 0
    # the tab moved to the History view of the first product and says where it came from
    page.wait_for_selector("#futures-results .tile", timeout=60_000)
    assert page.locator("#view-futures .tabsmall button[data-mode=history]").get_attribute("aria-selected") == "true"
    assert page.locator("#futures select[data-key=series]").input_value() == "US_NQ"
    page.wait_for_selector("#futures-results svg.chart", timeout=60_000)
    text = page.inner_text("#futures-results")
    for word in ("price history", "Last price", "Volatility, a year", "Worst fall", "The eight largest one-day moves"):
        assert word in text, word
    notes = page.locator("#futures-results").text_content()                                    # the notes sit in a closed section: text_content reads it, inner_text does not
    assert "no forward curve" in notes and "joined" in notes and "iTick" in notes
    assert "Downloaded: free data" in page.locator("#futures-results .prov .src").first.inner_text() and page.locator("#futures-results .banner.err").count() == 0
    assert page.locator("#futures-results svg.chart").count() >= 3 and page.locator("#futures-results table.data").count() >= 1
    page.wait_for_selector("#futures .fdlist li[data-name=US_NQ]")
    assert "iTick · price history" in page.inner_text("#futures .fdlist li[data-name=US_NQ]") and page.locator("#futures .fdlist ul.dl li").count() == 2
    select = page.locator("#futures select[data-key=series]")
    assert select.locator("option").count() == 2 and "NQ: E-mini Nasdaq-100 Futures (US)" in select.locator("option:checked").inner_text()

    # a second product in the menu, then the curve views know nothing of a price history
    select.select_option("HK_HSI")
    page.wait_for_selector("#futures-results h1:has-text('Hang Seng')", timeout=60_000)
    _mode(page, "futures", "curve")
    page.wait_for_selector("#futures select[data-key=root]", timeout=60_000)
    assert page.locator("#futures select[data-key=root] option[value='US_NQ']").count() == 0 and page.locator("#futures select[data-key=root] option[value='HK_HSI']").count() == 0

    # update is one call (a top-up), delete asks twice and takes the product out of the menu
    _mode(page, "futures", "history")
    before = len(page.live.fake.calls)
    page.click("#futures .fdlist li[data-name=US_NQ] button[data-action=update]")
    _wait_until(page, "!!document.querySelector('#futures details.freedata .fdres')")
    _wait_until(page, "!document.querySelector('#futures details.freedata .fdlive .progress')")
    assert len(page.live.fake.calls) - before == 1
    item = page.locator("#futures .fdlist li[data-name=US_NQ]")
    item.locator("button[data-action=delete]").click()
    item.locator("button[data-action=delete]").click()
    page.wait_for_selector("#futures .fdlist li[data-name=US_NQ]", state="detached", timeout=30_000)
    assert page.locator("#futures select[data-key=series] option[value='US_NQ']").count() == 0
    page.live.app.data.store.delete("futures", "HK_HSI")


def test_with_no_key_the_futures_card_shows_the_steps_and_the_free_source_still_works(browser, live_nokey):  # noqa: F811
    page = _page(browser, live_nokey)
    try:
        page.click("#tab-futures")
        page.wait_for_selector("#futures details.freedata button[data-action=download]", timeout=90_000)
        _source(page, "futures", "itick")
        page.wait_for_selector("#futures .itick .setup", timeout=20_000)
        text = page.inner_text("#futures .itick")
        assert "iTick is not set up yet" in text and "ITICK_API_KEY" in text and "Tools, then Secrets" in text and "Come back to this card" in text
        assert page.locator("#futures .itick button:text('Copy the secret name')").count() == 1 and page.locator("#futures .itick button:text('Test the iTick connection')").count() == 0
        page.click("#futures details.freedata button[data-action=download]")
        page.wait_for_selector("#futures details.freedata .banner.err", timeout=30_000)
        assert "ITICK_API_KEY" in page.inner_text("#futures details.freedata .banner.err") and page.locator("#futures details.freedata .fdres").count() == 0
        assert len(live_nokey.fake.calls) == 0, "nothing was asked of iTick"
        _source(page, "futures", "yahoo")                                                         # the free source needs no key
        page.locator("#futures details.freedata .picks button[data-value=ES]").click()
        page.locator("#futures details.freedata .picks button[data-value=GC]").click()
        _finish(page, "futures")
        assert page.locator("#futures .fdres li:not(.fail)").count() == 1
        assert all("status of 400" in p for p in page.problems), page.problems                  # the browser logs the request the server refused on purpose, and nothing else
    finally:
        page.close()
        live_nokey.app.data.store.delete("futures", "CL")


def test_the_crypto_tab_downloads_a_coins_daily_history_from_itick(itick):
    page = itick
    page.click("#tab-crypto")
    page.wait_for_selector("#crypto-results .tile", timeout=90_000)
    card = page.locator("#crypto details.freedata")
    page.wait_for_selector("#crypto details.freedata button[data-action=download]")
    assert card.locator("input[data-value]").count() == 3, "the exchanges' card has the three things to download"
    _source(page, "crypto", "itick")
    assert card.locator("input[data-value]").count() == 0 and "spot only" in card.inner_text() and "Your iTick key is set" in card.inner_text()
    card.locator("input[data-key=extra]").fill("doge notacoin")
    card.locator("select[data-key=days]").select_option("260")
    _finish(page, "crypto")
    ok, bad = page.locator("#crypto .fdres li:not(.fail)"), page.locator("#crypto .fdres li.fail")
    assert [t.replace("✓", "").strip() for t in page.locator("#crypto .fdres li:not(.fail) b").all_inner_texts()] == ["SOL · daily history", "DOGE · daily history"]
    assert ok.count() == 2 and bad.count() == 1 and "NOTACOIN" in bad.first.inner_text() and "does not have NOTACOIN" in bad.first.inner_text()
    page.wait_for_selector("#crypto .fdlist li[data-name=SOL]", timeout=30_000)
    assert "iTick (USDT)" in page.inner_text("#crypto .fdlist li[data-name=SOL]")
    select = page.locator("#crypto select[data-key=symbol]")
    assert select.input_value() == "SOL" and "downloaded" in select.locator("option:checked").inner_text()
    page.wait_for_selector("#crypto-results .prov .src", timeout=60_000)
    assert "iTick" in page.locator("#crypto-results").text_content() and page.locator("#crypto-results .banner.err").count() == 0
    for name in ("SOL", "DOGE"):
        page.live.app.data.store.delete("crypto", name)


def test_the_options_tab_can_also_download_the_underlyings_prices_and_then_shows_a_realised_volatility(itick):
    page = itick
    page.click("#tab-options")
    page.wait_for_selector("#options-results .tile", timeout=90_000)
    card = page.locator("#options details.freedata")
    page.wait_for_selector("#options details.freedata button[data-action=download]")
    assert "iTick has no option data" in card.inner_text() and card.locator("select[data-key=pricesSource]").count() == 0
    card.locator("input[data-key=prices]").check()
    assert card.locator("select[data-key=pricesSource]").input_value() == "yahoo" and card.locator("select[data-key=pricesYears]").input_value() == "3"
    card.locator("select[data-key=pricesSource]").select_option("itick")
    assert "Your iTick key is set" in card.inner_text()
    _finish(page, "options")
    labels = [t.replace("✓", "").strip() for t in page.locator("#options .fdres li:not(.fail) b").all_inner_texts()]
    assert labels == ["SPY", "SPY · underlying prices"] and page.locator("#options .fdres li.fail").count() == 0
    page.wait_for_selector("#options .fdlist li[data-name=SPY_prices]")
    assert "iTick" in page.inner_text("#options .fdlist li[data-name=SPY_prices]") and "daily prices" in page.inner_text("#options .fdlist li[data-name=SPY_prices]")
    assert page.locator("#options select[data-key=underlying]").input_value() == "dl:SPY" and page.locator("#options select[data-key=underlying] option[value='dl:SPY_prices']").count() == 0, "a price file is not an underlying"
    page.wait_for_selector("#options-results .tile", timeout=60_000)
    tile = page.locator("#options-results .tile", has_text="Realised volatility")
    assert "n/a" not in tile.first.inner_text() and "%" in tile.first.inner_text(), "the downloaded prices give the chain a realised volatility"
    assert "daily prices of SPY downloaded from iTick" in page.locator("#options-results").text_content()
    for name in ("SPY", "SPY_prices"):
        page.live.app.data.store.delete("options", name)


def test_the_test_button_on_the_futures_and_crypto_cards_asks_for_that_tabs_own_data_and_names_the_host(itick):
    page = itick
    for tab, asked, path in (("futures", "(futures data)", "/future/kline?region=US&code=ES"), ("crypto", "(crypto data)", "/crypto/kline?region=BA&code=BTCUSDT")):
        page.click(f"#tab-{tab}")
        page.wait_for_selector(f"#{tab}-results .tile", timeout=90_000)
        page.wait_for_selector(f"#{tab} details.freedata button[data-action=download]")
        _source(page, tab, "itick")
        card = page.locator(f"#{tab} details.freedata")
        assert "Requests go to api-free.itick.org" in card.inner_text(), "the card says which iTick host the key goes to"
        before = len(page.live.fake.urls)
        card.locator("button", has_text="Test the iTick connection").click()
        _wait_until(page, f"(document.querySelector('#{tab} [data-role=itick-test-result]') || {{textContent: ''}}).textContent.indexOf('iTick works') === 0")
        text = card.locator("[data-role=itick-test-result]").inner_text()
        assert f"iTick works at api-free.itick.org {asked}" in text and KEY not in text
        assert len(page.live.fake.urls) == before + 1 and path in page.live.fake.urls[-1], "one call, for that tab's own data"


def test_the_finder_marks_the_products_iTick_returned_or_refused_for_the_key(itick):
    page = itick
    provider = page.live.app.builder.sources["itick"].provider
    provider.note_access("future", "US:ES", False, "iTick does not have this symbol for your key (E001 not found this produce)")
    provider.note_access("future", "US:NQ", True)
    try:
        page.click("#tab-futures")
        page.wait_for_selector("#futures-results .tile", timeout=90_000)
        page.wait_for_selector("#futures details.freedata button[data-action=download]")
        _source(page, "futures", "itick")
        page.wait_for_selector("#futures .finder .hits button.hit")
        es = page.locator("#futures .finder .hits button.hit[data-value='US:ES']")
        nq = page.locator("#futures .finder .hits button.hit[data-value='US:NQ']")
        assert es.locator(".access.no").inner_text() == "✗ not for your key" and "E001" in es.locator(".access.no").get_attribute("title")
        assert nq.locator(".access.ok").inner_text() == "✓ worked"
        assert page.locator("#futures .finder .hits button.hit[data-value='US:CL'] .access").count() == 0, "a product never asked for has no mark"
    finally:
        provider._access.clear()


def test_the_futures_card_searches_every_future_yahoo_lists_asks_yahoo_for_a_word_and_reads_the_history(itick):
    page = itick
    routes = page.live.feeds.routes
    routes[YAHOO + "LBR=F"] = yahoo_route(500.0, days=300)
    routes[YAHOO + "OATZ=F"] = yahoo_route(300.0, days=300)
    routes[yf.LOOKUP] = fh.yahoo_lookup([fh.yahoo_future_doc("OATZ=F", "Oat Futures,Dec-2026", "CBT"), fh.yahoo_future_doc("OATZ26.CBT", "Oat Futures,Dec-2026", "CBT")])
    page.click("#tab-futures")
    page.wait_for_selector("#futures-results .tile", timeout=90_000)
    page.wait_for_selector("#futures details.freedata button[data-action=download]")
    card = page.locator("#futures details.freedata")
    _source(page, "futures", "yahoo_series")
    page.wait_for_selector("#futures .finder .hits button.hit")
    assert card.locator("[data-role=source] button[aria-pressed=true]").inner_text().startswith("Yahoo: search") and "No key is needed" in card.inner_text()
    assert card.locator("[data-role=finder-region]").count() == 0 and card.locator("[data-role=finder-group] option").count() >= 8, "Yahoo's list has groups, not markets"
    assert _first_hit(page, "futures") == "CL=F" and "ES=F" in page.locator("#futures .finder .chosen button").inner_text() and card.locator("[data-role=finder-ask]").is_hidden()
    page.fill("#futures [data-role=finder-search]", "lumber")
    _wait_until(page, "(document.querySelector('#futures .finder .hits button.hit b') || {textContent: ''}).textContent.indexOf('LBR=F') >= 0")
    page.click("#futures .finder .hits button[data-value='LBR=F']")
    page.fill("#futures [data-role=finder-search]", "zzoats")
    page.wait_for_selector("#futures .finder .note, #futures .finder p.hint", timeout=20_000)
    _wait_until(page, "document.querySelectorAll('#futures .finder .hits button.hit').length === 0")
    assert card.locator("[data-role=finder-ask]").is_visible() and "ask yahoo" in page.inner_text("#futures .finder").lower()
    card.locator("[data-role=finder-ask]").click()
    _wait_until(page, "document.querySelectorAll('#futures .finder .hits button.hit').length === 2")
    assert "Yahoo found 2 futures" in page.inner_text("#futures .finder") and "OATZ=F" in page.locator("#futures .finder .hits").inner_text()
    page.click("#futures .finder .hits button[data-value='OATZ=F']")
    assert [b.get_attribute("data-value") for b in page.locator("#futures .finder .chosen button").all()] == ["ES=F", "LBR=F", "OATZ=F"]
    page.click("#futures .finder .chosen button[data-value='ES=F']")                            # ES=F is not asked for: no route, and a chosen product is removed by clicking it
    card.locator("select[data-key=yyears]").select_option("3")
    _finish(page, "futures")
    ok = [t.replace("✓", "").strip() for t in page.locator("#futures .fdres li:not(.fail) b").all_inner_texts()]
    assert ok == ["LBR=F · price history", "OATZ=F · price history"] and page.locator("#futures .fdres li.fail").count() == 0
    page.wait_for_selector("#futures .fdlist li[data-name=YF_LBR]", timeout=30_000)
    assert "Yahoo Finance · price history" in page.inner_text("#futures .fdlist li[data-name=YF_LBR]") and "Lumber" in page.inner_text("#futures .fdlist li[data-name=YF_LBR]")
    assert page.locator("#view-futures .tabsmall button[aria-selected=true]").get_attribute("data-mode") == "history"
    page.wait_for_selector("#futures-results .prov .src", timeout=60_000)
    text = page.locator("#futures-results").text_content()
    assert "Lumber" in text and "rolled by Yahoo" in text and "Downloaded: free data" in text and page.locator("#futures-results .banner.err").count() == 0
    for name in ("YF_LBR", "YF_OATZ"):
        page.live.app.data.store.delete("futures", name)
    page.live.app.data._yahoo_list().unlink(missing_ok=True)


@pytest.mark.parametrize("tab, results", [("futures", "#futures-results .tile"), ("crypto", "#crypto-results .tile"), ("options", "#options-results .tile")])
def test_the_itick_cards_fit_a_phone(browser, live_itick, tab, results):  # noqa: F811
    pg = _page(browser, live_itick, width=390, height=800)
    try:
        pg.click(f"#tab-{tab}")
        pg.wait_for_selector(results, timeout=90_000)
        if tab == "options":
            pg.locator("#options input[data-key=prices]").check()
            pg.locator("#options select[data-key=pricesSource]").select_option("itick")
        else:
            _source(pg, tab, "itick")
        if tab == "futures":
            pg.wait_for_selector("#futures .finder .hits button.hit")
            pg.fill("#futures [data-role=finder-search]", "micro")
            _wait_until(pg, "(document.querySelector('#futures .finder .hits button.hit b') || {textContent: ''}).textContent.length > 0")
            pg.click("#futures .finder .hits button.hit >> nth=0")
        assert pg.evaluate("document.documentElement.scrollWidth") <= 392, tab
        assert pg.evaluate("document.querySelector('#" + tab + " details.freedata').getBoundingClientRect().right") <= 392
        assert not pg.problems, pg.problems
    finally:
        pg.close()
