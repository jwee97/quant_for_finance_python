"""Finzdata in the Free data card of the Futures tab, in a real browser: the steps that turn it on when there is no key, finding a market, listing every contract of it, downloading through a fake
service, the stored history and how long it has before a window of 30 days passes it, and the curve listing every contract. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assets.free_data.http import Http
from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.finzdata import FinzdataClient
from src.webapp.universe import TickerStore, UniverseBuilder
from tests import free_data_helpers as fh
from tests.fake_finzdata import KEY, FakeFinzdata
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _wait_until

pytest.importorskip("playwright.sync_api")

NOW = pd.Timestamp("2026-10-29")


def _serve(tmp_path_factory, key):
    config = load_config()
    root = tmp_path_factory.mktemp("finz_root")
    (root / "data").mkdir()
    (root / "data" / "raw").symlink_to(config.root / "data" / "raw", target_is_directory=True)
    (root / "docs").symlink_to(config.root / "docs", target_is_directory=True)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("finz_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    app.data.http = Http(fh.FakeFeeds({}), sleep=lambda s: None, pause=0.0, retries=0)
    fake = FakeFinzdata(today=str(NOW.date()))
    app.data.finz.client = FinzdataClient(key=key, root=root, base_url="https://api.finzdata.com", transport=fake, sleep=lambda s: None, pause=0.0)
    app.data.finz.today = lambda: NOW
    running = server.start(app, "127.0.0.1", 0)
    running.app, running.fake = app, fake
    return running


@pytest.fixture()
def live_finz(tmp_path_factory, monkeypatch):
    monkeypatch.delenv("FINZDATA_API_KEY", raising=False)
    running = _serve(tmp_path_factory, KEY)
    yield running
    running.stop()


@pytest.fixture()
def live_nokey(tmp_path_factory, monkeypatch):
    monkeypatch.delenv("FINZDATA_API_KEY", raising=False)
    running = _serve(tmp_path_factory, "")
    yield running
    running.stop()


def _open(browser, live):  # noqa: F811
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live.url)
    pg.wait_for_selector(".chip")
    pg.click("#tab-futures")
    pg.wait_for_selector("#futures details.freedata button[data-action=download]", timeout=90_000)
    pg.click("#futures [data-role=source] button[data-value=finzdata]")
    return pg


def _finish(page):
    page.click("#futures details.freedata button[data-action=download]")
    page.wait_for_selector("#futures details.freedata .fdres", timeout=90_000)
    _wait_until(page, "!document.querySelector('#futures details.freedata .fdlive .progress')")


def test_without_a_key_the_card_gives_the_steps_to_add_the_secret_and_shows_no_finder(browser, live_nokey):  # noqa: F811
    page = _open(browser, live_nokey)
    page.wait_for_selector("#futures [data-role=finzdata-box] .setup")
    guide = page.inner_text("#futures [data-role=finzdata-box]")
    assert "Finzdata is not set up yet" in guide and "FINZDATA_API_KEY" in guide and "finzdata.com" in guide and "Copy the secret name" in guide
    assert page.locator("#futures .finder").count() == 0 and "iTick" not in guide
    assert [b.inner_text() for b in page.locator("#futures [data-role=source] button").all()] == ["Yahoo: curves", "Finzdata: every contract", "Yahoo: search any future", "iTick: any product"]
    assert not page.problems, page.problems
    page.close()


def test_a_market_is_found_its_contracts_listed_downloaded_and_kept_and_the_curve_shows_all_of_them(browser, live_finz):  # noqa: F811
    page = _open(browser, live_finz)
    card = page.locator("#futures details.freedata")
    page.wait_for_selector("#futures .finder .hits button.hit")
    text = card.inner_text()
    assert "Your Finzdata key is set" in text and "0 of the 900 calls" in text and "Nothing is stored from Finzdata yet" in text and "last 30 days" in text and "forbid publishing" in text
    assert "ES" in page.locator("#futures .finder .chosen button").first.inner_text() and "markets people look for most, of the 139" in text

    page.fill("#futures [data-role=finder-search]", "crude")
    _wait_until(page, "(document.querySelector('#futures .finder .hits button.hit b') || {textContent: ''}).textContent.indexOf('CL') >= 0")
    first = page.locator("#futures .finder .hits button.hit").first.inner_text()
    assert "WTI Crude Oil" in first and "NYMEX" in first and "Energy" in first and "since 2008" in first
    page.click("#futures .finder .hits button[data-value='CL']")
    assert [b.get_attribute("data-value") for b in page.locator("#futures .finder .chosen button").all()] == ["ES", "CL"]

    page.click("#futures [data-action=list-contracts]")                                              # the last market chosen: CL
    page.wait_for_selector("#futures .finzcontracts table.data tbody tr", timeout=30_000)
    summary = page.inner_text("#futures .finzcontracts .hint")
    assert summary.startswith("CL · WTI Crude Oil: 72 contracts (26 still trading, 46 expired); 0 stored here.") and "Fetched just now" in summary      # the name is the bundled list's: the service's list was not asked for
    rows = page.locator("#futures .finzcontracts table.data tbody tr")
    assert rows.count() == 72 and "CLF23" in rows.first.inner_text() and "expired" in rows.first.inner_text() and "live" in page.inner_text("#futures .finzcontracts")

    _finish(page)
    done = page.locator("#futures .fdres li:not(.fail) b").all_inner_texts()
    assert [t.replace("✓", "").strip() for t in done] == ["ES", "CL"]
    listing = page.inner_text("#futures details.freedata .fdlist")
    assert "Finzdata" in listing and "with open interest" in listing
    page.wait_for_selector("#futures [data-role=finzdata-box] table.data")
    box = page.inner_text("#futures [data-role=finzdata-box]")
    assert "stored from finzdata" in box.lower() and "ES · E-mini S&P 500" in box and "CL · WTI Crude Oil" in box and "2026-09-29 to 2026-10-29" in box and "30 days" in box and "none" in box

    page.select_option("#view-futures select[data-key=root]", "CL")                                  # the curve of the crude oil file, nearest twelve by default
    _wait_until(page, "document.querySelector('#futures-results').innerText.includes('Downloaded: free data') && document.querySelector('#futures-results').innerText.includes('(CL)')")
    hint = page.inner_text("#futures-results")
    assert "lists the nearest 12 of the" in hint and "Choose \"every contract the file has\"" in hint
    page.select_option("#view-futures select[data-key=contracts]", "0")
    _wait_until(page, "!document.querySelector('#futures-results').innerText.includes('lists the nearest 12 of the')")
    page.locator("#futures-results .chartcard", has_text="Forward curve").locator(".seg button", has_text="Table").click()
    table = page.locator("#futures-results .chartcard", has_text="Forward curve").locator("table.data tbody tr")
    assert table.count() > 12 and "M" in table.first.inner_text()
    assert not page.problems, page.problems
    page.close()


def test_the_stored_markets_are_chosen_again_with_one_button_and_a_refusal_is_shown_not_hidden(browser, live_finz):  # noqa: F811
    page = _open(browser, live_finz)
    page.wait_for_selector("#futures .finder .hits button.hit")
    _finish(page)                                                                                       # ES, chosen to begin with
    page.reload()
    page.wait_for_selector("body[data-ready]")                                                          # the page opens on the tab it was left on, so the Backtest chips are not there to wait for
    page.click("#tab-futures")
    page.wait_for_selector("#futures details.freedata", timeout=90_000)
    page.evaluate("document.querySelectorAll('#futures details.freedata').forEach(d => d.open = true)")      # closed once there is something downloaded
    page.wait_for_selector("#futures details.freedata button[data-action=download]")
    page.click("#futures [data-role=source] button[data-value=finzdata]")
    page.click("#futures .finder .chosen button")                                                       # take ES off the choice
    assert "Nothing chosen yet" in page.inner_text("#futures .finder .chosen")
    page.click("#futures [data-action=choose-stored]")
    page.wait_for_selector("#futures .finder .chosen button[data-value='ES']")
    live_finz.fake.quota = 0                                                                            # the day's allowance is over
    page.click("#futures details.freedata button[data-action=download]")
    page.wait_for_selector("#futures details.freedata .banner.err", timeout=60_000)
    banner = page.inner_text("#futures details.freedata .banner.err")
    assert "Not downloaded" in banner and "limit of calls for the day" in banner
    assert not page.problems, page.problems
    page.close()
