"""SEC EDGAR in the Fundamentals tab, in a real browser: a card of its own in the Data manager that asks for a contact in a secret and takes none, a search that is free and aimed at the SEC's list of
US-listed companies, a fetch that costs two requests and brings back the whole history, a company page that lets the user choose whose statements to look at when two providers know the company, and a
page that never shows the contact. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.fundamentals.keys import KeyPool
from src.fundamentals.providers.openfilings import OpenFilingsProvider
from src.fundamentals.providers.sec_edgar import SecEdgarProvider
from src.fundamentals.usage import Allowance
from src.fundamentals.workbench import SECRET_NAMES
from src.webapp import server
from tests.test_fund_data import KEY_1, FakeOpenFilings
from tests.test_fund_sec_edgar import CONTACT, FakeSec
from tests.test_webapp_fundamentals_browser import LONG, _app, _open, _page, _tables_have_names
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _wait_until

pytest.importorskip("playwright.sync_api")


def _serve(tmp_path_factory, tag, sec: bool, openfilings: bool):
    saved = pytest.MonkeyPatch()
    for name in SECRET_NAMES:
        saved.delenv(name, raising=False)
    app = _app(tmp_path_factory, tag)
    svc = app.fundamentals.bench("real").service
    fakes = {}
    if openfilings:
        fakes["openfilings"] = FakeOpenFilings({KEY_1: "a"})
        pool = KeyPool("openfilings", [("OPENFILINGS_API_KEY_1", KEY_1)], svc.usage, svc.store.salt(), Allowance("credits", 100, 1000, "test"))
        svc.register("openfilings", OpenFilingsProvider(pool, transport=fakes["openfilings"], sleep=lambda s: None))
    if sec:
        fakes["sec"] = FakeSec()
        svc.register("sec_edgar", SecEdgarProvider(pool=svc.pool("sec_edgar"), transport=fakes["sec"], user_agent=CONTACT, cache_path=tmp_path_factory.mktemp(f"{tag}_cache") / "tickers.json",
                                                   sleep=lambda s: None))
    asked = []

    def prices(symbols, start):                                                             # whatever is asked, ten days: nothing here goes to the network
        asked.append(list(symbols))
        idx = pd.bdate_range("2020-01-01", periods=10)
        return {sym: pd.DataFrame({"close": 10.0, "adj_close": 9.5, "volume": 100.0}, index=idx) for sym in symbols}

    app.fundamentals.fetch = prices
    running = server.start(app, "127.0.0.1", 0)
    running.app, running.fakes, running.asked = app, fakes, asked
    return running, saved


@pytest.fixture()
def live_sec(tmp_path_factory):
    """A real database that can be filled from the SEC (a stand-in for its servers) and from OpenFilings (a stand-in for its API)."""
    running, saved = _serve(tmp_path_factory, "sec", sec=True, openfilings=True)
    yield running
    running.stop()
    saved.undo()


@pytest.fixture()
def live_unset(tmp_path_factory):
    """A real database with nothing set up: no key and no contact."""
    running, saved = _serve(tmp_path_factory, "unset", sec=False, openfilings=False)
    yield running
    running.stop()
    saved.undo()


def _provider(page, name):
    page.select_option("select[data-key=provider]", name)


def _search_the_directory(page, text):
    """Type a name, press Enter (that searches the database, which is free) and then ask the provider's directory. The page does one thing at a time: a click that arrives while the database search is
    still running gets the toast "Still working" and sends nothing (seen on a loaded machine, where the button in the results card is not locked as the one in the form is), so the click waits until the
    form's own buttons are unlocked, which is when the first search has answered."""
    page.fill("input[data-key=q]", text)
    page.press("input[data-key=q]", "Enter")
    page.wait_for_selector("#fundamentals .panel [data-action=search-provider]:not([data-busy-lock])")
    page.click("[data-role=fund-remote] [data-action=search-provider]")


# ------------------------------------------------------------------------------------------------------ nothing set up
def test_the_data_manager_says_how_to_give_the_sec_its_contact_in_a_secret_and_takes_none(browser, live_unset):  # noqa: F811
    page = _page(browser, live_unset)
    try:
        _open(page, "data", demo=False)
        page.wait_for_selector("[data-role=sec-contact-guide]")
        guide = page.inner_text("[data-role=sec-contact-guide]")
        assert "SEC_EDGAR_USER_AGENT" in guide and "Secrets" in guide and "Stop and then Run" in guide and "needs no key" in guide
        assert [c.get_attribute("data-provider") for c in page.locator("[data-role=fund-provider]").all()] == ["openfilings", "sec_edgar"]
        names = page.evaluate("""[...document.querySelectorAll('#fundamentals input, #fundamentals textarea, #fundamentals select')].map((e) =>
            [e.type, e.getAttribute('aria-label') || '', e.placeholder || '', (e.labels && e.labels.length ? e.labels[0].textContent : '')].join(' | '))""")
        assert names and not [n for n in names if re.search(r"password|secret|api key|token|\bkey\b|contact|user.?agent|e-?mail", n, re.I)], names                  # nothing on the page takes a key or a contact
        assert page.locator("[data-provider=sec_edgar] [data-action=test-provider]").count() == 0 and _tables_have_names(page) == []
        assert page.problems == []
    finally:
        page.close()


def test_the_explorer_says_the_sec_is_not_set_up_and_where_to_look(browser, live_unset):  # noqa: F811
    page = _page(browser, live_unset)
    try:
        _open(page, "companies", demo=False)
        page.wait_for_selector("[data-role=fund-remote]")
        _provider(page, "sec_edgar")
        page.wait_for_selector("#fundamentals .panel [data-action=search-provider]:has-text('Search SEC EDGAR (free)')")
        assert page.locator("select[data-key=market]").count() == 0                                   # the SEC list has one market
        assert "SEC EDGAR is not set up yet" in page.inner_text("[data-role=fund-remote]") and "SEC_EDGAR_USER_AGENT" in page.inner_text("[data-role=fund-remote]")
        assert page.locator("#fundamentals .panel [data-action=search-provider]").is_disabled()
        page.click("[data-role=fund-remote] button:has-text('Open the Data manager')")
        page.wait_for_selector("[data-role=sec-contact-guide]")
        assert page.problems == []
    finally:
        page.close()


# ------------------------------------------------------------------------------------------------------ the SEC set up
def test_a_company_is_found_in_the_sec_list_added_for_nothing_and_fetched_for_two_free_requests_with_its_whole_history(browser, live_sec):  # noqa: F811
    page = _page(browser, live_sec)
    fake = live_sec.fakes["sec"]
    try:
        _open(page, "companies", demo=False)
        page.wait_for_selector("[data-role=fund-remote]")
        _provider(page, "sec_edgar")
        page.wait_for_selector("#fundamentals .panel [data-action=search-provider]:has-text('Search SEC EDGAR (free)')")
        assert "Search the SEC's list of US-listed companies" in page.inner_text("#fundamentals .panel") and page.locator("#fundamentals .panel [role=img]").count() == 0       # no allowance meter: it is free
        _search_the_directory(page, "Apple")
        page.wait_for_selector("[data-role=fund-remote] tr[data-symbol='us:AAPL']")
        remote = page.inner_text("[data-role=fund-remote]")
        assert "SEC EDGAR: “Apple”" in remote and "read as a company name" in remote and "nothing spent" in remote and "Apple Inc." in remote and "about 2 free requests" in remote
        assert "about 2 requests" in page.inner_text("[data-role=fund-remote] [data-action=add-fetch-found]") and fake.kinds() == ["tickers"]                  # the list, once; a search asks nothing
        page.click("[data-role=fund-remote] [data-action=add-fetch-found]")
        page.wait_for_selector("[data-role=fund-fetch-report]", timeout=LONG)
        text = page.inner_text("[data-role=fund-fetch-report]")
        assert "Fetched 10 annual periods for us:AAPL and spent 2 requests." in text and "Loaded 10 daily prices (free)" in text
        assert fake.kinds() == ["tickers", "submissions", "facts"] and live_sec.asked == [["AAPL"]]
        row = page.locator("[data-role=fund-companies] tbody tr").first.locator("td").all_inner_texts()
        assert row[0].startswith("AAPL") and row[1] == "Apple Inc." and row[2] == "US" and row[3] == "Technology" and row[4] == "10"
        page.locator("[data-role=fund-companies] tbody tr").first.locator("[data-action=open-company]").click()
        page.wait_for_selector("[data-role=fund-coverage] .years")
        card = "[data-role=fund-coverage]"
        assert "Annual statements held: 10 years, FY2016 to FY2025" in page.inner_text("[data-role=fund-years-summary]")
        reading = page.inner_text("[data-role=fund-years-reading]")
        assert "the request to SEC EDGAR for every year brought back 10 annual statements (FY2016 to FY2025) and cost 1 request" in reading and "whole history in one free request" in reading
        assert "Credits pay" not in page.inner_text(card) and "SEC EDGAR sends a company's whole history in one free request" in page.inner_text(card)
        assert page.locator("[data-role=fund-years-provider]").count() == 0                           # only one provider knows the company: nothing to choose
        assert page.locator(f"{card} button.y.held").count() == 10 and page.locator(f"{card} button.y.empty").count() > 0
        assert page.inner_text("[data-role=fund-coverage]").count("credit") == 0
        assert CONTACT not in page.content() and "example.org" not in page.inner_text("body") and page.problems == []
    finally:
        page.close()


def test_a_company_two_providers_know_shows_whose_statements_are_on_view_and_each_asks_in_its_own_unit(browser, live_sec):  # noqa: F811
    page = _page(browser, live_sec)
    try:
        _open(page, "companies", demo=False)
        page.wait_for_selector("[data-role=fund-remote]")
        _search_the_directory(page, "Apple")                                                          # OpenFilings first (its default), for 11 credits
        page.wait_for_selector("[data-role=fund-remote] tr[data-symbol='us:AAPL']")
        page.click("[data-role=fund-remote] [data-action=add-fetch-found]")
        page.wait_for_selector("[data-role=fund-fetch-report]", timeout=LONG)
        assert "spent 11 credits" in page.inner_text("[data-role=fund-fetch-report]")
        _provider(page, "sec_edgar")                                                                  # the same company, from the SEC: the same company in the database
        page.click("[data-role=fund-remote] [data-action=search-provider]")
        page.wait_for_selector("[data-role=fund-remote] tr[data-symbol='us:AAPL']")
        assert "Yes" not in page.inner_text("[data-role=fund-remote] tr[data-symbol='us:AAPL']")      # the SEC does not hold it yet
        page.click("[data-role=fund-remote] [data-action=add-found]")
        _wait_until(page, "document.querySelector(\"[data-role=fund-remote] tr[data-symbol='us:AAPL']\").innerText.includes('Yes, no statements yet')", 30)
        assert page.locator("[data-role=fund-companies] tbody tr").count() == 1                      # one company, not two
        page.click("[data-role=fund-remote] [data-action=fetch-statements]")
        _wait_until(page, "document.querySelector('[data-role=fund-fetch-report]') && document.querySelector('[data-role=fund-fetch-report]').innerText.includes('spent 1 request.')", 60)                 # its record is known already (OpenFilings gave the sector): the SEC is asked for the facts only
        page.locator("[data-role=fund-companies] tbody tr").first.locator("[data-action=open-company]").click()
        page.wait_for_selector("[data-role=fund-years-provider]")
        both = page.locator("[data-role=fund-years-provider] button")
        assert both.all_inner_texts() == ["OpenFilings (12 years)", "SEC EDGAR (10 years)"] and both.nth(0).get_attribute("aria-pressed") == "true"
        assert "the request to OpenFilings" in page.inner_text("[data-role=fund-years-reading]") and "Credits pay for a request" in page.inner_text("[data-role=fund-coverage]")
        both.nth(1).click()
        _wait_until(page, "document.querySelector('[data-role=fund-years-reading]') && document.querySelector('[data-role=fund-years-reading]').innerText.includes('SEC EDGAR')", 30)
        assert "whole history in one free request" in page.inner_text("[data-role=fund-years-reading]")
        assert page.locator("[data-role=fund-years-provider] button").nth(1).get_attribute("aria-pressed") == "true"
        page.click("[data-role=fund-coverage] button.y.empty >> nth=0")                               # a year before the first XBRL filing: found empty, offered again on request
        page.wait_for_selector("[data-role=fund-years-plan-text]")
        assert "SEC EDGAR" in page.inner_text("[data-role=fund-years-plan-text]") or "1 request" in page.inner_text("[data-role=fund-years-plan-text]")
        assert page.problems == []
    finally:
        page.close()


def test_the_data_manager_adds_and_fetches_from_the_sec_within_a_budget_of_requests(browser, live_sec):  # noqa: F811
    page = _page(browser, live_sec)
    fake = live_sec.fakes["sec"]
    try:
        _open(page, "data", demo=False)
        page.wait_for_selector("[data-provider=sec_edgar] [data-action=test-provider]")
        card = page.inner_text("[data-provider=sec_edgar]")
        assert "free" in card and "A new company takes 2 requests" in card and "Your contact is set, but has no email address" not in card and "example.org" not in card
        page.click("[data-provider=sec_edgar] [data-action=test-provider]")
        page.wait_for_selector("[data-role=fund-test-result]")
        assert "Works: the SEC accepted the contact" in page.inner_text("[data-role=fund-test-result]") and fake.kinds() == ["submissions"]
        _provider(page, "sec_edgar")
        assert "Fetch statements (free)" in page.inner_text("#fundamentals .panel") and page.locator("#fundamentals .panel select[data-key=market]").count() == 0
        assert "Most requests this run may make" in page.inner_text("#fundamentals .panel") and "Tickers, one or several (AAPL, MSFT, BRK-B)" in page.inner_text("#fundamentals .panel")
        page.fill("[data-role=fund-symbols]", "WMT, JPM")
        page.click("#fundamentals .panel [data-action=add-companies]")
        page.wait_for_selector(".toast")
        page.click("#fundamentals .panel [data-action=plan-sync]")
        page.wait_for_selector("[data-role=fund-plan]")
        plan = page.inner_text("[data-role=fund-plan]")
        assert "4 requests" in plan and "2 missing" in plan
        page.click("#fundamentals .panel [data-action=start-sync]")
        page.wait_for_selector("[data-role=fund-report]", timeout=LONG)
        report = page.inner_text("[data-role=fund-report]")
        assert "Fetched" in report and "Spent" in report and "4 requests" in report and "Failed" in report
        assert fake.kinds().count("facts") == 2 and fake.kinds().count("submissions") == 3                # (the test above is one of them)
        assert CONTACT not in page.content() and page.problems == []
    finally:
        page.close()


def test_the_overview_counts_the_sec_as_a_data_source_and_says_what_a_company_costs_in_requests(browser, live_unset):  # noqa: F811
    page = _page(browser, live_unset)
    try:
        page.click("#tab-fundamentals")
        page.wait_for_selector("[data-role=fund-not-ready]")
        card = page.locator("[data-role=fund-not-ready]")
        titles = card.locator("li .body b").all_inner_texts()
        assert titles == ["Set up your OpenFilings key", "Add companies", "Fetch their statements", "Load their prices"]
        assert "SEC EDGAR" in card.locator("li").nth(0).inner_text() and "Replit's Secrets" in card.locator("li").nth(0).inner_text()
        assert page.problems == []
    finally:
        page.close()


def test_with_only_the_sec_set_up_the_first_step_is_done_and_its_cost_is_in_requests(browser, tmp_path_factory):  # noqa: F811
    running, saved = _serve(tmp_path_factory, "seconly", sec=True, openfilings=False)
    page = _page(browser, running)
    try:
        page.click("#tab-fundamentals")
        page.wait_for_selector("[data-role=fund-not-ready]")
        card = page.locator("[data-role=fund-not-ready]")
        assert card.locator("li.done").count() == 1 and "Connect a data source" in card.locator("li.done").inner_text() and "SEC EDGAR is ready" in card.locator("li.done").inner_text()
        assert "About 2 requests a company" in card.locator("li").nth(2).inner_text() and "credits" not in card.locator("li").nth(2).inner_text()
        assert page.problems == []
    finally:
        page.close()
        running.stop()
        saved.undo()
