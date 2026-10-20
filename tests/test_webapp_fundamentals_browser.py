"""The Fundamentals tab in a real browser: seven views over two separate databases (the real one, empty until a provider is set up, and a simulated demo), a Data manager that explains the key without ever
asking for one and shows no key it holds, company pages that print the day each figure became public, a factor study that reads in words with a table behind every chart, a search whose every formula is
counted and whose survivors are Candidates and never Validated, a library that keeps its failures, and the correlation, combination, model, backtest and monitor of the alphas. Skipped when Playwright or
Chromium is missing."""

from __future__ import annotations

import re

import pytest

from src.fundamentals import expr as X
from src.fundamentals.keys import KeyPool
from src.fundamentals.providers.openfilings import OpenFilingsProvider
from src.fundamentals.usage import Allowance
from src.fundamentals.workbench import KEY_NAMES
from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.ml_helpers import make_bundle
from tests.test_fund_api import run
from tests.test_fund_data import KEY_1, FakeOpenFilings
from tests.test_webapp import _fetch
from tests.test_webapp_browser import _page_with, browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _wait_until

sync_api = pytest.importorskip("playwright.sync_api")

MODES = ["Overview", "Company explorer", "Factor lab", "Alpha discovery", "Alpha library", "Alpha monitor", "Data manager"]
SECRETS = ("of_zzzzSECRETzzzz_one_0123456789", "of_zzzzSECRETzzzz_two_9876543210")
WORLD = dict(db="demo", companies=40, start_year=2010, end_year=2018, seed=5, planted={"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045})
SEARCH = {"n_candidates": 30, "seed": 4, "seeds": ["div(fcf,market_cap)", "delta(roic,12)"], "finalists": 3}
LONG = 240_000


def _app(tmp_path_factory, tag):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp(f"{tag}_root"), config)
    bundle = make_bundle(n_assets=4, n=400, seed=3, alpha=0.0)
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp(f"{tag}_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)


@pytest.fixture(scope="module")
def live_fund(tmp_path_factory):
    """An app whose real database is empty and whose demo database holds a simulated world and one finished search (so the library has Candidates and the graveyard has failures)."""
    saved = pytest.MonkeyPatch()
    for name in KEY_NAMES:
        saved.delenv(name, raising=False)                                                   # a key in the developer's own environment would change what an empty setup looks like
    app = _app(tmp_path_factory, "fund")
    run(app, "/api/fund/simulate", **WORLD)
    run(app, "/api/fund/discover", db="demo", spec=SEARCH)
    thin = {"universe": {}, "freq": "M", "horizon": "1m", "entry_lag": 1, "min_assets": 20, "direction": 1.0, "steps": [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}]}
    app.fundamentals.bench("demo").library.add_candidate(X.parse("div(fcf,enterprise_value)"), thin, {"ic": 0.08, "direction": 1.0, "n_tested": 40, "oos_t": 0.2})      # evidence too thin to validate
    running = server.start(app, "127.0.0.1", 0)
    running.app = app
    yield running
    running.stop()
    saved.undo()


@pytest.fixture()
def live_real(tmp_path_factory):
    """An app whose real database is empty and whose OpenFilings is a stand-in that answers as its documentation says (a free plan: 100 credits a day, 1 for a search, 10 for statements, 1 for a lookup)."""
    saved = pytest.MonkeyPatch()
    for name in KEY_NAMES:
        saved.delenv(name, raising=False)
    app = _app(tmp_path_factory, "real")
    wb = app.fundamentals.bench("real")
    fake = FakeOpenFilings({KEY_1: "a"})
    pool = KeyPool("openfilings", [("OPENFILINGS_API_KEY_1", KEY_1)], wb.service.usage, wb.store.salt(), Allowance("credits", 100, 1000, "test"))
    wb.service.register("openfilings", OpenFilingsProvider(pool, transport=fake, sleep=lambda s: None))
    running = server.start(app, "127.0.0.1", 0)
    running.app, running.fake = app, fake
    yield running
    running.stop()
    saved.undo()


def _page(browser, live, width=1400):  # noqa: F811
    pg = browser.new_page(viewport={"width": width, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.on("dialog", lambda d: d.dismiss())                                                  # a prompt for a reason is answered with "cancel" unless a test says otherwise
    pg.goto(live.url)
    pg.wait_for_selector("body[data-ready]")
    return pg


def _open(page, mode, demo=True):
    page.click("#tab-fundamentals")
    page.wait_for_selector("#fundamentals .panel")
    page.click(f"[data-role=fund-db] button:has-text('{'Demo' if demo else 'Real'}')")
    page.click(f"#fundamentals .labtop button[data-mode={mode}]")


def _go(page):
    page.click("#fundamentals .runbar button.btn.primary")


def _tile(page, label):
    """The big number of the tile with this label."""
    return page.locator("#fundamentals-results .tile", has=page.locator(f"text=/^{re.escape(label)}$/")).first.locator(".value").inner_text()


def _tables_have_names(page) -> list[str]:
    return page.evaluate("[...document.querySelectorAll('#fundamentals th')].filter((t) => !t.textContent.trim()).map((t) => t.parentElement.textContent.slice(0, 60))")


# ------------------------------------------------------------------------------------------------------ the frame
def test_the_tab_has_seven_views_and_opens_on_a_real_database_that_has_nothing_in_it(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        page.click("#tab-fundamentals")
        page.wait_for_selector("#fundamentals .panel")
        assert page.locator("#fundamentals .labtop [role=tab]").all_inner_texts() == MODES
        assert page.locator("[data-role=fund-db] button[aria-pressed=true]").inner_text() == "Real data"
        page.wait_for_selector("[data-role=fund-not-ready]")
        assert "Not ready to study yet" in page.inner_text("[data-role=fund-not-ready]") and _tile(page, "Companies") == "0"
        assert page.locator("[data-role=fund-demo]").count() == 0
        page.click("[data-role=fund-db] button:has-text('Demo')")
        page.wait_for_selector("[data-role=fund-demo]")                                     # the demo says what it is at the top of every view
        assert "simulated" in page.inner_text("[data-role=fund-demo]") and "says anything about real markets" in page.inner_text("[data-role=fund-demo]")
        page.wait_for_selector("#fundamentals-results .tile")
        assert _tile(page, "Companies") == "40" and page.locator("[data-role=fund-not-ready]").count() == 0
        page.click("[data-role=fund-db] button:has-text('Real')")
        page.wait_for_selector("[data-role=fund-not-ready]")
        assert _tile(page, "Companies") == "0" and page.locator("[data-role=fund-demo]").count() == 0
        assert page.problems == []
    finally:
        page.close()


def test_a_study_cannot_start_on_an_empty_database_and_the_page_says_what_to_do(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        for mode in ("factors", "discovery"):
            _open(page, mode, demo=False)
            page.wait_for_selector("#fundamentals .guard:not([hidden])")
            assert "Add companies and fetch their statements in the Data manager first" in page.inner_text("#fundamentals .guard")
            assert page.locator("#fundamentals .runbar button.btn.primary").is_disabled(), mode
        _open(page, "library", demo=False)
        page.click("[data-role=fund-lib-view] button:has-text('Backtest')")
        page.wait_for_selector("#fundamentals .guard:not([hidden])")
        assert page.locator("#fundamentals .runbar button.btn.primary").is_disabled()
        _open(page, "factors", demo=True)
        page.wait_for_selector("select[data-role=fund-factor]")
        assert page.locator("#fundamentals .runbar button.btn.primary").is_enabled() and page.locator("#fundamentals .guard:not([hidden])").count() == 0
        assert page.problems == []
    finally:
        page.close()


# ------------------------------------------------------------------------------------------------------ the data manager
def test_the_data_manager_explains_the_key_and_never_asks_for_one(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "data", demo=False)
        page.wait_for_selector("[data-role=fund-key-guide]")
        guide = page.inner_text("[data-role=fund-key-guide]")
        assert "OPENFILINGS_API_KEY_1" in guide and "never goes through this page" in guide and "Tools" in guide and "Secrets" in guide
        names = page.evaluate("""[...document.querySelectorAll('#fundamentals input, #fundamentals textarea, #fundamentals select')].map((e) =>
            [e.type, e.getAttribute('aria-label') || '', e.placeholder || '', (e.labels && e.labels.length ? e.labels[0].textContent : '')].join(' | '))""")
        assert names and not [n for n in names if re.search(r"password|secret|api key|token|\\bkey\\b", n, re.I)], names                  # nothing on the page takes a key
        assert page.locator("#fundamentals input[type=password]").count() == 0
        slots = page.locator("[data-role=fund-provider] table >> nth=0").locator("tbody tr")
        assert slots.count() == 5 and [r.locator("td").nth(1).inner_text() for r in slots.all()] == [f"OPENFILINGS_API_KEY_{i}" for i in range(1, 6)]
        assert all(r.locator("td").nth(3).inner_text() == "Not set" for r in slots.all())
        policy = page.inner_text("[data-role=fund-policy]")
        assert "do not add up" in policy and "does not allow rotating keys or accounts" in policy
        page.locator("#fundamentals details.how summary", has_text="credits").click()
        assert "rotating accounts or API keys" in page.inner_text("#fundamentals details.how[open]")
        assert page.inner_text("#fundamentals-results").count("100 credits") >= 1 and "1,000 credits" in page.inner_text("#fundamentals-results")
        assert page.problems == []
        _open(page, "data", demo=True)                                                      # the demo never calls a provider: it has no key card and a form to make the world instead
        page.wait_for_selector("[data-role=fund-provider][data-provider=simulated]")
        assert page.locator("[data-role=fund-provider][data-provider=openfilings]").count() == 0 and page.locator("[data-action=make-demo]").count() == 1
    finally:
        page.close()


def test_a_key_in_the_environment_shows_as_a_slot_and_never_as_a_value(browser, tmp_path_factory, monkeypatch):  # noqa: F811
    monkeypatch.setenv("OPENFILINGS_API_KEY_1", SECRETS[0])
    monkeypatch.setenv("OPENFILINGS_API_KEY_2", SECRETS[1])
    running = server.start(_app(tmp_path_factory, "keys"), "127.0.0.1", 0)
    page = _page(browser, running)
    try:
        _open(page, "data", demo=False)
        page.wait_for_selector("[data-role=fund-provider] table tbody tr")
        rows = page.locator("[data-role=fund-provider] table >> nth=0").locator("tbody tr")
        assert [r.locator("td").nth(2).inner_text() for r in rows.all()][:2] == ["active", "standby"]
        assert page.locator("[data-role=fund-key-guide]").count() == 0 and page.locator("[data-action=test-provider]").count() == 1
        status = page.evaluate("window.QL.api('/api/fund/status?db=real').then((r) => JSON.stringify(r))")
        for blob in (page.content(), page.inner_text("body"), status):
            assert not any(secret in blob or secret[:12] in blob or secret[-10:] in blob for secret in SECRETS)
        assert page.problems == []
    finally:
        page.close()
        running.stop()


# ------------------------------------------------------------------------------------------------------ the companies
def test_a_company_page_prints_the_day_each_figure_became_public_and_every_chart_has_a_table(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "companies")
        page.wait_for_selector("[data-role=fund-companies] tbody tr")
        assert page.locator("[data-role=fund-companies] tbody tr").count() == 25 and "1 to 25 of 40" in page.inner_text("[data-role=fund-companies]")
        page.click("[data-role=fund-companies] button:has-text('Next')")
        _wait_until(page, "document.querySelector('[data-role=fund-companies]').innerText.includes('26 to 40 of 40')", 30)
        assert page.locator("[data-role=fund-companies] tbody tr").count() == 15
        page.click("[data-role=fund-companies] button:has-text('Previous')")
        _wait_until(page, "document.querySelector('[data-role=fund-companies]').innerText.includes('1 to 25 of 40')", 30)
        ticker = page.locator("[data-role=fund-companies] tbody tr").first.locator("td").first.inner_text().split(" ")[0]
        page.locator("[data-role=fund-companies] tbody tr").first.locator("[data-action=open-company]").click()
        page.wait_for_selector("[data-role=fund-statements]")
        heads = page.locator("[data-role=fund-statements] th").all_inner_texts()
        assert heads[:3] == ["Year", "Period end", "Public from"] and ticker in page.inner_text("#fundamentals-results h2")
        dates = page.locator("[data-role=fund-statements] tbody tr").first.locator("td").all_inner_texts()[1:3]
        assert dates[1][:10] > dates[0][:10]                                                # a statement is public after its period ends, never on it
        assert page.locator("[data-role=fund-statements] tbody tr").count() >= 8
        for role in ("fund-dupont", "fund-piotroski", "fund-anomaly", "fund-price-symbol"):
            assert page.locator(f"[data-role={role}]").count() == 1, role
        assert "Contribution to the change" in page.inner_html("[data-role=fund-dupont]") and "never evidence of fraud" in page.inner_text("#fundamentals-results")
        assert page.locator("[data-role=fund-price-symbol] input").is_disabled()           # a simulated company has no market-data symbol
        cards = page.locator("#fundamentals-results .chartcard")
        assert cards.count() >= 4
        for i in range(cards.count()):
            cards.nth(i).locator(".seg button:has-text('Table')").click()
            assert cards.nth(i).locator("table tbody tr").count() >= 2, i
        page.click("[data-action=back-to-list]")
        page.wait_for_selector("[data-role=fund-companies]")
        assert page.problems == []
    finally:
        page.close()


def test_a_company_is_found_by_name_in_openfilings_added_for_nothing_and_fetched_for_the_cost_the_button_states(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    fake = live_real.fake
    fake.skipped = ["jp"]
    try:
        _open(page, "companies", demo=False)
        page.wait_for_selector("[data-role=fund-companies]")
        assert "There are no companies in your database yet" in page.inner_text("[data-role=fund-companies]")
        assert page.locator("select[data-key=country], select[data-key=sector]").count() == 0                    # a list with only "All" in it is not a choice, so it is not shown
        assert "fill in as companies are added" in page.inner_text("#fundamentals .panel")
        page.fill("input[data-key=q]", "Nvidia")
        page.press("input[data-key=q]", "Enter")                                            # Enter searches the database
        _wait_until(page, "document.querySelector('[data-role=fund-companies] h3') && document.querySelector('[data-role=fund-companies] h3').innerText.includes('no companies in your database yet')", 30)
        assert "No company matches" not in page.inner_text("#fundamentals-results") and fake.requests == []     # an empty database is not "no match", and looking at the database never asks the provider
        page.click("[data-role=fund-remote] [data-action=search-provider]")
        page.wait_for_selector("[data-role=fund-remote] tr[data-symbol='us:NVDA']")
        remote = page.inner_text("[data-role=fund-remote]")
        assert "read as a company name" in remote and "1 credit spent" in remote and "99 credits left today" in remote and "NVIDIA Corp" in remote and "also listed as mx:NVDA" in remote
        assert [r["query"] for r in fake.requests] == [{"query": "Nvidia"}] and page.locator("[data-role=fund-remote] tr[data-symbol='us:NVDA'] [data-action=add-found]").count() == 1
        assert "did not search these markets this time: JP" in page.inner_text("[data-role=fund-skipped]")     # a market the directory left out is said, not hidden
        assert "about 11 credits" in page.inner_text("[data-role=fund-remote] [data-action=add-fetch-found]")
        page.click("[data-role=fund-remote] [data-action=add-found]")                       # adding is free and keeps the name the search paid for
        _wait_until(page, "document.querySelector(\"[data-role=fund-remote] tr[data-symbol='us:NVDA']\").innerText.includes('Yes, no statements yet')", 30)
        assert len(fake.requests) == 1 and page.locator("[data-role=fund-companies] tbody tr").count() == 1 and "NVIDIA Corp" in page.inner_text("[data-role=fund-companies]")
        page.click("[data-role=fund-remote] [data-action=fetch-statements]")
        page.wait_for_selector("[data-role=fund-fetch-report]", timeout=LONG)
        said = re.search(r"Fetched (\d+) annual periods for us:NVDA and spent 11 credits", page.inner_text("[data-role=fund-fetch-report]"))
        assert said and int(said.group(1)) >= 8 and "88 credits left today" in page.inner_text("[data-role=fund-remote]")
        assert int(page.locator("[data-role=fund-companies] tbody tr td").nth(4).inner_text()) >= 8
        assert "saved answer, free" in page.inner_text("[data-role=fund-remote]") and len(fake.requests) == 3        # the search, the lookup and the statements: the page read the saved search again for nothing
        assert _tables_have_names(page) == [] and page.locator("[data-role=fund-remote] button:not([aria-label])").count() == 0
        page.fill("input[data-key=q]", "Zzyzx")                                             # a company the directory does not know
        page.press("input[data-key=q]", "Enter")
        page.click("[data-role=fund-remote] [data-action=search-provider]")
        page.wait_for_selector("[data-role=fund-remote-none]")
        assert "no company that matches" in page.inner_text("[data-role=fund-remote]") and page.locator("[data-role=fund-remote] [data-action=search-as-symbol]").count() == 1
        page.click("[data-role=fund-remote] [data-action=search-as-symbol]")                # the other reading is a second, explicit credit
        page.wait_for_selector("[data-role=fund-remote] [data-action=search-as-name]")
        assert fake.requests[-1]["query"] == {"ticker": "ZZYZX"}
        page.fill("input[data-key=q]", "x")                                                 # too short: the page says so and asks nothing
        before = len(fake.requests)
        page.click("#fundamentals .panel [data-action=search-provider]")                   # the button in the panel is always there
        page.wait_for_selector(".toast")
        assert len(fake.requests) == before
        blob = page.content() + page.inner_text("body")
        assert KEY_1 not in blob and page.problems == []
    finally:
        page.close()


def test_without_a_key_the_search_of_openfilings_says_it_is_not_set_up_and_where_to_look(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "companies", demo=False)
        page.wait_for_selector("[data-role=fund-remote]")
        assert "not set up yet" in page.inner_text("[data-role=fund-remote]") and page.locator("[data-role=fund-remote] [data-action=search-provider]").count() == 0
        assert page.locator("#fundamentals .panel [data-action=search-provider]").is_disabled() and "Data manager shows how to add a key" in page.inner_text("#fundamentals .panel")
        page.click("[data-role=fund-remote] button:has-text('Open the Data manager')")
        page.wait_for_selector("[data-role=fund-key-guide]")
        _open(page, "companies", demo=True)                                                 # the demo is a simulated world: nothing in it can be asked of a provider
        page.wait_for_selector("[data-role=fund-companies] tbody tr")
        assert page.locator("[data-role=fund-remote]").count() == 0 and page.locator("[data-action=search-provider]").count() == 0
        assert "40 matches" in page.inner_text("#fundamentals-results .head")
        assert page.problems == []
    finally:
        page.close()


# ------------------------------------------------------------------------------------------------------ the factor lab
def test_a_factor_study_reads_in_words_with_a_table_behind_every_chart_and_counts_its_years_right(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "factors")
        page.wait_for_selector("select[data-role=fund-factor]")
        assert page.locator("select[data-role=fund-factor]").input_value() == "fcf_yield"
        _go(page)
        page.wait_for_selector("[data-role=fund-verdict]", timeout=LONG)
        verdict = page.locator("[data-role=fund-verdict]")
        assert verdict.get_attribute("data-level") in ("good", "mixed") and "Interesting is not validated" in verdict.inner_text() and "fcf_yield" in verdict.inner_text()
        assert _tile(page, "Mean rank IC").startswith("+") and page.locator("[data-role=fund-breakdown-country] tbody tr").count() >= 2
        cards = page.locator("#fundamentals-results .chartcard")
        assert cards.count() >= 10
        year = cards.filter(has=page.locator("h3", has_text="IC by year")).first
        caption = year.locator("p.note").inner_text()
        said = re.search(r"Positive in (\d+) of (\d+) years", caption)
        assert said, caption
        year.locator(".seg button:has-text('Table')").click()
        shown = [r.locator("td").nth(1).inner_text() for r in year.locator("tbody tr").all()]
        assert int(said.group(2)) == len(shown) and int(said.group(1)) == sum(1 for v in shown if v.startswith("+"))      # the count in the sentence is the count in the table
        for i in range(cards.count()):
            cards.nth(i).locator(".seg button:has-text('Table')").click()
            assert cards.nth(i).locator("table tbody tr").count() >= 2, i
        with page.expect_download() as got:
            page.click("button:has-text('Download the IC series')")
        assert got.value.suggested_filename.endswith("_ic.csv")
        page.click("[data-action=backtest-this]")                                           # the factor goes to the backtest with its own name
        page.wait_for_selector("[data-role=fund-lib-view] button[aria-pressed=true]:has-text('Backtest')")
        assert page.locator("#fundamentals select[aria-label='signal 1 name']").input_value() == "fcf_yield"
        assert page.problems == []
    finally:
        page.close()


def test_a_scan_an_audit_and_a_formula_of_your_own(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "factors")
        page.wait_for_selector("select[data-role=fund-factor]")
        page.click("[data-action=scan-factors]")
        page.wait_for_selector("[data-role=fund-scan]", timeout=LONG)
        assert page.locator("#fundamentals-results .banner:has-text('Many factors looked at')").count() == 1 and page.locator("[data-role=fund-scan] tbody tr").count() >= 20
        assert page.locator("[data-role=fund-scan] tbody tr[data-factor=fcf_yield]").count() == 1 and _tables_have_names(page) == []
        page.click("[data-action=audit-factor]")
        page.wait_for_selector("[data-role=fund-audit]", timeout=LONG)
        assert page.locator("[data-role=fund-audit] tbody tr").count() == 2 and "from the day it was public" in page.inner_text("[data-role=fund-audit]")
        page.click("[data-role=fund-source] button:has-text('My own formula')")
        page.fill("#fundamentals .panel [data-key=formula]", "div(fcf,market_cap")
        page.click("[data-action=check-formula]")
        _wait_until(page, "document.querySelector('[data-role=fund-formula-check]').textContent.length > 0", 30)
        assert "Written canonically" not in page.inner_text("[data-role=fund-formula-check]")      # a formula that does not parse is refused, with the reason
        page.fill("#fundamentals .panel [data-key=formula]", "div( fcf , market_cap )")
        page.click("[data-action=check-formula]")
        _wait_until(page, "document.querySelector('[data-role=fund-formula-check]').textContent.includes('Written canonically')", 30)
        assert "div(fcf,market_cap)" in page.inner_text("[data-role=fund-formula-check]") and "Reads fcf, market_cap" in page.inner_text("[data-role=fund-formula-check]")
        _go(page)
        page.wait_for_selector("[data-role=fund-verdict]", timeout=LONG)
        assert "div(fcf,market_cap)" in page.inner_text("[data-role=fund-verdict]")
        assert page.problems == []
    finally:
        page.close()


# ------------------------------------------------------------------------------------------------------ discovery and the library
def test_the_library_lists_the_candidates_and_the_failures_and_a_view_that_was_left_comes_back(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "library")
        page.wait_for_selector("[data-role=fund-alphas]")
        kept = page.locator("[data-role=fund-alphas] tbody tr").count()
        assert kept >= 1 and set(page.locator("[data-role=fund-alphas] tbody tr td:nth-child(4)").all_inner_texts()) == {"Candidate"}          # nothing the search finds is Validated
        assert _tile(page, "Validated") == "0" and _tile(page, "Candidate") == str(kept)
        assert all(w in ("higher is better", "lower is better") for w in page.locator("[data-role=fund-alphas] tbody tr td:nth-child(3)").all_inner_texts())
        assert _tables_have_names(page) == []
        page.click("[data-role=fund-lib-view] button:has-text('Graveyard')")                # the lists are read when the view opens: no button to press, no "Reading…" left behind
        page.wait_for_selector("[data-role=fund-graveyard]")
        graves = page.locator("[data-role=fund-graveyard] tbody tr")
        assert graves.count() >= 1 and "Failed at" in page.locator("[data-role=fund-graveyard] th").all_inner_texts() and all(t.strip() for t in page.locator("[data-role=fund-graveyard] tbody tr td:nth-child(3)").all_inner_texts())
        total = int(re.search(r"(\d+) tried in all", page.inner_text("#fundamentals-results")).group(1))
        assert total == graves.count() + kept                                                # every idea tried is either kept or in the graveyard
        page.click("[data-role=fund-lib-view] button:has-text('Alphas')")
        page.wait_for_selector("[data-role=fund-alphas]")
        page.locator("[data-role=fund-alphas] [data-action=open-alpha]").last.click()           # the one with the thinnest evidence: it is listed last
        page.wait_for_selector("[data-role=fund-alpha-detail]")
        detail = page.inner_text("[data-role=fund-alpha-detail]").lower()                 # the headings are set in capitals by the stylesheet
        assert "passed every screen" in detail and "status history" in detail and "ic on the locked dates" in detail and "the ic figures above are in that direction" in detail
        page.click("[data-action=validate-alpha]")                                          # the evidence is thin: it is refused, and the prompt for a reason to approve it anyway is cancelled
        page.wait_for_timeout(1500)
        assert page.locator("[data-role=fund-alpha-detail] .pill", has_text="Candidate").count() == 1 and page.locator("[data-role=fund-alpha-detail] .pill", has_text="Validated").count() == 0
        assert [p for p in page.problems if "status of 400" not in p] == []                  # the refusal of the validation is a 400 the browser logs; nothing else went wrong
    finally:
        page.close()


def test_a_search_counts_every_formula_keeps_the_survivors_as_candidates_and_remembers_the_run(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "discovery")
        page.wait_for_selector("[data-role=fund-runs]")                                      # the search made before the page opened is listed
        assert page.locator("[data-role=fund-runs] li.finished").count() == 1 and page.locator("[data-role=fund-runs] [data-action=resume-run]").count() == 0
        page.fill("#fundamentals .panel [data-key=n]", "30")
        page.fill("#fundamentals .panel [data-key=seed]", "11")
        page.fill("#fundamentals .panel [data-key=finalists]", "3")
        page.select_option("#fundamentals .panel [data-key=seed_mode]", "mine")
        page.fill("[data-role=fund-seeds]", "div(net_income,market_cap)\ndelta(roic,12)\ngrowth(revenue,12)")
        _go(page)
        page.wait_for_selector("[data-role=fund-leaderboard]", timeout=LONG)
        assert _tile(page, "Formulas tried") == "30"
        heads = page.locator("[data-role=fund-leaderboard] th").all_inner_texts()
        assert heads[:2] == ["Formula", "Used as"] and {"IC", "Locked IC", "q", "DSR"} <= set(heads)
        statuses = set(page.locator("[data-role=fund-leaderboard] tbody tr td:nth-child(3)").all_inner_texts())
        assert statuses <= {"Candidate", "Survivor"} and "Validated" not in page.inner_text("#fundamentals-results")
        assert "Nothing here is validated" in page.inner_text("[data-role=fund-leaderboard]")
        funnel = page.locator("#fundamentals-results .chartcard", has=page.locator("h3", has_text="Where the formulas fell"))
        funnel.locator(".seg button:has-text('Table')").click()
        passed = [int(t) for t in funnel.locator("tbody tr td:nth-child(2)").all_inner_texts()]
        assert passed == sorted(passed, reverse=True) and passed[0] >= 5                     # a funnel only narrows
        assert page.locator("[data-role=fund-runs] li.finished").count() == 2 and page.locator("[data-role=fund-runs] [data-action=open-run]").count() == 2
        page.locator("[data-role=fund-runs] [data-action=open-run]").nth(1).click()
        page.wait_for_selector("[data-role=fund-leaderboard]")
        assert _tile(page, "Formulas tried") == "30" and page.problems == []
    finally:
        page.close()


def test_alphas_are_correlated_combined_modelled_and_backtested_and_the_views_keep_their_results(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "library")
        page.wait_for_selector("[data-role=fund-alphas]")
        boxes = page.locator("[data-role=fund-alphas] tbody input[type=checkbox]")
        assert boxes.count() >= 3
        for i in range(3):
            boxes.nth(i).check()
        page.click("[data-action=use-correlation]")
        assert page.locator("#fundamentals select[aria-label$='kind']").count() == 3
        _go(page)
        page.wait_for_selector("[data-role=fund-clusters]", timeout=LONG)
        assert "independent ideas" in page.inner_text("#fundamentals-results")
        cards = page.locator("#fundamentals-results .chartcard")
        cards.first.locator(".seg button:has-text('Table')").click()
        header = [h for h in cards.first.locator("th").all_inner_texts()][1:]
        rows = [r.locator("td").first.inner_text().split(":")[0] for r in cards.first.locator("tbody tr").all()]
        assert header == rows and sorted(rows) == ["1", "2", "3"]                           # each column carries the number of the alpha its row names
        page.click("[data-role=fund-lib-view] button:has-text('Combine')")
        page.wait_for_selector("#fundamentals select[aria-label$='kind']")
        assert page.locator("[data-role=fund-clusters]").count() == 0                       # the result of one view is not shown in another
        _go(page)
        page.wait_for_selector("[data-role=fund-verdict]", timeout=LONG)
        assert "Combined mean IC" in page.inner_text("#fundamentals-results") and page.locator("#fundamentals-results .banner:has-text('Watch.')").count() == 1       # the warnings of three alphas come as one notice
        page.click("[data-role=fund-lib-view] button:has-text('Correlation')")
        page.wait_for_selector("[data-role=fund-clusters]")                                 # and the correlation is still there when the view is opened again
        page.click("[data-role=fund-lib-view] button:has-text('Model')")
        page.wait_for_selector("#fundamentals select[data-key=model]")
        _go(page)
        page.wait_for_selector("[data-role=fund-model-verdict]", timeout=LONG)
        assert "simple rules" in page.inner_text("[data-role=fund-model-verdict]") and page.locator("[data-role=fund-model-compare] tbody tr").count() == 3
        assert _tables_have_names(page) == []
        page.click("[data-role=fund-lib-view] button:has-text('Backtest')")
        page.wait_for_selector("#fundamentals [data-key=side]")
        page.fill("#fundamentals .panel [data-key=n]", "8")
        _go(page)
        page.wait_for_selector("[data-role=fund-backtest-verdict]", timeout=LONG)
        text = page.inner_text("#fundamentals-results")
        assert "Recorded as experiment bt-" in text and "not that it will keep doing so" in text and "Deflated Sharpe" in text
        notice = page.locator("#fundamentals-results .banner:has-text('Watch.')").first
        assert "things to bear in mind" in notice.inner_text() and "survivorship" in notice.inner_text() and "Candidate" in notice.inner_text()
        assert page.locator("[data-role=fund-holdings] tbody tr").count() >= 1 and _tables_have_names(page) == []
        assert page.problems == []
    finally:
        page.close()


def test_the_monitor_judges_every_alpha_that_is_still_in_use(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "monitor")
        _go(page)
        page.wait_for_selector("[data-role=fund-monitor]", timeout=LONG)
        states = page.locator("[data-role=fund-monitor] tbody tr td:nth-child(2)").all_inner_texts()
        assert states and all(re.sub(r"^\S+ ", "", s) in ("Healthy", "Weakening", "Decaying", "Broken") for s in states)
        assert sum(int(_tile(page, w)) for w in ("Healthy", "Weakening", "Decaying", "Broken")) == len(states)
        assert page.problems == []
    finally:
        page.close()


def test_an_alpha_is_moved_between_statuses_with_a_reason_and_is_never_deleted(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund)
    try:
        _open(page, "library")
        page.wait_for_selector("[data-role=fund-alphas]")
        before = page.locator("[data-role=fund-alphas] tbody tr").count()
        page.locator("[data-role=fund-alphas] [data-action=open-alpha]").last.click()
        page.wait_for_selector("[data-action=set-status]")
        page.select_option("[aria-label='new status']", "Rejected")
        page.click("[data-action=set-status]")                                              # no reason, no move
        _wait_until(page, "!!document.querySelector('.toast') && document.querySelector('.toast').textContent.includes('needs a reason')", 15)
        page.fill("[aria-label='reason for the new status']", "does not hold up in the second half")
        page.click("[data-action=set-status]")
        page.wait_for_selector("[data-role=fund-alphas] tbody tr td .pill:has-text('Rejected')")
        assert page.locator("[data-role=fund-alphas] tbody tr").count() == before           # the rejected alpha is still listed
        assert _tile(page, "Rejected") == "1" and "does not hold up in the second half" in page.inner_text("[data-role=fund-alpha-detail]")
        assert page.problems == []
    finally:
        page.close()


def test_an_alpha_is_validated_from_the_page_only_when_the_evidence_holds(browser, live_fund):  # noqa: F811
    """Last of the tests that share one library: it validates the best alpha, whose evidence does hold after the number of ideas tried."""
    page = _page(browser, live_fund)
    try:
        _open(page, "library")
        page.wait_for_selector("[data-role=fund-alphas]")
        page.locator("[data-role=fund-alphas] [data-action=open-alpha]").first.click()
        page.wait_for_selector("[data-action=validate-alpha]")
        page.click("[data-action=validate-alpha]")
        page.wait_for_selector("[data-role=fund-alpha-detail] .pill:has-text('Validated')")
        assert _tile(page, "Validated") == "1" and "evidence holds after the correction for the number of ideas tried" in page.inner_text("[data-role=fund-alpha-detail]")
        assert page.problems == []
    finally:
        page.close()


# ------------------------------------------------------------------------------------------------------ for everyone
def test_every_view_names_its_table_headers_and_meters_and_colours_text_with_classes(browser, live_fund):  # noqa: F811
    """An automated audit with axe-core found these once in 29 states of this tab (the demo and the empty real database, every view and every sub-view with a result, light and dark): headers that held
    only a button, meters without a name, and text coloured from a variable read at draw time (so it kept the light colour on the dark page). These checks keep them found."""
    page = _page(browser, live_fund)
    try:
        for demo, mode in ((True, "overview"), (True, "companies"), (True, "data"), (False, "data"), (False, "overview"), (True, "library"), (True, "discovery")):
            _open(page, mode, demo=demo)
            page.wait_for_timeout(1200)
            assert _tables_have_names(page) == [], (demo, mode)
            assert page.evaluate("[...document.querySelectorAll('#fundamentals [role=img]')].filter((e) => !(e.getAttribute('aria-label') || '').trim()).length") == 0, (demo, mode)
            assert page.evaluate("document.querySelectorAll('#fundamentals :not(svg):not(svg *)[style*=\"color\"]').length") == 0, (demo, mode)
        assert page.problems == []
    finally:
        page.close()


def test_the_views_fit_a_phone(browser, live_fund):  # noqa: F811
    page = _page(browser, live_fund, width=390)
    try:
        for mode in ("overview", "companies", "factors", "discovery", "library", "monitor", "data"):
            _open(page, mode)
            page.wait_for_timeout(900)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), mode
        _open(page, "factors")
        page.wait_for_selector("select[data-role=fund-factor]")
        _go(page)
        page.wait_for_selector("[data-role=fund-verdict]", timeout=LONG)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        assert page.problems == []
    finally:
        page.close()


def test_an_app_older_than_the_tab_says_so(browser, live_fund):  # noqa: F811
    page = _page_with(browser, live_fund, rewrite=lambda c: c.update(api_version=10))
    try:
        page.wait_for_selector(".chip")
        page.click("#tab-fundamentals")
        page.wait_for_selector("#fundamentals .card")
        text = page.inner_text("#fundamentals")
        assert "needs a newer app" in text and "reports version 10; this tab needs 11" in text and "Stop the app" in text
        assert page.locator("#fundamentals .labtop [role=tab]").count() == 0
    finally:
        page.close()
