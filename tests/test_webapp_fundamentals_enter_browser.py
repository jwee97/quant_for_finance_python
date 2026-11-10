"""The Fundamentals tab's Enter reports view, driven in a real browser: a table pasted from Excel fills the grid, "Calculate" works out the ratios on a temporary copy and stores nothing, a mistake is
marked on its cell, "Save" writes the periods to the real database (and only the real one), a saved period can be loaded back, corrected and removed, and a restatement needs its date."""

from __future__ import annotations

import pytest

from tests.test_fund_manual import TABLE
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_fundamentals_browser import _page, live_real  # noqa: F401
from tests.test_webapp_workspaces_browser import _wait_until

pytest.importorskip("playwright.sync_api")



def _page_accepting(browser, live):  # noqa: F811
    """The page as ``_page`` opens it, except that the question "remove this period?" is answered yes."""
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.on("dialog", lambda d: d.accept())
    pg.on("response", lambda r: pg.problems.append(f"HTTP {r.status} {r.url} {r.text()[:400]}") if r.status >= 400 else None)
    pg.goto(live.url)
    pg.wait_for_selector("body[data-ready]")
    return pg


def _enter(page):
    page.click("#tab-fundamentals")
    page.wait_for_selector("#fundamentals .panel")
    page.click("#fundamentals .labtop button[data-mode=enter]")
    page.wait_for_selector("[data-role=manual-grid]")


def _new_company(page, ticker="TEST1"):
    page.click("[data-role=manual-target] button:has-text('A new company')")
    page.fill("input[data-key=ticker]", ticker)
    page.fill("input[data-key=name]", "Test Co")
    page.fill("input[data-key=country]", "US")
    page.fill("input[data-key=currency]", "USD")
    page.fill("input[data-key=sector]", "Technology")


def _paste(page, text=TABLE):
    page.fill("[data-role=manual-paste]", text)
    page.click("button[data-action=manual-parse]")
    page.wait_for_selector("[data-role=manual-report]")


def test_the_view_is_in_the_tab_and_real_data_only(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    try:
        page.click("#tab-fundamentals")
        page.wait_for_selector("#fundamentals .panel")
        assert "Enter reports" in page.locator("#fundamentals .labtop [role=tab]").all_inner_texts()
        page.click("#fundamentals .labtop button[data-mode=enter]")
        page.wait_for_selector("[data-role=manual-grid]")
        assert "grid is empty" in page.inner_text("[data-role=manual-grid]")
        page.click("[data-role=fund-db] button:has-text('Demo')")
        page.wait_for_selector("text=Reports are typed into your real database")
        assert page.locator("[data-role=manual-grid]").count() == 0
        assert page.problems == []
    finally:
        page.close()


def test_a_pasted_table_fills_the_grid_and_the_ratios_are_worked_out_without_storing_anything(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    try:
        _enter(page)
        _new_company(page)
        _paste(page)
        report = page.inner_text("[data-role=manual-report]")
        assert "Read 3 periods" in report and "A line nobody knows" in report and "calendar end" in report
        assert page.locator("input[data-head=period_end]").evaluate_all("els => els.map(e => e.value)") == ["2021-12-31", "2022-12-31", "2023-12-31"]
        assert page.locator("input[data-head=filing_date]").evaluate_all("els => els.map(e => e.value)") == ["2022-02-10", "2023-02-09", "2024-02-08"]
        assert page.input_value("input[data-line=revenue][data-col='2']") == "1500"
        assert page.input_value("input[data-line=cost_of_revenue][data-col='0']") == "-600"      # as typed: the server stores the amount spent, whichever sign
        page.click("#fundamentals .runbar button.btn.primary")
        page.wait_for_selector("[data-role=manual-ratios]")
        text = page.inner_text("[data-role=manual-ratios]")
        assert "Gross margin" in text and "0.433" in text                                          # 650 / 1,500
        assert "Net margin" in page.inner_text("[data-role=manual-ratios]")
        assert page.locator("[data-role=manual-multiples]").count() == 1                           # the prices typed in the grid give yields and multiples
        assert "Price / earnings" in page.inner_text("[data-role=manual-multiples]")
        assert page.locator("[data-role=manual-company]").count() == 1                             # the Company explorer's view, drawn from the figures
        counts = live_real.app.fundamentals.bench("real").store.counts()
        assert counts["companies"] == 0 and counts["observations"] == 0                            # calculating stored nothing
        assert page.problems == []
    finally:
        page.close()


def test_a_mistake_is_marked_on_its_cell_and_stops_the_calculation(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    try:
        _enter(page)
        _new_company(page)
        _paste(page)
        page.fill("input[data-line=total_assets][data-col='1']", "abc")
        page.click("#fundamentals .runbar button.btn.primary")
        page.wait_for_selector("[data-role=manual-issues]")
        assert "Not calculated" in page.inner_text("[data-role=manual-issues]")
        assert page.locator("[data-role=manual-ratios]").count() == 0
        assert "bad" in page.get_attribute("input[data-line=total_assets][data-col='1']", "class")
        assert "not a number" in page.get_attribute("input[data-line=total_assets][data-col='1']", "title")
        page.fill("input[data-line=total_assets][data-col='1']", "1700")
        page.fill("input[data-head=filing_date][data-col='0']", "2020-01-01")                   # published before the period it covers ended
        page.click("#fundamentals .runbar button.btn.primary")
        _wait_until(page, "!!document.querySelector('[data-role=manual-issues]') && document.querySelector('[data-role=manual-issues]').textContent.includes('before the period end')")
        assert "before the period end" in page.inner_text("[data-role=manual-issues]")
        assert page.problems == []
    finally:
        page.close()


def test_saving_stores_the_periods_in_the_real_database_and_they_can_be_corrected_and_removed(browser, live_real):  # noqa: F811
    page = _page_accepting(browser, live_real)
    try:
        _enter(page)
        _new_company(page)
        _paste(page)
        page.click("button[data-action=manual-save]")
        page.wait_for_selector("[data-role=manual-saved]")
        saved = page.inner_text("[data-role=manual-saved]")
        assert "TEST1" in saved and "Added" in saved
        store = live_real.app.fundamentals.bench("real").store
        assert store.counts()["companies"] == 1 and store.counts()["annual_periods"] == 3
        assert store.securities()["ticker"].tolist() == ["TEST1"] and store.securities()["source"].tolist() == ["manual"]
        # the company is now chosen, so a second save corrects instead of refusing a ticker that exists
        page.fill("input[data-line=revenue][data-col='0']", "1010")
        page.click("button[data-action=manual-save]")
        _wait_until(page, "[...document.querySelectorAll('[data-role=manual-saved] .tile')].some((t) => t.textContent.includes('Changed') && t.textContent.includes('1'))")
        assert page.locator("[data-role=manual-stored] tbody tr").count() == 3
        page.click("[data-role=manual-stored] tr[data-period='FY:2021-12-31'] button[data-action=manual-delete-period]")
        _wait_until(page, "document.querySelectorAll('[data-role=manual-stored] tbody tr').length === 2")
        assert store.counts()["annual_periods"] == 2
        assert page.problems == []
    finally:
        page.close()


def test_figures_typed_by_hand_or_pasted_into_the_grid_turn_the_calculate_button_on_and_work(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    try:
        _enter(page)
        _new_company(page)
        run = "#fundamentals .runbar button.btn.primary"
        page.click("button[data-action=manual-start]")                                             # three empty years
        assert page.is_disabled(run) and "figure" in page.inner_text("#fundamentals .runbar")         # a company, but nothing to calculate yet
        page.fill("input[data-line=revenue][data-col='2']", "1,500")
        page.fill("input[data-line=net_income][data-col='2']", "260")
        page.fill("input[data-line=total_assets][data-col='2']", "2000")
        assert not page.is_disabled(run)                                                           # typed, not pasted: the button followed
        # a block copied in Excel, pasted into a cell, lands from that cell down and to the right
        page.evaluate("""() => {
            const cell = document.querySelector("input[data-line=revenue][data-col='1']");
            const data = new DataTransfer(); data.setData('text/plain', '1200\\t1500\\n700\\t850');
            cell.dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true }));
        }""".replace("\\\\t", "\t").replace("\\\\n", "\n"))
        assert page.input_value("input[data-line=revenue][data-col='1']") == "1200" and page.input_value("input[data-line=revenue][data-col='2']") == "1500"
        assert page.input_value("input[data-line=cost_of_revenue][data-col='1']") == "700" and page.input_value("input[data-line=cost_of_revenue][data-col='2']") == "850"
        page.click(run)
        page.wait_for_selector("[data-role=manual-ratios]")
        assert "Gross margin" in page.inner_text("[data-role=manual-ratios]") and page.locator("[data-role=manual-stale]").is_hidden()
        assert "no publication date" in page.inner_text("[data-role=manual-issues]")                  # the dates were left empty: said, not assumed silently
        page.fill("input[data-line=net_income][data-col='2']", "270")                               # an edit after the calculation says the ratios shown are old
        assert page.locator("[data-role=manual-stale]").is_visible()
        assert page.problems == []
    finally:
        page.close()
