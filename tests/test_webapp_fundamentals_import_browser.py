"""The Fundamentals tab's Report import and Data coverage views in a real browser: a filing and a spreadsheet are dropped in, read, reviewed, corrected and approved into the database with their source
shown, a repeat is recognised, the coverage grid counts what is held and missing, and both pages fit a phone."""

from __future__ import annotations

import pytest

from tests.test_fund_import import csv_report, ixbrl
from tests.test_webapp_browser import browser, open_tab  # noqa: F401
from tests.test_webapp_fundamentals_browser import _page, live_real  # noqa: F401
from tests.test_webapp_workspaces_browser import _wait_until

pytest.importorskip("playwright.sync_api")


def _open(page, mode):
    open_tab(page, "fundamentals")
    page.wait_for_selector("#fundamentals .panel")
    page.click(f"#fundamentals .labtop button[data-mode={mode}]")


def _company(page):
    return page.evaluate("""async () => (await QL.api('/api/fund/manual/save', {company: {ticker: 'RNI', name: 'Round Numbers Inc', exchange: 'US', country: 'US', currency: 'USD'}, units: {money: 'units', shares: 'units'},
        periods: [{fiscal_period: 'FY', period_end: '2021-12-31', filing_date: '2022-02-10', values: {revenue: 900e6, total_assets: 4000e6}}]})).security_id""")


def test_the_modes_are_in_the_tab_and_a_filing_is_read_reviewed_and_approved_with_its_source(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    try:
        page.click("#tab-fundamentals")
        page.wait_for_selector("#fundamentals .panel")
        names = page.locator("#fundamentals .labtop [role=tab]").all_inner_texts()
        assert names.index("Enter reports") < names.index("Report import") < names.index("Data coverage") < names.index("Factor lab")
        _company(page)
        page.click("#fundamentals .labtop button[data-mode=import]")
        page.wait_for_selector("[data-role=import-drop]")
        page.fill("[data-key=import-ticker]", "RNI")
        page.click("button[data-action=import-find]")
        _wait_until(page, "document.querySelector('[data-role=import-company]').textContent.includes('annual and')", 20)
        assert "1 annual" in page.inner_text("[data-role=import-company]")
        page.set_input_files("[data-role=import-file]", files=[{"name": "0000000001-24-000005.htm", "mimeType": "text/html", "buffer": ixbrl()},
                                                                {"name": "FY2023.csv", "mimeType": "text/csv", "buffer": csv_report()},
                                                                {"name": "copy.htm", "mimeType": "text/html", "buffer": ixbrl()}])
        _wait_until(page, "document.querySelector('[data-role=import-notice]') && document.querySelector('[data-role=import-notice]').textContent.includes('files added')", 30)
        assert "already imported" in page.inner_text("[data-role=import-notice]") or page.locator("[data-role=import-items] tbody tr").count() == 3
        page.click("#fundamentals .runbar button.btn.primary")
        page.wait_for_selector("[data-role=import-items] tbody tr[data-status=extracted], [data-role=import-items] tbody tr[data-status=needs_review]", timeout=60_000)
        rows = page.locator("[data-role=import-items] tbody tr")
        assert rows.count() == 3 and page.locator("[data-role=import-items] tbody tr[data-status=duplicate]").count() == 1
        page.locator("[data-role=import-items] tbody tr", has_text="0000000001-24-000005").first.click()
        page.wait_for_selector("[data-role=import-review]")
        review = page.locator("[data-role=import-review]")
        assert "Inline XBRL filing" in review.inner_text() and "0000000001-24-000005" in review.inner_text()
        revenue = review.locator("input[data-code=revenue][data-col='0']")
        assert float(revenue.input_value()) == 1200e6 and "us-gaap:Revenues" in revenue.get_attribute("title")
        review.locator("button[data-step='4']").click()                                                   # step 4: warnings
        assert "no filing date" in review.locator("[data-role=import-issues]").inner_text()
        review.locator("button[data-step='1']").click()                                                   # step 1: filing details
        review.locator("[data-role=import-filing-date]").first.fill("2024-02-08")
        review.locator("button[data-action=import-correct]").click()
        _wait_until(page, "!document.querySelector('[data-role=import-issues]').textContent.includes('no filing date')", 20)
        review = page.locator("[data-role=import-review]")
        review.locator("button[data-step='3']").click()                                                   # step 3: nothing is confirmed until the person says so
        try:
            page.wait_for_selector("[data-role=obs-table] tbody tr", timeout=8000)
        except Exception:
            pane = review.locator("[data-pane='3']").inner_text()[:500]
            raise AssertionError(f"{page.problems} :: {pane}")
        assert page.locator("[data-role=obs-table] tbody tr[data-status=UNCONFIRMED]").count() >= 5 and page.locator("[data-role=obs-table] tbody tr[data-status=CONFIRMED]").count() == 0
        review.locator("button[data-step='5']").click()                                                   # step 5: approval is refused until something is confirmed
        page.wait_for_selector("[data-role=plan-counts]")
        assert review.locator("button[data-action=import-approve]").is_disabled()
        review.locator("button[data-step='3']").click()
        review.locator("button[data-action=obs-confirm-suggested]").click()
        _wait_until(page, "document.querySelectorAll('[data-role=obs-table] tbody tr[data-status=CONFIRMED]').length >= 5", 20)
        review.locator("button[data-step='5']").click()
        page.wait_for_selector("[data-role=plan-table] tbody tr")
        review.locator("[data-role=plan-ack]").check()
        page.locator("[data-role=import-review] button[data-action=import-approve]").click()
        _wait_until(page, "document.querySelector('[data-role=import-items] tbody tr[data-status=saved]') !== null", 30)
        saved = page.evaluate("async () => (await QL.api('/api/fund/import/provenance?security_id=' + %d)).rows.length" % page.evaluate("async () => (await QL.api('/api/fund/companies?q=RNI')).companies[0].security_id"))
        assert saved >= 5
        assert not [p for p in page.problems if not p.startswith("HTTP 400")], page.problems
    finally:
        page.close()


def test_the_coverage_grid_shows_held_and_missing_years_and_quarters_and_the_problems(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    try:
        page.click("#tab-fundamentals")
        page.wait_for_selector("#fundamentals .panel")
        page.evaluate("""async () => QL.api('/api/fund/manual/save', {company: {ticker: 'GAP', name: 'Gap Co', exchange: 'US', country: 'US', currency: 'USD'}, units: {money: 'units', shares: 'units'},
            periods: [2019, 2021, 2022].map((y) => ({fiscal_period: 'FY', period_end: y + '-12-31', filing_date: (y + 1) + '-02-15', values: {revenue: 100 * (y - 2018), total_assets: 500}}))})""")
        page.click("#fundamentals .labtop button[data-mode=coverage]")
        page.wait_for_selector("[data-role=coverage-grid] .ec-chart", timeout=30_000)
        text = page.inner_text("#fundamentals-results")
        assert "Annual statements held" in text and "Quarters held" in text and "Awaiting review" in text and "GAP" in page.inner_text("#fundamentals-results")
        assert page.locator("button[data-action=coverage-backfill-one]").count() >= 1 and page.locator("button[data-action=coverage-backfill]").count() == 1
        page.locator("[data-role=coverage-view] button", has_text="Quarterly").click()
        page.wait_for_selector("[data-role=coverage-grid] .ec-chart")
        assert "Fiscal quarters and half-years by company" in page.inner_text("[data-role=coverage-grid]")
    finally:
        page.close()


def test_both_views_fit_a_phone(browser, live_real):  # noqa: F811
    page = _page(browser, live_real, width=390)
    try:
        for mode in ("import", "coverage"):
            _open(page, mode)
            page.wait_for_timeout(1200)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), mode
    finally:
        page.close()


def test_a_country_source_is_listed_with_what_it_permits_and_a_links_only_source_lists_nothing(browser, live_real):  # noqa: F811
    page = _page(browser, live_real)
    try:
        page.click("#tab-fundamentals")
        page.wait_for_selector("#fundamentals .panel")
        page.click("#fundamentals .labtop button[data-mode=import]")
        page.evaluate("document.querySelectorAll('#fundamentals details').forEach(d => d.open = true)")
        page.wait_for_selector("select[data-role=import-connector] option", state="attached")
        names = page.eval_on_selector_all("select[data-role=import-connector] option", "els => els.map(e => e.textContent)")
        assert any("EDINET" in n and "needs a key" in n for n in names) and any("CVM" in n and "needs a key" not in n for n in names) and any("HKEXnews" in n for n in names)
        page.evaluate("document.querySelectorAll('#fundamentals details').forEach(d => d.open = true)")
        page.select_option("select[data-role=import-connector]", "hkex")
        assert "Links and files only" in page.inner_text("#fundamentals .panel")
        page.click("button[data-action=import-connector-list]")
        page.wait_for_selector("[data-role=import-connector-filings]", timeout=30_000)
        assert "no interface for programs" in page.inner_text("[data-role=import-connector-filings]")
        page.select_option("select[data-role=import-connector]", "edinet")
        assert "EDINET_API_KEY" in page.inner_text("#fundamentals .panel")
    finally:
        page.close()
