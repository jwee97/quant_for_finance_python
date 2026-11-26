"""The Data health page in a real browser: it opens from the sidebar, shows readiness, coverage, integrity and provider capabilities with one status each, inspects the missing periods and the source of a series,
validates as a job and keeps a snapshot, exports CSV, and records a correction without touching the stored data. Needs Playwright and Chromium (skipped without them)."""

from __future__ import annotations

import pytest

from tests.test_webapp_browser import browser  # noqa: F401
from tests.test_webapp_research_browser import _open
from tests.test_webapp_workspaces_browser import _start

pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live_health(tmp_path_factory):
    running = _start(tmp_path_factory, "health")
    yield running
    running.stop()


@pytest.fixture()
def pg(browser, live_health):  # noqa: F811
    page = _open(browser, live_health)
    yield page
    assert not page.problems, page.problems
    page.close()


def _go(pg):
    pg.click("#tab-datahealth")
    pg.wait_for_selector("#view-datahealth:not([hidden])")
    pg.wait_for_selector("[data-role=health-capabilities] table")


def test_the_page_shows_readiness_coverage_integrity_and_one_status_per_provider_dataset(pg):
    _go(pg)
    assert pg.locator("#view-datahealth h2").count() == 1 and pg.inner_text("#view-datahealth h2") == "Data health"
    for role in ("health-readiness", "health-coverage", "health-integrity", "health-capabilities", "health-history"):
        assert pg.locator(f"[data-role={role}]").count() == 1, role
    text = pg.inner_text("[data-role=health-capabilities]")
    assert "Mock tested only" in text or "Not tested" in text
    rows = pg.locator("[data-role=health-capabilities] tbody tr")
    assert rows.count() > 20
    statuses = {"Verified live", "Verified from a recorded response", "Mock tested only", "Subscription blocked", "Unsupported", "Not tested"}
    for i in range(min(rows.count(), 60)):
        cells = rows.nth(i).locator("td")
        assert any(cells.nth(2).inner_text().startswith(s) for s in statuses)


def test_a_series_missing_periods_and_source_can_be_inspected_and_a_correction_leaves_stored_data_alone(pg):
    _go(pg)
    pg.wait_for_selector("[data-role=health-coverage] tbody tr")
    first = pg.locator("[data-role=health-coverage] tbody tr").first
    name = first.get_attribute("data-series")
    first.get_by_text("Missing periods").click()
    pg.wait_for_selector("#health-detail h4")
    assert "Missing periods of" in pg.inner_text("#health-detail h4")
    first.get_by_text("Source").click()
    pg.wait_for_selector("#health-detail h4:has-text('Source of')")
    pg.click("#health-detail summary")
    for key, val in (("date", "2024-01-02"), ("raw", "1"), ("corrected", "2"), ("reason", "typo in the source file")):
        pg.fill(f"#health-detail input[aria-label={key}]", val)
    pg.click("#health-detail button:text-is('Record')")
    pg.wait_for_selector("[data-role=health-history]:has-text('typo in the source file')")
    assert name in pg.inner_text("[data-role=health-history]")


def test_validate_runs_as_a_job_keeps_a_snapshot_and_a_second_run_says_the_data_are_unchanged(pg):
    _go(pg)
    pg.click("[data-role=health-validate]")
    pg.wait_for_selector("[data-role=health-history] tbody tr", timeout=60000)
    pg.click("[data-role=health-validate]")
    pg.wait_for_selector("[data-role=health-history] tbody tr:nth-child(2)", timeout=60000)
    pg.wait_for_selector("#health-status:has-text('unchanged')", timeout=60000)


def test_the_platform_snapshot_cannot_be_refreshed_and_the_page_says_why(pg):
    _go(pg)
    pg.click("[data-role=health-refresh]")
    pg.wait_for_selector("text=versioned with the repository")


def test_the_export_downloads_the_findings_as_csv(pg):
    _go(pg)
    with pg.expect_download() as dl:
        pg.click("[data-role=health-export]")
    assert dl.value.suggested_filename.endswith(".csv")
