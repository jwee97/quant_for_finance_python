"""The Options Strategy Studio, the Crypto workflows and the data-mode switch in a real browser. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from tests.test_webapp_browser import browser  # noqa: F401
from tests.test_webapp_workspaces_browser import _mode, live_ws, ws  # noqa: F401

sync_api = pytest.importorskip("playwright.sync_api")


def run(ws):
    ws.wait_for_selector("#options .runbar button.primary:not([disabled])", timeout=60_000)
    ws.click("#options .runbar button.primary")


def open_options(ws):
    ws.click("#tab-options")
    ws.wait_for_selector("#options-results .tile", timeout=90_000)


def test_the_options_studio_is_four_workflows_and_the_scanner_lists_leads_with_quote_quality(ws):
    open_options(ws)
    groups = {g.locator(".tabgroup-label").text_content(): [b.inner_text() for b in g.locator("button").all()] for g in ws.locator("#view-options .tabgroup").all()}
    assert groups["Strategy Studio"][:5] == ["Opportunity scanner", "Ideas from a view", "Strategy builder", "Scenario laboratory", "Trade validation"]
    assert "Synthetic validation" in groups["Portfolio and validation"]
    _mode(ws, "options", "scan")
    run(ws)
    ws.wait_for_selector("#options-results .card h3:has-text('Leads')", timeout=120_000)
    text = ws.inner_text("#options-results")
    assert "Implied / realised" in text or "against its own history" in text
    assert ws.locator("#options-results [data-role=data-mode]").count() == 0, "the switch lives in the setup panel, not in the results"


def test_a_position_of_real_contracts_is_priced_revalued_checked_and_can_be_saved(ws):
    open_options(ws)
    _mode(ws, "options", "builder")
    ws.wait_for_selector("#options select[aria-label^='expiry']", state="attached", timeout=60_000) if ws.locator("#options select[aria-label^='expiry']").count() else None
    ws.click("#options button:has-text('+ Option')")
    ws.wait_for_selector("#options input[aria-label^='strike']", timeout=30_000)
    strike = ws.locator("#options input[aria-label^='strike']").first
    spot = float(ws.evaluate("() => (window.QL.lab('options').spot || 100)"))
    strike.fill(str(round(spot)))
    ws.locator("#options input[aria-label^='quantity']").first.fill("1")
    ws.click("#options button:has-text('+ Option')")
    ws.locator("#options input[aria-label^='strike']").nth(1).fill(str(round(spot) + 5))
    ws.locator("#options input[aria-label^='quantity']").nth(1).fill("-1")
    ws.click("#options .runbar button.primary")
    ws.wait_for_selector("#options-results .card h3:has-text('Legs')", timeout=120_000)
    text = ws.inner_text("#options-results")
    assert "Crossing every spread" in text and "Maximum loss" in text and ws.locator("#options-results [data-role=save-experiment]").count() == 1
    _mode(ws, "options", "scenarios")
    ws.click("#options .runbar button.primary")
    ws.wait_for_selector("#options-results .card h3:has-text('Scenarios and where their P&L comes from')", timeout=120_000)
    assert "Residual" in ws.inner_text("#options-results") and ws.locator("#options-results [aria-label*='Spot and volatility'], #options-results h3:has-text('Spot and volatility')").count() >= 1
    _mode(ws, "options", "tradecheck")
    ws.click("#options .runbar button.primary")
    ws.wait_for_selector("#options-results .card h3:has-text('Checks')", timeout=120_000)
    assert "every contract is listed on this chain" in ws.inner_text("#options-results")


def test_ideas_rank_structures_from_a_view_and_load_their_legs_into_the_builder(ws):
    open_options(ws)
    _mode(ws, "options", "ideas")
    run(ws)
    ws.wait_for_selector("#options-results .card h3:has-text('Ranked by profit if you are right')", timeout=120_000)
    assert "If wrong" in ws.inner_text("#options-results")
    ws.locator("#options-results button:has-text('Use these legs')").first.click()
    ws.wait_for_selector("#options input[aria-label^='strike']", timeout=30_000)
    assert ws.locator("#options input[aria-label^='strike']").count() >= 1 and ws.locator("#options .tabsmall button[aria-selected=true]").inner_text() == "Strategy builder"


def test_the_data_mode_switch_refuses_simulated_data_and_offers_to_allow_it(ws):
    open_options(ws)
    ws.click("#options [data-role=data-mode] button[data-mode=observed]")
    ws.wait_for_selector("#options-results .banner.err", timeout=60_000)
    assert "observed-only mode" in ws.inner_text("#options-results")
    ws.problems.clear()                                                                                          # the refusal is a 400, which the console reports
    ws.click("#options-results [data-role=allow-model]")
    ws.wait_for_selector("#options-results .tile", timeout=90_000)
    assert ws.locator("#options [data-role=data-mode] button[data-mode=model]").get_attribute("aria-pressed") == "true"
    ws.wait_for_timeout(500)
    ws.problems[:] = [p for p in ws.problems if "400" not in p]                                                  # the refusal is a 400, which the console reports


def test_the_crypto_tab_has_five_workflows_and_its_new_views_say_what_is_not_stored(ws):
    ws.click("#tab-crypto")
    ws.wait_for_selector("#crypto-results .tile", timeout=90_000)
    labels = [b.text_content() for b in ws.locator("#view-crypto .tabsmall button[data-workflow]").all()]
    assert labels == ["Market Overview", "Derivatives Intelligence", "Liquidity & Arbitrage", "Strategy Laboratory", "Portfolio Risk"]
    assert ws.locator("#view-crypto .tabsmall button[data-mode=funding]").count() == 0 and ws.locator("#view-crypto .tabsmall button[data-mode=basis]").count() == 0, "the merged views are not destinations of their own"
    assert "not stored by the platform" in ws.inner_text("#crypto-results") and "24h return" in ws.inner_text("#crypto-results")
    _mode(ws, "crypto", "derivs")
    ws.wait_for_selector("#crypto-results .card h3:has-text('Funding by coin and venue')", timeout=120_000)
    ws.wait_for_selector("#crypto-results .card h3:has-text('Funding by venue and coin')", timeout=60_000)
    text = ws.inner_text("#crypto-results")
    assert "Open interest by exchange" in text and "Liquidations" in text and "Dated futures term structure" in text
    _mode(ws, "crypto", "carry")
    ws.wait_for_selector("#crypto-results h3:has-text('Funding reversal and margin risk')", timeout=120_000)
    assert "Residual" in ws.inner_text("#crypto-results") and ws.locator("#crypto-results [data-role=save-experiment]").count() == 1
    ws.wait_for_selector("#crypto-results .card h3:has-text('Basis of the perpetual over spot')", timeout=60_000)
    _mode(ws, "crypto", "replay")
    ws.click("#crypto .runbar button.primary")
    ws.wait_for_selector("#crypto-results .card h3:has-text('Recorded history')", timeout=60_000)
    assert "Nothing is recorded yet" in ws.inner_text("#crypto-results") or "Replay" in ws.inner_text("#crypto-results")
    ws.problems[:] = [p for p in ws.problems if "400" not in p]
    _mode(ws, "crypto", "liq")
    ws.click("#crypto .runbar button.primary")
    ws.wait_for_selector("#crypto-results h3:has-text('Shocks')", timeout=60_000)
    assert "Liquidated on a fall of" in ws.inner_text("#crypto-results")
