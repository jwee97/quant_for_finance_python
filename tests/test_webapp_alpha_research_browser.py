"""The five-stage Alpha research workflow in a real browser: Discovery, Validation, Combination, Portfolio construction, Performance and attribution. Each stage is run from its own button, draws what the server
returned (not a recomputation), and a result whose sample or settings no longer match the form is marked stale instead of being drawn next to the new ones. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from tests.test_webapp_browser import browser, open_tab  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _start, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")

EIGHT = "A00, A01, A02, A03, A04, A05, A06, A07"


@pytest.fixture(scope="module")
def live_alpha(tmp_path_factory):
    running = _start(tmp_path_factory, "alpharesearch")
    yield running
    running.stop()


@pytest.fixture()
def page(browser, live_alpha):  # noqa: F811
    pg = browser.new_page(viewport={"width": 1400, "height": 1100})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live_alpha.url)
    pg.wait_for_selector("body[data-ready]")
    open_tab(pg, "alphalab")
    pg.wait_for_selector("[data-role=alpha-research]")
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def stage(page, key):
    page.click(f"[data-role=alpha-research] button[data-stage={key}]")
    page.wait_for_selector(f"[data-role=stage-{key}]")


def test_the_breadth_guard_is_said_before_the_run_and_the_run_is_refused(page):
    page.fill("#alphalab input[data-key=tickers]", "A00, A01, A02")
    page.wait_for_selector("[data-role=breadth-guard]")
    text = page.inner_text("[data-role=breadth-guard]")
    assert "at least 6 tickers" in text and "3 assets" in text and "never scored as zero" in text
    assert page.is_disabled("[data-role=run-factors]"), "the run button is off while the cross-section is too thin"
    page.fill("#alphalab input[data-key=tickers]", EIGHT)
    assert page.locator("[data-role=breadth-guard]").count() == 0 and page.is_enabled("[data-role=run-factors]")


def test_all_five_stages_run_in_order_and_draw_what_the_server_found(page):
    page.fill("#alphalab input[data-key=tickers]", EIGHT)
    assert all(page.get_attribute(f"button[data-stage={k}]", "data-state") == "not run" for k in ("discovery", "portfolio", "performance"))
    page.click("[data-role=run-factors]")
    page.wait_for_selector("[data-role=disc-table]", timeout=240_000)
    disc = page.inner_text("[data-role=disc-table]")
    assert "Median assets per date" in disc and "Rank persistence" in disc and "Drift p" in disc and page.locator("[data-role=disc-table] tbody tr").count() == 4
    page.wait_for_selector("[data-role=disc-corr] canvas, [data-role=disc-corr] svg", timeout=30_000)
    assert page.get_attribute("button[data-stage=validation]", "data-state") == "current"

    stage(page, "validation")
    val = page.inner_text("[data-role=val-table]")
    assert "p Holm" in val and "p BH" in val and "IC after beta and momentum" in val and "IC by thirds" in val
    assert page.locator("[data-role=val-table] tbody tr").count() == 4
    assert "Does clipping the outliers change the answer" in page.inner_text("[data-role=val-sensitivity]") and "beta, momentum_12_1" in page.inner_text("[data-role=val-sensitivity]")
    assert page.locator("[data-role=val-ic] canvas, [data-role=val-ic] svg").count() >= 1

    stage(page, "combination")
    comb = page.inner_text("[data-role=comb-table]")
    assert "equal weight" in comb and "ridge" in comb and "orthogonalised" in comb and "Beats best single by" in comb and "ic weighted" in comb
    assert "nothing was fitted" in page.inner_text("[data-role=stage-combination]")

    stage(page, "portfolio")
    page.click("[data-role=run-portfolio]")
    page.wait_for_selector("[data-role=pf-concentration]", timeout=300_000)
    pf = page.inner_text("[data-role=pf-concentration]")
    for head in ("Effective N", "Largest weight", "Weight turnover / rebalance", "Moves with noisy returns", "Moves with another covariance", "Cost drag / year"):
        assert head in pf, head
    assert page.locator("[data-role=pf-concentration] tbody tr").count() == 6, "the six default methods, all on the same windows, costs and cap"
    assert "same lookback" in page.inner_text("[data-role=pf-head]") and "cap" in page.inner_text("[data-role=pf-head]")
    for role in ("pf-effn", "pf-sensitivity", "pf-frontier"):
        page.wait_for_selector(f"[data-role={role}] canvas, [data-role={role}] svg", timeout=30_000)

    stage(page, "performance")
    page.click("[data-role=run-performance]")
    page.wait_for_selector("[data-role=perf-waterfall]", timeout=240_000)
    attr = page.inner_text("[data-role=perf-attribution]")
    assert "Cost drag / year" in attr and "5 bp" in page.inner_text("[data-role=perf-head]") + attr or "bp" in attr
    assert "Four ways to trade the same alpha" in page.inner_text("[data-role=perf-policies]")
    assert all(page.get_attribute(f"button[data-stage={k}]", "data-state") == "current" for k in ("discovery", "validation", "combination", "portfolio", "performance"))


def test_a_result_from_another_sample_or_other_settings_is_marked_and_not_drawn_with_the_new_ones(page):
    page.fill("#alphalab input[data-key=tickers]", EIGHT)
    page.click("[data-role=run-factors]")
    page.wait_for_selector("[data-role=disc-table]", timeout=240_000)
    page.fill("#alphalab input[data-key=horizon]", "10")                       # other settings, the same sample
    stage(page, "validation")
    assert page.get_attribute("button[data-stage=validation]", "data-state") == "settings changed"
    assert "other settings" in page.inner_text("[data-role=stale-factors]") and page.locator("[data-role=val-table]").count() == 1
    page.fill("#alphalab input[data-key=tickers]", "A00, A01, A02, A03, A04, A05, A06")      # another sample
    stage(page, "validation")
    assert page.get_attribute("button[data-stage=validation]", "data-state") == "different sample"
    assert "different sample" in page.inner_text("[data-role=stale-factors]") and page.locator("[data-role=val-table]").count() == 0, "it is not drawn next to results of the current sample"
    page.fill("#alphalab input[data-key=tickers]", EIGHT)
    page.fill("#alphalab input[data-key=horizon]", "5")
    stage(page, "validation")
    assert page.get_attribute("button[data-stage=validation]", "data-state") == "current" and page.locator("[data-role=val-table]").count() == 1, "back on the sample it was run on, it is current again"


def _seed_fundamentals(page, tickers):
    """Annual statements for 2014 to 2018, filed on 20 February of the next year, with profitability that differs by company and year, through the Fundamentals tab's own endpoint."""
    page.evaluate("""async (tickers) => {
        for (const [i, t] of tickers.entries()) {
            await QL.api('/api/fund/manual/save', {company: {ticker: t, name: t + ' Inc', exchange: 'US', country: 'US', currency: 'USD'}, units: {money: 'units', shares: 'units'},
                periods: [2014, 2015, 2016, 2017, 2018].map((y) => ({fiscal_period: 'FY', period_end: y + '-12-31', filing_date: (y + 1) + '-02-20',
                    values: {revenue: 1000 + 20 * i, net_income: 50 + 10 * i + 3 * (y - 2014) * (i % 2 ? 1 : -1), total_assets: 2000, shareholders_equity: 800 + 15 * i, gross_profit: 400 + 12 * i}}))});
        }
    }""", tickers)


def test_fundamental_factors_enter_after_the_point_in_time_check_and_studies_are_kept_in_a_research_session(page):
    _seed_fundamentals(page, ["A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07"])
    page.fill("#alphalab input[data-key=tickers]", EIGHT)
    page.wait_for_selector("[data-role=fundamental-factors] input[data-fundamental=roe]", timeout=30_000)
    page.check("[data-role=fundamental-factors] input[data-fundamental=roe]")
    page.check("[data-role=fundamental-factors] input[data-fundamental=net_margin]")
    assert "2 fundamental" in page.inner_text("[data-role=research-context]")
    page.click("[data-role=run-factors]")
    page.wait_for_selector("[data-role=disc-table]", timeout=240_000)
    disc = page.inner_text("[data-role=disc-table]")
    assert "Fundamental: ROE" in disc or "Fundamental: Return on equity" in disc or "Fundamental:" in disc
    assert page.locator("[data-role=disc-table] tbody tr").count() == 6, "four price signals and two fundamental ones"
    stage(page, "validation")
    pit = page.inner_text("[data-role=val-pit]")
    assert "point-in-time check that admitted them" in pit and "Ready" in pit and "Companies with a value" in pit and "8 of 8" in pit
    assert page.locator("[data-role=val-pit] .banner[data-status]").get_attribute("data-status") == "ready"
    assert "Fundamental:" in page.inner_text("[data-role=val-table]") and "p Holm" in page.inner_text("[data-role=val-table]")
    stage(page, "combination")
    page.click("[data-role=keep-study]")
    page.wait_for_selector("[data-role=save-study]")
    page.click("[data-role=save-confirm]")
    _wait_until(page, "[...document.querySelectorAll('[data-role=stage-combination] .hint[role=status]')].some((e) => e.textContent.startsWith('Saved as'))", 30)
    kept = page.evaluate("""async () => { const s = await QL.api('/api/research/sessions'); const d = await QL.api('/api/research/session?id=' + s.sessions[0].id); const a = await QL.api('/api/research/artifact?id=' + d.artifacts[0].id); return [a.kind, a.config.horizon, a.config.uses_fundamentals, a.config.fundamental.pit_status, a.metrics.n_tested]; }""")
    assert kept == ["factor_study", 5, True, "ready", 6], kept
    stage(page, "portfolio")
    page.click("[data-role=run-portfolio]")
    page.wait_for_selector("[data-role=pf-keep]", timeout=300_000)
    page.click("[data-role=keep-method]")
    page.wait_for_selector("[data-role=save-study]")
    page.click("[data-role=save-confirm]")
    _wait_until(page, "[...document.querySelectorAll('[data-role=pf-keep] .hint[role=status]')].some((e) => e.textContent.startsWith('Saved as'))", 30)
    kinds = page.evaluate("""async () => { const s = await QL.api('/api/research/sessions'); const d = await QL.api('/api/research/session?id=' + s.sessions[0].id); return d.artifacts.map((a) => a.kind).sort(); }""")
    assert kinds == ["factor_study", "portfolio_lab"], kinds
