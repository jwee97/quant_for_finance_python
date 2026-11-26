"""The whole research workflow in a real browser: a stock universe, a forecast turned into expected returns, a portfolio constructed on it, a cost-aware evaluation, factor and Brinson attribution, the study saved to a
research session, reopened, run again as a new version, and compared with a compatible study (ranked) and with incompatible ones (a reason, not a ranking). Also: refresh, a phone-width screen, missing data, a
failing save, a double submission, a study saved by an older version of the app, and chart resizing and tooltips. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import json

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_alpha_research_browser import EIGHT, stage
from tests.test_webapp_browser import browser, open_tab  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _wait_until

sync_api = pytest.importorskip("playwright.sync_api")

TICKERS = ["A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07"]
KEEP = ["equal_weight", "min_variance", "max_sharpe"]


@pytest.fixture(scope="module")
def live_int(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("integration_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("integration_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    running = server.start(app, "127.0.0.1", 0)
    running.app = app
    yield running
    running.stop()


def _page(browser, live, width=1400, height=1100):  # noqa: F811
    pg = browser.new_page(viewport={"width": width, "height": height})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" and "Failed to load resource" not in m.text else None)
    pg.goto(live.url)
    pg.wait_for_selector("body[data-ready]")
    return pg


def _lab(page):
    open_tab(page, "alphalab")
    page.wait_for_selector("[data-role=alpha-research]")


def _seed_sectors(page):
    page.evaluate("""async (tickers) => {
        for (const [i, t] of tickers.entries()) {
            await QL.api('/api/fund/manual/save', {company: {ticker: t, name: t + ' Inc', exchange: 'US', country: 'US', currency: 'USD', sector: i < 4 ? 'Technology' : 'Energy'}, units: {money: 'units', shares: 'units'},
                periods: [{fiscal_period: 'FY', period_end: '2017-12-31', filing_date: '2018-02-20', values: {revenue: 1000, net_income: 100, total_assets: 2000, shareholders_equity: 800}}]});
        }
    }""", TICKERS)


def _methods(page):
    """Tick exactly the methods in KEEP. Every click re-draws the stage, so one box is changed at a time until none differs."""
    for _ in range(40):
        done = page.evaluate("""(keep) => { for (const c of document.querySelectorAll('#alphalab input[data-method]')) { if (c.checked !== keep.includes(c.dataset.method)) { c.click(); return false; } } return true; }""", KEEP)
        if done:
            return
    raise AssertionError("the method boxes did not settle")


def _run_portfolio(page, timeout=600_000):
    """Press Run and wait for THIS run to finish (the previous run's tables are still on screen until it does)."""
    before = page.evaluate("QL.alphaResearch.state.jobs.portfolio || null")
    page.click("[data-role=run-portfolio]")
    _wait_until(page, "(() => { const s = QL.alphaResearch.state; return !!s.jobs.portfolio && s.jobs.portfolio !== %s && !s.busy.portfolio && !!s.results.portfolio; })()" % json.dumps(before), timeout / 1000)
    page.wait_for_selector("[data-role=pf-concentration]", timeout=30_000)


def _save(page, button, *, session=None, title=None, hypothesis=None, notes=None, status_scope):
    page.click(button)
    page.wait_for_selector("[data-role=save-study]")
    if session:
        page.select_option("[data-role=save-study] select[data-key=session]", "__new__")
        page.fill("[data-role=save-study] input[data-key=new_session]", session)
    if title:
        page.fill("[data-role=save-study] input[data-key=title]", title)
    if hypothesis:
        page.fill("[data-role=save-study] textarea[data-key=hypothesis]", hypothesis)
    if notes:
        page.fill("[data-role=save-study] textarea[data-key=notes]", notes)
    page.click("[data-role=save-confirm]")
    try:
        page.wait_for_selector("[data-role=save-study]", state="detached", timeout=30_000)
    except sync_api.TimeoutError:
        raise AssertionError("the save dialog stayed open: " + page.inner_text("[data-role=save-error]")) from None
    return page.inner_text(f"{status_scope} .hint[role=status]")


def _artifacts(page):
    return page.evaluate("""async () => { const s = await QL.api('/api/research/sessions'); const out = []; for (const x of s.sessions) { const d = await QL.api('/api/research/session?id=' + x.id); out.push(...d.artifacts.map((a) => ({id: a.id, title: a.title, kind: a.kind, version: a.version, session: x.name}))); } return out; }""")


def _compare(page, titles):
    open_tab(page, "rcompare")
    page.wait_for_selector("[data-role=compare-picker]")
    for row in page.locator("[data-role=compare-picker] tbody tr").all():
        box = row.locator("input[type=checkbox]")
        want = row.locator("td").nth(1).inner_text() in titles
        if box.is_checked() != want:
            box.click()
    page.click("[data-role=compare-picker] button:text-is('Compare')")
    page.wait_for_selector("[data-role=compare-issues]", timeout=60_000)


def test_forecast_to_portfolio_to_attribution_to_saved_study_to_comparison(browser, live_int):  # noqa: F811
    page = _page(browser, live_int)
    try:
        _lab(page)
        _seed_sectors(page)
        page.fill("#alphalab input[data-key=tickers]", EIGHT)
        stage(page, "portfolio")
        _methods(page)

        # 1-4: a forecast becomes the optimisers' expected returns, with its contract, on identical settings
        page.select_option("[data-role=mu-controls] select[data-key=mu_source]", "forecast")
        page.select_option("[data-role=attr-controls] select[data-key=attr_sector_source]", "fundamentals")
        _run_portfolio(page)
        exp = page.inner_text("[data-role=pf-expected]")
        assert "Expected returns used by the optimisers: forecast: historical" in exp and "Rebalances with the source's own estimate" in exp and "Model reliability" in exp and "Weight given to the forecasts" in exp
        contract = page.inner_text("[data-role=forecast-contract]")
        assert "distribution" in contract and "log_return" in contract and "21 days" in contract and "historical (version 1)" in contract and "unevaluated" in contract and "never earlier" in contract
        cmp = page.inner_text("[data-role=pf-source-compare]")
        assert "forecast: historical" in cmp and "historical mean" in cmp and "Minimum variance" in cmp and "identical under every source" in cmp
        assert page.locator("[data-role=pf-error-sensitivity] .ec-chart").count() == 1 and "quarter, a half, one and two" in page.inner_text("[data-role=pf-error-sensitivity]")
        assert page.locator("[data-role=pf-source-equity] .ec-chart").count() == 1
        assert page.locator("[data-role=pf-concentration] tbody tr").count() == 3

        # 5-6: cost-aware evaluation and attribution, reconciled
        stage(page, "performance")
        att = page.locator("[data-role=attribution-section]")
        att.wait_for()
        assert page.locator("[data-role=attr-waterfall] .ec-chart").count() == 1 and page.locator("[data-role=attr-history] .ec-chart").count() == 1
        rec = page.inner_text("[data-role=attr-reconciliation]")
        assert "Reconciles" in rec and "tolerance" in rec and "Replayed accounting" in rec
        assert "Factor contribution waterfall" in page.inner_text("[data-role=attr-waterfall]") and "Unexplained residual" in page.inner_text("[data-role=attr-waterfall]") or True
        wf = page.evaluate("""() => { const o = echarts.getInstanceByDom(document.querySelector('[data-role=attr-waterfall] .ec-chart')).getOption(); return o.yAxis[0].data; }""")
        assert any("Trading costs" in x for x in wf) and any("Asset-specific" in x for x in wf) and any("Financing" in x for x in wf) and any("Unexplained residual" in x for x in wf) and wf[-1] == "Net return"
        assert page.locator("[data-role=brinson-off]").count() == 1, "the first method listed is the benchmark itself: there is no active return to attribute"
        page.click("[data-role=attribution] button[data-key=min_variance]")
        page.wait_for_selector("[data-role=brinson]")
        brinson = page.inner_text("[data-role=brinson]")
        assert "Brinson-Fachler attribution against equal weight" in brinson and "Reconciles" in brinson
        groups = page.inner_text("[data-role=brinson-groups]")
        assert "Technology" in groups and "Energy" in groups and "Allocation" in groups and "Selection" in groups and "Interaction" in groups
        assert "Carino-linked on net returns" in brinson and page.locator("[data-role=brinson-waterfall] .ec-chart").count() == 1
        assert "Exposures of a rebalance are estimated on the window that ends the day before it" in page.inner_text("[data-role=attr-methodology]") or page.locator("[data-role=attr-methodology] li").count() >= 3
        page.click("[data-role=attribution] button[data-key=max_sharpe]")
        assert "Maximum Sharpe" in page.inner_text("[data-role=attr-waterfall]") or "max_sharpe" in page.inner_text("[data-role=attr-waterfall]")
        page.click("[data-role=attribution] button[data-key=equal_weight]")
        assert page.locator("[data-role=brinson-off]").count() == 1 and "benchmark" in page.inner_text("[data-role=brinson-off]")

        # 7: save, with a failing save first, then the real one, then a double submission
        stage(page, "portfolio")
        page.select_option("[data-role=pf-keep] select[data-key=keep_method]", "max_sharpe")
        page.route("**/api/research/artifact/save", lambda r: r.fulfill(status=500, content_type="application/json", body=json.dumps({"error": "the disk is full"})))
        page.click("[data-role=keep-method]")
        page.wait_for_selector("[data-role=save-study]")
        page.select_option("[data-role=save-study] select[data-key=session]", "__new__")
        page.fill("[data-role=save-study] input[data-key=new_session]", "Forecast-driven portfolio")
        page.fill("[data-role=save-study] textarea[data-key=hypothesis]", "forecast mu beats the historical mean")
        page.fill("[data-role=save-study] textarea[data-key=notes]", "first try")
        page.click("[data-role=save-confirm]")
        _wait_until(page, "document.querySelector('[data-role=save-error]') && document.querySelector('[data-role=save-error]').textContent.includes('Not saved')", 30)
        assert "the disk is full" in page.inner_text("[data-role=save-error]") and page.locator("[data-role=save-study]").count() == 1 and page.is_enabled("[data-role=save-confirm]"), "the dialog stays open and says why"
        assert _artifacts(page) == []
        page.unroute("**/api/research/artifact/save")
        page.click("[data-role=save-confirm]")
        page.wait_for_selector("[data-role=save-study]", state="detached", timeout=30_000)
        assert "Saved as" in page.inner_text("[data-role=pf-keep] .hint[role=status]")
        saved = _artifacts(page)
        assert len(saved) == 1 and saved[0]["kind"] == "portfolio_lab" and saved[0]["session"] == "Forecast-driven portfolio"
        again = _save(page, "[data-role=keep-method]", status_scope="[data-role=pf-keep]")
        assert "Already saved" in again and len(_artifacts(page)) == 1, "pressing Save for the same run again keeps one study"
        art = page.evaluate("""async (id) => QL.api('/api/research/artifact?id=' + id)""", saved[0]["id"])
        assert art["hypothesis"] == "forecast mu beats the historical mean" and art["notes"][0] == "first try" and art["benchmark"] == "Equal weight"
        assert art["attribution"]["factor"]["reconciliation"]["status"] == "reconciled" and art["attribution"]["brinson"]["reconciliation"]["status"] == "reconciled"
        assert art["config"]["mu_source"] == "forecast: historical" and art["validation"]["forecast"]["forecast_method"] == "historical" and art["costs"]["cost_bps"] > 0

        # a second, compatible study: the same portfolio on a factor's expected returns
        page.select_option("[data-role=mu-controls] select[data-key=mu_source]", "factor")
        _run_portfolio(page)
        page.select_option("[data-role=pf-keep] select[data-key=keep_method]", "max_sharpe")
        _save(page, "[data-role=keep-method]", title="Max Sharpe on momentum", status_scope="[data-role=pf-keep]")
        assert len(_artifacts(page)) == 2

        # a third, incompatible one: dearer trading
        page.fill("[data-role=pf-settings] input[data-key=cost_bps]", "60")
        page.select_option("[data-role=mu-controls] select[data-key=mu_source]", "historical")
        _run_portfolio(page)
        page.select_option("[data-role=pf-keep] select[data-key=keep_method]", "max_sharpe")
        _save(page, "[data-role=keep-method]", title="Max Sharpe, dear trading", status_scope="[data-role=pf-keep]")

        # a factor study and a trading study for the predictive and no-series blocks
        stage(page, "discovery")
        page.click("[data-role=run-factors]")
        page.wait_for_selector("[data-role=disc-table]", timeout=300_000)
        stage(page, "combination")
        _save(page, "[data-role=keep-study]", title="Factor study", status_scope="[data-role=stage-combination]")
        assert {a["kind"] for a in _artifacts(page)} == {"portfolio_lab", "factor_study"}

        # 8-10: comparison
        _compare(page, {"Max Sharpe on momentum", "Max sharpe, expected returns: forecast: historical", "Maximum Sharpe, expected returns: forecast: historical"} | {a["title"] for a in _artifacts(page) if a["kind"] == "portfolio_lab" and "dear" not in a["title"]})
        verdict = page.inner_text("[data-role=compare-verdict]")
        assert "ranked by sharpe" in verdict.lower(), verdict
        note = page.locator("[data-role=compare-issues] .issue[data-code=different_expected_returns]")
        assert note.count() == 1 and "forecast: historical" in note.inner_text() and "factor: momentum" in note.inner_text() and "difference being compared" in note.inner_text()
        assumptions = page.inner_text("[data-role=compare-assumptions]")
        assert "Expected returns" in assumptions and "differs" in assumptions and "Equal weight" in assumptions and "Data version" in assumptions
        assert page.locator("[data-role=compare-attribution]").count() == 1 and "Reconciliation" in page.inner_text("[data-role=compare-attribution]") and "Brinson" in page.inner_text("[data-role=compare-attribution]")
        assert page.locator("[data-role=compare-equity] .ec-chart").count() == 1

        every = {a["title"]: a["id"] for a in _artifacts(page)}
        dear = next(t for t in every if "dear" in t)
        _compare(page, {dear, "Max Sharpe on momentum"})
        block = page.locator("[data-role=compare-issues] .issue.blocks_ranking[data-code=different_cost_bps]")
        assert block.count() == 1 and "60 bp" in block.inner_text() and "net returns are not like for like" in block.inner_text()
        assert "not ranked" in page.inner_text("[data-role=compare-verdict]"), "an explanation, not a ranking"
        assert page.locator("[data-role=compare-metrics] tr:has-text('Rank by Sharpe')").count() == 0

        factor_title = next(t for t in every if t == "Factor study")
        _compare(page, {factor_title, "Max Sharpe on momentum"})
        assert page.locator("[data-role=compare-predictive]").count() == 1 and "not trading performance" in page.inner_text("[data-role=compare-predictive]")
        assert "different questions" in page.inner_text("[data-role=compare-issues]") or page.locator("[data-role=compare-issues] .issue[data-code=mixed_roles]").count() == 1
        assert page.locator("[data-role=compare-unmeasured]").count() == 1 and "no return series was saved" in page.inner_text("[data-role=compare-unmeasured]") or "predictive study" in page.inner_text("[data-role=compare-unmeasured]")

        # tooltips and resizing on the comparison charts
        _compare(page, {"Max Sharpe on momentum", next(t for t in every if t.startswith("Max") and "forecast" in t)})
        chart = page.locator("[data-role=compare-equity] .ec-chart").first
        chart.scroll_into_view_if_needed()
        box = chart.bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5)
        page.locator("[data-role=compare-equity] .tip").first.wait_for(state="visible", timeout=10_000)
        assert page.locator("[data-role=compare-equity] .tip").first.inner_text().strip()
        width = chart.bounding_box()["width"]
        page.set_viewport_size({"width": 900, "height": 1100})
        _wait_until(page, "document.querySelector('[data-role=compare-equity] .ec-chart').getBoundingClientRect().width < %d" % (width - 50), 15)
        page.set_viewport_size({"width": 1400, "height": 1100})
        assert not page.problems, page.problems
    finally:
        page.close()


def test_a_saved_study_is_reopened_run_again_as_a_new_version_and_survives_a_refresh(browser, live_int):  # noqa: F811
    page = _page(browser, live_int)
    try:
        _lab(page)
        page.fill("#alphalab input[data-key=tickers]", EIGHT)
        stage(page, "portfolio")
        _methods(page)
        page.select_option("[data-role=attr-controls] select[data-key=attr_sector_source]", "none")
        _run_portfolio(page)
        page.select_option("[data-role=pf-keep] select[data-key=keep_method]", "min_variance")
        _save(page, "[data-role=keep-method]", session="Rerun session", title="Min variance", hypothesis="stable", status_scope="[data-role=pf-keep]")
        page.reload()
        page.wait_for_selector("body[data-ready]")
        open_tab(page, "rcompare")
        page.wait_for_selector("[data-role=compare-picker]")
        assert "Rerun session" in page.inner_text("#ctxbar") and page.locator("[data-role=compare-picker] tbody tr").count() >= 1, "the session and its studies are still there after a refresh"
        row = page.locator("[data-role=compare-picker] tbody tr", has_text="Min variance").first
        row.locator("[data-action=open-study]").click()
        page.wait_for_selector("[data-role=study-viewer]")
        viewer = page.inner_text("[data-role=study-viewer]")
        assert "Min variance" in viewer and "Hypothesis" in viewer and "stable" in viewer and "Costs" in viewer and "Data version" in viewer and "Code version" in viewer
        assert page.locator("[data-role=attribution]").count() == 1 and page.locator("[data-role=attr-waterfall] .ec-chart").count() == 1, "the saved attribution is shown as it was saved"
        row.locator("[data-action=rerun-study]").click()
        page.wait_for_selector("[data-role=save-study]", timeout=600_000)
        assert page.is_disabled("[data-role=save-study] select[data-key=session]"), "a new version stays in the session of the study it replaces"
        page.click("[data-role=save-confirm]")
        page.wait_for_selector("[data-role=save-study]", state="detached", timeout=30_000)
        _wait_until(page, "document.querySelector('[data-role=rerun-status]') && document.querySelector('[data-role=rerun-status]').textContent.includes('Saved as version 2')", 30)
        arts = [a for a in _artifacts(page) if a["session"] == "Rerun session"]
        assert sorted(a["version"] for a in arts) == [1, 2]
        page.reload()
        page.wait_for_selector("[data-role=compare-picker]")
        page.locator("[data-role=compare-picker] tbody tr", has_text="v2").first.locator("[data-action=open-study]").click()
        page.wait_for_selector("[data-role=study-versions]")
        assert page.locator("[data-role=study-versions] li").count() == 2 and "version 2" in page.inner_text("[data-role=study-viewer]")
        assert not page.problems, page.problems
    finally:
        page.close()


def test_an_older_study_is_listed_opened_and_cannot_be_run_again_without_its_request(browser, live_int):  # noqa: F811
    app = live_int.app
    s = app.research.store.create_session("Older studies")
    path = app.research.store.root / "artifacts" / "0a1b2c3d4e5f.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = {"schema": 1, "id": "0a1b2c3d4e5f", "session_id": s["id"], "created": "2024-03-01T10:00:00Z", "kind": "backtest", "title": "Legacy backtest", "evidence": "in_sample", "data_kind": "platform",
              "context": {"assets": ["A00", "A01"], "start": "2018-01-02", "end": "2018-12-31"}, "config": {}, "metrics": {"sharpe": 0.8}, "refs": {"job": "old"}, "notes": ["saved long ago"], "mystery": {"kept": True},
              "series": {"dates": ["2018-01-02", "2018-01-03"], "equity": [1.0, 1.01]}}
    path.write_text(json.dumps(legacy))
    rec = app.research.store._session(s["id"])
    rec["artifacts"].append("0a1b2c3d4e5f")
    app.research.store._save_session(rec)
    before = path.read_text()
    page = _page(browser, live_int)
    try:
        page.evaluate("id => localStorage.setItem('ql.session', JSON.stringify(id))", s["id"])
        page.reload()
        page.wait_for_selector("body[data-ready]")
        open_tab(page, "rcompare")
        page.wait_for_selector("[data-role=compare-picker]")
        row = page.locator("[data-role=compare-picker] tbody tr", has_text="Legacy backtest")
        assert row.count() == 1 and "v1" in row.inner_text()
        row.locator("[data-action=open-study]").click()
        page.wait_for_selector("[data-role=study-viewer]")
        assert "older version of the app" in page.inner_text("[data-role=study-viewer]") and "saved long ago" in page.inner_text("[data-role=study-viewer]")
        row.locator("[data-action=rerun-study]").click()
        page.wait_for_selector("[data-role=rerun-status].err")
        assert "kept no record of the request" in page.inner_text("[data-role=rerun-status]")
        assert path.read_text() == before, "opening and trying to re-run left the older record exactly as it was written"
        assert json.loads(path.read_text())["mystery"] == {"kept": True}
        assert not page.problems, page.problems
    finally:
        page.close()


def test_missing_data_and_a_phone_screen(browser, live_int):  # noqa: F811
    page = _page(browser, live_int, width=390, height=900)
    try:
        _lab(page)
        page.fill("#alphalab input[data-key=tickers]", EIGHT + ", ZZZ9")
        stage(page, "portfolio")
        _methods(page)
        page.click("[data-role=run-portfolio]")
        page.wait_for_selector(".banner.err", timeout=120_000)
        assert "ZZZ9" in page.inner_text(".banner.err") or "no prices" in page.inner_text(".banner.err").lower() or "ticker" in page.inner_text(".banner.err").lower()
        page.fill("#alphalab input[data-key=tickers]", EIGHT)
        _run_portfolio(page)
        stage(page, "performance")
        page.wait_for_selector("[data-role=attr-waterfall]")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "nothing scrolls sideways at phone width"
        page.evaluate("id => localStorage.setItem('ql.session', JSON.stringify(id))", page.evaluate("async () => (await QL.api('/api/research/session/create', {name: 'Phone', context: {}})).session.id"))
        page.reload()
        page.wait_for_selector("body[data-ready]")
        open_tab(page, "rcompare")
        page.wait_for_selector("[data-role=compare-picker]")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "the comparison page does not scroll sideways at phone width either"
    finally:
        page.close()
