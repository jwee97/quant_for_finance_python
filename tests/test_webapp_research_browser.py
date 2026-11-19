"""The redesigned shell and the research workspace in a real browser: the grouped sidebar and its deep links, the command palette, the phone drawer, the Overview, a research session carried from a backtest
to a regime estimate to a comparison, and the ways each of those refuses to say more than the evidence does. Needs Playwright and Chromium (skipped without them)."""

from __future__ import annotations

import re
from json import dumps as json_dumps

import pytest

from tests.test_webapp_browser import browser, open_tab  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _start, _wait_until

pytest.importorskip("playwright.sync_api")

NEW_PAGES = {"overview": "Overview", "saved": "Saved research", "experiments": "Recent experiments", "regime": "Market Regime Lab", "rcompare": "Research comparison", "jobs": "Background jobs", "data": "Data management", "settings": "Settings"}


@pytest.fixture(scope="module")
def live_research(tmp_path_factory):
    running = _start(tmp_path_factory, "research")
    yield running
    running.stop()


def _open(browser, live, width=1400, height=1000, path=""):  # noqa: F811
    pg = browser.new_page(viewport={"width": width, "height": height})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live.url + path)
    pg.wait_for_selector("body[data-ready]")
    return pg


@pytest.fixture()
def pg(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    yield page
    assert not page.problems, page.problems
    page.close()


def _hash(page):
    return page.evaluate("location.hash")


def _create_session(page, name="Study", assets="A00, A01, A02, A03, A04, A05"):
    page.click("#ctxbar .btn")
    page.fill("#ctxdlg input[data-key=name]", name)
    page.fill("#ctxdlg input[data-key=assets]", assets)
    page.click("#ctxdlg button[type=submit]")
    page.wait_for_selector(f"#ctx-select option:text-is('{name}')", state="attached")


# ------------------------------------------------------------------------------------------------------ navigation
def test_every_new_page_opens_from_the_sidebar_has_its_own_address_and_one_heading(pg):
    for tab, title in NEW_PAGES.items():
        pg.click(f"#tab-{tab}")
        pg.wait_for_selector(f"#view-{tab}:not([hidden])")
        assert _hash(pg) == f"#/{tab}"
        assert pg.get_attribute(f"#tab-{tab}", "aria-selected") == "true"
        assert pg.inner_text("#crumbs").replace("\n", " ").endswith(title) or title in pg.inner_text("#crumbs")
        assert pg.title().startswith(title)


def test_an_address_opens_its_page_and_the_back_button_returns_to_the_previous_one(browser, live_research):  # noqa: F811
    page = _open(browser, live_research, path="#/options")
    try:
        assert page.get_attribute("#tab-options", "aria-selected") == "true" and page.is_visible("#view-options")
        page.click("#tab-regime")
        page.wait_for_selector("#view-regime:not([hidden])")
        page.go_back()
        page.wait_for_selector("#view-options:not([hidden])")
        assert page.get_attribute("#tab-options", "aria-selected") == "true"
        page.goto(live_research.url + "#/nonsense")
        page.reload()
        page.wait_for_selector("body[data-ready]")
        assert page.is_visible("#view-backtest") or page.is_visible("#view-options"), "an unknown address falls back to a real page"
        assert not page.problems, page.problems
    finally:
        page.close()


def test_the_sidebar_groups_fold_and_the_rail_narrows_and_both_choices_survive_a_reload(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        names = page.eval_on_selector_all(".navgroup .navhead", "(e) => e.map((x) => x.textContent.trim())")
        assert names == ["Workspace", "Research", "Markets & Analytics", "System"]
        page.click("#grp-markets")
        assert page.get_attribute("#grp-markets", "aria-expanded") == "false" and not page.is_visible("#tab-fundamentals")
        page.click("#railtoggle")
        assert page.evaluate("document.body.classList.contains('rail')")
        page.reload()
        page.wait_for_selector("body[data-ready]")
        assert page.evaluate("document.body.classList.contains('rail')") and page.get_attribute("#railtoggle", "aria-pressed") == "true"
        page.click("#railtoggle")
        assert page.get_attribute("#grp-markets", "aria-expanded") == "false", "a folded group stays folded"
        assert page.get_attribute("#grp-research", "aria-expanded") == "true"
        page.evaluate("QL.showTab('fundamentals')")
        page.wait_for_selector("#view-fundamentals:not([hidden])")
        assert page.get_attribute("#grp-markets", "aria-expanded") == "true", "opening a page in a folded group opens the group"
        assert page.evaluate("JSON.parse(localStorage.getItem('ql.shell')).recent.length") >= 1
        assert not page.problems, page.problems
    finally:
        page.close()


def test_the_command_palette_finds_pages_by_name_and_by_what_they_do_and_the_chords_navigate(pg):
    pg.keyboard.press("Control+k")
    pg.wait_for_selector("#palette[open] input")
    pg.keyboard.type("regime")
    assert pg.inner_text("#pal-list li:first-child").startswith("Market Regime Lab")
    pg.keyboard.press("Enter")
    pg.wait_for_selector("#view-regime:not([hidden])")
    pg.keyboard.press("/")
    pg.wait_for_selector("#palette[open]")
    pg.keyboard.type("hedging")                                                                   # a word that is in no page's name, only in its search words
    assert "Options Lab" in pg.inner_text("#pal-list")
    pg.keyboard.press("Escape")
    assert not pg.is_visible("#palette[open]")
    _wait_until(pg, "document.activeElement === document.body")                                  # the focus leaves the closed dialog a moment after it closes; a key sent in between would go to its search box
    pg.keyboard.press("g")
    pg.keyboard.press("j")
    pg.wait_for_selector("#view-jobs:not([hidden])")
    pg.keyboard.press("?")
    pg.wait_for_selector("#keys[open]")
    assert "Ctrl K" in pg.inner_text("#keys")
    pg.keyboard.press("Escape")
    pg.keyboard.press("Control+k")
    pg.keyboard.type("zzzzqqq")
    assert "Nothing matches" in pg.inner_text("#pal-list")


def test_on_a_phone_the_sidebar_is_a_drawer_and_no_new_page_widens_the_screen(browser, live_research):  # noqa: F811
    page = _open(browser, live_research, width=390, height=800)
    try:
        assert not page.is_visible("#tab-overview") and page.is_visible("#navtoggle")
        page.click("#navtoggle")
        assert page.get_attribute("#navtoggle", "aria-expanded") == "true"
        page.wait_for_selector("#tab-overview", state="visible")                                   # the drawer slides in over a short transition
        page.keyboard.press("Escape")
        page.wait_for_selector("#tab-overview", state="hidden")
        for tab in NEW_PAGES:
            open_tab(page, tab)
            page.wait_for_selector(f"#view-{tab}:not([hidden])")
            page.wait_for_timeout(300)
            page.wait_for_selector("#tab-overview", state="hidden")                                  # choosing a page closes the drawer
            assert page.evaluate("document.documentElement.scrollWidth") <= 392, tab
        assert not page.problems, page.problems
    finally:
        page.close()


def test_at_a_tablet_width_the_pages_fit_too(browser, live_research):  # noqa: F811
    page = _open(browser, live_research, width=820, height=1000)
    try:
        for tab in ("overview", "backtest", "regime", "rcompare"):
            open_tab(page, tab)
            page.wait_for_timeout(300)
            assert page.evaluate("document.documentElement.scrollWidth") <= 822, tab
        assert not page.problems, page.problems
    finally:
        page.close()


# ------------------------------------------------------------------------------------------------------ the overview
def test_the_overview_labels_every_block_with_its_source_and_shows_no_made_up_numbers(pg):
    pg.click("#tab-overview")
    pg.wait_for_selector("[data-role=widgets] .widget")
    text = pg.inner_text("#overview")
    assert "undefined" not in text and "NaN" not in text and "null" not in text
    widgets = pg.locator("[data-role=widgets] .widget")
    assert widgets.count() == 11
    for i in range(widgets.count()):
        assert widgets.nth(i).locator(".ev").count() >= 1, "every widget carries a badge saying where it comes from"
    assert "not a live feed" in pg.inner_text("[data-widget=market]")
    regime_text = pg.inner_text("[data-widget=regime]")
    assert "no regime estimate" in regime_text or "filtered" in regime_text
    options_text = pg.inner_text("[data-widget=options]")                                       # another test may have saved an option backtest to this app
    assert "no option strategy" in options_text or "simulated option market" in options_text


def test_dashboard_widgets_can_be_hidden_moved_and_reset_and_the_layout_is_kept(pg):
    pg.click("#tab-overview")
    pg.wait_for_selector("[data-widget=market]")
    first = pg.eval_on_selector_all("[data-role=widgets] .widget", "(e) => e.map((x) => x.dataset.widget)")
    pg.click("[data-widget=market] button[aria-label='Move later: Market overview']")
    pg.wait_for_selector("[data-widget=market]")
    moved = pg.eval_on_selector_all("[data-role=widgets] .widget", "(e) => e.map((x) => x.dataset.widget)")
    assert moved[0] == first[1] and moved[1] == "market"
    pg.click("[data-widget=jobs] button[aria-label='Hide this widget: Job status']")
    pg.wait_for_selector("button:text-is('Show Job status')")
    pg.reload()
    pg.wait_for_selector("body[data-ready]")
    pg.wait_for_selector("[data-widget=market]")
    assert pg.locator("[data-widget=jobs]").count() == 0 and pg.eval_on_selector_all("[data-role=widgets] .widget", "(e) => e.map((x) => x.dataset.widget)")[:2] == moved[:2]
    pg.click("button:text-is('Reset layout')")
    pg.wait_for_selector("[data-widget=jobs]")
    assert pg.eval_on_selector_all("[data-role=widgets] .widget", "(e) => e.map((x) => x.dataset.widget)") == first


def test_a_widget_opens_the_module_it_summarises(pg):
    pg.click("#tab-overview")
    pg.wait_for_selector("[data-widget=regime]")
    pg.click("[data-widget=regime] .go")
    pg.wait_for_selector("#view-regime:not([hidden])")
    assert _hash(pg) == "#/regime"


# ------------------------------------------------------------------------------------------------------ a research session
def test_a_session_validates_its_context_and_is_still_open_after_a_reload(pg):
    pg.click("#ctxbar .btn")
    pg.fill("#ctxdlg input[data-key=name]", "Bad one")
    pg.fill("#ctxdlg input[data-key=assets]", "A00, NOT A TICKER!")
    pg.click("#ctxdlg button[type=submit]")
    _wait_until(pg, "document.querySelector('#ctxdlg .err').textContent.includes('ticker')")
    assert pg.is_visible("#ctxdlg[open]"), "a refused context keeps the form open and says what is wrong"
    pg.fill("#ctxdlg input[data-key=assets]", "A00, A01, A02, A03")
    pg.fill("#ctxdlg input[data-key=start]", "2021-01-04")
    pg.fill("#ctxdlg input[data-key=end]", "2020-01-04")
    pg.click("#ctxdlg button[type=submit]")
    _wait_until(pg, "document.querySelector('#ctxdlg .err').textContent.includes('before')")
    assert len(pg.problems) == 2 and all("400" in p for p in pg.problems), pg.problems          # the browser logs each refused request; these two are the ones provoked above
    pg.problems.clear()
    pg.fill("#ctxdlg input[data-key=start]", "")
    pg.fill("#ctxdlg input[data-key=end]", "")
    pg.click("#ctxdlg button[type=submit]")
    pg.wait_for_selector("#ctx-select option:text-is('Bad one')", state="attached")
    pg.reload()
    pg.wait_for_selector("body[data-ready]")
    _wait_until(pg, "document.querySelector('#ctx-select') && document.querySelector('#ctx-select').selectedOptions[0].textContent === 'Bad one'")
    assert "A00" in pg.inner_text("#ctxbar .ctxsummary")


def test_the_context_fills_a_tab_and_never_runs_anything(pg):
    _create_session(pg, "Fill", "A01, A02, A03")
    pg.click("#ctxbar button:text-is('Use in this tab')")
    _wait_until(pg, "document.querySelectorAll('#u-chips .chip').length === 3")
    assert pg.eval_on_selector_all("#u-chips .chip b", "(e) => e.map((x) => x.textContent)") == ["A01", "A02", "A03"]
    assert pg.text_content("#runcount") == "0", "using the context changes the form, it does not start a run"


# ------------------------------------------------------------------------------------------------------ the workflow
def test_a_backtest_is_saved_analysed_by_regime_and_compared_with_its_evidence_labels_kept(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        _create_session(page, "Workflow")
        page.click("#run")
        page.wait_for_selector("#runcount:text-is('1')", timeout=180_000)
        card = page.locator("[data-role=research-actions]")
        assert "historical simulation" in card.inner_text()
        card.locator("button:text-is('Save to session')").click()
        page.wait_for_selector("[data-role=research-actions] button:text-is('Saved')")
        # the server holds it, with its labels
        sid = page.evaluate("QL.research.active().id")
        arts = page.evaluate("(sid) => fetch('/api/research/session?id=' + sid, { headers: { 'X-Token': document.querySelector('meta[name=token]').content } }).then((r) => r.json())", sid)["artifacts"]
        assert [a["kind"] for a in arts] == ["backtest"] and arts[0]["evidence"] == "historical_simulation" and arts[0]["refs"]["job"]
        # Analyze by Market Regime: the lab opens on the same assets
        card.locator("button:text-is('Analyze by Market Regime')").click()
        page.wait_for_selector("#view-regime:not([hidden])")
        assert _hash(page).startswith("#/regime?result=")
        assert "A00" in page.input_value("#regime input[data-key=assets]")
        page.fill("#regime input[data-key=min_days]", "40")
        page.click("#regime button:text-is('Estimate regimes')")
        page.wait_for_selector("[data-role=regime-now]", timeout=240_000)
        now = page.inner_text("[data-role=regime-now]")
        assert "Filtered (causal)" in now and "as of" in now
        assert page.locator("[data-role=regime-retro]").count() == 1 and "Retrospective" in page.inner_text("[data-role=regime-retro]")
        assert "not tradeable" in page.inner_text("[data-role=regime-retro]").lower() or "cannot be traded" in page.inner_text("[data-role=regime-retro]").lower()
        for role in ("regime-transitions", "regime-durations", "regime-perf-next", "regime-perf-same", "regime-stability"):
            assert page.locator(f"[data-role={role}]").count() == 1, role
        assert "next-day" in page.inner_text("[data-role=regime-perf-next]") and "descriptive" in page.inner_text("[data-role=regime-perf-same]")
        # the saved backtest scored by the saved regimes
        page.wait_for_selector("[data-role=regime-by-result] button:text-is('Score by regime')")
        page.click("[data-role=regime-by-result] button:text-is('Score by regime')")
        page.wait_for_selector("[data-role=regime-by-result-table]")
        assert "historical simulation" in page.inner_text("[data-role=regime-by-result]").lower()
        # the comparison: a backtest and a regime estimate are different kinds of evidence, so there is no ranking
        page.click("#tab-rcompare")
        page.wait_for_selector("[data-role=compare-picker] input[type=checkbox]")
        for cb in page.locator("[data-role=compare-picker] input[type=checkbox]").all():
            cb.check()
        page.click("[data-role=compare-picker] button:text-is('Compare')")
        page.wait_for_selector("[data-role=compare-issues]")
        assert page.locator("[data-role=compare-issues] .issue.blocks_ranking").count() >= 1
        assert "not ranked" in page.inner_text("[data-role=compare-verdict]").lower()
        for role in ("compare-equity", "compare-drawdown", "compare-rolling-sharpe", "compare-distribution"):
            page.wait_for_selector(f"[data-role={role}] svg.chart")
        assert re.search(r"Shared window: \d{4}-\d\d-\d\d to \d{4}-\d\d-\d\d", page.inner_text("[data-role=compare-window]"))
        page.locator("[data-role=compare-equity] svg.chart rect[tabindex]").scroll_into_view_if_needed()
        box = page.locator("[data-role=compare-equity] svg.chart rect[tabindex]").bounding_box()      # one cursor for the comparison charts: a day read on one is shown on the others
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
        page.wait_for_selector("[data-role=compare-drawdown] .tip:not([hidden])")
        day = page.locator("[data-role=compare-equity] .tip .t").inner_text()
        assert page.locator("[data-role=compare-drawdown] .tip .t").inner_text() == day and re.fullmatch(r"\d{4}-\d\d-\d\d", day)
        page.mouse.move(0, 0)
        page.wait_for_selector("[data-role=compare-drawdown] .tip", state="hidden")
        with page.expect_download() as dl:
            page.click("[data-role=compare-tools] button:text-is('Download report (Markdown)')")
        report = open(dl.value.path(), encoding="utf-8").read()
        assert "Window shared by all results" in report and "does not describe future returns" in report and arts[0]["refs"]["job"] in report
        with page.expect_download() as dls:
            page.click("[data-role=compare-equity] button[aria-label='Save chart']")
            page.click("[data-role=compare-equity] button:text-is('Save as SVG')")
        svg = open(dls.value.path(), encoding="utf-8").read()
        assert svg.startswith("<svg") and "<path" in svg and "stroke:" in svg, "the exported chart carries its own colours"
        with page.expect_download() as dl2:
            page.click("[data-role=compare-tools] button:text-is('Download metrics (CSV)')")
        assert open(dl2.value.path(), encoding="utf-8").read().startswith("window,")
        # saved research keeps the session after a reload
        page.reload()
        page.wait_for_selector("body[data-ready]")
        page.click("#tab-saved")
        page.wait_for_selector("[data-role=session-results] table")
        assert page.locator("[data-role=session-results] tbody tr").count() == 2
        assert not page.problems, page.problems
    finally:
        page.close()


def test_changing_the_context_flags_a_saved_result_and_never_rewrites_it(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        _create_session(page, "Drift")
        page.click("#run")
        page.wait_for_selector("#runcount:text-is('1')", timeout=180_000)
        page.locator("[data-role=research-actions] button:text-is('Save to session')").click()
        page.wait_for_selector("[data-role=research-actions] button:text-is('Saved')")
        sid = page.evaluate("QL.research.active().id")
        token = "document.querySelector('meta[name=token]').content"
        before = page.evaluate(f"(sid) => fetch('/api/research/session?id=' + sid, {{ headers: {{ 'X-Token': {token} }} }}).then((r) => r.json())", sid)["artifacts"][0]
        page.click("#ctxbar button:text-is('Edit context')")
        page.fill("#ctxdlg input[data-key=assets]", "A00, A01")
        page.click("#ctxdlg button[type=submit]")
        _wait_until(page, "QL.research.active().context.assets.length === 2")
        page.evaluate("QL.showTab('compare'); QL.showTab('backtest')")
        page.wait_for_selector("[data-role=research-actions] .banner:has-text('context has changed')")
        text = page.inner_text("[data-role=research-actions] .banner")
        assert "assets" in text.lower() or "A00" in text
        assert page.locator("[data-role=research-actions] button:text-is('Apply the context and run again')").count() == 1
        after = page.evaluate(f"(sid) => fetch('/api/research/session?id=' + sid, {{ headers: {{ 'X-Token': {token} }} }}).then((r) => r.json())", sid)["artifacts"][0]
        assert after == before, "the saved result is exactly as it was made"
        assert not page.problems, page.problems
    finally:
        page.close()


def test_settings_change_the_theme_and_the_page_the_app_opens_on(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        assert page.get_attribute("html", "data-theme") == "dark", "dark is the default"
        page.click("#tab-settings")
        page.check("#settings input[name=theme][value=light]")
        assert page.get_attribute("html", "data-theme") == "light"
        page.select_option("#settings select", "overview")
        page.goto(live_research.url)
        page.wait_for_selector("body[data-ready]")
        page.wait_for_selector("#view-overview:not([hidden])")
        assert page.get_attribute("html", "data-theme") == "light"
        assert not page.problems, page.problems
    finally:
        page.close()


def test_the_jobs_page_lists_a_run_and_its_state(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        page.click("#run")
        page.wait_for_selector("#runcount:text-is('1')", timeout=180_000)
        page.click("#tab-jobs")
        page.wait_for_selector("#jobs table tbody tr")
        row = page.locator("#jobs tbody tr").first
        assert row.locator("td").nth(1).inner_text() == "backtest" and row.locator(".ev").inner_text() == "done"
        assert not page.problems, page.problems
    finally:
        page.close()


def test_a_backtest_result_opens_the_ml_and_options_labs_on_the_same_assets_and_says_the_hedge_is_a_simulation(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        _create_session(page, "Across", "A00, A01, A02, A03, A04, A05")
        page.click("#run")
        page.wait_for_selector("#runcount:text-is('1')", timeout=180_000)
        card = page.locator("[data-role=research-actions]")
        card.locator("button:text-is('Evaluate with ML')").click()
        page.wait_for_selector("#view-ml:not([hidden])")
        assert _hash(page) == "#/ml"
        _wait_until(page, "QL.lab('ml').form.universe === 'custom' && QL.lab('ml').form.tickers.includes('A00')")
        page.evaluate("QL.showTab('backtest')")
        page.wait_for_selector("[data-role=research-actions]")
        page.locator("[data-role=research-actions] button:text-is('Explore Options Hedge')").click()
        page.wait_for_selector("#view-options:not([hidden])")
        assert page.evaluate("QL.lab('options').mode") == "hedge"
        toasts = " ".join(page.locator(".toast").all_inner_texts())
        assert "not historical trades" in toasts, toasts
        assert not page.problems, page.problems
    finally:
        page.close()


def test_the_option_assumptions_of_the_context_reach_the_options_lab_when_asked_and_not_before(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        page.click("#ctxbar .btn")
        page.fill("#ctxdlg input[data-key=name]", "Options")
        page.fill("#ctxdlg input[data-key=assets]", "A01, A02")
        page.fill("#ctxdlg input[data-key=orate]", "3.5")
        page.fill("#ctxdlg input[data-key=odiv]", "1.25")
        page.click("#ctxdlg button[type=submit]")
        page.wait_for_selector("#ctx-select option:text-is('Options')", state="attached")
        page.click("#tab-options")
        page.wait_for_selector("#options-results .tile", timeout=90_000)
        assert page.evaluate("QL.lab('options').form.rate") == "", "an open session does not rewrite a tab on its own"
        page.click("#ctxbar button:text-is('Use in this tab')")
        _wait_until(page, "QL.lab('options').form.rate === '3.5' && QL.lab('options').form.dividend === '1.25'")
        assert page.evaluate("QL.lab('options').form.underlying") == "A01"
        assert not page.problems, page.problems
    finally:
        page.close()


def test_every_page_old_and_new_fits_a_phone_in_both_themes(browser, live_research):  # noqa: F811
    page = _open(browser, live_research, width=390, height=800)
    try:
        tabs = page.eval_on_selector_all("nav.tabs [role=tab]", "(e) => e.map((x) => x.id.slice(4))")
        assert len(tabs) == 24
        for theme in ("dark", "light"):
            page.evaluate(f"QL.setTheme('{theme}')")
            for tab in tabs:
                open_tab(page, tab)
                page.wait_for_timeout(250)
                assert page.evaluate("document.documentElement.scrollWidth") <= 392, (theme, tab)
        assert not page.problems, page.problems
    finally:
        page.close()


def test_an_option_strategy_backtest_can_be_saved_to_a_session_from_the_options_lab(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        _create_session(page, "Opt", "A01, A02")
        page.click("#tab-options")
        page.wait_for_selector("#options-results .tile", timeout=90_000)
        page.click("#view-options .tabsmall button[data-mode=backtest]")
        page.click("#options .runbar button.primary")
        page.wait_for_selector("[data-role=research-actions]", timeout=180_000)
        assert "simulated" in page.inner_text("[data-role=research-actions]").lower()
        page.click("[data-role=research-actions] button:text-is('Save to session')")
        page.wait_for_selector("[data-role=research-actions] button:text-is('Saved')")
        page.click("#tab-overview")
        page.wait_for_selector("[data-widget=options] .kpis")
        assert "simulated option market" in page.inner_text("[data-widget=options]")
        assert not page.problems, page.problems
    finally:
        page.close()


def test_a_strategy_setup_is_saved_as_a_configuration_and_loaded_back_without_running_anything(browser, live_research):  # noqa: F811
    page = _open(browser, live_research)
    try:
        _create_session(page, "Configs", "A00, A01, A02, A03, A04, A05")
        page.click("#run")
        page.wait_for_selector("#runcount:text-is('1')", timeout=180_000)
        original = page.evaluate("QL.S.models.map((m) => m.name)")
        page.once("dialog", lambda d: d.accept("My momentum setup"))
        page.locator("[data-role=research-actions] button:text-is('Save configuration')").click()
        page.wait_for_selector(".toast:has-text('My momentum setup')")
        page.evaluate("QL.S.models = [{ name: 'mean_reversion', params: {} }]")                 # the form moves on to something else
        page.click("#tab-saved")
        page.wait_for_selector("[data-role=session-configs] button:text-is('Load into Strategy Lab')")
        assert "My momentum setup" in page.inner_text("[data-role=session-configs]")
        page.click("[data-role=session-configs] button:text-is('Load into Strategy Lab')")
        page.wait_for_selector("#view-backtest:not([hidden])")
        _wait_until(page, f"JSON.stringify(QL.S.models.map((m) => m.name)) === '{json_dumps(original)}'")
        assert page.text_content("#runcount") == "1", "loading a configuration does not start a run"
        assert not page.problems, page.problems
    finally:
        page.close()
