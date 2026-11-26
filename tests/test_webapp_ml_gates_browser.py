"""The research gates in a real browser: three cards that say what each gate is for, its status, the exact obstacle and kind of obstacle, and one next action; the verification control that runs a registered
requirement (P0-08, which has been "never run" because nothing in the page could start it) and shows its progress and its persisted result; the plan and prospective-evidence workflows that explain what
cannot be done now; and a phone-width layout without clipped text or sideways scrolling. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_assets_ml import finished
from tests.test_webapp_browser import browser, open_tab  # noqa: F401
from tests.test_webapp_workspaces_browser import _mode, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")
DEEP = {"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "explain": False, "name": "gate-demo",
        "model": {"kind": "nbeats", "params": {"epochs": 2, "max_rows": 800, "width": 8, "blocks": 1, "layers": 1, "batch_size": 128, "lookback": 8, "patience": 3, "num_threads": 1}}}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("gate_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("gate_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    exp = finished(app, app.extra_post["/api/ml/train"](DEEP))                                   # an exploratory run (its protected outcomes were read outside a plan) of a deep network
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, exp
    running.stop()


def open_controls(browser, running, width=1400, height=1000):  # noqa: F811
    pg = browser.new_page(viewport={"width": width, "height": height})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(running.url)
    pg.wait_for_selector(".chip")
    open_tab(pg, "ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "controls")
    _wait_until(pg, "document.querySelectorAll('#ml-results [data-role=gate-card]').length === 3", timeout=120.0)
    return pg


def test_three_cards_say_purpose_status_obstacle_and_next_action(browser, live):  # noqa: F811
    running, app, exp = live
    pg = open_controls(browser, running)
    cards = {c.get_attribute("data-gate"): c for c in pg.query_selector_all("#ml-results [data-role=gate-card]")}
    assert set(cards) == {"A", "B", "C"}
    a, b, c = (cards[k].inner_text() for k in "ABC")
    assert "Algorithm Verified" in a and "Is the algorithm implemented the way its documentation says" in a and "Pending" in a and "missing evidence" in a.lower() and "Never verified" in a
    assert "checks met" in a and "Run Gate A verification" in a
    assert "Strategy Validated" in b and "contaminated holdout" in b.lower() and "outside a frozen plan" in b and "Plan a new frozen evaluation" in b
    assert "Operationally Ready" in c and cards["C"].get_attribute("data-status") in ("blocked", "pending")
    assert "Read first." not in pg.locator("#ml-results").text_content(), "the repeated warning banners are gone"
    explain = pg.locator("[data-role=gates-explainer]")
    assert not explain.is_visible()
    pg.click("[data-role=gates-info]")
    text = explain.inner_text()
    assert explain.is_visible() and "Nothing about profit" in text and "never reclassified as untouched" in text and "Permission to trade automatically" in text
    cards["A"].query_selector("[data-role=gate-details] summary").click()
    assert pg.locator("tr[data-item='verification:P0-08']").count() == 1
    assert not pg.problems, pg.problems
    pg.close()


def test_the_p0_08_button_runs_the_registered_tests_shows_progress_and_turns_the_item_green(browser, live):  # noqa: F811
    running, app, exp = live
    pg = open_controls(browser, running)
    pg.locator("[data-gate=A] [data-role=gate-details] summary").click()
    row = pg.locator("tr[data-item='verification:P0-08']")
    assert "not yet produced" in row.inner_text()
    row.locator("button[data-requirement='P0-08'][data-action=run_requirement]").click()
    _wait_until(pg, "!document.querySelector('[data-role=verify-progress]').hidden", timeout=60.0)
    shown = pg.locator("[data-role=verify-progress]").inner_text()
    assert "P0-08" in shown or "Starting" in shown
    _wait_until(pg, "document.querySelector('[data-role=verify-summary]') !== null", timeout=300.0)
    assert "P0-08: pass" in pg.locator("[data-role=verify-summary]").inner_text() and "Verification finished" in pg.locator("[data-role=verify-summary]").inner_text()
    _wait_until(pg, "document.querySelector(\"tr[data-item='verification:P0-08']\") && document.querySelector(\"tr[data-item='verification:P0-08']\").dataset.state === 'ok'", timeout=120.0)
    pg.locator("[data-gate=A] [data-role=gate-details] summary").click()
    detail = pg.locator("tr[data-item='verification:P0-08']").inner_text()
    assert "21 passed, 0 failed" in detail and "test_ml_architecture_contracts.py" in detail
    pg.locator("tr[data-item='verification:P0-08'] button[data-action=history]").click()
    _wait_until(pg, "document.querySelector('[data-role=verify-history] table') !== null", timeout=30.0)
    hist = pg.locator("[data-role=verify-history]").inner_text()
    assert "Verification history: P0-08" in hist and "pass" in hist
    assert not pg.problems, pg.problems
    pg.close()


def test_the_plan_and_prospective_workflows_explain_what_cannot_be_done_now(browser, live):  # noqa: F811
    running, app, exp = live
    pg = open_controls(browser, running)
    pg.click("[data-gate=B] button[data-action=plan_new_look]")
    _wait_until(pg, "document.querySelector('[data-role=plan-readiness]') !== null && document.querySelector('[data-role=cannot-complete]') !== null", timeout=30.0)
    text = pg.locator("[data-role=plan-readiness]").inner_text()
    assert "first date a new look could protect" in text and "cannot be undone" in text and "Write the plan before looking" in text and "No validation policy is adopted" in text
    nxt = pg.locator("[data-gate=C] [data-role=gate-card]").count() or 1
    action = pg.locator("[data-gate=C] .gatecard-action button, [data-gate=C] .gatecard-action .gatecard-next").first.inner_text()
    assert action in ("Adopt a validation policy", "Finish Gates A and B first")
    assert not pg.problems, pg.problems
    pg.close()
    res = app.extra_get["/api/ml/prospective-readiness"]({"id": exp["id"]})
    assert "only by waiting" in res["cannot_complete_now"] and res["steps"][0]["step"] == "Adopt a validation policy"


def test_on_a_phone_the_cards_stack_without_clipped_text_or_sideways_scrolling(browser, live):  # noqa: F811
    running, app, exp = live
    pg = open_controls(browser, running, width=390, height=844)
    wide = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert wide <= 1, f"the page scrolls sideways by {wide}px"
    boxes = pg.evaluate("[...document.querySelectorAll('[data-role=gate-card]')].map(c => { const r = c.getBoundingClientRect(); return [r.left, r.right, c.scrollWidth - c.clientWidth]; })")
    assert len(boxes) == 3 and all(l >= -1 and r <= 391 and over <= 1 for l, r, over in boxes), boxes
    xs = {round(b[0]) for b in boxes}
    assert len(xs) == 1, "one column"
    assert pg.evaluate("[...document.querySelectorAll('.gate-badge')].every(b => b.scrollWidth <= b.clientWidth + 1)")
    assert not pg.problems, pg.problems
    pg.close()
