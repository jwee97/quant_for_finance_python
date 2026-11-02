"""The constrained selection and the adaptive retraining in a real browser (P1-15, P2-17, P3-02).

The Select view chooses among saved experiments on their validation rows and shows the eligibility, the ranking, the Pareto comparison and the simplicity step with what it does and does not claim; the Search
form offers the selection as a switch; the research record shows the retraining triggers, runs the challenger search of an approved proposal and shows its evidence, the versions and a rollback. Skipped when
Playwright or Chromium is missing.
"""

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
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _mode, _open_all, _page, _run, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "explain": False, "mode": "development"}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("selection_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=7, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("selection_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    ids = {}
    for name, model in (("ridge", {"kind": "ridge", "params": {"alpha": 10.0}}), ("lasso", {"kind": "lasso", "params": {"alpha": 0.001}}), ("forest", {"kind": "random_forest", "params": {"n_estimators": 20, "max_depth": 3}})):
        ids[name] = finished(app, app.extra_post["/api/ml/train"]({**BASE, "model": model, "name": f"select-{name}"}))["id"]
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, ids
    running.stop()


def _until_shown(pg, needle, timeout=120.0):
    """Wait for text in the results, and when it never comes say what the page showed instead."""
    try:
        _wait_until(pg, f"document.querySelector('#ml-results').textContent.includes({needle!r})", timeout=timeout)
    except AssertionError:
        raise AssertionError(f"{needle!r} never appeared; the results said: {pg.locator('#ml-results').text_content()[:700]!r}; page problems: {pg.problems}") from None


def _open(browser, running, exp_id=None):  # noqa: F811
    pg = _page(browser, running)
    if exp_id:
        pg.evaluate("id => localStorage.setItem('ql.ml.cur', JSON.stringify(id))", exp_id)
        pg.reload()
        pg.wait_for_selector(".chip")
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    return pg


def test_the_select_view_ranks_on_validation_data_shows_the_pareto_comparison_and_says_what_the_simplicity_step_claims(browser, live):  # noqa: F811
    running, _, ids = live
    pg = _open(browser, running)
    _mode(pg, "ml", "select")
    _wait_until(pg, "document.querySelectorAll('#ml .picklist input').length >= 3", timeout=60.0)
    for eid in ids.values():
        pg.check(f"#ml .picklist input[data-value='{eid}']")
    _open_all(pg, "ml")
    assert pg.locator("#ml select[data-key=selPrimary]").input_value() == "net_sharpe" and pg.locator("#ml input[data-key=selMaxDd]").input_value() == "40"
    _run(pg, "ml")
    _until_shown(pg, 'Constrained selection', timeout=120.0)
    text = pg.locator("#ml-results").text_content()
    for needle in ("pooled chronological validation returns", "Highest ranked", "Selected", "What the choice means", "Pareto comparison", "Nested estimate", "The best candidate in hindsight", "Audit of the windows", "Simplicity: a selection heuristic, not a test of equivalence", "does not show the models perform equally",
                   "nothing here is a statement about test, forward or live performance", "Why it is not eligible"):
        assert needle in text, needle
    assert pg.locator("#ml-results tbody tr").count() >= 3 and not pg.problems, pg.problems
    _open_all(pg, "ml")
    pg.select_option("#ml select[data-key=selRule]", "non_inferiority")
    _open_all(pg, "ml")
    pg.fill("#ml input[data-key=selMargin]", "5")
    _run(pg, "ml")
    _until_shown(pg, 'Simplicity: a formal non-inferiority test', timeout=120.0)
    assert "declared assumption" in pg.locator("#ml-results").text_content() and not pg.problems, pg.problems
    pg.close()


def test_the_select_view_refuses_an_empty_choice_and_the_search_form_offers_the_selection_as_a_switch(browser, live):  # noqa: F811
    running, _, ids = live
    pg = _open(browser, running)
    _mode(pg, "ml", "select")
    _wait_until(pg, "document.querySelectorAll('#ml .picklist input').length >= 3", timeout=60.0)
    assert "Choose between one and twenty experiments" in pg.locator("#ml").text_content() or pg.locator("#ml .runbar button.primary").is_disabled()
    _mode(pg, "ml", "search")
    _open_all(pg, "ml")
    assert pg.locator("#ml input[data-key=selection]").get_attribute("type") == "checkbox" and not pg.locator("#ml input[data-key=selection]").is_checked()
    assert pg.locator("#ml select[data-key=selPrimary]").count() == 0, "the thresholds appear when the switch is on"
    pg.check("#ml input[data-key=selection]")
    _open_all(pg, "ml")
    assert pg.locator("#ml select[data-key=selPrimary]").input_value() == "net_sharpe" and pg.locator("#ml input[data-key=selK]").input_value() == "1"
    assert not pg.problems, pg.problems
    pg.close()


def test_the_research_record_runs_the_challenger_search_of_an_approved_proposal_and_shows_the_evidence_and_never_a_promotion(browser, live):  # noqa: F811
    running, app, ids = live
    act = app.extra_post["/api/ml/retraining/action"]
    exp_id = ids["ridge"]
    prop = act({"id": exp_id, "action": "trigger", "kind": "manual", "actor": "monitor"})
    assert prop["outcome"] == "proposed"
    pg = _open(browser, running, exp_id)
    _mode(pg, "ml", "controls")
    _until_shown(pg, 'Retraining: schedule, diagnostics and a constrained challenger search', timeout=120.0)
    start = pg.locator("#ml-results button", has_text="Run the challenger search")
    assert start.is_disabled(), "an unapproved proposal cannot be started"
    pg.click("#ml-results button:has-text('Check the triggers now')")
    _until_shown(pg, 'Triggers as of', timeout=120.0)
    text = pg.locator("#ml-results").text_content()
    for needle in ("Schedule", "Drift alert", "Calibration alert", "Performance alert", "a monthly schedule", "Autonomous retraining is off"):
        assert needle.lower() in text.lower(), needle
    act({"id": exp_id, "action": "approve", "proposal": prop["proposal"], "actor": "alice", "note": "the monthly review asked for it"})
    pg.close()
    pg = _open(browser, running, exp_id)                                                                    # a new page: the approval is on record now
    _mode(pg, "ml", "controls")
    _until_shown(pg, 'Retraining: schedule', timeout=120.0)
    pg.fill("#ml-results input[aria-label='Who decides']", "alice")
    start = pg.locator("#ml-results button", has_text="Run the challenger search")
    assert start.is_enabled()
    start.click()
    _until_shown(pg, 'Latest challenger search', timeout=600.0)
    text = pg.locator("#ml-results").text_content()
    for needle in ("Decision", "Promoted", "nothing was promoted", "protected segments read: no", "Same observations", "Versions, kept for rollback", "Holm over"):
        assert needle in text, needle
    assert pg.locator("#ml-results button", has_text="Roll back").count() == 1 and not pg.problems, pg.problems
    pg.close()
