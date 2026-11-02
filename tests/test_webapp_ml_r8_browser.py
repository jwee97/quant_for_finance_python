"""The research record in a real browser shows Release 8's optional section. Nothing in it is computed until it is asked for; drift is computed once, kept, and read back by a fresh page; a model that states no
uncertainty is shown as having nothing to monitor, never as healthy; a manual retraining trigger is recorded as a proposal or as a suppressed trigger, never as a retraining; and a built report is listed.
Skipped when Playwright or Chromium is missing."""

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
from tests.test_webapp_workspaces_browser import _mode, _page, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge"}, "explain": False}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("r8_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=7, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("r8_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    exp = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "r8-record", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, exp
    running.stop()


def _open(browser, running):  # noqa: F811
    pg = _page(browser, running)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "controls")
    return pg


def _has(pg, text, timeout=180.0):
    _wait_until(pg, f"document.querySelector('#ml-results').textContent.includes({text!r})", timeout=timeout)


def test_nothing_is_computed_until_asked_and_drift_is_kept_for_a_fresh_page(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running)
    _has(pg, "Advanced (optional)")
    text = pg.locator("#ml-results").text_content()
    assert "Off by default: nothing here runs until you ask" in text and "Drift has not been checked for this experiment" in text
    assert "No report has been built for this experiment" in text and "PSI (a heuristic)" not in text
    pg.get_by_role("button", name="Check for drift").click()
    _has(pg, "PSI (a heuristic)")
    text = pg.locator("#ml-results").text_content()
    for needle in ("fixed at the fit cutoff", "KS p, adjusted", "Settled-IC drift", "an alert is a proposal to look, not an action"):
        assert needle in text, needle
    assert not pg.problems, pg.problems
    pg.close()
    pg = _open(browser, running)
    _has(pg, "PSI (a heuristic)")
    assert "Drift has not been checked for this experiment" not in pg.locator("#ml-results").text_content()
    assert not pg.problems, pg.problems
    pg.close()


def test_a_model_with_no_stated_uncertainty_is_not_applicable_and_never_healthy(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running)
    _has(pg, "Advanced (optional)")
    pg.get_by_role("button", name="Monitor uncertainty").click()
    _has(pg, "not a clean bill of health")
    text = pg.locator("#ml-results").text_content()
    assert "reports no predictive distribution" in text
    assert not pg.problems, pg.problems
    pg.close()


def test_a_manual_trigger_is_recorded_as_a_proposal_that_needs_a_person_and_a_report_is_listed(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running)
    _has(pg, "Advanced (optional)")
    pg.get_by_role("button", name="Record a manual retraining trigger").click()
    _has(pg, "1 event")
    text = pg.locator("#ml-results").text_content()
    assert "Open proposal:" in text and "Approval, rejection, recovery and rollback name a person" in text and "a challenger is never promoted" in text
    assert ("proposal" in text.lower()) and "retrained" not in text.lower()
    pg.get_by_role("button", name="Build a report").click()
    _has(pg, "Snapshot taken")
    text = pg.locator("#ml-results").text_content()
    assert "rp-" in text and "does not change when models or data are updated" in text
    assert not pg.problems, pg.problems
    pg.close()


def test_the_catalogue_and_a_champion_pair_can_be_read_from_the_page(browser, live):  # noqa: F811
    running, app, exp = live
    pair = app.extra_post["/api/ml/champion/register"]({"champion": "page-champ", "challenger": "page-chall", "universe_version": "u1", "horizon": 5, "looks": [10, 20], "actor": "alice"})
    pair_id = pair.get("data", pair)["pair_id"]
    pg = _open(browser, running)
    _has(pg, "Advanced (optional)")
    assert "The twelve optional modules" not in pg.locator("#ml-results").text_content(), "the list is read when asked for"
    pg.get_by_role("button", name="List the advanced modules").click()
    _has(pg, "The twelve optional modules")
    text = pg.locator("#ml-results").text_content()
    for needle in ("Drift monitoring", "Tax-lot and after-tax accounting", "Off-policy evaluation", "api and cli", "library", "all of these are off by default"):
        assert needle in text, needle
    pg.get_by_label("Champion and challenger pair id").fill(pair_id)
    pg.get_by_role("button", name="Show the pair").click()
    _has(pg, "Next look needs")
    text = pg.locator("#ml-results").text_content()
    for needle in ("page-champ", "page-chall", "Settled pairs", "pending", "A look is a statement about the evidence at that look and promotes nothing"):
        assert needle in text, needle
    assert not pg.problems, pg.problems
    pg.close()
