"""The research record and the forward test in a real browser show Release 5: the validation policy (or its absence), the three statements of Sharpe uncertainty, matched and unmatched benchmarks, regimes with
insufficient evidence said so, the strategy version with its risk limits, and the prospective ledger with replays kept apart. Skipped when Playwright or Chromium is missing."""

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
from tests.test_webapp_workspaces_browser import _mode, _page, _run, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge"}, "explain": False}
POLICY = {"name": "browser-policy", "version": "v1", "criteria": {"segment": "validation", "min_segment_days": 60, "interval_method": "hac", "min_sharpe_ci_lower": -9.0}}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("val_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("val_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    exp = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "validated", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, exp
    running.stop()


def _open(browser, running, mode):  # noqa: F811
    pg = _page(browser, running)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", mode)
    return pg


def test_the_record_says_no_policy_is_adopted_and_then_shows_the_diagnostics_read_against_it(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running, "controls")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Validation policy')", timeout=120.0)
    text = pg.locator("#ml-results").text_content()
    assert "No policy is adopted" in text and "implies no Sharpe" in text and "No diagnostics have been computed" in text and "No strategy version is registered" in text
    assert not pg.problems, pg.problems
    pg.close()
    app.extra_post["/api/ml/validation-policy"]({"policy": POLICY, "actor": "jo", "note": "declared before the evaluation"})
    finished(app, app.extra_post["/api/ml/statistical-diagnostics/run"]({"id": exp["id"], "n_boot": 200, "delays": [0, 1]}))
    app.extra_post["/api/ml/governance/version"]({"id": exp["id"], "actor": "jo", "risk_limits": {"max_gross": 1.5, "max_drawdown": 0.2}})
    pg = _open(browser, running, "controls")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Naive (treats days as independent)')", timeout=120.0)
    text = pg.locator("#ml-results").text_content()
    for needle in ("browser-policy v1", "adopted by jo", "HAC (Newey-West, delta method)", "Stationary bootstrap", "Against matched benchmarks", "no: descriptive only", "Regimes (declared before the evaluation)",
                   "insufficient evidence", "Delay between decision and fill", "Read against the adopted policy", "Strategy version, risk limits and prospective evidence", "max_gross 1.5", "Nothing has been committed yet"):
        assert needle in text, needle
    assert not pg.problems, pg.problems
    pg.close()


def test_the_forward_test_shows_the_ledger_with_replays_kept_apart(browser, live):  # noqa: F811
    running, app, exp = live
    finished(app, app.extra_post["/api/ml/forward/step"]({"id": exp["id"], "capital": 50_000}))
    pg = _open(browser, running, "forward")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Prospective ledger')", timeout=120.0)
    text = pg.locator("#ml-results").text_content()
    assert "replay (never counted)" in text and "listed and never counted as prospective evidence" in text and "not before the label's first return day" in text
    assert "daily equity of the committed holdings is built from the prices when a step runs" in text
    assert not pg.problems, pg.problems
    pg.close()
    pg = _open(browser, running, "forward")                                                    # pressing the button runs the step and draws the equity of the committed holdings
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Prospective ledger')", timeout=120.0)
    _run(pg, "ml")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Daily equity of the committed holdings')", timeout=180.0)
    assert not pg.problems, pg.problems
    pg.close()
