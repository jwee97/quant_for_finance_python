"""The research record in a real browser shows P1-22: the class probabilities' Brier score, log loss and calibration error beside the base-rate forecast, and, for a model that stored a predictive distribution, the
interval's coverage with its interval, width, interval score and pinball loss, with the two conformal procedures and what each guarantees; a model with neither says so. Skipped when Playwright or Chromium is
missing."""

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
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "explain": False, "mode": "development"}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("interval_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=7, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("interval_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    ids = {}
    for name, over in (("classifier", {"model": {"kind": "logistic"}, "dataset": {**BASE["dataset"], "target": "direction_5"}}),
                       ("ridge", {"model": {"kind": "ridge"}}),
                       ("ensemble", {"model": {"kind": "model_ensemble", "params": {"members": "ridge,lasso", "weighting": "equal"}}})):          # the newest is the one the page opens on
        exp = finished(app, app.extra_post["/api/ml/train"]({**BASE, **over, "name": f"interval-{name}"}))
        finished(app, app.extra_post["/api/ml/statistical-diagnostics/run"]({"id": exp["id"], "n_boot": 200}))
        ids[name] = exp["id"]
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, ids
    running.stop()


def _open(browser, running, exp_id):  # noqa: F811
    pg = _page(browser, running)
    pg.evaluate("id => localStorage.setItem('ql.ml.cur', JSON.stringify(id))", exp_id)
    pg.reload()
    pg.wait_for_selector(".chip")
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "controls")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Naive (treats days as independent)')", timeout=180.0)
    return pg


def test_a_model_with_a_predictive_distribution_shows_its_interval_scores_and_the_conformal_intervals(browser, live):  # noqa: F811
    running, _, ids = live
    pg = _open(browser, running, ids["ensemble"])
    text = pg.locator("#ml-results").text_content()
    for needle in ("Prediction interval of the model_ensemble model", "Coverage (interval)", "Interval score", "Pinball: lower / centre / upper", "from the 0.10 and 0.90 quantiles",
                   "time-series conformal prediction, calibrated only on outcomes that had ended", "Rolling split conformal", "Adaptive conformal inference", "none for time series", "not conditional on the features"):
        assert needle in text, needle
    assert "Class probabilities" not in text and not pg.problems, pg.problems
    pg.close()


def test_a_classifier_shows_brier_log_loss_and_calibration_error_beside_the_base_rate_and_no_interval(browser, live):  # noqa: F811
    running, _, ids = live
    pg = _open(browser, running, ids["classifier"])
    text = pg.locator("#ml-results").text_content()
    for needle in ("Class probabilities: Brier score, log loss and calibration error", "Brier (base rate)", "Log loss (base rate)", "Calibration error", "Interval calibration: the model stored no predictive distribution"):
        assert needle in text, needle
    assert "Prediction interval of the" not in text and not pg.problems, pg.problems
    pg.close()


def test_a_point_forecast_says_it_has_neither(browser, live):  # noqa: F811
    running, _, ids = live
    pg = _open(browser, running, ids["ridge"])
    text = pg.locator("#ml-results").text_content()
    assert "Probability calibration: the model produced no class probabilities" in text and "Interval calibration: the model stored no predictive distribution" in text
    assert "Class probabilities: Brier" not in text and "Prediction interval of the" not in text and not pg.problems, pg.problems
    pg.close()
