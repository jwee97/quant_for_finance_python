"""The research record in a real browser shows Release 6: the alpha diagnostics (the IC with its dependence-aware interval, the decay table, coverage and stability), the portfolio diagnostics (the book held,
the covariance estimators on realised risk, risk contributions, the allocation sensitivity) and the controlled comparisons (a card is frozen, run once, and its verdict and negative findings stay listed).
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
    root = isolated_root(tmp_path_factory.mktemp("alpha_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("alpha_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    exp = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "alpha-record", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, exp
    running.stop()


def _open(browser, running):  # noqa: F811
    pg = _page(browser, running)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "controls")
    return pg


def _has(pg, text, timeout=120.0):
    _wait_until(pg, f"document.querySelector('#ml-results').textContent.includes({text!r})", timeout=timeout)


def test_the_record_computes_and_keeps_the_alpha_and_portfolio_diagnostics(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running)
    _has(pg, "Controlled comparisons against this experiment")
    text = pg.locator("#ml-results").text_content()
    assert "Alpha diagnostics" in text and "Portfolio diagnostics" in text and text.count("have not been computed for this experiment") >= 2 and "No card is stored yet." in text
    pg.get_by_role("button", name="Compute the alpha diagnostics").click()
    _has(pg, "Decay: mean rank IC by extra delay")
    text = pg.locator("#ml-results").text_content()
    for needle in ("Newey-West, lag", "Dates with an IC", "Hit rate", "Coverage of investable cells", "Blocks with the same sign", "20-session return"):
        assert needle in text, needle
    pg.get_by_role("button", name="Compute the portfolio diagnostics").click()
    _has(pg, "Covariance estimators on the risk the book realised")
    text = pg.locator("#ml-results").text_content()
    for needle in ("The book it held", "Effective positions", "lw_constant_correlation", "Risk contributions of the last book", "Cost-aware allocation: how the optimal book moves", "Grid points / without a solution"):
        assert needle in text, needle
    assert not pg.problems, pg.problems
    pg.close()
    pg = _open(browser, running)                                                               # a fresh page reads the stored records: computing and recording happened once
    _has(pg, "Decay: mean rank IC by extra delay")
    assert "Covariance estimators on the risk the book realised" in pg.locator("#ml-results").text_content()
    assert not pg.problems, pg.problems
    pg.close()


def test_a_comparison_card_is_frozen_run_and_listed_with_its_verdict(browser, live):  # noqa: F811
    running, app, exp = live
    pg = _open(browser, running)
    _has(pg, "Controlled comparisons against this experiment")
    pg.get_by_role("button", name="Freeze the card and run it").click()                          # nothing declared yet: refused, and nothing was run
    pg.wait_for_timeout(500)
    assert "No card is stored yet." in pg.locator("#ml-results").text_content()
    pg.fill("label:has-text('Hypothesis') input", "a market-neutral target ranks assets better")
    pg.fill("label:has-text('The one change') input", "the target only")
    pg.get_by_role("button", name="Freeze the card and run it").click()
    _wait_until(pg, "/(supported on development|inconclusive|worse)/.test(document.querySelector('#ml-results').textContent) && document.querySelector('#ml-results').textContent.includes('cmp-')", timeout=300.0)
    text = pg.locator("#ml-results").text_content()
    assert "Target engineering" in text and "a market-neutral target ranks assets better" in text
    assert not pg.problems, pg.problems
    pg.close()
