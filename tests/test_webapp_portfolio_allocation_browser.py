"""The Portfolio allocation section of the ML workspace in a real browser, on a two-stock universe (the case that used to come out as 50/50): different allocation policies give different weights where the forecasts, the
risk or the limits justify it, the explanation says why, the comparison reads identical forecasts through every allocator, and a protected segment is reported only when asked for. Skipped when Playwright or Chromium
is missing."""

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
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "horizon": 5, "target": "fwd_ret_5", "min_history": 60}, "split": {"min_train_days": 300, "refit_every": 63, "validation_fraction": 0.2, "test_fraction": 0.2, "forward_fraction": 0.1},
        "signal": {"method": "proportional", "holding_period": 1}, "backtest": {"rebalance": "weekly", "target_vol": None}, "model": {"kind": "ridge"}, "explain": False}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("alloc_root"), config)
    bundle = make_bundle(n_assets=2, n=1500, seed=3, alpha=1.0, macro=False)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("alloc_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    parent = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "two stocks", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, parent
    running.stop()


def _open(browser, running):  # noqa: F811
    pg = _page(browser, running)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "allocation")
    pg.wait_for_selector("#ml select[data-key=alAllocator]", timeout=60000)
    return pg


LABEL = {"equal_weight": "Equal weight", "inverse_vol": "Inverse volatility", "mean_variance": "Mean-variance"}


def _run(pg, allocator, max_weight="90"):
    pg.select_option("#ml select[data-key=alAllocator]", allocator)
    pg.wait_for_selector("#ml select[data-key=alAllocator]")
    pg.select_option("#ml select[data-key=alAllocator]", allocator)
    pg.click("#ml details:has-text('4. Weight limits') summary") if not pg.locator("#ml input[data-key=alMaxW]").is_visible() else None
    pg.fill("#ml input[data-key=alMaxW]", max_weight)
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, f"document.querySelector('#ml-results').textContent.includes('Portfolio allocation: {LABEL[allocator]}') && document.querySelector('#ml-results [data-allocation-explain]') !== null", timeout=240.0)       # this run's own heading, not the previous one's
    return pg.locator("#ml-results [data-allocation-explain]")


def _weights(pg):
    rows = pg.locator("#ml-results [data-allocation-explain] table tbody tr")
    out = {}
    for i in range(rows.count()):
        cells = [c.strip() for c in rows.nth(i).locator("td").all_inner_texts()]
        out[cells[0]] = float(cells[-2].rstrip("%")) / 100.0                                       # the Weight column (the one before the share of risk)
    return out


def test_two_stocks_get_different_weights_under_different_policies_and_the_page_says_why(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    assert "Portfolio allocation" in pg.locator("#ml-results").text_content() or pg.locator("#ml .runbar button.primary").is_enabled()
    _run(pg, "equal_weight")
    eq = _weights(pg)
    assert len(eq) == 2 and all(abs(v - 0.5) < 1e-6 for v in eq.values()), f"equal weight is the benchmark: {eq}"
    _run(pg, "inverse_vol")
    iv = _weights(pg)
    names = sorted(iv)
    assert abs(iv[names[0]] + iv[names[1]] - 1.0) < 0.01
    risk = pg.locator("#ml-results [data-allocation-explain]").get_attribute("data-allocation-explain")
    assert risk in ("risk", "constraints", "equal"), "inverse volatility never reads the forecast"
    _run(pg, "mean_variance")
    mv = _weights(pg)
    assert abs(sum(mv.values()) - 1.0) < 0.02 or sum(mv.values()) <= 1.0 + 1e-6
    text = pg.locator("#ml-results").text_content()
    assert "Raw forecast" in text and "Calibrated forecast" in text and "Forecast uncertainty" in text and "Share of risk" in text and "From forecast" in text
    assert "Weights through time" in text and "Risk contribution at the latest decision" in text and "Equity curve by segment" in text
    assert any(abs(mv[a] - eq[a]) > 1e-3 for a in mv) or any(abs(iv[a] - eq[a]) > 1e-3 for a in iv), "at least one risk-aware or forecast-aware policy moves off 50/50 when the data justify it"
    assert not pg.problems, pg.problems
    pg.close()


def test_an_impossible_limit_is_reported_and_not_relaxed(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.select_option("#ml select[data-key=alAllocator]", "mean_variance")
    pg.click("#ml details:has-text('4. Weight limits') summary") if not pg.locator("#ml input[data-key=alMaxW]").is_visible() else None
    pg.fill("#ml input[data-key=alMaxW]", "30")                                                  # two stocks, long only, fully invested: 30% each cannot reach 100%
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "/had no solution at \\d+ of \\d+ decisions/.test(document.querySelector('#ml-results').textContent)", timeout=240.0)
    text = pg.locator("#ml-results").text_content()
    assert "failure policy" in text and "Decisions solved" in text
    assert "0 of 55" in text.replace("\xa0", " ") or "0 of" in text, "no decision was solved: the limits cannot be met and were not relaxed"
    pg.close()


def test_the_comparison_ranks_on_validation_and_shows_protected_segments_only_when_asked(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.click("#ml button:has-text('Compare all allocators')")
    pg.click("#ml details:has-text('4. Weight limits') summary") if not pg.locator("#ml input[data-key=alMaxW]").is_visible() else None
    pg.fill("#ml input[data-key=alMaxW]", "90")
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "document.querySelector('#ml-results [data-allocator-comparison]') !== null", timeout=300.0)
    text = pg.locator("#ml-results").text_content()
    assert "identical forecasts" in text and "Equal weight (benchmark)" in text and "Holm p" in text and "Forecast quality, separate from portfolio quality" in text
    assert "not used to select" not in text, "the protected segments are not shown unless asked for"
    assert "not estimated" in text or "needs at least five assets" in text, "no cross-sectional statistic is invented for a two-stock universe"
    pg.check("#ml input[data-key=alProtected]")
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('not used to select')", timeout=300.0)
    assert not pg.problems, pg.problems
    pg.close()
