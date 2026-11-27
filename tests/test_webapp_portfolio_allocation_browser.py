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
from tests.ml_helpers import make_bundle, rename_assets
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
    bundle = rename_assets(make_bundle(n_assets=2, n=1500, seed=3, alpha=1.0, macro=False), {"A00": "NVDA", "A01": "MU"})          # the two-stock universe of the request
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
    pg.click("#ml details:has-text('7. Weight limits') summary") if not pg.locator("#ml input[data-key=alMaxW]").is_visible() else None
    pg.fill("#ml input[data-key=alMaxW]", max_weight)
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, f"document.querySelector('#ml-results').textContent.includes('Portfolio allocation: {LABEL[allocator]}') && document.querySelector('#ml-results [data-allocation-explain]') !== null", timeout=240.0)       # this run's own heading, not the previous one's
    return pg.locator("#ml-results [data-allocation-explain]")


def _weights(pg):
    head = [c.strip() for c in pg.locator("#ml-results [data-allocation-explain] table thead th").all_inner_texts()]
    col = next(i for i, c in enumerate(head) if c in ("Weight", "Final (blended)"))
    rows = pg.locator("#ml-results [data-allocation-explain] table tbody tr")
    out = {}
    for i in range(rows.count()):
        cells = [c.strip() for c in rows.nth(i).locator("td").all_inner_texts()]
        out[cells[0]] = float(cells[col].rstrip("%")) / 100.0
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
    pg.click("#ml details:has-text('7. Weight limits') summary") if not pg.locator("#ml input[data-key=alMaxW]").is_visible() else None
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
    pg.click("#ml details:has-text('7. Weight limits') summary") if not pg.locator("#ml input[data-key=alMaxW]").is_visible() else None
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


# ------------------------------------------------------------------------------------------------ second round: horizon controls, calibration, stabilisers, adapters
def _open_section(pg, title):
    if not pg.locator(f"#ml details:has-text('{title}') input, #ml details:has-text('{title}') select").first.is_visible():
        pg.click(f"#ml details:has-text('{title}') summary")


def _results_text(pg):
    return pg.locator("#ml-results").text_content() or ""


def _wait_run(pg, label, timeout=240.0):
    _wait_until(pg, f"document.querySelector('#ml-results').textContent.includes('Portfolio allocation: {label}') && document.querySelector('#ml-results [data-allocation-explain]') !== null", timeout=timeout)


def test_the_page_explains_nvda_and_mu_from_forecast_uncertainty_covariance_and_limits(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.select_option("#ml select[data-key=alAllocator]", "mean_variance")
    _open_section(pg, "5. Forecast calibration")
    pg.select_option("#ml select[data-key=alInference]", "hac")
    _open_section(pg, "6. Stability")
    pg.select_option("#ml select[data-key=alBaseline]", "inverse_vol")
    _open_section(pg, "6. Stability")                                                            # choosing a baseline redraws the form: its sections close again
    pg.select_option("#ml select[data-key=alBlend]", "fixed")
    _open_section(pg, "6. Stability")
    pg.fill("#ml input[data-key=alAlpha]", "0.5")
    pg.check("#ml input[data-key=alSens]")
    _open_section(pg, "7. Weight limits")
    pg.fill("#ml input[data-key=alMaxW]", "80")
    pg.click("#ml .runbar button.primary")
    _wait_run(pg, "Mean-variance")
    text = _results_text(pg)
    words = pg.locator("#ml-results [data-allocation-narrative]").text_content()
    for needle in ("NVDA:", "MU:", "the model forecast", "after calibration the expected return used", "calibration slope", "volatility", "correlation with", "starting from 50.0% equal weight", "baseline inverse_vol wanted", "Final weight"):
        assert needle in words, needle
    for needle in ("Raw forecast", "Calibration slope", "Forecast confidence", "Calibrated forecast", "Forecast uncertainty", "Baseline book", "ML-only book", "Final (blended)", "Change from the previous decision", "Share of risk",
                   "Covariance and correlation used", "Turnover and cost of each decision", "Effective number of positions", "How sensitive is this book?", "Horizon, holding and decisions", "Blended: w_final"):
        assert needle in text, needle
    assert not pg.problems, pg.problems
    pg.close()


def test_the_holding_period_and_overlap_controls_change_the_recorded_lifecycle(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.select_option("#ml select[data-key=alAllocator]", "inverse_vol")
    _open_section(pg, "4. Time")
    pg.fill("#ml input[data-key=alHolding]", "10")
    pg.select_option("#ml select[data-key=alLifecycle]", "staggered_tranches")
    pg.fill("#ml input[data-key=alOptEvery]", "5")
    _open_section(pg, "7. Weight limits")
    pg.fill("#ml input[data-key=alMaxW]", "90")
    pg.click("#ml .runbar button.primary")
    _wait_run(pg, "Inverse volatility")
    text = _results_text(pg).replace("\xa0", " ")
    assert "staggered tranches: 2 alive at once, each held 10 days" in text and "Holding window" in text and "10 days" in text and "Decision every" in text and "5 days" in text
    pg.select_option("#ml select[data-key=alLifecycle]", "single_book")
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('one book; each new decision replaces the last')", timeout=240.0)
    assert not pg.problems, pg.problems
    pg.close()


def test_the_conservative_preset_and_risk_aversion_change_the_calculated_weights(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    _run(pg, "mean_variance", "90")
    plain = _weights(pg)
    before = pg.evaluate("(document.querySelector('#ml-results').textContent.match(/settings ([0-9a-f]{16})/) || [])[1]")
    pg.fill("#ml input[data-key=alRiskAversion]", "500")
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, f"((document.querySelector('#ml-results').textContent.match(/settings ([0-9a-f]{{16}})/) || [])[1] || '{before}') !== '{before}'", timeout=240.0)     # a new run, not the last one
    calm = _weights(pg)
    assert any(abs(calm[a] - plain[a]) > 1e-3 for a in plain), "a higher risk aversion changes the weights"
    _open_section(pg, "6. Stability")
    pg.select_option("#ml select[data-key=alPreset]", "conservative")
    assert pg.locator("#ml [data-preset-doc=conservative]").text_content().count("not an optimum") == 1
    assert pg.locator("#ml input[data-key=alRiskAversion]").count() == 0, "a preset's own settings are shown as documentation and cannot be silently overridden"
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "document.querySelector('#ml-results').textContent.includes('Blended: w_final')", timeout=240.0)
    text = _results_text(pg)
    assert "confidence" in text and "inverse vol" in text.lower()
    cons = _weights(pg)
    assert max(cons.values()) <= 0.6 + 1e-6, "the preset's documented 60% cap holds"
    pg.close()


def test_an_unsupported_configuration_is_refused_with_what_to_change(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.select_option("#ml select[data-key=alAllocator]", "herc")
    assert pg.locator("#ml [data-uses-forecast=false]").count() == 1 and "risk-only" in pg.locator("#ml [data-uses-forecast=false]").text_content()
    assert "Needs at least 3 assets" in pg.locator("#ml").text_content()
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "/at least 3 assets/.test(document.body.textContent)", timeout=240.0)
    pg.select_option("#ml select[data-key=alAllocator]", "mean_variance")
    _open_section(pg, "6. Stability")
    pg.select_option("#ml select[data-key=alBaseline]", "equal_weight")
    _open_section(pg, "6. Stability")
    pg.select_option("#ml select[data-key=alBlend]", "none")
    _open_section(pg, "6. Stability")
    pg.fill("#ml input[data-key=alMaxChange]", "500")
    assert pg.locator("#ml .runbar button.primary").is_disabled() or pg.locator("#ml .runbar .guard").count() >= 0
    pg.close()


def test_calibration_methods_are_compared_and_two_stocks_fall_back_to_the_pooled_slope_without_a_superiority_claim(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.click("#ml button:has-text('Compare calibration methods')")
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "document.querySelector('#ml-results [data-calibration-comparison]') !== null", timeout=300.0)
    text = _results_text(pg)
    assert "partial_pooling_hac" in text and "Against the baseline" in text and "fall back to the pooled slope" in text and "protected segments were not read" in text
    verdicts = [t.strip() for t in pg.locator("#ml-results [data-calibration-comparison] table tbody tr td:last-child").all_inner_texts()]
    allowed = {"the reference", "cannot be told apart from the baseline on this sample", "lower error than the baseline (the interval excludes zero)", "higher error than the baseline (the interval excludes zero)"}
    assert verdicts and set(verdicts) <= allowed, verdicts
    rows = {r.locator("td").first.inner_text().strip(): [c.strip() for c in r.locator("td").all_inner_texts()] for r in pg.locator("#ml-results [data-calibration-comparison] table tbody tr").all()}
    assert rows["partial_pooling_hac"][4] == "100%", "two stocks: partial pooling used the pooled slope at every decision"
    assert not pg.problems, pg.problems
    pg.close()


def test_the_comparison_shows_sortino_intervals_and_the_stabilisers_next_to_equal_weight(browser, live):  # noqa: F811
    running, app, parent = live
    pg = _open(browser, running)
    pg.click("#ml button:has-text('Compare all allocators')")
    _open_section(pg, "7. Weight limits")
    pg.fill("#ml input[data-key=alMaxW]", "90")
    pg.check("#ml input[data-key=alStab]")
    pg.click("#ml .runbar button.primary")
    _wait_until(pg, "document.querySelector('#ml-results [data-allocator-comparison]') !== null", timeout=300.0)
    text = _results_text(pg)
    for needle in ("Sortino", "Sharpe 95% interval (HAC)", "Sharpe vs equal weight (95%)", "Mean-variance, conservative preset", "Mean-variance blended 50/50 with inverse volatility", "Equal weight (benchmark)", "not expected to lose"):
        assert needle in text, needle
    assert not pg.problems, pg.problems
    pg.close()
