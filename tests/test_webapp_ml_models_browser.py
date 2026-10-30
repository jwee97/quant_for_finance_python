"""The Machine learning tab's new model families in a real browser: every family is selectable with its own settings (choices, switches, text), the position rule follows an agent, and a trained model's report
shows what it adds (intervals, ranking quality, regime probabilities, ensemble weights, the asset graph, compute and provenance), each chart with its table. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.ml import models as md
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _mode, _open_all, _page, _run, _start, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live_models(tmp_path_factory):
    running = _start(tmp_path_factory, "models")
    yield running
    running.stop()


@pytest.fixture()
def mp(browser, live_models):  # noqa: F811
    pg = _page(browser, live_models)
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "train")
    pg.wait_for_selector("#ml select[data-key=model]", state="attached", timeout=60_000)
    _open_all(pg, "ml")
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def _available():
    return {m["kind"]: m for m in md.model_catalog() if m["available"]}


def _choose(page, kind):
    page.select_option("#ml select[data-key=model]", kind)
    _open_all(page, "ml")


def _set(page, key, value):
    page.fill(f"#ml input[data-key={key}]", str(value))


def _train(page, name):
    """Train with the form as it stands and wait for the report of THIS run: the page may still hold the report of the one before, so the wait is for the header that carries the name."""
    _set(page, "minTrain", 400)
    _set(page, "refit", 252)
    page.fill("#ml input[data-key=name]", name)
    _run(page, "ml")
    _wait_until(page, f"[...document.querySelectorAll('#ml-results .head h2')].some(h => h.textContent.trim() === {name!r}) || document.querySelector('#ml-results').textContent.includes('That did not run')", timeout=900.0)
    assert "That did not run" not in page.locator("#ml-results").text_content(), page.locator("#ml-results").text_content()[:400]


def _card(page, title):
    return page.locator(f"#ml-results .chartcard:has(h3:text-is('{title}'))")


def test_every_available_family_is_in_the_model_menu_with_its_family_and_the_missing_ones_are_explained(mp):
    options = mp.locator("#ml select[data-key=model] option").all_text_contents()
    regression = {k: m for k, m in _available().items() if "regression" in m["tasks"]}                       # the default target is a return; logistic regression waits for a classification target
    for kind, info in regression.items():
        assert any(o.startswith(info["label"]) and f"({info['family']})" in o for o in options), (kind, options[:8])
    families = {info["family"] for info in regression.values()}
    assert {"ranking", "probabilistic", "regime", "deep sequence", "forecasting", "representation", "graph", "ensemble"} <= families
    unavailable = [m for m in md.model_catalog() if not m["available"]]
    if unavailable:
        assert "Not available here" in mp.inner_text("#ml .panel")


def test_a_model_s_own_settings_are_menus_switches_and_text_as_well_as_numbers(mp):
    avail = _available()
    if "xgb_ranker" in avail:
        _choose(mp, "xgb_ranker")
        assert mp.locator("#ml select[data-key=objective] option").all_text_contents() == ["rank:ndcg", "rank:pairwise", "rank:map"]
        assert mp.locator("#ml input[data-key=relevance_levels]").input_value() == "5"
    if "hmm_regime" in avail:
        _choose(mp, "hmm_regime")
        assert mp.locator("#ml select[data-key=covariance]").input_value() == "diag" and "regime probabilities" in mp.inner_text("#ml .panel")
    if "model_ensemble" in avail:
        _choose(mp, "model_ensemble")
        assert mp.locator("#ml input[data-key=members]").input_value() == "ridge,lightgbm,lgbm_quantile" and mp.locator("#ml select[data-key=weighting]").input_value() == "dynamic"
        assert mp.locator("#ml input[data-key=refit_full]").get_attribute("type") == "checkbox" and mp.locator("#ml input[data-key=refit_full]").is_checked()
    if "lstm" in avail:
        _choose(mp, "lstm")
        assert mp.locator("#ml select[data-key=device] option").all_text_contents() == ["auto", "cpu", "cuda"] and mp.locator("#ml input[data-key=horizons]").get_attribute("type") == "text"
        assert "trains a network" in mp.inner_text("#ml .panel") and "training curves" in mp.inner_text("#ml .panel")
    if "tft" in avail:
        _choose(mp, "tft")
        assert "prediction intervals" in mp.inner_text("#ml .panel") and mp.locator("#ml input[data-key=alpha_high]").input_value() == "0.9"


def test_an_agent_trades_its_own_weights_and_the_rule_comes_back_with_another_model(mp):
    if "ppo" not in _available():
        pytest.skip("Stable-Baselines3 is not installed")
    _choose(mp, "ppo")
    assert mp.locator("#ml select[data-key=signal]").input_value() == "target_weights"
    assert mp.locator("#ml select[data-key=reward] option").all_text_contents() == ["risk_adjusted", "dsr", "return"]
    assert mp.locator("#ml input[data-key=long_only]").get_attribute("type") == "checkbox" and not mp.locator("#ml input[data-key=long_only]").is_checked()
    _choose(mp, "ridge")
    assert mp.locator("#ml select[data-key=signal]").input_value() == "top_n_long_short"


def test_the_uncertainty_rules_have_their_own_settings(mp):
    mp.select_option("#ml select[data-key=signal]", "uncertainty_weighted")
    _open_all(mp, "ml")
    assert mp.locator("#ml input[data-key=kappa]").input_value() == "0" and mp.locator("#ml input[data-key=uncertainty_power]").input_value() == "1"
    mp.select_option("#ml select[data-key=signal]", "uncertainty_mean_variance")
    _open_all(mp, "ml")
    assert mp.locator("#ml input[data-key=mv_lookback]").input_value() == "126"


def test_a_quantile_model_report_shows_intervals_ranking_quality_and_what_the_run_cost(mp):
    if "lgbm_quantile" not in _available():
        pytest.skip("LightGBM is not available")
    _choose(mp, "lgbm_quantile")
    _set(mp, "n_estimators", 40)
    mp.select_option("#ml select[data-key=signal]", "uncertainty_weighted")
    _open_all(mp, "ml")
    _train(mp, "quantile")
    for title in ("Prediction intervals", "Does the uncertainty mean anything?", "Ranking quality"):
        assert _card(mp, title).count() == 1, title
    assert "calibrated" in _card(mp, "Prediction intervals").inner_text() and mp.locator("#ml-results .chartcard:has(h3:text-is('Prediction intervals')) .bar select option").count() >= 2
    _card(mp, "Prediction intervals").locator(".seg button", has_text="Table").click()
    header = _card(mp, "Prediction intervals").locator("thead th").all_text_contents()
    assert [h.lower() for h in header] == ["date", "lower", "forecast", "upper", "realised"]
    _card(mp, "Ranking quality").locator(".seg button", has_text="Table").click()
    assert "Validation" in _card(mp, "Ranking quality").inner_text()
    text = mp.locator("#ml-results").text_content()                                                       # the headings are set in capitals, which inner_text would return
    assert "Compute and provenance" in text and "Training time" in text and "SHA-256" in text and "Libraries" in text and "lightgbm" in text


def test_a_regime_model_report_shows_filtered_probabilities_and_a_graph_model_its_edges(mp):
    if "hmm_regime" not in _available():
        pytest.skip("hmmlearn is not available")
    _choose(mp, "hmm_regime")
    _train(mp, "regimes")
    card = _card(mp, "Regime probabilities")
    assert card.count() == 1 and "FILTERED" in card.inner_text() and "never smoothed" in card.inner_text()
    card.locator(".seg button", has_text="Table").click()
    assert card.locator("tbody tr").count() == 3
    if "gcn" in _available():
        _mode(mp, "ml", "train")
        _open_all(mp, "ml")
        _choose(mp, "gcn")
        _set(mp, "epochs", 4)
        _set(mp, "hidden", 8)
        _train(mp, "graph")
        assert _card(mp, "The asset graph").count() == 1 and _card(mp, "Portfolio-level forecast").count() == 1
        assert _card(mp, "Training curves").count() == 1


def test_an_ensemble_report_shows_member_weights_and_an_agent_report_its_rewards(mp):
    if "model_ensemble" not in _available() or "lgbm_quantile" not in _available():
        pytest.skip("LightGBM is not available")
    _choose(mp, "model_ensemble")
    mp.fill("#ml input[data-key=members]", "ridge,lgbm_quantile[n_estimators=30]")
    _train(mp, "ensemble")
    card = _card(mp, "Ensemble weights")
    assert card.count() == 1 and "outcomes already known" in card.inner_text()
    card.locator(".seg button", has_text="Table").click()
    assert [r.lower() for r in card.locator("tbody tr td:first-child").all_text_contents()] == ["ridge", "lgbm_quantile"]
    if "ppo" in _available():
        _mode(mp, "ml", "train")
        _open_all(mp, "ml")
        _choose(mp, "ppo")
        for key, value in (("total_timesteps", 800), ("n_steps", 128), ("episode_days", 100), ("hidden", 16)):
            _set(mp, key, value)
        _train(mp, "agent")
        assert _card(mp, "Agent reward while training").count() == 1 and _card(mp, "Agent on its own training days").count() == 1
        _wait_until(mp, "document.querySelectorAll('#ml-results .chartcard').length >= 6")
