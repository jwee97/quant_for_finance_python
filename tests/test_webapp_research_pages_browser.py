"""Research Evidence, Model Comparison and the scheduled simulation in a real browser, each run through its form on the synthetic dataset. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.ml_helpers import make_bundle
from tests.test_ml_strategy_api import BASE
from tests.test_webapp import _fetch
from tests.test_webapp_assets_ml import finished
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _page, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("rp_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("rp_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "policy run", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running
    running.stop()


def test_the_evidence_page_lists_every_check_with_its_rule_and_runs_the_control_and_the_feature_study(browser, live):  # noqa: F811
    pg = _page(browser, live)
    pg.click("#tab-evidence")
    pg.wait_for_selector("[data-role=evidence-list] details", timeout=60000)
    text = pg.locator("#evidence").inner_text()
    assert "Evidence checklist" in text and "no combined score" in text.lower() and "Feature registry" in text
    assert pg.locator("[data-role=evidence-list] details").count() >= 10
    assert "Rule:" in pg.locator("[data-role=evidence-list]").inner_html() and "Source:" in pg.locator("[data-role=evidence-list]").inner_html()
    unknown = pg.locator("[data-check=random_signals]")
    assert unknown.get_attribute("data-status") == "unknown"
    unknown.locator("summary").click()
    unknown.locator("button:text-is('Run the random-signal control')").click()
    _wait_until(pg, "document.querySelector('[data-check=random_signals]').dataset.status !== 'unknown'", timeout=300.0)
    # the registry and its study
    assert pg.locator("[data-role=feature-registry] table").count() >= 2 and "Jegadeesh" in pg.locator("[data-role=feature-registry]").inner_text()
    pg.fill("#evidence input[data-key=min_train]", "504")
    pg.uncheck("#evidence input[data-key=ablate]")
    pg.click("#evidence button:text-is('Run the feature study')")
    _wait_until(pg, "[...document.querySelectorAll('[data-role=feature-registry] th')].some(th => th.textContent.includes('Mean rank IC'))", timeout=300.0)
    assert not pg.problems, pg.problems
    pg.close()


def test_the_model_comparison_page_runs_models_and_baselines_and_separates_predictive_from_trading_quality(browser, live):  # noqa: F811
    pg = _page(browser, live)
    pg.click("#tab-modelcompare")
    pg.wait_for_selector("[data-role=compare-form]", timeout=30000)
    pg.fill("#modelcompare input[data-key=min_train]", "504")
    pg.fill("#modelcompare input[data-key=draws]", "3")
    pg.click("#modelcompare button:text-is('Run the comparison')")
    pg.wait_for_selector("[data-role=compare-trading] table", timeout=300000)
    trading = pg.locator("[data-role=compare-trading]").inner_text()
    assert "equal weight, rebalanced monthly" in trading and "12-1 momentum" in trading and "random-signal books" in trading and "ridge" in trading
    assert "Rank IC" in pg.locator("[data-role=compare-predictive]").inner_text() and "forecast of 0" in pg.locator("[data-role=compare-predictive]").inner_text()
    assert pg.locator("#modelcompare svg.chart").count() >= 1 and "multiple comparison" in pg.locator("#modelcompare").inner_text()
    assert not pg.problems, pg.problems
    pg.close()


def test_the_strategy_builder_runs_a_scheduled_simulation_and_the_page_shows_the_versions_and_choices(browser, live):  # noqa: F811
    pg = _page(browser, live)
    pg.click("#tab-strategy")
    pg.wait_for_selector("[data-role=strategy-tiles] .tile", timeout=120000)
    pg.click("[data-role=strategy-builder] summary")
    pg.fill("[data-role=strategy-builder] input[data-key=min_train_days]", "504")
    pg.check("[data-role=strategy-builder] input[data-key=simulate]")
    pg.click("[data-role=strategy-builder] button:text-is('Run the strategy')")
    pg.wait_for_selector("[data-role=simulation-card]", timeout=400000)
    card = pg.locator("[data-role=simulation-card]").inner_text()
    assert "Model versions" in card and "Policy chosen at each refit" in card and "no prediction made by a model that had seen its date" in card and "Reproducibility" in card
    assert not pg.problems, pg.problems
    pg.close()
