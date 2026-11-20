"""The Stock Universe Explorer in a real browser: the form runs a study as a job and the page shows the hindsight warning, the out-of-sample result, the block-by-block choices, the robustness card and the
in-sample table labelled as such. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _page

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp("universe_root"), config)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("universe_prices"), fetch=_fetch), default_loader=_default), root=root)
    running = server.start(app, "127.0.0.1", 0)
    yield running
    running.stop()


def test_a_study_runs_from_the_form_and_the_page_separates_out_of_sample_from_in_sample(browser, live):  # noqa: F811
    pg = _page(browser, live)
    pg.click("#tab-universe")
    pg.wait_for_selector("[data-role=universe-form]", timeout=30000)
    pg.fill("#universe input[data-key=tickers]", "AAA, BBB, CCC, DDD, EEE, FFF")
    pg.fill("#universe input[data-key=folds]", "2")
    pg.fill("#universe input[data-key=lookbacks]", "126")
    pg.uncheck("#universe input[data-screen=momentum]")
    for m in ("min_variance", "risk_parity", "hrp"):
        if pg.locator(f"#universe input[data-method={m}]").count():
            pg.uncheck(f"#universe input[data-method={m}]")
    pg.click("#universe button:text-is('Run the universe study')")
    pg.wait_for_selector("[data-role=universe-tiles] .tile", timeout=300000)
    text = pg.locator("#universe").inner_text()
    assert "survivorship" in text.lower() or "survivors" in text.lower(), "the hindsight warning is on the page"
    assert "Candidates tried" in text and "Deflated Sharpe probability" in text and "Out-of-sample Sharpe" in text
    assert "What was chosen, block by block" in text and "selection optimism" in text
    assert "How far to believe it" in text and "Block bootstrap" in pg.locator("[data-role=universe-robustness]").inner_text()
    assert "in-sample" in pg.locator("[data-role=universe-insample]").inner_text().lower() and "not a finding" in pg.locator("[data-role=universe-insample]").inner_text()
    assert pg.locator("#universe svg.chart").count() >= 1
    assert not pg.problems, pg.problems
    pg.close()
