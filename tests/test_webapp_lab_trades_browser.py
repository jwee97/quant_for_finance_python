"""The Machine Learning Lab's report shows the buys and sells of a run that uses an ordinary position rule (not only decision_policy), and the Strategy Explorer and Model Comparison let the person choose the
universe. In a real browser; skipped when Playwright or Chromium is missing."""

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
    root = isolated_root(tmp_path_factory.mktemp("lab_trades_root"), config)
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("lab_trades_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    run = finished(app, app.extra_post["/api/ml/train"]({"dataset": BASE["dataset"], "split": BASE["split"], "model": BASE["model"], "explain": False, "name": "lab run", "mode": "development"}))
    running = server.start(app, "127.0.0.1", 0)
    yield running, app, run
    running.stop()


def test_the_labs_report_draws_the_buys_and_sells_of_an_ordinary_position_rule(browser, live):  # noqa: F811
    running, app, run = live
    pg = _page(browser, running)
    pg.evaluate("id => localStorage.setItem('ql.ml.cur', JSON.stringify(id))", run["id"])
    pg.reload()
    pg.wait_for_selector(".chip")
    pg.click("#tab-ml")
    pg.wait_for_selector("#ml .panel")
    pg.click("#ml button:has-text('Experiments')")
    pg.locator("#ml button:has-text('Open')").first.click()
    pg.wait_for_selector("[data-role=lab-trades] svg.chart .event", timeout=60000)
    card = pg.locator("[data-role=lab-trades]")
    text = card.inner_text()
    assert "Buys and sells" in text and "keeps no decision log" in text and "Turnover / year" in text and "Cost drag / year" in text
    assert "Win rate" not in text, "a rule that holds every asset has no round trips to win or lose, so those tiles are not shown"
    assert card.locator("svg.chart .event").count() > 0, "every trade is a marker on the price"
    assert card.locator("svg.chart .event title").first.text_content().strip(), "each marker says why"
    first = card.locator("svg.tp-svg").get_attribute("aria-label")
    card.locator("select[data-key=asset]").select_option(index=1)
    _wait_until(pg, f"document.querySelector('[data-role=lab-trades] svg.tp-svg').getAttribute('aria-label') !== {first!r}", timeout=30.0)
    card.locator("button:has-text('Open in the ML Strategy Explorer')").click()
    pg.wait_for_selector("#strategy [data-role=strategy-tiles] .tile", timeout=60000)
    assert pg.url.endswith(f"#/strategy?id={run['id']}") and not pg.problems, pg.problems
    pg.close()


def test_the_strategy_explorer_and_the_comparison_offer_every_universe(browser, live):  # noqa: F811
    running, _, _ = live
    pg = _page(browser, running)
    for tab, form in (("strategy", "[data-role=strategy-builder]"), ("modelcompare", "[data-role=compare-form]")):
        pg.click(f"#tab-{tab}")
        pg.wait_for_selector(form)
        if tab == "strategy":
            pg.locator("[data-role=strategy-builder] summary").click()
        values = pg.locator(f"{form} select[data-key=universe] option").evaluate_all("els => els.map(e => e.value)")
        assert values == ["etf", "crypto", "futures", "custom"], values
        assert pg.locator(f"{form} input[data-key=tickers]").count() == 1
    pg.close()


def test_the_replay_works_for_a_run_without_a_decision_log(browser, live):  # noqa: F811
    """On such a run the replay used to answer every date with the error 'asset' (a missing column); now its dates are the days the trades were decided and the table names each decision's trade."""
    running, _, run = live
    pg = _page(browser, running)
    pg.goto(running.url + f"#/strategy?id={run['id']}")
    pg.wait_for_selector("[data-role=replay]", timeout=60000)
    pg.locator("[data-role=replay]").scroll_into_view_if_needed()
    pg.click("[data-role=replay] button:has-text('Replay')")
    pg.wait_for_selector("[data-role=replay] table", timeout=30000)
    assert "banner err" not in pg.locator("[data-role=replay]").inner_html()
    pg.click("[data-role=replay] button:has-text('Next decision day')")
    pg.wait_for_selector("[data-role=replay] table", timeout=30000)
    assert not pg.problems, pg.problems
    pg.close()


def test_the_trade_panel_shares_one_axis_zooms_and_stays_inside_a_phone(browser, live):  # noqa: F811
    running, _, run = live
    pg = _page(browser, running)
    pg.goto(running.url + f"#/strategy?id={run['id']}")
    pg.wait_for_selector("[data-role=trade-panel] svg.tp-svg", timeout=60000)
    panel = pg.locator("[data-role=trade-panel]")
    assert panel.locator("svg.tp-svg").count() == 1, "price, trades, position and the overview strip are one drawing, so they zoom together"
    for label in ("Price", "Position", "Book exposure", "Realized P&L", "Unrealized P&L", "Strategy return"):
        assert label in panel.locator("[data-role=trade-metrics]").inner_text(), label
    full = panel.locator("svg.tp-svg").get_attribute("aria-label")
    panel.locator(".tp-seg button:has-text('3M')").click()
    assert panel.locator("svg.tp-svg").get_attribute("aria-label") != full, "a range button changes the window of every panel at once"
    assert panel.locator(".tp-seg button:has-text('3M')").get_attribute("aria-pressed") == "true"
    assert panel.locator("details.tp-legend[data-role=marker-legend]").count() == 1 and panel.locator("details.tp-legend").get_attribute("open") is None, "the legend is collapsed until asked for"
    box = panel.locator("svg.tp-svg").bounding_box()
    pg.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + 80)
    assert panel.locator("svg.tp-svg line[stroke-dasharray]").count() >= 1
    panel.locator(".tp-link").click()
    assert panel.locator("svg.tp-svg").get_attribute("aria-label") == full, "Reset zoom returns to the whole history"
    assert panel.locator(".event title").count() > 0 and not pg.problems, pg.problems
    pg.close()
    phone = browser.new_page(viewport={"width": 390, "height": 800}, has_touch=True)
    phone.goto(running.url + f"#/strategy?id={run['id']}")
    phone.wait_for_selector("[data-role=trade-panel] svg.tp-svg", timeout=60000)
    assert phone.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "no sideways scrolling on a phone"
    assert phone.locator("[data-role=trade-panel] svg.tp-svg").bounding_box()["width"] <= 390
    phone.close()


def test_two_fingers_pinch_and_pan_the_trade_panel(browser, live):  # noqa: F811
    """On a touch screen two fingers zoom (spread) and pan the window of the whole panel; one finger only moves the crosshair."""
    running, _, run = live
    phone = browser.new_page(viewport={"width": 390, "height": 800}, has_touch=True)
    phone.goto(running.url + f"#/strategy?id={run['id']}")
    phone.wait_for_selector("[data-role=trade-panel] svg.tp-svg", timeout=60000)
    label = lambda: phone.locator("[data-role=trade-panel] svg.tp-svg").get_attribute("aria-label")  # noqa: E731
    whole = label()
    box = phone.locator("[data-role=trade-panel] svg.tp-svg").bounding_box()
    y, cx = box["y"] + 100, box["x"] + box["width"] * 0.6

    def touch(kind, pid, x):
        phone.evaluate("""([kind, pid, x, y]) => {
          const host = document.querySelector('[data-role=trade-panel]'), svg = host.querySelector('svg.tp-svg');
          (kind === 'pointerdown' ? svg : host).dispatchEvent(new PointerEvent(kind, { pointerId: pid, pointerType: 'touch', clientX: x, clientY: y, bubbles: true, isPrimary: pid === 1 }));
        }""", [kind, pid, x, y])

    touch("pointerdown", 1, cx - 20)
    assert label() == whole, "one finger does not zoom"
    touch("pointerdown", 2, cx + 20)
    touch("pointermove", 1, cx - 90)
    touch("pointermove", 2, cx + 90)                                                          # the fingers spread: zoom in
    phone.wait_for_function("l => document.querySelector('[data-role=trade-panel] svg.tp-svg').getAttribute('aria-label') !== l", arg=whole, timeout=5000)
    zoomed = label()
    assert zoomed != whole and phone.locator("[data-role=trade-panel] .tp-link").is_enabled()
    touch("pointermove", 1, cx - 40)
    touch("pointermove", 2, cx + 140)                                                         # both move right with the same spread: pan
    phone.wait_for_function("l => document.querySelector('[data-role=trade-panel] svg.tp-svg').getAttribute('aria-label') !== l", arg=zoomed, timeout=5000)
    touch("pointerup", 1, cx - 40)
    touch("pointerup", 2, cx + 140)
    assert phone.locator("[data-role=trade-panel] svg.tp-svg").count() == 1 and label() != zoomed, "moving both fingers pans the window"
    phone.close()


def test_the_vendored_charting_library_is_served_with_its_licence(live):
    """ECharts (Apache-2.0) is part of the repository, not fetched: the app's content-security policy allows scripts from its own origin only. Its licence and notice travel with it."""
    import urllib.request

    running, _, _ = live
    for name, needle in (("vendor/echarts.min.js", b"echarts"), ("vendor/echarts.LICENSE", b"Apache License"), ("vendor/echarts.NOTICE", b"Apache ECharts"), ("charts_ec.js", b"window.Charts")):
        with urllib.request.urlopen(running.url + "static/" + name) as r:
            assert r.status == 200 and needle in r.read(), name
    html = urllib.request.urlopen(running.url).read().decode()
    assert html.index("vendor/echarts.min.js") < html.index("charts.js") < html.index("charts_ec.js"), "the library loads first, then the SVG kit, then the adapter that takes charts over"
