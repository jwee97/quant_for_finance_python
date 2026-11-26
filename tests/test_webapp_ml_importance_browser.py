"""The permutation-importance chart in a real browser: bars are actually drawn (measured, not assumed) for positive, negative and mixed values; an all-zero or all-undefined result gets an explanation instead of
an empty axis; undefined rows are left out of chart and table alike and counted in a note; long labels, a narrow phone screen, resizing and switching tabs leave the chart intact; and the Chart and Table views show the same rows.
The experiments behind two of the tests are real: a four-asset universe draws bars, a two-asset universe (where a rank IC does not exist) shows the explanation. Skipped without Playwright or Chromium."""

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
from tests.test_webapp_browser import browser, open_tab  # noqa: F401
from tests.test_webapp_workspaces_browser import _mode, _wait_until

sync_api = pytest.importorskip("playwright.sync_api")
FEATURES = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 1}, "model": {"kind": "ridge"}, "explain": True}


def make_app(tmp_path_factory, n_assets, name):
    config = load_config()
    root = isolated_root(tmp_path_factory.mktemp(name + "_root"), config)
    bundle = make_bundle(n_assets=n_assets, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp(name + "_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    exp = finished(app, app.extra_post["/api/ml/train"]({**FEATURES, "name": name, "mode": "development"}))
    return app, exp, server.start(app, "127.0.0.1", 0)


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    app, exp, running = make_app(tmp_path_factory, 4, "four-assets")
    two_app, two_exp, two_running = make_app(tmp_path_factory, 2, "two-assets")                 # a separate app: the dataset builder caches the universe it first loaded
    running.two = two_running
    yield running, app, {"four": exp["id"], "two": two_exp["id"]}
    running.stop()
    two_running.stop()


def page(browser, running, width=1200, height=900):  # noqa: F811
    pg = browser.new_page(viewport={"width": width, "height": height})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(running.url)
    pg.wait_for_selector(".chip")
    return pg


HARNESS = """([rows, status, width, top]) => {
  let host = document.getElementById('permtest');
  if (host) host.remove();
  host = document.createElement('div'); host.id = 'permtest'; host.style.cssText = 'position:relative;width:' + width + 'px;margin:0 auto;padding:8px;';
  document.body.prepend(host);
  window.QLImportance.card(host, rows, status, { top });
  return true;
}"""

MEASURE = """() => {
  const host = document.getElementById('permtest'), svg = host.querySelector('.ec-chart svg');
  if (!svg) return { svg: false };
  const bars = [...svg.querySelectorAll('path')].map((p) => { const b = p.getBBox(), r = p.getBoundingClientRect(); return { x: b.x, w: b.width, h: b.height, l: r.left, r: r.right, fill: p.getAttribute('fill') }; }).filter((b) => Math.abs(b.h - 14) < 1.5 && b.w >= 1);
  const labels = [...svg.querySelectorAll('text')].map((t) => { const b = t.getBoundingClientRect(); return { text: t.textContent, l: b.left, r: b.right, t: b.top, b: b.bottom }; });
  const box = host.getBoundingClientRect(), card = host.querySelector('.chartcard').getBoundingClientRect(), caption = host.querySelector('.note').getBoundingClientRect();
  return { svg: true, bars, labels, hostRight: box.right, hostLeft: box.left, chartBottom: svg.getBoundingClientRect().bottom, captionTop: caption.top, scrollW: document.documentElement.scrollWidth, innerW: window.innerWidth,
           toolbarOverlapsCaption: [...host.querySelectorAll('.chartkit button, .seg button')].some((b) => { const r = b.getBoundingClientRect(); return r.bottom > caption.top && r.top < caption.bottom && r.right > caption.left && r.left < caption.right; }) };
}"""


def rows(values, std=0.01, n=5):
    return [{"feature": f"f{i}", "mean_drop": v, "std_drop": std, "n": n, "ci_low": None if v is None else v - 0.02, "ci_high": None if v is None else v + 0.02} for i, v in enumerate(values)]


def draw(pg, data, status=None, width=700, top=15):
    pg.evaluate(HARNESS, [data, status, width, top])
    pg.wait_for_timeout(500)
    return pg.evaluate(MEASURE)


def test_positive_values_draw_bars_that_start_at_zero_and_grow_right(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    m = draw(pg, rows([0.20, 0.15, 0.10, 0.05, 0.02]))
    assert m["svg"] and len(m["bars"]) == 5, m
    assert len({round(b["x"]) for b in m["bars"]}) == 1, "every positive bar starts at the zero line"
    widths = [b["w"] for b in m["bars"]]
    assert widths == sorted(widths, reverse=True) and widths[0] > 3 * widths[-1] > 0, "bar length follows the value, largest first"
    assert not pg.problems, pg.problems
    pg.close()


def test_negative_and_mixed_values_diverge_around_one_zero_line_in_two_colours(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    m = draw(pg, rows([0.12, 0.06, 0.01, -0.01, -0.04, -0.09]))
    bars = m["bars"]
    assert len(bars) == 6
    pos, neg = bars[:3], bars[3:]
    zero = round(pos[0]["x"])
    assert all(round(b["x"]) == zero for b in pos), "positive bars start at zero"
    assert all(abs(b["x"] + b["w"] - zero) <= 1.5 for b in neg), "negative bars end at zero, extending left"
    assert {b["fill"] for b in pos}.isdisjoint({b["fill"] for b in neg}) and len({b["fill"] for b in pos}) == 1, "positive and negative are told apart by colour"
    assert any(t["text"] == "-0.090" or t["text"] == "−0.090" for t in m["labels"]) and any(t["text"] == "0.120" for t in m["labels"]), "exact values are printed"
    assert pg.evaluate("document.querySelectorAll('#permtest .ec-chart svg path').length") > 0
    pg.close()


def test_all_negative_values_still_draw_every_bar(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    m = draw(pg, rows([-0.01, -0.03, -0.08]))
    assert len(m["bars"]) == 3 and all(b["w"] > 1 for b in m["bars"])
    pg.close()


def test_all_zero_values_say_so_instead_of_pretending_a_chart(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    m = draw(pg, rows([0.0, 0.0, 0.0]))
    assert m["svg"] and all(b["w"] >= 1 for b in m["bars"]) and len(m["bars"]) == 3, "zero is a value: a one-pixel mark at the zero line, not an empty chart"
    pg.close()


def test_undefined_values_get_an_explanation_not_an_empty_axis(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    status = {"computable": False, "reason": "the universe has 2 asset(s) and a cross-sectional rank IC needs at least 3 on a date: the drop in rank IC is undefined for every feature"}
    m = draw(pg, rows([None, None, None, None]), status)
    assert m["svg"] is False, "no chart is drawn"
    text = pg.locator("#permtest [data-role=permutation-empty]").inner_text()
    assert "No permutation importance" in text and "2 asset(s)" in text and "fewer than three assets" in text
    assert pg.locator("#permtest .ec-chart").count() == 0
    pg.close()


def test_undefined_rows_are_left_out_of_chart_and_table_and_the_card_counts_them(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    status = {"computable": True, "reason": "1 of 4 features have no defined drop (left out of the chart and the table)", "interval": "mean ± 1.96 · spread / √n"}
    m = draw(pg, rows([0.1, None, -0.02, 0.05]), status)
    assert len(m["bars"]) == 3
    assert "1 feature has no defined drop" in pg.locator("#permtest").inner_text()
    pg.locator("#permtest .seg button", has_text="Table").click()
    names = pg.eval_on_selector_all("#permtest table tbody tr", "rs => rs.map(r => r.cells[0].textContent)")
    assert names == ["f0", "f3", "f2"], "the undefined row is not in the table either: it is counted in the note, never given a value"
    pg.close()


def test_chart_and_table_show_the_same_rows_in_the_same_order(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    data = rows([0.03, 0.2, -0.05, 0.11, 0.0, -0.01, 0.07])
    m = draw(pg, data, None, top=7)
    chart = [t["text"] for t in m["labels"] if t["text"].startswith("f") and t["text"][1:].isdigit()]
    pg.locator("#permtest .seg button", has_text="Table").click()
    pg.wait_for_selector("#permtest table")
    table = pg.eval_on_selector_all("#permtest table tbody tr", "rs => rs.map(r => [r.cells[0].textContent, r.cells[1].textContent])")
    expect = sorted(data, key=lambda r: -r["mean_drop"])
    assert [t[0] for t in table] == [r["feature"] for r in expect] == chart
    assert [float(t[1]) for t in table] == [round(r["mean_drop"], 4) for r in expect], "identical underlying data"
    pg.close()


def test_the_top_n_control_limits_the_bars_but_keeps_the_most_negative_visible(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    vals = [0.30 - 0.02 * i for i in range(24)] + [-0.06, -0.04]
    m = draw(pg, rows(vals), top=10)
    assert len(m["bars"]) == 10 + 3, "the top ten and the three most negative features (noise reliance must not be hidden)"
    assert pg.locator("#permtest select[data-role=perm-top] option").count() >= 3
    pg.select_option("#permtest select[data-role=perm-top]", "5")
    pg.wait_for_timeout(500)
    assert len(pg.evaluate(MEASURE)["bars"]) == 5 + 3
    pg.close()


def test_long_labels_do_not_overlap_the_bars_or_each_other(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    data = rows([0.2, 0.1, -0.05])
    data[0]["feature"] = "a_really_quite_long_feature_name_for_the_first_row_63"
    data[1]["feature"] = "short"
    m = draw(pg, data, width=500)
    names = [t for t in m["labels"] if t["text"] in ("short",) or t["text"].startswith("a_really")]
    first_bar_left = min(b["l"] for b in m["bars"])
    assert names and all(t["r"] <= first_bar_left + 1 for t in names), (names, first_bar_left)
    ys = sorted((t["t"], t["b"]) for t in m["labels"] if t["text"] in ("short",) or t["text"].startswith("a_really") or t["text"] == "f2")
    assert all(ys[i][1] <= ys[i + 1][0] + 1 for i in range(len(ys) - 1)), "rows do not overlap vertically"
    pg.close()


def test_on_a_phone_the_axis_labels_do_not_overlap_and_the_caption_is_not_covered(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running, width=360, height=800)
    m = draw(pg, rows([0.20, 0.10, -0.02, -0.05]), width=330)
    numeric = [t for t in m["labels"] if t["text"].replace(".", "").replace("-", "").replace("−", "").isdigit()]
    bottom = max(t["t"] for t in numeric)
    axis = sorted([t for t in numeric if abs(t["t"] - bottom) < 2], key=lambda t: t["l"])                       # the value axis is the lowest row of numbers; the bars' own value labels sit beside their bars
    assert len(axis) >= 2 and all(axis[i]["r"] <= axis[i + 1]["l"] + 0.5 for i in range(len(axis) - 1)), [(t["text"], round(t["l"]), round(t["r"])) for t in axis]
    assert m["toolbarOverlapsCaption"] is False and m["captionTop"] >= m["chartBottom"] - 1, "the caption sits below the chart, uncovered"
    assert m["scrollW"] <= m["innerW"] + 1, "no sideways scrolling"
    assert len(m["bars"]) == 4
    pg.close()


def test_resizing_the_window_redraws_the_chart_with_its_bars(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running, width=1200)
    m1 = draw(pg, rows([0.2, 0.1, -0.05]), width=0)
    pg.evaluate("document.getElementById('permtest').style.width = '100%'")
    pg.set_viewport_size({"width": 1200, "height": 800})
    pg.wait_for_timeout(500)
    wide = pg.evaluate(MEASURE)
    pg.set_viewport_size({"width": 420, "height": 800})
    pg.wait_for_timeout(800)
    narrow = pg.evaluate(MEASURE)
    assert len(wide["bars"]) == len(narrow["bars"]) == 3
    assert max(b["w"] for b in wide["bars"]) > max(b["w"] for b in narrow["bars"]) > 5, "the bars follow the width"
    assert narrow["hostRight"] <= 421
    pg.close()


def test_a_chart_drawn_in_a_hidden_tab_has_its_bars_once_the_tab_is_shown(browser, live):  # noqa: F811
    running, *_ = live
    pg = page(browser, running)
    pg.evaluate(HARNESS, [rows([0.2, 0.1, -0.05]), None, 700, 15])
    pg.evaluate("document.getElementById('permtest').hidden = true; document.getElementById('permtest').style.display = 'none'")
    pg.wait_for_timeout(300)
    pg.evaluate("document.getElementById('permtest').style.display = 'block'; document.getElementById('permtest').hidden = false; window.dispatchEvent(new Event('resize'))")
    pg.wait_for_timeout(800)
    assert len(pg.evaluate(MEASURE)["bars"]) == 3
    pg.close()


def test_a_real_four_asset_experiment_draws_bars_in_the_explain_view(browser, live):  # noqa: F811
    running, app, ids = live
    pg = page(browser, running)
    open_tab(pg, "ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "experiments")
    pg.wait_for_selector("#ml-results tr", timeout=60_000)
    pg.locator("#ml-results tr", has_text="four-assets").locator("button").first.click()
    _wait_until(pg, "document.querySelector('#ml-results [data-role=permutation]') !== null", timeout=120.0)
    pg.locator("#ml-results [data-role=permutation]").scroll_into_view_if_needed()
    pg.wait_for_timeout(800)
    n = pg.evaluate("[...document.querySelectorAll('#ml-results [data-role=permutation] .ec-chart svg path')].filter(p => { const b = p.getBBox(); return Math.abs(b.height - 14) < 1.5 && b.width >= 1; }).length")
    assert n >= 5, f"{n} bars drawn"
    assert "How to read this chart" in pg.locator("#ml-results [data-role=permutation]").inner_text()
    assert not pg.problems, pg.problems
    pg.close()


def test_a_real_two_asset_experiment_explains_why_there_is_no_chart(browser, live):  # noqa: F811
    running, app, ids = live
    pg = page(browser, running.two)
    open_tab(pg, "ml")
    pg.wait_for_selector("#ml .panel")
    _mode(pg, "ml", "experiments")
    pg.wait_for_selector("#ml-results tr", timeout=60_000)
    pg.locator("#ml-results tr").nth(1).locator("button").first.click()
    _wait_until(pg, "document.querySelector('#ml-results [data-role=permutation-empty]') !== null", timeout=120.0)
    text = pg.locator("#ml-results [data-role=permutation-empty]").inner_text()
    assert "No permutation importance" in text and "2 asset(s)" in text and "at least 3" in text
    assert pg.locator("#ml-results [data-role=permutation-empty] .ec-chart").count() == 0
    assert not pg.problems, pg.problems
    pg.close()
