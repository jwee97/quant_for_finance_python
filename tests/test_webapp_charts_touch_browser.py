"""Charts on a phone and under zoom, in a real browser: a tap shows a chart's tooltip and it stays until a tap elsewhere (a finger cannot hover); a line chart zooms in, moves, zooms from the
keyboard and resets, and its tooltip then reads the zoomed data; a bar chart answers a tap too; and nothing scrolls sideways at phone width."""

from __future__ import annotations

import pytest

from tests.test_webapp_browser import browser, open_tab  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_forecast_browser import _only, _open, live_forecast  # noqa: F401
from tests.test_webapp_workspaces_browser import _wait_until, live_ws  # noqa: F401

pytest.importorskip("playwright.sync_api")


@pytest.fixture()
def phone(browser, live_ws):  # noqa: F811
    context = browser.new_context(viewport={"width": 390, "height": 800}, has_touch=True, is_mobile=True)
    page = context.new_page()
    page.problems = []
    page.on("pageerror", lambda e: page.problems.append(f"pageerror: {e}"))
    page.on("console", lambda m: page.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    page.goto(live_ws.url)
    page.wait_for_selector("body[data-ready]")
    yield page
    assert not page.problems, page.problems
    context.close()


def _curve(page):
    open_tab(page, "rates")
    page.wait_for_selector("[data-role=rates-text]", timeout=60_000)
    _wait_until(page, "document.querySelector('[data-role=rates-text]').value.includes('3M 5.10')")
    page.click("#rates .runbar button.primary")
    page.wait_for_selector("[data-role=rates-parameters]", timeout=60_000)
    card = page.locator("#rates-results .chartcard").first
    card.scroll_into_view_if_needed()
    return card


def test_a_tap_shows_the_tooltip_and_it_stays_until_a_tap_elsewhere(phone):
    card = _curve(phone)
    box = _plot(card.locator(".ec-chart"))
    phone.touchscreen.tap(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5)
    tip = card.locator(".tip")
    tip.wait_for(state="visible")
    assert "Zero curve" in tip.inner_text() and "%" in tip.inner_text()
    phone.wait_for_timeout(600)
    assert tip.is_visible(), "a finger lifted is not a mouse that left: the tooltip stays to be read"
    first = tip.inner_text()
    phone.touchscreen.tap(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.5)
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .tip') && !document.querySelector('#rates-results .chartcard .tip').hidden")
    assert tip.inner_text() != first, "a tap further along reads another point"
    head = phone.locator("#rates-results h2").first.bounding_box()
    phone.touchscreen.tap(head["x"] + 5, head["y"] + 5)
    tip.wait_for(state="hidden")
    assert phone.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"), "nothing scrolls sideways on a phone"


def _plot(chart):
    """The plot area of an ECharts chart (without its axis labels and its zoom strip), as an absolute box: where a tap or a drag reads data."""
    return chart.evaluate("""el => { const c = echarts.getInstanceByDom(el), r = c.getModel().getComponent('grid').coordinateSystem.getRect(), b = el.getBoundingClientRect(); return { x: b.left + r.x, y: b.top + r.y, width: r.width, height: r.height }; }""")


def _window(page, selector):
    """The visible stretch of an ECharts line chart, as [start, end, first, last] along its x axis (None when the chart is the SVG kit's)."""
    return page.evaluate("""(sel) => { const el = document.querySelector(sel); const c = el && window.echarts && echarts.getInstanceByDom(el); if (!c) return null;
      const z = c.getOption().dataZoom[0], d = c.getOption().series[0].data; return [z.startValue, z.endValue, d[0][0], d[d.length - 1][0]]; }""", selector)


def test_a_line_chart_zooms_moves_and_resets_and_the_tooltip_reads_the_zoomed_data(phone):
    """The compact controls: a Reset button, a readout of how much is shown, a strip under the chart to drag and the keyboard (+ - 0). The old plus, minus and arrow buttons are gone: the strip and a pinch do their work."""
    card = _curve(phone)
    bar = card.locator(".zoombar")
    assert bar.locator("button").all_inner_texts() == ["Reset"] and bar.locator("button[aria-label='Show everything']").is_disabled()
    sel = "#rates-results .chartcard .ec-chart"
    a0, b0, first, last = _window(phone, sel)
    assert (a0, b0) == (first, last), "it opens on the whole curve"
    chart = card.locator(".ec-chart")
    chart.focus()
    phone.keyboard.press("+")
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') !== null")
    assert " of " in card.locator(".zoombar span").inner_text()
    shown = int(card.locator(".zoombar span").inner_text().split()[0])
    total = int(card.locator(".zoombar span").inner_text().split()[2])
    a1, b1, _, _ = _window(phone, sel)
    assert shown < total and (b1 - a1) < (b0 - a0) * 0.75 and bar.locator("button").is_enabled()
    phone.evaluate("""(sel) => { const c = echarts.getInstanceByDom(document.querySelector(sel)), z = c.getOption().dataZoom[0], d = c.getOption().series[0].data, end = d[d.length - 1][0], w = z.endValue - z.startValue;
      c.dispatchAction({ type: 'dataZoom', startValue: end - w, endValue: end }); }""", sel)                    # what dragging the strip to the right-hand end does
    box = _plot(chart)
    phone.touchscreen.tap(box["x"] + box["width"] - 6, box["y"] + box["height"] * 0.4)
    card.locator(".tip").wait_for(state="visible")
    assert "years" in card.locator(".tip").inner_text(), "the right edge of the zoomed chart is still the last maturity"
    chart.focus()
    phone.keyboard.press("-")
    a2, b2, _, _ = _window(phone, sel)
    assert (b2 - a2) > (b1 - a1), "minus zooms out"
    phone.keyboard.press("0")
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') === null")
    phone.keyboard.press("+")
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') !== null")
    card.locator("button[aria-label='Show everything']").tap()
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') === null")
    assert _window(phone, sel)[:2] == [first, last]


def test_a_bar_chart_answers_a_tap_too(phone):
    open_tab(phone, "futures")
    phone.wait_for_selector("#futures-results .tile", timeout=120_000)
    hit = phone.locator("#futures-results :is(.ec-chart, svg.chart rect[fill=transparent])").first
    hit.scroll_into_view_if_needed()
    box = hit.bounding_box()
    phone.touchscreen.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    _wait_until(phone, "[...document.querySelectorAll('#futures-results .tip')].some((t) => !t.hidden)")
    phone.wait_for_timeout(500)
    assert phone.evaluate("[...document.querySelectorAll('#futures-results .tip')].filter((t) => !t.hidden).length") == 1, "one tooltip at a time, and it stays"


def test_a_tap_on_the_forecast_part_of_the_fan_reads_the_forecast(browser, live_forecast):  # noqa: F811
    """The forecast is the right-hand part of the fan chart. On a phone it must be wide enough to tap, and a tap there must read the forecast (the median and the ranges), not only the past price."""
    context = browser.new_context(viewport={"width": 390, "height": 800}, has_touch=True, is_mobile=True)
    page = context.new_page()
    try:
        page.goto(live_forecast.url)
        page.wait_for_selector("body[data-ready]")
        _open(page)
        _only(page, ["riskmetrics"])
        page.locator("#forecast [data-role=forecast-effort] button", has_text="Quick").click()
        page.click("#forecast .runbar button.primary")
        page.wait_for_selector("[data-role=forecast-fan] svg.chart", timeout=240_000)
        card = page.locator("[data-role=forecast-fan]")
        card.scroll_into_view_if_needed()
        box = _plot(card.locator(".ec-chart"))
        share = page.evaluate("""() => { const c = echarts.getInstanceByDom(document.querySelector('[data-role=forecast-fan] .ec-chart')), o = c.getOption(), ml = o.series[0].markLine.data.find((d) => d.xAxis !== undefined),
          rect = c.getModel().getComponent('grid').coordinateSystem.getRect(), x = c.convertToPixel({ xAxisIndex: 0 }, ml.xAxis); return (rect.x + rect.width - x) / rect.width; }""")
        assert share > 0.3, f"the forecast is {share:.0%} of the chart: too narrow to tap on a phone"
        page.touchscreen.tap(box["x"] + box["width"] * 0.9, box["y"] + box["height"] * 0.5)
        tip = card.locator(".tip")
        tip.wait_for(state="visible")
        text = tip.inner_text()
        assert "Median forecast" in text and "90% range" in text and "Middle 50%" in text and " to " in text, text
        assert "Price" not in text.replace("Median forecast", ""), "a future date has no price yet"
        assert card.locator(".zoombar span").count() == 1 and card.locator("button[aria-label='Show everything']").is_enabled(), "it opens zoomed on the forecast; the whole history is one tap away"
        page.touchscreen.tap(box["x"] + box["width"] * 0.1, box["y"] + box["height"] * 0.5)
        _wait_until(page, "document.querySelector('[data-role=forecast-fan] .tip').textContent.includes('Price')")
    finally:
        context.close()
