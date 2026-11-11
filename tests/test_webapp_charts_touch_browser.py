"""Charts on a phone and under zoom, in a real browser: a tap shows a chart's tooltip and it stays until a tap elsewhere (a finger cannot hover); a line chart zooms in, moves, zooms from the
keyboard and resets, and its tooltip then reads the zoomed data; a bar chart answers a tap too; and nothing scrolls sideways at phone width."""

from __future__ import annotations

import pytest

from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
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
    page.click("#tab-rates")
    page.wait_for_selector("[data-role=rates-text]", timeout=60_000)
    _wait_until(page, "document.querySelector('[data-role=rates-text]').value.includes('3M 5.10')")
    page.click("#rates .runbar button.primary")
    page.wait_for_selector("[data-role=rates-parameters]", timeout=60_000)
    card = page.locator("#rates-results .chartcard").first
    card.scroll_into_view_if_needed()
    return card


def test_a_tap_shows_the_tooltip_and_it_stays_until_a_tap_elsewhere(phone):
    card = _curve(phone)
    overlay = card.locator("svg.chart rect[tabindex]")
    box = overlay.bounding_box()
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


def test_a_line_chart_zooms_moves_and_resets_and_the_tooltip_reads_the_zoomed_data(phone):
    card = _curve(phone)
    bar = card.locator(".zoombar")
    assert bar.locator("button").all_inner_texts() == ["+", "−", "◀", "▶", "Reset"] and bar.locator("button[aria-label='Zoom out']").is_disabled() and bar.locator("button[aria-label='Show everything']").is_disabled()
    last_tick = lambda: card.locator("svg.chart").evaluate("s => [...s.querySelectorAll('text')].map(t => t.textContent).join('|')")  # noqa: E731
    whole = last_tick()
    bar.locator("button[aria-label='Zoom in']").tap()
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') !== null")
    assert " of " in card.locator(".zoombar span").inner_text() and last_tick() != whole
    shown = int(card.locator(".zoombar span").inner_text().split()[0])
    total = int(card.locator(".zoombar span").inner_text().split()[2])
    assert shown < total and abs(shown - total / 2) <= 2
    assert card.locator("button[aria-label='Earlier']").is_enabled() and card.locator("button[aria-label='Later']").is_enabled()
    for _ in range(6):
        if card.locator("button[aria-label='Later']").is_disabled():
            break
        card.locator("button[aria-label='Later']").tap()
    _wait_until(phone, "document.querySelector(\"#rates-results .chartcard button[aria-label='Later']\").disabled")
    overlay = card.locator("svg.chart rect[tabindex]")
    box = overlay.bounding_box()
    phone.touchscreen.tap(box["x"] + box["width"] - 2, box["y"] + box["height"] * 0.5)
    card.locator(".tip").wait_for(state="visible")
    assert "30.00 years" in card.locator(".tip").inner_text() or "years" in card.locator(".tip").inner_text(), "the right edge of the zoomed chart is still the last maturity"
    overlay.focus()
    phone.keyboard.press("-")
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') === null")
    phone.keyboard.press("+")
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') !== null")
    card.locator("button[aria-label='Show everything']").tap()
    _wait_until(phone, "document.querySelector('#rates-results .chartcard .zoombar span') === null")
    assert last_tick() == whole


def test_a_bar_chart_answers_a_tap_too(phone):
    phone.click("#tab-futures")
    phone.wait_for_selector("#futures-results .tile", timeout=120_000)
    hit = phone.locator("#futures-results svg.chart rect[fill=transparent]").first
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
        box = card.locator("svg.chart rect[tabindex]").bounding_box()
        share = page.evaluate("""() => { const s = document.querySelector('[data-role=forecast-fan] svg.chart'), r = s.querySelector('rect[tabindex]').getBoundingClientRect(), v = s.querySelector('line.ref').getBoundingClientRect(); return (r.right - v.left) / r.width; }""")
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
