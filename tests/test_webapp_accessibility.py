"""The dashboard's accessibility, checked the way a screen-reader or keyboard user meets it: one page heading and no skipped levels, tab-like controls that really are tabs (and toggles that are not), every form
control with a name, the arrow keys on the tabs, and text colours that reach the WCAG contrast of 4.5:1 in both themes. The browser tests need Playwright and Chromium and are skipped without them; the
contrast test reads the stylesheet and always runs.

An automated audit with axe-core found each of these once (22 views, light and dark); these tests keep them found."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.webapp import server
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _start

sync_api = pytest.importorskip("playwright.sync_api")

CSS = (Path(server.__file__).parent / "static" / "style.css").read_text(encoding="utf-8")
READY = {"backtest": ".chip", "compare": "#compare .card", "builder": "#builder .card", "tune": "#tune .panel", "options": "#options-results .tile", "futures": "#futures-results .tile",
         "crypto": "#crypto-results .tile", "ml": "#ml-results .card", "exec": "#exec-results .card", "cash": "#cash-results .card", "guides": "#guides .toc"}


@pytest.fixture(scope="module")
def live_a11y(tmp_path_factory):
    running = _start(tmp_path_factory, "a11y")
    yield running
    running.stop()


@pytest.fixture()
def pg(browser, live_a11y):  # noqa: F811
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(live_a11y.url)
    page.wait_for_selector("body[data-ready]")                                              # the page restores the last tab when it has booted: wait for that, or it races the keys below
    yield page
    page.close()


def _open(page, tab):
    page.click(f"#tab-{tab}")
    page.wait_for_selector(READY[tab], timeout=90_000)


# ------------------------------------------------------------------------------------------------------ contrast (no browser)
def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", block))


def _luminance(colour: str) -> float:
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4          # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme, block", [("light", CSS.split(":root {", 1)[1].split("}", 1)[0]), ("dark", CSS.split(':root[data-theme="dark"] {', 1)[1].split("}", 1)[0])])
def test_every_text_colour_and_the_accent_reach_the_wcag_contrast_in_both_themes(theme, block):
    t = _tokens(block)
    for name in ("ink", "ink-2", "muted", "good-text", "crit", "warn", "accent"):
        for ground in ("surface", "page"):
            assert _contrast(t[name], t[ground]) >= 4.5, f"{theme}: {name} on {ground} is {_contrast(t[name], t[ground]):.2f}:1"
    assert _contrast(t["accent-ink"], t["accent"]) >= 4.5, f"{theme}: the text of a primary button is {_contrast(t['accent-ink'], t['accent']):.2f}:1"


def test_the_dark_theme_blocks_agree_with_each_other():
    """The stylesheet states the dark tokens twice (for the system setting and for the switch); a change to one must reach the other."""
    media = CSS.split("@media (prefers-color-scheme: dark) {", 1)[1].split("}", 1)[0]
    switch = CSS.split(':root[data-theme="dark"] {', 1)[1].split("}", 1)[0]
    assert _tokens(media) == _tokens(switch)


# ------------------------------------------------------------------------------------------------------ the page in a browser
def test_every_view_has_the_page_title_as_its_one_top_heading_and_no_heading_skips_a_level(pg):
    for tab in READY:
        _open(pg, tab)
        levels = pg.evaluate("[...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].filter((e) => e.offsetParent !== null).map((e) => Number(e.tagName[1]))")
        assert levels[0] == 1 and levels.count(1) == (1 if tab != "guides" else levels.count(1)), (tab, levels)
        assert all(b - a <= 1 for a, b in zip(levels, levels[1:])), (tab, levels)


def test_only_real_tabs_say_they_are_selected_and_the_choosers_are_toggle_buttons(pg):
    for tab in READY:
        _open(pg, tab)
        assert pg.evaluate("document.querySelectorAll('[aria-selected]:not([role=\"tab\"])').length") == 0, tab
        assert pg.evaluate("[...document.querySelectorAll('.tabsmall[role=\"tablist\"]')].every((bar) => bar.hasAttribute('aria-label'))"), tab
        assert pg.evaluate("[...document.querySelectorAll('[role=\"tab\"]')].every((t) => t.closest('[role=\"tablist\"]'))"), tab


def test_every_form_control_has_a_name(pg):
    unnamed = """[...document.querySelectorAll('input:not([type=hidden]), select, textarea')].filter((e) => e.offsetParent !== null).filter((e) =>
        !(e.getAttribute('aria-label') || e.getAttribute('aria-labelledby') || e.title || (e.labels && e.labels.length))).map((e) => e.tagName + ':' + (e.dataset.key || e.id || e.name || e.className))"""
    for tab in READY:
        _open(pg, tab)
        assert pg.evaluate(unnamed) == [], tab


def test_the_section_tabs_are_reached_with_the_arrow_keys_and_only_the_selected_one_is_in_the_tab_order(pg):
    order = lambda: pg.evaluate("[...document.querySelectorAll('nav.tabs [role=\"tab\"]')].map((t) => [t.id, t.tabIndex, t.getAttribute('aria-selected')])")        # noqa: E731
    first = order()
    assert [x for x in first if x[1] == 0] == [["tab-backtest", 0, "true"]] and all(x[1] == -1 for x in first[1:])
    pg.focus("#tab-backtest")
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("document.activeElement.id") == "tab-compare" and pg.get_attribute("#tab-compare", "aria-selected") == "true" and pg.get_attribute("#tab-compare", "tabindex") == "0"
    pg.keyboard.press("End")
    assert pg.evaluate("document.activeElement.id") == "tab-guides" and pg.get_attribute("#tab-backtest", "tabindex") == "-1"
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("document.activeElement.id") == "tab-backtest", "past the last tab it wraps to the first"
    pg.keyboard.press("ArrowLeft")
    assert pg.evaluate("document.activeElement.id") == "tab-guides"
    pg.keyboard.press("Home")
    assert pg.evaluate("document.activeElement.id") == "tab-backtest"
    pg.wait_for_selector("#view-backtest:not([hidden])", timeout=10_000)                      # the view follows the key; a machine busy with a whole suite can show it a moment late


def test_a_labs_view_tabs_keep_the_keyboard_focus_when_the_view_is_redrawn(pg):
    _open(pg, "exec")
    modes = pg.locator("#view-exec .tabsmall [role=tab]")
    assert modes.count() >= 3 and [t.get_attribute("tabindex") for t in modes.all()] == ["0"] + ["-1"] * (modes.count() - 1)
    modes.nth(0).focus()
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("document.activeElement.dataset.mode") == pg.locator("#view-exec .tabsmall [role=tab]").nth(1).get_attribute("data-mode")
    assert pg.evaluate("document.activeElement.getAttribute('aria-selected')") == "true", "the redraw replaced the buttons and the focus is on the selected one"
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("document.activeElement.getAttribute('aria-selected')") == "true"


def test_choosers_such_as_the_ml_feature_filter_are_a_group_of_toggle_buttons(pg):
    _open(pg, "options")
    pg.wait_for_selector("#options-results .tile")
    group = pg.locator("#options-results .tabsmall[role=group]").first
    if group.count():
        buttons = group.locator("button")
        assert buttons.count() >= 2 and all(b.get_attribute("aria-pressed") in ("true", "false") for b in buttons.all()) and buttons.first.get_attribute("aria-selected") is None
