"""The shared chart and table kit in a real desktop browser: legends that hide and show series, range presets, drag-to-zoom and double-click reset, a chart opened large and closed with Esc, a chart saved
as SVG, PNG and CSV, and tables that sort by their headings and filter when they are long."""

from __future__ import annotations

import pytest

from tests.test_webapp_browser import _run_and_wait, browser, live  # noqa: F401  (the module's fixtures)

pytest.importorskip("playwright.sync_api")


@pytest.fixture()
def page(browser, live):  # noqa: F811
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live.url)
    pg.wait_for_selector(".chip")
    yield pg
    assert not pg.problems, pg.problems
    pg.close()


def _growth(page):
    _run_and_wait(page)
    card = page.locator(".chartcard:has(.seg button:has-text('Log'))").first
    card.scroll_into_view_if_needed()
    return card


def test_a_legend_entry_hides_and_shows_its_series_and_the_last_one_stays(page):
    card = _growth(page)
    names = card.locator(".legend [role=button]")
    assert names.count() >= 2
    lines = lambda: card.locator("svg.chart path[fill=none]").count()  # noqa: E731
    before = lines()
    names.nth(1).click()
    assert card.locator(".legend [role=button]").nth(1).get_attribute("aria-pressed") == "false" and lines() == before - 1
    assert page.evaluate("document.activeElement.dataset.series") == "1", "the keyboard keeps its place in the legend after the chart is redrawn"
    card.locator(".legend [role=button]").nth(1).click()
    assert lines() == before
    for k in range(1, names.count()):
        card.locator(f".legend [data-series='{k}']").click()
    assert card.locator(".legend [aria-pressed=true]").count() >= 1 and lines() >= 1, "the last visible series cannot be hidden"


def test_a_range_preset_and_a_drag_zoom_narrow_the_chart_and_a_double_click_shows_everything(page):
    card = _growth(page)
    presets = card.locator(".rangebar button")
    assert "All" in presets.all_inner_texts() and presets.filter(has_text="1Y").count() == 1
    assert presets.filter(has_text="All").get_attribute("aria-pressed") == "true"
    presets.filter(has_text="1Y").click()
    assert "of" in card.locator(".zoombar span").inner_text() and card.locator(".rangebar button:has-text('1Y')").get_attribute("aria-pressed") == "true"
    card.locator(".rangebar button:has-text('All')").click()
    assert card.locator(".zoombar span").count() == 0
    box = card.locator("svg.chart rect[tabindex]").bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] * 0.3, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.45, y, steps=6)
    assert card.locator(".brush").count() == 1, "the stretch being selected is shaded while the mouse is down"
    page.mouse.move(box["x"] + box["width"] * 0.6, y, steps=6)
    page.mouse.up()
    page.wait_for_selector(".zoombar span")
    assert card.locator(".brush").count() == 0 and "points" in card.locator(".zoombar span").inner_text()
    card.locator("svg.chart rect[tabindex]").dblclick()
    assert card.locator(".zoombar span").count() == 0


def test_a_chart_opens_large_and_closes_with_escape(page):
    card = _growth(page)
    small = card.locator("svg.chart").bounding_box()["height"]
    card.locator("button[aria-label='Expand chart']").click()
    assert "expanded" in card.get_attribute("class") and page.locator(".chartback").count() == 1
    assert card.locator("svg.chart").bounding_box()["height"] > small * 1.3
    page.keyboard.press("Escape")
    assert "expanded" not in card.get_attribute("class") and page.locator(".chartback").count() == 0
    assert card.locator("svg.chart").bounding_box()["height"] == pytest.approx(small, abs=2)
    assert page.evaluate("document.activeElement.getAttribute('aria-label')") == "Expand chart"


def test_a_chart_is_saved_as_svg_png_and_csv(page):
    card = _growth(page)
    save = card.locator("button[aria-label='Save chart']")
    with page.expect_download() as d:
        save.click()
        page.click("text=Save as SVG")
    svg = open(d.value.path(), encoding="utf-8").read()
    assert d.value.suggested_filename.endswith(".svg") and svg.startswith("<svg") and "stroke:" in svg
    with page.expect_download() as d:
        save.click()
        page.click("text=Save as PNG")
    assert open(d.value.path(), "rb").read(8) == b"\x89PNG\r\n\x1a\n" and d.value.suggested_filename.endswith(".png")
    with page.expect_download() as d:
        save.click()
        page.click("text=Save numbers (CSV)")
    rows = open(d.value.path(), encoding="utf-8").read().strip().splitlines()
    assert len(rows) > 5 and rows[0].count(",") >= 1 and d.value.suggested_filename.endswith(".csv")
    save.click()
    assert page.locator(".savemenu").count() == 1
    page.keyboard.press("Escape")
    assert page.locator(".savemenu").count() == 0


def test_a_table_sorts_by_its_headings_and_a_long_one_filters(page):
    card = _growth(page)
    card.locator(".seg button:has-text('Table')").click()
    table = card.locator("table.data")
    table.wait_for()
    first_col = lambda: table.locator("tbody tr td:nth-child(1)").all_inner_texts()  # noqa: E731
    original = first_col()
    assert len(original) > 14 and card.locator(".tbltools input[aria-label='Filter rows']").count() == 1
    head = table.locator("th").nth(1)
    head.locator("button").click()
    assert head.get_attribute("aria-sort") in ("ascending", "descending")
    head.locator("button").click()
    assert head.get_attribute("aria-sort") in ("ascending", "descending")
    head.locator("button").click()
    assert head.get_attribute("aria-sort") is None and first_col() == original, "a third press restores the original order"
    card.locator(".tbltools input").fill(original[3])
    assert table.locator("tbody tr").count() >= 1 and "of" in card.locator(".tbltools .hint").inner_text()
    card.locator(".tbltools input").fill("zzzzqq")
    assert "No row matches" in table.inner_text()


LIVE = {"context": {"model": "lstm", "fold": 2, "folds": 5}, "fits": 2, "updated": 0,
        "done": [{"n": 1, "context": {"model": "lstm", "fold": 1, "folds": 5}, "epochs_run": 12, "epochs": 40, "seconds": 31.2, "best_epoch": 9, "best_val": 0.4123, "stopped_early": True, "final_train": 0.39}],
        "run": {"n": 2, "context": {"model": "lstm", "fold": 2, "folds": 5}, "epochs": 40, "patience": 6, "device": "cpu", "n_params": 18305, "n_train": 5000, "n_val": 900, "batch_size": 128, "name": None,
                "batch": {"epoch": 8, "index": 12, "of": 40, "loss": 0.3951},
                "history": {"epoch": [0, 1, 2, 3, 4, 5, 6, 7], "train": [0.9, 0.7, 0.6, 0.52, 0.47, 0.44, 0.41, 0.39], "val": [0.95, 0.8, 0.7, 0.62, 0.6, 0.601, 0.605, 0.612], "lr": [1e-3] * 7 + [5e-4], "grad": [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5], "seconds": [2.1] * 8},
                "current": {"epoch": 8, "epochs": 40, "train": 0.39, "val": 0.612, "best": 0.6, "best_epoch": 5, "lr": 5e-4, "grad": 0.5, "seconds": 2.1, "bad": 3, "patience": 6, "improved": False, "eta_seconds": 67.2, "peak_mb": 312.0}}}


def test_the_live_training_panel_says_how_the_network_is_learning(page):
    page.evaluate("(live) => { const d = document.createElement('div'); d.id = 'probe'; document.body.appendChild(d); QL.livePanel(d, live); }", LIVE)
    panel = page.locator("#probe [data-role=live-training]")
    text = panel.inner_text()
    low = text.lower()                                                                    # the tile labels are set in capitals by the stylesheet
    assert "lstm · fold 2 of 5" in low and "18,305 parameters" in low and "epoch 8 of 40 finished" in low and "batch 12 of 40" in low
    assert "training loss" in low and "validation loss" in low and "best validation loss" in low and "epoch 5" in low and "3 of 6" in low and "about 67 s left" in low
    assert "just halved" in low and "312 mb" in low, "the learning-rate drop and the memory are shown"
    assert "not improving: no new best validation loss for 3 epoch(s) (best at epoch 5)" in low and "memorise" in low, "it says that the validation loss turned while the training loss kept falling"
    bar = panel.locator("[role=progressbar]")
    assert bar.get_attribute("aria-valuenow") == str(round(((8 + 12 / 40) / 40) * 100)) and bar.get_attribute("aria-label") == "Training progress"
    assert panel.locator("svg.chart").count() == 1 and panel.locator(".legend span").count() == 2, "the loss curves are drawn: training and validation"
    panel.locator("details.lt-done summary").click()
    earlier = panel.locator("details.lt-done table.data").inner_text()
    el = earlier.lower()
    assert "fold 1" in el and "12 of 40" in el and "early, by patience" in el and "0.4123" in el
    page.evaluate("(live) => { const d = document.getElementById('probe'); d.replaceChildren(); const l = JSON.parse(JSON.stringify(live)); l.run.current.improved = true; l.run.current.bad = 0; l.run.history.val[7] = 0.55; l.run.current.val = 0.55; QL.livePanel(d, l); }", LIVE)
    assert "improving: the validation loss reached a new best" in page.inner_text("#probe").lower()
