"""The Machine learning tab finds the runs again: a training started before the page was reloaded (or by a page that has since been closed) is listed with what it is doing, follows it to its end, opens the
experiment it made, and says plainly when a run was lost because the app stopped. Skipped when Playwright or Chromium is missing."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch
from tests.test_webapp_browser import browser  # noqa: F401  (the module's Chromium fixture)
from tests.test_webapp_workspaces_browser import _wait_until

sync_api = pytest.importorskip("playwright.sync_api")

BODY = {"dataset": {"feature_groups": ["price", "momentum"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge", "params": {"alpha": 30}}, "name": "left running", "explain": False}


@pytest.fixture(scope="module")
def live_runs(tmp_path_factory):
    config = load_config()
    root = tmp_path_factory.mktemp("runs_root")
    (root / "data").mkdir()
    (root / "data" / "raw").symlink_to(config.root / "data" / "raw", target_is_directory=True)
    (root / "docs").symlink_to(config.root / "docs", target_is_directory=True)
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path_factory.mktemp("runs_prices"), fetch=_fetch), default_loader=lambda c: bundle), root=root)
    running = server.start(app, "127.0.0.1", 0)
    running.app, running.root = app, root
    yield running
    running.stop()


def _page(browser, live):  # noqa: F811
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.goto(live.url)
    pg.wait_for_selector(".chip")
    return pg


def _status(page, label):
    """The status of the run with this label in the card."""
    return page.locator("#ml-results .runs li", has_text=label).first.get_attribute("data-status")


def test_a_reloaded_page_finds_a_run_follows_it_to_its_end_and_opens_the_experiment_it_made(browser, live_runs):  # noqa: F811
    app, root = live_runs.app, live_runs.root
    gate, started = threading.Event(), threading.Event()

    def blocker(progress):
        progress("holding the worker")
        started.set()
        gate.wait(120)
        return {"held": True}

    first = app.submit_task("ml: a long search", blocker, "ml")
    assert started.wait(30)
    job = app.extra_post["/api/ml/train"](BODY)["job"]                              # queued behind the first: the single worker is busy
    built = app.submit_task("dataset: sample", lambda progress: {"rows": 10}, "ml")  # a dataset build makes no experiment
    (root / "data" / "user" / "jobs" / "aaaaaaaaaaaa.json").write_text(json.dumps({"id": "aaaaaaaaaaaa", "label": "ml: lightgbm on crypto", "kind": "ml", "status": "interrupted", "pid": 1,
                                                                                    "message": "the app stopped while this was running: training fold 3 of 9", "error": None, "created": time.time() - 120, "finished": time.time() - 60, "ref": None}))
    page = _page(browser, live_runs)                                                # a page that did not start either job
    try:
        page.click("#tab-ml")
        page.wait_for_selector("#ml-results .runs li", timeout=60_000)
        _wait_until(page, "document.querySelectorAll('#ml-results .runs li').length >= 3")
        assert (_status(page, "a long search"), _status(page, "left running"), _status(page, "lightgbm on crypto")) == ("running", "queued", "interrupted")
        text = page.inner_text("#ml-results .runs")
        assert "Runs in the background" in text and "Running · a long search" in text and "holding the worker" in text and "Waiting · left running" in text
        assert "Stopped · lightgbm on crypto" in text and "training fold 3 of 9" in text, "a run lost when the app stopped says so, and what it was doing"
        assert "go on at the app's computer if you close this page" in text and "has to be started again from the beginning" in text
        page.reload()                                                               # reloading changes nothing: the list comes from the server (the page reopens on the tab it was on)
        page.wait_for_selector("#ml-results .runs li", timeout=60_000)
        assert page.locator("#ml-results .runs li[data-status=running]").count() == 1

        gate.set()                                                                  # the first run ends; the training starts, then ends
        _wait_until(page, "Array.from(document.querySelectorAll('#ml-results .runs li[data-status=done]')).some((li) => li.textContent.includes('left running') && !!li.querySelector('button'))", 300.0)
        done = page.locator("#ml-results .runs li[data-status=done]", has_text="left running")
        assert done.count() == 1 and "Finished · left running" in done.inner_text() and "ended just now" in done.inner_text() and done.locator("button[data-action=open-run]").inner_text() == "Open"
        assert page.locator("#ml-results .runs li[data-status=done]", has_text="a long search").locator("button[data-action=open-run]").inner_text() == "Show the experiments", "a run that made no single experiment points at the list"
        _wait_until(page, "Array.from(document.querySelectorAll('#ml-results .runs li[data-status=done]')).some((li) => li.textContent.includes('dataset: sample') || li.textContent.includes('sample'))", 120.0)
        sample = page.locator("#ml-results .runs li[data-status=done]", has_text="sample")
        assert sample.count() == 1 and sample.locator("button[data-action=open-run]").count() == 0, "a dataset build made no experiment, so there is nothing to open"
        done.locator("button[data-action=open-run]").click()
        page.wait_for_selector("#ml-results .head h2", timeout=60_000)
        assert "left running" in page.inner_text("#ml-results"), "Open shows the experiment the run made"
        assert page.locator("#view-ml .tabsmall button[aria-selected=true]").get_attribute("data-mode") == "train"
        assert [r for r in app.extra_get["/api/ml/experiments"]({})["rows"]][0]["name"] == "left running"
        assert not [p for p in page.problems if "status of 4" not in p], page.problems
    finally:
        gate.set()
        page.close()
    assert app.job(first["job"])["status"] == "done" and app.job(job)["status"] == "done" and app.job(built["job"])["status"] == "done"


def test_a_page_never_waits_for_a_run_that_was_lost_and_a_stuck_row_can_be_cleared(browser, live_runs):  # noqa: F811
    """After a crash the record of a run can say running for a while (another process may have the old process id), or interrupted. The page must stop waiting for the second, and the person must be
    able to take either off the list."""
    app, root = live_runs.app, live_runs.root
    app.jobs_clear({"ended": True})
    folder = root / "data" / "user" / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])           # has the process id the crashed app had
    try:
        now = time.time()

        def record(job_id, status, label, **more):
            return json.dumps({"id": job_id, "label": label, "kind": "ml", "status": status, "message": "training fold 3 of 9", "error": None, "created": now - 300, "finished": None, "ref": None,
                               "pid": other.pid, "process": "the app before the crash", "updated": now, **more})

        (folder / "aaaaaaaaaaaa.json").write_text(record("aaaaaaaaaaaa", "running", "ml: stuck by a crash"))
        (folder / "bbbbbbbbbbbb.json").write_text(record("bbbbbbbbbbbb", "interrupted", "ml: lost one", message="the app stopped while this was running: training fold 2 of 9", finished=now - 100))
        (folder / "cccccccccccc.json").write_text(record("cccccccccccc", "error", "ml: failed one", error="boom", finished=now - 50))
        page = _page(browser, live_runs)
        try:
            page.click("#tab-ml")
            page.wait_for_selector("#ml-results .runs li", timeout=60_000)
            assert _status(page, "stuck by a crash") == "running" and _status(page, "lost one") == "interrupted" and _status(page, "failed one") == "error"
            assert page.locator("#ml-results .runs li", has_text="stuck by a crash").locator("button[data-action=clear-run]").inner_text() == "Not running? Clear"
            assert page.locator("#ml-results .runs li", has_text="lost one").locator("button[data-action=clear-run]").inner_text() == "Clear"
            lost = page.evaluate("""async () => { try { await window.QL.waitJob("bbbbbbbbbbbb", null, 50); return "returned"; } catch (e) { return e.message; } }""")
            assert "training fold 2 of 9" in lost and "start it again" in lost, "a page waiting for a lost run is told, instead of waiting for ever"
            unknown = page.evaluate("""async () => { try { await window.QL.waitJob("dddddddddddd", null, 50); return "returned"; } catch (e) { return e.message; } }""")
            assert "no longer knows this run" in unknown

            assert page.locator("#ml-results .runs button[data-action=clear-ended]").count() == 1, "two ended runs: one button clears both"
            page.click("#ml-results .runs button[data-action=clear-ended]")
            _wait_until(page, "document.querySelectorAll('#ml-results .runs li').length === 1")
            assert _status(page, "stuck by a crash") == "running" and not (folder / "bbbbbbbbbbbb.json").exists() and not (folder / "cccccccccccc.json").exists()
            page.locator("#ml-results .runs li", has_text="stuck by a crash").locator("button[data-action=clear-run]").click()
            _wait_until(page, "document.querySelectorAll('#ml-results .runs li').length === 0")
            assert not (folder / "aaaaaaaaaaaa.json").exists() and page.locator("#ml-results .runs").count() == 0, "the card goes when nothing is left"
            assert not [p for p in page.problems if "status of 4" not in p], page.problems
        finally:
            page.close()
    finally:
        other.kill()
        other.wait()
