"""Cancelling, retrying and linking jobs: what the Background jobs page offers on top of the job log. A cancelled run keeps no result, a run that cannot be stopped says so, and a retry starts again from the
beginning with the request that started the first run, naming the one it repeats."""

from __future__ import annotations

import threading
import time

import pytest

from src.webapp.api import ApiError
from src.utils.config import load_config
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def _wait_status(app, job_id, wanted, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        view = app.job(job_id)
        if view["status"] in wanted:
            return view
        time.sleep(0.05)
    raise AssertionError(f"job stayed {app.job(job_id)['status']}")


def test_a_job_that_has_not_started_is_taken_off_the_queue_and_never_runs(app):
    gate, ran = threading.Event(), []
    first = app.submit_task("blocker", lambda progress: gate.wait(30) and {"x": 1}, kind="task")["job"]
    second = app.submit_task("waiting", lambda progress: ran.append(1) or {"x": 2}, kind="task")["job"]
    _wait_status(app, first, ("running",))
    assert app.job(second)["status"] == "queued"
    out = app.jobs_cancel({"id": second})
    assert out["status"] == "cancelled" and "before it started" in out["message"]
    gate.set()
    _wait(app, first)
    time.sleep(0.3)
    assert ran == [], "a cancelled job never starts"
    view = app.job(second)
    assert view["status"] == "cancelled" and view["result"] is None


def test_a_running_job_stops_at_its_next_progress_message_and_keeps_no_result(app):
    started, steps = threading.Event(), []

    def work(progress):
        started.set()
        for i in range(200):
            progress(f"step {i}")
            steps.append(i)
            time.sleep(0.02)
        return {"done": True}

    job = app.submit_task("long", work, kind="task")["job"]
    assert started.wait(10)
    listed = {j["id"]: j for j in app.jobs_list()["jobs"]}[job]
    assert listed["cancellable"] is True and listed["cancelling"] is False
    app.jobs_cancel({"id": job})
    assert {j["id"]: j for j in app.jobs_list()["jobs"]}[job]["cancelling"] is True
    view = _wait_status(app, job, ("cancelled",))
    assert view["result"] is None and "cancelled" in view["message"] and len(steps) < 200


def test_a_job_that_finishes_before_it_can_stop_is_cancelled_and_its_result_is_discarded(app):
    gate = threading.Event()

    def work(progress):                                                         # never reports progress: nothing to stop it at
        gate.wait(30)
        return {"answer": 42}

    job = app.submit_task("silent", work, kind="task")["job"]
    _wait_status(app, job, ("running",))
    app.jobs_cancel({"id": job})
    gate.set()
    view = _wait_status(app, job, ("cancelled", "done"))
    assert view["status"] == "cancelled" and view["result"] is None and "discarded" in view["message"]


def test_cancelling_what_cannot_be_cancelled_is_an_api_error(app):
    done = app.submit_task("quick", lambda progress: {"x": 1}, kind="task")["job"]
    _wait(app, done)
    with pytest.raises(ApiError, match="already done"):
        app.jobs_cancel({"id": done})
    with pytest.raises(ApiError, match="not going in this app"):
        app.jobs_cancel({"id": "0123456789ab"})


def test_a_cancelled_job_is_remembered_on_disk_and_listed_as_cancelled(app):
    gate = threading.Event()
    blocker = app.submit_task("blocker", lambda progress: gate.wait(30) and {}, kind="task")["job"]
    queued = app.submit_task("queued", lambda progress: {}, kind="task")["job"]
    app.jobs_cancel({"id": queued})
    gate.set()
    _wait(app, blocker)
    kept = app.joblog.get(queued)
    assert kept["status"] == "cancelled" and kept["result"] is None
    row = {j["id"]: j for j in app.jobs_list()["jobs"]}[queued]
    assert row["status"] == "cancelled" and row["cancellable"] is False


def test_a_failed_backtest_is_started_again_with_its_own_request_and_names_the_run_it_repeats(app):
    body = {"tickers": ["AAA", "BBB", "CCC"], "models": [{"name": "expression", "params": {"expr": "-ret(5)"}}], "allocator": "score_stack", "session": "0123456789ab"}
    first = app.with_request("/api/backtest", app.submit, dict(body))["job"]
    view = _wait(app, first)
    assert view["status"] == "done"
    assert app.joblog.get(first)["session"] == "0123456789ab", "the run is linked to the session its request named"
    row = {j["id"]: j for j in app.jobs_list()["jobs"]}[first]
    assert row["retryable"] is False and row["session"] == "0123456789ab"
    with pytest.raises(ApiError, match="only a failed, cancelled or interrupted"):
        app.jobs_retry({"id": first})
    app.jobs[first].status = "error"                                              # as if it had failed
    app.jobs[first].save(final=True)
    row = {j["id"]: j for j in app.jobs_list()["jobs"]}[first]
    assert row["retryable"] is True
    again = app.jobs_retry({"id": first})["job"]
    assert again != first and _wait(app, again)["status"] == "done"
    row = {j["id"]: j for j in app.jobs_list()["jobs"]}[again]
    assert row["retry_of"] == first and row["session"] == "0123456789ab"
    assert app.jobs[again].request["path"] == "/api/backtest", "a retried run keeps the original request, so that it can be repeated in its turn"


def test_a_run_whose_kind_has_files_or_keys_behind_it_is_not_offered_a_retry(app):
    job = app.with_request("/api/anything", lambda b: app.submit_task("data", lambda progress: (_ for _ in ()).throw(ValueError("boom")), kind="data"), {"x": 1})["job"]
    _wait(app, job)
    row = {j["id"]: j for j in app.jobs_list()["jobs"]}[job]
    assert row["status"] == "error" and row["retryable"] is False
    with pytest.raises(ApiError, match="cannot be started again"):
        app.jobs_retry({"id": job})


def test_an_oversized_request_is_not_kept_with_the_job(app):
    big = {"tickers": ["AAA", "BBB", "CCC"], "models": [{"name": "momentum"}], "pad": "x" * 30_000}
    job = app.with_request("/api/backtest", app.submit, big)["job"]
    _wait(app, job)
    assert "request" not in next(r for r in app.joblog.records() if r["id"] == job)
