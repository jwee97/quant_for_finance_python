"""Two workers instead of one: a fetch of statements or prices (which waits for a provider) is not held up by a long search, and a backtest is not held up by a fetch; work in one lane still waits its
turn in that lane."""

from __future__ import annotations

import threading
import time

import pytest

from tests.test_fund_api import make_app, post, wait
from tests.test_fund_sec_edgar import sec_app  # noqa: F401  (the fixture: an app whose real database can be filled from a stand-in for the SEC)


def _status(app, started) -> str:
    return app.job(started["job"])["status"]


def _until(condition, seconds=10.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_a_job_on_the_data_lane_runs_while_a_long_job_holds_the_compute_lane(tmp_path):
    app = make_app(tmp_path)
    release = threading.Event()
    long = app.submit_task("a long search", lambda progress: (release.wait(30), "searched")[1])
    assert _until(lambda: _status(app, long) == "running")
    fetch = app.submit_task("a fetch", lambda progress: "fetched", "fundamentals", "data")
    assert _until(lambda: _status(app, fetch) == "done") and app.job(fetch["job"])["result"] == "fetched"
    assert _status(app, long) == "running"                                                      # the search is still going: the fetch did not wait for it
    behind = app.submit_task("another backtest", lambda progress: "ran")
    time.sleep(0.3)
    assert _status(app, behind) == "queued"                                                      # work in the compute lane still takes its turn
    release.set()
    assert wait(app, long["job"])["status"] == "done" and wait(app, behind["job"])["result"] == "ran"


def test_two_jobs_on_the_data_lane_take_turns_because_they_share_one_provider_allowance(tmp_path):
    app = make_app(tmp_path)
    release, order = threading.Event(), []

    def first(progress):
        order.append("first starts")
        release.wait(30)
        order.append("first ends")
        return 1

    def second(progress):
        order.append("second starts")
        return 2
    a = app.submit_task("fetch one", first, "fundamentals", "data")
    assert _until(lambda: order == ["first starts"])
    b = app.submit_task("fetch two", second, "fundamentals", "data")
    time.sleep(0.3)
    assert _status(app, b) == "queued" and order == ["first starts"]
    release.set()
    assert wait(app, a["job"])["status"] == "done" and wait(app, b["job"])["status"] == "done"
    assert order == ["first starts", "first ends", "second starts"]


def test_a_lane_that_does_not_exist_is_refused_and_an_error_in_a_job_does_not_stop_its_lane(tmp_path):
    app = make_app(tmp_path)
    with pytest.raises(ValueError, match="lane must be"):
        app.submit_task("x", lambda progress: 1, "task", "gpu")

    def broken(progress):
        raise RuntimeError("the provider fell over")
    bad = app.submit_task("a fetch that fails", broken, "fundamentals", "data")
    assert wait(app, bad["job"])["status"] == "error" and "fell over" in app.job(bad["job"])["error"]
    good = app.submit_task("the next fetch", lambda progress: "fine", "fundamentals", "data")
    assert wait(app, good["job"])["result"] == "fine"


def test_fetching_statements_through_the_route_does_not_wait_for_a_search_that_is_running(sec_app):  # noqa: F811
    app, fake = sec_app
    app.fundamentals.fetch = lambda symbols, start: {}                                          # the market-data source answers with nothing: no test here goes to the network
    release = threading.Event()
    long = app.submit_task("a long discovery run", lambda progress: (release.wait(60), "ok")[1], "fundamentals")
    assert _until(lambda: _status(app, long) == "running")
    post(app, "/api/fund/companies/add", symbols=["AAPL"], provider="sec_edgar")
    started = post(app, "/api/fund/sync", provider="sec_edgar", budget=2)
    view = wait(app, started["job"], timeout=30)
    assert view["status"] == "done" and view["result"]["fetched"] == 1 and _status(app, long) == "running"
    prices = post(app, "/api/fund/prices", security_ids=[1])
    assert wait(app, prices["job"], timeout=30)["status"] in ("done", "error") and _status(app, long) == "running"           # prices are on the data lane as well
    study = post(app, "/api/fund/simulate", db="demo", companies=30, start_year=2012, end_year=2018, seed=1)
    time.sleep(0.5)
    assert _status(app, study) == "queued"                                                       # everything else waits for the compute lane
    release.set()
    assert wait(app, long["job"])["status"] == "done"
    assert wait(app, study["job"], timeout=300)["status"] == "done"
