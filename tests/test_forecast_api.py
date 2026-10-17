"""The Forecast tab's endpoints: the plan of methods the form lists, a forecast-and-evaluation run as a job on prices the app resolves like any other tab (kept in the job list, touching no trial count), the requests
refused before a job starts, a Stop that keeps what is scored, a history too short, and the script being served."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from src.forecast import METHODS
from src.forecast.base import Forecast, Forecaster
from src.utils.config import load_config
from src.webapp import api, server
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch


def make_app(tmp_path, n):
    config = load_config()
    bundle = make_bundle(n_assets=4, n=n, seed=3, alpha=0.0)
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=lambda c: bundle), root=tmp_path)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return make_app(tmp_path_factory.mktemp("forecast_app"), 1400)


def _wait(app, job, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        view = app.job(job)
        if view["status"] in ("done", "error", "interrupted"):
            return view
        time.sleep(0.05)
    raise AssertionError("the job did not end")


def _run(app, **body):
    return app.extra_post["/api/forecast/run"](body)


@pytest.fixture(scope="module")
def finished(app):
    """One real run, shared by the tests that read its report."""
    trials_before = {k: set(v) for k, v in app.trials.items()}
    view = _wait(app, _run(app, ticker="A01", methods=["riskmetrics", "garch", "hull_white"], horizon=21, effort="quick", seed=2)["job"])
    assert view["status"] == "done", view
    return view["result"], trials_before


# ------------------------------------------------------------------------------------------------------------------------------ the plan
def test_the_plan_lists_the_methods_with_what_each_assumes_and_the_choices_of_the_form(app):
    plan = app.extra_get["/api/forecast/plan"]({})
    ids = [m["id"] for m in plan["methods"]]
    assert ids == list(METHODS) + ["blend"] and plan["baseline"] == "random_walk" and plan["blend"] == "blend" and set(plan["default"]) <= set(ids)
    assert all(m["label"] and m["short"] and m["text"] and m["assumes"] and m["fails"] for m in plan["methods"])
    assert plan["horizons"] == [5, 10, 21, 63] and plan["default_horizon"] == 21 and [e["id"] for e in plan["efforts"]] == ["quick", "standard", "thorough"]
    assert set(m["family"] for m in plan["methods"]) <= set(plan["families"]) and plan["scored_horizons"] == [1, 5, 21] and plan["min_days"] == 800
    assert "A01" in app.builder.default_tickers()


def test_the_api_version_says_the_forecast_tab_is_there(app):
    assert app.catalog()["api_version"] == api.API_VERSION == 10
    assert "forecasting" in api.DOC_ROOT_FILES


# ------------------------------------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("body, message", [
    ({"methods": "garch"}, "methods must be a list"),
    ({"methods": [3]}, "methods must be a list"),
    ({"methods": ["astrology"]}, "unknown forecasting method 'astrology'"),
    ({"methods": ["random_walk"]}, "choose at least one method besides the random walk"),
    ({"methods": []}, "choose at least one method"),
    ({"methods": ["garch"], "horizon": 7}, "the horizon must be one of 5, 10, 21, 63"),
    ({"methods": ["garch"], "horizon": "soon"}, "the horizon must be a number"),
    ({"methods": ["garch"], "horizon": 500}, "the horizon must be between"),
    ({"methods": ["garch"], "effort": "lavish"}, "effort must be one of quick, standard, thorough"),
    ({"methods": ["garch"], "seed": -1}, "the seed must be between"),
    ({"methods": ["garch"], "ticker": "not a ticker!"}, ""),
])
def test_a_request_that_cannot_run_is_refused_before_a_job_starts(app, body, message):
    jobs_before = len(app.jobs)
    with pytest.raises(ApiError) as error:
        _run(app, **body)
    assert message in str(error.value) and len(app.jobs) == jobs_before


def test_stopping_a_forecast_that_is_not_running_is_refused(app):
    with pytest.raises(ApiError, match="not running in this app"):
        app.extra_post["/api/forecast/stop"]({"job": "000000000000"})
    with pytest.raises(ApiError, match="not running in this app"):
        app.extra_post["/api/forecast/stop"]({})


# ---------------------------------------------------------------------------------------------------------------------------------- a run
def test_a_run_reports_the_fan_the_scores_and_a_verdict_for_the_ticker_it_was_asked_about(finished):
    d, _ = finished
    assert d["ticker"] == "A01" and d["horizon"] == 21 and d["methods"] == ["riskmetrics", "garch", "hull_white"] and not d["stopped"] and d["history_days"] == 1400
    assert len(d["future_dates"]) == 21 and d["future_dates"][0] > d["asof"] and d["asof"] == d["history"]["dates"][-1] and len(d["history"]["dates"]) == len(d["history"]["close"]) == 300
    assert d["last_price"] == pytest.approx(d["history"]["close"][-1])
    ev = d["evaluation"]
    assert [m["id"] for m in ev["methods"]] == ["random_walk", "riskmetrics", "garch", "hull_white"] and ev["origins"]["count"] >= 30 and ev["origins"]["effort"] == "quick"
    assert d["verdict"]["level"] in ("good", "mixed", "bad") and d["verdict"]["headline"] and len(d["verdict"]["reasons"]) >= 4
    now = d["now"]
    assert set(now["methods"]) == {"random_walk", "riskmetrics", "garch", "hull_white"} and now["errors"] == {} and now["last_price"] == pytest.approx(d["last_price"])
    assert all(len(m["fan"]["0.5"]) == 21 for m in now["methods"].values())
    assert any("weekends" in n for n in d["notes"]) and d["universe"]["tickers"] == ["A01"]


def test_a_forecast_is_not_a_trial_and_the_job_is_kept_in_the_list_with_its_kind(app, finished):
    _, before = finished
    assert {k: set(v) for k, v in app.trials.items()} == before, "forecasting does not count against the Backtest tab's trial counter"
    listed = [j for j in app.jobs_list()["jobs"] if j["kind"] == "forecast"]
    assert listed and listed[0]["status"] == "done" and "A01" in listed[0]["label"] and "21 days" in listed[0]["label"]


def test_the_default_ticker_and_the_first_of_a_list_are_used_when_none_is_named(app):
    done = _wait(app, _run(app, tickers=["A02", "A03"], methods=["riskmetrics"], horizon=5, effort="quick")["job"])
    assert done["status"] == "done" and done["result"]["ticker"] == "A02" and done["result"]["horizon"] == 5
    done = _wait(app, _run(app, methods=["riskmetrics"], effort="quick")["job"])
    assert done["status"] == "done" and done["result"]["ticker"] == app.builder.default_tickers()[0] and done["result"]["horizon"] == 21


def test_a_history_that_is_too_short_for_forecasting_says_how_long_it_must_be(tmp_path):
    short = make_app(tmp_path, 700)
    view = _wait(short, _run(short, ticker="A01", methods=["riskmetrics"], effort="quick")["job"])
    assert view["status"] == "error" and "700 days" in view["error"] and "at least 800" in view["error"]
    tight = make_app(tmp_path / "tight", 820)                                   # long enough to start, too short to test a method that needs 800 days to learn from
    view = _wait(tight, _run(tight, ticker="A01", methods=["gp_returns"], effort="quick")["job"])
    assert view["status"] == "error" and "too short to evaluate" in view["error"] and "820" in view["error"]


class Gated(Forecaster):
    """A random walk that holds the run at its 34th forecast until the test lets it go, so that a Stop can be sent while the run is on."""

    name, short, label, min_history = "gated", "Gated", "Gated", 300
    gate = threading.Event()
    reached = threading.Event()
    calls = 0

    def fit(self, h, horizons):
        return {}

    def predict(self, state, h, horizons, n, rng):
        type(self).calls += 1
        if type(self).calls == 34:
            type(self).reached.set()
            assert type(self).gate.wait(30)
        return Forecast(np.cumsum(0.01 * rng.standard_normal((n, max(horizons))), axis=1))


def test_stop_ends_the_run_with_the_forecasts_made_so_far_scored_and_reported(app, monkeypatch):
    Gated.gate, Gated.reached, Gated.calls = threading.Event(), threading.Event(), 0
    monkeypatch.setitem(METHODS, "gated", Gated())
    job = _run(app, ticker="A01", methods=["gated"], effort="quick")["job"]
    assert Gated.reached.wait(120), "the run never got to the gate"
    assert app.extra_post["/api/forecast/stop"]({"job": job}) == {"ok": True}
    Gated.gate.set()
    view = _wait(app, job)
    assert view["status"] == "done" and view["result"]["stopped"] is True
    count = view["result"]["evaluation"]["origins"]["count"]
    assert 30 <= count <= 40, count


# --------------------------------------------------------------------------------------------------------------------------------- over HTTP
def test_the_routes_and_the_script_are_served_and_a_forecast_can_be_started_over_http(app):
    running = server.start(app, "127.0.0.1", 0)
    try:
        def call(path, body=None, token=True):
            h = {"X-Token": running.token} if token else {}
            if body is not None:
                h["Content-Type"] = "application/json"
            req = urllib.request.Request(running.url.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()

        status, script = call("/static/forecast.js", token=False)
        page = call("/", token=False)[1]
        assert status == 200 and b"/api/forecast/run" in script and b"/static/forecast.js" in page and b'id="tab-forecast"' in page and b'id="view-forecast"' in page
        assert call("/api/forecast/plan", token=False)[0] == 403 and call("/api/forecast/run", {"methods": ["garch"]}, token=False)[0] == 403
        assert json.loads(call("/api/forecast/plan")[1])["baseline"] == "random_walk"
        status, body = call("/api/forecast/run", {"methods": ["astrology"]})
        assert status == 400 and "unknown forecasting method" in json.loads(body)["error"]
        status, body = call("/api/forecast/run", {"ticker": "A00", "methods": ["riskmetrics"], "effort": "quick", "horizon": 5})
        job = json.loads(body)["job"]
        for _ in range(1200):
            view = json.loads(call(f"/api/job/{job}")[1])
            if view["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert status == 200 and view["status"] == "done" and view["result"]["horizon"] == 5
        assert any(j["kind"] == "forecast" for j in json.loads(call("/api/jobs")[1])["jobs"])
    finally:
        running.stop()
