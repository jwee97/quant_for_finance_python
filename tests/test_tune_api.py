"""The Tune tab's endpoints: the plan of a strategy's settings, a search run as a job on the real pipeline (kept across a restart, counted once in the trial counter, the same fingerprint a backtest of the
winner has), the requests that are refused before a job starts, the Stop button, and the script being served."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.webapp import server, tune_api
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=lambda c: bundle), root=tmp_path)


def _wait(app, job, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        view = app.job(job)
        if view["status"] in ("done", "error", "interrupted"):
            return view
        time.sleep(0.05)
    raise AssertionError("the job did not end")


def _run(app, **body):
    return app.extra_post["/api/tune/run"](body)


def _body(**over):
    base = {"strategy": "kalman_trend", "tune": [{"name": "level_noise", "low": 0.002, "high": 0.5, "scale": "log"}, {"name": "slope_noise", "low": 1e-7, "high": 1e-3, "scale": "log"}],
            "allocator": "sleeves", "effort": "custom", "trials": 8, "seconds": 120}
    return {**base, **over}


class FakePipeline:
    """Stands in for ``Pipeline``: a strategy whose edge peaks at level_noise 0.1, on the dates of the bundle, with an optional gate that holds a run until it is released."""

    runs = 0
    gate: threading.Event | None = None
    specs: list = []

    def __init__(self, spec, config, bundle):
        self.spec, self.bundle = spec, bundle

    def run(self, validate=False):
        type(self).runs += 1
        type(self).specs.append(self.spec.to_dict())
        if type(self).gate is not None and type(self).runs > 3:
            assert type(self).gate.wait(30), "the gate was never opened"
        params = self.spec.models[0]["params"]
        level = params.get("level_noise", 0.02)
        idx = self.bundle.returns.index[1:]
        noise = np.random.default_rng(7).normal(0, 0.01, len(idx))
        return SimpleNamespace(window=pd.Series(noise + 0.0015 * float(np.exp(-(((np.log(level) - np.log(0.1)) / 0.8) ** 2))), index=idx), weights=None)


@pytest.fixture()
def fake(monkeypatch):
    FakePipeline.runs, FakePipeline.gate, FakePipeline.specs = 0, None, []
    monkeypatch.setattr(tune_api, "Pipeline", FakePipeline)
    return FakePipeline


# ------------------------------------------------------------------------------------------------------------------ the plan
def test_the_plan_lists_a_strategys_settings_with_ranges_help_and_the_choices_of_the_form(app):
    plan = app.extra_get["/api/tune/plan"]({"strategy": "kalman_trend"})
    names = {s["name"]: s for s in plan["settings"]}
    assert list(names) == ["level_noise", "slope_noise", "vol_halflife", "clip", "scale"] and names["clip"]["tune"] is False and names["scale"]["tune"] is True
    assert names["level_noise"]["scale"] == "log" and names["level_noise"]["low"] == 0.002 and names["level_noise"]["default"] == 0.02 and "Higher" in names["level_noise"]["help"]
    assert plan["note"] is None and [e["id"] for e in plan["efforts"]] == ["quick", "standard", "thorough"] and plan["efforts"][1]["trials"] == 50
    assert [o["id"] for o in plan["objectives"]] == ["sharpe", "sortino", "calmar", "cagr"] and plan["holdouts"] == [0.2, 0.3, 0.4] and plan["max_settings"] == 12
    cross = app.extra_get["/api/tune/plan"]({"strategy": "dual_momentum"})
    assert {s["name"] for s in cross["settings"]} == {"lookback", "skip", "top_k"}        # the text setting (the cash ticker) cannot be searched
    with pytest.raises(ApiError, match="unknown strategy 'nope'"):
        app.extra_get["/api/tune/plan"]({"strategy": "nope"})


def test_a_strategy_with_nothing_to_tune_says_so(app):
    plan = app.extra_get["/api/tune/plan"]({"strategy": "basis_momentum"})
    assert plan["settings"] == [] and "no numeric or yes/no settings" in plan["note"]


# ------------------------------------------------------------------------------------------------------------------ a search on the real pipeline
def test_a_search_runs_as_a_job_on_the_real_pipeline_and_is_kept_across_a_restart(app, tmp_path):
    job = _run(app, **_body())["job"]
    view = _wait(app, job)
    assert view["status"] == "done" and view["kind"] == "tune", view
    r = view["result"]
    assert r["strategy"] == "kalman_trend" and r["search"]["tried"] == 8 and r["search"]["scored"] + r["search"]["failed"] == 8 and r["same_as_start"] in (True, False)
    assert r["start"]["params"] == {"level_noise": 0.02, "slope_noise": 1e-05} and set(r["best"]["params"]) == {"level_noise", "slope_noise"}
    assert r["windows"]["tuning_days"] + r["windows"]["holdout_days"] == r["start"]["tuning"]["days"] + r["start"]["holdout"]["days"] and r["verdict"]["level"] in ("good", "mixed", "bad", "keep")
    assert r["universe"]["tickers"] == [f"A{i:02d}" for i in range(8)] and r["use"]["models"][0]["name"] == "kalman_trend" and r["trials_in_universe"] == 8
    assert {t["ticker"] for t in r["tickers"]} == set(r["universe"]["tickers"]) and r["benchmark"]["holdout"]["sharpe"] is not None and len(r["curves"]["dates"]) <= 500
    listed = [j for j in app.jobs_list()["jobs"] if j["id"] == job][0]
    assert listed["kind"] == "tune" and listed["status"] == "done" and listed["label"].startswith("tune: kalman_trend on 8 tickers")
    again = App(app.config, builder=app.builder, root=tmp_path)                                        # a restarted app finds the finished search again
    assert again.job(job)["result"]["use"] == r["use"]


def test_every_try_is_a_trial_on_the_universe_and_the_winner_run_in_the_backtest_tab_is_not_a_new_one(app):
    result = _wait(app, _run(app, **_body())["job"])["result"]
    key = app.universe_key(result["universe"])
    assert len(app.trials[key]) == result["search"]["tried"] == 8
    best = {"models": [{"name": "kalman_trend", "params": result["use"]["models"][0]["params"]}], "allocator": "sleeves"}
    backtest = _wait(app, app.submit(best)["job"])
    assert backtest["status"] == "done" and backtest["result"]["trials_in_universe"] == 8             # the same fingerprint: found by the search, then run, and counted once
    other = _wait(app, app.submit({"models": [{"name": "kalman_trend", "params": {"level_noise": 0.31337}}], "allocator": "sleeves"})["job"])
    assert other["result"]["trials_in_universe"] == 9
    again = _wait(app, _run(app, **_body(seed=1))["job"])["result"]
    assert again["search"]["tried"] == 8 and again["trials_in_universe"] >= 9 and again["selection"]["n_trials_counted"] >= 8 + 9            # earlier looks count against the new search's winner


# ------------------------------------------------------------------------------------------------------------------ requests
def test_requests_that_cannot_run_are_refused_before_a_job_starts(app):
    jobs_before = len(app.jobs)
    for over, message in [({"tune": []}, "tick at least one setting"), ({"tune": None}, "tick at least one setting"), ({"strategy": "nope"}, "unknown strategy"),
                          ({"tune": [{"name": "banana", "low": 1, "high": 2}]}, "'banana' is not a setting of kalman_trend"),
                          ({"tune": [{"name": "level_noise", "low": 0.5, "high": 0.1}]}, "lowest value to try must be below the highest"),
                          ({"tune": [{"name": "level_noise", "low": -1, "high": 1, "scale": "log"}]}, "must be above zero"),
                          ({"tune": [{"name": "level_noise", "low": "x", "high": 1}]}, "must be a number"),
                          ({"tune": [{"name": "level_noise", "low": 0.1, "high": 1, "scale": "sideways"}]}, "scale must be linear or log"),
                          ({"tune": [{"name": "level_noise", "low": 0.1, "high": 1}] * 2}, "listed twice"), ({"tune": ["level_noise"]}, "each setting to tune must be an object"),
                          ({"tune": [None]}, "each setting to tune must be an object"), ({"tune": {"name": "level_noise"}}, "tick at least one setting"),
                          ({"objective": "nonsense"}, "measure must be one of"), ({"effort": "forever"}, "effort must be one of"), ({"holdout": 0.9}, "between 0.1 and 0.5"),
                          ({"effort": "custom", "trials": 1}, "between 2 and"), ({"stable": "yes"}, "must be true or false"), ({"values": {"level_noise": "big"}}, "must be a number"),
                          ({"values": {"banana": 1}}, "'banana' is not a setting"), ({"values": {"clip": -1.0}}, "must be positive"), ({"values": {"level_noise": -1.0}}, "must be above zero"), ({"values": []}, "mapping of setting name to value"),
                          ({"tickers": ["ZZZ1", "bad ticker!"]}, "not a valid ticker"), ({"allocator": "nope"}, "unknown allocator")]:
        with pytest.raises(ApiError, match=message):
            _run(app, **_body(**over))
    with pytest.raises(ApiError, match="needs at least 2 tickers and you have 1"):                   # a strategy that compares tickers needs several
        _run(app, strategy="xs_momentum", tune=[{"name": "lookback", "low": 63, "high": 504}], tickers=["A00"], effort="quick")
    assert len(app.jobs) == jobs_before


def test_a_start_outside_the_range_widens_the_range_and_says_so_and_a_changed_value_is_used_as_the_start(app, fake):
    r = _wait(app, _run(app, **_body(tune=[{"name": "level_noise", "low": 0.002, "high": 0.01, "scale": "log"}], values={"level_noise": 0.05, "clip": 2.5}))["job"])["result"]
    assert any("widened to 0.002 to 0.05" in n for n in r["notes"]) and r["start"]["params"] == {"level_noise": 0.05} and r["fixed"] == {"clip": 2.5}
    assert r["settings"][0]["high"] == 0.05 and r["use"]["models"][0]["params"]["clip"] == 2.5          # the setting that stays is carried to the Backtest tab with its value
    assert all(s["models"][0]["params"].get("clip") == 2.5 for s in fake.specs)                         # every try used it


def test_what_is_sent_to_the_pipeline_is_what_the_backtest_tab_would_send(app, fake):
    _wait(app, _run(app, **_body())["job"])
    first, others = fake.specs[0], fake.specs[1:]
    assert first["models"][0]["params"] == {} and first["allocation"] == {"allocator": "sleeves", "params": {}}            # the starting settings are the strategy's own: nothing is sent that is a default
    assert all(set(s["models"][0]["params"]) <= {"level_noise", "slope_noise"} for s in others) and first["evaluation"]["benchmarks"] == []


def test_a_small_universe_is_run_with_the_one_asset_minimum_the_backtest_tab_uses(app, fake):
    _wait(app, _run(app, **_body(tickers=["A00", "A01"]))["job"])
    assert all(s["execution"]["min_assets"] == 1 for s in fake.specs)
    _wait(app, _run(app, **_body())["job"])
    assert "min_assets" not in (fake.specs[-1].get("execution") or {})


def test_a_history_too_short_for_a_hold_out_is_said_in_words_and_the_job_ends_with_that_message(app, tmp_path, fake):
    config = load_config()
    short = make_bundle(n_assets=8, n=320, seed=3)
    small = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "p2", fetch=_fetch), default_loader=lambda c: short), root=tmp_path / "short")
    view = _wait(small, _run(small, **_body())["job"])
    assert view["status"] == "error" and "trading days to work with" in view["error"] and "Use a longer history" in view["error"]


# ------------------------------------------------------------------------------------------------------------------ stop
def test_stop_ends_a_running_search_after_the_try_it_is_on_and_reports_the_tries_so_far(app, fake):
    fake.gate = threading.Event()
    job = _run(app, **_body(trials=40))["job"]
    t0 = time.time()
    while fake.runs < 4 and time.time() - t0 < 30:                                                      # the starting settings and two tries have run; the fourth waits at the gate
        time.sleep(0.02)
    assert app.job(job)["status"] == "running" and "try" in app.job(job)["message"]
    assert app.extra_post["/api/tune/stop"]({"job": job}) == {"ok": True}
    fake.gate.set()
    view = _wait(app, job)
    assert view["status"] == "done" and view["result"]["search"]["stopped"] == "user" and 4 <= view["result"]["search"]["tried"] < 40
    with pytest.raises(ApiError, match="not running in this app"):
        app.extra_post["/api/tune/stop"]({"job": "000000000000"})
    with pytest.raises(ApiError, match="not running in this app"):
        app.extra_post["/api/tune/stop"]({})


# ------------------------------------------------------------------------------------------------------------------ over HTTP
def test_the_routes_and_the_script_are_served_and_a_search_can_be_started_over_http(app, fake):
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

        status, script = call("/static/tune.js", token=False)
        assert status == 200 and b"/api/tune/run" in script and b"/static/tune.js" in call("/", token=False)[1] and b'id="tab-tune"' in call("/", token=False)[1]
        status, body = call("/api/tune/plan?strategy=kalman_trend")
        assert status == 200 and json.loads(body)["settings"][0]["name"] == "level_noise"
        assert call("/api/tune/plan?strategy=kalman_trend", token=False)[0] == 403 and call("/api/tune/run", _body(), token=False)[0] == 403
        status, body = call("/api/tune/run", _body(tune=[]))
        assert status == 400 and "tick at least one setting" in json.loads(body)["error"]
        status, body = call("/api/tune/run", _body())
        job = json.loads(body)["job"]
        for _ in range(600):
            view = json.loads(call(f"/api/job/{job}")[1])
            if view["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert status == 200 and view["status"] == "done" and view["result"]["search"]["tried"] == 8
        assert call("/api/jobs")[0] == 200 and any(j["kind"] == "tune" for j in json.loads(call("/api/jobs")[1])["jobs"])
    finally:
        running.stop()


def test_the_api_version_says_the_tune_tab_is_there(app):
    assert app.catalog()["api_version"] == 8
