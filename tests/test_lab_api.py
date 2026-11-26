"""The Alpha, Risk and Portfolio Lab through the app's request handlers: the plan, the checks on a request, and each study as a job on the synthetic platform dataset (AAA to FFF, 2,300 days)."""

from __future__ import annotations

import json

import pytest

from src.portfolio import lab as portfolio_lab
from src.utils.config import load_config
from src.webapp import api
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait

SIX = "AAA, BBB, CCC, DDD, EEE, FFF"


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def _done(app, route, body, timeout=240):
    view = _wait(app, app.extra_post[route](body)["job"], timeout)
    assert view["status"] == "done", view
    json.dumps(view["result"])                                                       # the whole result goes over HTTP
    return view["result"]


def test_routes_and_the_plan_say_what_each_study_does(app):
    assert api.API_VERSION >= 21 and {"/api/lab/plan", "/api/lab/signals"} <= set(app.extra_get)
    assert {"/api/lab/alpha", "/api/lab/risk", "/api/lab/portfolio", "/api/lab/trading", "/api/lab/signals/register", "/api/lab/signals/delete"} <= set(app.extra_post)
    plan = app.extra_get["/api/lab/plan"]({})
    assert {m["key"] for m in plan["methods"]} == set(portfolio_lab.METHODS) and {s["key"] for s in plan["studies"]} == {"alpha", "risk", "portfolio", "trading"}
    assert set(plan["defaults_methods"]) <= set(portfolio_lab.METHODS) and "graphical_lasso" in plan["cov_methods"] and {s["key"] for s in plan["settings"]} <= set(plan["limits"]) | {"cov_method"}
    assert all(s["help"] for s in plan["settings"]) and "inside the noise" in " ".join(plan["notes"])


def test_bad_requests_are_api_errors_that_say_what_to_change(app):
    post = app.extra_post
    for route, body, word in (("/api/lab/alpha", {"tickers": "AAA, BBB"}, "at least 6"), ("/api/lab/alpha", {"tickers": SIX, "signal": "oracle"}, "signal must be"), ("/api/lab/alpha", {"tickers": SIX, "horizon": 200}, "horizon"),
                              ("/api/lab/risk", {"tickers": SIX, "kind": "magic"}, "kind must be"), ("/api/lab/risk", {"tickers": SIX, "n_factors": 99}, "n_factors must be"), ("/api/lab/risk", {"tickers": "AAA"}, "at least 4"),
                              ("/api/lab/portfolio", {"tickers": SIX, "methods": ["oracle"]}, "unknown method"), ("/api/lab/portfolio", {"tickers": SIX, "methods": []}, "choose at least one"),
                              ("/api/lab/portfolio", {"tickers": SIX, "config": {"bogus": 1}}, "unknown setting"), ("/api/lab/portfolio", {"tickers": SIX, "config": {"lookback": 5}}, "between"),
                              ("/api/lab/portfolio", {"tickers": SIX, "config": {"step": 2.5}}, "whole number"), ("/api/lab/portfolio", {"tickers": SIX, "config": {"cov_method": "magic"}}, "cov_method"),
                              ("/api/lab/portfolio", {"tickers": SIX, "config": {"max_weight": 0.1}}, "cannot be fully invested"), ("/api/lab/trading", {"tickers": SIX, "config": {"lag": 99}}, "between"),
                              ("/api/lab/signals/delete", {"name": "nope"}, "no signal")):
        with pytest.raises(ApiError, match=word):
            post[route](body)


def test_an_alpha_study_is_a_job_with_the_ic_the_decay_and_the_calibration(app):
    r = _done(app, "/api/lab/alpha", {"tickers": SIX, "signal": "reversal", "horizon": 5})
    assert r["signal"] == "reversal" and r["horizon"] == 5 and len(r["assets"]) == 6 and r["days"] > 1500
    assert {"n", "mean", "t", "ir_annual", "hit_rate", "stability"} <= set(r["ic"]) and r["ic"]["n"] > 1000
    assert len(r["decay"]["table"]) >= 4 and {"ic", "se", "t"} <= set(r["decay"]["table"][0]) and "half_life" in r["decay"]["fit"]
    assert r["calibration"]["verdict"] and {"n_effective", "breadth", "rho"} <= set(r["breadth"]) and 1 <= r["breadth"]["n_effective"] <= 6
    assert r["transfer"]["ok"] and 0 <= r["transfer"]["long_only_tc"] <= 1 + 1e-9 and all(a["long_only_weight"] >= -1e-9 for a in r["transfer"]["assets"])
    assert len(r["ic_series"]["dates"]) == len(r["ic_series"]["cumulative"]) and any("residual" in n for n in r["notes"])


def test_a_risk_study_decomposes_volatility_and_checks_the_forecast(app):
    r = _done(app, "/api/lab/risk", {"tickers": SIX, "kind": "statistical"})
    assert r["kind"] == "statistical" and r["model"]["factors"] >= 1 and 0 < r["risk"]["volatility"] < 1
    total = sum(c["contribution"] for c in r["contributions"])
    assert abs(total - r["risk"]["volatility"]) < 1e-6, "factor and specific contributions add to the volatility"
    assert abs(sum(a["contribution"] for a in r["assets_risk"]) - r["risk"]["volatility"]) < 1e-6
    assert r["bias"]["verdict"] and len(r["bias"]["dates"]) == len(r["bias"]["values"]) and r["spectrum"]["above_noise"] >= 0
    t = _done(app, "/api/lab/risk", {"tickers": SIX, "kind": "time_series"})
    assert t["model"]["factors"] == 1 and t["neutral"] and abs(t["neutral"]["exposure_after"]) < 1e-9


def test_a_portfolio_study_runs_the_same_windows_for_every_method(app):
    r = _done(app, "/api/lab/portfolio", {"tickers": SIX, "methods": ["equal_weight", "min_variance", "hrp", "nco", "cdar"], "config": {"lookback": 252, "step": 63, "cost_bps": 10, "perturb": 2}}, 300)
    rows = {x["key"]: x for x in r["table"]}
    assert set(rows) == {"equal_weight", "min_variance", "hrp", "nco", "cdar"} and all(x["fallbacks"] == 0 for x in rows.values())
    assert rows["equal_weight"]["instability"] == pytest.approx(0, abs=1e-9) and rows["min_variance"]["instability"] is not None and all(0 < x["effective_n"] <= 6 + 1e-9 for x in rows.values())
    assert len(r["equity"]["dates"]) == len(r["equity"]["series"]["hrp"]) and abs(sum(r["last_weights"]["nco"].values()) - 1) < 1e-9 and r["config"]["assets"] == 6
    capped = _done(app, "/api/lab/portfolio", {"tickers": SIX, "methods": ["min_variance"], "config": {"step": 63, "max_weight": 0.3}}, 300)
    assert max(capped["last_weights"]["min_variance"].values()) <= 0.3 + 1e-9


def test_a_trading_study_shows_the_waterfall_and_four_policies(app):
    r = _done(app, "/api/lab/trading", {"tickers": SIX, "signal": "momentum", "horizon": 5, "config": {"cost_bps": 10}}, 400)
    assert [x["index"] for x in r["waterfall"]] == ["paper", "constraints", "costs", "cost-aware trading", "delay"] and {x["index"] for x in r["policies"]} == {"full", "partial", "band", "threshold"}
    pol = {x["index"]: x for x in r["policies"]}
    assert pol["band"]["turnover_per_year"] < pol["full"]["turnover_per_year"] and pol["partial"]["turnover_per_year"] < pol["full"]["turnover_per_year"]
    assert all(x["sharpe"] is None or x["sharpe"] <= x["gross_sharpe"] + 1e-9 for x in r["policies"]), "costs only ever lower the Sharpe ratio; a book that never trades has none (null)"


def test_signals_are_kept_with_versions_and_an_unchanged_signal_keeps_its_version(app):
    first = _done(app, "/api/lab/signals/register", {"tickers": SIX, "signal": "trend", "name": "trend_a", "horizon": 5})
    assert first["version"] == 1 and first["new"] and first["name"] == "trend_a" and first["dates"] > 1500
    again = _done(app, "/api/lab/signals/register", {"tickers": SIX, "signal": "trend", "name": "trend_a", "horizon": 5})
    assert again["version"] == 1 and not again["new"]
    listing = app.extra_get["/api/lab/signals"]({})["signals"]
    assert [s["name"] for s in listing] == ["trend_a"] and listing[0]["n_assets"] == 6 and listing[0]["versions"] == 1
    assert app.extra_post["/api/lab/signals/delete"]({"name": "trend_a"})["deleted"] == "trend_a" and app.extra_get["/api/lab/signals"]({})["signals"] == []
    bad = _wait(app, app.extra_post["/api/lab/signals/register"]({"tickers": SIX, "signal": "trend", "name": "../x"})["job"], 60)
    assert bad["status"] == "error" and "name" in bad["error"]


def test_a_method_of_a_portfolio_study_is_kept_in_a_session_with_its_benchmark_and_the_number_of_methods_tried(app):
    view = _wait(app, app.extra_post["/api/lab/portfolio"]({"tickers": SIX, "methods": ["equal_weight", "hrp", "nco"], "config": {"step": 63}})["job"], 300)
    assert view["status"] == "done", view
    post = app.extra_post
    session = post["/api/research/session/create"]({"name": "Portfolio lab", "context": {"assets": ["aaa", "bbb", "ccc", "ddd", "eee", "fff"], "start": "2012-01-02"}})["session"]
    art = post["/api/research/artifact/save"]({"session": session["id"], "job": view["id"], "method": "hrp"})["artifact"]
    assert art["kind"] == "portfolio_lab" and art["evidence"] == "historical_simulation" and art["title"].startswith("Portfolio lab: Hierarchical")
    full = app.extra_get["/api/research/artifact"]({"id": art["id"]})
    dates, equity = full["series"]["dates"], full["series"]["equity"]
    assert len(dates) == len(equity) > 1000 and len(full["series"]["benchmarks"]["Equal weight"]) == len(dates), "the whole daily curve is kept, not a thinned one"
    assert full["metrics"]["n_trials"] == 3 and "sharpe" in full["metrics"] and full["config"]["method"] == "hrp" and any("not corrected for having chosen it" in n for n in full["notes"])
    assert full["context"]["assets"] == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"] and full["refs"]["job"] == view["id"]
    eq_art = post["/api/research/artifact/save"]({"session": session["id"], "job": view["id"], "method": "equal_weight"})["artifact"]
    cmp = post["/api/research/compare"]({"artifacts": [art["id"], eq_art["id"]]})
    assert len(cmp["artifacts"]) == 2 and not any(a["kind"] != "portfolio_lab" for a in cmp["artifacts"])
    with pytest.raises(ApiError, match="which method"):
        post["/api/research/artifact/save"]({"session": session["id"], "job": view["id"]})
    with pytest.raises(ApiError, match="no method"):
        post["/api/research/artifact/save"]({"session": session["id"], "job": view["id"], "method": "oracle"})


def test_a_factor_study_compares_corrects_and_combines_from_the_past(app):
    assert "/api/lab/factors" in app.extra_post
    for body, word in (({"tickers": "AAA, BBB"}, "at least 6"), ({"tickers": SIX, "signals": ["oracle", "momentum"]}, "signals must be"), ({"tickers": SIX, "signals": ["momentum", "momentum"]}, "distinct"),
                       ({"tickers": SIX, "horizon": 200}, "horizon")):
        with pytest.raises(ApiError, match=word):
            app.extra_post["/api/lab/factors"](body)
    r = _done(app, "/api/lab/factors", {"tickers": SIX, "signals": ["momentum", "reversal", "low_vol"], "horizon": 5})
    cmp_, comb = r["comparison"], r["combination"]
    assert cmp_["n_tested"] == 3 and {x["factor"] for x in cmp_["table"]} == {"momentum", "reversal", "low_vol"}
    for row in cmp_["table"]:
        assert row["ci_low"] <= row["rank_ic"] <= row["ci_high"] and row["p_holm"] >= row["p_raw"] - 1e-12 and isinstance(row["survives_holm"], bool)
    assert comb["train_dates"][1] < comb["test_dates"][0] and set(comb["verdicts"]) == {"equal_weight", "ic_weighted", "ridge", "orthogonalised"}
    assert comb["best_single_on_training"] in ("momentum", "reversal", "low_vol") and any("past only" in n or "only the past" in n or "from the past" in n for n in r["notes"] + [comb["note"]])


def test_a_factor_study_with_a_single_signal_says_what_to_change(app):
    view = _wait(app, app.extra_post["/api/lab/factors"]({"tickers": SIX, "signals": ["momentum"]})["job"], 120)
    assert view["status"] == "error" and "at least two signals" in view["error"]
