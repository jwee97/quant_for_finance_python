"""The research workspace through the app's own request handlers (the HTTP layer is covered by the existing server tests): sessions, saving a finished run, comparing, the Market Regime Lab
and the Overview feed. Runs on the synthetic platform dataset of ``tests/test_webapp.py``."""

from __future__ import annotations

import json
import time

import pytest

from src.webapp import api
from src.webapp.api import ApiError
from src.utils.config import load_config
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _run, _wait


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def _post(app, path, body):
    return app.extra_post[path](body)


def _get(app, path, **q):
    return app.extra_get[path](q)


def _saved_backtest(app, session, expr="-ret(5)", **body):
    """A backtest that really trades (the expression is traded as written), saved from its finished job."""
    view = _run(app, tickers=["AAA", "BBB", "CCC"], models=[{"name": "expression", "params": {"expr": expr}}], allocator="score_stack", **body)
    assert view["status"] == "done", view
    return _post(app, "/api/research/artifact/save", {"session": session, "job": view["id"]})["artifact"], view


def test_the_routes_exist_and_the_version_is_announced(app):
    assert app.catalog()["api_version"] == api.API_VERSION == 21
    for path in ("/api/research/sessions", "/api/research/session", "/api/research/artifact", "/api/research/overview", "/api/regime/plan", "/api/research/compat"):
        assert path in app.extra_get, path
    for path in ("/api/research/session/create", "/api/research/artifact/save", "/api/research/compare", "/api/research/export", "/api/regime/run", "/api/regime/by-result"):
        assert path in app.extra_post, path


def test_a_session_is_created_changed_and_deleted_and_bad_input_is_an_api_error(app):
    s = _post(app, "/api/research/session/create", {"name": "Study", "context": {"assets": ["spy", "qqq"], "start": "2015-01-01"}})["session"]
    assert s["context"]["assets"] == ["SPY", "QQQ"]
    listed = _get(app, "/api/research/sessions")
    assert [x["id"] for x in listed["sessions"]] == [s["id"]] and listed["schema"] == 2 and "yahoo" in listed["sources"]
    got = _post(app, "/api/research/session/update", {"id": s["id"], "name": "Renamed", "context": {"benchmark": "SPY", "start": None}})["session"]
    assert got["name"] == "Renamed" and got["context"]["benchmark"] == "SPY" and "start" not in got["context"]
    for body, word in (({"context": {"assets": ["no way!"]}}, "ticker"), ({"context": {"data_source": "nope"}}, "data source"), ({"context": {"bogus": 1}}, "unknown context field")):
        with pytest.raises(ApiError, match=word):
            _post(app, "/api/research/session/create", body)
    with pytest.raises(ApiError, match="unknown session"):
        _get(app, "/api/research/session", id="0123456789ab")
    with pytest.raises(ApiError, match="id is required"):
        _get(app, "/api/research/session")
    _post(app, "/api/research/session/delete", {"id": s["id"]})
    assert _get(app, "/api/research/sessions")["sessions"] == []


def test_a_finished_backtest_is_saved_from_the_servers_own_record_and_labelled(app):
    s = _post(app, "/api/research/session/create", {"name": "Study", "context": {"assets": ["AAA", "BBB", "CCC"]}})["session"]
    art, view = _saved_backtest(app, s["id"])
    assert art["kind"] == "backtest" and art["evidence"] == "historical_simulation" and art["refs"]["job"] == view["id"] and art["refs"]["code_version"]
    full = _get(app, "/api/research/artifact", id=art["id"])
    assert full["series"]["dates"] == view["result"]["dates"] and full["context"]["assets"] == ["AAA", "BBB", "CCC"] and full["metrics"]["sharpe"] == pytest.approx(view["result"]["metrics"]["sharpe"])
    detail = _get(app, "/api/research/session", id=s["id"])
    assert [a["id"] for a in detail["artifacts"]] == [art["id"]] and "series" not in json.dumps(detail["artifacts"])
    json.dumps(full, allow_nan=False)


def test_only_a_finished_job_of_a_known_kind_can_be_saved(app):
    s = _post(app, "/api/research/session/create", {"name": "x"})["session"]
    with pytest.raises(ApiError, match="unknown job"):
        _post(app, "/api/research/artifact/save", {"session": s["id"], "job": "0123456789ab"})
    with pytest.raises(ApiError, match="say which run"):
        _post(app, "/api/research/artifact/save", {"session": s["id"]})
    task = app.submit_task("a task", lambda progress: {"x": 1}, kind="task")["job"]
    _wait(app, task)
    with pytest.raises(ApiError, match="cannot be saved here yet"):
        _post(app, "/api/research/artifact/save", {"session": s["id"], "job": task})
    with pytest.raises(ApiError, match="unknown session"):
        _post(app, "/api/research/artifact/save", {"session": "0123456789ab", "job": task})


def test_the_context_check_flags_a_result_made_on_other_assets_without_touching_it(app):
    s = _post(app, "/api/research/session/create", {"name": "x", "context": {"assets": ["AAA", "BBB", "CCC"]}})["session"]
    art, _ = _saved_backtest(app, s["id"])
    assert _get(app, "/api/research/compat", session=s["id"], artifact=art["id"])["compatible"] is True
    _post(app, "/api/research/session/update", {"id": s["id"], "context": {"assets": ["AAA", "DDD"]}})
    got = _get(app, "/api/research/compat", session=s["id"], artifact=art["id"])
    assert got["compatible"] is False and got["issues"][0]["field"] == "assets" and "Run it again" in got["advice"]
    assert _get(app, "/api/research/artifact", id=art["id"])["context"]["assets"] == ["AAA", "BBB", "CCC"]


def test_two_saved_backtests_are_compared_exported_and_kept_as_a_group(app):
    s = _post(app, "/api/research/session/create", {"name": "x", "context": {"assets": ["AAA", "BBB", "CCC"]}})["session"]
    a, _ = _saved_backtest(app, s["id"])
    b, _ = _saved_backtest(app, s["id"], expr="ret(21)", label="second")
    out = _post(app, "/api/research/compare", {"artifacts": [a["id"], b["id"]], "benchmark": a["id"]})
    assert out["window"] and out["metrics"] and "versus_reference" in out and out["rankable"] in (True, False)
    g = _post(app, "/api/research/group/save", {"session": s["id"], "name": "pair", "artifacts": [a["id"], b["id"]], "benchmark": a["id"]})["group"]
    again = _post(app, "/api/research/compare", {"session": s["id"], "group": g["id"]})
    assert again["window"] == out["window"]
    md = _post(app, "/api/research/export", {"session": s["id"], "group": g["id"], "format": "markdown"})
    assert md["filename"].endswith(".md") and "Research comparison" not in md["text"].splitlines()[0] and "pair" in md["text"].splitlines()[0] and a["refs"]["job"] in md["text"]
    csv_out = _post(app, "/api/research/export", {"artifacts": [a["id"], b["id"]], "format": "csv"})
    assert csv_out["text"].startswith("window,")
    with pytest.raises(ApiError, match="format"):
        _post(app, "/api/research/export", {"artifacts": [a["id"]], "format": "pdf"})
    with pytest.raises(ApiError, match="between 1 and 12"):
        _post(app, "/api/research/compare", {"artifacts": []})
    with pytest.raises(ApiError, match="unknown comparison group"):
        _post(app, "/api/research/compare", {"session": s["id"], "group": "0123456789ab"})


def test_the_regime_lab_runs_as_a_job_saves_a_causal_estimate_and_scores_a_backtest_by_regime(app):
    s = _post(app, "/api/research/session/create", {"name": "x", "context": {"assets": ["AAA", "BBB", "CCC"]}})["session"]
    plan = _get(app, "/api/regime/plan")
    assert {m["name"] for m in plan["models"]} == {"hmm", "vol_state", "gmm"} and plan["defaults"]["min_train"] == 750
    job = _post(app, "/api/regime/run", {"tickers": ["AAA", "BBB", "CCC"], "model": "hmm", "session": s["id"], "min_days": 40})["job"]
    view = _wait(app, job, 300)
    assert view["status"] == "done", view
    res = view["result"]
    assert res["evidence"] == "walk_forward" and res["names"] == ["LowVol", "HighVol"] and res["retrospective"]["label"].startswith("retrospective") and res["current"]["basis"].startswith("filtered")
    assert res["artifact"]["kind"] == "regime" and res["artifact"]["evidence"] == "walk_forward"
    reg = _get(app, "/api/research/artifact", id=res["artifact"]["id"])
    assert set(reg["series"]["extra"]) == {"label", "p_LowVol", "p_HighVol"} and reg["series"]["dates"] == res["timeline"]["dates"]
    bt, _ = _saved_backtest(app, s["id"])
    by = _post(app, "/api/regime/by-result", {"regime": reg["id"], "result": bt["id"]})
    assert by["lag"] == 1 and by["label_basis"] == "filtered (causal)" and by["rows"] and "historical simulation" in by["caution"]
    with pytest.raises(ApiError, match="saved regime"):
        _post(app, "/api/regime/by-result", {"regime": bt["id"], "result": bt["id"]})
    with pytest.raises(ApiError, match="lag"):
        _post(app, "/api/regime/by-result", {"regime": reg["id"], "result": bt["id"], "lag": 5})
    json.dumps(res, allow_nan=False)


def test_the_regime_lab_refuses_bad_settings_before_it_starts_a_job(app):
    for body, word in (({"model": "oracle"}, "model must be"), ({"n_states": 7}, "2 or 3"), ({"min_train": 10}, "min_train"), ({"min_days": "x"}, "whole numbers")):
        with pytest.raises(ApiError, match=word):
            _post(app, "/api/regime/run", {"tickers": ["AAA", "BBB", "CCC"], **body})
    assert app.jobs_list()["jobs"] == [] or all(j["kind"] != "regime" for j in app.jobs_list()["jobs"])


def test_the_overview_shows_only_what_exists_and_labels_where_it_comes_from(app):
    empty = _get(app, "/api/research/overview")
    assert empty["sessions"]["n"] == 0 and empty["artifacts"]["n"] == 0
    for block in ("regime", "portfolio_risk", "options"):
        assert empty[block]["available"] is False and empty[block]["reason"], block
    assert empty["data"]["platform"]["last"] and empty["data"]["source"] and empty["market"]["kind"] == "historical" and "not a live feed" in empty["market"]["source"]
    assert {r["ticker"] for r in empty["market"]["rows"]} <= {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}
    s = _post(app, "/api/research/session/create", {"name": "x", "context": {"assets": ["AAA", "BBB", "CCC"]}})["session"]
    art, _ = _saved_backtest(app, s["id"])
    full = _get(app, "/api/research/overview")
    assert full["sessions"]["n"] == 1 and full["artifacts"]["recent"][0]["id"] == art["id"]
    pr = full["portfolio_risk"]
    assert pr["available"] and pr["evidence"] == "historical_simulation" and "historical simulation" in pr["source"] and pr["session"] == s["id"]
    json.dumps(full, allow_nan=False)


def test_a_job_started_here_is_listed_with_its_kind(app):
    job = _post(app, "/api/regime/run", {"tickers": ["AAA", "BBB", "CCC"], "model": "vol_state"})["job"]
    view = _wait(app, job, 300)
    assert view["status"] == "done" and view["result"]["model"] == "vol_state" and "no retrospective counterpart" in view["result"]["retrospective_note"] and "retrospective" not in view["result"]
    assert any(j["id"] == job and j["kind"] == "regime" for j in app.jobs_list()["jobs"])
    time.sleep(0)


def test_a_saved_machine_learning_result_is_scored_by_regime_with_its_own_walk_forward_label(app):
    """ML to regime: an experiment's out-of-sample returns, stored as a result, are paired with the regime known the day before."""
    import numpy as np
    import pandas as pd

    from src.research import extract

    s = _post(app, "/api/research/session/create", {"name": "x", "context": {"assets": ["AAA", "BBB", "CCC"]}})["session"]
    job = _post(app, "/api/regime/run", {"tickers": ["AAA", "BBB", "CCC"], "model": "vol_state", "session": s["id"], "min_days": 40})["job"]
    res = _wait(app, job, 300)["result"]
    reg_id = res["artifact"]["id"]
    dates = pd.to_datetime(res["timeline"]["dates"])
    rng = np.random.default_rng(0)
    ret = pd.DataFrame({"net": rng.normal(0.0004, 0.008, len(dates)), "turnover": 0.1, "costs": 0.0001, "segment": ["test"] * len(dates)}, index=dates)
    report = {"id": "exp_x", "name": "ridge", "config": {"dataset": {"target": "fwd_5d"}}, "config_hash": "h", "dataset_key": "dk", "dataset": {"target": "fwd_5d", "horizon": 5}, "provenance": {"source": "platform", "label": "platform"},
              "metrics": {"oos": {"sharpe": 0.5}}, "trials": {"dataset_trials": 3}}
    art = app.research.store.add_artifact(s["id"], extract.ml_artifact(report, ret, code_version="x"))
    out = _post(app, "/api/regime/by-result", {"regime": reg_id, "result": art["id"], "min_days": 40})
    assert out["result_evidence"] == "walk_forward" and "caution" not in out, "a walk-forward result is not told it is only a simulation"
    assert out["lag"] == 1 and {r["state"] for r in out["rows"]} <= set(res["names"]) and sum(r["n_days"] for r in out["rows"]) == out["n_days"]
    assert out["window"]["start"] >= res["timeline"]["dates"][0]


def test_results_on_different_data_versions_are_flagged_before_they_are_compared(app):
    s = _post(app, "/api/research/session/create", {"name": "x", "context": {"assets": ["AAA", "BBB", "CCC"]}})["session"]
    a, _ = _saved_backtest(app, s["id"])
    b = app.research.store.add_artifact(s["id"], {**app.research.store.get_artifact(a["id"]), "context": {**a["context"], "dataset_version": "another-vintage"}})
    out = _post(app, "/api/research/compare", {"artifacts": [a["id"], b["id"]]})
    assert "different_dataset" in {i["code"] for i in out["issues"]} and out["ranking"] is None and out["metrics"]


def test_an_option_strategy_backtest_is_saved_as_simulated_and_never_as_a_historical_result(app):
    s = _post(app, "/api/research/session/create", {"name": "x"})["session"]
    job = _post(app, "/api/options/strategy/backtest", {"strategy": "short_put", "days": 90, "params": {"delta": 0.2}})["job"]
    view = _wait(app, job, 300)
    assert view["status"] == "done"
    art = _post(app, "/api/research/artifact/save", {"session": s["id"], "job": job})["artifact"]
    assert art["kind"] == "options" and art["evidence"] == "simulated" and art["data_kind"] == "simulated"
    full = _get(app, "/api/research/artifact", id=art["id"])
    assert len(full["series"]["dates"]) == len(full["series"]["equity"]) > 30 and any("simulated" in n for n in full["notes"])
    json.dumps(full, allow_nan=False)
    ov = _get(app, "/api/research/overview")["options"]
    assert ov["available"] and ov["evidence"] == "simulated" and "simulated option market" in ov["source"]


def test_a_sweep_or_comparison_is_refused_because_it_is_not_one_result(app):
    from src.research import extract
    from src.research.store import StoreError

    with pytest.raises(StoreError, match="not a strategy backtest"):
        extract.strategy_artifact({"kind": "options_sweep", "dates": [], "rows": []})
    assert app


def test_a_strategy_configuration_is_saved_listed_and_deleted_through_the_api(app):
    s = _post(app, "/api/research/session/create", {"name": "x"})["session"]
    body = {"models": [{"name": "expression", "params": {"expr": "-ret(5)"}}], "tickers": ["AAA", "BBB", "CCC"], "allocator": "score_stack", "session": s["id"]}
    cfg = _post(app, "/api/research/config/save", {"session": s["id"], "name": "Reversal", "body": body})["config"]
    got = _get(app, "/api/research/session", id=s["id"])["session"]
    assert got["schema"] == 2 and [c["name"] for c in got["configs"]] == ["Reversal"] and "session" not in got["configs"][0]["body"]
    assert _get(app, "/api/research/sessions")["sessions"][0]["n_configs"] == 1
    with pytest.raises(ApiError, match="at least one strategy"):
        _post(app, "/api/research/config/save", {"session": s["id"], "name": "empty", "body": {"tickers": ["AAA"]}})
    _post(app, "/api/research/config/delete", {"session": s["id"], "config": cfg["id"]})
    assert _get(app, "/api/research/session", id=s["id"])["session"]["configs"] == []
