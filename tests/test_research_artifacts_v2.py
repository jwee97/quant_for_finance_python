"""Saved studies: the version-2 artifact schema, reading older artifacts without altering them, keeping unknown fields, saving once however many times the button is pressed, new versions that leave the earlier study
exactly as it was, re-running from the stored request, and the new kinds (portfolio study with attribution, trading study)."""

from __future__ import annotations

import json

import pytest

from src.research import store as rs
from src.utils.config import load_config
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait

SIX = "AAA, BBB, CCC, DDD, EEE, FFF"


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


@pytest.fixture()
def store(tmp_path):
    return rs.ResearchStore(tmp_path / "research")


def _art(**kw):
    base = {"kind": "backtest", "title": "A study", "evidence": "historical_simulation", "data_kind": "platform", "context": {"assets": ["AAA", "BBB"], "start": "2020-01-01"}, "metrics": {"sharpe": 1.1}}
    return {**base, **kw}


# ------------------------------------------------------------------------------------------------------------------------ the schema, migration and unknown fields
def test_a_version_one_artifact_is_read_as_version_two_and_its_file_is_not_touched(store, tmp_path):
    s = store.create_session("old")
    legacy = {"schema": 1, "id": "0123456789ab", "session_id": s["id"], "created": "2024-01-01T00:00:00Z", "kind": "backtest", "title": "Old", "evidence": "in_sample", "data_kind": "platform",
              "context": {"assets": ["AAA"], "costs": {"cost_bps": 5.0}}, "config": {"x": 1}, "metrics": {"sharpe": 0.9}, "refs": {}, "notes": ["n"], "mystery": {"kept": True},
              "series": {"dates": ["2020-01-01", "2020-01-02"], "equity": [1.0, 1.01], "benchmarks": {"Bench": [1.0, 1.0]}}}
    path = tmp_path / "research" / "artifacts" / "0123456789ab.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(legacy)
    path.write_text(text)
    rec = store._session(s["id"])
    rec["artifacts"].append("0123456789ab")
    store._save_session(rec)
    a = store.get_artifact("0123456789ab")
    assert a["schema"] == 2 and a["migrated_from"] == 1 and a["version"] == 1 and a["parent"] == "" and a["hypothesis"] == "" and a["attribution"] is None
    assert a["costs"] == {"cost_bps": 5.0} and a["benchmark"] == "Bench" and a["extra"] == {"mystery": {"kept": True}} and "mystery" not in a
    assert path.read_text() == text, "the older record stays exactly as it was written"
    assert store.list_artifacts(s["id"])[0]["version"] == 1


def test_an_artifact_from_a_newer_version_is_refused_not_rewritten(store, tmp_path):
    s = store.create_session("new")
    a = store.add_artifact(s["id"], _art())
    path = tmp_path / "research" / "artifacts" / f"{a['id']}.json"
    rec = json.loads(path.read_text())
    rec["schema"] = 3
    path.write_text(json.dumps(rec))
    with pytest.raises(rs.StoreError, match="newer version"):
        store.get_artifact(a["id"])


def test_every_field_of_a_version_two_artifact_survives_a_round_trip_and_unknown_fields_are_kept(store):
    s = store.create_session("rt")
    raw = _art(hypothesis="momentum survives costs", benchmark="Equal weight", costs={"cost_bps": 10, "charged": "on rebalance"}, validation={"point_in_time": "yes", "recon": "reconciled"},
               attribution={"method": "hrp", "factor": {"totals": {"market": 0.1}}}, source={"job": "j1", "request": {"path": "/api/lab/portfolio", "body": {"a": 1}}}, future_field={"x": [1, 2]})
    a = store.add_artifact(s["id"], raw)
    back = store.get_artifact(a["id"])
    for k in ("hypothesis", "benchmark", "costs", "validation", "attribution", "source"):
        assert back[k] == raw[k], k
    assert back["extra"] == {"future_field": {"x": [1, 2]}} and back["schema"] == 2 and back["version"] == 1 and "migrated_from" not in back
    assert json.loads(json.dumps(back)) == back


# ------------------------------------------------------------------------------------------------------------------------ immutability, duplicates and versions
def test_saving_the_same_run_twice_keeps_one_study_and_says_so(store):
    s = store.create_session("dup")
    first = store.add_artifact(s["id"], _art(refs={"save_key": "k1"}))
    second = store.add_artifact(s["id"], _art(refs={"save_key": "k1"}, title="pressed again"))
    assert second["duplicate"] is True and second["id"] == first["id"] and len(store.list_artifacts(s["id"])) == 1
    assert store.get_artifact(first["id"])["title"] == "A study"
    other = store.add_artifact(s["id"], _art(refs={"save_key": "k2"}))
    assert other["id"] != first["id"] and len(store.list_artifacts(s["id"])) == 2


def test_a_new_version_is_a_separate_artifact_and_the_earlier_one_is_not_changed(store, tmp_path):
    s = store.create_session("v")
    v1 = store.add_artifact(s["id"], _art(refs={"save_key": "k"}))
    before = (tmp_path / "research" / "artifacts" / f"{v1['id']}.json").read_text()
    v2 = store.add_artifact(s["id"], _art(refs={"save_key": "k"}, parent=v1["id"], title="re-run", metrics={"sharpe": 1.3}))
    assert not v2.get("duplicate"), "a new version of a study is not a duplicate of it"
    assert v2["version"] == 2 and v2["parent"] == v1["id"] and v2["id"] != v1["id"]
    v3 = store.add_artifact(s["id"], _art(refs={"save_key": "k3"}, parent=v2["id"], title="re-run again"))
    assert v3["version"] == 3
    assert (tmp_path / "research" / "artifacts" / f"{v1['id']}.json").read_text() == before
    assert [(v["version"], v["id"]) for v in store.versions(v2["id"])] == [(1, v1["id"]), (2, v2["id"]), (3, v3["id"])]
    assert store.get_artifact(v1["id"])["metrics"] == {"sharpe": 1.1}
    other = store.create_session("elsewhere")
    with pytest.raises(rs.StoreError, match="same session"):
        store.add_artifact(other["id"], _art(parent=v1["id"]))
    with pytest.raises(rs.StoreError, match="no longer exists"):
        store.add_artifact(s["id"], _art(parent="ffffffffffff"))


# ------------------------------------------------------------------------------------------------------------------------ through the endpoints
def _job(app, route, body, timeout=600):
    from src.webapp import api
    token = api._REQUEST_HINT.set({"path": route, "body": body})                          # what the HTTP layer does for every request: the job remembers the request that started it
    try:
        started = app.extra_post[route](body)
    finally:
        api._REQUEST_HINT.reset(token)
    view = _wait(app, started["job"], timeout)
    assert view["status"] == "done", view
    return view


def test_a_portfolio_study_is_saved_with_attribution_validation_benchmark_and_costs_and_can_be_re_run_as_a_new_version(app):
    post, get = app.extra_post, app.extra_get
    view = _job(app, "/api/lab/portfolio", {"tickers": SIX, "methods": ["equal_weight", "min_variance", "max_sharpe"], "config": {"step": 21, "lookback": 252, "cost_bps": 12}, "expected_returns": {"source": "factor", "factor": "momentum"}})
    saved = post["/api/research/artifact/save"]({"new_session": "Portfolio studies", "job": view["id"], "method": "max_sharpe", "title": "Momentum-informed max Sharpe", "notes": "tried after the forecast study", "hypothesis": "factor mu beats history"})
    assert saved["session_created"] and saved["duplicate"] is False
    art = get["/api/research/artifact"]({"id": saved["artifact"]["id"]})
    assert art["kind"] == "portfolio_lab" and art["title"] == "Momentum-informed max Sharpe" and art["hypothesis"] == "factor mu beats history" and art["notes"][0] == "tried after the forecast study"
    assert art["benchmark"] == "Equal weight" and art["costs"]["cost_bps"] == 12 and art["context"]["costs"]["cost_bps"] == 12
    assert art["attribution"]["factor"]["reconciliation"]["status"] == "reconciled" and art["attribution"]["method"] == "max_sharpe" and art["attribution"]["methodology"]
    assert art["validation"]["expected_returns_source"] == "factor: momentum" and art["validation"]["attribution_reconciliation"] == "reconciled" and "before" in art["validation"]["point_in_time"]
    assert art["config"]["mu_source"] == "factor: momentum" and art["source"]["request"]["path"] == "/api/lab/portfolio" and art["refs"]["save_key"]
    again = post["/api/research/artifact/save"]({"session": saved["session"], "job": view["id"], "method": "max_sharpe"})
    assert again["duplicate"] is True and again["artifact"]["id"] == art["id"], "a second press keeps the one study"
    other_method = post["/api/research/artifact/save"]({"session": saved["session"], "job": view["id"], "method": "min_variance"})
    assert other_method["duplicate"] is False and other_method["artifact"]["id"] != art["id"]
    spec = get["/api/research/artifact/rerun"]({"id": art["id"]})
    assert spec["route"] == "/api/lab/portfolio" and spec["body"]["expected_returns"] == {"source": "factor", "factor": "momentum"} and spec["body"]["config"]["cost_bps"] == 12
    new = _job(app, spec["route"], {**spec["body"], "config": {**spec["body"]["config"], "cost_bps": 3}})
    v2 = post["/api/research/artifact/save"]({"session": saved["session"], "job": new["id"], "method": "max_sharpe", "parent": art["id"], "title": "Momentum-informed max Sharpe, cheaper trading"})["artifact"]
    assert v2["version"] == 2 and v2["parent"] == art["id"]
    chain = get["/api/research/artifact/versions"]({"id": v2["id"]})["versions"]
    assert [c["version"] for c in chain] == [1, 2]
    assert get["/api/research/artifact"]({"id": art["id"]})["costs"]["cost_bps"] == 12, "the first study still says what it said"


def test_a_trading_study_can_be_kept_and_a_legacy_study_without_a_stored_request_cannot_be_re_run(app):
    post, get = app.extra_post, app.extra_get
    view = _job(app, "/api/lab/trading", {"tickers": SIX, "signal": "momentum", "horizon": 5, "config": {"cost_bps": 5}})
    saved = post["/api/research/artifact/save"]({"new_session": "Trading", "job": view["id"]})
    art = get["/api/research/artifact"]({"id": saved["artifact"]["id"]})
    assert art["kind"] == "trading_study" and art["costs"]["cost_bps"] == 5 and "paper_sharpe" in art["metrics"] and art["validation"]["waterfall"] and "ranked by performance" in " ".join(art["notes"])
    assert art["context"]["assets"] == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    legacy = app.research.store.add_artifact(saved["session"], _art(kind="backtest", title="from before"))
    with pytest.raises(ApiError, match="kept no record of the request"):
        get["/api/research/artifact/rerun"]({"id": legacy["id"]})
    with pytest.raises(ApiError, match="name the new session"):
        post["/api/research/artifact/save"]({"new_session": "  ", "job": view["id"]})
    with pytest.raises(ApiError, match="unknown job"):
        post["/api/research/artifact/save"]({"session": saved["session"], "job": "0123456789ab"})
