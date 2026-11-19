"""The Financial ML Lab through the app's request handlers: the plan, the checks on a request, and a pipeline and a structure run as jobs on the synthetic platform dataset."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import api
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def test_the_routes_and_the_plan_say_what_the_lab_does_and_does_not_do(app):
    assert api.API_VERSION >= 19 and "/api/finml/plan" in app.extra_get and {"/api/finml/run", "/api/finml/structure"} <= set(app.extra_post)
    plan = app.extra_get["/api/finml/plan"]({})
    keys = {s["key"] for s in plan["settings"]}
    assert {"primary", "cusum_k", "pt", "sl", "vertical_bars", "entry_lag", "holdout", "threshold", "n_trials", "same_bar"} <= keys and keys <= set(plan["defaults"])
    assert plan["choices"]["primary"] == ["momentum", "mean_reversion", "ma_cross", "always_long"] and all(s["help"] for s in plan["settings"])
    text = " ".join(plan["notes"])
    assert "individual trades" in text and "not a prospective test" in text and "negative result" in text


def test_bad_requests_are_api_errors_that_say_what_to_change(app):
    run = app.extra_post["/api/finml/run"]
    for body, word in (({}, "one ticker"), ({"ticker": "AAA, BBB"}, "one ticker"), ({"ticker": "AAA", "config": {"primary": "oracle"}}, "primary must be one of"), ({"ticker": "AAA", "config": {"holdout": 0.9}}, "holdout must be"),
                       ({"ticker": "AAA", "config": {"vertical_bars": 2.5}}, "whole number"), ({"ticker": "AAA", "config": {"bogus": 1}}, "unknown setting"), ({"ticker": "AAA", "config": {"cost_bps": "x"}}, "must be a number"),
                       ({"ticker": "AAA", "config": {"threshold": 0.3}}, "threshold must be")):
        with pytest.raises(ApiError, match=word):
            run(body)
    with pytest.raises(ApiError, match="window"):
        app.extra_post["/api/finml/structure"]({"ticker": "AAA", "window": 10})


def test_a_pipeline_run_is_a_job_with_the_whole_report_and_a_short_history_is_refused_clearly(app):
    view = _wait(app, app.extra_post["/api/finml/run"]({"ticker": "AAA", "config": {"importance": True, "cpcv_groups": 4, "cusum_k": 0.8, "lookback": 20}})["job"], 240)
    assert view["status"] == "done", view
    r = view["result"]
    assert r["ticker"] == "AAA" and r["universe"]["days"] > 1000 and r["split"]["holdout_starts"] and r["validation"]["purging_leak_free"]
    assert {"development", "holdout", "cpcv", "quality", "importance", "verdict", "series"} <= set(r) and r["importance"]["mda"] and r["development"]["evidence"]["meta_filter"]["dsr"] is not None
    import json
    json.dumps(r)                                                                         # the whole report goes over HTTP
    bad = _wait(app, app.extra_post["/api/finml/run"]({"ticker": "AAA", "config": {"cusum_k": 9.5}})["job"], 120)
    assert bad["status"] == "error" and "labelled events" in bad["error"]


def test_a_structure_run_reports_explosiveness_level_shifts_and_entropy(app):
    view = _wait(app, app.extra_post["/api/finml/structure"]({"ticker": "AAA", "window": 400})["job"], 240)
    assert view["status"] == "done", view
    r = view["result"]
    assert r["days"] == 400 and r["sadf"]["critical_95"] > -5 and len(r["sadf"]["dates"]) == len(r["sadf"]["values"]) and 0 <= r["cusum"]["break_share"] <= 1
    assert r["entropy"]["whole_sample"]["sign"]["plug_in_bits_per_symbol"] > 0.8 and "last_250_days" in r["entropy"]
