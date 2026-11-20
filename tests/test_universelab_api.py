"""The Stock Universe Explorer through the app's request handlers, on the synthetic platform dataset (AAA to FFF)."""

from __future__ import annotations

import json

import pytest

from src.utils.config import load_config
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait

SIX = "AAA, BBB, CCC, DDD, EEE, FFF"
FAST = {"folds": 2, "min_train_days": 504, "bootstrap": 50}


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def _done(app, body, timeout=300):
    view = _wait(app, app.extra_post["/api/universe/explore"](body)["job"], timeout)
    assert view["status"] == "done", view
    json.dumps(view["result"], allow_nan=False)
    return view["result"]


def test_the_plan_lists_screens_methods_limits_and_the_honest_notes(app):
    plan = app.extra_get["/api/universe/plan"]({})
    assert {s["key"] for s in plan["screens"]} >= {"all", "liquidity", "size", "low_vol", "momentum", "sector", "manual"} and plan["max_candidates"] == 60
    assert any("survivorship" in n for n in plan["notes"]) and any("proxy" in n for n in plan["notes"]) and "membership_columns" in plan


@pytest.mark.parametrize("body, text", [
    ({"tickers": "AAA, BBB"}, "at least 6"),
    ({"tickers": SIX, "screens": [{"kind": "nope"}]}, "screen kind"),
    ({"tickers": SIX, "methods": ["magic"]}, "unknown method"),
    ({"tickers": SIX, "lookbacks": [5]}, "lookbacks"),
    ({"tickers": SIX, "design": {"folds": 50}}, "between"),
    ({"tickers": SIX, "design": {"sql": 1}}, "has no setting"),
    ({"tickers": SIX, "membership": "ticker\nAAA\n"}, "needs the columns"),
    ({"tickers": SIX, "screens": [{"kind": "low_vol", "top": t} for t in (0.2, 0.3, 0.4, 0.5, 0.6)], "methods": ["equal_weight", "hrp", "min_variance", "risk_parity"], "lookbacks": [126, 252, 504, 63]}, "narrow the grid"),
])
def test_a_bad_request_is_refused_before_a_job_is_queued(app, body, text):
    with pytest.raises(ApiError, match=text):
        app.extra_post["/api/universe/explore"](body)


def test_a_study_runs_as_a_job_and_returns_the_whole_evidence(app):
    r = _done(app, {"tickers": SIX, "screens": [{"kind": "all"}, {"kind": "low_vol", "top": 0.5}], "methods": ["equal_weight", "min_variance"], "lookbacks": [126], "design": FAST})
    assert r["n_candidates"] == 4 and r["status"] == "selected_on_prior_data_only" and len(r["folds"]) == 2
    assert r["eligibility"]["hindsight"] is True and r["warnings"] and "survivors" in r["warnings"][0]
    assert {"oos", "benchmarks", "deflated_sharpe", "bootstrap", "regimes", "removal", "in_sample", "equity", "stability", "concentration"} <= set(r)
    assert r["universe"]["tickers"] == SIX.split(", ") and len(r["equity"]["dates"]) == len(r["equity"]["oos"])


def test_a_membership_table_removes_the_survivorship_warning_only_when_complete(app):
    table = "ticker,start,end,status\n" + "\n".join(f"{t},2000-01-03,," for t in SIX.split(", "))
    partial = _done(app, {"tickers": SIX, "membership": table, "methods": ["equal_weight"], "design": FAST})
    assert partial["eligibility"]["basis"] == "point_in_time" and "survivorship" in partial["warnings"][0]
    complete = _done(app, {"tickers": SIX, "membership": table, "membership_complete": True, "methods": ["equal_weight"], "design": FAST})
    assert complete["eligibility"]["coverage"] == "complete" and "warnings" not in complete
