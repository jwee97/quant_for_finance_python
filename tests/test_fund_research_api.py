"""The research-reliability routes (health, screen, scenarios, events, families) on the simulated demo world."""

from __future__ import annotations

import pytest

from src.webapp.api import ApiError
from tests.test_fund_api import app, get, post  # noqa: F401  (the module's simulated world)


def test_the_routes_exist(app):
    for p in ("/api/fund/health", "/api/fund/events", "/api/fund/sectors", "/api/fund/lineage", "/api/fund/alpha/inputs"):
        assert p in app.extra_get
    for p in ("/api/fund/screen", "/api/fund/screen/history", "/api/fund/valuation/scenarios", "/api/fund/events/record", "/api/fund/family", "/api/fund/sources/prefer"):
        assert p in app.extra_post


def test_data_health_lists_every_company_with_what_is_missing(app):
    d = get(app, "/api/fund/health", db="demo")
    assert d["n"] == 40 and all("factor_readiness" in c and "status" in c for c in d["companies"])
    assert get(app, "/api/fund/health", db="demo", security_id=d["companies"][0]["security_id"])["n"] == 1


def test_a_screen_over_the_api_ranks_and_explains(app):
    out = post(app, "/api/fund/screen", db="demo", rules=[{"factor": "earnings_yield", "op": ">", "value": 0}], top=5)
    assert out["counts"]["pass"] >= len(out["rows"]) and len(out["rows"]) <= 5 and out["universe"] == 40 and all(e["why"] for e in out["excluded"])
    hist = post(app, "/api/fund/screen/history", db="demo", rules=[{"factor": "earnings_yield", "op": "top_pct", "value": 30}], step=12)
    assert len(hist["dates"]) >= 3
    with pytest.raises(ApiError, match="rules must be a list"):
        post(app, "/api/fund/screen", db="demo", rules="x")
    with pytest.raises(ApiError, match="no factor"):
        post(app, "/api/fund/screen", db="demo", rules=[{"factor": "nope", "op": ">", "value": 1}])


def test_scenarios_show_the_cases_side_by_side_with_a_grid_and_refuse_impossible_assumptions(app):
    sid = get(app, "/api/fund/health", db="demo")["companies"][0]["security_id"]
    out = post(app, "/api/fund/valuation/scenarios", db="demo", security_id=sid, cases={"base": [0.09, 0.02], "bear": [0.11, 0.0]})
    assert set(out["cases"]) == {"base", "bear"} and out["sensitivity"]["rates"] and out["yardsticks"]
    with pytest.raises(ApiError, match="discount rate"):
        post(app, "/api/fund/valuation/scenarios", db="demo", security_id=sid, cases={"x": [0.03, 0.02]})
    with pytest.raises(ApiError, match="cases must be"):
        post(app, "/api/fund/valuation/scenarios", db="demo", security_id=sid, cases=[1])


def test_a_corporate_event_and_a_family_can_be_recorded_and_are_validated(app):
    sid = get(app, "/api/fund/health", db="demo")["companies"][1]["security_id"]
    post(app, "/api/fund/events/record", db="demo", security_id=sid, event_date="2015-06-30", kind="spin_off", value_fraction=0.2, counterparty="NewCo")
    ev = get(app, "/api/fund/events", db="demo")
    assert [e["kind"] for e in ev["events"] if e["security_id"] == sid] == ["spin_off"] and "candidates" in ev and "splits" in ev
    with pytest.raises(ApiError, match="kind is one of"):
        post(app, "/api/fund/events/record", db="demo", security_id=sid, event_date="2015-06-30", kind="teleport")
    post(app, "/api/fund/family", db="demo", security_id=sid, family="bank")
    with pytest.raises(ApiError, match="bank, insurer, reit or general"):
        post(app, "/api/fund/family", db="demo", security_id=sid, family="spaceship")
    post(app, "/api/fund/family", db="demo", security_id=sid, family=None)
