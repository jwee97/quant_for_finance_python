"""The Fundamentals tab's endpoints: the routes and the version, the two databases (a simulated demo that never mixes with real data), what the status shows and never shows (a key), the readers, the
refusals before a job starts, the data jobs (statements fetched within a budget, prices loaded), the research jobs (scan, study, formula, look-ahead, discovery, correlation, combination, model, backtest,
monitor), stopping a long job, and the answers going out as plain JSON over HTTP."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from src.fundamentals import expr as X
from src.fundamentals.keys import KeyPool
from src.fundamentals.providers.openfilings import OpenFilingsProvider
from src.fundamentals.usage import Allowance
from src.utils.config import load_config
from src.webapp import api, server
from src.webapp.api import ApiError, App
from src.webapp.fundamentals_api import FundamentalsApi
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_fund_data import KEY_1, FakeOpenFilings
from tests.test_webapp import _fetch

PARTS = [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}, {"kind": "technical", "name": "mom_12_1"}]


def make_app(tmp_path):
    config = load_config()
    bundle = make_bundle(n_assets=4, n=400, seed=3, alpha=0.0)
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=lambda c: bundle), root=tmp_path)


def wait(app, job, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        view = app.job(job)
        if view["status"] in ("done", "error", "interrupted"):
            return view
        time.sleep(0.05)
    raise AssertionError("the job did not end")


def post(app, path, **body):
    return app.extra_post[path](body)


def get(app, path, **query):
    return app.extra_get[path](query)


def run(app, path, **body):
    started = post(app, path, **body)
    view = wait(app, started["job"])
    assert view["status"] == "done", view
    return view["result"]


def failed(app, path, **body):
    started = post(app, path, **body)
    view = wait(app, started["job"])
    assert view["status"] == "error", view
    return view["error"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    app = make_app(tmp_path_factory.mktemp("fund_app"))
    made = run(app, "/api/fund/simulate", db="demo", companies=40, start_year=2010, end_year=2018, seed=5, planted={"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045})
    assert made["companies"] == 40
    return app


# ------------------------------------------------------------------------------------------------------------------------------ the routes
def test_the_routes_are_in_the_app_and_the_api_version_says_so(app):
    for path in ("/api/fund/status", "/api/fund/overview", "/api/fund/catalogue", "/api/fund/companies", "/api/fund/company", "/api/fund/library", "/api/fund/run"):
        assert path in app.extra_get
    for path in ("/api/fund/sync", "/api/fund/simulate", "/api/fund/scan", "/api/fund/discover", "/api/fund/backtest", "/api/fund/stop", "/api/fund/ml"):
        assert path in app.extra_post
    assert api.API_VERSION == 11 and app.catalog()["api_version"] == 11
    assert (Path(api.__file__).resolve().parents[2] / "docs" / api.DOC_ROOT_FILES["fundamentals"]).is_file()      # the guides viewer can open the chapter


def test_the_two_databases_are_separate_files_and_the_demo_never_calls_a_provider(app):
    status = get(app, "/api/fund/status", db="demo")
    assert status["database"] == "demo" and status["databases"]["demo"]["companies"] == 40 and status["databases"]["real"]["companies"] == 0
    assert status["readiness"]["ready"] and status["simulated"].startswith("40:2010:2018:5")
    for bad, why in (({"planted": {"magic": 1}}, "planted is a dictionary"), ({"planted": {"fcf_yield": 5}}, "between"), ({"companies": 10}, "between"), ({"start_year": 2016, "end_year": 2018}, "between")):
        with pytest.raises(ApiError, match=why):
            post(app, "/api/fund/simulate", db="demo", **bad)
    real = get(app, "/api/fund/status")
    assert real["database"] == "real" and not real["readiness"]["ready"] and "add some in the Data Manager" in real["readiness"]["problems"][0]
    assert app.root.joinpath("data", "user", "fundamentals_demo.db").exists() and app.root.joinpath("data", "user", "fundamentals.db").exists()
    for path, body in (("/api/fund/companies/add", {"symbols": ["AAPL"]}), ("/api/fund/sync", {"budget": 10}), ("/api/fund/sync/plan", {}), ("/api/fund/provider/test", {}),
                       ("/api/fund/prices", {}), ("/api/fund/price-symbol", {"security_id": 1, "symbol": "X"})):
        with pytest.raises(ApiError, match="Switch to Real data|switch to Real data"):
            post(app, path, db="demo", **body)
    with pytest.raises(ApiError, match="made in the demo database"):
        post(app, "/api/fund/simulate", companies=40)
    with pytest.raises(ApiError, match="db must be one of"):
        get(app, "/api/fund/status", db="both")


def test_the_status_shows_the_slots_the_allowance_and_the_policy_but_never_a_key(app, monkeypatch):
    monkeypatch.setenv("OPENFILINGS_API_KEY_1", "of_zzzzSECRETzzzz_one_0123456789")
    monkeypatch.setenv("OPENFILINGS_API_KEY_2", "of_zzzzSECRETzzzz_two_9876543210")
    fresh = FundamentalsApi(app)
    status = fresh.get_routes()["/api/fund/status"]({})
    text = json.dumps(status, allow_nan=False)
    assert "SECRET" not in text and "of_zzzz" not in text and "0123456789" not in text
    openf = next(p for p in status["providers"] if p["name"] == "openfilings")
    assert [k["role"] for k in openf["keys"]] == ["active", "standby"] and openf["daily_limit"] == 100 and "rotating accounts or API keys" in status["policy"]
    assert not any("key" in str(route).lower() and "set" in str(route).lower() for route in fresh.post_routes())                 # nothing accepts a key


def test_readers_answer_from_the_demo_database(app):
    over = get(app, "/api/fund/overview", db="demo")
    assert over["counts"]["companies"] == 40 and over["library"]["tried"] == 0 and over["readiness"]["ready"]
    page = get(app, "/api/fund/companies", db="demo", limit="5", q="")
    assert page["total"] == 40 and len(page["companies"]) == 5
    company = get(app, "/api/fund/company", db="demo", id=str(page["companies"][0]["security_id"]))
    assert company["profile"]["ticker"] == page["companies"][0]["ticker"] and len(company["annual"]) >= 8 and company["prices"]["days"] > 1500
    cat = get(app, "/api/fund/catalogue")
    assert len(cat["factors"]) >= 80 and cat["portfolio"]["defaults"]["side"] == "long_only"
    assert get(app, "/api/fund/quality", db="demo")["counts"]["observations"]["ok"] > 1000
    assert get(app, "/api/fund/library", db="demo")["alphas"] == [] and get(app, "/api/fund/graveyard", db="demo")["rows"] == [] and get(app, "/api/fund/runs", db="demo")["runs"] == []
    with pytest.raises(ApiError, match="say which company"):
        get(app, "/api/fund/company", db="demo")
    with pytest.raises(ApiError, match="no company with id"):
        get(app, "/api/fund/company", db="demo", id="999999")
    with pytest.raises(ApiError, match="status must be one of"):
        get(app, "/api/fund/library", db="demo", status="Great")
    with pytest.raises(ApiError, match="between"):
        get(app, "/api/fund/companies", db="demo", limit="0")


# ------------------------------------------------------------------------------------------------------------------------------ the data jobs
def test_statements_are_fetched_within_a_budget_that_the_request_must_state(tmp_path):
    local = make_app(tmp_path)
    wb = local.fundamentals.bench("real")
    fake = FakeOpenFilings({KEY_1: "a"})
    pool = KeyPool("openfilings", [("OPENFILINGS_API_KEY_1", KEY_1)], wb.service.usage, wb.store.salt(), Allowance("credits", 100, 1000, "test"))
    wb.service.register("openfilings", OpenFilingsProvider(pool, transport=fake, sleep=lambda s: None))
    added = post(local, "/api/fund/companies/add", symbols="AAPL, jp:7203, us:MSFT")
    assert added["added"] == 3
    plan = post(local, "/api/fund/sync/plan", periods=["annual"], start_year=2015)
    assert plan["plan"]["cost"] == 33 and plan["plan"]["unit"] == "credits" and plan["affordability"]["remaining_today"] == 100 and fake.requests == []
    with pytest.raises(ApiError, match="how many credits"):
        post(local, "/api/fund/sync", periods=["annual"])
    with pytest.raises(ApiError, match="between"):
        post(local, "/api/fund/sync", budget=0)
    with pytest.raises(ApiError, match="from OpenFilings"):
        post(local, "/api/fund/sync", budget=20, provider="simulated")
    report = run(local, "/api/fund/sync", budget=22, periods=["annual"], start_year=2015)
    assert report["fetched"] == 2 and report["cost"] == 22 and "budget of 22" in report["stopped"] and "1 companies remain" in report["stopped"]
    assert local.extra_get["/api/fund/status"]({})["counts"]["with_financials"] == 2
    again = run(local, "/api/fund/sync", budget=50, periods=["annual"], start_year=2015)
    assert again["fetched"] == 1 and again["cost"] == 11
    tested = post(local, "/api/fund/provider/test")
    assert tested["ok"] and tested["plan"] == "free"
    for bad, why in (({"symbols": []}, "name at least one company"), ({"symbols": ["bad symbol!"]}, "not a valid ticker"), ({"symbols": ["A"] * 300}, "at most 200"), ({"symbols": "AAPL", "market": "united states"}, "short code")):
        with pytest.raises(ApiError, match=why):
            post(local, "/api/fund/companies/add", **bad)


def test_a_company_is_found_in_the_directory_by_name_then_added_for_nothing_and_fetched(tmp_path, monkeypatch, app):
    for name in [n for n in list(os.environ) if n.startswith("OPENFILINGS_")]:
        monkeypatch.delenv(name, raising=False)
    local = make_app(tmp_path)
    empty = get(local, "/api/fund/companies", q="Nvidia")                                      # the database is empty, so the list is: the page must not call that "no match"
    assert (empty["total"], empty["countries"], empty["sectors"]) == (0, [], []) and empty["lookup"]["available"] is False and "OPENFILINGS_API_KEY" in empty["lookup"]["reason"]
    with pytest.raises(ApiError, match="no OPENFILINGS_API_KEY"):
        post(local, "/api/fund/companies/search", q="Nvidia")
    wb = local.fundamentals.bench("real")
    fake = FakeOpenFilings({KEY_1: "a"})
    pool = KeyPool("openfilings", [("OPENFILINGS_API_KEY_1", KEY_1)], wb.service.usage, wb.store.salt(), Allowance("credits", 100, 1000, "test"))
    wb.service.register("openfilings", OpenFilingsProvider(pool, transport=fake, sleep=lambda s: None))
    assert get(local, "/api/fund/companies", q="Nvidia")["lookup"] == {"available": True, "reason": "", "cost": 1, "fetch_cost": 11, "unit": "credits", "remaining_today": 100}
    assert fake.requests == []                                                                  # looking at the list never asks the provider
    found = post(local, "/api/fund/companies/search", q="Nvidia")
    json.dumps(found, allow_nan=False)                                                          # plain JSON
    hit = found["results"][0]
    assert (found["count"], found["credits"], found["cached"], found["remaining_today"]) == (1, 1, False, 99) and (hit["symbol"], hit["name"], hit["in_database"]) == ("us:NVDA", "NVIDIA Corp", False)
    assert len(fake.requests) == 1 and KEY_1 not in json.dumps(found)
    added = post(local, "/api/fund/companies/add", symbols=["us:NVDA"])
    assert added["added"] == 1 and added["companies"] == [{"security_id": added["security_ids"][0], "ticker": "NVDA", "name": "NVIDIA Corp"}] and len(fake.requests) == 1
    page = get(local, "/api/fund/companies", q="nvidia")                                       # now the name finds it, and the filters have something to offer
    assert page["total"] == 1 and page["companies"][0]["company_name"] == "NVIDIA Corp" and page["countries"] == ["US"] and page["lookup"]["remaining_today"] == 99
    assert post(local, "/api/fund/companies/search", q="nvidia")["cached"] is True and len(fake.requests) == 1
    report = run(local, "/api/fund/sync", symbols=["us:NVDA"], budget=11)
    assert report["fetched"] == 1 and report["cost"] == 11 and not report["failed"]
    row = get(local, "/api/fund/companies", q="NVDA")["companies"][0]
    assert row["annual"] >= 8 and row["sector"] == "Technology"
    assert post(local, "/api/fund/companies/search", q="Nvidia")["results"][0]["periods"] >= 8
    for bad, why in (({"q": ""}, "at least two characters"), ({"q": "x" * 121}, "at least two characters"), ({"q": 7}, "at least two characters"), ({"q": "Nvidia", "kind": "isbn"}, "kind must be one of"),
                     ({"q": "Nvidia", "market": "united states"}, "short code")):
        with pytest.raises(ApiError, match=why):
            post(local, "/api/fund/companies/search", **bad)
    with pytest.raises(ApiError, match="real data"):
        post(local, "/api/fund/companies/search", db="demo", q="Nvidia")                      # the demo is a simulated world and never calls a provider
    assert "lookup" not in get(app, "/api/fund/companies", db="demo", limit="1")


def test_prices_are_loaded_by_a_job_and_a_symbol_can_be_set_by_hand(tmp_path):
    local = make_app(tmp_path)
    wb = local.fundamentals.bench("real")
    ids = wb.service.add_companies("openfilings", ["us:AAA", "us:BBB", "jp:7203"])
    asked = []

    def fetch(symbols, start):
        asked.append(list(symbols))
        idx = pd.bdate_range("2020-01-01", periods=10)
        return {s: pd.DataFrame({"close": 10.0, "adj_close": 9.5, "volume": 100.0}, index=idx) for s in symbols}

    local.fundamentals.fetch = fetch
    report = run(local, "/api/fund/prices", start="2020-01-01")
    assert report["loaded"] == 3 and report["rows"] == 30 and asked[0] == ["AAA", "BBB", "7203.T"] and "fx" in report and not report["failed"]
    assert post(local, "/api/fund/price-symbol", security_id=ids[0], symbol="aaa.x")["symbol"] == "AAA.X"
    with pytest.raises(ApiError, match="price symbol"):
        post(local, "/api/fund/price-symbol", security_id=ids[0], symbol="a b")
    with pytest.raises(ApiError, match="say which company"):
        post(local, "/api/fund/price-symbol", symbol="X")


# ------------------------------------------------------------------------------------------------------------------------------ the research jobs
def test_a_scan_sets_factors_side_by_side_and_a_study_gives_the_full_tear_sheet(app):
    scan = run(app, "/api/fund/scan", db="demo", names=["fcf_yield", "d_roic", "roic", "revenue_cagr_3y"], spec={"horizon": "1m"})
    assert scan["tested"] >= 3 and scan["rows"][0]["name"] in ("fcf_yield", "d_roic") and "map of where to look" in scan["note"]
    study = run(app, "/api/fund/evaluate", db="demo", spec={"factor": "fcf_yield"}, universe={"countries": []}, freq="M")
    assert study["ic"]["mean_ic"] > 0.03 and "Interesting is not validated" in study["verdict"] and set(study["charts"]) >= {"ic", "ls_cumulative_net", "quantile_mean"}
    audit = run(app, "/api/fund/lookahead", db="demo", spec={"factor": "fcf_yield"})
    assert set(audit) >= {"honest", "naive", "flattered", "verdict"}
    for body, why in (({"spec": {}}, "name a factor"), ({"spec": {"factor": "fcf_yield", "horizon": "9y"}}, "horizon must be"), ({"spec": {"factor": "fcf_yield", "bogus": 1}}, "unknown evaluation setting"),
                      ({"spec": {"factor": "fcf_yield"}, "freq": "D"}, None)):
        if why:
            with pytest.raises(ApiError, match=why):
                post(app, "/api/fund/evaluate", db="demo", **body)
    assert "freq must be" in failed(app, "/api/fund/evaluate", db="demo", spec={"factor": "fcf_yield"}, freq="D")
    assert "no factor called" in failed(app, "/api/fund/evaluate", db="demo", spec={"factor": "astrology"})
    with pytest.raises(ApiError, match="names must be a list"):
        post(app, "/api/fund/scan", db="demo", names="roic")


def test_a_formula_is_checked_and_studied(app):
    uni = {"countries": ["JP", "AU", "DE"]}                                            # a universe nothing else in this module has built the data for
    ok = post(app, "/api/fund/formula/check", db="demo", formula="div(fcf,market_cap)", universe=uni)
    assert ok["ok"] and ok["fields"] == ["fcf", "market_cap"] and "coverage" not in ok
    assert not post(app, "/api/fund/formula/check", db="demo", formula="div(fcf,")["ok"]
    study = run(app, "/api/fund/evaluate/formula", db="demo", formula="div(fcf,market_cap)", spec={"direction": "auto", "min_assets": 8}, universe=uni)
    assert study["formula"]["ok"] and study["universe"]["companies"] < 40 and study["ic"]["n_dates"] > 40
    after = post(app, "/api/fund/formula/check", db="demo", formula="div(fcf,market_cap)", universe=uni)
    assert after["coverage"] > 0.4 and after["well_typed"]                       # a study has built the data, so the check can use it
    with pytest.raises(ApiError, match="write a formula"):
        post(app, "/api/fund/formula/check", db="demo", formula=" ")
    with pytest.raises(ApiError, match="write a formula"):
        post(app, "/api/fund/evaluate/formula", db="demo", formula="x" * 3000)


def test_alphas_are_correlated_combined_modelled_and_backtested(app):
    corr = run(app, "/api/fund/correlation", db="demo", parts=PARTS)
    assert corr["alphas"] == 3 and corr["heatmap"]["names"] and "independent ideas" in corr["reading"]
    combo = run(app, "/api/fund/combine", db="demo", parts=PARTS, method="icir", horizon="1m")
    assert combo["method"] == "icir" and len(combo["parts"]) == 3 and combo["combined"]["n_dates"] > 50 and combo["weights"]["names"]
    model = run(app, "/api/fund/ml", db="demo", parts=PARTS, model="ridge", options={"permutation_repeats": 1, "min_train": 24})
    assert model["folds"] and model["comparison"]["verdict"] and model["importance"]["alphas"]
    bt = run(app, "/api/fund/backtest", db="demo", signal=PARTS[:2], portfolio={"side": "long_short", "n": 8, "min_names": 4}, name="demo book")
    assert bt["experiment_id"].startswith("bt-") and bt["summary"]["n_obs"] > 1000 and bt["limits"]["dates"] > 50 and any("survivorship" in w for w in bt["warnings"])
    assert "between 2 and" in failed(app, "/api/fund/combine", db="demo", parts=PARTS[:1], method="equal")
    assert "method must be one of" in failed(app, "/api/fund/combine", db="demo", parts=PARTS, method="magic")
    assert "model must be one of" in failed(app, "/api/fund/ml", db="demo", parts=PARTS, model="neural")
    assert "whole number" in failed(app, "/api/fund/backtest", db="demo", signal=PARTS[0], portfolio={"n": 0})
    with pytest.raises(ApiError, match="say what to backtest"):
        post(app, "/api/fund/backtest", db="demo", portfolio={})
    with pytest.raises(ApiError, match="options must be a dictionary"):
        post(app, "/api/fund/combine", db="demo", parts=PARTS, options=[1])
    assert any(e["kind"] == "backtest" for e in get(app, "/api/fund/experiments", db="demo", kind="backtest")["experiments"])


def test_a_discovery_run_is_counted_kept_resumable_and_never_called_validated(app):
    started = post(app, "/api/fund/discover", db="demo", spec={"n_candidates": 30, "seed": 4, "seeds": ["div(fcf,market_cap)", "delta(roic,12)"], "finalists": 3})
    run_id = started["run_id"]
    view = wait(app, started["job"])
    assert view["status"] == "done", view
    summary = view["result"]
    assert summary["run_id"] == run_id and summary["progress"] == {"done": 30, "total": 30} and summary["counts"]["tested"] >= 10 and summary["funnel"][0]["stage"] == "sanity"
    assert "Nothing here is validated" in summary["note"]
    detail = get(app, "/api/fund/run", db="demo", id=run_id, limit="50")
    assert detail["run_id"] == run_id and detail["candidates"] and {"text", "status", "stage_reached", "metrics"} <= set(detail["candidates"][0])
    runs = get(app, "/api/fund/runs", db="demo")["runs"]
    assert runs[0]["run_id"] == run_id and runs[0]["status"] == "finished"
    lib = get(app, "/api/fund/library", db="demo")
    assert all(a["status"] == "Candidate" for a in lib["alphas"]) and lib["counts"]["tried"] >= 1
    with pytest.raises(ApiError, match="that run has finished"):
        post(app, "/api/fund/discover", db="demo", run_id=run_id)                                                      # its locked test has been looked at: looking again would spend it
    grave = get(app, "/api/fund/graveyard", db="demo")
    assert grave["counts"]["graveyard"] >= 1 and all(r["reason_rejected"] for r in grave["rows"])
    with pytest.raises(ApiError, match="no discovery run called"):
        post(app, "/api/fund/discover", db="demo", run_id="Rnothere")
    with pytest.raises(ApiError, match="between"):
        post(app, "/api/fund/discover", db="demo", spec={"n_candidates": 0})


def test_a_stopped_discovery_run_resumes_exactly_where_it_left_off(app):
    started = post(app, "/api/fund/discover", db="demo", spec={"n_candidates": 60, "seed": 9, "seeds": ["div(fcf,market_cap)", "delta(roic,12)"], "finalists": 3})
    deadline = time.time() + 120
    while time.time() < deadline and "formulas" not in app.job(started["job"])["message"]:
        time.sleep(0.02)
    assert post(app, "/api/fund/stop", job=started["job"]) == {"ok": True}
    stopped = wait(app, started["job"])
    assert stopped["status"] == "done" and 0 < stopped["result"]["progress"]["done"] < 60 and stopped["result"]["status"] == "stopped"
    done_before = stopped["result"]["progress"]["done"]
    final = run(app, "/api/fund/discover", db="demo", run_id=started["run_id"])
    assert final["progress"] == {"done": 60, "total": 60} and final["status"] == "finished" and final["counts"]["tested"] >= stopped["result"]["counts"]["tested"] and done_before < 60


def test_the_library_moves_an_alpha_only_along_allowed_paths_and_the_monitor_judges_it(app):
    wb = app.fundamentals.bench("demo")
    alpha_id = wb.library.add_candidate(X.parse("div(fcf,market_cap)"), {"universe": {}, "freq": "M", "horizon": "1m", "entry_lag": 1, "min_assets": 20, "direction": 1.0,
                                                                      "steps": [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}]}, {"ic": 0.08, "direction": 1.0, "n_tested": 40, "oos_t": 0.2})
    detail = get(app, "/api/fund/alpha", db="demo", id=alpha_id)
    assert detail["status"] == "Candidate" and detail["transitions"] == ["Validated", "Rejected", "Retired"] and detail["status_log"][0]["to_status"] == "Candidate"
    with pytest.raises(ApiError, match="not validated"):
        post(app, "/api/fund/library/validate", db="demo", alpha_id=alpha_id)
    with pytest.raises(ApiError, match="cannot go straight to Production"):
        post(app, "/api/fund/library/status", db="demo", alpha_id=alpha_id, status="Production", reason="why not")
    with pytest.raises(ApiError, match="needs a reason"):
        post(app, "/api/fund/library/status", db="demo", alpha_id=alpha_id, status="Rejected", reason=" ")
    moved = post(app, "/api/fund/library/status", db="demo", alpha_id=alpha_id, status="Rejected", reason="not convincing")
    assert moved["status"] == "Rejected"
    kept = get(app, "/api/fund/library", db="demo", status="Rejected")
    assert [a["alpha_id"] for a in kept["alphas"]] == [alpha_id]                       # a rejected alpha stays in the library
    result = run(app, "/api/fund/monitor", db="demo", alpha_ids=[alpha_id])
    assert result["checked"] == 1 and result["results"][0]["monitor"] in ("Healthy", "Weakening", "Decaying", "Broken")
    with pytest.raises(ApiError, match="alpha_ids must be"):
        post(app, "/api/fund/monitor", db="demo", alpha_ids="x")


def test_a_long_job_can_be_stopped_and_keeps_what_it_has(app):
    started = post(app, "/api/fund/scan", db="demo", spec={"horizon": "1m"})
    deadline = time.time() + 120
    while time.time() < deadline and "(" not in app.job(started["job"])["message"]:
        time.sleep(0.02)
    assert post(app, "/api/fund/stop", job=started["job"]) == {"ok": True}
    view = wait(app, started["job"])
    assert view["status"] == "done" and 0 < view["result"]["tested"] + sum("error" in r for r in view["result"]["rows"]) < 84
    with pytest.raises(ApiError, match="not running in this app"):
        post(app, "/api/fund/stop", job="nothere")


def test_what_goes_out_over_http_is_plain_json(app):
    running = server.start(app, port=0)
    try:
        def call(path, body=None):
            req = urllib.request.Request(running.url.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST",
                                         headers={"X-Token": running.token, **({"Content-Type": "application/json"} if body is not None else {})})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        assert call("/api/fund/status?db=demo")["counts"]["companies"] == 40
        assert call("/api/fund/companies?db=demo&limit=3")["total"] == 40
        assert call("/api/fund/formula/check", {"db": "demo", "formula": "roic"})["ok"]
    finally:
        running.stop()
