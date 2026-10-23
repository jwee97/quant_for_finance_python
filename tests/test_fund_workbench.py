"""The workbench the page and the command line share: what is ready and what is missing, the lists and one company's view, the catalogue of choices, a research context kept while the data does not change,
strict checking of everything a form sends, a scan of many factors, formulas, combinations, a model, a backtest and the monitor, and no key in anything that leaves."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import expr as X
from src.fundamentals import prices
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.research import UniverseSpec
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile
from src.fundamentals.workbench import Workbench, eval_spec, pit_settings, universe_spec

FAKE_KEYS = {"OPENFILINGS_API_KEY_1": "of_zzzzSECRETzzzz_key_one_0123456789", "OPENFILINGS_API_KEY_2": "of_zzzzSECRETzzzz_key_two_9876543210"}
PARTS = [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}, {"kind": "technical", "name": "mom_12_1"}]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("workbench"))
    svc.load_simulated(n_companies=100, start_year=2009, end_year=2018, seed=13, planted={"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045})
    with svc.store.db.transaction() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return Workbench(service=svc)


def json_safe(value) -> str:
    return json.dumps(value, allow_nan=False)


# ------------------------------------------------------------------------------------------------------------------------ what is ready
def test_an_empty_database_says_what_to_do_first_and_a_loaded_one_is_ready(tmp_path, world):
    empty = Workbench(tmp_path)
    r = empty.readiness()
    assert not r["ready"] and r["companies"] == 0 and "add some in the Data Manager" in r["problems"][0]
    empty.store.upsert_security(CompanyProfile("openfilings", "us:AAA", "AAA", "A Co", exchange="US", country="US"))
    assert "No company has financial statements" in empty.readiness()["problems"][0]
    assert world.readiness()["ready"] and world.readiness()["companies"] == 100


def test_the_status_carries_the_quota_the_slots_and_the_policy_but_never_a_key(world, monkeypatch):
    for name, value in FAKE_KEYS.items():
        monkeypatch.setenv(name, value)
    fresh = Workbench(service=FundamentalsService(world.service.root, db_path=world.store.db.path))
    status = fresh.status()
    text = json_safe(status)
    assert "SECRET" not in text and "of_zzzz" not in text and "0123456789" not in text and "9876543210" not in text
    openf = next(p for p in status["providers"] if p["name"] == "openfilings")
    assert [k["slot"] for k in openf["keys"]] == ["key 1", "key 2"] and [k["variable"] for k in openf["keys"]] == ["OPENFILINGS_API_KEY_1", "OPENFILINGS_API_KEY_2"]
    assert [k["role"] for k in openf["keys"]] == ["active", "standby"] and openf["unit"] == "credits" and openf["daily_limit"] == 100 and openf["monthly_limit"] == 1000
    assert "rotating accounts or API keys" in status["policy"] and "OPENFILINGS_API_KEY_1" in status["key_setup"]["text"] and status["readiness"]["ready"]
    assert status["simulated"].startswith("100:2009:2018:13")


def test_the_overview_gathers_the_counts_the_coverage_the_library_and_the_recent_work(world):
    o = world.overview()
    json_safe(o)
    assert o["counts"]["companies"] == 100 and o["counts"]["with_financials"] == 100 and {c["country"] for c in o["by_country"]} == {"US", "JP", "AU", "DE", "GB", "SG"}
    assert sum(s["companies"] for s in o["by_sector"]) == 100 and o["metric_coverage"][0]["company_share"] <= 1.0 and o["library"]["tried"] == 0 and o["pipeline"][0] == "providers"
    assert o["quality"]["periods"] and o["readiness"]["ready"]


def test_companies_are_listed_searched_filtered_and_paged(world):
    page = world.companies(limit=10)
    assert page["total"] == 100 and len(page["companies"]) == 10 and "US" in page["countries"] and "Energy" in page["sectors"]
    first = page["companies"][0]
    assert first["annual"] >= 10 and first["price_days"] > 2000 and first["last_period_end"] and first["quarterly"] > 0 and first["first_period_end"] < first["last_period_end"]
    ticker = first["ticker"]
    hit = world.companies(query=ticker[:4])
    assert any(c["ticker"] == ticker for c in hit["companies"]) and hit["total"] <= 100
    energy = world.companies(sector="Energy", limit=500)
    assert energy["total"] == 12 and all(c["sector"] == "Energy" for c in energy["companies"])
    us = world.companies(country="US", sector="Energy", limit=500)
    assert all(c["country"] == "US" for c in us["companies"]) and us["total"] <= energy["total"]
    second = world.companies(limit=10, offset=10)
    assert {c["security_id"] for c in second["companies"]}.isdisjoint({c["security_id"] for c in page["companies"]})
    assert world.companies(query="no-such-company-xyz")["total"] == 0
    for bad in ({"limit": 0}, {"limit": 1000}, {"offset": -1}):
        with pytest.raises(FundamentalsError, match="between"):
            world.companies(**bad)
    with pytest.raises(FundamentalsError, match="must be a number"):
        world.companies(limit="many")


def test_one_company_shows_its_figures_ratios_decomposition_and_screens_with_the_dates_they_became_public(world):
    sid = int(world.store.securities().iloc[3]["security_id"])
    c = world.company(sid)
    json_safe(c)
    assert c["profile"]["security_id"] == sid and c["prices"]["days"] > 2000 and c["prices"]["last_close"] > 0 and c["summary"]["annual"] >= 10
    rows = c["annual"]
    assert len(rows) == 11 and rows == sorted(rows, key=lambda r: r["period_end"]) and all(r["available"] > r["period_end"] for r in rows)
    assert {"revenue", "net_income", "total_assets", "roe", "roic", "piotroski_f"} <= set(rows[-1]) and rows[-1]["fiscal_year"] == 2018
    assert len(c["dupont"]) == 11 and len(c["piotroski"]) == 11 and len(c["anomaly"]) == 11 and [s["key"] for s in c["dupont_steps"]][0] == "tax_burden" and len(c["piotroski_signals"]) == 9
    complete = [d for d in c["dupont"] if all(d[k["key"]] is not None for k in c["dupont_steps"]) and d["roe"] is not None]
    assert complete and complete[-1]["roe"] == pytest.approx(np.prod([complete[-1][k["key"]] for k in c["dupont_steps"]]), abs=1e-6)
    assert c["dupont"][-1]["method"] in ("log-mean (exact)", "needs every step positive in both years", "no prior year")
    assert c["dates"][-1] == rows[-1]["period_end"] and len(c["series"]["revenue"]) == 11 and any("screens" in n for n in c["notes"])
    with pytest.raises(FundamentalsError, match="no company with id"):
        world.company(10_000_000)


# ------------------------------------------------------------------------------------------------------------------------ the catalogue and the checks
def test_the_catalogue_lists_every_choice_the_forms_offer(world):
    c = world.catalogue()
    json_safe(c)
    assert len(c["factors"]) >= 80 and {f["family"] for f in c["factors"]} <= set(c["families"]) and "fcf_yield" in {f["name"] for f in c["factors"]}
    assert [t["op"] for t in c["transforms"]][:2] == ["winsorize", "zscore"] and all(t["text"] for t in c["transforms"]) and c["winsor_levels"] == [0.5, 1.0, 2.5, 5.0]
    assert [h["id"] for h in c["horizons"]][:3] == ["1d", "1w", "2w"] and {f["id"] for f in c["frequencies"]} == {"W", "M", "Q"}
    assert "rank" in c["operators"]["cross_sectional"] and c["formula"]["max_nodes"] == X.MAX_NODES and len(c["metrics"]) > 80
    assert c["discovery"]["stages"][0] == "sanity" and c["discovery"]["stages"][-1] == "locked_test" and c["combination"]["methods"] == ["equal", "ic", "icir", "diversified", "cluster"]
    assert "ridge" in c["combination"]["models"] and c["portfolio"]["defaults"]["side"] == "long_only" and "mom_12_1" in c["portfolio"]["technical"]
    assert c["library"]["statuses"][0] == "Candidate" and c["library"]["transitions"]["Retired"] == [] and c["providers"] == ["openfilings", "sec_edgar", "simulated"] and "not a validated alpha" in c["rule"]


def test_what_a_form_sends_is_checked_strictly():
    assert universe_spec({"countries": ["US", "JP"], "min_market_cap_usd": "1e9", "developed_only": 1}).countries == ("US", "JP") and universe_spec(None) == UniverseSpec()
    for bad, why in (({"countrys": ["US"]}, "unknown universe setting"), ({"countries": "US"}, "list of names"), ({"min_market_cap_usd": -1}, "between"), ({"sectors": [3]}, "list of names")):
        with pytest.raises(FundamentalsError, match=why):
            universe_spec(bad)
    assert pit_settings({"info_lag_days": 5, "naive_availability": True}).info_lag_days == 5
    for bad in ({"lag": 1}, {"info_lag_days": 100}):
        with pytest.raises(FundamentalsError):
            pit_settings(bad)
    s = eval_spec({"factor": "roic", "horizon": "3m", "quantiles": 4, "top_fraction": 0.1, "start": "2012-01-31", "direction": -1})
    assert s.factor == "roic" and s.quantiles == 4 and s.start == "2012-01-31" and s.direction == -1
    for bad, why in (({"factors": "roic"}, "unknown evaluation setting"), ({"direction": "up"}, "direction must be"), ({"horizon": "9y"}, "horizon must be"), ({"quantiles": 40}, "between"),
                     ({"top_fraction": 0.9}, "between"), ({"weighting": "kelly"}, "weighting must be"), ({"cost_bps": 9999}, "between"), ({"start": "yesterday-ish"}, "date such as"),
                     ({"steps": [{"op": "rm -rf"}]}, "list of at most 12 operations"), ({"steps": "zscore"}, "list of at most 12"), ({"breakdowns": ["planet"]}, "breakdowns are")):
        with pytest.raises(FundamentalsError, match=why):
            eval_spec(bad)


def test_a_context_is_built_once_for_a_universe_and_again_when_the_data_changes(world, tmp_path):
    a = world.context()
    assert world.context() is a and world.context({"countries": ["US"]}) is not a and world.context(freq="Q") is not a
    with pytest.raises(FundamentalsError, match="freq must be"):
        world.context(freq="D")
    fresh = Workbench(service=FundamentalsService(tmp_path, db_path=_copy(world, tmp_path)))
    c1 = fresh.context()
    assert fresh.context() is c1
    fresh.store.upsert_security(CompanyProfile("openfilings", "us:NEW", "NEW", "New Co", exchange="US", country="US"))
    fresh.store.store_prices(pd.DataFrame({"security_id": [fresh.store.get_security(1)["security_id"]], "date": ["2030-01-01"], "adj_close": [1.0]}))
    assert fresh.context() is not c1                                                                   # a new price row changes the data the context was built from
    for country in ("US", "JP", "AU"):
        fresh.context({"countries": [country]})
    assert len(fresh._contexts) == 2                                                                   # only the last two are kept
    fresh.forget()
    assert not fresh._contexts


def _copy(world, tmp_path):
    target = tmp_path / "copy.db"
    shutil.copy(world.store.db.path, target)
    return target


# ------------------------------------------------------------------------------------------------------------------------ factor studies, formulas, combinations
def test_a_scan_sets_many_factors_side_by_side_and_says_how_many_were_looked_at(world):
    ctx = world.context()
    seen = []
    scan = world.scan(ctx, ["fcf_yield", "d_roic", "roic", "asset_growth", "revenue_cagr_3y"], eval_spec({"horizon": "1m"}), progress=seen.append)
    json_safe(scan)
    rows = scan["rows"]
    assert [r["name"] for r in rows][:2] == ["fcf_yield", "d_roic"] and rows[0]["t_stat"] > rows[1]["t_stat"] > 2 and rows[0]["coverage"] > 0.5 and rows[0]["net_sharpe"] > 1.0
    assert abs(next(r for r in rows if r["name"] == "roic")["t_stat"]) < 2 and scan["tested"] == len(rows) - sum("error" in r for r in rows) and "map of where to look" in scan["note"]
    assert len(seen) == 5 and seen[0].startswith("fcf_yield (1 of 5)") and rows[0]["direction"] == 1
    stop = world.scan(ctx, ["fcf_yield", "d_roic", "roic"], eval_spec({}), stop=lambda: True)
    assert stop["rows"] == []
    with pytest.raises(FundamentalsError, match="no factor called"):
        world.scan(ctx, ["not_a_factor"], eval_spec({}))
    with pytest.raises(FundamentalsError, match="between 1 and"):
        world.scan(ctx, ["roic"] * 200, eval_spec({}))


def test_one_factor_is_studied_in_full_and_the_look_ahead_audit_runs(world):
    ctx = world.context()
    r = world.evaluate(ctx, eval_spec({"factor": "fcf_yield"}))
    json_safe(r)
    assert r["ic"]["t_stat"] > 5 and set(r["charts"]) >= {"ic", "quantile_mean", "ls_cumulative_net"} and "Interesting is not validated" in r["verdict"]
    audit = world.lookahead(eval_spec({"factor": "fcf_yield"}), None, "M", None)
    json_safe(audit)
    assert set(audit) >= {"honest", "naive", "flattered", "verdict"} and audit["honest"]["mean_ic"] > 0.05


def test_a_formula_is_checked_then_evaluated_like_a_factor(world):
    ctx = world.context()
    ok = world.check_formula("div(fcf,market_cap)", ctx)
    assert ok["ok"] and ok["text"] == "div(fcf,market_cap)" and ok["fields"] == ["fcf", "market_cap"] and 0.5 < ok["coverage"] <= 1.0 and ok["tree"] == X.parse("div(fcf,market_cap)")
    assert not world.check_formula("div(fcf,")["ok"] and "ends too soon" in world.check_formula("div(fcf,")["error"]
    typed = world.check_formula("add(fcf,roic)", ctx)
    assert not typed["ok"] and typed["error"]
    r = world.evaluate_formula(ctx, "div(fcf,market_cap)", eval_spec({"direction": "auto"}))
    json_safe(r)
    assert r["ic"]["mean_ic"] > 0.05 and r["formula"]["ok"] and any("direction 'auto'" in w for w in r["warnings"])
    flipped = world.evaluate_formula(ctx, "div(fcf,market_cap)", eval_spec({"direction": -1}))
    assert flipped["ic"]["mean_ic"] == pytest.approx(-r["ic"]["mean_ic"], abs=1e-6)


def test_alphas_are_correlated_combined_and_fed_to_a_model(world):
    ctx = world.context()
    corr = world.correlation(ctx, PARTS)
    json_safe(corr)
    assert corr["alphas"] == 3 and 1.0 <= corr["effective_number"] <= 3.0 and sorted(m for c in corr["clusters"] for m in c["members"]) == sorted(corr["names"])
    combo = world.combination(ctx, PARTS, "icir", "1m")
    json_safe(combo)
    assert combo["method"] == "icir" and len(combo["parts"]) == 3 and combo["combined"]["mean_ic"] > combo["equal_weight"]["mean_ic"] and combo["weights"]["names"][0] == "fcf_yield"
    assert len(combo["weights"]["dates"]) == len(combo["weights"]["values"]) and abs(sum(combo["weights"]["values"][-1]) - 1.0) < 1e-3 and combo["evaluation"]["ic"]["n_dates"] > 80
    ml = world.machine_learning(ctx, PARTS, "ridge", "1m", {"permutation_repeats": 1})
    json_safe(ml)
    assert ml["model"] == "ridge" and ml["folds"] and ml["comparison"]["model"]["mean_ic"] > 0 and set(ml["importance"]["permutation"]) == set(ml["importance"]["alphas"])
    assert ml["evaluation"]["ic"]["n_dates"] > 50 and ml["parts"] == ["fcf_yield", "d_roic", "mom_12_1"]
    with pytest.raises(FundamentalsError, match="between 2 and"):
        world.correlation(ctx, PARTS[:1])


def test_a_backtest_comes_back_as_plain_json(world):
    ctx = world.context()
    r = world.backtest(ctx, PARTS[:2], {"side": "long_short", "n": 15})
    json_safe(r)
    assert r["summary"]["sharpe"] > 1.0 and r["experiment_id"].startswith("bt-") and r["signal"]["label"].startswith("1 x fcf_yield")
    assert world.experiments("backtest", 5)[0]["experiment_id"] == r["experiment_id"]


def test_the_monitor_judges_the_alphas_in_the_library_and_records_it(world, tmp_path):
    bench = Workbench(service=FundamentalsService(tmp_path, db_path=_copy(world, tmp_path)))
    alpha_id = bench.library.add_candidate(X.parse("div(fcf,market_cap)"), {"universe": {}, "freq": "M", "horizon": "1m", "entry_lag": 1, "min_assets": 20, "direction": 1.0,
                                                                         "steps": [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}]}, {"ic": 0.08, "direction": 1.0, "n_tested": 50})
    seen = []
    out = bench.monitor(progress=seen.append)
    json_safe(out)
    assert out["checked"] == 1 and out["results"][0]["alpha_id"] == alpha_id and out["results"][0]["monitor"] in ("Healthy", "Weakening", "Decaying", "Broken") and seen[0].startswith("div(fcf,market_cap)")
    a = bench.alpha(alpha_id)
    json_safe(a)
    assert a["status"] == "Candidate" and a["monitor"][0]["status"] == out["results"][0]["monitor"] and a["status_log"][0]["to_status"] == "Candidate" and a["transitions"] == ["Validated", "Rejected", "Retired"]
    assert bench.monitor([alpha_id], stop=lambda: True)["checked"] == 0


def test_a_discovery_request_is_checked_before_a_run_is_created(world):
    spec = world.discovery_spec({"n_candidates": 40, "seed": 3, "horizon": "3m", "seeds": ["div(fcf,market_cap)"], "thresholds": {"t_ic": 2.0}})
    assert spec.n_candidates == 40 and spec.horizon == "3m" and spec.limits().t_ic == 2.0
    for bad, why in (({"n_candidate": 5}, "unknown discovery setting"), ({"n_candidates": 0}, "between"), ({"horizon": "2y"}, "horizon must be"), ({"freq": "D"}, "freq must be"),
                     ({"train_fraction": 0.8, "validation_fraction": 0.3}, "locked test"), ({"seeds": ["div(fcf,"]}, "ends too soon"), ({"seeds": "roic"}, "list of formulas"),
                     ({"thresholds": {"magic": 1}}, "unknown threshold"), ({"universe": {"cities": []}}, "unknown universe setting"), ({"max_nodes": 100}, "between")):
        with pytest.raises(FundamentalsError, match=why):
            world.discovery_spec(bad)
    ctx = world.context()
    assert "div(fcf,market_cap)" in world.seeds_for(ctx) and world.splits_for(ctx, world.discovery_spec({}))["train"][0] >= "2010"


def test_prices_loaded_through_the_workbench_drop_the_cached_contexts(world, tmp_path):
    bench = Workbench(service=FundamentalsService(tmp_path, db_path=_copy(world, tmp_path)))
    bench.context()
    assert bench._contexts
    for sid in bench.store.securities()["security_id"]:
        prices.set_price_symbol(bench.store, int(sid), f"S{int(sid)}")                                   # the simulated companies are on no real exchange: say what to ask for

    def fetch(symbols, start):
        return {s: pd.DataFrame({"close": 10.0, "adj_close": 10.0, "volume": 5.0}, index=pd.bdate_range("2019-01-01", periods=5)) for s in symbols}

    report = bench.load_prices(start="2019-01-01", fetch=fetch, refresh=True)
    json_safe(report)
    assert report["loaded"] == 100 and report["rows"] == 500 and report["fx"]["currencies"] and not bench._contexts
