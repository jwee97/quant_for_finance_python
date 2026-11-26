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


# ------------------------------------------------------------------------------------------------------ fundamental factors, admitted only after the point-in-time check
def _fundamentals(app, tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"), filing_lag=50, early=None):
    """Annual statements for 2011 to 2019 whose profitability differs by company and year; ``early`` makes one company's filings predate their periods (a wrong date)."""
    from src.fundamentals.providers.base import CompanyProfile
    from src.fundamentals.store import PeriodStatement
    store = app.fundamentals.bench("real").store
    for i, t in enumerate(tickers):
        sid = store.upsert_security(CompanyProfile("manual", f"us:{t}", t, f"{t} Inc", exchange="US", market="US", country="US", currency="USD"))
        periods = []
        for y in range(2011, 2020):
            filed = f"{y}-11-01" if t == early else (__import__("datetime").date(y, 12, 31) + __import__("datetime").timedelta(days=filing_lag)).isoformat()
            periods.append(PeriodStatement(fiscal_period="FY", period_end=f"{y}-12-31", fiscal_year=y, filing_date=filed, reporting_currency="USD",
                                           values={"revenue": 1000.0 + 20 * i, "net_income": 50.0 + 10 * i + 3 * (y - 2011) * (1 if i % 2 else -1), "total_assets": 2000.0, "shareholders_equity": 800.0 + 15 * i,
                                                   "gross_profit": 400.0 + 12 * i, "operating_cash_flow": 90.0 + 5 * i}))
        store.store_financials(sid, "manual", periods)
    return store


def test_fundamental_factors_enter_the_study_after_the_point_in_time_check_and_the_audit_travels_with_the_result(app):
    store = _fundamentals(app)
    out = _done(app, "/api/lab/factors", {"tickers": SIX, "signals": ["momentum"], "fundamental_factors": ["net_margin", "roe"], "horizon": 21})
    names = [r["factor"] for r in out["comparison"]["table"]]
    assert set(names) == {"momentum", "fund:net_margin", "fund:roe"} and out["comparison"]["n_tested"] == 3, "the family corrected for is every factor tried, fundamental ones included"
    fund = out["fundamental"]
    assert fund["pit"]["verdict"]["status"] == "ready" and fund["pit"]["estimated"] == 0 and fund["coverage"]["roe"]["tickers_with_a_value"] == 6 and fund["settings"]["info_lag_days"] == 1
    assert any("point-in-time" in n for n in out["notes"]) and fund["database"] == "real"
    sig = out["discovery"]["signals"]["fund:roe"]
    assert sig["rank_persistence"]["lag21"] > 0.9, "a figure that changes once a year barely changes rank from one month to the next"
    # the factor on the study's own calendar is the figure public that day, never the one for a period still being reported
    from src.alphalab.fundamental import panels
    import pandas as pd
    got = panels(store, ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], ["roe"], pd.bdate_range("2013-01-01", "2013-12-31"))["panels"]["fund:roe"]
    assert got["AAA"].loc["2013-02-01"] == pytest.approx(50.0 / 800.0), "on 1 Feb 2013 the latest public year is 2011: the 2012 statements are filed on 19 Feb 2013"
    before, after = got["AAA"].loc["2013-02-19"], got["AAA"].loc["2013-02-20"]
    assert before == pytest.approx(50.0 / 800.0) and after == pytest.approx(47.0 / 800.0), "the 2012 figure appears the day after its filing (19 Feb plus one day) and not on the filing day"


def test_a_study_with_a_wrong_filing_date_is_refused_with_the_reason(app):
    _fundamentals(app, early="CCC")
    view = _wait(app, app.extra_post["/api/lab/factors"]({"tickers": SIX, "signals": ["momentum"], "fundamental_factors": ["roe"]})["job"], 120)
    assert view["status"] == "error" and "point-in-time check blocks these figures" in view["error"] and "wrong date" in view["error"]


def test_fundamental_factors_need_statements_for_enough_of_the_chosen_tickers(app):
    _fundamentals(app, tickers=("AAA", "BBB"))
    view = _wait(app, app.extra_post["/api/lab/factors"]({"tickers": SIX, "signals": ["momentum"], "fundamental_factors": ["roe"]})["job"], 120)
    assert view["status"] == "error" and "at least 6 of the chosen tickers" in view["error"] and "holds 2" in view["error"]


def test_the_fundamental_request_is_checked_before_a_job_starts(app):
    post = app.extra_post["/api/lab/factors"]
    with pytest.raises(ApiError, match="distinct metric names"):
        post({"tickers": SIX, "fundamental_factors": ["roe", "roe"]})
    with pytest.raises(ApiError, match="fundamentals_db"):
        post({"tickers": SIX, "fundamental_factors": ["roe"], "fundamentals_db": "elsewhere"})
    _fundamentals(app)
    view = _wait(app, post({"tickers": SIX, "signals": [], "fundamental_factors": ["roe"]})["job"], 60)
    assert view["status"] == "error" and "at least two" in view["error"], "one factor cannot be compared or combined, fundamental or not"


# ------------------------------------------------------------------------------------------------------ keeping studies in a research session, and not comparing what cannot be compared
def _factor_job(app, horizon=21, **body):
    view = _wait(app, app.extra_post["/api/lab/factors"]({"tickers": SIX, "signals": ["momentum", "reversal", "low_vol"], "horizon": horizon, **body})["job"], 240)
    assert view["status"] == "done", view
    return view


def test_a_factor_study_is_kept_with_its_assumptions_and_two_studies_of_different_horizons_cannot_be_ranked(app):
    post = app.extra_post
    session = post["/api/research/session/create"]({"name": "Factors", "context": {"assets": ["aaa", "bbb", "ccc", "ddd", "eee", "fff"], "start": "2012-01-02"}})["session"]
    a_view, b_view = _factor_job(app, 5), _factor_job(app, 21)
    a = post["/api/research/artifact/save"]({"session": session["id"], "job": a_view["id"]})["artifact"]
    b = post["/api/research/artifact/save"]({"session": session["id"], "job": b_view["id"]})["artifact"]
    full = app.extra_get["/api/research/artifact"]({"id": a["id"]})
    assert full["kind"] == "factor_study" and full["evidence"] == "historical_simulation" and not full.get("series")
    assert full["config"]["horizon"] == 5 and set(full["config"]["factors"]) == {"momentum", "reversal", "low_vol"}
    assert full["metrics"]["n_tested"] == 3 and full["metrics"]["n_trials"] == 3 and full["context"]["assets"] == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"] and full["refs"]["job"] == a_view["id"]
    assert any("not ranked by performance" in n for n in full["notes"]) and full["context"]["dataset_version"]
    cmp = post["/api/research/compare"]({"artifacts": [a["id"], b["id"]]})
    codes = {i["code"]: i for i in cmp["issues"]}
    assert codes["different_horizon"]["severity"] == "blocks_ranking" and "5 days" in codes["different_horizon"]["message"] and "21 days" in codes["different_horizon"]["message"]
    assert cmp["rankable"] is False and set(cmp["missing_series"]) == {a["id"], b["id"]}, "listed and checked for comparability, never ranked"
    same = post["/api/research/compare"]({"artifacts": [a["id"], post["/api/research/artifact/save"]({"session": session["id"], "job": a_view["id"]})["artifact"]["id"]]})
    assert not [i for i in same["issues"] if i["severity"] == "blocks_ranking"], "the same study twice is comparable"


def test_a_fundamental_study_is_kept_with_its_point_in_time_status_and_the_comparison_warns_about_assumed_dates(app):
    _fundamentals(app, tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"))
    store = app.fundamentals.bench("real").store
    from src.fundamentals.providers.base import CompanyProfile
    from src.fundamentals.store import PeriodStatement
    sid = int(store.securities().iloc[0]["security_id"])
    store.store_financials(sid, "manual", [PeriodStatement(fiscal_period="FY", period_end="2020-12-31", fiscal_year=2020, values={"revenue": 1.0, "net_income": 0.1, "shareholders_equity": 5.0}, reporting_currency="USD")])   # no filing date: assumed
    post = app.extra_post
    session = post["/api/research/session/create"]({"name": "Fundamentals"})["session"]
    with_f = _factor_job(app, 21, signals=["momentum"], fundamental_factors=["roe", "net_margin"])
    price_only = _factor_job(app, 21)
    a = post["/api/research/artifact/save"]({"session": session["id"], "job": with_f["id"]})["artifact"]
    b = post["/api/research/artifact/save"]({"session": session["id"], "job": price_only["id"]})["artifact"]
    full = app.extra_get["/api/research/artifact"]({"id": a["id"]})
    fund = full["config"]["fundamental"]
    assert fund["pit_status"] == "ready_with_assumptions" and fund["assumed_dates"] == 1 and full["config"]["uses_fundamentals"] is True and fund["pit_settings"]["info_lag_days"] == 1
    assert any("assumed" in n for n in full["notes"]) and any("point-in-time check" in n for n in full["notes"])
    cmp = post["/api/research/compare"]({"artifacts": [a["id"], b["id"]]})
    codes = {i["code"]: i for i in cmp["issues"]}
    assert codes["pit_assumptions"]["severity"] == "warning" and codes["fundamentals_mixed"]["severity"] == "warning"


# ------------------------------------------------------------------------------------------------------ expected-return sources and attribution through the portfolio endpoint
def _pf(app, **body):
    base = {"tickers": SIX, "methods": ["equal_weight", "min_variance", "max_sharpe"], "config": {"step": 21, "lookback": 252}}
    view = _wait(app, app.extra_post["/api/lab/portfolio"]({**base, **body})["job"], 600)
    return view


def test_the_portfolio_study_attributes_its_returns_and_says_why_brinson_is_off_without_a_classification(app):
    view = _pf(app, sector_source="none")
    assert view["status"] == "done", view
    out = view["result"]
    json.dumps(out)
    a = out["attribution"]
    assert a["kind"] == "market" and a["brinson_available"] is False and "classification" in a["brinson_reason"]
    auto = _pf(app)["result"]["attribution"]
    assert auto["brinson_available"] is True and auto["methods"]["max_sharpe"]["brinson"]["reconciliation"]["status"] == "reconciled", "the platform data carry asset classes, which are used when no classification is given"
    for key, m in a["methods"].items():
        rec = m["factor"]["reconciliation"]
        assert rec["status"] == "reconciled" and rec["max_daily_residual"] < 1e-12, key
        assert m["factor"]["totals"]["costs"] < 0 and {"market", "specific", "costs", "financing", "residual"} <= set(m["factor"]["totals"]) and m["factor"]["ex_ante"]
    assert out["expected_returns"]["source"] == "historical mean" and out["source_comparison"] is None and out["config"]["mu_source"] == "historical mean"
    with_groups = _pf(app, sectors={"AAA": "X", "BBB": "X", "CCC": "X", "DDD": "Y", "EEE": "Y", "FFF": "Y"}, attribution={"kind": "statistical", "n_factors": 2})["result"]["attribution"]
    assert with_groups["brinson_available"] and with_groups["kind"] == "statistical"
    b = with_groups["methods"]["max_sharpe"]["brinson"]
    assert b["reconciliation"]["status"] == "reconciled" and set(b["groups"]) == {"X", "Y"} and b["totals"]["costs"] < 0 or b["totals"]["costs"] == 0
    assert "PC1" in with_groups["methods"]["max_sharpe"]["factor"]["totals"]
    off = _pf(app, attribution=None)["result"]
    assert off["attribution"] is None


def test_forecast_and_factor_expected_returns_are_run_on_identical_settings_and_compared(app):
    view = _pf(app, expected_returns=[{"source": "historical"}, {"source": "factor", "factor": "momentum"}, {"source": "forecast", "forecast_method": "historical", "calibrate": "none"}])
    assert view["status"] == "done", view
    out = view["result"]
    cmp = out["source_comparison"]
    assert list(cmp) == ["historical mean", "factor: momentum", "forecast: historical"]
    assert out["expected_returns"]["source"] == "historical mean"
    mv = {lbl: next(r for r in v["table"] if r["key"] == "min_variance")["sharpe"] for lbl, v in cmp.items()}
    assert len(set(mv.values())) == 1, "a method that ignores expected returns is identical whatever the source"
    ms = {lbl: next(r for r in v["table"] if r["key"] == "max_sharpe")["sharpe"] for lbl, v in cmp.items()}
    assert len(set(ms.values())) > 1, "and one that uses them is not"
    fore = cmp["forecast: historical"]
    assert fore["meta"]["forecast_contract"]["model_id"] == "historical" and fore["meta"]["forecast_contract"]["kind"] == "distribution" and fore["meta"]["training_fingerprint_example"]
    assert any(f > 0 for f in fore["mu_report"]["filled_with_historical"]), "early rebalances have no usable forecast and use the historical mean, which is counted"


def test_the_expected_return_request_is_checked_before_a_job_starts(app):
    post = app.extra_post["/api/lab/portfolio"]
    for er, word in (({"source": "oracle"}, "historical, factor, forecast or combined"), ({"source": "factor", "factor": "alpha"}, "factor must be"), ({"source": "forecast", "forecast_method": "garch"}, "forecast_method must be"),
                     ({"source": "forecast", "horizon": 5}, "only at its own horizon"), ({"source": "forecast", "calibrate": "x"}, "calibrate"), ({"source": "forecast", "horizon": 99, "allow_horizon_scaling": True}, "1 to 63"),
                     ([{"source": "factor"}, {"source": "factor"}], "twice")):
        with pytest.raises(ApiError, match=word):
            post({"tickers": SIX, "expected_returns": er, "config": {"step": 21}})
    with pytest.raises(ApiError, match="attribution.kind"):
        post({"tickers": SIX, "attribution": {"kind": "magic"}})
    with pytest.raises(ApiError, match="sectors must map"):
        post({"tickers": SIX, "sectors": ["a"]})
