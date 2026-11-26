"""Comparing saved studies without misleading: recomputation over the common window from the saved series, the rules that stop a ranking (costs, construction, currency, horizon, dates, data version, kind of study),
the ones that only warn, and the separation of predictive quality from trading performance. Small deterministic series with answers worked out by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research import compare as rc
from src.research import store as rs


@pytest.fixture()
def store(tmp_path):
    return rs.ResearchStore(tmp_path / "research")


def _curve(start, days, daily, seed=0):
    idx = pd.bdate_range(start, periods=days)
    r = np.full(days, daily) + (np.random.default_rng(seed).standard_normal(days) * 0.002 if seed else 0.0)
    return idx, r


def _trading(store, s, title, start, days, daily, seed, cost=10.0, mu="historical mean", method="hrp", **study):
    idx, r = _curve(start, days, daily, seed)
    eq = (1 + pd.Series(r, index=idx)).cumprod()
    cfg_study = {"lookback": 252, "step": 21, "cov_method": "shrinkage", "max_weight": 1.0, **study}
    return store.add_artifact(s, {"kind": "portfolio_lab", "title": title, "evidence": "historical_simulation", "data_kind": "platform",
                                  "context": {"assets": ["A", "B", "C"], "start": str(idx[0].date()), "end": str(idx[-1].date()), "costs": {"cost_bps": cost}, "dataset_version": "v1"},
                                  "config": {"method": method, "study": cfg_study, "mu_source": mu}, "metrics": {"sharpe": 1.0}, "benchmark": "Equal weight", "costs": {"cost_bps": cost},
                                  "series": {"dates": [str(d.date()) for d in idx], "equity": [float(x) for x in eq]}}), pd.Series(r, index=idx)


def _factor(store, s, title, horizon=5, assets=("A", "B", "C"), start="2020-01-01", end="2022-01-01", ic=0.05, version="v1"):
    return store.add_artifact(s, {"kind": "factor_study", "title": title, "evidence": "historical_simulation", "data_kind": "platform",
                                  "context": {"assets": list(assets), "start": start, "end": end, "dataset_version": version}, "config": {"horizon": horizon, "factors": ["momentum", "reversal"]},
                                  "metrics": {"best_combination_test_ic": ic, "n_tested": 2.0, "survive_bh": 1.0}})


def _forecast(store, s, title, horizon=5, skill=0.04):
    return store.add_artifact(s, {"kind": "forecast", "title": title, "evidence": "walk_forward", "data_kind": "platform",
                                  "context": {"assets": ["A"], "start": "2020-01-01", "end": "2022-01-01", "dataset_version": "v1"}, "config": {"horizon": horizon}, "metrics": {"best_skill_common": skill}})


def codes(out):
    return {i["code"]: i for i in out["issues"]}


# ------------------------------------------------------------------------------------------------------------------------ recomputation on the common window
def test_series_on_different_dates_are_measured_over_the_days_they_share_and_the_numbers_are_the_ones_computed_by_hand(store):
    s = store.create_session("c")["id"]
    a, ra = _trading(store, s, "long", "2020-01-01", 600, 0.0006, 1)
    b, rb = _trading(store, s, "short", "2020-09-01", 400, 0.0004, 2)
    out = rc.compare([store.get_artifact(a["id"]), store.get_artifact(b["id"])])
    lo, hi = max(ra.index[0], rb.index[0]), min(ra.index[-1], rb.index[-1])
    assert out["window"] == {"start": str(lo.date()), "end": str(hi.date())}
    for art, r in ((a, ra), (b, rb)):
        want = rc.measure(r.loc[lo:hi])
        got = out["metrics"][art["id"]]
        assert got["sharpe"] == pytest.approx(want["sharpe"], abs=1e-9) and got["total_return"] == pytest.approx(float((1 + r.loc[lo:hi]).prod() - 1), abs=1e-9)
    assert codes(out)["window_cut"]["severity"] == "note" and out["rankable"] is True and len(out["ranking"]) == 2
    assert out["studies"]["roles"] == {a["id"]: "trading", b["id"]: "trading"}


def test_a_window_too_short_to_compare_is_reported_and_blocks_a_ranking(store):
    s = store.create_session("c")["id"]
    a, _ = _trading(store, s, "early", "2020-01-01", 100, 0.0005, 1)
    b, _ = _trading(store, s, "late", "2020-05-01", 100, 0.0005, 2)
    out = rc.compare([store.get_artifact(a["id"]), store.get_artifact(b["id"])])
    assert out["rankable"] is False and out["ranking"] is None and "short_overlap" in codes(out) and out["metrics"] == {}


# ------------------------------------------------------------------------------------------------------------------------ what stops a ranking and what only warns
def test_different_trading_costs_stop_a_ranking_and_the_same_cost_does_not(store):
    s = store.create_session("c")["id"]
    a, _ = _trading(store, s, "cheap", "2020-01-01", 300, 0.0005, 1, cost=5.0)
    b, _ = _trading(store, s, "dear", "2020-01-01", 300, 0.0005, 2, cost=25.0)
    c, _ = _trading(store, s, "same", "2020-01-01", 300, 0.0005, 3, cost=5.0)
    out = rc.compare([store.get_artifact(x["id"]) for x in (a, b)])
    assert codes(out)["different_cost_bps"]["severity"] == "blocks_ranking" and "5 bp" in codes(out)["different_cost_bps"]["message"] and "25 bp" in codes(out)["different_cost_bps"]["message"]
    assert out["rankable"] is False and out["ranking"] is None and "not ranked" in out["reason"]
    ok = rc.compare([store.get_artifact(x["id"]) for x in (a, c)])
    assert "different_cost_bps" not in codes(ok) and ok["rankable"] is True


def test_different_construction_stops_a_ranking_but_a_different_source_of_expected_returns_is_the_point_of_the_comparison(store):
    s = store.create_session("c")["id"]
    a, _ = _trading(store, s, "history", "2020-01-01", 300, 0.0005, 1, method="max_sharpe", mu="historical mean")
    b, _ = _trading(store, s, "forecast", "2020-01-01", 300, 0.0005, 2, method="max_sharpe", mu="forecast: historical")
    c, _ = _trading(store, s, "wider window", "2020-01-01", 300, 0.0005, 3, method="max_sharpe", lookback=504)
    d, _ = _trading(store, s, "another optimiser", "2020-01-01", 300, 0.0005, 4, method="hrp", mu="forecast: historical")
    same_build = rc.compare([store.get_artifact(x["id"]) for x in (a, b)])
    assert codes(same_build)["different_expected_returns"]["severity"] == "note" and same_build["rankable"] is True and "different_construction" not in codes(same_build)
    built = rc.compare([store.get_artifact(x["id"]) for x in (a, c)])
    assert codes(built)["different_construction"]["severity"] == "blocks_ranking" and "lookback" in codes(built)["different_construction"]["message"] and built["rankable"] is False
    both = rc.compare([store.get_artifact(x["id"]) for x in (a, d)])
    assert codes(both)["different_methods_and_sources"]["severity"] == "warning" and "cannot say which of the two" in codes(both)["different_methods_and_sources"]["message"]


def test_results_in_different_currencies_or_against_different_benchmarks_are_flagged(store):
    s = store.create_session("c")["id"]
    a, _ = _trading(store, s, "usd", "2020-01-01", 300, 0.0005, 1)
    raw_b, _ = _trading(store, s, "eur", "2020-01-01", 300, 0.0005, 2)
    full = store.get_artifact(a["id"])
    eur = {**full, "config": {**full["config"], "currency": "EUR"}, "benchmark": "MSCI Europe", "id": "bbbbbbbbbbbb"}
    usd = {**full, "config": {**full["config"], "currency": "USD"}}
    out = rc.compare([usd, eur])
    assert codes(out)["different_currency"]["severity"] == "blocks_ranking" and codes(out)["different_benchmark"]["severity"] == "warning" and out["rankable"] is False


# ------------------------------------------------------------------------------------------------------------------------ predictive quality is not trading performance
def test_factor_studies_and_forecast_evaluations_are_compared_only_on_the_same_horizon_dates_instruments_and_data(store):
    s = store.create_session("c")["id"]
    a, b = _factor(store, s, "f1", ic=0.06), _factor(store, s, "f2", ic=0.02)
    ok = rc.compare([store.get_artifact(x["id"]) for x in (a, b)])
    assert ok["studies"]["predictive"]["ranking"] == [a["id"], b["id"]] and ok["studies"]["predictive"]["blocked_because"] == []
    assert "do not by themselves" in ok["studies"]["predictive"]["note"] or "does not by itself" in ok["studies"]["predictive"]["note"]
    for kw, word in (({"horizon": 21}, "horizons"), ({"assets": ("A", "B")}, "instruments"), ({"end": "2021-01-01"}, "evaluation dates"), ({"version": "v2"}, "data versions")):
        other = _factor(store, s, "other", **kw)
        out = rc.compare([store.get_artifact(a["id"]), store.get_artifact(other["id"])])
        assert out["studies"]["predictive"]["ranking"] is None and any(word in r for r in out["studies"]["predictive"]["blocked_because"]), kw
    fc = _forecast(store, s, "fc")
    mixed = rc.compare([store.get_artifact(a["id"]), store.get_artifact(fc["id"])])
    assert mixed["studies"]["predictive"]["ranking"] is None and "different questions" in " ".join(mixed["studies"]["predictive"]["blocked_because"])
    assert codes(mixed)["different_studies"]["severity"] == "blocks_ranking"


def test_a_predictive_study_beside_trading_results_does_not_stop_the_trading_ranking_and_is_never_in_it(store):
    s = store.create_session("c")["id"]
    a, _ = _trading(store, s, "p1", "2020-01-01", 300, 0.0005, 1)
    b, _ = _trading(store, s, "p2", "2020-01-01", 300, 0.0004, 2)
    f = _forecast(store, s, "forecast")
    out = rc.compare([store.get_artifact(x["id"]) for x in (a, b, f)])
    assert out["rankable"] is True and set(out["ranking"]) == {a["id"], b["id"]} and f["id"] not in out["metrics"]
    assert out["studies"]["roles"][f["id"]] == "predictive" and codes(out)["mixed_roles"]["severity"] == "note"
    assert any(u["id"] == f["id"] and "not trading performance" in u["reason"] for u in out["studies"]["unmeasured"])
    assert out["missing_series"] == [f["id"]]


def test_a_study_with_no_series_says_why_nothing_was_recomputed(store):
    s = store.create_session("c")["id"]
    a, b = _factor(store, s, "f1"), _factor(store, s, "f2")
    out = rc.compare([store.get_artifact(a["id"]), store.get_artifact(b["id"])])
    assert out["rankable"] is False and "compared on their saved summaries" in out["reason"] and out["metrics"] == {}
    assert all("no return series was saved" in u["reason"] or "predictive study" in u["reason"] for u in out["studies"]["unmeasured"])


def test_the_assumptions_are_listed_side_by_side_and_the_ones_that_differ_are_named(store):
    s = store.create_session("c")["id"]
    a, _ = _trading(store, s, "a", "2020-01-01", 300, 0.0005, 1, cost=5.0, mu="historical mean")
    b, _ = _trading(store, s, "b", "2020-01-01", 300, 0.0005, 2, cost=5.0, mu="forecast: historical")
    out = rc.compare([store.get_artifact(x["id"]) for x in (a, b)])["studies"]["assumptions"]
    assert out["differing"] == ["mu"] and out["values"][a["id"]]["costs"] == "cost_bps 5.0" and out["values"][a["id"]]["benchmark"] == "Equal weight" and out["values"][b["id"]]["mu"] == "forecast: historical"
    assert {r["key"] for r in out["rows"]} >= {"assets", "dataset", "horizon", "costs", "benchmark", "construction", "mu", "evidence", "pit", "version"}


# ------------------------------------------------------------------------------------------------------------------------ attribution
def test_attribution_reports_are_compared_line_by_line_only_when_they_use_the_same_factor_model(store):
    s = store.create_session("c")["id"]

    def with_attr(title, kind):
        a, _ = _trading(store, s, title, "2020-01-01", 300, 0.0005, 1 if kind == "market" else 2)
        art = store.get_artifact(a["id"])
        return {**art, "attribution": {"method": "hrp", "kind": kind, "benchmark": "equal_weight", "factor": {"totals": {"market": 0.1, "specific": 0.02, "costs": -0.003}, "reconciliation": {"status": "reconciled"}},
                                       "brinson": {"totals": {"allocation": 0.004}, "reconciliation": {"status": "reconciled"}}}}
    same = rc.compare([with_attr("m1", "market"), {**with_attr("m2", "market"), "id": "cccccccccccc"}])
    assert "different_attribution_models" not in codes(same) and len(same["studies"]["attribution"]) == 2 and same["studies"]["attribution"][0]["factor_status"] == "reconciled"
    mixed = rc.compare([with_attr("m", "market"), {**with_attr("s", "statistical"), "id": "dddddddddddd"}])
    assert codes(mixed)["different_attribution_models"]["severity"] == "warning" and "not the same quantities" in codes(mixed)["different_attribution_models"]["message"]
