"""The backend of the Execution and Cash flows tabs: simulations on a stylised market and a portfolio followed through money in and out, validated like any other request."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.framework import bundle_from_prices
from src.utils.config import load_config
from src.webapp import api, server
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.test_webapp import CLASSES, IDX, N, _default, _fetch, _http, _prices


def _rich(config):
    """A stand-in for the platform dataset that also has dollar volume and the ten-year yield, which the redemption and liability tools use."""
    rng = np.random.default_rng(1)
    macro = pd.DataFrame({"VIX": 20 + np.cumsum(rng.normal(0, 0.3, N)), "DGS10": (3 + np.cumsum(rng.normal(0, 0.02, N))).clip(0.5, 6)}, index=IDX)
    prices = _prices(1, CLASSES)
    volume = pd.DataFrame(1e6 * (1 + rng.random(prices.shape)), index=prices.index, columns=prices.columns)
    return bundle_from_prices(prices, asset_class=dict(CLASSES), macro=macro, volume=volume, name="fake-platform")


def _app(tmp_path, loader):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=loader), root=config.root)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return _app(tmp_path_factory.mktemp("rich"), _rich)


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    """An app whose data has no volume and no interest rates."""
    return _app(tmp_path_factory.mktemp("bare"), _default)


def _json_safe(value):
    json.dumps(value, allow_nan=False)                         # the server refuses NaN and infinity, so a result that carried one would be a 500
    return value


# ------------------------------------------------------------------------------------------------------ the page and its guides
def test_the_version_the_new_tabs_need_is_announced_and_the_coverage_guide_is_listed(lab):
    assert lab.catalog()["api_version"] == api.API_VERSION == 3
    slugs = [d["slug"] for d in lab.docs_index()["docs"]]
    assert "algorithmic_trading" in slugs
    assert lab.doc("algorithmic_trading")["markdown"].lstrip().startswith("#")
    assert {"technique-execution-algorithms", "technique-basket-and-liquidity-algorithms", "technique-black-box-and-high-frequency-strategies", "technique-cash-flow-strategies"} <= set(slugs)


# ------------------------------------------------------------------------------------------------------ the execution lab
def test_comparing_algorithms_returns_summaries_schedules_the_market_and_the_numbers_that_made_them(lab):
    r = _json_safe(lab.exec_lab({"action": "run", "algos": ["twap", "vwap", "is:risk_aversion=0.003"], "paths": 40, "shares": 100_000, "adv": 1_000_000, "sigma": 0.015}))
    assert [x["algo"] for x in r["rows"]] == ["twap", "vwap", "is:risk_aversion=0.003"] and all(x["paths"] == 40 for x in r["rows"])
    assert set(r["schedules"]) == {"twap", "vwap", "is:risk_aversion=0.003"} and all(abs(sum(v) - 1.0) < 1e-6 for v in r["schedules"].values())
    assert len(r["volume_profile"]) == r["intervals"] == 26
    assert r["order"]["shares"] == 100_000 and r["order"]["participation"] == pytest.approx(0.1)
    assert r["inputs"]["adv"] == 1_000_000 and r["inputs"]["sigma"] == 0.015 and r["inputs"]["side"] == "buy"
    cost = {x["algo"]: x for x in r["rows"]}
    assert cost["is:risk_aversion=0.003"]["std_bps"] < cost["vwap"]["std_bps"]                  # paying for urgency buys a narrower range of outcomes


def test_the_same_seed_gives_the_same_days_and_another_seed_gives_other_days(lab):
    body = {"action": "run", "algos": ["vwap"], "paths": 30}
    a, b = lab.exec_lab({**body, "seed": 4}), lab.exec_lab({**body, "seed": 4})
    c = lab.exec_lab({**body, "seed": 5})
    assert a["rows"][0]["shortfall_bps"] == b["rows"][0]["shortfall_bps"] and a["rows"][0]["shortfall_bps"] != c["rows"][0]["shortfall_bps"]


def test_algorithms_may_be_sent_as_one_string_and_a_blank_target_means_none(lab):
    r = lab.exec_lab({"action": "run", "algos": "twap, vwap", "target_bps": "", "paths": 20})
    assert [x["algo"] for x in r["rows"]] == ["twap", "vwap"]
    assert lab.exec_lab({"paths": 20})["rows"][0]["algo"] == "twap"                                # the action and the algorithms have defaults


def test_a_tactic_and_a_style_and_a_scenario_are_accepted(lab):
    r = lab.exec_lab({"action": "run", "algos": ["aim+vwap", "target_cost+is:risk_aversion=0.003"], "style": "passive", "scenario": "crisis", "side": "sell", "paths": 20, "target_bps": 15})
    assert r["style"]["name"] == "passive" and r["scenario"]["name"] == "crisis" and r["order"]["side"] == "sell"


def test_the_frontier_trades_cost_for_risk(lab):
    r = _json_safe(lab.exec_lab({"action": "frontier", "shares": 400_000}))
    risk, cost = [p["risk_bps"] for p in r["frontier"]], [p["cost_bps"] for p in r["frontier"]]
    assert all(b <= a + 1e-3 for a, b in zip(risk, risk[1:])) and all(b >= a - 1e-3 for a, b in zip(cost, cost[1:]))
    assert r["vwap"]["cost_bps"] <= r["twap"]["cost_bps"] and r["inputs"]["shares"] == 400_000
    assert r["frontier"][-1]["first_slice"] > r["frontier"][0]["first_slice"]                          # the more urgent, the more is traded at the open


def test_the_basket_report_has_the_joint_schedule_the_partial_executions_and_the_dark_set(lab):
    r = _json_safe(lab.exec_lab({"action": "basket", "size": 6, "seed": 3, "share": 0.4}))
    assert r["size"] == 6 and len(r["names"]) == len(r["value"]) == len(r["side"]) == 6
    assert r["joint"]["objective"] <= r["independent"]["objective"] + 1e-9
    assert len(r["risk_left_bps"]["joint"]) == len(r["executed"]["independent"]) == r["intervals"]
    assert r["mtrq"]["share"] == 0.4 and r["mtrq"]["residual_risk"] <= r["mtrq"]["naive_risk"] + 1e-6 <= r["mtrq"]["original_risk"] + 1e-6
    assert set(r["program_block"]["dark"]) <= set(r["program_block"]["block"]) and r["inputs"] == {"size": 6, "seed": 3, "risk_aversion": 0.001, "block_threshold": 0.005, "share": 0.4}


@pytest.mark.parametrize("sim", ["pairs", "etf", "amm", "rebate"])
def test_each_high_frequency_simulation_returns_a_table_and_the_figure_to_chart(lab, sim):
    r = _json_safe(lab.exec_lab({"action": "hft", "sim": sim, "seed": 1}))
    assert r["sim"] == sim and r["rows"] and r["chart"] in r["columns"] and all(set(r["columns"]) <= set(row) and row["label"] for row in r["rows"])
    assert "stylised market" in r["note"] and r["inputs"] == {"sim": sim, "seed": 1}
    if sim == "pairs":
        assert len(r["series"]["x"]) == len(r["series"]["y"]) == r["rows"][0]["days"]
    if sim == "etf":
        assert [row["label"] for row in r["rows"]][:2] == ["0 bars late", "1 bar late"] and r["rows"][0]["mean_daily_bps"] > r["rows"][-1]["mean_daily_bps"]
    if sim == "amm":
        assert {row["label"] for row in r["rows"]} == {"inventory-shaded (Avellaneda-Stoikov)", "symmetric quotes"}


@pytest.mark.parametrize("body, message", [
    ({"action": "nope"}, "action must be one of"),
    ({"action": "run", "shares": "lots"}, "shares must be a number"),
    ({"action": "run", "shares": float("nan")}, "shares must be between"),
    ({"action": "run", "paths": 10**9}, "paths must be between"),
    ({"action": "run", "sigma": 5}, "sigma must be between"),
    ({"action": "run", "algos": ["bogus"]}, "unknown algorithm"),
    ({"action": "run", "algos": ["pov:nope=1"]}, "does not accept those settings"),
    ({"action": "run", "algos": ["zzz+vwap"]}, "unknown tactic"),
    ({"action": "run", "algos": ["twap"] * 9}, "between one and eight"),
    ({"action": "run", "algos": 5}, "algos must be a list"),
    ({"action": "run", "style": "sneaky"}, "style must be one of"),
    ({"action": "run", "scenario": "boom"}, "scenario must be one of"),
    ({"action": "run", "side": "hold"}, "side must be one of"),
    ({"action": "frontier", "adv": 0}, "adv must be between"),
    ({"action": "basket", "size": 1}, "size must be between"),
    ({"action": "basket", "size": 17}, "size must be between"),
    ({"action": "basket", "share": 2}, "share must be between"),
    ({"action": "hft", "sim": "zzz"}, "sim must be one of"),
])
def test_a_request_that_cannot_run_is_refused_with_the_reason(lab, body, message):
    with pytest.raises(ApiError, match=message):
        lab.exec_lab(body)


# ------------------------------------------------------------------------------------------------------ the cash-flow lab
def test_simulating_flows_follows_each_policy_and_the_accounts_add_up(lab):
    r = _json_safe(lab.cash_lab({"action": "simulate", "tickers": ["AAA", "CCC"], "weights": {"AAA": 60, "CCC": 40}, "deposit": 1000, "initial": 50_000, "policies": ["pro_rata", "correct_drift"]}))
    assert r["mix"] == pytest.approx({"AAA": 0.6, "CCC": 0.4}) and [x["policy"] for x in r["rows"]] == ["pro_rata", "correct_drift"]
    assert len(r["dates"]) == len(r["nav"]["pro_rata"]) == len(r["deviation"]["correct_drift"]) and r["period"][0] < r["period"][1]
    for row in r["rows"]:
        assert row["deposits"] > 0 and row["final_value"] == pytest.approx(50_000 + row["deposits"] - row["withdrawals"] + row["profit"])           # withdrawals are reported as a positive amount
    assert r["inputs"]["initial"] == 50_000 and r["inputs"]["rebalance"] == "none" and r["universe"]["tickers"] == ["AAA", "CCC"]


def test_withdrawals_dividends_and_a_scheduled_rebalance_are_passed_through(lab):
    body = {"action": "simulate", "tickers": ["AAA", "CCC"], "weights": {"AAA": 0.5, "CCC": 0.5}, "initial": 1_000_000, "withdraw": 3000, "dividend_yield": 0.02, "dividend_policy": "reinvest",
            "policies": ["pro_rata"], "cost_bps": 0}
    free, restored = lab.cash_lab({**body, "rebalance": "none"}), lab.cash_lab({**body, "rebalance": "quarterly"})
    row = restored["rows"][0]
    assert row["withdrawals"] > 0 and row["dividends"] > 0 and restored["inputs"]["dividend_policy"] == "reinvest" and restored["inputs"]["rebalance"] == "quarterly"
    assert row["max_deviation"] < free["rows"][0]["max_deviation"] and row["turnover"] > free["rows"][0]["turnover"]            # restoring the mix keeps it closer and trades more


def test_with_no_weights_the_portfolio_is_equal_weight_and_the_default_tickers_are_used(lab):
    r = lab.cash_lab({"action": "simulate", "tickers": ["AAA", "BBB", "CCC", "DDD"], "policies": ["pro_rata"]})
    assert r["mix"] == pytest.approx({t: 0.25 for t in ["AAA", "BBB", "CCC", "DDD"]})
    assert set(lab.cash_lab({"action": "redeem"})["mix"]) == set(CLASSES)


def test_spending_rules_report_ruin_cuts_the_bands_and_the_highest_safe_rate(lab):
    r = _json_safe(lab.cash_lab({"action": "spending", "tickers": ["AAA", "CCC"], "weights": {"AAA": 0.6, "CCC": 0.4}, "paths": 100, "years": 10, "rules": ["fixed_real", "percent_of_nav"], "rate": 0.04, "target_ruin": 0.1}))
    rows = {x["rule"]: x for x in r["rows"]}
    assert set(rows) == {"fixed_real", "percent_of_nav"} and rows["percent_of_nav"]["ruin_probability"] == 0.0                         # a share of the value never runs out
    assert 0.0 < r["sustainable_rate"] <= 0.15 and r["target_ruin"] == 0.1 and r["years"] == 10 and r["paths"] == 100 and r["days"] > 1000
    band = r["bands"]["fixed_real"]
    assert band["quantiles"] == [0.05, 0.25, 0.5, 0.75, 0.95] and len(band["wealth"]) == 5 and len(band["wealth"][0]) == 11 and len(band["spending"][0]) == 10
    assert all(a <= b + 1e-9 for a, b in zip(band["wealth"][1], band["wealth"][3]))


def test_every_spending_rule_faces_the_same_futures_whatever_the_other_rules_are(lab):
    body = {"action": "spending", "tickers": ["AAA", "CCC"], "weights": {"AAA": 0.6, "CCC": 0.4}, "paths": 100, "years": 10, "seed": 3}
    alone = lab.cash_lab({**body, "rules": ["guardrails"]})["rows"][0]
    together = {x["rule"]: x for x in lab.cash_lab({**body, "rules": ["fixed_real", "endowment", "guardrails"]})["rows"]}["guardrails"]
    assert alone == together                                           # the same draw, so the same numbers


def test_a_redemption_is_priced_by_policy_and_a_cash_buffer_sells_less(lab):
    r = _json_safe(lab.cash_lab({"action": "redeem", "tickers": ["AAA", "CCC", "DDD"], "weights": {"AAA": 0.5, "CCC": 0.3, "DDD": 0.2}, "aum": 1e9, "redemption": 0.1, "cash": 0.05}))
    rows = {x["policy"]: x for x in r["rows"]}
    assert list(rows) == ["pro_rata", "liquid", "cash"] and rows["pro_rata"]["raised"] == pytest.approx(1e8) and rows["cash"]["raised"] == pytest.approx(5e7)
    assert rows["cash"]["cost_bps_of_fund"] < rows["pro_rata"]["cost_bps_of_fund"] and rows["liquid"]["assets_sold"] == 1 and rows["pro_rata"]["drift_after"] < rows["liquid"]["drift_after"]
    assert "median of the last 60 days" in r["note"] and r["inputs"] == {"aum": 1e9, "redemption": 0.1, "cash": 0.05, "participation": 0.1}


def test_without_volume_in_the_data_the_redemption_says_what_it_assumed(bare):
    r = bare.cash_lab({"action": "redeem", "tickers": ["AAA", "CCC"], "weights": {"AAA": 0.5, "CCC": 0.5}})
    assert "no volume" in r["note"] and r["rows"]


def test_the_liability_plan_compares_no_hedge_a_glide_path_and_a_full_hedge(lab):
    r = _json_safe(lab.cash_lab({"action": "ldi", "tickers": ["AAA", "BBB", "CCC"], "hedge": ["ccc"], "seeking": ["AAA", "BBB"], "funding_ratio": 0.9, "liability_years": 20}))
    assert [x["plan"] for x in r["rows"]] == ["unhedged", "glide path", "fully hedged"] and r["hedge"] == ["CCC"] and r["seeking"] == ["AAA", "BBB"]
    assert set(r["funding_ratio"]) == set(r["hedge_weight"]) == {"unhedged", "glide path", "fully hedged"} and len(r["funding_ratio"]["unhedged"]) == len(r["dates"])
    assert all(x["funding_ratio_start"] == pytest.approx(0.9) for x in r["rows"]) and r["inputs"] == {"funding_ratio": 0.9, "liability_years": 20, "payment": 100.0}
    assert set(r["hedge_weight"]["unhedged"]) <= {0.0, None} and set(r["hedge_weight"]["fully hedged"]) <= {1.0, None}


def test_the_liability_plan_needs_interest_rates_in_the_data(bare):
    with pytest.raises(ApiError, match="10-year Treasury yield"):
        bare.cash_lab({"action": "ldi", "tickers": ["AAA", "CCC"], "hedge": ["CCC"], "seeking": ["AAA"]})


@pytest.mark.parametrize("body, message", [
    ({"action": "nope"}, "action must be one of"),
    ({"action": "simulate", "policies": ["zzz"]}, "policies must be 1 to 5"),
    ({"action": "simulate", "policies": "pro_rata"}, "policies must be 1 to 5"),
    ({"action": "simulate", "deposit": "abc"}, "deposit must be a number"),
    ({"action": "simulate", "deposit": -5}, "deposit must be between"),
    ({"action": "simulate", "rebalance": "hourly"}, "rebalance must be one of"),
    ({"action": "simulate", "weights": {"AAA": -1}}, "weights must be non-negative"),
    ({"action": "simulate", "weights": {"ZZZ": 1}, "tickers": ["AAA", "CCC"]}, "name tickers that are in the list"),
    ({"action": "simulate", "weights": [1, 2]}, "weights must be a mapping"),
    ({"action": "simulate", "weights": {"AAA": "x"}}, "weights must be numbers"),
    ({"action": "spending", "rules": ["zzz"]}, "rules must be 1 to 4"),
    ({"action": "spending", "years": 100}, "years must be between"),
    ({"action": "spending", "paths": 5000}, "paths must be between"),
    ({"action": "redeem", "redemption": 5}, "redemption must be between"),
    ({"action": "redeem", "participation": 0}, "participation must be between"),
    ({"action": "ldi"}, "name at least one bond fund"),
    ({"action": "ldi", "hedge": 5, "seeking": ["AAA"]}, "hedge and seeking are lists"),
    ({"action": "ldi", "hedge": ["ZZZ"], "seeking": ["AAA"]}, "name at least one bond fund"),
    ({"action": "simulate", "tickers": ["BADX", "AAA"]}, "BADX"),
    ({"action": "simulate", "tickers": ["NEWTINY", "AAA"]}, "too little history"),
    ({"action": "simulate", "tickers": ["not a ticker!"]}, "not a valid ticker"),
])
def test_a_cash_request_that_cannot_run_is_refused_with_the_reason(lab, body, message):
    with pytest.raises(ApiError, match=message):
        lab.cash_lab(body)


# ------------------------------------------------------------------------------------------------------ over HTTP
@pytest.fixture()
def running(lab):
    running = server.start(lab, "127.0.0.1", 0)
    yield running
    running.stop()


def test_both_endpoints_need_the_token_and_answer_json_or_a_reason(running):
    for path in ("/api/exec", "/api/cash"):
        assert _http(running, path, body={}, token=False)[0] == 403
        assert _http(running, path, raw=b"x=1", headers={"Content-Type": "text/plain"})[0] == 415
        assert _http(running, path, raw=b"{not json")[0] == 400
    status, body, headers = _http(running, "/api/exec", body={"action": "frontier"})
    assert status == 200 and "application/json" in headers["Content-Type"] and json.loads(body)["frontier"]
    status, body, _ = _http(running, "/api/exec", body={"action": "run", "algos": ["bogus"]})
    assert status == 400 and "unknown algorithm" in json.loads(body)["error"]
    status, body, _ = _http(running, "/api/cash", body={"action": "simulate", "tickers": ["AAA", "CCC"], "weights": {"AAA": 1, "CCC": 1}, "policies": ["pro_rata"]})
    assert status == 200 and json.loads(body)["rows"][0]["policy"] == "pro_rata"
    status, body, _ = _http(running, "/api/cash", body={"action": "spending", "rules": ["zzz"]})
    assert status == 400 and "rules must be" in json.loads(body)["error"]
