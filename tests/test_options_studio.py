"""Options Strategy Studio: scanner, real-contract builder, scenarios, ranking and the margin of a backtest (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import options_analytics as oa
from src.assets import options_studio as st

AS_OF = pd.Timestamp("2025-01-02")


@pytest.fixture(scope="module")
def raw():
    return oa.build_model_chain(100.0, AS_OF, 0.2)


@pytest.fixture(scope="module")
def chain(raw):
    return oa.prepare_chain(raw)


def pick(chain, i=9, offset=0.0):
    e = sorted(chain["expiry"].unique())[i]
    g = chain[(chain["expiry"] == e) & (chain["right"] == "C")]
    k = float(g.iloc[(g["strike"] - (100.0 + offset)).abs().argsort()].iloc[0]["strike"])
    return pd.Timestamp(e), k


def row(chain, e, k, right="C"):
    return chain[(chain["expiry"] == e) & (chain["strike"] == k) & (chain["right"] == right)].iloc[0]


def test_a_call_spread_is_priced_at_the_touch_the_mid_and_the_limits_exactly(chain):
    e, k = pick(chain)
    k2 = k + 5.0
    long_, short_ = row(chain, e, k), row(chain, e, k2)
    legs = [{"type": "option", "expiry": str(e.date()), "strike": k, "right": "C", "quantity": 2, "limit": float(long_["ask"])}, {"type": "option", "expiry": str(e.date()), "strike": k2, "right": "C", "quantity": -2}]
    out = st.build_real(chain, legs, 100.0, commission=0.65)
    t = out["totals"]
    assert t["at_touch"] == pytest.approx(2 * 100 * (long_["ask"] - short_["bid"]) + 4 * 0.65)
    assert t["at_mid"] == pytest.approx(2 * 100 * (long_["mid"] - short_["mid"]) + 4 * 0.65)
    assert t["at_limits"] == pytest.approx(2 * 100 * (long_["ask"] - short_["mid"]) + 4 * 0.65) and t["spread_cost"] == pytest.approx(t["at_touch"] - t["at_mid"])
    assert out["legs"][0]["status"].startswith("marketable") and out["legs"][1]["status"].startswith("no limit")
    a = out["analysis"]
    assert a["cost"] == pytest.approx(t["at_limits"]) and a["max_profit"] == pytest.approx(2 * 100 * 5.0 - t["at_limits"], abs=1.0) and a["max_loss"] == pytest.approx(-t["at_limits"], abs=1.0)
    assert a["breakevens"][0] == pytest.approx(k + t["at_limits"] / 200.0, abs=0.1)


def test_limits_that_rest_are_flagged_and_unlisted_contracts_are_refused_with_the_nearest_strikes(chain):
    e, k = pick(chain)
    r = row(chain, e, k)
    below = [{"type": "option", "expiry": str(e.date()), "strike": k, "right": "C", "quantity": 1, "limit": float(r["bid"]) - 0.01}]
    out = st.build_real(chain, below, 100.0)
    assert "unlikely to fill" in out["legs"][0]["status"] and out["fill"]["legs_resting"] == 1 and "may not fill" in out["fill"]["reading"]
    inside = st.build_real(chain, [{**below[0], "limit": float((r["bid"] + r["ask"]) / 2)}], 100.0)
    assert "inside the spread" in inside["legs"][0]["status"]
    sell_low = st.build_real(chain, [{**below[0], "quantity": -1, "limit": float(r["bid"])}], 100.0)
    assert sell_low["legs"][0]["status"].startswith("marketable")
    with pytest.raises(ValueError, match="nearest listed strikes"):
        st.build_real(chain, [{**below[0], "strike": k + 0.123}], 100.0)
    with pytest.raises(ValueError, match="no C contracts are listed"):
        st.build_real(chain, [{**below[0], "expiry": "2031-01-01"}], 100.0)
    with pytest.raises(ValueError, match="zero"):
        st.build_real(chain, [{**below[0], "quantity": 0}], 100.0)


def test_scenario_pnl_splits_into_underlying_volatility_skew_time_costs_and_an_explicit_residual(chain):
    e, k = pick(chain, offset=0.0)
    built = st.build_real(chain, [{"type": "option", "expiry": str(e.date()), "strike": k, "right": "C", "quantity": 1}, {"type": "option", "expiry": str(e.date()), "strike": k + 5, "right": "C", "quantity": -1}], 100.0)
    lab = st.scenario_lab(built["analysis"]["legs"], 100.0, 0.04, 0.0, AS_OF, built["totals"]["at_limits"], built["totals"]["spread_cost"] + built["totals"]["commission"])
    centre = lab["heatmaps"]["spot_vol"]
    assert centre["pnl"][centre["vol"].index(0.0)][centre["spot"].index(0.0)] == pytest.approx(0.0, abs=1e-9)
    up = lab["heatmaps"]["spot_vol"]["pnl"][2]
    assert up[-1] > up[0], "a bull call spread gains when the spot rises"
    for sc in lab["scenarios"]:
        a = sc["attribution"]
        assert sum(a.values()) == pytest.approx(sc["pnl"], abs=1e-6), sc["name"]
        assert a["costs"] == pytest.approx(-(built["totals"]["spread_cost"] + built["totals"]["commission"]))
    quiet = next(s for s in lab["scenarios"] if s["name"].startswith("quiet"))
    assert abs(quiet["attribution"]["delta"]) < 1e-6 and quiet["attribution"]["theta"] != 0.0
    skew = next(s for s in lab["scenarios"] if s["name"].startswith("skew steepens"))
    assert skew["attribution"]["skew"] != 0.0 and skew["attribution"]["delta"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        st.scenario_lab([{"type": "stock", "quantity": 100}], 100.0, 0.04, 0.0, AS_OF)


def test_the_ranking_orders_by_payoff_if_right_shows_the_wrong_way_loss_and_excludes_with_reasons(chain):
    out = st.recommend(chain, 100.0, "bullish", 0.05, "long", 0.03, 30, max_loss=2000.0)
    assert out["ranked"] and all(r["delta_dollars"] > 0 and r["vega_per_point"] > 0 for r in out["ranked"])
    assert [r["pnl_if_right"] for r in out["ranked"]] == sorted((r["pnl_if_right"] for r in out["ranked"]), reverse=True)
    assert all(r["pnl_if_wrong"] < r["pnl_if_right"] for r in out["ranked"])
    reasons = " ".join(x["reason"] for x in out["excluded"])
    assert "loss is unlimited" in reasons and "net delta points against" in reasons
    short = st.recommend(chain, 100.0, "neutral", 0.03, "short", 0.03, 30)
    names = [r["strategy"] for r in short["ranked"]]
    assert names and all(r["vega_per_point"] <= 0 for r in short["ranked"]) and "long_straddle" not in names
    for bad in ({"direction": "up"}, {"vol_view": "x"}, {"move": 0.0001}):
        with pytest.raises(ValueError):
            st.recommend(chain, 100.0, **{"direction": "neutral", **bad})


def test_the_scanner_flags_a_planted_kink_with_its_quality_and_stale_quotes_are_not_tradable(raw):
    c = raw.copy()
    exps = sorted(c["expiry"].unique())
    m = c["expiry"] == exps[9]
    for col in ("bid", "ask"):
        c.loc[m, col] = c.loc[m, col] * 1.25
    c["mid"], c["iv"] = 0.5 * (c["bid"] + c["ask"]), np.nan
    out = st.scan(oa.prepare_chain(c), 100.0)
    kinds = {i["kind"] for i in out["items"]}
    kink = next(i for i in out["items"] if i["kind"] == "term_kink" and i["expiries"] == [str(pd.Timestamp(exps[9]).date())])
    assert kink["metric"] > 0.03 and kink["quality"]["tradable"] and "negative_forward_variance" in kinds and out["iv_history"]["state"] == "unavailable"
    assert out["realised"]["state"] == "unavailable"
    stale = raw.copy()
    stale.loc[stale["expiry"] == exps[9], "quote_age_min"] = 90.0
    p = oa.prepare_chain(stale)
    q = st.quality(p[p["expiry"] == exps[9]], oa.quote_staleness(p))
    assert not q["tradable"] and "stale" in q["reason"]
    clean = st.scan(oa.prepare_chain(raw), 100.0, prices=pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 400))), iv_history=pd.Series(np.linspace(0.15, 0.25, 40)))
    assert clean["iv_history"]["state"] == "ok" and 0 <= clean["iv_history"]["percentile"] <= 1 and clean["realised"]["state"] == "ok" and clean["realised"]["ratio"] > 0


def test_a_backtest_reports_its_margin_apart_from_its_pnl_and_labels_itself_synthetic():
    out = oa.run_strategy_backtest("short_put", n_days=120)
    m = out["margin"]
    assert m["available"] and m["peak"] > 0 and m["peak_to_capital"] == pytest.approx(m["peak"] / out["capital"]) and m["return_on_margin"] is not None and m["not_modelled"]
    assert out["mode"]["id"] == "synthetic_validation" and out["mode"]["historical"] is False and {l["id"] for l in out["mode"]["limitations"]} >= {"synthetic_market", "not_historical", "margin_not_charged"}
    spread = oa.run_strategy_backtest("put_spread", n_days=120)["margin"]
    assert spread["peak"] < m["peak"] * 5 and spread["peak"] > 0
    legs = [{"type": "option", "right": "P", "strike": 100.0, "quantity": -1.0}]
    assert oa.backtest_margin(legs, 105.0) == pytest.approx(oa.margin_estimate(legs, 105.0, 0.0)["naked_requirement"])
    hedged = [{"type": "option", "right": "P", "strike": 100.0, "quantity": -1.0}, {"type": "option", "right": "P", "strike": 95.0, "quantity": 1.0}]
    assert oa.backtest_margin(hedged, 105.0) == pytest.approx(500.0, abs=15.0)


def test_the_ranking_reports_how_stable_it_is_and_the_baselines(chain):
    out = st.recommend(chain, 100.0, "bullish", 0.05, "long", 0.03, 30, max_loss=2000.0)
    s = out["stability"]
    assert s["of"] == 4 and 0 <= s["top_unchanged_in"] <= 4 and s["reading"] and all(r["top"] for r in s["runs"])
    assert out["baselines"]["hold_100_shares"] == pytest.approx(100 * 100.0 * 0.05) and out["baselines"]["cash"] == 0.0
    assert st.recommend(chain, 100.0, "bullish", 0.05, "long", 0.03, 30, stability=False)["stability"] is None


def test_a_proposed_position_changes_the_book_by_exactly_its_own_greeks(chain):
    e, k = pick(chain)
    legs = st.build_real(chain, [{"type": "option", "expiry": str(e.date()), "strike": k, "right": "C", "quantity": 2}], 100.0)["analysis"]["legs"]
    book = [{"type": "stock", "quantity": 100.0}, {"type": "option", "right": "P", "strike": 95.0, "dte": 45.0, "quantity": -1.0, "iv": 0.22}]
    out = st.portfolio_impact(book, legs, 100.0, 0.04, 0.0, AS_OF)
    alone = st.portfolio_impact([], legs, 100.0, 0.04, 0.0, AS_OF)
    for key in ("delta_dollars", "vega_per_point", "theta_per_day", "gamma_pnl_1pct"):
        assert out["change"][key] == pytest.approx(alone["after"][key], rel=1e-9, abs=1e-9), key
    assert alone["hypothetical_book"] and not out["hypothetical_book"] and out["worst_stress"]["scenario"]
    with pytest.raises(ValueError):
        st.portfolio_impact(book, [], 100.0, 0.04, 0.0)
