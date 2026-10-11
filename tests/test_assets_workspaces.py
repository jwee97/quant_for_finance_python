"""The options, futures and crypto workspaces: provenance, closed forms against the numbers they report, and the identities each dashboard must satisfy."""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assets import common, crypto_analytics as ca, futures_analytics as fa, options_analytics as oa
from src.assets.futures import carry as futures_carry, term_structure
from src.utils.config import load_config

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------------------------------- common
def test_provenance_labels_combine_and_refuse_unknown_sources():
    assert common.overall_source({"a": "platform", "b": "platform"}) == "platform"
    assert common.overall_source({"a": "platform", "b": "simulated"}) == "mixed"
    assert common.overall_source({"a": "model"}) == "simulated"
    with pytest.raises(ValueError):
        common.overall_source({"a": "invented"})
    env = common.envelope("x", {"rows": [1]}, {"p": "simulated"}, ["careful"], "2024-01-05")
    assert env["source"] == "simulated" and env["warnings"] == ["careful"] and env["as_of"] == "2024-01-05" and env["rows"] == [1]
    with pytest.raises(ValueError):                              # data may not overwrite the provenance fields
        common.envelope("x", {"source": "platform"}, {"p": "platform"})


def test_rank_percentile_and_liquidity_helpers():
    h = np.arange(1.0, 101.0)
    assert common.percentile_rank(h, 50) == pytest.approx(0.5) and common.range_rank(h, 50.5) == pytest.approx(0.5)
    assert common.range_rank(h, 1e9) == 1.0 and math.isnan(common.range_rank([1.0, 1.0], 1.0))
    assert common.liquidity_score(0.5, 1e9) > common.liquidity_score(5.0, 1e9) > common.liquidity_score(50.0, 1e9)
    assert common.liquidity_score(1.0, 1e9) > common.liquidity_score(1.0, 1e6)
    assert common.nice_step(0.6) == 1.0 and common.nice_step(1.9) == 2.0 and common.nice_step(3.0) == 5.0 and common.nice_step(0.12) == 0.2
    z = pd.Series(np.r_[np.zeros(89), 10.0]) + np.random.default_rng(0).normal(0, 0.01, 90)
    assert common.last_zscore(z, 90) > 5


# --------------------------------------------------------------------------------------------------------------------- options
SPOT, VOL, LONG_RUN, RATE, DIV = 500.0, 0.16, 0.20, 0.04, 0.013
AS_OF = pd.Timestamp("2024-06-03")


@pytest.fixture(scope="module")
def chain():
    return oa.prepare_chain(oa.build_model_chain(SPOT, AS_OF, VOL, LONG_RUN, RATE, DIV, 0.15, "TST"))


def test_the_model_chain_is_reproducible_and_priced_off_the_model_surface(chain):
    again = oa.build_model_chain(SPOT, AS_OF, VOL, LONG_RUN, RATE, DIV, 0.15, "TST")
    assert np.allclose(again["mid"].to_numpy(), oa.prepare_chain(again)["mid"].to_numpy()) and again["mid"].equals(chain["mid"])
    for row in oa.term_structure(chain):
        if row["dte"] >= 14:
            assert row["atm_iv"] == pytest.approx(oa.model_atm_iv(VOL, LONG_RUN, row["dte"], 0.15), rel=0.02)
    assert (chain["bid"] <= chain["ask"]).all() and (chain["volume"] >= 0).all() and chain["date"].nunique() == 1


def test_term_structure_has_no_calendar_arbitrage_and_a_sane_slope(chain):
    term = oa.term_structure(chain)
    total_var = [t["atm_iv"] ** 2 * t["dte"] / 365.0 for t in term]
    assert all(b >= a - 1e-9 for a, b in zip(total_var, total_var[1:]))
    tm = oa.term_metrics(term)
    assert tm["iv_90d"] > tm["iv_30d"] and not tm["inverted"]           # low current vol against a higher long-run level: upward-sloping
    assert oa.interpolate_iv(term, 1e4) == term[-1]["atm_iv"]
    low = oa.build_model_chain(SPOT, AS_OF, 0.40, 0.20, RATE, DIV, 0.15, "TST")
    assert oa.term_metrics(oa.term_structure(oa.prepare_chain(low)))["inverted"]    # stress: the front above the back


def test_skew_metrics_have_the_equity_signs_and_deltas_are_consistent(chain):
    expiry = oa.expiry_list(chain)[6]["expiry"]
    sk = oa.skew_metrics(chain, expiry)
    assert sk["rr_25d"] < 0 and sk["rr_10d"] < sk["rr_25d"] < 0                       # puts dearer than calls, more so further out
    assert sk["put_25d_iv"] > sk["atm_iv"] > sk["call_25d_iv"] and sk["bf_25d"] > 0 and sk["skew_slope"] < 0
    view = oa.chain_view(chain, expiry, window=0.2)
    mid_row = next(r for r in view["rows"] if r["atm"])
    T = view["dte"] / 365.0
    for r in view["rows"]:                                                           # call delta - put delta = e^{-qT} at the same strike and (nearly) the same volatility
        if r["call"] and r["put"] and r["call"]["delta"] is not None and r["put"]["delta"] is not None:
            assert r["call"]["delta"] - r["put"]["delta"] == pytest.approx(math.exp(-DIV * T), abs=0.03)
    assert abs(mid_row["strike"] / view["forward"] - 1.0) < 0.01


def test_chain_view_totals_match_its_rows_and_flags_stale_quotes(chain):
    view = oa.chain_view(chain, None, window=0.12)
    calls = [r["call"] for r in view["rows"] if r["call"]]
    assert view["totals"]["call_volume"] == pytest.approx(sum(c["volume"] for c in calls))
    assert view["totals"]["put_call_volume"] == pytest.approx(view["totals"]["put_volume"] / view["totals"]["call_volume"])
    aged = chain.copy()
    aged["quote_age_min"] = 60.0
    assert oa.quote_staleness(aged).all() and oa.chain_view(aged, view["expiry"])["stale_quotes"] == view["quotes"]
    with pytest.raises(ValueError, match="not on the chain"):
        oa.chain_view(chain, "2031-01-01")


def test_the_surface_grid_is_not_extrapolated(chain):
    grid = oa.surface_grid(chain)
    assert len(grid["iv"]) == len(grid["expiries"]) > 5 and all(len(r) == len(grid["moneyness"]) for r in grid["iv"])
    assert any(v is None for r in grid["iv"] for v in r)                              # the wings of short expiries are not quoted, so they are empty
    centre = len(grid["moneyness"]) // 2
    assert all(r[centre] is not None for r in grid["iv"])


def test_overview_reports_the_cards_and_labels_the_rank_as_model_based(chain):
    prices = pd.Series(500.0 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0004, 0.01, 600))), index=pd.bdate_range(end=AS_OF, periods=600))
    ctx = oa.underlying_context("TST", prices)
    ov = oa.overview(chain, ctx, 0.15, None, True)
    assert 0.0 <= ov["iv_rank"] <= 1.0 and 0.0 <= ov["iv_percentile"] <= 1.0 and "model" in ov["iv_rank_basis"]
    assert ov["snapshot"]["asset_class"] == "option" and ov["snapshot"]["source"] == "simulated"
    assert ov["nearest_dte"] == min(e["dte"] for e in oa.expiry_list(chain) if e["dte"] >= 1)
    file_like = oa.overview(chain, ctx, 0.15, None, False)
    assert np.isnan(file_like["iv_rank"]) and file_like["snapshot"]["source"] == "user_file"
    with pytest.raises(ValueError, match="130 days"):
        oa.underlying_context("TST", prices.head(50))


def test_a_vendor_chain_gets_its_volatility_from_the_mid():
    from src.derivatives.pricing import bsm_price

    rows = []
    for right in ("C", "P"):
        for k in (90.0, 100.0, 110.0):
            rows.append({"date": "2024-06-03", "expiry": "2024-07-19", "strike": k, "right": right, "bid": 0, "ask": 0, "underlying": 100.0, "rate": 0.03, "dividend": 0.0})
    df = pd.DataFrame(rows)
    T = 46 / 365.0
    px = [float(bsm_price(100.0, r.strike, T, 0.03, 0.0, 0.25, r.right == "C")) for r in df.itertuples()]
    df["bid"], df["ask"] = np.array(px) - 0.005, np.array(px) + 0.005
    from src.derivatives.schema import normalise_chain

    got = oa.prepare_chain(normalise_chain(df, "V"))
    assert np.allclose(got["iv"], 0.25, atol=2e-3)


def test_portfolio_greeks_add_up_and_full_revaluation_matches_the_taylor_expansion():
    pos = [{"expiry": "2024-07-19", "strike": 500.0, "right": "C", "quantity": 2, "iv": 0.18}, {"expiry": "2024-07-19", "strike": 500.0, "right": "P", "quantity": 2, "iv": 0.18},
           {"expiry": "2024-09-20", "strike": 540.0, "right": "C", "quantity": -3, "iv": 0.17}, {"type": "stock", "quantity": 50}]
    risk = oa.portfolio_risk(pos, SPOT, RATE, DIV, AS_OF)
    t = risk["totals"]
    assert t["delta_shares"] == pytest.approx(sum(p["delta"] for p in risk["positions"])) and t["vega_per_point"] == pytest.approx(sum(p["vega"] for p in risk["positions"]))
    zero = next(r for r in risk["stress"] if r["spot_shock"] == 0.0 and r["vol_shock"] == 0.0)
    assert zero["pnl"] == pytest.approx(0.0, abs=1e-6)
    small = next(r for r in risk["stress"] if r["spot_shock"] == 0.02 and r["vol_shock"] == 0.0)
    assert abs(small["residual"]) < 0.15 * abs(small["pnl"]) + 5.0                    # delta and gamma explain most of a 2% move
    assert len(risk["stress"]) == 15 and risk["worst"]["pnl"] == min(r["pnl"] for r in risk["stress"])
    short_gamma = oa.portfolio_risk([{"expiry": "2024-07-19", "strike": 500.0, "right": "C", "quantity": -4, "iv": 0.18}, {"expiry": "2024-07-19", "strike": 500.0, "right": "P", "quantity": -4, "iv": 0.18}],
                                    SPOT, RATE, DIV, AS_OF)
    assert short_gamma["totals"]["gamma"] < 0 and short_gamma["totals"]["vega_per_point"] < 0 and short_gamma["totals"]["theta_per_day"] > 0
    assert any("short gamma" in f for f in short_gamma["risk_flags"])
    with pytest.raises(ValueError, match="implied volatility"):
        oa.portfolio_risk([{"expiry": "2024-07-19", "strike": 500.0, "right": "C", "quantity": 1}], SPOT, RATE, DIV, AS_OF)


def _analyse(name, **kw):
    legs = oa.strategy_legs(name, SPOT, 503.0, 1.0, dte=30, **kw)
    return legs, oa.strategy_analysis(legs, SPOT, 503.0, 0.18, RATE, DIV, skew_slope=-0.5, commission=0.65)


def test_strategy_payoffs_have_the_textbook_extremes_and_breakevens():
    legs, long_call = _analyse("long_call")
    assert long_call["unlimited_profit"] and not long_call["unlimited_loss"] and long_call["max_loss"] == pytest.approx(-long_call["cost"])
    k = legs[0]["strike"]
    assert long_call["breakevens"][0] == pytest.approx(k + (long_call["cost"]) / 100.0, abs=0.5)
    _, straddle = _analyse("long_straddle")
    lo, hi = straddle["breakevens"]
    assert lo < SPOT < hi and (hi - lo) / 2.0 == pytest.approx(straddle["cost"] / 100.0, rel=0.05) and straddle["greeks"]["gamma"] > 0 and straddle["greeks"]["vega_per_point"] > 0
    legs, condor = _analyse("iron_condor")
    width = (legs[1]["strike"] - legs[0]["strike"]) * 100.0
    assert condor["max_profit"] == pytest.approx(condor["net_credit"] + 0.0, abs=1.0) and condor["max_loss"] == pytest.approx(-(width - condor["net_credit"]), abs=2.0)
    assert not condor["unlimited_loss"] and condor["margin"]["estimate"] == pytest.approx(width - condor["net_credit"], abs=2.0) and condor["prob_profit"] > 0.5
    _, short_straddle = _analyse("short_straddle")
    assert short_straddle["unlimited_loss"] and short_straddle["max_loss"] is None and not short_straddle["margin"]["defined_risk"] and short_straddle["margin"]["estimate"] > 5000
    _, calendar = _analyse("calendar_spread")
    assert not calendar["unlimited_loss"] and calendar["max_loss"] == pytest.approx(-calendar["cost"], rel=0.1)
    _, covered = _analyse("covered_call")
    assert covered["greeks"]["delta_shares"] < 100 and covered["margin"]["estimate"] == pytest.approx(0.5 * SPOT * 100.0, rel=0.02) and not covered["unlimited_loss"]
    _, ratio = _analyse("ratio_spread")
    assert ratio["unlimited_loss"] and not ratio["margin"]["defined_risk"]


def test_strategy_value_today_starts_at_the_cost_of_trading_and_expected_value_follows_the_view():
    legs, a = _analyse("long_call")
    spots, now = np.array(a["curve"]["spot"]), np.array(a["curve"]["now"])
    assert float(np.interp(SPOT, spots, now)) == pytest.approx(-0.65, abs=0.1 * a["cost"] * 0.05 + 3.0)         # marked at the model price, so only the commission is lost
    bull = oa.strategy_analysis(legs, SPOT, 503.0, 0.18, RATE, DIV, skew_slope=-0.5, view_drift=0.6)
    bear = oa.strategy_analysis(legs, SPOT, 503.0, 0.18, RATE, DIV, skew_slope=-0.5, view_drift=-0.6)
    assert bull["expected_value"] > a["expected_value"] > bear["expected_value"] and bull["prob_profit"] > bear["prob_profit"]
    worst = oa.strategy_analysis(legs, SPOT, 503.0, 0.18, RATE, DIV, skew_slope=-0.5, fill="worst", spread_pct=0.10)
    assert worst["cost"] > a["cost"]
    with pytest.raises(ValueError, match="unknown strategy"):
        oa.strategy_legs("jade_lizard", SPOT, 503.0, 1.0)
    with pytest.raises(ValueError, match="horizon"):
        oa.strategy_analysis(legs, SPOT, 503.0, 0.18, horizon_days=400)


def test_every_listed_strategy_builds_and_prices():
    for name in oa.STRATEGY_INFO:
        legs, a = _analyse(name)
        assert a["legs"] and np.isfinite(a["cost"]) and 0.0 <= a["prob_profit"] <= 1.0 and a["curve"]["spot"][0] == 0.0


def test_the_strategy_backtest_runs_on_the_simulated_market_and_says_so():
    out = oa.run_strategy_backtest("short_put", None, n_days=60, seed=1)
    assert out["strategy"] == "short_put" and len(out["equity"]) > 20 and "sharpe" in out["summary"] and out["cvar_95"] >= out["var_95"] >= 0
    assert any("simulated" in n for n in out["notes"])
    with pytest.raises(ValueError, match="unknown strategy"):
        oa.run_strategy_backtest("naked_hope")
    with pytest.raises(ValueError, match="between 60 and 756"):
        oa.run_strategy_backtest("short_put", None, n_days=10)


# --------------------------------------------------------------------------------------------------------------------- futures
@pytest.fixture(scope="module")
def cl():
    return fa.simulated_curve("CL", "2024-06-03", 300)


def test_the_curve_snapshot_matches_the_term_structure_panel(cl):
    snap = fa.curve_snapshot(cl)
    ts = term_structure(cl.long[["date", "contract", "expiry", "price"]], 4)
    assert snap["annualised_carry"] == pytest.approx(float(futures_carry(ts["price"], ts["ttm"]).iloc[-1]))
    prices = [r["price"] for r in snap["rows"]]
    assert snap["front_price"] == prices[0] and [r["rank"] for r in snap["rows"]] == list(range(1, len(prices) + 1))
    ry = [r["roll_yield"] for r in snap["rows"][:-1]]
    assert ry[0] == pytest.approx(math.log(prices[0] / prices[1]) / ((snap["rows"][1]["dte"] - snap["rows"][0]["dte"]) / 365.0), rel=0.05)
    assert snap["rows"][-1]["roll_yield"] is None and snap["notional"] == pytest.approx(prices[0] * 1000.0)


def test_curve_state_classification():
    assert fa.curve_state(np.array([0.08, 0.05]), -0.05) == "backwardation"
    assert fa.curve_state(np.array([-0.08, -0.05]), 0.05) == "contango"
    assert fa.curve_state(np.array([0.08, -0.05]), 0.05) == "mixed" and fa.curve_state(np.array([0.001]), 0.0) == "mixed"


def test_open_interest_moves_to_the_next_contract_before_expiry(cl):
    roll = fa.roll_analysis(cl, 5)
    assert roll["current_contract"] != roll["next_contract"] and roll["days_to_expiry"] >= 0
    for past in roll["past_rolls"]:
        gap = (pd.Timestamp(past["expiry"]) - pd.Timestamp(past["oi_crossover"])).days
        assert abs(gap - fa.ROOTS["CL"].roll_oi_days) <= 4
    assert roll["calendar_roll_date"] == str((pd.Timestamp(roll["front_expiry"]) - pd.Timedelta(days=5)).date())
    spec = fa.ROOTS["CL"]
    assert roll["roll_cost_dollars"] == pytest.approx(roll["calendar_spread_points"] * spec.multiplier + spec.spread_ticks * spec.tick_value + 2.0 * spec.commission)
    assert min(pd.Timestamp(roll["suggested_roll_date"]), pd.Timestamp(roll["calendar_roll_date"])) == pd.Timestamp(roll["suggested_roll_date"])
    h = roll["history"]
    assert len(h["dates"]) == len(h["front"]) == len(h["next"]) == len(h["spread"]) and h["spread"][-1] == pytest.approx(h["next"][-1] - h["front"][-1], abs=1e-3)


def test_the_carry_and_basis_dashboards_agree_with_the_curve(cl):
    c = fa.carry_dashboard(cl)
    assert c["carry"] == pytest.approx(fa.curve_snapshot(cl, 6)["annualised_carry"]) and 0.0 <= c["carry_percentile_1y"] <= 1.0
    assert c["carry_to_vol"] == pytest.approx(c["carry"] / c["excess_return_vol"])
    b = fa.basis_dashboard(cl)
    assert b["has_spot"] and b["basis_points"] == pytest.approx(b["basis"] * b["spot"], rel=1e-6) and b["calendar_series"]["values"]


def test_margin_dashboard_arithmetic_and_warnings(cl):
    curves = {"CL": cl, "ES": fa.simulated_curve("ES", "2024-06-03", 300)}
    front = fa.curve_snapshot(cl)["front_price"]
    m = fa.margin_dashboard([{"root": "CL", "quantity": 4}, {"root": "ES", "quantity": -1}], curves, 40_000.0)
    cl_row = next(r for r in m["positions"] if r["root"] == "CL")
    assert cl_row["notional"] == pytest.approx(4 * front * 1000.0) and m["initial_margin"] == pytest.approx(4 * 6500.0 + 12500.0)
    assert m["margin_usage"] == pytest.approx(m["initial_margin"] / 40_000.0) and m["leverage"] == pytest.approx(m["gross_notional"] / 40_000.0)
    assert any("margin use" in f for f in m["risk_flags"]) and next(r for r in m["positions"] if r["root"] == "ES")["notional"] < 0
    safe = fa.margin_dashboard([{"root": "CL", "quantity": 1}], curves, 1_000_000.0)
    assert not any("margin use" in f or "maintenance" in f for f in safe["risk_flags"])
    near = fa.margin_dashboard([{"root": "CL", "quantity": 1}], curves, 1e6, near_expiry_days=400)
    assert any("expires in" in f for f in near["risk_flags"])
    with pytest.raises(ValueError, match="no curve"):
        fa.margin_dashboard([{"root": "ZZ", "quantity": 1}], curves, 1e5)
    with pytest.raises(ValueError, match="not on the CL curve"):
        fa.margin_dashboard([{"root": "CL", "contract": "Z99", "quantity": 1}], curves, 1e5)


def test_a_curve_file_round_trips_and_is_labelled_as_yours(tmp_path, cl):
    path = tmp_path / "CL.csv"
    cl.long[["date", "contract", "expiry", "price"]].to_csv(path, index=False)
    loaded = fa.load_curve_file(path, "CL")
    assert loaded.source == "user_file" and loaded.spot is None and any("no open interest" in n for n in loaded.notes)
    assert fa.curve_snapshot(loaded)["annualised_carry"] == pytest.approx(fa.curve_snapshot(cl)["annualised_carry"])
    assert fa.roll_analysis(loaded)["oi_crossover"] is None                       # no open interest in the file: nothing to cross
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-02"], "price": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="needs the columns"):
        fa.load_curve_file(bad, "CL")
    unknown = fa.load_curve_file(path, "XX")
    assert unknown.spec.multiplier == 1.0 and any("unknown" in n for n in unknown.notes)


def test_simulated_curves_are_deterministic_and_the_unknown_root_is_refused():
    a, b = fa.simulated_curve("GC", "2024-06-03", 200), fa.simulated_curve("GC", "2024-06-03", 200)
    assert a is b or a.long.equals(b.long)
    assert fa.simulated_curve("GC", "2024-06-04", 200).long["price"].iloc[0] != a.long["price"].iloc[0]
    with pytest.raises(ValueError, match="unknown futures root"):
        fa.simulated_curve("ZZ")
    snap = fa.futures_snapshot(a)
    assert snap.asset_class == "future" and snap.source == "simulated" and 0 <= snap.liquidity_score <= 100


# --------------------------------------------------------------------------------------------------------------------- crypto
def _series(n=500, seed=0, funding_level=0.0002, source="platform"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n)
    spot = pd.Series(30000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n))), index=idx)
    basis = 0.0005 + 0.0003 * rng.standard_normal(n)
    funding = pd.Series(funding_level + 0.0001 * rng.standard_normal(n), index=idx)
    return ca.CryptoSeries("BTC", spot, spot * (1.0 + basis), funding, source)


def test_funding_by_venue_keeps_deribit_real_and_marks_the_rest_simulated():
    s = _series()
    d, src = ca.venue_funding(s, "DERIBIT")
    assert d.equals(s.funding) and src == "platform"
    b, src_b = ca.venue_funding(s, "BINANCE")
    assert src_b == "simulated" and not b.equals(s.funding) and b.corr(s.funding) > 0.7
    assert ca.venue_funding(s, "BINANCE")[0].equals(b)                           # deterministic
    dash = ca.funding_dashboard({"BTC": s})
    assert len(dash["rows"]) == 4 and {r["venue"] for r in dash["rows"]} == set(ca.VENUE_ORDER)
    deribit = next(r for r in dash["rows"] if r["venue"] == "DERIBIT")
    assert deribit["annualised"] == pytest.approx(float(s.funding.iloc[-1]) * 365.0) and deribit["avg_7d"] == pytest.approx(float(s.funding.tail(7).mean()) * 365.0)
    assert all(r["source"] == "simulated" for r in dash["rows"] if r["venue"] != "DERIBIT")
    with pytest.raises(ValueError, match="unknown venue"):
        ca.venue_funding(s, "NOWHERE")


def test_simulated_coins_come_from_btc_and_say_so():
    btc = _series(source="platform")
    sol = ca.simulated_series("SOL", btc)
    assert sol.source == "simulated" and any("SIMULATED" in n for n in sol.notes) and sol.spot.index.equals(btc.spot.index)
    r_btc, r_sol = np.log(btc.spot).diff().dropna(), np.log(sol.spot).diff().dropna()
    assert 0.3 < r_btc.corr(r_sol) < 0.95 and ca.simulated_series("SOL", btc).spot.equals(sol.spot)
    with pytest.raises(ValueError, match="no simulation"):
        ca.simulated_series("DOGE", btc)


def test_overview_cards_and_the_states_they_report():
    s = _series()
    ov = ca.overview(s, pd.Series(1e11 * np.exp(0.0008 * np.arange(500)), index=s.spot.index), ca.order_book_summary("BTC", "BINANCE", 30000.0))
    cards = {c["name"]: c for c in ov["cards"]}
    assert set(cards) == {"trend", "funding", "momentum", "volatility", "liquidity", "onchain"} and cards["liquidity"]["source"] == "simulated"
    assert cards["onchain"]["state"] == "inflow" and cards["funding"]["value"] == pytest.approx(float(s.funding.tail(7).mean()) * 365.0)
    assert ov["open_interest"] is None and ov["exchange_net_flow"] is None and any("not estimated" in n for n in ov["notes"])
    no_funding = ca.CryptoSeries("BTC", s.spot, None, None, "platform")
    assert {c["name"]: c for c in ca.overview(no_funding)["cards"]}["funding"]["state"] == "unavailable"
    with pytest.raises(ValueError, match="at least 250"):
        ca.overview(_series(100))


def test_basis_dashboard_break_even_is_fees_over_funding():
    s = _series(funding_level=0.0003)
    b = ca.basis_dashboard(s, fee_bps=5.0)
    assert b["breakeven_days"] == pytest.approx(4 * 5.0 / 1e4 / float(s.funding.tail(30).mean()))
    assert b["funding_annualised_30d"] == pytest.approx(float(s.funding.tail(30).mean()) * 365.0)
    sp = ca.spot_perp_view(s, 200)
    assert len(sp["dates"]) == 200 and sp["stats"]["return_correlation"] > 0.99
    with pytest.raises(ValueError, match="perpetual"):
        ca.spot_perp_view(ca.CryptoSeries("BTC", s.spot, None, None, "platform"))


def test_cross_exchange_edges_net_off_both_fees_and_slippage():
    s = _series()
    out = ca.cross_exchange(s, slippage_bps=1.0, notional=2e6)
    quotes = {q["venue"]: q for q in out["quotes"]}
    assert quotes["DERIBIT"]["source"].startswith("platform") and quotes["BINANCE"]["source"] == "simulated"
    for r in out["pairs"]:
        a, b = quotes[r["buy"]], quotes[r["sell"]]
        assert r["gross_bps"] == pytest.approx((b["bid"] / a["ask"] - 1.0) * 1e4)
        assert r["net_bps"] == pytest.approx(r["gross_bps"] - a["taker_bps"] - b["taker_bps"] - 2.0) and r["net_dollars"] == pytest.approx(r["net_bps"] / 1e4 * 2e6)
    assert [r["net_bps"] for r in out["pairs"]] == sorted((r["net_bps"] for r in out["pairs"]), reverse=True) and len(out["pairs"]) == 12
    assert any("simulated" in f for f in out["risk_flags"])


def test_the_order_book_impact_grows_with_size_and_is_labelled():
    book = ca.order_book_summary("BTC", "BINANCE", 60000.0)
    impacts = [book["impact_bps"][k]["buy"] for k in book["impact_bps"]]
    assert impacts == sorted(impacts) and impacts[0] >= book["spread_bps"] / 2.0 and book["source"] == "simulated"
    assert book["depth"]["50bp"]["ask"] > book["depth"]["25bp"]["ask"] > book["depth"]["10bp"]["ask"]
    assert ca.order_book_summary("BTC", "DERIBIT", 60000.0)["depth"]["10bp"]["bid"] < book["depth"]["10bp"]["bid"]
    assert ca.order_book_summary("SOL", "BINANCE", 150.0)["spread_bps"] > book["spread_bps"]
    with pytest.raises(ValueError, match="unknown venue"):
        ca.order_book_summary("BTC", "NOWHERE", 1.0)


def test_a_user_crypto_file_replaces_the_simulation(tmp_path):
    s = _series(300)
    path = tmp_path / "SOL.csv"
    pd.DataFrame({"time": s.spot.index, "spot": s.spot.values, "perp": s.perp.values, "funding_8h": (s.funding / 3.0).values}).to_csv(path, index=False)
    u = ca.user_series(path, "SOL")
    assert u.source == "user_file" and u.funding.iloc[-1] == pytest.approx(float(s.funding.iloc[-1]))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"time": ["2024-01-01"], "price": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="time and spot"):
        ca.user_series(bad, "SOL")


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "crypto" / "BTC_funding.csv.gz").exists(), reason="the Deribit files are not present")
def test_the_platform_data_loads_and_its_strategies_run_through_the_pipeline():
    from src.data.crypto import build_crypto_frames

    frames = build_crypto_frames(ROOT / "data" / "raw" / "crypto")
    btc = ca.platform_series(frames, "BTC")
    assert btc.source == "platform" and btc.perp is not None and btc.funding.notna().sum() > 1000
    config = load_config()
    out = ca.run_crypto_strategy(config, "funding_carry", ["BTC", "ETH"], "weekly", None, None, 5.0, 2.0)
    assert out["universe"] == ["BTC_CARRY", "ETH_CARRY"] and out["spec"]["execution"]["cost_bps"] == 7.0 and out["metrics"]["sharpe"] > 0 and out["source"] == "platform"
    assert len(out["dates"]) == len(out["equity"]) == len(out["drawdown"]) and min(out["drawdown"]) <= 0
    pricey = ca.run_crypto_strategy(config, "spot_momentum", ["BTC"], "weekly", 60, None, 50.0, 50.0)
    cheap = ca.run_crypto_strategy(config, "spot_momentum", ["BTC"], "weekly", 60, None, 0.0, 0.0)
    assert cheap["metrics"]["sharpe"] > pricey["metrics"]["sharpe"] and cheap["metrics"]["ann_cost_bps"] < pricey["metrics"]["ann_cost_bps"]
    with pytest.raises(ValueError, match="exchange net-flow"):
        ca.run_crypto_strategy(config, "exchange_flow")
    with pytest.raises(ValueError, match="platform's data"):
        ca.run_crypto_strategy(config, "spot_momentum", ["SOL"])
    with pytest.raises(ValueError, match="unknown crypto strategy"):
        ca.run_crypto_strategy(config, "moon")


def test_futures_strategies_run_on_the_simulated_bundle_through_the_same_pipeline():
    from src.assets.strategy_runs import run_strategy

    bundle = fa.futures_bundle_for(["CL", "GC", "ES"], "2024-06-03")
    assert set(bundle.assets) == {"CL", "GC", "ES"} and {"CARRY_CL", "BASISMOM_GC"} <= set(bundle.macro.columns)
    out = run_strategy(bundle, load_config(), "carry_ts", {}, rebalance="weekly", cost_bps=3.0, label="carry", allocator="sleeves")
    assert out["n_days"] > 100 and out["universe"] == ["CL", "ES", "GC"] and out["spec"]["models"][0]["name"] == "carry_ts" and out["metrics"]["ann_turnover"] > 0
    free = run_strategy(bundle, load_config(), "tsmom", {}, rebalance="weekly", cost_bps=0.0, allocator="score_stack")
    costly = run_strategy(bundle, load_config(), "tsmom", {}, rebalance="weekly", cost_bps=100.0, allocator="score_stack")
    assert free["metrics"]["ann_cost_bps"] == 0.0 < costly["metrics"]["ann_cost_bps"] and free["metrics"]["sharpe"] > costly["metrics"]["sharpe"]
    with pytest.raises(ValueError, match="frequency"):
        run_strategy(bundle, load_config(), "carry_ts", {}, rebalance="hourly")
    with pytest.raises(ValueError, match="basis points"):
        run_strategy(bundle, load_config(), "carry_ts", {}, cost_bps=900.0)
    wrapped = fa.run_futures_strategy(load_config(), "carry_ts", ["CL", "GC", "ES"], "weekly", None, None, 2.0, "2024-06-03")
    assert wrapped["source"] == "simulated" and wrapped["roots"] == ["CL", "ES", "GC"] and wrapped["spec"]["allocation"] == {"allocator": "sleeves"}
    with pytest.raises(ValueError, match="at least two"):
        fa.run_futures_strategy(load_config(), "carry_xs", ["CL"])
    with pytest.raises(ValueError, match="commodity roots"):
        fa.run_futures_strategy(load_config(), "basis_momentum", ["CL", "ES"])
    with pytest.raises(ValueError, match="unknown futures strategy"):
        fa.run_futures_strategy(load_config(), "alchemy")
