"""Mathematical audit of the options and futures analytics: identities, boundaries, finite differences, convergence, arbitrage constraints, and what is observed against what is assumed."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.assets import options_analytics as oa
from src.assets import provenance as pv
from src.derivatives import greeks as G
from src.derivatives import iv as IV
from src.derivatives import pricing as P
from tests.test_options_chain_exact import flat_chain


# ------------------------------------------------------------------------------------------------------------------ pricing identities
@pytest.mark.parametrize("S,K,T,r,q,sig", [(100, 100, 1.0, 0.05, 0.0, 0.2), (100, 80, 0.5, 0.03, 0.02, 0.35), (50, 70, 2.0, 0.01, 0.0, 0.5), (100, 100, 1 / 365, 0.04, 0.01, 0.15)])
def test_put_call_parity_and_the_textbook_value(S, K, T, r, q, sig):
    c, p = float(P.bsm_price(S, K, T, r, q, sig, True)), float(P.bsm_price(S, K, T, r, q, sig, False))
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)
    assert abs(float(P.put_call_parity_gap(c, p, S, K, T, r, q))) < 1e-9
    assert float(P.bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, True)) == pytest.approx(10.450583572185565, abs=1e-9)


def test_black_76_is_black_scholes_on_the_forward_and_parity_holds_with_discounting():
    F, K, T, r, sig = 95.0, 100.0, 0.75, 0.04, 0.3
    c, p = float(P.black76_price(F, K, T, r, sig, True)), float(P.black76_price(F, K, T, r, sig, False))
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-10)
    S, q = F * math.exp(-(r - 0.0) * T), 0.0
    assert c == pytest.approx(float(P.bsm_price(S, K, T, r, q, sig, True)), abs=1e-9)


def test_boundaries_expiry_zero_volatility_and_monotonicity():
    assert float(P.bsm_price(100, 90, 1e-9, 0.03, 0.0, 0.2, True)) == pytest.approx(10.0, abs=1e-6)
    assert float(P.bsm_price(100, 110, 1e-9, 0.03, 0.0, 0.2, True)) == pytest.approx(0.0, abs=1e-6)
    S, K, T, r = 100.0, 100.0, 1.0, 0.03
    assert float(P.bsm_price(S, K, T, r, 0.0, 1e-6, True)) == pytest.approx(max(S - K * math.exp(-r * T), 0.0), abs=1e-4)
    assert float(P.bsm_price(S, K, T, r, 0.0, 6.0, True)) < S and float(P.bsm_price(S, K, T, r, 0.0, 6.0, True)) > S * 0.95, "a call is bounded by the underlying and approaches it as volatility explodes"
    vols = np.linspace(0.05, 1.0, 20)
    prices = [float(P.bsm_price(S, K, T, r, 0.0, v, True)) for v in vols]
    assert all(b > a for a, b in zip(prices, prices[1:])), "increasing in volatility"
    strikes = np.linspace(60, 140, 41)
    calls = np.array([float(P.bsm_price(S, k, T, r, 0.0, 0.25, True)) for k in strikes])
    assert (np.diff(calls) < 0).all() and (np.diff(calls, 2) > 0).all(), "decreasing and convex in strike: the butterfly condition"


# ------------------------------------------------------------------------------------------------------------------ Greeks
@pytest.mark.parametrize("call", [True, False])
def test_analytic_greeks_match_finite_differences_and_their_units(call):
    S, K, T, r, q, sig = 100.0, 105.0, 0.5, 0.03, 0.01, 0.28
    g = G.bsm_greeks(S, K, T, r, q, sig, call)
    f = lambda s=S, t=T, v=sig, rr=r: float(P.bsm_price(s, K, t, rr, q, v, call))  # noqa: E731
    h = 1e-4 * S
    assert float(g["delta"]) == pytest.approx((f(s=S + h) - f(s=S - h)) / (2 * h), abs=1e-7)
    assert float(g["gamma"]) == pytest.approx((f(s=S + h) - 2 * f() + f(s=S - h)) / h ** 2, abs=1e-6)
    assert float(g["vega"]) == pytest.approx((f(v=sig + 1e-5) - f(v=sig - 1e-5)) / 2e-5, abs=1e-5), "vega is per unit of volatility (1.00 = 100 points)"
    assert float(g["theta"]) == pytest.approx(-(f(t=T + 1e-5) - f(t=T - 1e-5)) / 2e-5, abs=1e-4), "theta is per year, the change as time passes"
    assert float(g["rho"]) == pytest.approx((f(rr=r + 1e-5) - f(rr=r - 1e-5)) / 2e-5, abs=1e-4), "rho is per unit of rate"
    assert float(g["vanna"]) == pytest.approx(((float(G.bsm_greeks(S, K, T, r, q, sig + 1e-5, call)["delta"]) - float(G.bsm_greeks(S, K, T, r, q, sig - 1e-5, call)["delta"])) / 2e-5), abs=1e-5)
    assert float(g["volga"]) == pytest.approx((float(G.bsm_greeks(S, K, T, r, q, sig + 1e-4, call)["vega"]) - float(G.bsm_greeks(S, K, T, r, q, sig - 1e-4, call)["vega"])) / 2e-4, abs=1e-3)


def test_black_76_greeks_match_finite_differences_with_respect_to_the_forward():
    F, K, T, r, sig = 100.0, 95.0, 0.4, 0.02, 0.3
    g = G.black76_greeks(F, K, T, r, sig, True)
    f = lambda ff=F, v=sig, t=T: float(P.black76_price(ff, K, t, r, v, True))  # noqa: E731
    h = 1e-3
    assert float(g["delta"]) == pytest.approx((f(ff=F + h) - f(ff=F - h)) / (2 * h), abs=1e-7)
    assert float(g["gamma"]) == pytest.approx((f(ff=F + h) - 2 * f() + f(ff=F - h)) / h ** 2, abs=1e-5)
    assert float(g["vega"]) == pytest.approx((f(v=sig + 1e-5) - f(v=sig - 1e-5)) / 2e-5, abs=1e-4)
    assert float(g["theta"]) == pytest.approx(-(f(t=T + 1e-5) - f(t=T - 1e-5)) / 2e-5, abs=1e-3)


# ------------------------------------------------------------------------------------------------------------------ implied volatility inversion
def test_implied_volatility_round_trips_across_moneyness_and_tenor_and_refuses_what_has_no_answer():
    S, r, q = 100.0, 0.03, 0.01
    for T in (1 / 52, 0.25, 1.0, 3.0):
        for K in (60, 85, 100, 115, 160):
            for sig in (0.08, 0.25, 0.9):
                for call in (True, False):
                    price = float(P.bsm_price(S, K, T, r, q, sig, call))
                    iv = float(IV.implied_vol(price, S, K, T, r, q, call))
                    if not np.isfinite(iv):
                        assert price < 1e-8 * S or abs(price - max((S * math.exp(-q * T) - K * math.exp(-r * T)) * (1 if call else -1), 0.0)) < 1e-8 * S, "only an option with no time value may fail"
                        continue
                    assert float(P.bsm_price(S, K, T, r, q, iv, call)) == pytest.approx(price, abs=1e-8 * max(S, 1)), (T, K, sig, call)
    assert np.isnan(float(IV.implied_vol(1.0, 100, 90, 1.0, 0.03, 0.0, True))), "below intrinsic value: no volatility"
    assert np.isnan(float(IV.implied_vol(120.0, 100, 90, 1.0, 0.03, 0.0, True))), "above the underlying: no volatility"
    assert np.isnan(float(IV.implied_vol(5.0, 100, 100, 0.0, 0.03, 0.0, True))), "no time left"


def test_the_black_76_and_bachelier_inverters_round_trip():
    F, K, T, r = 100.0, 105.0, 0.5, 0.02
    for sig in (0.1, 0.4):
        assert float(IV.implied_vol_black76(float(P.black76_price(F, K, T, r, sig, True)), F, K, T, r, True)) == pytest.approx(sig, abs=1e-6)
    pn = float(P.bachelier_price(F, K, T, r, 12.0, True))
    assert float(IV.implied_vol_bachelier(pn, F, K, T, r, True)) == pytest.approx(12.0, abs=1e-6)


# ------------------------------------------------------------------------------------------------------------------ American options and convergence
def test_the_binomial_tree_converges_to_black_scholes_and_the_early_exercise_premium_has_the_right_sign():
    S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    bs = float(P.bsm_price(S, K, T, r, q, sig, False))
    errs = [abs(float(P.binomial_price(S, K, T, r, q, sig, False, False, n)) - bs) for n in (50, 200, 800)]
    assert errs[2] < errs[0] and errs[2] < 5e-3, errs
    am_put = float(P.binomial_price(S, K, T, r, q, sig, False, True, 800))
    assert am_put > bs + 0.1, "an American put on a non-dividend stock with positive rates is worth more than the European"
    am_call = float(P.binomial_price(S, K, T, r, q, sig, True, True, 800))
    assert am_call == pytest.approx(float(P.bsm_price(S, K, T, r, q, sig, True)), abs=5e-3), "no dividends: early exercise of a call is never optimal"
    assert float(P.binomial_price(S, 140.0, T, r, q, sig, False, True, 400)) == pytest.approx(140.0 - S, abs=1e-6), "a deep in-the-money American put is worth its intrinsic value"
    big_div_call = float(P.binomial_price(S, 80.0, T, 0.02, 0.08, sig, True, True, 800))
    assert big_div_call > float(P.bsm_price(S, 80.0, T, 0.02, 0.08, sig, True)) + 0.01, "with a large dividend yield the American call has an early-exercise premium"


def test_heston_collapses_to_black_scholes_when_volatility_of_volatility_vanishes_and_merton_to_it_without_jumps():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    v = 0.04
    h = float(P.heston_price(S, K, T, r, q, v, 2.0, v, 1e-3, 0.0, True))
    assert h == pytest.approx(float(P.bsm_price(S, K, T, r, q, math.sqrt(v), True)), abs=2e-3)
    assert float(P.merton_price(S, K, T, r, q, 0.2, 0.0, 0.0, 0.1, True)) == pytest.approx(float(P.bsm_price(S, K, T, r, q, 0.2, True)), abs=1e-8)


# ------------------------------------------------------------------------------------------------------------------ no-arbitrage on a chain
def _count(view, name):
    return next(r for r in view["rows"] if r["check"].startswith(name))


def test_a_flat_chain_has_no_violations_and_a_corrupted_quote_is_caught_by_the_vertical_butterfly_and_parity_checks():
    chain = flat_chain()
    clean = oa.arbitrage_view(chain)
    assert all(r["violations"] == 0 and r["tested"] > 0 for r in clean["rows"])
    bad = chain.copy()
    idx = bad[(bad["right"] == "C") & (bad["expiry"] == bad["expiry"].min()) & (bad["strike"] == 100.0)].index
    for col in ("bid", "ask", "mid"):
        bad.loc[idx, col] = bad.loc[idx, col] * 3.0                                                         # one call priced far too high
    caught = oa.arbitrage_view(bad)
    assert _count(caught, "butterfly")["violations"] >= 1 and _count(caught, "vertical")["violations"] >= 1 and _count(caught, "put-call")["violations"] >= 1


def test_calendar_arbitrage_is_caught_when_a_later_expiry_has_less_total_variance():
    chain = flat_chain(days=(20, 60)).copy()
    late = chain["expiry"] == chain["expiry"].max()
    chain.loc[late, "iv"] = 0.05                                                                              # total variance falls with expiry
    assert _count(oa.arbitrage_view(chain), "calendar")["violations"] > 0


def test_american_parity_is_a_pair_of_bounds_not_an_equality():
    chain = flat_chain(days=(60,), dividend=0.0, rate=0.05).copy()
    # an American put is worth more than a European one: raise the in-the-money puts by the (small) early-exercise premium and parity as an equality breaks, the bounds do not
    itm = (chain["right"] == "P") & (chain["strike"] > chain["underlying"] * 1.1)
    for col in ("bid", "ask", "mid"):
        chain.loc[itm, col] = chain.loc[itm, col] * 1.02
    eu, am = oa.arbitrage_view(chain, exercise="european"), oa.arbitrage_view(chain, exercise="american")
    assert _count(eu, "put-call")["violations"] > 0 and _count(am, "put-call")["violations"] == 0 and "American" in _count(am, "put-call")["check"]


# ------------------------------------------------------------------------------------------------------------------ gamma exposure: observed against assumed
def test_gamma_exposure_separates_observed_gross_gamma_from_assumed_dealer_positioning():
    chain = flat_chain(open_interest=100.0, days=(30,))
    out = oa.positioning(chain, chain["expiry"].min())
    assert out["gross_gamma"]["total"] > 0 and out["gross_gamma"]["basis"].startswith("observed")
    assert out["gex"]["basis"] == "assumed dealer positioning" and out["gex"]["scenario"] == "calls_long_puts_short"
    assert set(out["scenarios"]) == {"calls_long_puts_short", "dealers_short_all", "dealers_long_all", "calls_short_puts_long"}
    assert out["scenarios"]["dealers_long_all"]["net"] == pytest.approx(out["gross_gamma"]["total"]) and out["scenarios"]["dealers_short_all"]["net"] == pytest.approx(-out["gross_gamma"]["total"])
    assert out["scenarios"]["calls_long_puts_short"]["net"] == pytest.approx(-out["scenarios"]["calls_short_puts_long"]["net"])
    assert out["uncertainty"]["sign_agrees_across_scenarios"] is False and out["uncertainty"]["net_range"][0] < 0 < out["uncertainty"]["net_range"][1]
    assert any("ASSUMED" in a for a in out["assumptions"]) and any("no side" in a for a in out["assumptions"])


def test_the_dealer_assumption_is_configurable_and_scales_with_the_share_of_open_interest_assumed_held():
    chain = flat_chain(open_interest=100.0, days=(30,))
    e = chain["expiry"].min()
    base = oa.positioning(chain, e, dealer="dealers_long_all")
    half = oa.positioning(chain, e, dealer="dealers_long_all", dealer_share=0.5)
    assert half["gex"]["net"] == pytest.approx(0.5 * base["gex"]["net"])
    custom = oa.positioning(chain, e, dealer=(1.0, 0.0))
    assert custom["gex"]["scenario"] == "custom" and custom["gex"]["net"] == pytest.approx(base["gex"]["calls"])
    assert oa.positioning(chain, e)["gex"]["net"] == pytest.approx(oa.positioning(chain, e, dealer="calls_long_puts_short")["gex"]["net"])
    for bad in ({"dealer": "nobody"}, {"dealer": (2.0, 0.0)}, {"dealer_share": 1.5}):
        with pytest.raises(ValueError):
            oa.positioning(chain, e, **bad)


# ------------------------------------------------------------------------------------------------------------------ provenance
def test_provenance_classifies_data_and_never_calls_a_simulation_empirical():
    sim = pv.build({"chain": "simulated", "underlying_price": "platform"}, as_of="2026-11-20")
    assert sim["kind"] == "mixed" and sim["parts"]["chain"] == "synthetic" and any("synthetic" in w.lower() for w in sim["warnings"])
    only = pv.build({"chain": "simulated"}, as_of="2026-11-20")
    assert only["kind"] == "synthetic" and only["freshness"] == "not_applicable"
    live = pv.build({"chain": "downloaded"}, as_of="2026-11-20", observed_at="2026-11-20T15:30:00", instrument="SPY", now="2026-11-20T16:00:00")
    assert live["kind"] == "observed_delayed" and live["freshness"] == "fresh" and live["instrument"] == "SPY" and live["age_hours"] == 0.5
    stale = pv.build({"chain": "downloaded"}, as_of="2026-11-10", observed_at="2026-11-10T15:30:00", now="2026-11-20T16:00:00")
    assert stale["freshness"] == "stale" and any("hours old" in w for w in stale["warnings"])
    hist = pv.build({"curve": "platform"}, as_of="2020-03-01")
    assert hist["kind"] == "historical_observed" and hist["freshness"] == "historical"
    cal = pv.build({"price": "model"}, as_of="2026-11-20", calibration={"status": "calibrated", "rmse": 0.004})
    assert cal["kind"] == "calibrated_model"
    nocal = pv.build({"price": "model"}, as_of="2026-11-20", calibration={"status": "failed"})
    assert any("not calibrated" in w for w in nocal["warnings"])


def test_every_options_and_futures_payload_carries_the_provenance_block(tmp_path):
    from src.assets import common
    out = common.envelope("x", {"a": 1}, {"chain": "simulated"}, as_of="2026-11-20")
    assert out["provenance"]["kind"] == "synthetic" and out["source"] == "simulated"


# ------------------------------------------------------------------------------------------------------------------ futures
from src.assets import futures as F  # noqa: E402
from src.assets import futures_analytics as fa  # noqa: E402


@pytest.fixture(scope="module")
def cl():
    return fa.simulated_curve("CL", as_of="2026-09-18")


def test_contract_terms_and_tick_values_match_the_exchange_specifications():
    spec = {r: (s.multiplier, s.tick_size, s.tick_value) for r, s in fa.ROOTS.items()}
    assert spec["CL"] == (1000.0, 0.01, 10.0) and spec["NG"][0] == 10000.0 and spec["NG"][2] == pytest.approx(10.0)
    assert spec["GC"] == (100.0, 0.10, 10.0) and spec["HG"][2] == pytest.approx(12.5) and spec["ES"] == (50.0, 0.25, 12.5)


def test_carry_is_the_log_slope_annualised_and_its_sign_names_the_curve_state():
    price = pd.DataFrame({"F1": [100.0, 100.0], "F2": [102.0, 98.0]})
    ttm = pd.DataFrame({"F1": [0.1, 0.1], "F2": [0.2, 0.2]})
    c = F.carry(price, ttm)
    assert c.iloc[0] == pytest.approx(math.log(100 / 102) / 0.1) and c.iloc[0] < 0 < c.iloc[1], "contango has negative carry for a long, backwardation positive"
    assert fa.curve_state(np.array([-0.05]), 0.02) == "contango" and fa.curve_state(np.array([0.05]), -0.02) == "backwardation" and fa.curve_state(np.array([0.001]), 0.0) == "mixed"
    assert fa.curve_state(np.array([-0.05]), -0.02) == "mixed", "a humped curve is neither"


def test_the_roll_separates_a_price_difference_from_a_cash_cost_and_from_realised_results(cl):
    roll = fa.roll_analysis(cl, 5)
    d, spec = roll["decomposition"], cl.spec
    assert d["calendar_spread"]["dollars_per_contract"] == pytest.approx(roll["calendar_spread_points"] * spec.multiplier)
    assert d["implied_roll_down"]["dollars_per_contract_long"] == pytest.approx(-d["calendar_spread"]["dollars_per_contract"]), "under a static curve a long rolls down by exactly the spread"
    assert d["execution_cost"]["dollars_per_contract"] == pytest.approx(spec.spread_ticks * spec.tick_value + 2 * spec.commission) and roll["roll_cost_dollars"] == d["execution_cost"]["dollars_per_contract"]
    assert not any("costs" in f and "rolling a long position costs" in f for f in roll["risk_flags"]), "a calendar spread is never reported as the cost of rolling"
    if d["realised_roll_contribution"]:
        assert d["realised_excess_return"]["log_return"] == pytest.approx(d["realised_roll_contribution"]["log_return"] + d["realised_roll_contribution"]["spot_log_return"])
    else:
        assert "spot" in roll["decomposition"]["realised_roll_contribution_note"]


def test_the_continuous_series_has_no_artificial_jump_and_the_excess_return_charges_no_roll():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", "2024-12-31")
    rows = []
    for i, (c, exp) in enumerate((("A", "2024-04-19"), ("B", "2024-07-19"), ("C", "2024-10-18"), ("D", "2025-01-17"))):
        for d in dates:
            if d < pd.Timestamp(exp):
                rows.append({"date": d, "contract": c, "expiry": pd.Timestamp(exp), "price": 100.0 + 2.0 * i + 0.01 * (d - dates[0]).days + 0.1 * rng.normal(), "open_interest": 1000.0})
    long = pd.DataFrame(rows)
    ratio = F.continuous_series(long, 5, "ratio")
    none = F.continuous_series(long, 5, "none")
    assert len(ratio.roll_dates) == 3 and ratio.returns.abs().max() < 0.05, "back-adjusted: no jump at the roll"
    assert none.returns.abs().max() > ratio.returns.abs().max(), "unadjusted splices keep the artificial gap"
    ex = F.excess_return_series(long, 5)
    d = ratio.roll_dates[0]
    day = ex.loc[d]
    prev = long[(long["contract"] == ratio.contract.shift(1).loc[d]) & (long["date"] == d)]["price"].iloc[0] / long[(long["contract"] == ratio.contract.shift(1).loc[d]) & (long["date"] == dates[list(dates).index(d) - 1])]["price"].iloc[0] - 1.0
    assert day == pytest.approx(prev), "on the roll day the return is the OLD contract's: rolling itself earns and costs nothing"


def test_the_margin_dashboard_uses_the_contract_multiplier_and_scales_margin_by_the_quantity(cl):
    out = fa.margin_dashboard([{"root": "CL", "quantity": -2}], {"CL": cl}, equity=100_000.0)
    row = out["positions"][0]
    assert row["notional"] == pytest.approx(-2 * row["price"] * cl.spec.multiplier) and row["gross_notional"] == pytest.approx(abs(row["notional"]))
    assert row["initial_margin"] == pytest.approx(2 * cl.spec.initial_margin) and out["leverage"] == pytest.approx(row["gross_notional"] / 100_000.0)
    assert any("illustrative" in n.lower() for n in out["notes"]), "margins are labelled illustrative, never presented as the exchange's"
