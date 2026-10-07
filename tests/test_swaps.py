"""The swaps module: schedules and day counts, curve construction and bootstrapping, cashflow tables, pricing identities, risk (PV01, carry and roll-down) and engine integration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.engine import CostSchedule, Engine, EngineConfig, FinancingModel, Schedule, SpreadModel, Strategy, Target
from src.instruments import InstrumentRegistry, make_irs
from src.instruments.rates import BasisSwap
from src.swaps.cashflows import cashflow_table
from src.swaps.contracts import par_swap
from src.swaps.curves import CurveSet, DiscountCurve, bootstrap_par_curve, tenor_basis_curve
from src.swaps.pricing import bid_ask, explain_pnl, par_rate, par_spread, price
from src.swaps.risk import butterfly_weights, carry_rolldown, convexity_pv, dv01_neutral_ratio, key_rate_pv01, parallel, pv01, scenario, steepener
from src.swaps.schedules import generate_schedule, notional_profile, unadjusted_dates, year_fraction
from src.swaps.synthetic import synthetic_rates_market

warnings.filterwarnings("ignore")
VAL = pd.Timestamp("2024-01-02")


def usd_curves(level=0.045, slope=0.0, basis_bp=0.0):
    tenors = (0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)
    zeros = tuple(level + slope * np.log1p(t) / 3.4 for t in tenors)
    disc = DiscountCurve(VAL, tenors, zeros, name="USD-DISCOUNT")
    proj = DiscountCurve(VAL, tenors, tuple(z + basis_bp * 1e-4 for z in zeros), name="USD-PROJ-3M")
    return CurveSet({"USD-DISCOUNT": disc}, {"USD-PROJ-3M": proj})


# ------------------------------------------------------------------------------------------------------------------------------ day counts and schedules
def test_day_count_conventions():
    a, b = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-07-31")
    assert year_fraction(a, b, "ACT/360") == 182 / 360 and year_fraction(a, b, "ACT/365F") == 182 / 365
    assert year_fraction(a, b, "30/360") == 0.5 and year_fraction(a, b, "30E/360") == 0.5
    assert abs(year_fraction("2023-07-01", "2024-07-01", "ACT/ACT") - (184 / 365 + 182 / 366)) < 1e-12   # ISDA: each year-slice over its own year length
    assert year_fraction(b, a, "ACT/360") == -year_fraction(a, b, "ACT/360")
    with pytest.raises(ValueError):
        year_fraction(a, b, "BANANA")


def test_us_30_360_end_of_month_rule():
    assert year_fraction("2024-01-30", "2024-03-31", "30/360") == (360 * 0 + 30 * 2 + (30 - 30)) / 360   # 31st becomes 30th when the start is 30th or 31st
    assert year_fraction("2024-01-15", "2024-03-31", "30/360") == (30 * 2 + 16) / 360                     # but not otherwise


def test_schedule_stubs_and_adjustment():
    d = unadjusted_dates("2024-01-15", "2025-01-15", "3M")
    assert len(d) == 5 and d[0] == pd.Timestamp("2024-01-15") and d[-1] == pd.Timestamp("2025-01-15")
    short = unadjusted_dates("2024-02-01", "2025-01-15", "3M", "short_front")
    assert short[0] == pd.Timestamp("2024-02-01") and short[1] == pd.Timestamp("2024-04-15")           # a short first period
    stub = unadjusted_dates("2024-03-20", "2025-01-15", "3M", "short_front")
    merged = unadjusted_dates("2024-03-20", "2025-01-15", "3M", "long_front")                           # a 26-day front stub is merged into the next period
    assert len(merged) == len(stub) - 1 and merged[0] == stub[0] and merged[1] == stub[2]
    s = generate_schedule("2024-01-15", "2025-01-15", "3M", "ACT/360", "US")
    assert (s["accrual"] > 0).all() and abs(s["accrual"].sum() - 366 / 360) < 0.01
    assert all(get_ok for get_ok in (s["pay_date"].dt.weekday < 5))
    assert (s["fixing_date"] < s["reset_date"]).all()                                                   # fixed two business days before the reset


def test_modified_following_moves_a_holiday_payment():
    s = generate_schedule("2024-01-04", "2024-07-04", "6M", "30/360", "US")
    assert s["end"].iloc[-1] == pd.Timestamp("2024-07-04") and s["adj_end"].iloc[-1] == pd.Timestamp("2024-07-05")


def test_notional_profiles():
    assert notional_profile(100.0, 4, "bullet") == [100.0] * 4
    am = notional_profile(100.0, 4, "amortizing", (10.0, 20.0))
    assert am == [100.0, 90.0, 70.0, 50.0]
    with pytest.raises(ValueError):
        notional_profile(100.0, 4, "amortizing")
    assert notional_profile(100.0, 3, "custom", (100.0, 50.0)) == [100.0, 50.0, 50.0]


# ----------------------------------------------------------------------------------------------------------------------------------------- curves
def test_discount_curve_interpolation_and_forward_consistency():
    c = DiscountCurve.flat(VAL, 0.04)
    assert abs(float(c.df(2.0)) - np.exp(-0.08)) < 1e-12 and float(c.df(0.0)) == 1.0
    f = c.forward_rate(1.0, 2.0)
    assert abs(f - (np.exp(0.04) - 1.0)) < 1e-9                                                         # the SIMPLE forward over a year on a flat 4% continuously-compounded curve
    u = usd_curves().discount["USD-DISCOUNT"]
    assert abs(float(u.df(3.0)) * float(u.df_date(VAL)) - float(u.df(3.0))) < 1e-12
    assert np.all(np.diff(u.df(np.linspace(0.1, 30, 200))) < 0)                                         # discount factors fall
    with pytest.raises(ValueError):
        DiscountCurve(VAL, (1.0, 1.0), (0.01, 0.01))


def test_bump_and_roll_behave():
    c = usd_curves().discount["USD-DISCOUNT"]
    b = c.bumped(10.0)
    assert np.allclose(np.asarray(b.zeros) - np.asarray(c.zeros), 10e-4)
    rolled = c.rolled(1.0)
    assert abs(float(rolled.zero(2.0)) - float(c.zero(3.0))) < 1e-2                                     # the curve keeps its shape in tenor space


def test_par_curve_bootstrap_reprices_the_par_swaps_it_was_built_from():
    tenors, rates = [1, 2, 3, 5, 7, 10], [0.040, 0.0415, 0.043, 0.045, 0.0465, 0.048]
    curve = bootstrap_par_curve(VAL, tenors, rates, freq=2, name="USD")
    for T, r in zip(tenors, rates):
        grid = np.arange(0.5, T + 1e-9, 0.5)
        annuity = 0.5 * float(np.sum(curve.df(grid)))
        implied = (1.0 - float(curve.df(T))) / annuity
        assert abs(implied - r) < 2e-5, (T, implied, r)


def test_tenor_basis_curve_adds_the_spread():
    d = usd_curves().discount["USD-DISCOUNT"]
    p = tenor_basis_curve(d, {1: 5.0, 10: 15.0})
    assert abs(float(p.zero(1.0)) - float(d.zero(1.0)) - 5e-4) < 1e-9 and abs(float(p.zero(10.0)) - float(d.zero(10.0)) - 15e-4) < 1e-9


# ---------------------------------------------------------------------------------------------------------------------------------- pricing identities
def test_par_swap_has_zero_value_and_par_rate_is_recoverable():
    cs = usd_curves(slope=0.01, basis_bp=3.0)
    swap = par_swap(cs, VAL, "2024-01-04", "2029-01-04")
    assert abs(price(swap, cs, VAL).pv) < 0.01 * swap.principal * 1e-4 * 5           # well under a hundredth of a basis point of annuity
    k = par_rate(make_irs("USD", "2024-01-04", "2029-01-04", 0.0), cs, VAL)
    assert abs(k - swap.fixed_rate) < 1e-7 and 0.03 < k < 0.07


def test_floating_leg_equals_df_start_minus_df_end_on_a_single_curve():
    c = DiscountCurve.from_values(VAL, {0.25: 0.04, 1: 0.042, 5: 0.045, 10: 0.047}, name="C")
    cs = CurveSet({"C": c}, {"C": c})
    swap = make_irs("USD", "2024-01-04", "2029-01-04", 0.04, 1_000_000.0, discount_curve_id="C", projection_curve_id="C", daycount_float="ACT/360") if False else \
        make_irs("USD", "2024-01-04", "2029-01-04", 0.04, 1_000_000.0, discount_curve_id="C", projection_curve_id="C")
    v = price(swap, cs, VAL)
    start_df, end_df = float(c.df_date(pd.Timestamp("2024-01-04"))), float(c.df_date(pd.Timestamp(swap.expiry)))
    # a floating leg that pays the projected forward on its own accrual (no spread, no payment lag) is worth N (DF(start) - DF(end)) up to the day-count/business-day alignment
    assert abs(abs(v.pv_by_leg["float"]) - 1_000_000.0 * (start_df - end_df)) < 1_000_000.0 * 2e-4


def test_pay_fixed_and_receive_fixed_are_opposites():
    cs = usd_curves(slope=0.01)
    pay = make_irs("USD", "2024-01-04", "2029-01-04", 0.04, pay_fixed=True)
    rec = make_irs("USD", "2024-01-04", "2029-01-04", 0.04, pay_fixed=False)
    assert abs(price(pay, cs, VAL).pv + price(rec, cs, VAL).pv) < 1e-6
    assert price(pay, cs, VAL).pv > 0 if par_rate(pay, cs, VAL) > 0.04 else price(pay, cs, VAL).pv < 0       # paying below par is worth money


def test_cashflow_table_amounts_and_signs():
    cs = usd_curves()
    swap = make_irs("USD", "2024-01-04", "2026-01-04", 0.04, 2_000_000.0, pay_fixed=True)
    t = cashflow_table(swap, cs, VAL)
    fixed = t[t["leg"] == "fixed"]
    assert len(fixed) == 4 and (fixed["amount"] < 0).all()                                                  # the payer pays fixed
    assert np.allclose(fixed["amount"].abs(), 2_000_000.0 * 0.04 * fixed["accrual"])
    assert (t[t["leg"] == "float"]["amount"] > 0).all()
    assert (t["pv"].abs() <= t["amount"].abs() + 1e-9).all()


def test_basis_swap_par_spread_zeroes_the_value():
    cs = usd_curves(basis_bp=0.0)
    cs.projection["USD-PROJ-6M"] = tenor_basis_curve(cs.discount["USD-DISCOUNT"], {1: 10.0, 10: 10.0})
    eff = pd.Timestamp("2024-01-04")
    spec = BasisSwap(instrument_id="B", asset_class="swap", instrument_type="basis_swap", currency="USD", tick_size=1e-6, lot_size=1e-3, calendar="US", expiry=eff + pd.DateOffset(years=5),
                     settlement_type="cash", effective_date=eff, principal=1_000_000.0, payment_calendar="US", reset_calendar="US", discount_curve_id="USD-DISCOUNT",
                     projection_curve_id="USD-PROJ-3M", index_1="USD-3M", tenor_1="3M", frequency_1="3M", index_2="USD-6M", tenor_2="6M", frequency_2="6M",
                     projection_curve_id_1="USD-PROJ-3M", projection_curve_id_2="USD-PROJ-6M")
    s = par_spread(spec, cs, VAL)
    assert abs(price(spec.replace(spread_1=s), cs, VAL).pv) < 1.0
    assert abs(s) > 5e-4                                                                                    # the 6M leg pays ~10bp more, so the 3M leg's spread is about that large


# ------------------------------------------------------------------------------------------------------------------------------------------ risk
def test_pv01_matches_a_finite_difference_and_has_the_right_sign():
    cs = usd_curves(slope=0.01, basis_bp=2.0)
    pay = par_swap(cs, VAL, "2024-01-04", "2034-01-04", pay_fixed=True, notional=10_000_000.0)
    d = pv01(pay, cs, VAL)
    manual = price(pay, cs.bumped(1.0), VAL).pv - price(pay, cs, VAL).pv
    assert abs(d - manual) < 0.02 * abs(d) and d > 0                       # a payer gains when rates rise
    assert abs(pv01(pay.reversed(), cs, VAL) + d) < 1e-6
    assert 6_000 < d < 10_000                                              # ~ 10m x annuity (~8) x 1bp


def test_key_rate_pv01_sums_to_the_parallel_pv01():
    cs = usd_curves(slope=0.01)
    swap = par_swap(cs, VAL, "2024-01-04", "2029-01-04", notional=5_000_000.0)
    kr = key_rate_pv01(swap, cs, VAL)
    assert abs(kr.sum() - pv01(swap, cs, VAL)) < 0.05 * abs(pv01(swap, cs, VAL))
    assert kr.idxmax() in (3, 5, 7)                                         # the risk sits near the swap's maturity


def test_convexity_is_small_and_scenarios_reprice():
    cs = usd_curves()
    swap = par_swap(cs, VAL, "2024-01-04", "2034-01-04", notional=10_000_000.0)
    up = scenario(swap, cs, VAL, parallel(100.0))
    assert up > 0 > scenario(swap, cs, VAL, parallel(-100.0)) and abs(convexity_pv(swap, cs, VAL)) < abs(pv01(swap, cs, VAL))
    assert abs(scenario(swap, cs, VAL, steepener(20.0))) > 0


def test_carry_plus_rolldown_equals_the_total_and_follows_the_curve_slope():
    steep = usd_curves(level=0.03, slope=0.03)
    swap = par_swap(steep, VAL, "2024-01-04", "2034-01-04", pay_fixed=True, notional=10_000_000.0)
    cr = carry_rolldown(swap, steep, VAL, 91)
    assert abs(cr["carry"] + cr["rolldown"] - cr["total"]) < 1e-6
    flat = usd_curves(level=0.04, slope=0.0)
    swap_flat = par_swap(flat, VAL, "2024-01-04", "2034-01-04", pay_fixed=True, notional=10_000_000.0)
    cr_flat = carry_rolldown(swap_flat, flat, VAL, 91)
    assert abs(cr_flat["rolldown"]) < 0.5 * abs(cr["rolldown"])                                             # much less roll-down on a flat curve (day-count residue remains)
    rec = carry_rolldown(swap.reversed(), steep, VAL, 91)
    assert abs(rec["total"] + cr["total"]) < 1e-6                                                           # the receiver earns exactly what the payer loses


def test_explain_pnl_adds_up():
    c0, c1 = usd_curves(), usd_curves(level=0.05, slope=0.005)
    swap = par_swap(c0, VAL, "2024-01-04", "2029-01-04", notional=5_000_000.0)
    out = explain_pnl(swap, c0, c1, VAL, VAL + pd.Timedelta(days=30))
    assert abs(out["time_effect"] + out["curve_effect"] - out["total"]) < 1e-6 and out["curve_effect"] > 0


def test_bid_ask_brackets_the_mid_by_a_fraction_of_annuity():
    cs = usd_curves()
    swap = par_swap(cs, VAL, "2024-01-04", "2029-01-04", notional=10_000_000.0)
    bid, mid, ask = bid_ask(swap, cs, VAL, 0.5)
    assert bid < mid < ask and abs((ask - bid) / 2.0 - 0.5e-4 * price(swap, cs, VAL).annuity) < 1e-6


def test_hedge_ratios():
    assert dv01_neutral_ratio(1000.0, -250.0) == 4.0
    f, b, k = butterfly_weights(2.0, 5.0, 8.0)
    assert abs(f * 2.0 + k * 8.0 - 5.0) < 1e-9 and abs(f * 2.0 - k * 8.0) < 1e-9        # the wings carry the belly's DV01, half each
    with pytest.raises(ValueError):
        dv01_neutral_ratio(1.0, 0.0)


# ------------------------------------------------------------------------------------------------------------------------------ engine integration
def swap_market(n=60):
    mk = synthetic_rates_market("2024-01-02", n, ("USD",), seed=3, fx_pairs={})
    return mk


def test_synthetic_rates_market_publishes_curves_and_fixings_at_their_availability():
    mk = swap_market(10)
    ev = mk["events"]
    assert {"curve", "fixing"} <= set(ev["event_type"])
    fix = ev[ev["event_type"] == "fixing"]
    assert (fix["available_at"] >= fix["timestamp"]).all()
    assert set(ev.loc[ev["event_type"] == "curve", "instrument_id"]) >= {"USD-DISCOUNT", "USD-PROJ-3M"}


def test_swap_traded_in_the_engine_pays_coupons_and_reconciles():
    mk = swap_market(70)
    dates = mk["dates"]
    cs_events = mk["events"]
    eff = pd.Timestamp("2024-01-08")
    swap = make_irs("USD", eff, eff + pd.DateOffset(months=6), 0.045, 5_000_000.0, pay_fixed=True)
    reg = InstrumentRegistry([swap])

    class Pay(Strategy):
        name = "pay"
        schedule = Schedule("daily", "16:30", 4, "US")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == dates[3]:
                ctx.set_targets([Target(swap.instrument_id, quantity=1.0)])

    cfg = EngineConfig(start=dates[0], end=dates[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 1_000_000.0})
    e = Engine(reg, cs_events, [Pay()], cfg, CostSchedule().set("swap", spread=SpreadModel(0.0, 0.0)), FinancingModel())
    res = e.run()
    assert res.reconciliation.ok, res.reconciliation.failures()
    j = res.journal
    coupons = j[j["category"] == "coupon"]
    assert len(coupons) == 1 and pd.Timestamp(coupons["ts"].iloc[0]).normalize() == pd.Timestamp("2024-04-08")      # the first quarterly floating payment (fixed side pays semi-annually)
    assert abs(coupons["cash_base"].iloc[0] - coupons["pnl"].iloc[0]) < 1e-9 and coupons["cash_base"].iloc[0] != 0.0
    assert (j["category"] == "mtm").sum() > 30 and abs(e.ledger.quantity(swap.instrument_id) - 1.0) < 1e-12
    # OTC swaps are entered at the mid-priced PV with the cost of crossing booked separately, and the position is carried at its model value
    assert res.attribution("category")["spread"] < 0


# --------------------------------------------------------------------------------------------------- FX forwards and cross-currency swaps
def test_fx_forward_in_the_engine_is_marked_from_curves_and_settles_by_exchanging_the_currencies():
    from src.instruments import fx_forward, fx_spot
    from src.marketdata import concat_events, events_from_prices, normalise_events

    idx = pd.bdate_range("2024-01-02", periods=60)
    spot_path = 1.10 + 0.0005 * np.arange(60)
    expiry = pd.Timestamp("2024-03-15")
    K = 1.10
    fwd = fx_forward("EUR", "USD", expiry, K)
    reg = InstrumentRegistry([fx_spot("EUR", "USD"), fwd])
    curves = pd.DataFrame({"timestamp": np.repeat(idx + pd.Timedelta(hours=16), 2), "instrument_id": [f"{c}-DISCOUNT" for _ in idx for c in ("USD", "EUR")], "event_type": "curve",
                           "curve_values": [{0.25: r, 1.0: r} for _ in idx for r in (0.045, 0.03)]})
    ev = concat_events(events_from_prices(pd.DataFrame({"EURUSD": spot_path}, index=idx), "bar", "16:00", spread_bps=0.5), normalise_events(curves, lag="0s"))

    class Buy(Strategy):
        name = "buy"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[1]:
                ctx.set_targets([Target(fwd.instrument_id, quantity=1_000_000.0)])

    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 2_000_000.0})
    e = Engine(reg, ev, [Buy()], cfg, CostSchedule().set("fx", spread=SpreadModel(0.0, 0.0)), FinancingModel())
    res = e.run()
    assert res.reconciliation.ok, res.reconciliation.failures()
    assert e.ledger.quantity(fwd.instrument_id) == 0.0                                       # settled at the value date and gone
    entry = float(res.fills["price"].iloc[0]) * 1_000_000.0                                   # the all-in price of entering the forward: its present value plus the cost of crossing
    assert abs(e.ledger.cash.get("EUR", 0.0) - 1_000_000.0) < 1e-6                           # the base currency was delivered at the value date
    assert abs(e.ledger.cash["USD"] - (2_000_000.0 - K * 1_000_000.0 - entry)) < 1e-6        # and the contracted dollars paid, on top of what it cost to enter
    final_spot = float(spot_path[-1])
    assert abs(res.equity_curve.iloc[-1] - (2_000_000.0 + 1_000_000.0 * (final_spot - K) - entry)) < 1e-6      # euros marked at the last spot
    assert entry > 0 and entry < 0.01 * 1_000_000.0                                          # the forward points were worth a few thousand dollars


def test_cross_currency_basis_swap_prices_with_notional_exchanges_and_a_spread():
    from src.instruments.rates import CrossCurrencyBasisSwap

    cs = usd_curves()
    cs.discount["EUR-DISCOUNT"] = DiscountCurve(VAL, (0.25, 1, 5, 10), (0.03, 0.03, 0.031, 0.032), name="EUR-DISCOUNT")
    cs.projection["EUR-PROJ-3M"] = cs.discount["EUR-DISCOUNT"]
    eff = pd.Timestamp("2024-01-04")
    spot = 1.10
    spec = CrossCurrencyBasisSwap(instrument_id="XCCY", asset_class="swap", instrument_type="xccy_basis_swap", currency="USD", tick_size=1e-6, lot_size=1e-3, calendar="US",
                                  expiry=eff + pd.DateOffset(years=3), settlement_type="cash", effective_date=eff, principal=11_000_000.0, payment_calendar="US", reset_calendar="US",
                                  discount_curve_id="USD-DISCOUNT", projection_curve_id="USD-PROJ-3M", currency_2="EUR", notional_2=10_000_000.0, fx_spot_id="EURUSD",
                                  index_1="USD-3M", index_2="EUR-3M", discount_curve_id_2="EUR-DISCOUNT", projection_curve_id_2="EUR-PROJ-3M")
    with pytest.raises(ValueError):
        price(spec, cs, VAL)                                                           # the spot rate is required: no guessing
    s = par_spread(spec, cs, VAL, spot=spot)
    assert abs(price(spec.replace(spread_2=s), cs, VAL, spot=spot).pv) < 1.0
    t = price(spec, cs, VAL, spot=spot).table
    assert set(t["leg"]) >= {"leg1", "leg2", "exchange"} and (t["leg"] == "exchange").sum() >= 2


def test_cleared_swap_settles_its_value_changes_in_cash_every_day():
    mk = swap_market(60)
    dates = mk["dates"]
    eff = pd.Timestamp("2024-01-08")
    cleared = make_irs("USD", eff, eff + pd.DateOffset(years=2), 0.045, 5_000_000.0, pay_fixed=True, settlement_mode="variation_margin")
    assert cleared.cash_style == "variation_margin"
    reg = InstrumentRegistry([cleared])

    class Pay(Strategy):
        name = "pay"
        schedule = Schedule("daily", "16:30", 4, "US")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == dates[3]:
                ctx.set_targets([Target(cleared.instrument_id, quantity=1.0)])

    cfg = EngineConfig(start=dates[0], end=dates[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 1_000_000.0})
    e = Engine(reg, mk["events"], [Pay()], cfg, CostSchedule().set("swap", spread=SpreadModel(0.0, 0.0)), FinancingModel())
    res = e.run()
    assert res.reconciliation.ok, res.reconciliation.failures()
    j = res.journal
    assert (j["category"] == "variation_margin").sum() > 30 and (j["category"] == "mtm").sum() == 0       # the PV change is cash, not a carried value
    assert e.ledger.positions[cleared.instrument_id].last_value == 0.0
    assert abs(e.ledger.cash["USD"] - res.equity_curve.iloc[-1]) < 1e-6                                    # nothing is carried on the balance sheet: equity is cash
