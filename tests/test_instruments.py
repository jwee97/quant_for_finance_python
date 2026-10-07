"""The unified instrument layer: calendars, contract specifications, chains, serialisation and the conventions each instrument family needs."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from src.instruments import (Instrument, InstrumentRegistry, Option, RollSpec, build_chain, crypto_future, crypto_perp, crypto_spot, easter_sunday, fx_forward, fx_spot, fx_swap,
                             get_calendar, instrument_from_dict, joint_calendar, make_irs, make_option)
from src.instruments.calendars import nth_weekday


# ------------------------------------------------------------------------------------------------------------------------------- calendars
def test_easter_and_nth_weekday():
    assert easter_sunday(2024) == dt.date(2024, 3, 31)
    assert easter_sunday(2025) == dt.date(2025, 4, 20)
    assert nth_weekday(2024, 11, 3, 4) == dt.date(2024, 11, 28)            # Thanksgiving: fourth Thursday
    assert nth_weekday(2024, 5, 0, -1) == dt.date(2024, 5, 27)             # Memorial Day: last Monday


def test_us_holidays_and_observance():
    us = get_calendar("US")
    assert not us.is_business_day("2024-07-04")
    assert not us.is_business_day("2024-11-28")
    assert not us.is_business_day("2024-03-29")                              # Good Friday
    assert us.is_business_day("2024-07-05")
    assert not us.is_business_day("2021-12-24")                              # Christmas on a Saturday is observed on the Friday
    assert not us.is_business_day("2022-06-20")                              # Juneteenth observed (the 19th is a Sunday)


def test_calendar_247_and_weekday():
    assert get_calendar("24x7").is_business_day("2024-06-15")                # a Saturday
    assert not get_calendar("WEEKDAY").is_business_day("2024-06-15")
    assert get_calendar("WEEKDAY").is_business_day("2024-12-25")             # no holidays at all


def test_business_day_conventions():
    us = get_calendar("US")
    sat = pd.Timestamp("2024-06-29")                                          # Saturday; Monday the 1st of July is in the next month
    assert us.adjust(sat, "following") == pd.Timestamp("2024-07-01")
    assert us.adjust(sat, "modified_following") == pd.Timestamp("2024-06-28")  # does not cross the month end
    assert us.adjust(sat, "preceding") == pd.Timestamp("2024-06-28")
    assert us.adjust("2024-07-04", "modified_following") == pd.Timestamp("2024-07-05")
    assert us.adjust("2024-07-03", "following") == pd.Timestamp("2024-07-03")
    with pytest.raises(ValueError):
        us.adjust(sat, "sideways")


def test_add_business_days_and_counts():
    us = get_calendar("US")
    assert us.add_business_days("2024-07-03", 1) == pd.Timestamp("2024-07-05")   # skips the 4th
    assert us.add_business_days("2024-07-05", -1) == pd.Timestamp("2024-07-03")
    assert us.business_days_between("2024-07-01", "2024-07-08") == 4


def test_joint_calendar_is_open_only_when_all_are():
    j = joint_calendar("US", "TARGET")
    assert not j.is_business_day("2024-07-04")                                # US holiday
    assert not j.is_business_day("2024-05-01")                                # TARGET holiday
    assert j.is_business_day("2024-07-08")


def test_session_bounds_are_utc():
    open_, close = get_calendar("US").session_bounds("2024-01-02")
    assert open_ == pd.Timestamp("2024-01-02 14:30") and close == pd.Timestamp("2024-01-02 21:00")      # EST = UTC-5
    open_, close = get_calendar("US").session_bounds("2024-07-01")
    assert open_ == pd.Timestamp("2024-07-01 13:30")                         # EDT = UTC-4


# --------------------------------------------------------------------------------------------------------------------------- base conventions
def test_validation_rejects_bad_specs():
    with pytest.raises(ValueError):
        Instrument(instrument_id="X", asset_class="spaceship")
    with pytest.raises(ValueError):
        Instrument(instrument_id="X", tick_size=0.0)
    with pytest.raises(ValueError):
        Instrument(instrument_id="")
    with pytest.raises(KeyError):
        Instrument(instrument_id="X", calendar="MARS")


def test_round_quantity_toward_zero_and_tick_value():
    inst = Instrument(instrument_id="X", lot_size=10.0, tick_size=0.25, contract_multiplier=50.0)
    assert inst.round_quantity(39.9) == 30.0 and inst.round_quantity(-39.9) == -30.0
    assert inst.round_quantity(9.99) == 0.0
    assert inst.tick_value == 12.5
    assert inst.round_price(100.13) == 100.25 and inst.round_price(100.12) == 100.0


def test_every_kind_round_trips_through_a_dict():
    exp = pd.Timestamp("2024-12-20")
    items = [fx_spot("EUR", "USD"), fx_forward("EUR", "USD", exp, 1.12), fx_swap("EUR", "USD", "2024-06-20", exp, 1.09, 1.12), crypto_spot("BINANCE", "BTC", "USDT"),
             crypto_perp("BINANCE", "BTC", "USDT"), crypto_future("BINANCE", "BTC", exp, "USDT"), make_option("SPY", exp, "put", 450.0),
             make_irs("USD", "2024-01-04", "2029-01-04", 0.04), build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), multiplier=50.0).contracts[0]]
    for it in items:
        back = instrument_from_dict(it.to_dict())
        assert back == it, it.instrument_id
        assert type(back) is type(it)


def test_registry_rejects_conflicting_duplicates_and_roundtrips(tmp_path):
    reg = InstrumentRegistry()
    spot = fx_spot("EUR", "USD")
    reg.add(spot)
    reg.add(spot)                                                              # identical: accepted (idempotent loading)
    with pytest.raises(ValueError):
        reg.add(fx_spot("EUR", "USD", lot_size=1.0))
    chain = build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), multiplier=50.0)
    reg.add_chain(chain)
    reg.add(make_option("ES", "2024-06-21", "call", 5000.0, underlying_kind="future"))
    path = tmp_path / "reg.json"
    reg.to_json(path)
    back = InstrumentRegistry.from_json(path)
    assert back.ids() == reg.ids() and back.is_chain("ES") and not back.is_chain(spot.instrument_id)
    assert back.get("EURUSD") == spot
    assert reg.validate() == []


def test_registry_validate_flags_missing_underlying():
    reg = InstrumentRegistry([make_option("NOPE", "2024-06-21", "call", 100.0)])
    assert any("underlying" in p for p in reg.validate())


# ---------------------------------------------------------------------------------------------------------------------------------- futures
def test_chain_expiries_and_front():
    chain = build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), expiry_rule="third_friday", multiplier=50.0, tick_size=0.25, roll=RollSpec("calendar", 5))
    c = {x.instrument_id: x for x in chain.contracts}
    assert c["ESH24"].expiry == pd.Timestamp("2024-03-15")                    # third Friday of March 2024
    assert c["ESM24"].expiry == pd.Timestamp("2024-06-21")
    front = chain.front("2024-03-01")
    assert front.instrument_id == "ESH24" and chain.front("2024-03-01", 1).instrument_id == "ESM24"
    assert chain.front("2024-03-16").instrument_id == "ESM24"                 # H24 has expired
    assert chain.roll_date(c["ESH24"]) == pd.Timestamp("2024-03-08")          # five business days before expiry
    sched = chain.roll_schedule()
    assert len(sched) >= 3


def test_future_pnl_and_inverse_perp_pnl():
    chain = build_chain("ES", "2024-01-01", "2024-06-30", months=(3, 6), multiplier=50.0, tick_size=0.25)
    f = chain.contracts[0]
    assert f.pnl(4000.0, 4010.0, 2) == 2 * 50.0 * 10.0
    inv = crypto_perp("DERIBIT", "BTC", "USD", contract_type="inverse")
    assert inv.inverse and inv.currency == "BTC"
    q = 10.0
    assert math.isclose(inv.pnl(20000.0, 25000.0, q), q * inv.contract_multiplier * (1 / 20000.0 - 1 / 25000.0))
    assert inv.pnl(20000.0, 25000.0, q) > 0                                    # a long gains when the price rises (in coin)


# -------------------------------------------------------------------------------------------------------------------------------- crypto
def test_perp_funding_sign_and_cap():
    perp = crypto_perp("BINANCE", "BTC", "USDT")
    pay = perp.funding_payment(2.0, 30000.0, 0.0001)
    assert pay < 0 and math.isclose(pay, -2.0 * perp.contract_multiplier * 30000.0 * 0.0001)      # longs pay when the rate is positive
    assert perp.funding_payment(-2.0, 30000.0, 0.0001) > 0                                           # shorts receive
    assert math.isclose(perp.funding_payment(1.0, 100.0, 5.0), -perp.contract_multiplier * 100.0 * perp.funding_rate_cap)   # capped


def test_perp_funding_times_and_liquidation_price():
    perp = crypto_perp("BINANCE", "BTC", "USDT")
    t = perp.funding_times("2024-01-01 00:00", "2024-01-02 00:00")
    assert list(t.hour) == [8, 16, 0]
    lq_long, lq_short = perp.liquidation_price(100.0, 1.0, 10.0), perp.liquidation_price(100.0, -1.0, 10.0)
    assert lq_long < 100.0 < lq_short
    assert math.isclose(lq_long, 100.0 * (1 - 0.1 + perp.maintenance_margin_rate))


def test_spot_and_dated_future_ids():
    assert crypto_spot("BINANCE", "ETH", "USDT").instrument_id == "BINANCE:ETH-USDT:SPOT"
    fut = crypto_future("BINANCE", "BTC", "2024-12-27", "USDT")
    assert fut.instrument_id.endswith(":FUT:20241227") and fut.expiry == pd.Timestamp("2024-12-27")
    with pytest.raises(KeyError):
        crypto_spot("NOSUCHVENUE", "BTC")


# --------------------------------------------------------------------------------------------------------------------------------- options
def test_option_payoff_exercise_and_margin():
    call = make_option("SPY", "2024-06-21", "call", 100.0)
    put = make_option("SPY", "2024-06-21", "put", 100.0)
    assert call.payoff(110.0, 2) == 2 * 100 * 10 and put.payoff(110.0, 2) == 0.0
    assert call.will_auto_exercise(100.5) and not call.will_auto_exercise(100.005)
    ex = call.exercise_settlement(1.0, 110.0)                                 # physical: buy 100 shares at 100
    assert ex["cash"] == {"USD": -100 * 100.0} and ex["delivery"] == ("SPY", 100.0, 100.0)
    ex = put.exercise_settlement(-1.0, 90.0)                                   # assigned short put: buy 100 shares at 100
    assert ex["cash"] == {"USD": -100 * 100.0} and ex["delivery"][1] == 100.0
    idx = make_option("SPX", "2024-06-21", "call", 4000.0, underlying_kind="index")
    assert idx.exercise_settlement(1.0, 4100.0) == {"cash": {"USD": 100 * 100.0}, "delivery": None}
    fut = make_option("ES", "2024-06-21", "call", 4000.0, underlying_kind="future")
    assert fut.exercise_settlement(1.0, 4100.0)["delivery"] == ("ES", 1.0, 4000.0)
    assert call.margin_requirement(1.0, 100.0, 5.0) == 0.0 and call.margin_requirement(-1.0, 100.0, 5.0) > 0


def test_option_greeks_and_validation():
    call = make_option("SPY", "2025-01-17", "call", 100.0)
    g = call.greeks(100.0, 0.2, "2024-07-17")
    assert 0.45 < g["delta"] < 0.65 and g["gamma"] > 0 and g["vega"] > 0 and g["theta"] < 0
    assert call.model_price(100.0, 0.2, "2024-07-17") > call.model_price(100.0, 0.1, "2024-07-17")
    with pytest.raises(ValueError):
        Option(instrument_id="bad", right="call", strike=-1.0, underlying_id="X", expiry=pd.Timestamp("2024-06-21"))


# ----------------------------------------------------------------------------------------------------------------------------------- FX
def test_fx_conventions_and_forward_settlement():
    spot = fx_spot("EUR", "USD")
    assert spot.pair == "EURUSD" and spot.cash_style == "currency_exchange" and spot.currency == "USD"
    assert fx_spot("USD", "JPY").price_precision == 3
    fwd = fx_forward("EUR", "USD", "2024-12-20", 1.10)
    assert fwd.settlement_amounts(1_000_000.0) == {"EUR": 1_000_000.0, "USD": -1_100_000.0}
    ndf = fx_forward("USD", "BRL", "2024-12-20", 5.0, deliverable=False)
    assert ndf.settlement_amounts(1_000.0, fixing=5.5) == {"BRL": 500.0}
    with pytest.raises(ValueError):
        ndf.settlement_amounts(1_000.0)
    assert math.isclose(fwd.pv_per_unit(1.12, 0.97), (1.12 - 1.10) * 0.97)


def test_fx_swap_legs_are_opposite_forwards():
    sw = fx_swap("EUR", "USD", "2024-06-20", "2024-12-20", 1.08, 1.11)
    legs = sw.legs(2.0)
    assert len(legs) == 2 and legs[0][1] == -legs[1][1] and math.isclose(sw.swap_points(), 0.03)


# ---------------------------------------------------------------------------------------------------------------------------------- swaps
def test_irs_fields_and_reverse():
    irs = make_irs("USD", "2024-01-04", "2029-01-04", 0.04, 5_000_000.0, pay_fixed=True)
    assert irs.instrument_type == "irs" and irs.cash_style == "otc_mtm"
    assert irs.notional(0.0, 2.0) == 2 * 5_000_000.0
    rev = irs.reversed()
    assert rev.pay_fixed != irs.pay_fixed and rev.fixed_rate == irs.fixed_rate
