"""Release 4, fixtures that reconcile the platform's ledger against an independent shadow computation (spec 7.3): a multi-currency book, a futures roll and perpetual funding, long and short.

The shadow is plain arithmetic on native balances, settlement prices and published rates written for the test; it shares no code with the ledger or with ``realism``. Where the production helpers
(``FXBook``, ``executable_contract``, ``roll_events``, ``funding_payment``, ``FeeSchedule``) are exercised they are compared with that arithmetic."""

import numpy as np
import pandas as pd
import pytest

from src.instruments import Instrument, InstrumentRegistry, build_chain, crypto_perp, fx_spot
from src.ledger import CurrencyConverter, Fill, Ledger, reconcile
from src.ledger.ledger import LedgerError
from src.ml import realism as rz

TOL = 1e-6


# --------------------------------------------------------------------------------------------------------------------- P1-12: a multi-currency book
def test_a_multi_currency_book_reconciles_to_native_balances_translated_at_the_rate_in_force():
    pair = fx_spot("EUR", "USD", lot_size=1.0)
    sap = Instrument(instrument_id="SAP", asset_class="equity", instrument_type="equity", currency="EUR")
    led = Ledger(InstrumentRegistry([pair, sap]), "USD", None, CurrencyConverter("USD", None), 1e-9)
    days = pd.bdate_range("2023-03-01", periods=6)
    close = [d + pd.Timedelta(hours=16) for d in days]
    # EUR/USD is observed at 11:00 on some days only: the closes of the other days must use the last rate seen, never a later one
    seen = {0: 1.10, 1: 1.12, 3: 1.08, 5: 1.09}
    book = rz.FXBook("USD", spread_bps=0.0, settlement_days={"EUR": 2})
    book.set_series("EUR", pd.Series(list(seen.values()), index=[days[i] + pd.Timedelta(hours=11) for i in seen]))
    sap_px = [100.0, 104.0, 103.0, 107.0, 105.0, 106.0]
    led.deposit(close[0], "USD", 200_000.0)
    # shadow account: native balances and shares, nothing else
    usd, eur, shares = 200_000.0, 0.0, 0.0
    shadow, rates_used, eur_net_before = [], [], []

    def shadow_equity(i):
        r = book.rate_at("EUR", close[i])
        return usd + r * (eur + shares * sap_px[i]), r

    for i, ts in enumerate(close):
        eur_net_before.append(eur + shares * (sap_px[i - 1] if i else 0.0))
        marks = {"SAP": sap_px[i]}
        if i in seen:
            marks["EURUSD"] = seen[i]
        if i == 0:
            led.converter.set_pair("EUR", "USD", seen[0])
            led.apply_fill(ts, Fill(ts, "EURUSD", 10_000.0, seen[0]), mid=seen[0])                  # buy 10,000 EUR with 11,000 USD
            usd, eur = usd - 10_000.0 * seen[0], eur + 10_000.0
            led.apply_fill(ts, Fill(ts, "SAP", 100.0, sap_px[0]), mid=sap_px[0])                    # 100 shares for 10,000 EUR
            eur, shares = eur - 100.0 * sap_px[0], shares + 100.0
        else:
            led.revalue(ts, marks)
        if i == 3:
            led.apply_fill(ts, Fill(ts, "SAP", -50.0, sap_px[3]), mid=sap_px[3])                    # sell half: EUR 5,350
            eur, shares = eur + 50.0 * sap_px[3], shares - 50.0
            led.apply_fill(ts, Fill(ts, "EURUSD", -5_350.0, seen[3]), mid=seen[3])                  # and convert it back at the rate in force
            usd, eur = usd + 5_350.0 * seen[3], eur - 5_350.0
        eq, r = shadow_equity(i)
        shadow.append(eq)
        rates_used.append(r)
        assert led.equity() == pytest.approx(eq, abs=TOL), f"day {i}"
        assert led.cash["USD"] == pytest.approx(usd, abs=TOL) and led.cash.get("EUR", 0.0) == pytest.approx(eur, abs=TOL)
    assert rates_used == [1.10, 1.12, 1.12, 1.08, 1.08, 1.09], "a close with no fresh observation uses the last one seen"
    # the translation line is the change of rate times the native net assets held across it: computed here from the shadow's own balances
    expected_translation = sum(eur_net_before[i] * (rates_used[i] - rates_used[i - 1]) for i in range(1, len(close)))
    assert led.pnl_by("category").get("fx_translation", 0.0) == pytest.approx(expected_translation, abs=1e-4)
    assert expected_translation != 0.0
    # history is not re-translated at the final rate: the day-0 purchase of SAP was booked at 1.10
    j = led.journal_frame()
    buy = j[(j["instrument_id"] == "SAP") & (j["category"] == "trade")].iloc[0]
    assert buy["cash_base"] == pytest.approx(-10_000.0 * 1.10), "EUR 10,000 paid, translated at the rate of the day, not at 1.09"
    rep = reconcile(led, expected_quantities={"SAP": 50.0, "EURUSD": 4_650.0})                  # 10,000 EUR bought, 5,350 sold
    assert rep.ok, rep.failures()
    # the settlement calendar: T+2 across a weekend, and what is in transit on a given day
    assert book.settle_date("EUR", "2023-03-03", pd.bdate_range("2023-03-01", periods=10)) == pd.Timestamp("2023-03-07")
    trades = [{"ccy": "EUR", "amount": -5_350.0, "trade_date": days[3]}]
    assert book.unsettled(trades, pd.bdate_range("2023-03-01", periods=10), days[4]) == {"EUR": -5_350.0} and book.unsettled(trades, pd.bdate_range("2023-03-01", periods=10), days[5]) == {}


def test_a_missing_rate_is_an_error_everywhere_and_a_later_rate_is_never_used():
    book = rz.FXBook("USD")
    book.set_series("EUR", pd.Series([1.10], index=[pd.Timestamp("2023-03-02 11:00")]))
    with pytest.raises(KeyError, match="a later rate is not used"):
        book.rate_at("EUR", "2023-03-01 16:00")
    with pytest.raises(KeyError, match="not guessed"):
        book.rate_at("JPY", "2023-03-02")
    sap = Instrument(instrument_id="SAP", asset_class="equity", instrument_type="equity", currency="EUR")
    led = Ledger(InstrumentRegistry([sap]), "USD", None, CurrencyConverter("USD", None), 1e-9)
    led.deposit(pd.Timestamp("2023-03-01"), "USD", 1_000.0)
    with pytest.raises(KeyError):
        led.apply_fill(pd.Timestamp("2023-03-02 16:00"), Fill(pd.Timestamp("2023-03-02 16:00"), "SAP", 1.0, 100.0), mid=100.0)             # no EUR/USD pair has ever been observed
    with pytest.raises(LedgerError, match="no rate for JPY"):
        led.deposit(pd.Timestamp("2023-03-02"), "JPY", 5_000.0)
    assert book.convert(1_000.0, "EUR", "2023-03-02 12:00")["rate"] == 1.10


# --------------------------------------------------------------------------------------------------------------------- P1-13: a futures roll
@pytest.mark.parametrize("held", [2.0, -2.0])
def test_a_futures_position_rolls_when_the_nearby_stops_being_executable_and_the_basis_is_not_pnl(held):
    chain = build_chain("ES", "2023-01-01", "2023-12-31", months=(3, 6, 9, 12), multiplier=50.0, tick_size=0.25, initial_margin=12000.0, maintenance_margin=10000.0)
    front, nxt = "ESH23", "ESM23"
    specs = [rz.FuturesSpec(c.instrument_id, c.contract_multiplier, c.initial_margin, c.maintenance_margin, c.expiry) for c in chain.contracts]
    led = Ledger(InstrumentRegistry([c for c in chain.contracts]), "USD", None, CurrencyConverter("USD", None), 1e-9)
    days = pd.bdate_range("2023-02-27", "2023-03-24")
    rng = np.random.default_rng(11)
    settle_h = pd.Series(4000.0 + np.cumsum(rng.normal(0, 8, len(days))), index=days)
    basis = pd.Series(12.0 + rng.normal(0, 0.5, len(days)), index=days)                                  # ESM trades above ESH: the roll gap
    settle_m = settle_h + basis
    events = rz.roll_events(specs, days, 5)
    assert [(e["date"], e["close"], e["open"]) for e in events] == [(pd.Timestamp("2023-03-13"), front, nxt)]
    roll_day = events[0]["date"]
    start, fee = 500_000.0, 2.0
    led.deposit(days[0], "USD", start)
    cash, contract, prev = start, front, None                                                           # the shadow account: a number and the contract held
    mult = 50.0
    variation = 0.0
    for d in days:
        t = d + pd.Timedelta(hours=16)
        px = {front: float(settle_h[d]), nxt: float(settle_m[d])}
        if prev is not None:
            held_contract = contract
            dv = held * mult * (px[held_contract] - prev[held_contract])                               # the day's settlement on the contract held overnight
            cash, variation = cash + dv, variation + dv
            led.revalue(t, {held_contract: px[held_contract]})
        if d == days[0]:
            led.apply_fill(t, Fill(t, front, held, px[front], fee=fee * abs(held)), mid=px[front])
            cash -= fee * abs(held)
        if d == roll_day:
            led.apply_fill(t, Fill(t, front, -held, px[front], fee=fee * abs(held)), mid=px[front])    # close ESH at its settlement ...
            led.apply_fill(t, Fill(t, nxt, held, px[nxt], fee=fee * abs(held)), mid=px[nxt])           # ... and open ESM at ITS settlement; the 12-point gap between the two is not a gain
            cash -= 2 * fee * abs(held)
            contract = nxt
        prev = px
        assert led.equity() == pytest.approx(cash, abs=TOL) and led.cash["USD"] == pytest.approx(cash, abs=TOL), f"{d.date()}"
    assert led.quantity(front) == 0.0 and led.quantity(nxt) == held
    assert led.pnl_by("category")["variation_margin"] == pytest.approx(variation, abs=TOL)
    # on the roll day the equity moved by the day's variation on the OLD contract less the two fees, and not by the basis
    j = led.journal_frame()
    on_roll = j[pd.to_datetime(j["ts"]).dt.normalize() == roll_day]
    assert on_roll[on_roll["category"].isin(["variation_margin", "fee", "spread", "impact", "slippage"])]["pnl"].sum() == pytest.approx(
        held * mult * (settle_h[roll_day] - settle_h[days[list(days).index(roll_day) - 1]]) - 2 * fee * abs(held), abs=TOL)
    rep = reconcile(led, expected_quantities={nxt: held})
    assert rep.ok, rep.failures()
    # a continuous series is not a contract: the executable one changes on the roll date and the position is never in a contract past it
    assert rz.executable_contract(specs, days[0], 5) == front and rz.executable_contract(specs, roll_day, 5) == nxt


# --------------------------------------------------------------------------------------------------------------------- P1-13: perpetual funding
@pytest.mark.parametrize("held", [2.0, -2.0])
def test_perpetual_funding_is_paid_by_longs_when_the_rate_is_positive_and_the_schedule_and_cap_are_the_venue_s(held):
    perp = crypto_perp("BINANCE", "BTC", "USDT")
    led = Ledger(InstrumentRegistry([perp]), "USD", None, CurrencyConverter("USD", {"USDT": 1.0}), 1e-9)
    start_ts, end_ts = pd.Timestamp("2023-03-01 00:00"), pd.Timestamp("2023-03-03 00:00")
    times = perp.funding_times(start_ts, end_ts)
    assert len(times) == 6 and list(times.hour[:3]) == [8, 16, 0], "three funding times a day on the 8-hour grid"
    rates = [0.0001, -0.0002, 0.0003, 0.0100, -0.0100, 0.0]                                          # 0.0100 is above the venue's cap of 0.0075
    marks = [30_000.0, 30_150.0, 29_900.0, 30_400.0, 30_050.0, 30_000.0]
    led.deposit(start_ts, "USDT", 100_000.0)
    led.apply_fill(start_ts, Fill(start_ts, perp.instrument_id, held, 30_000.0), mid=30_000.0)
    cash, prev, funding_total = 100_000.0, 30_000.0, 0.0
    size, cap = 1.0, 0.0075
    for ts, r, m in zip(times, rates, marks):
        led.revalue(ts, {perp.instrument_id: m})
        led.post_cashflow(ts, "funding", "USDT", perp.funding_payment(held, m, r), perp.instrument_id)
        pay = -held * size * m * float(np.clip(r, -cap, cap))                                         # longs pay shorts when the rate is positive; a rate beyond the cap is capped
        cash += held * size * (m - prev) + pay
        funding_total += pay
        prev = m
        assert led.cash["USDT"] == pytest.approx(cash, abs=TOL) and led.equity() == pytest.approx(cash, abs=TOL)
    assert led.pnl_by("category")["funding"] == pytest.approx(funding_total, abs=TOL)
    # signs: with the first (positive) rate the long paid and the short received
    first = -held * size * 30_000.0 * 0.0001
    assert (first < 0) == (held > 0)
    assert rz.funding_payment(held, 30_000.0, 0.0001) == pytest.approx(perp.funding_payment(held, 30_000.0, 0.0001)), "the helper and the instrument agree below the cap"
    assert perp.funding_payment(held, 30_000.0, 0.0100) == pytest.approx(perp.funding_payment(held, 30_000.0, 0.0075)), "above the venue cap the rate is capped"
    rep = reconcile(led, expected_quantities={perp.instrument_id: held})
    assert rep.ok, rep.failures()


def test_fee_tiers_are_applied_from_trailing_volume_and_are_never_guessed():
    fees = rz.FeeSchedule(((0, 2.0, 5.0), (1e6, 1.6, 4.0), (5e6, -0.5, 3.0)), venue="example", source="assumed")
    assert fees.rate_bps(0.0, maker=False) == 5.0 and fees.rate_bps(999_999.0, maker=False) == 5.0 and fees.rate_bps(1e6, maker=False) == 4.0 and fees.rate_bps(9e6, maker=True) == -0.5
    assert fees.fee(100_000.0, 2e6, maker=False) == pytest.approx(40.0) and fees.fee(100_000.0, 9e6, maker=True) == pytest.approx(-5.0), "a maker rebate is a negative fee"
    with pytest.raises(ValueError, match="not guessed"):
        fees.rate_bps(float("nan"), maker=True)
    with pytest.raises(ValueError):
        rz.FeeSchedule(((1e6, 1.0, 2.0),))
    with pytest.raises(ValueError):
        rz.FeeSchedule(((0, 1.0, 2.0), (0, 1.0, 2.0)))
