"""The portfolio ledger on its own: arithmetic of each cash style, the journal identity, conservation checks and the reconciliation that catches tampering."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.instruments import Instrument, InstrumentRegistry, build_chain, crypto_perp, crypto_spot, fx_spot, make_option
from src.ledger import CurrencyConverter, Fill, Ledger, MarginModel
from src.ledger.reconcile import reconcile

T0 = pd.Timestamp("2024-01-02 16:00")


def make_ledger(extra=(), cash=None, pegs=None):
    reg = InstrumentRegistry()
    reg.add(Instrument(instrument_id="STK", asset_class="equity", instrument_type="equity", currency="USD"))
    reg.add_chain(build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), multiplier=50.0, tick_size=0.25, initial_margin=12000.0, maintenance_margin=10000.0))
    for i in extra:
        reg.add(i)
    led = Ledger(reg, "USD", None, CurrencyConverter("USD", pegs), 1e-9)
    for c, a in (cash or {"USD": 1_000_000.0}).items():
        led.deposit(T0, c, a)
    return reg, led


def test_equity_stock_full_payment_round_trip():
    _, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    assert led.cash["USD"] == 1_000_000 - 5_000 and led.equity() == 1_000_000
    led.revalue(T0 + pd.Timedelta(days=1), {"STK": 55.0})
    assert math.isclose(led.equity(), 1_000_500.0)
    led.apply_fill(T0 + pd.Timedelta(days=1), Fill(T0, "STK", -100.0, 55.0), mid=55.0)
    assert math.isclose(led.cash["USD"], 1_000_500.0) and led.quantity("STK") == 0.0


def test_future_has_no_entry_cost_and_settles_variation_margin_daily():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, fut, 2.0, 4000.0), mid=4000.0)
    assert led.cash["USD"] == 1_000_000.0                                      # nothing paid at entry
    led.revalue(T0 + pd.Timedelta(days=1), {fut: 4010.0})
    assert math.isclose(led.cash["USD"], 1_000_000.0 + 2 * 50.0 * 10.0)       # the gain is cash at once
    led.revalue(T0 + pd.Timedelta(days=2), {fut: 4005.0})
    assert math.isclose(led.cash["USD"], 1_000_000.0 + 2 * 50.0 * 5.0)


def test_costs_are_separate_journal_entries_and_equity_falls_by_exactly_the_costs():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.05, fee=1.0, spread_price=0.05), mid=50.0)      # buys 5 cents above the mid
    assert math.isclose(led.equity(), 1_000_000.0 - 100 * 0.05 - 1.0)
    cats = led.pnl_by("category")
    assert math.isclose(cats["spread"], -5.0) and math.isclose(cats["fee"], -1.0)


def test_spot_fx_balances_are_the_position_and_translation_is_tracked():
    reg, led = make_ledger([fx_spot("EUR", "USD")], cash={"USD": 1_000_000.0})
    led.converter.set_pair("EUR", "USD", 1.10)
    led.apply_fill(T0, Fill(T0, "EURUSD", 100_000.0, 1.10), mid=1.10)
    assert math.isclose(led.cash["EUR"], 100_000.0) and math.isclose(led.cash["USD"], 1_000_000.0 - 110_000.0)
    assert math.isclose(led.equity(), 1_000_000.0)
    led.revalue(T0 + pd.Timedelta(days=1), {"EURUSD": 1.15})
    assert math.isclose(led.equity(), 1_000_000.0 + 100_000 * 0.05)            # EUR is worth 5 cents more
    assert math.isclose(led.pnl_by("category").get("fx_translation", 0.0), 5_000.0)


def test_option_premium_is_paid_and_short_option_is_a_liability():
    opt = make_option("STK", "2024-06-21", "call", 50.0)
    reg, led = make_ledger([opt])
    led.apply_fill(T0, Fill(T0, opt.instrument_id, -2.0, 3.0), mid=3.0)       # sell two contracts at 3.00 -> receive 600
    assert math.isclose(led.cash["USD"], 1_000_600.0) and math.isclose(led.equity(), 1_000_000.0)
    led.revalue(T0 + pd.Timedelta(days=1), {opt.instrument_id: 3.5})
    assert math.isclose(led.equity(), 1_000_000.0 - 2 * 100 * 0.5)


def test_perp_funding_and_margin_currency():
    perp, spot = crypto_perp("BINANCE", "BTC", "USDT"), crypto_spot("BINANCE", "BTC", "USDT")
    reg, led = make_ledger([perp, spot], cash={"USDT": 100_000.0, "USD": 0.0}, pegs={"USDT": 1.0})
    led.apply_fill(T0, Fill(T0, perp.instrument_id, 1.0, 30_000.0), mid=30_000.0)
    led.post_cashflow(T0, "funding", "USDT", perp.funding_payment(1.0, 30_000.0, 0.0001), perp.instrument_id)
    assert math.isclose(led.cash["USDT"], 100_000.0 - 3.0)


def test_journal_identity_holds_for_every_entry_and_reconciles():
    reg, led = make_ledger([fx_spot("EUR", "USD")])
    fut = reg.chain("ES").contracts[0].instrument_id
    led.converter.set_pair("EUR", "USD", 1.1)
    led.apply_fill(T0, Fill(T0, "STK", 10.0, 100.0, fee=1.0), mid=100.0)
    led.apply_fill(T0, Fill(T0, fut, -1.0, 4000.0, fee=2.0), mid=4000.0)
    led.apply_fill(T0, Fill(T0, "EURUSD", 50_000.0, 1.1), mid=1.1)
    for d, px in enumerate((101.0, 99.0, 103.0), start=1):
        t = T0 + pd.Timedelta(days=d)
        led.revalue(t, {"STK": px, fut: 4000.0 - 10 * d, "EURUSD": 1.1 + 0.01 * d})
    rep = reconcile(led, expected_quantities={"STK": 10.0, fut: -1.0, "EURUSD": 50_000.0})
    assert rep.ok, rep.failures()
    for e in led.journal:
        assert abs(e.cash_base + e.value + e.translation - e.pnl - e.transfer) < 1e-6


def test_reconcile_catches_tampering():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 10.0, 100.0), mid=100.0)
    assert reconcile(led, expected_quantities={"STK": 10.0}).ok
    led.cash["USD"] += 1.0                                                      # money from nowhere
    rep = reconcile(led, expected_quantities={"STK": 10.0})
    assert not rep.ok and any("cash[USD]" in f for f in rep.failures())
    led.cash["USD"] -= 1.0
    led.positions["STK"].quantity += 1.0                                        # a share from nowhere
    assert not reconcile(led, expected_quantities={"STK": 10.0}).ok


def test_expired_contracts_must_not_be_held():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0]
    led.apply_fill(T0, Fill(T0, fut.instrument_id, 1.0, 4000.0), mid=4000.0)
    rep = reconcile(led, expected_quantities={fut.instrument_id: 1.0}, ts=fut.expiry + pd.Timedelta(days=2))
    assert fut.instrument_id in rep.expired_held and not rep.ok


def test_no_conversion_path_is_an_error_not_a_guess():
    conv = CurrencyConverter("USD")
    with pytest.raises(KeyError, match="no conversion path"):
        conv.rate("EUR")
    conv.set_pair("EUR", "GBP", 0.85)
    conv.set_pair("GBP", "USD", 1.25)
    assert math.isclose(conv.rate("EUR"), 0.85 * 1.25)                          # a path through GBP
    assert math.isclose(conv.convert(100.0, "GBP"), 125.0)


def test_margin_model_requirements():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, fut, 3.0, 4000.0), mid=4000.0)
    led.revalue(T0, {fut: 4000.0})
    st = MarginModel().state(led, {fut: 4000.0})
    assert math.isclose(st.initial_margin, 3 * 12000.0) and math.isclose(st.maintenance_margin, 3 * 10000.0)
    assert st.equity == led.equity() and not st.margin_call
    led.cash["USD"] = 20_000.0                                                   # equity below the maintenance requirement
    assert MarginModel().state(led, {fut: 4000.0}).margin_call


def test_unknown_instrument_and_closing_a_position():
    reg, led = make_ledger()
    with pytest.raises(KeyError):
        led.apply_fill(T0, Fill(T0, "NOPE", 1.0, 1.0), mid=1.0)
    led.apply_fill(T0, Fill(T0, "STK", 10.0, 100.0), mid=100.0)
    led.close_position(T0 + pd.Timedelta(days=1), "STK", 110.0)
    assert led.quantity("STK") == 0.0 and math.isclose(led.cash["USD"], 1_000_000.0 + 100.0)
