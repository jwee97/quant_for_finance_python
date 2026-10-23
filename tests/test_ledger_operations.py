"""The ledger's operations one at a time, each against a figure worked out by hand: average cost through partial closes and a flip, the split of the cost of crossing the market, a stock split,
interest and borrowing, settlement, exposures, multi-currency equity, and the guards that reject an entry that does not balance.

A planted-bug audit (``experiments/plant_bugs.py``) of ``src/ledger/ledger.py`` found that the ledger's own tests and the engine's noticed only 42% of the changes made to it: the financing,
borrow, split, exposure and settlement paths were reached by a run but their figures were not checked. These tests check them."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.instruments import Instrument, fx_spot
from src.ledger import CurrencyConverter, Fill, Ledger
from src.ledger.ledger import LedgerError
from tests.test_ledger import T0, make_ledger

DAY = pd.Timedelta(days=1)


def cats(led: Ledger) -> dict:
    return led.pnl_by("category").to_dict()


# ------------------------------------------------------------------------------------------------------------------------ average cost
def test_average_cost_runs_through_adds_partial_closes_and_a_flip_and_the_realised_profit_adds_up():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 60.0), mid=60.0)
    pos = led.positions["STK"]
    assert pos.quantity == 200.0 and pos.avg_price == pytest.approx(55.0) and pos.opened_at == pd.Timestamp(T0)
    led.apply_fill(T0 + DAY, Fill(T0, "STK", -50.0, 70.0), mid=70.0)                                      # sells 50 against an average of 55
    assert pos.quantity == 150.0 and pos.avg_price == pytest.approx(55.0) and pos.realized_pnl == pytest.approx(750.0)
    led.apply_fill(T0 + 2 * DAY, Fill(T0, "STK", -250.0, 80.0), mid=80.0)                                 # closes the 150 and goes 100 short at 80
    assert pos.quantity == -100.0 and pos.avg_price == pytest.approx(80.0) and pos.realized_pnl == pytest.approx(750.0 + 150 * 25.0)
    led.apply_fill(T0 + 3 * DAY, Fill(T0, "STK", 40.0, 76.0), mid=76.0)                                   # covers 40 of the short at 4 below its average
    assert pos.quantity == -60.0 and pos.avg_price == pytest.approx(80.0) and pos.realized_pnl == pytest.approx(750.0 + 3750.0 + 40 * 4.0)
    led.apply_fill(T0 + 4 * DAY, Fill(T0, "STK", 60.0, 75.0), mid=75.0)                                   # covers the rest at 5 below
    assert "STK" not in led.positions and led.closed[-1] is pos and pos.avg_price == 0.0
    assert pos.realized_pnl == pytest.approx(750.0 + 3750.0 + 160.0 + 300.0)
    assert led.equity() == pytest.approx(1_000_000.0 + pos.realized_pnl)                                    # nothing else made or lost money: the mids were the fills


def test_adding_to_a_short_averages_the_entry_price_by_size():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", -100.0, 50.0), mid=50.0)
    led.apply_fill(T0, Fill(T0, "STK", -300.0, 54.0), mid=54.0)
    assert led.positions["STK"].quantity == -400.0 and led.positions["STK"].avg_price == pytest.approx((100 * 50.0 + 300 * 54.0) / 400.0)
    assert led.positions["STK"].fees_paid == 0.0


def test_a_fill_of_nothing_writes_nothing():
    reg, led = make_ledger()
    before = len(led.journal)
    led.apply_fill(T0, Fill(T0, "STK", 0.0, 50.0), mid=50.0)
    assert len(led.journal) == before and not led.fills and "STK" not in led.positions


# ------------------------------------------------------------------------------------------------------------------------ what crossing the market cost
@pytest.mark.parametrize("quantity, price", [(100.0, 50.10), (-100.0, 49.90)])
def test_the_distance_from_the_mid_is_split_into_spread_impact_and_the_slippage_that_is_left(quantity, price):
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", quantity, price, fee=0.0, spread_price=0.05, impact_price=0.03), mid=50.0)
    c = cats(led)
    assert c["spread"] == pytest.approx(-5.0) and c["impact"] == pytest.approx(-3.0) and c["slippage"] == pytest.approx(-2.0)        # 100 shares at 5, 3 and 2 cents
    assert led.cash["USD"] == pytest.approx(1_000_000.0 - quantity * 50.0 - 10.0)                           # the principal at the mid, then 10 of cost either way
    assert led.equity() == pytest.approx(1_000_000.0 - 10.0)


def test_a_delivery_at_a_contractual_price_books_the_whole_gap_as_lifecycle_profit_and_not_as_costs():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.10, spread_price=0.05, impact_price=0.03, settlement=True), mid=50.0)
    c = cats(led)
    assert c["lifecycle_settlement"] == pytest.approx(-10.0) and not {"spread", "impact", "slippage"} & set(c)


def test_a_fee_in_another_currency_is_paid_from_that_currencys_balance_and_valued_at_its_rate():
    reg, led = make_ledger(cash={"USD": 1_000_000.0, "EUR": 10_000.0}, pegs={"EUR": 1.10})
    led.apply_fill(T0, Fill(T0, "STK", 10.0, 100.0, fee=20.0, fee_currency="EUR"), mid=100.0)
    assert led.cash["EUR"] == pytest.approx(9_980.0) and led.cash["USD"] == pytest.approx(1_000_000.0 - 1_000.0)
    assert cats(led)["fee"] == pytest.approx(-22.0) and led.positions["STK"].fees_paid == 20.0


def test_a_fee_is_in_the_instruments_currency_when_none_is_named():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 10.0, 100.0, fee=3.0), mid=100.0)
    assert cats(led)["fee"] == -3.0 and led.cash["USD"] == pytest.approx(1_000_000.0 - 1_000.0 - 3.0)


# ------------------------------------------------------------------------------------------------------------------------ a stock split
def test_a_split_scales_the_quantity_up_and_the_prices_down_and_leaves_the_value_alone():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    led.revalue(T0 + DAY, {"STK": 55.0})
    equity = led.equity()
    led.split(T0 + DAY, "STK", 2.0)
    pos = led.positions["STK"]
    assert pos.quantity == 200.0 and pos.avg_price == pytest.approx(25.0) and pos.last_mark == pytest.approx(27.5) and pos.last_value == pytest.approx(5_500.0)
    assert led.equity() == pytest.approx(equity)
    entry = led.journal[-1]
    assert entry.category == "corporate_action" and entry.pnl == 0.0 and entry.note == "split 2:1" and entry.instrument_id == "STK"
    led.revalue(T0 + 2 * DAY, {"STK": 28.0})                                                                  # the next mark is in the new units
    assert led.equity() == pytest.approx(1_000_000.0 + 200 * 28.0 - 5_000.0)
    led.split(T0 + 3 * DAY, "STK", 0.5, note="reverse")
    assert led.positions["STK"].quantity == 100.0 and led.journal[-1].note == "reverse"


def test_a_split_that_means_nothing_is_ignored():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    entries = len(led.journal)
    for ratio in (0.0, -2.0):
        led.split(T0, "STK", ratio)
    led.split(T0, "NOT-HELD", 2.0)
    assert led.positions["STK"].quantity == 100.0 and len(led.journal) == entries


# ------------------------------------------------------------------------------------------------------------------------ interest, financing and borrow
def test_balances_earn_the_deposit_rate_and_debts_pay_the_borrow_rate_which_defaults_to_the_deposit_rate_plus_fifty_basis_points():
    reg, led = make_ledger(cash={"USD": 100_000.0, "GBP": -50_000.0, "JPY": 0.0}, pegs={"GBP": 1.25, "JPY": 0.0067})
    led.accrue_interest(T0 + DAY, 0.5, {"USD": 0.04, "GBP": 0.03})
    assert led.cash["USD"] == pytest.approx(100_000.0 + 2_000.0)                                           # 100,000 * 4% * half a year
    assert led.cash["GBP"] == pytest.approx(-50_000.0 - 50_000.0 * 0.035 * 0.5)                           # the debt grows at 3% + 0.5%
    assert cats(led)["interest"] == pytest.approx(2_000.0) and cats(led)["financing"] == pytest.approx(-875.0 * 1.25)
    assert [e.note for e in led.journal if e.category in ("interest", "financing")] == ["USD deposit", "GBP borrow"]
    led.accrue_interest(T0 + 2 * DAY, 1.0, {"USD": 0.0, "GBP": 0.03}, {"GBP": 0.08})                       # an explicit borrow rate replaces the default; a zero rate writes nothing
    assert led.cash["GBP"] == pytest.approx(-50_875.0 - 50_875.0 * 0.08)
    assert sum(e.category == "interest" for e in led.journal) == 1


def test_a_short_pays_the_borrow_fee_on_its_notional_and_only_a_short_with_a_rate_in_a_cash_style_that_borrows():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, "STK", -100.0, 50.0), mid=50.0)
    led.apply_fill(T0, Fill(T0, fut, -1.0, 4000.0), mid=4000.0)
    led.accrue_borrow(T0 + DAY, 0.25, {"STK": 0.02, fut: 0.05}, {"STK": 50.0, fut: 4000.0})
    assert cats(led)["borrow"] == pytest.approx(-25.0) and led.positions["STK"].funding_paid == pytest.approx(-25.0)          # 5,000 * 2% * a quarter; a future is not borrowed
    assert led.positions[fut].funding_paid == 0.0
    led.accrue_borrow(T0 + 2 * DAY, 0.25, {"STK": 0.02}, {"STK": 60.0})                                       # the mark of the day sets the notional
    assert cats(led)["borrow"] == pytest.approx(-25.0 - 30.0)
    led.accrue_borrow(T0 + 3 * DAY, 0.25, {}, {"STK": 60.0})                                                  # no rate for it: no fee
    assert cats(led)["borrow"] == pytest.approx(-55.0)
    led.apply_fill(T0 + 3 * DAY, Fill(T0, "STK", 300.0, 60.0), mid=60.0)                                      # now long 200
    led.accrue_borrow(T0 + 4 * DAY, 0.25, {"STK": 0.02}, {"STK": 60.0})
    assert cats(led)["borrow"] == pytest.approx(-55.0)


def test_a_short_with_no_mark_at_all_pays_no_borrow_fee():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", -100.0, 50.0), mid=50.0)
    led.positions["STK"].last_mark = float("nan")
    led.accrue_borrow(T0 + DAY, 0.25, {"STK": 0.02}, {})
    assert "borrow" not in cats(led)
    led.accrue_borrow(T0 + DAY, 0.25, {"STK": 0.02}, {"STK": float("nan")})
    assert "borrow" not in cats(led)


# ------------------------------------------------------------------------------------------------------------------------ settlement
def test_closing_a_share_position_pays_the_value_at_the_settlement_price_and_files_the_position_away():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    equity = led.equity()
    led.close_position(T0 + DAY, "STK", 60.0, note="called away")
    assert "STK" not in led.positions and led.cash["USD"] == pytest.approx(1_000_000.0 - 5_000.0 + 6_000.0)
    pos = led.closed[-1]
    assert pos.quantity == 0.0 and pos.realized_pnl == pytest.approx(1_000.0) and led.quantity("STK") == 0.0
    assert led.equity() == pytest.approx(equity + 1_000.0)                                                    # the move to 60 was profit; settling it in cash changed nothing more
    entry = led.journal[-1]
    assert entry.category == "lifecycle_settlement" and entry.note == "called away" and entry.pnl == pytest.approx(0.0)
    assert cats(led)["mtm"] == pytest.approx(1_000.0)
    led.close_position(T0 + 2 * DAY, "STK", 70.0)                                                             # nothing held: nothing happens
    assert led.journal[-1] is entry                                                                           # and nothing was written for it


def test_closing_a_future_at_its_final_price_settles_the_last_variation_margin_and_pays_nothing_more():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, fut, 2.0, 4000.0), mid=4000.0)
    led.close_position(T0 + DAY, fut, 4010.0)
    assert fut not in led.positions and led.cash["USD"] == pytest.approx(1_000_000.0 + 2 * 50.0 * 10.0)
    assert led.journal[-1].category == "lifecycle_settlement" and led.journal[-1].note == "position removed at settlement" and led.journal[-1].cash == {}
    assert led.closed[-1].realized_pnl == pytest.approx(1_000.0)


def test_a_currency_position_cannot_be_settled_like_a_share_it_is_traded_out():
    reg, led = make_ledger([fx_spot("EUR", "USD")])
    led.converter.set_pair("EUR", "USD", 1.10)
    led.apply_fill(T0, Fill(T0, "EURUSD", 10_000.0, 1.10), mid=1.10)
    with pytest.raises(LedgerError, match="trade it out"):
        led.close_position(T0 + DAY, "EURUSD", 1.12)


def test_exercise_removes_the_value_without_cash_and_says_how_much_was_held():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    assert led.remove_position_value(T0 + DAY, "STK", 55.0) == 100.0
    assert "STK" not in led.positions and led.cash["USD"] == pytest.approx(1_000_000.0 - 5_000.0)             # no cash came back
    assert led.equity() == pytest.approx(1_000_000.0 - 5_000.0)                                               # the value left the books: it reappears as whatever was delivered
    assert led.journal[-1].note == "converted by exercise" and led.closed[-1].realized_pnl == pytest.approx(500.0)
    assert led.remove_position_value(T0 + 2 * DAY, "STK", 55.0) == 0.0


def test_a_position_settled_for_stated_cash_books_the_difference_to_its_value_as_profit():
    reg, led = make_ledger()
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    removed = led.settle_with_cash(T0 + DAY, "STK", {"USD": 5_400.0}, mark=55.0, note="delivered")
    assert removed == 100.0 and "STK" not in led.positions and led.cash["USD"] == pytest.approx(1_000_000.0 - 5_000.0 + 5_400.0)
    entry = led.journal[-1]
    assert entry.category == "lifecycle_settlement" and entry.note == "delivered" and entry.pnl == pytest.approx(-100.0)       # worth 5,500, paid 5,400
    assert led.closed[-1].realized_pnl == pytest.approx(500.0)
    assert led.settle_with_cash(T0 + 2 * DAY, "STK", {"USD": 1.0}, mark=1.0) == 0.0


# ------------------------------------------------------------------------------------------------------------------------ what is held
def test_exposures_are_the_notional_in_the_base_currency_with_its_sign_and_the_profit_to_date():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, "STK", 100.0, 50.0), mid=50.0)
    led.apply_fill(T0, Fill(T0, fut, -2.0, 4000.0), mid=4000.0)
    marks = {"STK": 55.0, fut: 4010.0}
    frame = led.exposures(marks)
    assert set(frame.index) == {"STK", fut}
    assert frame.loc["STK", "notional_base"] == pytest.approx(5_500.0) and frame.loc["STK", "signed_notional_base"] == pytest.approx(5_500.0)
    assert frame.loc[fut, "notional_base"] == pytest.approx(2 * 50.0 * 4010.0) and frame.loc[fut, "signed_notional_base"] == pytest.approx(-2 * 50.0 * 4010.0)
    assert frame.loc["STK", "unrealized_pnl"] == pytest.approx(0.0) and frame.loc["STK", "quantity"] == 100.0 and frame.loc[fut, "cash_style"] == "variation_margin"
    assert led.gross_exposure(marks) == pytest.approx(5_500.0 + 401_000.0) and led.net_exposure(marks) == pytest.approx(5_500.0 - 401_000.0)
    led.revalue(T0 + DAY, marks)
    assert led.exposures({}).loc["STK", "unrealized_pnl"] == pytest.approx(500.0) and led.exposures({}).loc["STK", "mark"] == 55.0          # the last mark is used where none is given
    snap = led.snapshot(T0 + DAY, marks)
    assert snap["gross_exposure"] == pytest.approx(406_500.0) and snap["net_exposure"] == pytest.approx(-395_500.0) and snap["n_positions"] == 2 and snap["equity"] == pytest.approx(led.equity())
    assert "gross_exposure" not in led.snapshot(T0 + DAY)


def test_a_position_without_a_mark_has_no_notional_and_an_empty_book_has_no_exposure():
    reg, led = make_ledger()
    assert led.exposures({}).empty and led.gross_exposure({}) == 0.0 and led.net_exposure({}) == 0.0 and led.snapshot(T0, {})["gross_exposure"] == 0.0
    led.apply_fill(T0, Fill(T0, "STK", -10.0, 50.0), mid=50.0)
    led.positions["STK"].last_mark = float("nan")
    frame = led.exposures({})
    assert math.isnan(frame.loc["STK", "notional_base"]) and frame.loc["STK", "unrealized_pnl"] == 0.0
    assert led.exposures({"STK": 40.0}).loc["STK", "signed_notional_base"] == pytest.approx(-400.0)


def test_equity_is_every_currencys_cash_and_position_value_at_its_rate():
    reg, led = make_ledger(cash={"USD": 1_000_000.0, "EUR": 100_000.0}, pegs={"EUR": 1.10})
    led.apply_fill(T0, Fill(T0, "STK", 10.0, 100.0), mid=100.0)
    assert led.net_assets() == pytest.approx({"USD": 1_000_000.0, "EUR": 100_000.0}) and led.cash_base() == pytest.approx(1_000_000.0 - 1_000.0 + 110_000.0)
    led.revalue(T0 + DAY, {"STK": 110.0})
    assert led.net_assets() == pytest.approx({"USD": 1_000_000.0 - 1_000.0 + 1_100.0, "EUR": 100_000.0})
    assert led.equity() == pytest.approx(1_000_000.0 + 100.0 + 110_000.0)
    assert led.cash_base() == pytest.approx(1_000_000.0 - 1_000.0 + 110_000.0)                              # the position is not cash
    led.cash["XXX"] = 0.0
    assert led.equity() == pytest.approx(1_110_100.0)                                                       # a zero balance in a currency with no rate is nothing, not an error
    led.cash["XXX"] = 1.0
    with pytest.raises(KeyError, match="no conversion path"):
        led.equity()
    assert led.position_value("STK") == pytest.approx(1_100.0) and led.position_value("NOT-HELD") == 0.0


# ------------------------------------------------------------------------------------------------------------------------ deposits, and the guards on every entry
def test_a_deposit_is_a_transfer_and_not_profit_and_a_currency_with_no_rate_cannot_be_deposited():
    reg, led = make_ledger()
    led.deposit(T0 + DAY, "USD", 50_000.0)
    led.deposit(T0 + DAY, "USD", -20_000.0, note="withdrawal")
    assert led.cash["USD"] == 1_030_000.0 and led.starting_cash["USD"] == 1_030_000.0 and led.transfers == pytest.approx(1_030_000.0)
    last = led.journal[-1]
    assert last.category == "transfer" and last.transfer == -20_000.0 and last.pnl == 0.0 and last.note == "withdrawal"
    assert cats(led).get("transfer", 0.0) == 0.0 and led.equity() == 1_030_000.0
    with pytest.raises(LedgerError, match="no rate for CHF"):
        led.deposit(T0, "CHF", 1.0)


def test_the_ledger_starts_at_the_time_given_or_at_the_epoch_and_with_the_cash_given():
    reg, _ = make_ledger()
    early = Ledger(reg, "USD", {"USD": 1_000.0})
    assert early.journal[0].ts == pd.Timestamp("1970-01-01") and early.ts == pd.Timestamp("1970-01-01") and early.cash == {"USD": 1_000.0}
    dated = Ledger(reg, "USD", {"USD": 1_000.0}, start=pd.Timestamp("2024-03-01"))
    assert dated.journal[0].ts == pd.Timestamp("2024-03-01") and dated.starting_cash == {"USD": 1_000.0}
    assert Ledger(reg, "USD").ts is None and Ledger(reg, "USD").cash == {}


def test_an_entry_that_does_not_balance_is_refused_at_once_and_one_within_the_tolerance_is_not():
    reg, led = make_ledger()
    with pytest.raises(LedgerError, match="breaks the accounting identity"):
        led._post(T0, "mtm", None, "USD", None, 100.0, 0.0, 50.0)                                              # 100 of value change claimed as 50 of profit
    assert len(led.journal) == 1 and led.cash == {"USD": 1_000_000.0}                                           # nothing was written
    led._post(T0, "mtm", None, "USD", None, 100.0, 0.0, 100.0 + 5e-8)                                         # off by less than 1e-9 of 100
    with pytest.raises(LedgerError, match="unknown journal category"):
        led._post(T0, "magic", None, "USD", {"USD": 1.0})
    assert led.journal[-1].category == "mtm"


def test_cashflows_that_are_not_cashflows_are_refused_and_a_zero_one_writes_nothing():
    reg, led = make_ledger()
    for category in ("trade", "transfer", "mtm", "variation_margin", "fx_translation"):
        with pytest.raises(LedgerError, match="cannot write"):
            led.post_cashflow(T0, category, "USD", 1.0)
    entries = len(led.journal)
    assert led.post_cashflow(T0, "dividend", "USD", 0.0) is None and len(led.journal) == entries
    entry = led.post_cashflow(T0, "dividend", "USD", 25.0, "STK", tags=("income",), strategy="carry", note="Q1")
    assert entry.pnl == 25.0 and entry.instrument_id == "STK" and entry.tags == ("income",) and entry.strategy == "carry" and entry.note == "Q1" and led.cash["USD"] == 1_000_025.0
    assert led.post_cashflow(T0, "interest", "USD", -3.0).pnl == -3.0 and led.journal[-1].instrument_id == ""


def test_a_balance_in_a_currency_with_no_rate_is_refused_when_the_book_is_valued_and_a_zero_one_is_not():
    reg, led = make_ledger()
    led.cash["GBP"] = 5.0
    with pytest.raises(LedgerError, match="cannot value the GBP balance"):
        led.revalue(T0 + DAY, {})
    led.cash["GBP"] = 0.0
    led.revalue(T0 + DAY, {})                                                                                  # nothing owned in it: nothing to value
    assert led.ts == T0 + DAY


def test_revaluing_skips_unknown_instruments_missing_marks_and_unchanged_futures_marks():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, "STK", 10.0, 100.0), mid=100.0)
    led.apply_fill(T0, Fill(T0, fut, 1.0, 4000.0), mid=4000.0)
    entries = len(led.journal)
    led.revalue(T0 + DAY, {"UNKNOWN": 1.0, "STK": float("nan"), fut: 4000.0})
    assert len(led.journal) == entries and led.positions["STK"].last_mark == 100.0
    led.revalue(T0 + 2 * DAY, {fut: 4004.0}, category="lifecycle_settlement")                                  # a caller can label the profit entries
    assert led.journal[-1].category == "lifecycle_settlement" and led.journal[-1].pnl == pytest.approx(200.0)
    led.revalue(T0 + 3 * DAY, {"STK": 100.0})                                                                  # an unchanged mark writes nothing
    assert len(led.journal) == entries + 1


def test_a_converter_with_pegs_values_a_pegged_currency_without_any_pair_price():
    reg = make_ledger()[0]
    led = Ledger(reg, "USD", None, CurrencyConverter("USD", {"USDT": 1.0}), 1e-9)
    led.deposit(T0, "USDT", 500.0)
    assert led.equity() == 500.0 and led.rate("USDT") == 1.0


# ------------------------------------------------------------------------------------------------------------------------ a position in another currency
def euro_stock() -> Instrument:
    return Instrument(instrument_id="SAP", asset_class="equity", instrument_type="equity", currency="EUR")


def test_a_share_priced_in_euros_is_traded_translated_marked_exposed_and_settled_at_the_euro_rate():
    reg, led = make_ledger([euro_stock()], cash={"USD": 1_000_000.0, "EUR": 50_000.0}, pegs={"EUR": 1.10})
    led.apply_fill(T0, Fill(T0, "SAP", 100.0, 100.0), mid=100.0)                                              # 10,000 euros of shares
    assert led.cash["EUR"] == 40_000.0 and led.positions["SAP"].last_value == 10_000.0
    assert led.equity() == pytest.approx(1_000_000.0 + 50_000.0 * 1.10) and led.net_assets() == pytest.approx({"USD": 1_000_000.0, "EUR": 50_000.0})
    led.converter.set_pair("EUR", "USD", 1.20)                                                                 # the euro gains ten cents
    led.revalue(T0 + DAY, {"SAP": 100.0})
    assert cats(led)["fx_translation"] == pytest.approx((40_000.0 + 10_000.0) * 0.10)                         # on the cash and on the shares
    assert led.equity() == pytest.approx(1_000_000.0 + 50_000.0 * 1.20)
    led.revalue(T0 + 2 * DAY, {"SAP": 110.0})
    assert cats(led)["mtm"] == pytest.approx(1_000.0 * 1.20)                                                  # a thousand euros, at today's rate
    frame = led.exposures({"SAP": 110.0})
    assert frame.loc["SAP", "notional_base"] == pytest.approx(100 * 110.0 * 1.20) and frame.loc["SAP", "currency"] == "EUR"
    equity = led.equity()
    led.close_position(T0 + 3 * DAY, "SAP", 120.0)
    assert led.cash["EUR"] == pytest.approx(40_000.0 + 12_000.0) and led.equity() == pytest.approx(equity + 1_000.0 * 1.20)


def test_a_euro_position_removed_for_nothing_or_for_stated_cash_leaves_its_value_at_the_euro_rate():
    reg, led = make_ledger([euro_stock()], cash={"USD": 1_000_000.0, "EUR": 50_000.0}, pegs={"EUR": 1.10})
    led.apply_fill(T0, Fill(T0, "SAP", 100.0, 100.0), mid=100.0)
    before = led.equity()
    led.remove_position_value(T0 + DAY, "SAP", 100.0)
    assert led.equity() == pytest.approx(before - 10_000.0 * 1.10)                                            # the shares' value left the books, in dollars at 1.10
    reg, led = make_ledger([euro_stock()], cash={"USD": 1_000_000.0, "EUR": 50_000.0}, pegs={"EUR": 1.10})
    led.apply_fill(T0, Fill(T0, "SAP", 100.0, 100.0), mid=100.0)
    led.settle_with_cash(T0 + DAY, "SAP", {"EUR": 9_000.0}, mark=100.0)
    assert led.journal[-1].pnl == pytest.approx(-1_000.0 * 1.10) and led.cash["EUR"] == pytest.approx(40_000.0 + 9_000.0)


def test_only_a_positive_finite_mark_of_a_currency_pair_moves_the_exchange_rate():
    reg, led = make_ledger([fx_spot("EUR", "USD")])
    led.converter.set_pair("EUR", "USD", 1.10)
    for bad in (0.0, -1.0, float("nan")):
        led.revalue(T0 + DAY, {"EURUSD": bad})
        assert led.converter.pairs[("EUR", "USD")] == 1.10
    led.revalue(T0 + DAY, {"EURUSD": 1.15})
    assert led.converter.pairs[("EUR", "USD")] == 1.15


def test_the_refusal_of_an_entry_that_does_not_balance_states_the_two_sides():
    reg, led = make_ledger()
    with pytest.raises(LedgerError, match=r"entry 'mtm' for  breaks the accounting identity: movements 100 != pnl \+ transfer 50"):
        led._post(T0, "mtm", None, "USD", None, 100.0, 0.0, 50.0)
    with pytest.raises(LedgerError, match=r"entry 'mtm' for STK breaks the accounting identity: movements 100 != pnl \+ transfer 60"):
        led._post(T0, "mtm", reg.get("STK"), "USD", None, 100.0, 0.0, 50.0, 10.0)
