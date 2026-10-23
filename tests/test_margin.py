"""Margin, collateral and the scenario scan on their own, each against a figure worked out by hand: every margin type follows its stated rule, the scan requires the worst loss on its grid and never a
negative one, offsetting positions pay less than their sum, collateral is haircut where it is an asset and not where it is a debt, and the call is made when equity falls below maintenance, not at it.

A planted-bug audit (``experiments/plant_bugs.py``) of ``src/ledger/margin.py`` found that the engine tests noticed 8 of 66 changes to it: only fixed futures margin was checked, and the percent, premium
and scan rules, the haircuts, the utilisation and the edges of the call could be changed without a test failing. These tests close that."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from src.instruments import Instrument, crypto_perp, crypto_spot, make_option
from src.ledger import Fill, MarginModel
from tests.test_ledger import T0, make_ledger


def stand_in(kind: str, initial: float = 0.0, maintenance: float = 0.0, multiplier: float = 1.0, **extra) -> SimpleNamespace:
    return SimpleNamespace(margin_type=kind, initial_margin=initial, maintenance_margin=maintenance, notional=lambda price, quantity=1.0: abs(quantity) * multiplier * price, **extra)


# ------------------------------------------------------------------------------------------------------------------------ the rule of each margin type
def test_no_margin_and_scan_margin_are_not_asked_for_per_position_and_nothing_held_needs_nothing():
    model = MarginModel()
    assert model.position_requirement(stand_in("none", 5.0, 4.0), 10.0, 100.0) == (0.0, 0.0)
    assert model.position_requirement(stand_in("portfolio", 5.0, 4.0), 10.0, 100.0) == (0.0, 0.0)             # the scan asks for it across the book instead
    assert model.position_requirement(stand_in("fixed", 12_000.0, 10_000.0), 0.0, 4000.0) == (0.0, 0.0)
    assert model.position_requirement(stand_in("percent", 0.1, 0.05), 0.0, 4000.0) == (0.0, 0.0)
    assert model.position_requirement(stand_in("unheard of", 5.0, 4.0), 10.0, 100.0) == (0.0, 0.0)


def test_fixed_margin_is_a_sum_per_contract_on_either_side_and_maintenance_defaults_to_the_initial_one():
    model = MarginModel()
    future = stand_in("fixed", 12_000.0, 10_000.0)
    assert model.position_requirement(future, 3.0, 4000.0) == (36_000.0, 30_000.0)
    assert model.position_requirement(future, -3.0, 4000.0) == (36_000.0, 30_000.0)                           # short or long
    assert model.position_requirement(future, 2.0, 1.0) == (24_000.0, 20_000.0)                              # the mark does not matter
    assert model.position_requirement(stand_in("fixed", 12_000.0, 0.0), 2.0, 4000.0) == (24_000.0, 24_000.0)


def test_percent_margin_is_a_share_of_the_notional_which_is_quantity_times_multiplier_times_the_mark():
    model = MarginModel()
    perp = stand_in("percent", 0.10, 0.05, multiplier=0.01)
    initial, maintenance = model.position_requirement(perp, -500.0, 30_000.0)                                 # notional 500 * 0.01 * 30,000 = 150,000
    assert initial == pytest.approx(15_000.0) and maintenance == pytest.approx(7_500.0)
    assert model.position_requirement(perp, 500.0, 30_000.0) == pytest.approx((15_000.0, 7_500.0))
    assert model.position_requirement(stand_in("percent", 0.10, 0.0, multiplier=0.01), 500.0, 30_000.0) == pytest.approx((15_000.0, 15_000.0))


def test_option_margin_is_nothing_for_a_long_and_the_listed_rule_for_a_short_with_maintenance_in_proportion():
    model = MarginModel()
    call = make_option("STK", "2024-06-21", "call", 50.0)                                                      # 20% initial, 10% maintenance, 100 shares
    assert model.position_requirement(call, 2.0, 3.0, 48.0) == (0.0, 0.0)                                      # the premium was paid
    # short two, underlying 48, premium 3: out of the money by 2; max(0.2 * 48 - 2, 0.1 * 48) + 3 = 10.6 per share
    initial, maintenance = model.position_requirement(call, -2.0, 3.0, 48.0)
    assert initial == pytest.approx(2 * 100 * 10.6) and maintenance == pytest.approx(2 * 100 * 10.6 * 0.5)
    assert call.margin_requirement(-2.0, 48.0, 3.0) == pytest.approx(2120.0)
    # no underlying price given: the mark stands in for it (documented behaviour; max(0.2 * 3 - 47, 0.1 * 3) + 3 = 3.3 per share)
    assert model.position_requirement(call, -2.0, 3.0) == pytest.approx((2 * 100 * 3.3, 2 * 100 * 3.3 * 0.5))
    assert model.position_requirement(stand_in("premium", 0.2, 0.1), -2.0, 3.0, 48.0) == (0.0, 0.0)           # an instrument with no rule of its own for it


# ------------------------------------------------------------------------------------------------------------------------ the scenario scan
def record(**fields) -> dict:
    return {"spot": 100.0, "vol": 0.2, "delta": 0.0, "gamma": 0.0, "vega": 0.0, **fields}


def test_the_scan_requires_the_worst_loss_on_its_grid_of_spot_and_volatility_shocks():
    model = MarginModel()
    assert model.portfolio_scan({"A": [record(delta=100.0)]})["A"] == pytest.approx(1500.0)                  # 100 delta, spot 100 down 15%
    assert model.portfolio_scan({"A": [record(delta=-100.0)]})["A"] == pytest.approx(1500.0)                 # a short loses when spot rises
    assert model.portfolio_scan({"A": [record(gamma=-2.0)]})["A"] == pytest.approx(225.0)                    # short gamma: 0.5 * 2 * 15^2 either way
    assert model.portfolio_scan({"A": [record(vega=1000.0)]})["A"] == pytest.approx(50.0)                    # long vega loses 1000 * 0.2 * 25%
    assert model.portfolio_scan({"A": [record(vega=-1000.0)]})["A"] == pytest.approx(50.0)
    assert model.portfolio_scan({"A": [record(delta=10.0, gamma=-0.1, vega=100.0)]})["A"] == pytest.approx(166.25)       # -150 - 11.25 - 5, at spot -15% and vol -25%
    assert model.portfolio_scan({"A": [record(delta=1.0, spot=200.0)]})["A"] == pytest.approx(30.0)          # each record is shocked in proportion to its own spot
    assert model.portfolio_scan({"A": [record(delta=1.0, vol=0.0, vega=500.0)]})["A"] == pytest.approx(0.15 * 100.0)      # a record with no volatility level has no vega loss


@pytest.mark.parametrize("delta, gamma, expected", [(10.0, 2.0, 25.0), (-10.0, 2.0, 25.0), (20.0, 2.0, 100.0), (-20.0, 2.0, 100.0), (30.0, 2.0, 225.0), (-30.0, 2.0, 225.0)])
def test_every_spot_shock_on_the_grid_can_be_the_worst_one(delta, gamma, expected):
    """With long gamma the loss is worst *between* the extremes, at spot * (-delta / gamma): 5%, 10% or 15% down (or up) from a spot of 100 for these records. Each of the six shocks other than zero
    is the unique worst one for one of them, so a changed shock changes an answer."""
    assert MarginModel().portfolio_scan({"A": [record(delta=delta, gamma=gamma)]})["A"] == pytest.approx(expected)


def test_the_sign_of_the_volatility_term_matters_when_the_grid_is_not_symmetric():
    rises_only = MarginModel(spot_shocks=(0.0,), vol_shocks=(0.25,))
    assert rises_only.portfolio_scan({"A": [record(vega=1000.0)]})["A"] == 0.0                              # long vega gains when volatility rises
    assert rises_only.portfolio_scan({"A": [record(vega=-1000.0)]})["A"] == pytest.approx(50.0)             # short vega loses 1000 * 0.2 * 25%
    falls_only = MarginModel(spot_shocks=(0.0,), vol_shocks=(-0.25,))
    assert falls_only.portfolio_scan({"A": [record(vega=1000.0)]})["A"] == pytest.approx(50.0) and falls_only.portfolio_scan({"A": [record(vega=-1000.0)]})["A"] == 0.0


def test_the_scan_never_asks_for_a_negative_requirement_and_offsetting_positions_pay_less_than_their_sum():
    model = MarginModel()
    assert model.portfolio_scan({"A": [record(gamma=2.0)]})["A"] == 0.0                                      # long gamma gains on every shock
    long, short = record(delta=100.0), record(delta=-100.0)
    hedged = model.portfolio_scan({"A": [long, short]})["A"]
    apart = model.portfolio_scan({"A": [long], "B": [short]})
    assert hedged == 0.0 and apart == {"A": pytest.approx(1500.0), "B": pytest.approx(1500.0)} and hedged < sum(apart.values())
    spread = model.portfolio_scan({"A": [record(delta=100.0, vega=1000.0), record(delta=-100.0, vega=-900.0)]})["A"]
    assert spread == pytest.approx(5.0)                                                                       # only the vega that is left over, 100 * 0.05


def test_the_grid_is_the_models_own_and_a_missing_optional_field_counts_as_zero():
    one_sided = MarginModel(spot_shocks=(0.05, 0.10), vol_shocks=(0.0,))
    assert one_sided.portfolio_scan({"A": [record(delta=100.0)]})["A"] == 0.0                                 # spot only rises on this grid
    down = MarginModel(spot_shocks=(-0.5,), vol_shocks=(0.0, 1.0))
    assert down.portfolio_scan({"A": [{"spot": 100.0, "delta": 1.0, "gamma": 0.0}]})["A"] == pytest.approx(50.0)           # no vol and no vega given
    assert MarginModel().portfolio_scan({}) == {}


# ------------------------------------------------------------------------------------------------------------------------ the position of the account
def test_the_state_counts_each_instruments_requirement_in_the_base_currency():
    perp, spot = crypto_perp("BINANCE", "BTC", "USDT"), crypto_spot("BINANCE", "BTC", "USDT")
    reg, led = make_ledger([perp, spot], cash={"USDT": 100_000.0, "USD": 0.0}, pegs={"USDT": 1.01})
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, perp.instrument_id, -2.0, 30_000.0), mid=30_000.0)
    led.apply_fill(T0, Fill(T0, fut, 3.0, 4000.0), mid=4000.0)
    state = MarginModel().state(led, {perp.instrument_id: 31_000.0, fut: 4000.0})
    notional = 2.0 * perp.contract_multiplier * 31_000.0
    expected = (notional * perp.initial_margin * 1.01, notional * perp.maintenance_margin * 1.01)               # quoted in USDT, counted in dollars at the rate of the day
    assert state.by_instrument[perp.instrument_id] == pytest.approx(expected)
    assert state.by_instrument[fut] == pytest.approx((36_000.0, 30_000.0))
    assert state.initial_margin == pytest.approx(expected[0] + 36_000.0) and state.maintenance_margin == pytest.approx(expected[1] + 30_000.0)
    assert state.excess_initial == pytest.approx(state.equity - state.initial_margin) and state.excess_maintenance == pytest.approx(state.equity - state.maintenance_margin)
    assert state.utilization == pytest.approx(state.initial_margin / state.equity)


def test_the_last_mark_is_used_for_a_position_the_caller_gave_no_mark_for():
    perp = crypto_perp("BINANCE", "BTC", "USDT")
    reg, led = make_ledger([perp], cash={"USDT": 100_000.0, "USD": 0.0}, pegs={"USDT": 1.0})
    led.apply_fill(T0, Fill(T0, perp.instrument_id, 1.0, 30_000.0), mid=30_000.0)
    led.revalue(T0, {perp.instrument_id: 32_000.0})
    state = MarginModel().state(led, {})
    assert state.by_instrument[perp.instrument_id][0] == pytest.approx(perp.contract_multiplier * 32_000.0 * perp.initial_margin)


def test_collateral_is_haircut_where_it_is_an_asset_and_at_full_value_where_it_is_a_debt():
    reg, led = make_ledger(cash={"USD": 50_000.0, "USDT": 100_000.0, "BTC": 2.0}, pegs={"USDT": 1.0, "BTC": 30_000.0})
    model = MarginModel(haircuts={"BTC": 0.20, "USDT": 0.05})
    state = model.state(led, {})
    assert state.collateral_by_currency == pytest.approx({"USD": 50_000.0, "USDT": 95_000.0, "BTC": 48_000.0})
    assert state.collateral_value == pytest.approx(193_000.0) and state.equity == pytest.approx(210_000.0)
    led.cash["USD"] = -10_000.0                                                                               # borrowed dollars are not discounted, they are owed in full
    led.cash["BTC"] = -1.0                                                                                    # nor is a coin owed
    state = model.state(led, {})
    assert state.collateral_by_currency == pytest.approx({"USD": -10_000.0, "USDT": 95_000.0, "BTC": -30_000.0}) and state.collateral_value == pytest.approx(55_000.0)
    assert MarginModel().state(led, {}).collateral_by_currency["USDT"] == pytest.approx(100_000.0)             # no haircut set: the face value


def test_utilisation_is_requirement_over_equity_and_is_infinite_when_there_is_no_equity_to_meet_one():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    model = MarginModel()
    flat = model.state(led, {})
    assert flat.initial_margin == 0.0 and flat.utilization == 0.0 and not flat.margin_call and flat.by_instrument == {}
    led.apply_fill(T0, Fill(T0, fut, 3.0, 4000.0), mid=4000.0)
    state = model.state(led, {fut: 4000.0})
    assert state.utilization == pytest.approx(36_000.0 / 1_000_000.0)
    led.cash["USD"] = 0.0
    broke = model.state(led, {fut: 4000.0})
    assert broke.equity == 0.0 and math.isinf(broke.utilization) and broke.margin_call
    led.positions.clear()
    nothing = model.state(led, {})
    assert nothing.equity == 0.0 and nothing.utilization == 0.0 and not nothing.margin_call                  # no requirement and no equity: nothing to meet


def test_the_call_is_made_below_the_maintenance_requirement_and_not_at_it():
    reg, led = make_ledger()
    fut = reg.chain("ES").contracts[0].instrument_id
    led.apply_fill(T0, Fill(T0, fut, 3.0, 4000.0), mid=4000.0)
    model = MarginModel()
    led.cash["USD"] = 30_000.0                                                                                # equity equals the 30,000 maintenance requirement
    assert not model.state(led, {fut: 4000.0}).margin_call
    led.cash["USD"] = 29_999.99
    state = model.state(led, {fut: 4000.0})
    assert state.margin_call and state.excess_maintenance < 0 and state.excess_initial < 0
    led.cash["USD"] = 35_000.0                                                                                # above maintenance, below the initial 36,000: no call, but no room either
    state = model.state(led, {fut: 4000.0})
    assert not state.margin_call and state.excess_maintenance == pytest.approx(5_000.0) and state.excess_initial == pytest.approx(-1_000.0)


def test_a_call_needs_a_requirement_to_fall_below_so_an_account_in_debt_with_no_positions_is_not_called():
    reg, led = make_ledger()
    led.cash["USD"] = -5_000.0
    state = MarginModel().state(led, {})
    assert state.equity == -5_000.0 and state.maintenance_margin == 0.0 and not state.margin_call and state.utilization == 0.0 and state.excess_maintenance == -5_000.0


def test_the_state_is_a_series_of_its_numbers_without_the_breakdowns():
    reg, led = make_ledger()
    series = MarginModel().state(led, {}).to_series()
    assert isinstance(series, pd.Series)
    assert set(series.index) == {"equity", "initial_margin", "maintenance_margin", "collateral_value", "excess_initial", "excess_maintenance", "utilization", "margin_call"}
    assert series["equity"] == 1_000_000.0 and series["margin_call"] == False  # noqa: E712  (a numpy bool)


# ------------------------------------------------------------------------------------------------------------------------ margin by scenario scan, through the ledger
def scan_instrument(name: str, underlying: str | None = "U") -> Instrument:
    return Instrument(instrument_id=name, asset_class="option", instrument_type="option_book", currency="USD", margin_type="portfolio", underlying_id=underlying)


def test_scan_instruments_are_margined_together_by_underlying_and_maintenance_is_a_share_of_the_initial():
    a, b = scan_instrument("OPT-A"), scan_instrument("OPT-B")
    reg, led = make_ledger([a, b])
    led.apply_fill(T0, Fill(T0, "OPT-A", 2.0, 1.0), mid=1.0)
    led.apply_fill(T0, Fill(T0, "OPT-B", -1.0, 1.0), mid=1.0)
    risk = {"OPT-A": {"delta": 50.0, "gamma": 0.0, "vega": 0.0, "spot": 100.0, "vol": 0.2, "underlying": "U"},
            "OPT-B": {"delta": 100.0, "gamma": 0.0, "vega": 0.0, "spot": 100.0, "vol": 0.2, "underlying": "U"}}
    model = MarginModel(maintenance_ratio=0.5)
    state = model.state(led, {}, risk)
    # per-unit deltas times the quantities held: 2 * 50 - 1 * 100 = 0, so the book is hedged and needs nothing
    assert state.initial_margin == 0.0 and state.by_instrument == {"scan:U": (0.0, 0.0)}
    unhedged = model.state(led, {}, {**risk, "OPT-B": {**risk["OPT-B"], "delta": 20.0}})                     # 2 * 50 - 20 = 80 delta left: 80 * 15 = 1200
    assert unhedged.initial_margin == pytest.approx(1200.0) and unhedged.maintenance_margin == pytest.approx(600.0)
    assert unhedged.by_instrument["scan:U"] == pytest.approx((1200.0, 600.0))
    assert MarginModel().state(led, {}, {**risk, "OPT-B": {**risk["OPT-B"], "delta": 20.0}}).maintenance_margin == pytest.approx(900.0)        # the default ratio is three quarters


def test_the_gamma_and_vega_of_a_scan_instrument_are_scaled_by_the_quantity_held_as_the_delta_is():
    opt = scan_instrument("OPT-A")
    reg, led = make_ledger([opt])
    led.apply_fill(T0, Fill(T0, "OPT-A", 3.0, 1.0), mid=1.0)
    per_unit = {"OPT-A": {"delta": 0.0, "gamma": -0.2, "vega": 10.0, "spot": 100.0, "vol": 0.2, "underlying": "U"}}
    # three held: gamma -0.6 and vega 30; worst at spot -15% (0.5 * -0.6 * 225 = -67.5) and volatility -25% (30 * -0.05 = -1.5)
    assert MarginModel().state(led, {}, per_unit).initial_margin == pytest.approx(69.0)
    led.apply_fill(T0, Fill(T0, "OPT-A", -6.0, 1.0), mid=1.0)                                              # now short three: gamma +0.6 and vega -30
    short = MarginModel().state(led, {}, per_unit)
    assert short.initial_margin == pytest.approx(1.5)                                                        # long gamma gains on every spot shock; only the vega loss (volatility up 25%) is left


def test_scan_instruments_without_a_risk_record_are_left_out_and_separate_underlyings_are_scanned_apart():
    a, b, c = scan_instrument("OPT-A", "UA"), scan_instrument("OPT-B", "UB"), scan_instrument("OPT-C", None)
    reg, led = make_ledger([a, b, c])
    for name in ("OPT-A", "OPT-B", "OPT-C"):
        led.apply_fill(T0, Fill(T0, name, 1.0, 1.0), mid=1.0)
    record_for = {"delta": 100.0, "gamma": 0.0, "vega": 0.0, "spot": 100.0, "vol": 0.2}
    state = MarginModel().state(led, {}, {"OPT-A": dict(record_for), "OPT-B": dict(record_for, underlying="UB")})        # OPT-C has no record
    assert set(state.by_instrument) == {"scan:UA", "scan:UB"}
    assert state.initial_margin == pytest.approx(3000.0)                                                       # two separate scans of 1500, not one netted
    keyed_by_own_id = MarginModel().state(led, {}, {"OPT-C": dict(record_for)})
    assert set(keyed_by_own_id.by_instrument) == {"scan:OPT-C"}                                                # no underlying anywhere: its own id is the key
    assert MarginModel().state(led, {}).by_instrument == {}                                                    # no records at all: nothing is invented


def test_a_position_in_another_currency_is_counted_at_that_currencys_rate():
    foreign = Instrument(instrument_id="EUR-FUT", asset_class="future", instrument_type="future", currency="EUR", margin_type="fixed", initial_margin=1000.0, maintenance_margin=800.0,
                         contract_multiplier=10.0)
    reg, led = make_ledger([foreign], cash={"USD": 1_000_000.0})
    led.converter.set_pair("EUR", "USD", 1.25)
    led.apply_fill(T0, Fill(T0, "EUR-FUT", 4.0, 100.0), mid=100.0)
    state = MarginModel().state(led, {"EUR-FUT": 100.0})
    assert state.by_instrument["EUR-FUT"] == pytest.approx((4 * 1000.0 * 1.25, 4 * 800.0 * 1.25))
