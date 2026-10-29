"""The option structures and rolling strategies against closed forms: the payoff at expiry of every structure is the sum of its legs' intrinsic values less what it cost, the textbook maxima, minima and
break-evens of each new structure follow from its strikes and its price, the rolling strategies choose the strikes their rules say, and the filter, the sweep and the comparison reproduce the plain run."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.assets import options_analytics as oa
from src.derivatives.backtest import Hedge, Order, StrategyContext
from src.derivatives.strategies import STRATEGIES, Collar, IronButterfly, ProtectivePut, PutSpread
from tests.test_options_chain_exact import AS_OF, flat_chain

SPOT, FWD, MULT = 500.0, 503.0, 100.0
SINGLE_EXPIRY = [n for n in oa.STRATEGY_INFO if n not in ("calendar_spread", "diagonal_spread")]


def analyse(name, **kw):
    legs = oa.strategy_legs(name, SPOT, FWD, 1.0, dte=30, **kw)
    return legs, oa.strategy_analysis(legs, SPOT, FWD, 0.18, 0.04, 0.013, commission=0.65, cash_secured=(name == "cash_secured_put"))


def intrinsic_pnl(legs, prices, cost):
    """The profit at expiry by the definition, one leg at a time: shares at the price, a call at max(S - K, 0), a put at max(K - S, 0), times quantity and multiplier."""
    total = np.zeros_like(prices)
    for leg in legs:
        if leg["type"] == "stock":
            total += leg["quantity"] * prices
        elif leg["right"] == "C":
            total += leg["quantity"] * MULT * np.maximum(prices - leg["strike"], 0.0)
        else:
            total += leg["quantity"] * MULT * np.maximum(leg["strike"] - prices, 0.0)
    return total - cost


# ------------------------------------------------------------------------------------------------------------------ the structures
def test_the_structures_are_grouped_once_each_and_the_group_of_a_name_is_found():
    members = [n for names in oa.STRATEGY_GROUPS.values() for n in names]
    assert sorted(members) == sorted(oa.STRATEGY_INFO) and len(set(members)) == len(members) and all(oa.STRATEGY_GROUPS.values())
    assert oa.group_of("collar") == "Own the shares" and oa.group_of("iron_butterfly") == "Sell premium" and oa.group_of("long_condor") == "Bet on a range" and oa.group_of("nope") == "Other"
    assert len(oa.STRATEGY_INFO) == 25 and {"collar", "cash_secured_put", "iron_butterfly", "long_condor", "risk_reversal", "strap", "strip", "call_backspread", "put_backspread", "diagonal_spread",
                                            "tail_hedge"} <= set(oa.STRATEGY_INFO)


@pytest.mark.parametrize("name", SINGLE_EXPIRY)
def test_the_payoff_at_expiry_is_the_sum_of_the_legs_less_the_cost(name):
    legs, a = analyse(name)
    prices = np.array(a["curve"]["spot"])
    assert a["curve"]["expiry"] == pytest.approx(intrinsic_pnl(legs, prices, a["cost"]).tolist(), abs=0.03)                         # the prices are rounded to 1e-4 and the profit to a cent


@pytest.mark.parametrize("name", ["calendar_spread", "diagonal_spread"])
def test_a_structure_with_a_later_leg_still_has_that_leg_valued_by_the_model_at_the_horizon(name):
    legs, a = analyse(name)
    near, far = min(l["dte"] for l in legs), max(l["dte"] for l in legs)
    assert near < far and a["horizon_days"] == near
    prices = np.array(a["curve"]["spot"])
    at_zero = a["curve"]["expiry"][0]
    assert prices[0] == 0.0 and at_zero == pytest.approx(-a["cost"], rel=0.01)                                    # at a price of zero every call is worthless: the whole debit is lost
    assert a["max_loss"] == pytest.approx(-a["cost"], rel=0.02) and not a["unlimited_loss"]


def test_a_collar_holds_the_shares_between_a_floor_and_a_cap():
    legs, a = analyse("collar")
    put = next(l for l in legs if l["type"] == "option" and l["right"] == "P")
    call = next(l for l in legs if l["type"] == "option" and l["right"] == "C")
    assert put["quantity"] == 1 and call["quantity"] == -1 and put["strike"] < SPOT < call["strike"]
    assert a["max_profit"] == pytest.approx(MULT * call["strike"] - a["cost"], abs=0.01) and a["max_loss"] == pytest.approx(MULT * put["strike"] - a["cost"], abs=0.01)
    assert not a["unlimited_profit"] and not a["unlimited_loss"] and a["greeks"]["delta_shares"] < 100 and a["margin"]["estimate"] == pytest.approx(0.5 * SPOT * MULT, rel=0.02)
    cost_of_shares = 100 * SPOT
    assert a["cost"] == pytest.approx(cost_of_shares + sum(l["quantity"] * MULT * l["price"] for l in a["legs"] if l["type"] == "option") + 0.65 * 2, abs=0.01)


def test_a_cash_secured_put_risks_the_strike_less_the_premium_and_holds_that_in_cash():
    legs, a = analyse("cash_secured_put")
    k = legs[0]["strike"]
    credit = a["net_credit"]
    assert legs[0]["quantity"] == -1 and a["max_profit"] == pytest.approx(credit, abs=0.01) and a["max_loss"] == pytest.approx(-(MULT * k - credit), abs=0.01)
    assert a["breakevens"] == pytest.approx([k - credit / MULT], abs=0.02) and not a["unlimited_loss"]
    assert a["margin"]["estimate"] == pytest.approx(MULT * k - credit, abs=0.01) and a["margin"]["cash_secured"] and not any("naked" in f for f in a["risk_flags"])
    plain = oa.strategy_analysis(legs, SPOT, FWD, 0.18, 0.04, 0.013, commission=0.65)                                   # the same put margined as a naked short
    assert not plain["margin"]["cash_secured"] and plain["margin"]["estimate"] < a["margin"]["estimate"] and any("naked" in f for f in plain["risk_flags"])
    assert a["prob_profit"] > 0.7


def test_an_iron_butterfly_earns_its_credit_at_the_money_and_loses_the_wing_less_the_credit():
    legs, a = analyse("iron_butterfly")
    strikes = sorted({l["strike"] for l in legs})
    low, mid, high = strikes
    credit = a["net_credit"]
    assert [(l["right"], l["quantity"]) for l in legs] == [("P", 1), ("P", -1), ("C", -1), ("C", 1)] and mid == next(l["strike"] for l in legs if l["quantity"] == -1)
    assert a["max_profit"] == pytest.approx(credit, abs=0.01)
    assert a["max_loss"] == pytest.approx(-(MULT * (mid - low) - credit), abs=0.01) and mid - low == pytest.approx(high - mid, abs=1.0)
    assert a["breakevens"] == pytest.approx([mid - credit / MULT, mid + credit / MULT], abs=0.02)
    assert a["margin"]["estimate"] == pytest.approx(MULT * max(mid - low, high - mid) - credit, abs=1.5) and not a["unlimited_loss"]


def test_a_long_condor_pays_across_the_band_between_its_inner_strikes():
    legs, a = analyse("long_condor")
    k1, k2, k3, k4 = [l["strike"] for l in legs]
    debit = a["net_debit"]
    assert [l["quantity"] for l in legs] == [1, -1, -1, 1] and k1 < k2 < k3 < k4 and a["unlimited_profit"] is False
    assert a["max_loss"] == pytest.approx(-debit, abs=0.01) and a["max_profit"] == pytest.approx(MULT * (k2 - k1) - debit, abs=0.01)
    assert a["breakevens"] == pytest.approx([k1 + debit / MULT, k4 - debit / MULT], abs=0.02)
    inner = np.array(a["curve"]["spot"])
    plateau = np.array(a["curve"]["expiry"])[(inner >= k2) & (inner <= k3)]
    assert plateau == pytest.approx(a["max_profit"], abs=0.01)


def test_a_risk_reversal_is_long_a_call_and_short_a_put_and_loses_the_put_strike_at_zero():
    legs, a = analyse("risk_reversal")
    call = next(l for l in legs if l["right"] == "C")
    put = next(l for l in legs if l["right"] == "P")
    assert call["quantity"] == 1 and put["quantity"] == -1 and put["strike"] < SPOT < call["strike"] and a["unlimited_profit"] and not a["unlimited_loss"]
    assert a["max_loss"] == pytest.approx(-MULT * put["strike"] - a["cost"], abs=0.01)
    assert a["greeks"]["delta_shares"] > 0 and a["greeks"]["vega_per_point"] == pytest.approx(0.0, abs=0.35 * abs(a["greeks"]["vega_per_point"]) + 15.0)       # calls and puts of equal distance roughly cancel in volatility


def test_a_strap_and_a_strip_are_a_straddle_leaning_up_or_down():
    for name, up, down in (("strap", 2, 1), ("strip", 1, 2)):
        legs, a = analyse(name)
        k = legs[0]["strike"]
        assert {l["strike"] for l in legs} == {k} and a["unlimited_profit"] and not a["unlimited_loss"] and a["max_loss"] == pytest.approx(-a["cost"], abs=0.01)
        assert a["breakevens"] == pytest.approx([k - a["cost"] / (down * MULT), k + a["cost"] / (up * MULT)], abs=0.02), name
        prices, pnl = np.array(a["curve"]["spot"]), np.array(a["curve"]["expiry"])
        assert (pnl[-1] - pnl[-2]) / (prices[-1] - prices[-2]) == pytest.approx(up * MULT, rel=1e-4) and (pnl[1] - pnl[2]) / (prices[1] - prices[2]) == pytest.approx(-down * MULT, rel=1e-4)
    _, strap = analyse("strap")
    _, strip = analyse("strip")
    assert strap["greeks"]["delta_shares"] > 0 > strip["greeks"]["delta_shares"]


def test_the_backspreads_lose_most_at_the_far_strike_and_win_without_limit_on_one_side():
    legs, call = analyse("call_backspread")
    ka, kb = sorted({l["strike"] for l in legs})
    assert [l["quantity"] for l in legs] == [-1, 2] and call["unlimited_profit"] and not call["unlimited_loss"]
    assert call["max_loss"] == pytest.approx(-MULT * (kb - ka) - call["cost"], abs=0.01) and call["net_credit"] > 0
    legs, put = analyse("put_backspread")
    kb, ka = sorted({l["strike"] for l in legs})
    assert [l["quantity"] for l in legs] == [-1, 2] and not put["unlimited_profit"] and not put["unlimited_loss"]
    assert put["max_loss"] == pytest.approx(-MULT * (ka - kb) - put["cost"], abs=0.01) and put["max_profit"] == pytest.approx(MULT * (2 * kb - ka) - put["cost"], abs=0.01)


def test_a_tail_hedge_costs_little_and_pays_a_great_deal_in_a_crash():
    legs, a = analyse("tail_hedge")
    k = legs[0]["strike"]
    _, atm_put = analyse("long_put")
    assert legs[0]["quantity"] == 1 and legs[0]["right"] == "P" and k < 0.92 * SPOT and a["cost"] < 0.2 * atm_put["cost"]
    assert a["max_loss"] == pytest.approx(-a["cost"], abs=0.01) and a["max_profit"] == pytest.approx(MULT * k - a["cost"], abs=0.01) and a["breakevens"] == pytest.approx([k - a["cost"] / MULT], abs=0.02)
    assert a["prob_profit"] < 0.1 and a["max_profit"] > 50 * a["cost"]


def test_the_wings_of_a_structure_cannot_reach_a_zero_strike():
    with pytest.raises(ValueError, match="at or below zero"):
        oa.strategy_legs("tail_hedge", SPOT, FWD, 1.0, 30, offset=0.5, width=0.5)
    legs = oa.strategy_legs("long_condor", SPOT, FWD, 1.0, 30, offset=0.0, width=0.1)                                 # the inner strikes meet at the money: still a valid, four-legged payoff
    assert legs[1]["strike"] == legs[2]["strike"] and oa.strategy_analysis(legs, SPOT, FWD, 0.18)["max_profit"] is not None


def test_the_density_the_grid_and_the_greeks_of_a_structure_agree_with_the_curves():
    legs, a = analyse("iron_condor")
    d = a["density"]
    assert len(d["edges"]) == len(d["mass"]) + 1 and sum(d["mass"]) == pytest.approx(norm.cdf(3.5) - norm.cdf(-3.5), abs=1e-9) and d["edges"] == sorted(d["edges"])
    prob_inside = sum(m for lo, hi, m in zip(d["edges"][:-1], d["edges"][1:], d["mass"]) if lo >= legs[1]["strike"] and hi <= legs[2]["strike"])
    assert 0.0 < prob_inside < a["prob_profit"] + 0.01
    g = a["grid"]
    assert g["days"][0] == 0 and g["days"][-1] == a["horizon_days"] and len(g["pnl"]) == len(g["spot"]) and all(len(row) == len(g["days"]) for row in g["pnl"])
    now, at_end = [row[0] for row in g["pnl"]], [row[-1] for row in g["pnl"]]
    assert at_end == pytest.approx(np.interp(g["spot"], a["curve"]["spot"], a["curve"]["expiry"]).tolist(), abs=1.0) and now == pytest.approx(np.interp(g["spot"], a["curve"]["spot"], a["curve"]["now"]).tolist(), abs=1.0)
    c = a["greeks_curve"]
    k = int(np.argmin(np.abs(np.array(c["spot"]) - SPOT)))
    assert c["spot"] == g["spot"] and c["delta"][k] == pytest.approx(a["greeks"]["delta_shares"], abs=3.0) and c["gamma"][k] == pytest.approx(a["greeks"]["gamma"], rel=0.15, abs=0.02)
    assert c["vega"][k] == pytest.approx(a["greeks"]["vega_per_point"], rel=0.2, abs=2.0)
    assert a["greeks"]["gamma"] < 0 < a["greeks"]["theta_per_day"] and max(abs(x) for x in c["delta"]) < 100                  # short gamma and long time decay at the money; a condor never carries a full contract's delta
    assert c["gamma"][k] < 0 and c["theta"][k] == pytest.approx(a["greeks"]["theta_per_day"], rel=0.2, abs=2.0)


def test_named_structures_on_a_chain_use_its_strikes_and_volatilities():
    chain = flat_chain(spot=100.0, sigma=0.25, days=(20, 40, 90), step=1.0)
    legs, ctx = oa.named_structure(chain, 100.0, "iron_condor", dte=30, offset=0.05, width=0.05)
    assert ctx["dte"] in (20, 40) and ctx["step"] == 1.0 and all(l["iv"] == pytest.approx(0.25, abs=1e-6) for l in legs)
    assert all(l["dte"] == ctx["dte"] and float(l["strike"]).is_integer() for l in legs)
    legs, ctx = oa.named_structure(chain, 100.0, "diagonal_spread", dte=20, far_dte=90)
    assert sorted(l["dte"] for l in legs) == [20, 90] and ctx["far_dte"] == 90


def test_comparing_structures_reads_their_own_curves_on_one_axis():
    chain = flat_chain(spot=100.0, sigma=0.25, days=(30, 60), step=1.0)
    legs, ctx = oa.named_structure(chain, 100.0, "long_straddle", dte=30)
    main = oa.strategy_analysis(legs, 100.0, ctx["forward"], ctx["atm_iv"], 0.03, 0.01)
    other = oa.compare_structures(chain, 100.0, ["iron_condor", "collar"], main["curve"]["spot"], dte=30, analysis={"rate": 0.03, "dividend": 0.01})
    assert [o["strategy"] for o in other] == ["iron_condor", "collar"] and all(len(o["curve"]) == len(main["curve"]["spot"]) for o in other)
    condor_legs, _ = oa.named_structure(chain, 100.0, "iron_condor", dte=30)
    solo = oa.strategy_analysis(condor_legs, 100.0, ctx["forward"], ctx["atm_iv"], 0.03, 0.01)
    at = np.interp(main["curve"]["spot"], solo["curve"]["spot"], solo["curve"]["expiry"])
    assert other[0]["curve"] == pytest.approx(at.tolist(), abs=0.0101) and other[0]["max_profit"] == solo["max_profit"] and other[0]["group"] == "Sell premium"
    assert other[0]["net_credit"] > 0 and other[1]["net_debit"] > 5000 and other[1]["margin"] > 0


# ------------------------------------------------------------------------------------------------------------------ the rolling strategies
def context(chain, positions=None, units=0.0, date=AS_OF) -> StrategyContext:
    spot = float(chain["underlying"].iloc[0])
    return StrategyContext(date, chain, spot, pd.Series([spot], index=[date]), positions or {}, units, 100_000.0, 100_000.0, {"delta": units, "gamma": 0.0, "vega": 0.0, "theta": 0.0})


@pytest.fixture(scope="module")
def chain():
    return flat_chain(spot=100.0, sigma=0.25, rate=0.0, dividend=0.0, days=(30, 45, 60), step=1.0, width=0.5)


def put_delta(chain, expiry, strike):
    g = chain[(chain["expiry"] == expiry) & (chain["right"] == "P") & (chain["strike"] == strike)].iloc[0]
    d1 = (math.log(g["underlying"] / strike) + 0.5 * g["iv"] ** 2 * g["T"]) / (g["iv"] * math.sqrt(g["T"]))
    return abs(-norm.cdf(-d1))


def test_the_new_strategies_are_registered_and_their_settings_are_validated():
    assert {"put_spread", "iron_butterfly", "collar", "protective_put"} <= set(STRATEGIES) and set(oa.BACKTEST_STRATEGIES) == set(STRATEGIES)
    assert set(oa.BACKTEST_TEXT) == set(oa.BACKTEST_STRATEGIES) == set(oa.BACKTEST_PARAMS)
    for bad in ({"short_delta": 0.1, "long_delta": 0.2}, {"short_delta": 0.25, "long_delta": 0.25}):
        with pytest.raises(ValueError, match="long_delta"):
            oa.parse_backtest_params("put_spread", bad)
        with pytest.raises(ValueError, match="long_delta"):
            PutSpread(**bad)
    with pytest.raises(ValueError, match="wing"):
        IronButterfly(wing=0.8)
    with pytest.raises(ValueError, match="put_moneyness"):
        Collar(put_moneyness=1.02)
    with pytest.raises(ValueError, match="call_moneyness"):
        Collar(call_moneyness=0.98)
    with pytest.raises(ValueError, match="put_moneyness"):
        ProtectivePut(put_moneyness=1.0)
    with pytest.raises(ValueError, match="between"):
        oa.parse_backtest_params("collar", {"put_moneyness": 0.3})
    assert oa.parse_backtest_params("collar", {"call_moneyness": "1.1", "dte": "60"}) == {"call_moneyness": 1.1, "dte": 60}


def test_a_put_spread_sells_the_put_nearest_one_delta_and_buys_the_one_nearest_another(chain):
    expiry = chain.loc[(chain["T"] * 365).round() == 45, "expiry"].iloc[0]
    orders = PutSpread(0.25, 0.10).on_date(context(chain))
    strikes = np.sort(chain.loc[(chain["expiry"] == expiry) & (chain["right"] == "P"), "strike"].unique())
    nearest = lambda target: float(min(strikes, key=lambda k: abs(put_delta(chain, expiry, k) - target)))                  # noqa: E731
    assert orders == [Order(expiry, nearest(0.25), "P", -1.0), Order(expiry, nearest(0.10), "P", 1.0)] and orders[1].strike < orders[0].strike
    assert [o.quantity for o in PutSpread(size=3).on_date(context(chain))] == [-3.0, 3.0]


def test_an_iron_butterfly_sells_the_money_and_buys_wings_a_fifth_of_a_tenth_away(chain):
    expiry = chain.loc[(chain["T"] * 365).round() == 45, "expiry"].iloc[0]
    orders = IronButterfly(wing=0.05).on_date(context(chain))
    short_c, short_p, long_p, long_c = orders
    forward = float(chain.loc[chain["expiry"] == expiry, "F"].iloc[0])
    atm = min(chain.loc[chain["expiry"] == expiry, "strike"].unique(), key=lambda k: abs(k - forward))
    assert (short_c.right, short_p.right, long_p.right, long_c.right) == ("C", "P", "P", "C") and short_c.strike == short_p.strike == atm and short_c.quantity == short_p.quantity == -1.0
    assert long_p.strike == pytest.approx(95.0, abs=0.51) and long_c.strike == pytest.approx(105.0, abs=0.51) and long_p.quantity == long_c.quantity == 1.0
    wide = IronButterfly(wing=0.1).on_date(context(chain))
    assert wide[2].strike < long_p.strike and wide[3].strike > long_c.strike


def test_a_collar_buys_the_put_below_sells_the_call_above_and_holds_the_shares(chain):
    orders = Collar(0.95, 1.05, shares_per_contract=100.0, size=2.0).on_date(context(chain))
    put, call, hedge = orders
    assert (put.right, put.quantity) == ("P", 2.0) and (call.right, call.quantity) == ("C", -2.0) and hedge == Hedge(200.0)
    assert put.strike == pytest.approx(95.0, abs=0.51) and call.strike == pytest.approx(105.0, abs=0.51) and put.expiry == call.expiry
    asymmetric = Collar(0.9, 1.10).on_date(context(chain))
    assert asymmetric[0].strike < put.strike and asymmetric[1].strike > call.strike


def test_a_protective_put_buys_the_put_and_holds_the_shares(chain):
    put, hedge = ProtectivePut(0.90, shares_per_contract=50.0).on_date(context(chain))
    assert (put.right, put.quantity, put.strike) == ("P", 1.0, 90.0) and hedge == Hedge(50.0)


def test_the_package_is_closed_and_reopened_a_few_days_before_it_expires_and_otherwise_left_alone(chain):
    strategy = Collar()
    first = strategy.on_date(context(chain))
    positions = {(o.expiry, o.strike, o.right): o.quantity for o in first if isinstance(o, Order)}
    soon = min(k[0] for k in positions) - pd.Timedelta(days=3)                                                            # within close_dte (7) of the held expiry
    again = strategy.on_date(context(chain, positions, 100.0, date=soon))
    closes, reopened = again[:len(positions)], again[len(positions):]
    assert {(o.expiry, o.strike, o.right): o.quantity for o in closes} == {k: -q for k, q in positions.items()}           # first the package held is closed, contract by contract...
    assert reopened == first and any(isinstance(o, Hedge) for o in reopened)                                              # ...then the next one is opened (the same strikes: this chain has only one date), shares re-asserted
    quiet = strategy.on_date(context(chain, positions, 100.0, date=AS_OF))
    assert quiet == []                                                                                                    # nothing to do in the middle of the period


def test_a_strategy_that_cannot_find_a_quote_trades_nothing():
    empty = flat_chain(spot=100.0, sigma=0.25, days=(30,), step=1.0, width=0.3)
    empty["bid"] = 0.0
    for strategy in (PutSpread(), IronButterfly(), Collar(), ProtectivePut()):
        assert strategy.on_date(context(empty)) == []


# ------------------------------------------------------------------------------------------------------------------ the filter, the sweep and the comparison
DAYS = 90
KW = dict(n_days=DAYS, seed=3, capital=100_000.0, premium=0.25, commission=0.65, fill_fraction=1.0)


def test_the_variance_premium_filter_is_parsed_and_bounded():
    assert oa.parse_timing(None) is None and oa.parse_timing({}) is None
    assert oa.parse_timing({"window": "30", "threshold": "0.01"}) == {"window": 30, "threshold": 0.01} and oa.parse_timing({"window": 21}) == {"window": 21, "threshold": 0.0}
    for bad in ({"window": 3}, {"window": 500}, {"threshold": 2.0}, {"window": "x"}, {"speed": 1}, [1]):
        with pytest.raises(ValueError):
            oa.parse_timing(bad)


def test_a_filter_that_is_never_on_trades_nothing_and_one_that_is_always_on_waits_only_for_its_window():
    plain = oa.run_strategy_backtest("short_put", None, **KW)
    never = oa.run_strategy_backtest("short_put", None, timing={"window": 21, "threshold": 0.5}, **KW)
    always = oa.run_strategy_backtest("short_put", None, timing={"window": 21, "threshold": -0.2}, **KW)
    assert never["trades"] == 0 and never["share_in_market"] == 0.0 and set(never["equity"]) == {1.0} and plain["timing"] is None and plain["share_in_market"] > 0.5
    assert always["timing"] == {"window": 21, "threshold": -0.2} and always["trades"] > 0 and 0.0 < always["share_in_market"] <= plain["share_in_market"]
    assert set(always["equity"][:5]) == {1.0}                                                                         # no decision until the window of realised volatility is full (the first 21 days)


def test_a_backtest_carries_the_risk_views_the_other_tabs_show():
    out = oa.run_strategy_backtest("iron_butterfly", None, **KW)
    n = len(out["dates"])
    assert len(out["equity"]) == len(out["drawdown"]) == len(out["rolling_vol"]) == len(out["benchmark"]) == n and max(out["drawdown"]) <= 0 and min(out["drawdown"]) >= -1
    result, _ = oa._backtest_result("iron_butterfly", None, DAYS, 3, 100_000.0, 0.25, 0.65, 1.0, None)
    daily = result.returns.dropna()
    assert sum(y["days"] for y in out["years"]) == len(daily) and out["summary"]["total_return"] == pytest.approx(float((1.0 + daily).prod() - 1.0), rel=1e-9, abs=1e-12)
    assert sum(1 + y["value"] for y in out["years"]) > 0 and {"sortino", "calmar", "omega", "var_95", "es_95", "ulcer_index"} <= set(out["risk"]) and sum(out["distribution"]["counts"]) > 0 and out["monthly"]
    equity = (1.0 + daily).cumprod()
    assert out["equity"] == pytest.approx([round(float(v), 5) for v in equity.to_numpy()[::2]]) and out["drawdown"] == pytest.approx([round(float(v), 5) for v in (equity / equity.cummax() - 1.0).to_numpy()[::2]])


def test_a_sweep_runs_the_plain_backtest_at_each_value_and_checks_them_first():
    sweep = oa.run_strategy_sweep("short_put", "delta", [0.1, 0.25, 0.4], **KW)
    assert [r["value"] for r in sweep["rows"]] == [0.1, 0.25, 0.4] and sweep["base"] == 0.25 and sweep["default"] == 0.25 and sweep["param"] == "delta"
    single = oa.run_strategy_backtest("short_put", {"delta": 0.25}, **KW)
    assert sweep["rows"][1]["sharpe"] == single["summary"]["sharpe"] and sweep["rows"][1]["trades"] == single["trades"] and sweep["rows"][1]["costs"] == single["option_summary"]["costs_total"]
    assert sweep["rows"][0]["sharpe"] != sweep["rows"][2]["sharpe"]
    for bad_values in ([0.1], [0.1] * 10, "0.1"):
        with pytest.raises(ValueError, match="between 2 and 9"):
            oa.check_sweep("short_put", "delta", bad_values)
    with pytest.raises(ValueError, match="numeric setting"):
        oa.check_sweep("short_strangle", "hedge", [0, 1])
    with pytest.raises(ValueError, match="numeric setting"):
        oa.check_sweep("short_put", "nope", [1, 2])
    with pytest.raises(ValueError, match="between"):
        oa.check_sweep("short_put", "delta", [0.1, 0.9])
    with pytest.raises(ValueError, match="long_delta"):
        oa.check_sweep("put_spread", "short_delta", [0.05, 0.3], {"long_delta": 0.2})                                       # 0.05 would leave the bought put nearer the money


def test_a_comparison_runs_every_strategy_at_its_defaults_on_one_market():
    names = ["short_put", "put_spread", "iron_butterfly"]
    cmp = oa.run_strategy_comparison(names, **KW)
    assert [r["strategy"] for r in cmp["rows"]] == names and set(cmp["equity"]) == set(names) and all(len(v) == len(cmp["dates"]) == len(cmp["benchmark"]) for v in cmp["equity"].values())
    own = oa.run_strategy_backtest("put_spread", None, **KW)
    row = next(r for r in cmp["rows"] if r["strategy"] == "put_spread")
    assert row["sharpe"] == own["summary"]["sharpe"] and cmp["equity"]["put_spread"] == own["equity"]
    with pytest.raises(ValueError, match="at least two"):
        oa.run_strategy_comparison(["short_put"], **KW)
    with pytest.raises(ValueError, match="unknown strategy"):
        oa.run_strategy_comparison(["short_put", "naked_hope"], **KW)


def test_the_protective_strategies_hold_shares_and_the_credit_strategies_do_not():
    for name, shares in (("collar", True), ("protective_put", True), ("put_spread", False), ("iron_butterfly", False)):
        result, _ = oa._backtest_result(name, None, DAYS, 3, 100_000.0, 0.25, 0.65, 1.0, None)
        held = result.positions["underlying_units"].abs().max()
        assert (held > 0) == shares, name
        assert result.skipped_orders == 0 and len(result.trades) >= 2, name
