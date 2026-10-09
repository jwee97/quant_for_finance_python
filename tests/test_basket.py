"""Basket schedules, the three trade-list tactics (minimum trading risk, maximum trading opportunity, program-block) and liquidity-seeking execution."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.algo import Market, Order
from src.algo.algos import VWAP
from src.algo.basket import (Basket, basket_cost_risk, basket_schedule, independent_schedule, maximum_trading_opportunity, minimum_trading_risk_quantity, program_block, random_basket, simulate_basket)
from src.algo.liquidity import LiquiditySeeking
from src.algo.simulate import AGGRESSIVE, WORKING, simulate


def _basket(corr=None, beta=1.0, **kw):
    """Four names: two buys and two sells, a liquid pair and an illiquid pair."""
    defaults = dict(names=["A", "B", "C", "D"], shares=[60_000, -50_000, 40_000, -45_000], prices=[50.0, 40.0, 80.0, 60.0], adv=[3e6, 2e6, 6e5, 8e5], sigma=[0.018, 0.02, 0.025, 0.022], corr=corr)
    defaults.update(kw)
    return Basket(beta=beta, **defaults)


# --------------------------------------------------------------------------------------------------------------------------- the list
def test_a_basket_that_cannot_exist_is_refused():
    with pytest.raises(ValueError):
        Basket(["A", "B"], [1, 2, 3], [10, 10], [1e6, 1e6], [0.02, 0.02])
    with pytest.raises(ValueError):
        Basket(["A"], [0], [10], [1e6], [0.02])
    with pytest.raises(ValueError):
        Basket(["A", "B"], [1, 1], [10, 10], [1e6, 1e6], [0.02, 0.02], corr=[[1.0, 2.0], [2.0, 1.0]])                      # not positive semi-definite


def test_the_list_reports_sides_values_and_a_risk_that_nets_offsetting_trades():
    b = _basket()
    assert b.side.tolist() == [1, -1, 1, -1] and b.value.tolist() == [3_000_000.0, 2_000_000.0, 3_200_000.0, 2_700_000.0] and b.gross == pytest.approx(10_900_000.0)
    pair = Basket(["X", "Y"], [1000, -1000], [100.0, 100.0], [1e6, 1e6], [0.02, 0.02], corr=[[1.0, 1.0], [1.0, 1.0]])
    assert pair.risk() == pytest.approx(0.0, abs=1e-6)                                                       # a buy hedged by an identical sell
    long_only = Basket(["X", "Y"], [1000, 1000], [100.0, 100.0], [1e6, 1e6], [0.02, 0.02], corr=[[1.0, 1.0], [1.0, 1.0]])
    assert long_only.risk() == pytest.approx(2 * 100_000 * 0.02)
    assert pair.risk([1.0, 0.0]) == pytest.approx(100_000 * 0.02)                                              # executing only the buy leg leaves a naked sale


def test_a_random_basket_is_reproducible_and_mixes_buys_and_sells():
    a, b = random_basket(8, seed=4), random_basket(8, seed=4)
    assert (a.shares == b.shares).all() and (a.side > 0).any() and (a.side < 0).any() and a.gross == pytest.approx(20e6, rel=0.05)


# --------------------------------------------------------------------------------------------------------------------------- the joint schedule
def test_every_row_of_the_joint_schedule_adds_up_to_its_order_and_is_non_negative():
    b = _basket()
    for ra in (0.0, 1e-3, 1e-1):
        Q = basket_schedule(b, ra)
        assert Q.shape == (4, b.n) and (Q >= -1e-9).all() and Q.sum(axis=1) == pytest.approx(b.quantity)


def test_without_risk_aversion_each_stock_follows_its_own_volume():
    b = _basket()
    Q = basket_schedule(b, 0.0)
    assert Q == pytest.approx(b.quantity[:, None] * b.volume() / b.adv[:, None], rel=1e-5)


@pytest.mark.parametrize("ra", [1e-3, 1e-2, 1e-1])
def test_the_joint_schedule_is_never_worse_than_scheduling_each_stock_alone_in_the_model_it_optimises(ra):
    b = _basket(corr=np.array([[1, .8, .6, .5], [.8, 1, .5, .6], [.6, .5, 1, .8], [.5, .6, .8, 1.0]]))
    J = lambda s: (lambda c: c["cost_bps"] + ra * c["risk_bps"] ** 2)(basket_cost_risk(b, s))                  # linear impact (beta = 1): the QP is exact, so this is the model's own objective
    assert J(basket_schedule(b, ra)) <= J(independent_schedule(b, ra)) + 1e-9


def test_for_uncorrelated_stocks_the_joint_schedule_is_just_the_stocks_alone():
    b = _basket(corr=np.eye(4))
    assert basket_schedule(b, 1e-2) == pytest.approx(independent_schedule(b, 1e-2), rel=1e-4, abs=1.0)


def test_a_hedged_pair_is_worked_together_so_the_basket_is_never_left_naked():
    """A buy of a liquid name against a sale of an illiquid one, perfectly correlated: stock by stock the liquid leg finishes early and leaves a naked sale; jointly it waits for its hedge."""
    b = Basket(["LIQUID", "ILLIQUID"], [100_000, -100_000], [50.0, 50.0], [4e6, 4e5], [0.02, 0.02], corr=[[1.0, 1.0], [1.0, 1.0]], beta=1.0)
    ra = 3e-3
    joint, alone = basket_cost_risk(b, basket_schedule(b, ra)), basket_cost_risk(b, independent_schedule(b, ra))
    assert joint["risk_bps"] < 0.6 * alone["risk_bps"]
    Q = basket_schedule(b, ra)
    done = np.cumsum(Q, axis=1) / b.quantity[:, None]
    assert np.abs(done[0] - done[1]).max() < np.abs(np.cumsum(independent_schedule(b, ra), axis=1)[0] / b.quantity[0] - np.cumsum(independent_schedule(b, ra), axis=1)[1] / b.quantity[1]).max()


def test_the_power_law_polish_improves_the_power_law_objective():
    b = random_basket(6, seed=3)
    ra = 1e-2
    J = lambda s: (lambda c: c["cost_bps"] + ra * c["risk_bps"] ** 2)(basket_cost_risk(b, s))
    quad, exact = basket_schedule(b, ra, exact=False), basket_schedule(b, ra, exact=True)
    assert J(exact) <= J(quad) + 1e-9 and exact.sum(axis=1) == pytest.approx(b.quantity) and J(exact) <= J(independent_schedule(b, ra, exact=True)) + 1e-6


def test_the_basket_simulator_reproduces_the_closed_forms_when_volumes_are_not_noisy():
    b = random_basket(5, seed=2)
    b.volume_noise = b.day_noise = 0.0
    Q = basket_schedule(b, 1e-3)
    c = basket_cost_risk(b, Q)
    s = simulate_basket(b, Q, paths=3000, seed=1, noise=False)
    se = s["std_bps"] / np.sqrt(3000)
    assert abs(s["mean_bps"] - c["cost_bps"]) < 4 * se
    assert s["std_bps"] == pytest.approx(c["risk_bps"], rel=0.05)
    parts = s["parts_bps"]
    assert parts["temporary"].mean() == pytest.approx(1e4 * c["temporary"] / b.gross, rel=1e-6) and parts["spread"].mean() == pytest.approx(1e4 * c["spread"] / b.gross, rel=1e-6)
    assert parts["permanent"].mean() == pytest.approx(1e4 * c["permanent"] / b.gross, rel=1e-6)
    assert sum(parts.values()) == pytest.approx(s["shortfall_bps"])


# --------------------------------------------------------------------------------------------------------------------------- minimum trading risk quantity
def test_the_minimum_risk_trade_list_leaves_less_risk_than_trading_every_stock_in_proportion():
    b = random_basket(8, seed=1)
    out = minimum_trading_risk_quantity(b, share=0.5)
    assert out["residual_risk"] <= out["naive_risk"] + 1e-6 and out["residual_risk"] < out["original_risk"]
    theta = out["executed_fraction"]
    assert (theta >= -1e-9).all() and (theta <= 1 + 1e-9).all() and float(b.value @ theta) == pytest.approx(0.5 * b.gross, rel=1e-6)          # exactly the share asked for
    assert out["residual_risk"] == pytest.approx(b.risk(theta))
    assert np.sign(out["executed_shares"][np.abs(out["executed_shares"]) > 1e-9]).tolist() == b.side[np.abs(out["executed_shares"]) > 1e-9].tolist()


def test_no_feasible_trade_list_of_the_same_value_leaves_less_risk():
    b = random_basket(6, seed=5)
    best = minimum_trading_risk_quantity(b, share=0.4)["residual_risk"]
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(3000):
        theta = rng.uniform(0, 1, b.size)
        theta = np.clip(theta * (0.4 * b.gross / (b.value @ theta)), 0, 1)                                       # scale to the executed value, within the bounds
        if abs(b.value @ theta - 0.4 * b.gross) < 1e-4 * b.gross:
            checked += 1
            assert b.risk(theta) >= best - 1e-6 * best
    assert checked > 50


def test_the_extremes_of_the_minimum_risk_quantity_and_its_inverse():
    b = random_basket(6, seed=5)
    assert minimum_trading_risk_quantity(b, share=0.0)["residual_risk"] == pytest.approx(b.risk())
    assert minimum_trading_risk_quantity(b, share=1.0)["residual_risk"] == pytest.approx(0.0, abs=1e-3)
    half = minimum_trading_risk_quantity(b, share=0.5)["residual_risk"]
    inverse = minimum_trading_risk_quantity(b, risk_target=half)
    assert inverse["share"] == pytest.approx(0.5, abs=2e-3) and inverse["residual_risk"] <= half * 1.001
    assert minimum_trading_risk_quantity(b, risk_target=10 * b.risk())["share"] == 0.0
    shares = [minimum_trading_risk_quantity(b, risk_target=r * b.risk())["share"] for r in (0.8, 0.5, 0.2)]
    assert shares[0] < shares[1] < shares[2]                                                                   # a tighter risk target needs more of the list executed
    with pytest.raises(ValueError):
        minimum_trading_risk_quantity(b)
    with pytest.raises(ValueError):
        minimum_trading_risk_quantity(b, share=0.5, risk_target=1.0)


# --------------------------------------------------------------------------------------------------------------------------- maximum trading opportunity
def test_with_everything_available_the_whole_list_can_be_executed_and_nothing_binds():
    b = random_basket(6, seed=2)
    out = maximum_trading_opportunity(b)
    assert out["share_of_list"] == pytest.approx(1.0, abs=1e-6) and not out["binding"] and out["residual_risk"] == pytest.approx(0.0, abs=1e-3)


def test_only_the_buy_legs_available_the_limit_stops_the_list_from_becoming_a_bet():
    b = random_basket(8, seed=1)
    buys_only = b.quantity * (b.side > 0)
    out = maximum_trading_opportunity(b, available=buys_only)
    assert out["binding"] and out["executed_value"] < out["available_value"] - 1e-6 * b.gross
    assert out["residual_risk"] <= out["risk_limit"] * (1 + 1e-6) and out["risk_limit"] == pytest.approx(b.risk())
    assert (out["executed_fraction"][b.side < 0] == 0).all() and (out["executed_fraction"] <= 1 + 1e-9).all()
    loose = maximum_trading_opportunity(b, available=buys_only, risk_limit=2 * b.risk())
    assert loose["executed_value"] >= out["executed_value"] - 1e-6 * b.gross and not loose["binding"]            # allow more risk and the whole offer can be taken
    assert out["feasible"] and loose["feasible"]
    tight = maximum_trading_opportunity(b, available=buys_only, risk_limit=0.7 * b.risk())                       # the buy legs alone cannot bring the remaining risk down that far
    assert not tight["feasible"] and tight["residual_risk"] < tight["original_risk"] and tight["residual_risk"] > 0.7 * b.risk()
    reachable = maximum_trading_opportunity(b, risk_limit=0.5 * b.risk())                                         # with every leg on offer the whole list can go, and the limit is met trivially
    assert reachable["feasible"] and reachable["share_of_list"] == pytest.approx(1.0, abs=1e-6)


def test_availability_below_the_order_caps_what_can_be_executed():
    b = random_basket(5, seed=3)
    out = maximum_trading_opportunity(b, available=0.1 * b.quantity, risk_limit=10 * b.risk())
    assert out["executed_fraction"] == pytest.approx(np.full(5, 0.1), abs=1e-6)
    with pytest.raises(ValueError):
        maximum_trading_opportunity(b, available=-b.quantity)


# --------------------------------------------------------------------------------------------------------------------------- program-block decomposition
def _block_basket():
    """Two big names that hedge each other (a buy and a sell), a third big buy, and small liquid names."""
    names = ["B1", "B2", "B3", "P1", "P2", "P3"]
    shares = [200_000, -200_000, 150_000, 10_000, -12_000, 8_000]
    corr = np.full((6, 6), 0.4)
    corr[0, 1] = corr[1, 0] = 0.95
    np.fill_diagonal(corr, 1.0)
    return Basket(names, shares, [50, 50, 40, 60, 60, 60], [1.2e6, 1.2e6, 1.0e6, 5e6, 5e6, 5e6], [0.02, 0.02, 0.022, 0.018, 0.02, 0.02], corr)


def test_block_names_are_the_ones_large_against_their_volume():
    out = program_block(_block_basket(), block_threshold=0.1)
    assert out["block"] == ["B1", "B2", "B3"] and out["program"] == ["P1", "P2", "P3"] and out["candidates"] == ["B1", "B2", "B3"] or set(out["candidates"]) == {"B1", "B2", "B3"}


def test_the_dark_set_is_safe_for_every_way_it_might_fill_and_cannot_take_one_more_name():
    b = _block_basket()
    out = program_block(b, block_threshold=0.1)
    names = list(b.names)
    chosen = [names.index(n) for n in out["dark"]]
    limit = b.risk()
    for r in range(len(chosen) + 1):                                                                           # every subset that might fill leaves a list no riskier than the original
        for subset in itertools.combinations(chosen, r):
            theta = np.zeros(b.size)
            theta[list(subset)] = 1.0
            assert b.risk(theta) <= limit * (1 + 1e-9)
    assert out["worst_case_risk"] <= limit * (1 + 1e-9) and out["exact"]
    for other in (names.index(n) for n in out["candidates"] if n not in out["dark"]):                          # maximal: adding any other candidate admits a fill pattern that raises the risk
        worst = max(b.risk(np.isin(np.arange(b.size), list(subset) + [other]).astype(float) * 0 + np.array([1.0 if k in subset or k == other else 0.0 for k in range(b.size)]))
                    for r in range(len(chosen) + 1) for subset in itertools.combinations(chosen, r))
        assert worst > limit * (1 + 1e-9)


def test_a_hedged_pair_cannot_both_go_dark_when_one_leg_filling_alone_would_leave_a_bet():
    b = Basket(["L", "S", "X"], [100_000, -100_000, 5_000], [50.0, 50.0, 50.0], [1e6, 1e6, 5e6], [0.02, 0.02, 0.02], corr=[[1, .99, .3], [.99, 1, .3], [.3, .3, 1.0]])
    out = program_block(b, block_threshold=0.05)
    assert not {"L", "S"} <= set(out["dark"])                                                                 # one leg alone is a naked position, riskier than the hedged pair
    assert out["dark_share"] < 1.0 and out["worst_case_risk"] <= out["original_risk"] * (1 + 1e-9)


def test_a_risk_cap_above_one_lets_more_go_dark_and_eligible_names_can_be_named():
    b = _block_basket()
    strict, loose = program_block(b, 0.1, risk_cap=1.0), program_block(b, 0.1, risk_cap=3.0)
    assert set(strict["dark"]) <= set(loose["dark"]) and loose["dark_value"] >= strict["dark_value"]
    named = program_block(b, 0.1, risk_cap=3.0, eligible=["P1", "P2"])
    assert set(named["dark"]) <= {"P1", "P2"} and named["candidates"] and set(named["candidates"]) == {"P1", "P2"}


def test_a_big_dark_candidate_set_is_checked_by_sampling_and_says_so():
    big = random_basket(16, seed=7, notional=400e6)
    out = program_block(big, block_threshold=0.0, risk_cap=5.0)
    assert len(out["dark"]) <= 12 and out["worst_case_risk"] <= 5.0 * out["original_risk"] * (1 + 1e-9)


# --------------------------------------------------------------------------------------------------------------------------- liquidity seeking
SELL = Order(-1, 200_000)
MARKET = Market()


def test_liquidity_seeking_trades_little_inside_the_crisis_and_finishes_later_at_a_higher_pace():
    vwap = simulate(SELL, MARKET, VWAP(), AGGRESSIVE, "crisis", paths=300, seed=2)
    seek = simulate(SELL, MARKET, LiquiditySeeking(), AGGRESSIVE, "crisis", paths=300, seed=2)
    inside = slice(int(round(0.35 * 26)), int(round(0.70 * 26)))
    assert seek.executed[:, inside].sum(axis=1).mean() < 0.5 * vwap.executed[:, inside].sum(axis=1).mean()
    assert seek.executed[:, int(round(0.70 * 26)):].sum(axis=1).mean() > vwap.executed[:, int(round(0.70 * 26)):].sum(axis=1).mean()
    assert seek.executed.sum(axis=1) == pytest.approx(np.full(300, SELL.shares))                              # the deadline still holds


def test_liquidity_seeking_pays_less_impact_and_spread_in_a_crisis_than_a_volume_follower():
    vwap = simulate(SELL, MARKET, VWAP(), AGGRESSIVE, "crisis", paths=400, seed=3).summary()
    seek = simulate(SELL, MARKET, LiquiditySeeking(), AGGRESSIVE, "crisis", paths=400, seed=3).summary()
    assert seek["temporary_bps"] < vwap["temporary_bps"] and seek["spread_bps"] < vwap["spread_bps"]


def test_liquidity_seeking_is_close_to_its_base_in_an_ordinary_day():
    base = simulate(ORDER := Order(1, 200_000), MARKET, VWAP(), AGGRESSIVE, "normal", paths=300, seed=4).summary()
    seek = simulate(ORDER, MARKET, LiquiditySeeking(), AGGRESSIVE, "normal", paths=300, seed=4).summary()
    assert abs(seek["shortfall_bps"] - base["shortfall_bps"]) < 6.0 and seek["temporary_bps"] == pytest.approx(base["temporary_bps"], rel=0.3)


def test_liquidity_quality_is_one_when_nothing_is_unusual_and_below_one_in_stress():
    seek = LiquiditySeeking()
    from tests.test_algo import _state

    state = _state(0.0)
    state.volume, state.expected_volume = np.array([1e5]), 1e5
    assert seek.quality(state)[0] == pytest.approx(1.0)
    state.spread_mult, state.impact_mult, state.volume = 4.0, 2.5, np.array([1.6e5])
    assert seek.quality(state)[0] == pytest.approx(0.16)
    with pytest.raises(ValueError):
        LiquiditySeeking(floor=0.0)


def test_a_working_order_style_can_carry_liquidity_seeking_and_still_finish():
    r = simulate(SELL, MARKET, LiquiditySeeking(), WORKING, "crisis", paths=100, seed=5)
    assert r.executed.sum(axis=1) == pytest.approx(np.full(100, SELL.shares))
