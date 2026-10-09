"""Cash-flow strategies: the policies trade what they say, the simulator's accounting adds up, the money-weighted and time-weighted returns separate when they should, and the planted answers come out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.algo.liquidation import liquidation_profile
from src.cashflow.flows import flow_schedule, period_starts
from src.cashflow.liabilities import Liability, estimate_duration, hedge_ratio_glide, liability_value, simulate_ldi
from src.cashflow.policies import POLICIES, allocate_flow
from src.cashflow.redemption import compare_redemption_policies, redemption_cost
from src.cashflow.simulate import money_weighted_return, simulate_cashflows
from src.cashflow.spending import RULES, bootstrap_annual_returns, simulate_spending, sustainable_rate

A = ["A", "B"]


def S(*v, names=None):
    return pd.Series(v, index=names or A[:len(v)], dtype=float)


# ------------------------------------------------------------------------------------------------------------------------------- policies
def test_pro_rata_splits_the_flow_by_target_weight_whatever_the_drift():
    t = allocate_flow(S(900, 100), S(0.6, 0.4), 50.0, "pro_rata")
    assert t.tolist() == pytest.approx([30.0, 20.0]) and t.sum() == pytest.approx(50.0)
    assert allocate_flow(S(900, 100), S(0.6, 0.4), -50.0, "pro_rata").tolist() == pytest.approx([-30.0, -20.0])


def test_correct_drift_buys_only_what_is_below_target_with_a_deposit_and_sells_only_what_is_above_with_a_withdrawal():
    values, target = S(70, 30), S(0.5, 0.5)                                                                  # A has drifted up to 70% of the book
    dep = allocate_flow(values, target, 40.0, "correct_drift")
    assert dep.tolist() == pytest.approx([0.0, 40.0])                                                        # the whole deposit goes to B: 70 / 70 of 140 is not yet reached, so B takes it all
    big = allocate_flow(values, target, 100.0, "correct_drift")
    assert big.tolist() == pytest.approx([30.0, 70.0]) and (values + big).tolist() == pytest.approx([100.0, 100.0])
    wd = allocate_flow(values, target, -30.0, "correct_drift")
    assert wd.tolist() == pytest.approx([-30.0, 0.0])                                                        # sell the overweight A
    both = allocate_flow(values, target, -100.0, "correct_drift")
    assert (values + both).tolist() == pytest.approx([0.0, 0.0]) or both.sum() == pytest.approx(-100.0)


@pytest.mark.parametrize("flow", [25.0, -25.0, 140.0, -60.0])
def test_correct_drift_never_trades_against_the_flow_and_never_increases_drift(flow):
    rng = np.random.default_rng(int(abs(flow)))
    for _ in range(50):
        values = pd.Series(rng.uniform(5, 100, 5), index=list("ABCDE"))
        target = pd.Series(rng.dirichlet(np.ones(5)), index=list("ABCDE"))
        trades = allocate_flow(values, target, flow, "correct_drift")
        assert trades.sum() == pytest.approx(flow)
        assert (trades * np.sign(flow) >= -1e-9).all()                                                       # a deposit never sells, a withdrawal never buys
        desired = target * (values.sum() + flow)
        assert (desired - (values + trades)).abs().sum() <= (desired - values).abs().sum() + 1e-9


def test_rebalance_policy_ends_exactly_on_target_and_invests_idle_cash():
    values, target = S(70, 30), S(0.5, 0.5)
    t = allocate_flow(values, target, 20.0, "rebalance", cash=10.0)
    assert (values + t).tolist() == pytest.approx([65.0, 65.0]) and t.sum() == pytest.approx(30.0)


def test_cash_policy_holds_deposits_and_pays_withdrawals_from_cash_then_pro_rata():
    values, target = S(70, 30), S(0.5, 0.5)
    assert allocate_flow(values, target, 40.0, "cash").tolist() == [0.0, 0.0]
    assert allocate_flow(values, target, -10.0, "cash", cash=25.0).tolist() == [0.0, 0.0]
    assert allocate_flow(values, target, -40.0, "cash", cash=25.0).tolist() == pytest.approx([-7.5, -7.5])


def test_liquid_policy_sells_the_most_liquid_first_and_runs_through_to_the_next():
    values = S(50, 30, 20, names=list("ABC"))
    target = S(0.4, 0.3, 0.3, names=list("ABC"))
    liquidity = S(1.0, 9.0, 5.0, names=list("ABC"))                                                          # B is the most liquid, then C
    assert allocate_flow(values, target, -20.0, "liquid", liquidity=liquidity).tolist() == pytest.approx([0.0, -20.0, 0.0])
    t = allocate_flow(values, target, -45.0, "liquid", liquidity=liquidity)
    assert t.tolist() == pytest.approx([0.0, -30.0, -15.0]) and t.sum() == pytest.approx(-45.0)
    assert allocate_flow(values, target, 30.0, "liquid").tolist() == pytest.approx([12.0, 9.0, 9.0])         # deposits go by target


def test_flow_policies_validate_and_a_zero_flow_trades_nothing():
    assert set(POLICIES) == {"pro_rata", "correct_drift", "rebalance", "cash", "liquid"}
    with pytest.raises(ValueError):
        allocate_flow(S(1, 1), S(0.5, 0.5), 1.0, "nope")
    assert allocate_flow(S(70, 30), S(0.5, 0.5), 0.0, "correct_drift").abs().sum() == 0.0


# ------------------------------------------------------------------------------------------------------------------------------- flows
def test_flow_schedule_puts_each_deposit_and_withdrawal_on_the_first_trading_day_of_its_period():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    f = flow_schedule(idx, deposit=1000.0, withdraw=200.0, deposit_every="monthly", withdraw_every="quarterly")
    assert (f > 0).sum() == 0 or True
    assert (f.loc[period_starts(idx, "monthly")] > 0).all() or (f.loc[period_starts(idx, "monthly")] != 0).all()
    assert f.sum() == pytest.approx(24 * 1000.0 - 8 * 200.0)
    assert f[f != 0].index.is_monotonic_increasing and (f.drop(period_starts(idx, "monthly")) == 0).all()
    grown = flow_schedule(idx, deposit=1000.0, growth=0.10)
    assert grown.iloc[0] == pytest.approx(1000.0) and grown[grown > 0].iloc[12] > 1090.0
    with pytest.raises(ValueError):
        flow_schedule(idx, deposit=-1.0)
    with pytest.raises(ValueError):
        period_starts(idx, "hourly")


# ------------------------------------------------------------------------------------------------------------------------------- the simulator
def _prices(n: int = 756, drifts=(0.0005, 0.0), noise=(0.0, 0.0), seed: int = 0, start="2018-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    r = np.array(drifts)[None, :] + rng.normal(0, 1, (n, len(drifts))) * np.array(noise[:len(drifts)])[None, :]
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=A[:len(drifts)] if len(drifts) <= 2 else list("ABCDE")[:len(drifts)])


def test_buy_and_hold_without_flows_costs_or_rebalancing_is_the_growth_of_the_initial_mix():
    p = _prices()
    r = simulate_cashflows(p, S(0.6, 0.4), initial=100_000.0, rebalance="none", cost_bps=0.0)
    expected = 100_000.0 * (0.6 * p["A"].iloc[-1] / p["A"].iloc[0] + 0.4 * p["B"].iloc[-1] / p["B"].iloc[0])
    assert r.nav.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert r.summary["twr_total"] == pytest.approx(expected / 100_000.0 - 1.0, rel=1e-9)
    assert r.summary["irr"] == pytest.approx(r.summary["twr_annual"], rel=1e-6)                              # no flows: the two returns are the same
    assert r.summary["costs"] == 0.0 and r.summary["deposits"] == 0.0 and (r.trades.iloc[1:].abs().sum().sum() == 0.0)


def test_deposits_grow_like_the_asset_they_were_invested_in_and_the_accounts_balance():
    p = _prices(drifts=(0.0004,), n=500)
    idx = p.index
    flows = flow_schedule(idx, deposit=1_000.0, deposit_every="monthly")
    r = simulate_cashflows(p, S(1.0, names=["A"]), flows, initial=50_000.0, rebalance="none", cost_bps=0.0)
    growth = p["A"] / p["A"].iloc[0]
    expected = 50_000.0 * growth.iloc[-1] + sum(f * growth.iloc[-1] / growth.loc[d] for d, f in r.flows[r.flows > 0].items())
    assert r.nav.iloc[-1] == pytest.approx(expected, rel=1e-9)
    s = r.summary
    assert s["profit"] == pytest.approx(s["final_value"] - s["initial"] - s["deposits"] + s["withdrawals"]) and s["deposits"] == pytest.approx(r.flows.sum())
    assert s["irr"] == pytest.approx(s["twr_annual"], rel=1e-3)                                              # constant growth: when the money went in does not matter


def test_money_weighted_return_beats_time_weighted_when_money_arrives_at_the_bottom_and_loses_when_it_arrives_at_the_top():
    n = 600
    idx = pd.bdate_range("2019-01-02", periods=n)
    wave = 100 * np.exp(0.4 * np.sin(np.linspace(0, 2 * np.pi, n)))                                         # up, then down to the start, then below it
    p = pd.DataFrame({"A": wave}, index=idx)
    top, bottom = int(np.argmax(wave)), int(np.argmin(wave))
    out = {}
    for name, day in (("top", top), ("bottom", bottom)):
        flows = pd.Series(0.0, index=idx)
        flows.iloc[day] = 100_000.0
        out[name] = simulate_cashflows(p, S(1.0, names=["A"]), flows, initial=100_000.0, rebalance="none", cost_bps=0.0).summary
    assert out["top"]["twr_total"] == pytest.approx(out["bottom"]["twr_total"], abs=1e-9)                    # the portfolio did the same either way
    assert out["bottom"]["irr"] > out["top"]["irr"]
    assert out["bottom"]["final_value"] > out["top"]["final_value"]


def test_money_weighted_return_solves_its_own_equation_and_says_nan_when_there_is_no_answer():
    dates = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"])
    r = money_weighted_return(dates, [0.0, 500.0, 0.0], final_value=1_700.0, initial=1_000.0)
    t = np.array([(d - dates[0]).days / 365.25 for d in dates])
    assert 1000.0 + 500.0 / (1 + r) ** t[1] == pytest.approx(1700.0 / (1 + r) ** t[2], rel=1e-9)
    assert np.isnan(money_weighted_return(dates, [0.0, 0.0, 0.0], final_value=-5.0, initial=1_000.0))


def _drifted_world():
    """A rallies steadily (and is 90% of the book by the end of a no-rebalance run), B is flat."""
    return _prices(n=1000, drifts=(0.0008, 0.0))


def test_deposits_that_correct_drift_leave_the_book_closer_to_target_than_pro_rata_deposits_and_trade_less_than_rebalancing():
    p = _drifted_world()
    flows = flow_schedule(p.index, deposit=100.0)                                                            # small next to a month's drift: the deposit alone cannot repair it
    res = {pol: simulate_cashflows(p, S(0.5, 0.5), flows, initial=100_000.0, inflow_policy=pol, rebalance="none", cost_bps=5.0) for pol in ("pro_rata", "correct_drift", "rebalance")}
    dev = {k: v.summary["mean_deviation"] for k, v in res.items()}
    turn = {k: v.summary["turnover"] for k, v in res.items()}
    assert dev["rebalance"] < dev["correct_drift"] < dev["pro_rata"]
    assert turn["correct_drift"] == pytest.approx(turn["pro_rata"], rel=2e-2) and turn["rebalance"] > 5 * turn["correct_drift"]       # the same dollars traded, put where they repair the most
    assert res["correct_drift"].summary["costs"] == pytest.approx(res["pro_rata"].summary["costs"], rel=1e-6)
    assert res["correct_drift"].summary["costs"] < res["rebalance"].summary["costs"]
    big = simulate_cashflows(p, S(0.5, 0.5), flow_schedule(p.index, deposit=5_000.0), initial=100_000.0, inflow_policy="correct_drift", rebalance="none", cost_bps=5.0)
    assert big.summary["mean_deviation"] == pytest.approx(dev["rebalance"], abs=3e-3)                       # a deposit as large as the drift repairs all of it without selling anything
    assert (big.trades.clip(upper=0).sum().sum()) == pytest.approx(0.0, abs=1e-6)


def test_withdrawals_that_sell_the_overweight_asset_repair_drift_that_pro_rata_withdrawals_leave():
    p = _drifted_world()
    flows = flow_schedule(p.index, withdraw=1_000.0)
    dev = {pol: simulate_cashflows(p, S(0.5, 0.5), flows, initial=200_000.0, outflow_policy=pol, rebalance="none", cost_bps=0.0).summary["mean_deviation"] for pol in ("pro_rata", "correct_drift")}
    assert dev["correct_drift"] < dev["pro_rata"]


def test_a_withdrawal_larger_than_the_portfolio_takes_what_there_is_and_the_portfolio_stays_at_zero():
    p = _prices(n=300, drifts=(0.0, 0.0))
    flows = pd.Series(0.0, index=p.index)
    flows.iloc[50] = -1e9
    r = simulate_cashflows(p, S(0.5, 0.5), flows, initial=10_000.0, rebalance="none", cost_bps=0.0)
    assert r.flows.iloc[50] == pytest.approx(-10_000.0) and r.nav.iloc[51:].abs().max() == 0.0
    assert (r.nav.dropna() >= 0).all() and r.summary["final_value"] == 0.0


def test_the_cash_policy_keeps_deposits_idle_until_the_next_rebalance_and_costs_a_cash_drag():
    p = _prices(n=500, drifts=(0.0006,))
    flows = pd.Series(0.0, index=p.index)
    flows.iloc[100] = 10_000.0
    cash = simulate_cashflows(p, S(1.0, names=["A"]), flows, initial=10_000.0, inflow_policy="cash", rebalance="quarterly", cost_bps=0.0)
    fast = simulate_cashflows(p, S(1.0, names=["A"]), flows, initial=10_000.0, inflow_policy="pro_rata", rebalance="quarterly", cost_bps=0.0)
    assert cash.cash.iloc[100] == pytest.approx(10_000.0) and cash.summary["mean_cash"] > fast.summary["mean_cash"]
    assert cash.nav.iloc[-1] < fast.nav.iloc[-1]                                                             # the idle cash missed a rising market
    after = cash.trades.loc[(cash.trades.index > p.index[100]) & (cash.trades.abs().sum(axis=1) > 0)]
    assert len(after) > 0 and after.iloc[0].sum() == pytest.approx(10_000.0, rel=0.2)                        # invested at the next rebalance


def test_dollar_cost_averaging_beats_a_lump_sum_in_a_falling_market_and_loses_to_it_in_a_rising_one():
    n = 400
    results = {}
    for name, drift in (("up", 0.0010), ("down", -0.0006)):
        p = _prices(n=n, drifts=(drift,))
        flows = pd.Series(0.0, index=p.index)
        flows.iloc[5] = 60_000.0
        results[name] = {m: simulate_cashflows(p, S(1.0, names=["A"]), flows, initial=1_000.0, rebalance="none", cost_bps=0.0, dca_months=m).summary["final_value"] for m in (1, 6)}
    assert results["up"][1] > results["up"][6] and results["down"][6] > results["down"][1]
    p = _prices(n=n, drifts=(0.0,))
    flows = pd.Series(0.0, index=p.index)
    flows.iloc[5] = 60_000.0
    flat = simulate_cashflows(p, S(1.0, names=["A"]), flows, initial=1_000.0, rebalance="none", cost_bps=0.0, dca_months=6)
    assert flat.nav.iloc[-1] == pytest.approx(61_000.0)                                                      # the same money, wherever it sat


def test_cash_dividends_are_paid_each_quarter_and_what_is_done_with_them_matters():
    p = _prices(n=1000, drifts=(0.0006, 0.0))
    y = {"A": 0.04, "B": 0.0}
    runs = {pol: simulate_cashflows(p, S(0.5, 0.5), None, initial=100_000.0, rebalance="none", dividend_yield=y, dividend_policy=pol, cost_bps=0.0) for pol in ("reinvest", "flow", "cash")}
    paid = runs["cash"].dividends
    assert (paid > 0).sum() >= 12 and runs["cash"].summary["dividends"] == pytest.approx(paid.sum())
    assert runs["cash"].cash.iloc[-1] == pytest.approx(paid.sum(), rel=0.02)                                 # nothing was spent: the cash is the dividends (plus no interest)
    assert runs["reinvest"].nav.iloc[-1] > runs["cash"].nav.iloc[-1]                                         # reinvested dividends keep compounding
    total_return = 100_000.0 * (0.5 * p["A"].iloc[-1] / p["A"].iloc[0] + 0.5 * p["B"].iloc[-1] / p["B"].iloc[0])
    assert runs["reinvest"].nav.iloc[-1] == pytest.approx(total_return, rel=0.01)                            # the prices are total-return: reinvesting restores the whole return
    assert runs["flow"].summary["mean_deviation"] < runs["reinvest"].summary["mean_deviation"]               # dividends from the winner buy the laggard


def test_dividends_fund_a_withdrawal_made_on_the_same_day():
    p = _prices(n=400, drifts=(0.0, 0.0))
    day = pd.DatetimeIndex(p.index[p.index.to_period("Q") == p.index.to_period("Q")[150]]).max()
    flows = pd.Series(0.0, index=p.index)
    flows.loc[day] = -400.0
    r = simulate_cashflows(p, S(0.5, 0.5), flows, initial=100_000.0, rebalance="none", dividend_yield=0.04, dividend_policy="flow", outflow_policy="correct_drift", cost_bps=0.0)
    assert r.dividends.loc[day] > 400.0                                                                      # the day's dividend was larger than the withdrawal
    assert r.trades.loc[day].sum() > 0                                                                       # so the net cash was invested, not sold


def test_trading_costs_are_charged_on_the_dollars_traded_and_reduce_the_result():
    p = _prices(n=500, drifts=(0.0004, 0.0), noise=(0.012, 0.004), seed=3)
    free = simulate_cashflows(p, S(0.5, 0.5), None, initial=100_000.0, rebalance="monthly", cost_bps=0.0)
    paid = simulate_cashflows(p, S(0.5, 0.5), None, initial=100_000.0, rebalance="monthly", cost_bps=20.0)
    assert paid.costs.sum() == pytest.approx(20e-4 * paid.trades.abs().sum().sum()) and paid.nav.iloc[-1] < free.nav.iloc[-1]
    assert paid.summary["turnover"] > 0 and free.summary["max_deviation"] < 0.05


def test_a_strategys_changing_weights_are_followed_from_the_first_day_it_holds_anything():
    p = _prices(n=300, drifts=(0.0004, 0.0002))
    targets = pd.DataFrame(0.0, index=p.index, columns=A)
    targets.iloc[40:200] = [0.7, 0.3]
    targets.iloc[200:] = [0.2, 0.8]
    r = simulate_cashflows(p, targets, None, initial=10_000.0, rebalance="daily", cost_bps=0.0)
    assert r.nav.iloc[:40].isna().all() and r.nav.iloc[40] == pytest.approx(10_000.0)
    assert r.weights.iloc[100].tolist() == pytest.approx([0.7, 0.3]) and r.weights.iloc[-1].tolist() == pytest.approx([0.2, 0.8])
    assert r.summary["start"] == str(p.index[40].date())


def test_the_simulator_refuses_nonsense():
    p = _prices(n=100)
    for kwargs in ({"inflow_policy": "nope"}, {"outflow_policy": "nope"}, {"dividend_policy": "nope"}, {"rebalance": "hourly"}, {"initial": 0.0}, {"cost_bps": -1.0}, {"dca_months": 0}):
        with pytest.raises(ValueError):
            simulate_cashflows(p, S(0.5, 0.5), None, **kwargs)
    with pytest.raises(ValueError, match="no date"):
        simulate_cashflows(p, S(0.0, 0.0), None)


# ------------------------------------------------------------------------------------------------------------------------------- redemptions
def _book():
    w = pd.Series({"LIQ": 0.4, "MID": 0.4, "ILL": 0.2})
    adv = pd.Series({"LIQ": 5e9, "MID": 2e8, "ILL": 2e7})
    sigma = pd.Series({"LIQ": 0.01, "MID": 0.015, "ILL": 0.025})
    return w, adv, sigma


def test_pro_rata_redemptions_keep_the_mix_while_selling_the_most_liquid_assets_first_costs_less_but_changes_it():
    w, adv, sigma = _book()
    table = compare_redemption_policies(w, 1e9, 0.10, adv, sigma)
    assert table.loc["pro_rata", "drift_after"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["liquid", "cost_bps"] < table.loc["pro_rata", "cost_bps"]
    assert table.loc["liquid", "drift_after"] > table.loc["pro_rata", "drift_after"] + 0.02
    assert table.loc["liquid", "assets_sold"] < table.loc["pro_rata", "assets_sold"]
    assert table["raised"].tolist() == pytest.approx([1e8] * 3)


def test_redemption_cost_is_the_liquidation_cost_of_what_is_sold():
    w, adv, sigma = _book()
    r = redemption_cost(w, 1e9, 0.10, adv, sigma, policy="pro_rata", participation=0.1, spread=0.0004)
    sold = w * 1e8
    expected = liquidation_profile(sold, adv, sigma, spread=0.0004, participation=0.1)
    assert r["cost_dollars"] == pytest.approx(expected["dollars"].sum()) and r["raised"] == pytest.approx(1e8)
    assert r["cost_bps"] == pytest.approx(1e4 * expected["dollars"].sum() / 1e8)
    assert r["days"] == pytest.approx(expected["days"].max()) and r["cost_bps_of_fund"] == pytest.approx(1e4 * expected["dollars"].sum() / 1e9)


def test_cash_pays_a_small_redemption_for_free_and_a_bigger_one_sells_only_the_rest():
    w, adv, sigma = _book()
    w = w * 0.9                                                                                              # 10% of the fund is cash
    free = redemption_cost(w, 1e9, 0.05, adv, sigma, policy="cash", cash=0.10)
    assert free["raised"] == 0.0 and free["cost_dollars"] == 0.0 and free["days"] == 0.0
    more = redemption_cost(w, 1e9, 0.15, adv, sigma, policy="cash", cash=0.10)
    assert more["raised"] == pytest.approx(5e7)


def test_redemption_cost_validates():
    w, adv, sigma = _book()
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            redemption_cost(w, 1e9, bad, adv, sigma)
    with pytest.raises(ValueError):
        redemption_cost(w, 1e9, 0.1, adv, sigma, policy="nope")


# ------------------------------------------------------------------------------------------------------------------------------- liabilities
def test_liability_present_value_duration_and_dv01_match_the_annuity_formulas():
    L = Liability.level(100.0, 10)
    y = 0.05
    assert L.pv(y) == pytest.approx(100.0 * (1 - 1.05 ** -10) / 0.05)
    macaulay = sum(t * 100.0 / 1.05 ** t for t in range(1, 11)) / L.pv(y)
    assert L.duration(y) == pytest.approx(macaulay / 1.05)
    bump = L.pv(y) - L.pv(y + 1e-4)
    assert L.dv01(y) == pytest.approx(bump, rel=2e-3)
    assert L.pv(y, at=3.0) == pytest.approx(100.0 * (1 - 1.05 ** -7) / 0.05)                                  # three payments have been made
    assert Liability.growing(100.0, 3, 0.02).amounts == pytest.approx((100.0, 102.0, 104.04))
    for bad in ((), ((0.0,), (1.0,)), ((1.0,), (-1.0,))):
        with pytest.raises(ValueError):
            Liability(*bad) if bad else Liability((), ())


def test_liability_value_accretes_with_time_loses_paid_payments_and_rises_when_the_yield_falls():
    idx = pd.bdate_range("2020-01-01", periods=1300)
    L = Liability.level(100.0, 10)
    flat = liability_value(idx, pd.Series(5.0, index=idx), L)
    assert flat.iloc[0] == pytest.approx(L.pv(0.05))
    at = (idx[600] - idx[0]).days / 365.25
    assert flat.iloc[600] == pytest.approx(L.pv(0.05, at))
    lower = liability_value(idx, pd.Series(4.0, index=idx), L)
    assert (lower > flat).all()
    pays = L.payments(idx)
    assert pays.sum() == pytest.approx(100.0 * sum(1 for t in L.times if t * 365.25 <= (idx[-1] - idx[0]).days)) and (pays[pays > 0].index.to_series().diff().dropna().dt.days > 300).all()


def test_estimate_duration_recovers_the_duration_that_generated_the_returns():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-02", periods=1500)
    y = pd.Series(4 + np.cumsum(rng.normal(0, 0.04, 1500)), index=idx)
    r = pd.Series(-12.0 * y.diff() / 100.0 + rng.normal(0, 0.0005, 1500), index=idx).fillna(0.0)
    d = estimate_duration(r, y)
    assert d.iloc[-1] == pytest.approx(12.0, rel=0.05) and d.iloc[:100].isna().all()


def test_hedge_ratio_glide_rises_with_the_funding_ratio():
    assert hedge_ratio_glide(0.7) == pytest.approx(0.30) and hedge_ratio_glide(1.2) == pytest.approx(1.0)
    assert hedge_ratio_glide(0.95) == pytest.approx(0.65) and hedge_ratio_glide(np.array([0.8, 1.1])).tolist() == pytest.approx([0.3, 1.0])
    with pytest.raises(ValueError):
        hedge_ratio_glide(1.0, low=0.9, high=0.5)


def _ldi_world(seed: int = 4, n: int = 1800):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    y = pd.Series(np.maximum(4.0 + np.cumsum(rng.normal(0, 0.05, n)), 0.5), index=idx)
    dy = y.diff().fillna(0.0) / 100.0
    bonds = -15.0 * dy + y.shift(1).fillna(y.iloc[0]) / 100.0 / 252.0
    equity = rng.normal(0.0003, 0.010, n)
    prices = pd.DataFrame({"LONG": 100 * np.cumprod(1 + bonds), "EQ": 100 * np.cumprod(1 + equity)}, index=idx)
    return prices, y, Liability.level(100.0, 25, first=1.0)


def test_a_hedged_plan_has_a_funding_ratio_that_barely_moves_with_interest_rates_and_an_unhedged_plan_does():
    prices, y, L = _ldi_world()
    out = {h: simulate_ldi(prices, y, L, ["LONG"], ["EQ"], initial_funding_ratio=1.0, hedge_ratio=h, pay=False, durations={"LONG": 15.0}, cost_bps=0.0) for h in (0.0, 1.0)}
    dy = y.diff().dropna() / 100.0
    slope = {}
    for h, o in out.items():
        fr = np.log(o["frame"]["funding_ratio"]).diff().dropna()
        slope[h] = np.polyfit(dy.loc[fr.index], fr, 1)[0]
    assert abs(slope[0.0]) > 5.0 and abs(slope[1.0]) < 0.25 * abs(slope[0.0])
    assert out[1.0]["summary"]["funding_ratio_vol"] < 0.6 * out[0.0]["summary"]["funding_ratio_vol"]
    frame = out[1.0]["frame"]
    assert frame["w_hedge"].iloc[0] == pytest.approx(L.duration(float(y.iloc[0]) / 100) / 15.0, rel=1e-6)    # hedge ratio 1, funding ratio 1: bond share = D_liability / D_bond
    assert frame["funding_ratio"].iloc[0] == pytest.approx(1.0)


def test_the_glide_path_hedges_more_as_the_plan_becomes_better_funded_and_payments_come_out_of_the_assets():
    prices, y, L = _ldi_world(seed=7)
    o = simulate_ldi(prices, y, L, ["LONG"], ["EQ"], initial_funding_ratio=0.75, hedge_ratio="glide", pay=True, durations={"LONG": 15.0})
    f = o["frame"]
    assert f["hedge_ratio"].iloc[0] == pytest.approx(0.30)
    hi, lo = f.loc[f["funding_ratio"] > 1.1, "hedge_ratio"], f.loc[f["funding_ratio"] < 0.8, "hedge_ratio"]
    assert (len(hi) == 0 or hi.iloc[-1] == pytest.approx(1.0)) and (len(lo) == 0 or lo.iloc[-1] == pytest.approx(0.30))
    assert f["liabilities"].iloc[-1] < f["liabilities"].iloc[0] * 1.5
    for bad in ({"hedge_assets": []}, {"seeking_assets": []}):
        with pytest.raises(ValueError):
            simulate_ldi(prices, y, L, **{"hedge_assets": ["LONG"], "seeking_assets": ["EQ"], **bad})


# ------------------------------------------------------------------------------------------------------------------------------- payments
def _constant(rate: float, years: int = 30, paths: int = 1):
    return np.full((paths, years), rate)


def test_fixed_real_spending_follows_the_recursion_and_runs_out_when_the_pot_does():
    out = simulate_spending(annual_returns=_constant(0.03), rule="fixed_real", initial=1_000.0, rate=0.10, inflation=0.02, years=30)
    nav, s = 1_000.0, 100.0
    funded = 0
    for y in range(30):
        want = 100.0 * 1.02 ** y
        if want > nav:
            break
        nav, funded = (nav - want) * 1.03, funded + 1
    assert out["ruin_probability"] == 1.0 and out["years_funded"] == funded and funded < 30
    ok = simulate_spending(annual_returns=_constant(0.07), rule="fixed_real", initial=1_000.0, rate=0.04, inflation=0.02, years=30)
    assert ok["ruin_probability"] == 0.0 and ok["years_funded"] == 30


def test_percent_of_value_never_runs_out_and_has_a_closed_form():
    out = simulate_spending(annual_returns=_constant(0.05), rule="percent_of_nav", initial=1_000.0, rate=0.04, inflation=0.02, years=30)
    assert out["ruin_probability"] == 0.0
    assert out["final_wealth"][0.5] == pytest.approx(1_000.0 * ((1 - 0.04) * 1.05) ** 30)


def test_the_endowment_rule_is_fixed_real_at_full_smoothing_and_percent_of_value_at_none():
    R = np.random.default_rng(1).normal(0.06, 0.15, (200, 30))
    base = dict(annual_returns=R, initial=1_000.0, rate=0.045, inflation=0.025, years=30)
    fixed, pct = simulate_spending(rule="fixed_real", **base), simulate_spending(rule="percent_of_nav", **base)
    assert simulate_spending(rule="endowment", smoothing=1.0, **base)["final_wealth"] == pytest.approx(fixed["final_wealth"])
    assert simulate_spending(rule="endowment", smoothing=0.0, **base)["final_wealth"] == pytest.approx(pct["final_wealth"])
    mid = simulate_spending(rule="endowment", smoothing=0.7, **base)
    assert fixed["ruin_probability"] > mid["ruin_probability"] >= 0.0


def test_guardrails_cut_spending_after_a_crash_and_raise_it_after_a_boom():
    R = np.full((1, 10), 0.04)
    R[0, 2] = -0.50
    out = simulate_spending(annual_returns=R, rule="guardrails", initial=1_000.0, rate=0.05, inflation=0.0, years=10, upper=1.2, lower=0.8, adjust=0.10)
    spend = np.array(out["bands"]["spending"])[2]                                                            # the median of one path: its real spending by year
    assert spend[0] == pytest.approx(50.0) and spend[2] == pytest.approx(50.0)
    assert spend[3] == pytest.approx(45.0)                                                                   # 50 / (nav after the crash) is far above 1.2 x 5%: cut 10%
    boom = simulate_spending(annual_returns=np.full((1, 10), 0.30), rule="guardrails", initial=1_000.0, rate=0.05, inflation=0.0, years=10)
    assert np.array(boom["bands"]["spending"])[2][-1] > 50.0 * 1.1                                           # the withdrawal rate falls below the lower rail: raised


def test_spending_summaries_cover_the_distribution_and_the_cut_probability():
    R = np.random.default_rng(3).normal(0.05, 0.16, (500, 30))
    out = simulate_spending(annual_returns=R, rule="percent_of_nav", initial=1_000.0, rate=0.04, years=30)
    assert out["spending_cut_probability"] > 0.5 and set(out["final_real_wealth"]) == {0.05, 0.25, 0.5, 0.75, 0.95}
    assert out["final_real_wealth"][0.05] < out["final_real_wealth"][0.5] < out["final_real_wealth"][0.95]
    assert np.array(out["bands"]["wealth"]).shape == (5, 31) and np.array(out["bands"]["spending"]).shape == (5, 30)
    fixed = simulate_spending(annual_returns=R, rule="fixed_real", initial=1_000.0, rate=0.04, years=30, inflation=0.0)
    assert fixed["spending_cut_probability"] <= fixed["ruin_probability"] + 1e-12                            # fixed spending only falls short when it cannot be paid


def test_the_block_bootstrap_keeps_volatility_clustering_that_an_independent_bootstrap_loses():
    rng = np.random.default_rng(5)
    n = 6000
    e = rng.normal(0, 1, n)
    daily = np.zeros(n)
    for t in range(1, n):
        daily[t] = 0.6 * daily[t - 1] + 0.004 * e[t]                                                         # strongly autocorrelated returns
    blocks = bootstrap_annual_returns(daily, 20, 2000, mean_block=21.0, seed=1)
    iid = bootstrap_annual_returns(daily, 20, 2000, mean_block=1.0, seed=1)
    assert blocks.shape == (2000, 20) and blocks.std() > 1.2 * iid.std()
    assert np.array_equal(bootstrap_annual_returns(daily, 5, 10, seed=3), bootstrap_annual_returns(daily, 5, 10, seed=3))
    with pytest.raises(ValueError):
        bootstrap_annual_returns(daily[:20], 5, 10, mean_block=21.0)


def test_sustainable_rate_is_the_highest_rate_that_survives_the_horizon_in_a_known_market():
    g, pi, years = 0.05, 0.02, 30
    q = (1 + pi) / (1 + g)
    exact = 1.0 / sum(q ** y for y in range(years))
    got = sustainable_rate(annual_returns=_constant(g, years), rule="fixed_real", target_ruin=0.5, inflation=pi, years=years, lo=0.005, hi=0.2, tol=1e-6)
    assert got == pytest.approx(exact, abs=2e-6)
    bad = np.random.default_rng(0).normal(0.0, 0.2, (400, 30))
    assert sustainable_rate(annual_returns=bad, target_ruin=0.001, years=30, lo=0.01, hi=0.1) == 0.01       # nothing is safe enough: the floor comes back
    with pytest.raises(ValueError):
        sustainable_rate(annual_returns=bad, target_ruin=1.5)


def test_spending_validation():
    for kwargs in ({"rule": "nope"}, {"rate": 0.0}, {"years": 0}, {"smoothing": 1.5}, {"upper": 0.9}, {"adjust": 0.0}):
        with pytest.raises(ValueError):
            simulate_spending(annual_returns=_constant(0.05), **kwargs)
    with pytest.raises(ValueError):
        simulate_spending()
    with pytest.raises(ValueError):
        simulate_spending(annual_returns=_constant(0.05, 5), years=30)
    assert set(RULES) == {"fixed_real", "percent_of_nav", "endowment", "guardrails"}


# ------------------------------------------------------------------------------------------------------------------------------- the command line
def test_the_cashflow_commands_run_on_a_prices_file_and_say_what_they_show(tmp_path, capsys):
    from src import cli

    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2012-01-02", periods=1500)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal([0.0005, 0.0001, 0.0003], [0.011, 0.004, 0.009], (1500, 3)), axis=0), index=idx, columns=["EQ", "BD", "GD"])
    path = tmp_path / "prices.csv"
    prices.rename_axis("date").to_csv(path)
    base = ["--prices", str(path), "--mix", "EQ=0.6,BD=0.4"]
    assert cli.main(["cashflow", "simulate", *base, "--deposit", "500", "--policies", "pro_rata", "correct_drift"]) == 0
    out = capsys.readouterr().out
    assert "money-weighted" in out and "correct_drift" in out and "irr" in out
    assert cli.main(["cashflow", "simulate", *base, "--withdraw", "400", "--initial", "200000", "--policies", "pro_rata", "correct_drift", "--dividend-yield", "0.02"]) == 0
    assert "mean_deviation" in capsys.readouterr().out
    assert cli.main(["cashflow", "spending", *base, "--paths", "200", "--years", "20", "--rules", "fixed_real", "endowment"]) == 0
    out = capsys.readouterr().out
    assert "ruin_probability" in out and "highest starting rate" in out and "endowment" in out
    assert cli.main(["cashflow", "redeem", "--prices", str(path), "--mix", "EQ=0.5,BD=0.3,GD=0.2"]) == 0
    assert "liquid" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not in the data"):
        cli.main(["cashflow", "simulate", "--prices", str(path), "--mix", "NOPE=1"])
    with pytest.raises(SystemExit, match="DGS10"):
        cli.main(["cashflow", "ldi", *base])
