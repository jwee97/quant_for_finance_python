"""Hedging with transaction costs: Leland's adjusted volatility (closed form and the nonlinear grid for a payoff whose gamma changes sign), the Whalley-Wilmott band, a hedging simulation that
says honestly what each rule costs and how much risk it leaves, and the utility-indifference prices of a binomial market, which reduce to Cox-Ross-Rubinstein without cost and whose no-trade
band is the Whalley-Wilmott band for small cost."""

from __future__ import annotations

import numpy as np
import pytest

from src.derivatives import costs, greeks, levy
from src.derivatives import pricing as pr

S, K, T, R, Q, SIG = 100.0, 100.0, 0.5, 0.03, 0.0, 0.2


# ------------------------------------------------------------------------------------------------------------------------------ Leland
def test_the_leland_number_and_adjusted_volatilities():
    le = costs.leland_number(SIG, 1 / 52, 0.0025)
    assert le == pytest.approx(2 * np.sqrt(2 / np.pi) * 0.0025 / (SIG * np.sqrt(1 / 52)))
    assert costs.leland_sigma(SIG, 1 / 52, 0.0025, "sell") == pytest.approx(SIG * np.sqrt(1 + le)) and costs.leland_sigma(SIG, 1 / 52, 0.0025, "buy") == pytest.approx(SIG * np.sqrt(1 - le))
    assert costs.leland_sigma(SIG, 1 / 52, 0.0, "sell") == SIG
    assert costs.leland_number(SIG, 1 / 252, 0.0025) > costs.leland_number(SIG, 1 / 52, 0.0025) > costs.leland_number(SIG, 1 / 12, 0.0025) > 0, "rebalancing more often costs more"
    with pytest.raises(ValueError, match="at least 1"):
        costs.leland_sigma(SIG, 1 / 2520, 0.01, "buy")
    with pytest.raises(ValueError):
        costs.leland_number(0.0, 1 / 52, 0.01)
    with pytest.raises(ValueError):
        costs.leland_sigma(SIG, 1 / 52, 0.01, "hold")


def test_the_leland_price_brackets_black_scholes_and_the_grid_reproduces_it_for_a_vanilla():
    bs = float(pr.bsm_price(S, K, T, R, Q, SIG, True))
    sell, buy = (costs.leland_price(S, K, T, R, Q, SIG, 0.0025, 1 / 52, True, side) for side in ("sell", "buy"))
    assert buy < bs < sell and costs.leland_price(S, K, T, R, Q, SIG, 0.0, 1 / 52, True, "sell") == pytest.approx(bs)
    for side, exact in (("sell", sell), ("buy", buy)):
        grid = costs.leland_pde(S, [K], [1.0], T, R, Q, SIG, 0.0025, 1 / 52, side, n_space=800, n_time=400)
        assert grid["price"] == pytest.approx(exact, rel=1e-3), side
    free = costs.leland_pde(S, [K], [1.0], T, R, Q, SIG, 0.0, 1 / 52, "sell", n_space=800, n_time=400)
    assert free["price"] == pytest.approx(bs, rel=2e-4) and free["delta"] == pytest.approx(0.5702, abs=2e-3) and free["gamma"] > 0


def test_a_payoff_whose_gamma_changes_sign_is_priced_by_the_nonlinear_grid():
    """A butterfly is long gamma in the middle and short at the wings: the seller needs a higher volatility where the gamma is positive and a lower one where it is negative, so the price is
    neither of the two constant-volatility prices. The seller's price exceeds Black-Scholes and the buyer's falls short of it."""
    args = dict(S=S, strikes=[90.0, 100.0, 110.0], weights=[1.0, -2.0, 1.0], T=T, r=R, q=Q, sigma=SIG, cost=0.005, dt=1 / 52)
    bs = float(pr.bsm_price(S, 90.0, T, R, Q, SIG, True) - 2 * pr.bsm_price(S, 100.0, T, R, Q, SIG, True) + pr.bsm_price(S, 110.0, T, R, Q, SIG, True))
    free = costs.leland_pde(**{**args, "cost": 0.0})["price"]
    assert free == pytest.approx(bs, abs=1e-3)
    sell, buy = costs.leland_pde(**args, side="sell")["price"], costs.leland_pde(**args, side="buy")["price"]
    sell_flat, buy_flat = costs.leland_pde(**args, side="sell", nonlinear=False)["price"], costs.leland_pde(**args, side="buy", nonlinear=False)["price"]
    assert buy < bs < sell and abs(sell - sell_flat) > 0.3 and abs(buy - buy_flat) > 0.3, (buy, bs, sell, sell_flat, buy_flat)
    with pytest.raises(ValueError):
        costs.leland_pde(S, [K], [1.0, 2.0], T, R, Q, SIG, 0.001, 1 / 52)
    with pytest.raises(ValueError):
        costs.leland_pde(S, [K], [1.0], T, R, Q, SIG, 0.01, 1 / 2520, side="buy")


# ------------------------------------------------------------------------------------------------------------------------ Whalley-Wilmott
def test_the_whalley_wilmott_band_follows_its_formula_and_scales_as_cost_to_the_one_third():
    delta, half = costs.whalley_wilmott_band(S, K, T, R, Q, SIG, 0.005, 1.0)
    g = float(greeks.bsm_greeks(S, K, T, R, Q, SIG, True)["gamma"])
    assert float(half) == pytest.approx((1.5 * np.exp(-R * T) * 0.005 * S * g ** 2) ** (1 / 3)) and float(delta) == pytest.approx(0.5702, abs=1e-3)
    h8 = float(costs.whalley_wilmott_band(S, K, T, R, Q, SIG, 0.04, 1.0)[1])
    assert h8 / float(half) == pytest.approx(2.0, rel=1e-9), "eight times the cost doubles the band"
    assert float(costs.whalley_wilmott_band(S, K, T, R, Q, SIG, 0.005, 8.0)[1]) == pytest.approx(float(half) / 2.0, rel=1e-9), "eight times the risk aversion halves it"
    assert float(costs.whalley_wilmott_band(S, K, T, R, Q, SIG, 0.0, 1.0)[1]) == 0.0
    d_arr, h_arr = costs.whalley_wilmott_band(np.array([80.0, 100.0, 120.0]), K, T, R, Q, SIG, 0.005, 1.0)
    assert h_arr.shape == (3,) and h_arr[1] > h_arr[0] and h_arr[1] > h_arr[2], "the band is widest where the gamma is largest, at the money"
    with pytest.raises(ValueError):
        costs.whalley_wilmott_band(S, K, T, R, Q, SIG, 0.005, 0.0)


# ------------------------------------------------------------------------------------------------------------------------------ hedging
def test_without_cost_discrete_delta_hedging_is_fair_and_its_error_shrinks_with_the_frequency():
    daily = costs.simulate_hedging(S, K, T, R, Q, SIG, 0.0, 126, 20_000, "delta", seed=3)
    weekly = costs.simulate_hedging(S, K, T, R, Q, SIG, 0.0, 26, 20_000, "delta", seed=3)
    assert abs(daily["mean"]) < 4.5 * daily["se"] and abs(weekly["mean"]) < 4.5 * weekly["se"], "the hedge is unbiased at the true volatility"
    assert 1.9 < weekly["std"] / daily["std"] < 2.6 and daily["cost"] == 0.0, "the error falls like one over the square root of the number of hedges (here sqrt(126/26) = 2.2)"
    naked = costs.simulate_hedging(S, K, T, R, Q, SIG, 0.0, 26, 20_000, "none", seed=3)
    assert naked["std"] > 8 * weekly["std"] and abs(naked["mean"]) < 4.5 * naked["se"] and naked["trades"] == 0.0


def test_costs_eat_the_profit_in_proportion_to_the_trading_and_the_rules_trade_off_cost_against_risk():
    rows = {st: costs.simulate_hedging(S, K, T, R, Q, SIG, 0.005, 52, 20_000, st, risk_aversion=1.0, seed=3) for st in ("delta", "leland", "band")}
    d, le, b = rows["delta"], rows["leland"], rows["band"]
    assert d["mean"] == pytest.approx(-d["cost"], abs=0.08), "the mean loss of a fair hedge is the cost paid"
    assert le["premium"] > d["premium"] and le["mean"] > d["mean"] + 0.5, "Leland's premium pays for most of the hedge"
    assert le["mean"] < -0.1, "but not all of it: the adjustment is an approximation that falls short"
    assert b["trades"] < 0.6 * d["trades"] and b["cost"] < 0.7 * d["cost"], "the band trades much less"
    assert b["std"] > d["std"], "and leaves more risk: the price of the saving"
    few = costs.simulate_hedging(S, K, T, R, Q, SIG, 0.005, 13, 20_000, "delta", seed=3)
    many = costs.simulate_hedging(S, K, T, R, Q, SIG, 0.005, 130, 20_000, "delta", seed=3)
    assert many["cost"] > 1.8 * few["cost"] and many["std"] < few["std"], "more trades: more cost, less risk"
    assert abs(d["trades"] - 53.0) < 1.5, "one trade per date, plus the unwinding"


def test_the_hedging_simulation_accepts_a_jump_model_a_drift_and_a_put_and_checks_its_arguments():
    merton = levy.Merton(sigma=0.12, lam=3.0, mu_j=-0.06, sigma_j=0.08)
    jumpy = costs.simulate_hedging(S, K, T, R, Q, 0.2, 0.0, 52, 8_000, "delta", model=merton, seed=4)
    smooth = costs.simulate_hedging(S, K, T, R, Q, 0.2, 0.0, 52, 8_000, "delta", seed=4)
    assert jumpy["std"] > smooth["std"], "a hedge at the Black-Scholes volatility cannot follow jumps"
    trending = costs.simulate_hedging(S, K, T, R, Q, SIG, 0.0, 52, 8_000, "delta", mu=0.15, seed=4)
    assert trending["std"] == pytest.approx(smooth["std"], rel=0.15), "the drift barely matters to the error of a delta hedge"
    put = costs.simulate_hedging(S, K, T, R, Q, SIG, 0.0, 52, 8_000, "delta", call=False, seed=4)
    assert abs(put["mean"]) < 4.5 * put["se"] + 1e-9 and put["premium"] == pytest.approx(float(pr.bsm_price(S, K, T, R, Q, SIG, False)))
    rows = costs.compare_strategies(S, K, T, R, Q, SIG, 0.005, 26, 3_000, seed=1)
    assert [r_["strategy"] for r_ in rows] == ["delta", "leland", "band"] and all("pnl" not in r_ for r_ in rows)
    with pytest.raises(ValueError):
        costs.simulate_hedging(S, K, T, R, Q, SIG, 0.005, 26, 100, "momentum")


# ----------------------------------------------------------------------------------------------------------------------------- DPZ tree
@pytest.mark.parametrize("call", [True, False])
def test_without_cost_the_indifference_prices_are_the_cox_ross_rubinstein_price_whatever_the_risk_aversion_and_drift(call):
    for gamma, mu in ((0.002, None), (0.05, None), (0.01, 0.06), (0.01, 0.04)):                # a drift within what the holding range can carry
        res = costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.0, gamma, n_steps=20, call=call, n_hold=121, mu=mu)
        assert res["writer"] == pytest.approx(res["crr"], abs=5e-4) and res["buyer"] == pytest.approx(res["crr"], abs=5e-4), (gamma, mu)
    tree = pr.binomial_price(S, K, 0.25, 0.05, 0.0, SIG, call, False, 20)
    assert res["crr"] == pytest.approx(tree, rel=0.02), "the tree here is the standard one"


def test_with_cost_there_is_a_bid_and_an_ask_around_the_frictionless_price_that_widen_with_the_cost_and_the_risk_aversion():
    prev = 0.0
    for cost in (0.0002, 0.0005, 0.001, 0.002):
        res = costs.dpz_prices(S, K, 0.25, 0.05, SIG, cost, 0.01, n_steps=30, n_hold=161)
        assert res["buyer"] < res["crr"] < res["writer"] and res["spread"] == pytest.approx(res["writer"] - res["buyer"])
        assert res["spread"] > prev
        prev = res["spread"]
    low = costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.0005, 0.005, n_steps=30, n_hold=161)
    high = costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.0005, 0.04, n_steps=30, n_hold=161)
    assert high["spread"] > low["spread"], "a more risk-averse investor demands more to be indifferent"


def test_the_no_trade_band_of_the_tree_is_the_whalley_wilmott_band_for_small_cost_and_scales_as_risk_aversion_to_minus_one_third():
    """The asymptotic band assumes trading at every instant; the tree trades 40 times, so where the band is narrower than one period's move in the delta (about 0.04) the tree's band is a
    good deal narrower than the formula's. Where it is wider the two agree to within about a tenth."""
    widths, wws = [], []
    for gamma in (0.01, 0.02):
        res = costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.0005, gamma, n_steps=40, n_hold=201, bands=True)
        lo, hi = res["band"]
        delta, half = costs.whalley_wilmott_band(S, K, 0.25, 0.05, 0.0, SIG, 0.0005, gamma)
        widths.append((hi - lo) / 2.0)
        wws.append(float(half))
        assert lo < res["crr_delta"] < hi and (lo + hi) / 2 == pytest.approx(res["crr_delta"], abs=0.015), "the band is centred on the frictionless delta"
        assert 0.8 * float(half) < (hi - lo) / 2.0 < 1.02 * float(half), (gamma, widths[-1], float(half))
    assert widths[0] / widths[1] == pytest.approx(2 ** (1 / 3), rel=0.15) and wws[0] / wws[1] == pytest.approx(2 ** (1 / 3), rel=1e-9)
    narrow = costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.0002, 0.16, n_steps=40, n_hold=201, bands=True)["band"]
    assert (narrow[1] - narrow[0]) / 2.0 < 0.75 * float(costs.whalley_wilmott_band(S, K, 0.25, 0.05, 0.0, SIG, 0.0002, 0.16)[1]), "a coarse clock cannot hold a narrow band"
    put = costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.0005, 0.01, n_steps=30, call=False, n_hold=161, bands=True)
    assert put["band"][0] < put["crr_delta"] < put["band"][1] and put["crr_delta"] < 0


def test_the_indifference_price_arguments_are_checked():
    with pytest.raises(ValueError):
        costs.dpz_prices(S, K, 0.25, 0.05, SIG, -0.1, 0.01)
    with pytest.raises(ValueError):
        costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.001, 0.0)
    with pytest.raises(ValueError):
        costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.001, 0.01, n_steps=0)
    one = costs.dpz_prices(S, K, 0.25, 0.05, SIG, 0.0, 0.01, n_steps=1, n_hold=61)
    assert one["writer"] == pytest.approx(one["crr"], abs=2e-3)
