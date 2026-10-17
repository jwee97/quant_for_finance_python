"""Exotic options: the closed forms against their own identities and limits, and the Monte Carlo engine against the closed forms (the barrier bridge, the control variate, the discrete-monitoring
bias that the bridge removes)."""

from __future__ import annotations

import numpy as np
import pytest

from src.derivatives import exotics, levy
from src.derivatives import pricing as pr

S, K, T, R, Q, SIG = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
BS = levy.BlackScholes(sigma=SIG)


# --------------------------------------------------------------------------------------------------------------------------------- digital
def test_digitals_obey_the_identities_that_define_them():
    cash_c, cash_p = (exotics.digital_price(S, K, T, R, Q, SIG, c, "cash", 7.0) for c in (True, False))
    asset_c, asset_p = (exotics.digital_price(S, K, T, R, Q, SIG, c, "asset") for c in (True, False))
    assert cash_c + cash_p == pytest.approx(7.0 * np.exp(-R * T)) and asset_c + asset_p == pytest.approx(S * np.exp(-Q * T))
    assert asset_c - K * exotics.digital_price(S, K, T, R, Q, SIG, True, "cash") == pytest.approx(float(pr.bsm_price(S, K, T, R, Q, SIG, True)), abs=1e-12), "a call is asset-or-nothing less K cash-or-nothing"
    h = 1e-3
    slope = (pr.bsm_price(S, K - h, T, R, Q, SIG, True) - pr.bsm_price(S, K + h, T, R, Q, SIG, True)) / (2 * h)
    assert float(exotics.digital_price(S, K, T, R, Q, SIG, True, "cash")) == pytest.approx(float(slope) , rel=1e-6), "the cash digital is minus the slope of the call in the strike, discounted"
    strikes = exotics.digital_price(S, np.array([90.0, 100.0, 110.0]), T, R, Q, SIG, True, "cash")
    assert strikes.shape == (3,) and np.all(np.diff(strikes) < 0)
    with pytest.raises(ValueError):
        exotics.digital_price(S, K, T, R, Q, SIG, kind="binary")


# --------------------------------------------------------------------------------------------------------------------------------- barrier
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("direction,H", [("down", 85.0), ("down", 95.0), ("up", 105.0), ("up", 125.0)])
@pytest.mark.parametrize("strike", [90.0, 100.0, 115.0])
def test_a_knock_in_and_the_matching_knock_out_add_up_to_the_vanilla(call, direction, H, strike):
    out = exotics.barrier_price(S, strike, H, T, R, Q, SIG, call, f"{direction}-out")
    knock_in = exotics.barrier_price(S, strike, H, T, R, Q, SIG, call, f"{direction}-in")
    assert out >= 0 and knock_in >= 0
    assert out + knock_in == pytest.approx(float(pr.bsm_price(S, strike, T, R, Q, SIG, call)), abs=1e-10)


def test_barrier_limits_a_far_barrier_changes_nothing_and_a_near_one_kills_the_option():
    vanilla = float(pr.bsm_price(S, K, T, R, Q, SIG, True))
    assert exotics.barrier_price(S, K, 1.0, T, R, Q, SIG, True, "down-out") == pytest.approx(vanilla, abs=1e-6)
    assert exotics.barrier_price(S, K, 1e4, T, R, Q, SIG, True, "up-out") == pytest.approx(vanilla, abs=1e-6) and exotics.barrier_price(S, K, 1e4, T, R, Q, SIG, True, "up-in") < 1e-6
    assert exotics.barrier_price(S, K, 100.0001, T, R, Q, SIG, True, "up-out") < 1e-3
    assert exotics.barrier_price(S, K, 99.9999, T, R, Q, SIG, True, "down-out") < 1e-3 and exotics.barrier_price(S, K, 99.9999, T, R, Q, SIG, True, "down-in") == pytest.approx(vanilla, abs=1e-3)
    outs = [exotics.barrier_price(S, K, h, T, R, Q, SIG, True, "down-out") for h in (60.0, 75.0, 90.0, 97.0)]
    assert all(a > b for a, b in zip(outs, outs[1:])), "the closer the barrier, the less a knock-out is worth"
    assert exotics.barrier_price(S, K, 105.0, T, R, Q, SIG, True, "down-out") == 0.0 and exotics.barrier_price(S, K, 105.0, T, R, Q, SIG, True, "down-in") == pytest.approx(vanilla)
    assert exotics.barrier_price(S, K, 95.0, T, R, Q, SIG, False, "up-out") == 0.0
    with pytest.raises(ValueError):
        exotics.barrier_price(S, K, 90.0, T, R, Q, SIG, True, "sideways")


@pytest.mark.parametrize("call,kind,H,strike", [(True, "down-out", 90.0, 100.0), (False, "up-out", 115.0, 100.0), (True, "up-in", 125.0, 95.0), (False, "down-in", 85.0, 105.0)])
def test_the_brownian_bridge_removes_the_bias_of_discrete_monitoring(call, kind, H, strike):
    exact = exotics.barrier_price(S, strike, H, T, R, Q, SIG, call, kind)
    bridged = exotics.barrier_mc(BS, S, strike, H, T, R, Q, call, kind, steps=52, n_paths=60_000, seed=1, bridge=True)
    naive = exotics.barrier_mc(BS, S, strike, H, T, R, Q, call, kind, steps=52, n_paths=60_000, seed=1, bridge=False)
    assert bridged["bridge"] and not naive["bridge"]
    assert abs(bridged["price"] - exact) < 4.5 * bridged["se"], (kind, bridged, exact)
    assert abs(naive["price"] - exact) > 5 * naive["se"], "weekly monitoring misses crossings: the knock-out is priced too high and the knock-in too low"
    assert (naive["price"] > exact) == kind.endswith("out")


def test_barrier_monte_carlo_under_a_jump_model_is_monitored_at_the_dates_only():
    merton = levy.Merton(sigma=0.15, lam=1.5, mu_j=-0.07, sigma_j=0.1)
    res = exotics.barrier_mc(merton, S, K, 90.0, T, R, Q, True, "down-out", steps=52, n_paths=40_000, seed=2)
    assert not res["bridge"] and 0 < res["price"] < exotics.mc_price(merton, S, R, Q, T, lambda p: np.maximum(p[:, -1] - K, 0.0), steps=1, n_paths=40_000, seed=2)["price"]
    assert exotics.barrier_mc(BS, S, K, 105.0, T, R, Q, True, "down-out", steps=12, n_paths=2_000)["price"] == 0.0, "a barrier the spot is already beyond"
    with pytest.raises(ValueError):
        exotics.barrier_mc(BS, S, K, 90.0, T, R, Q, True, "sideways")


# ---------------------------------------------------------------------------------------------------------------------- perpetual American
def test_the_perpetual_put_solves_its_equation_and_pastes_smoothly_at_the_boundary():
    r, q, sig = 0.06, 0.02, 0.3
    p = exotics.perpetual_american(100.0, 100.0, r, q, sig, False)
    star, beta = p["boundary"], p["beta"]
    assert beta < 0 and star == pytest.approx(100.0 * beta / (beta - 1.0)) and star < 100.0
    V = lambda s: exotics.perpetual_american(s, 100.0, r, q, sig, False)["price"]               # noqa: E731
    h = 1e-3 * star
    assert V(star) == pytest.approx(100.0 - star, abs=1e-9) and V(star * 0.9) == pytest.approx(100.0 - star * 0.9)
    assert (V(star + h) - V(star - h)) / (2 * h) == pytest.approx(-1.0, abs=1e-3), "smooth pasting: the value meets the payoff with the same slope"
    s0, h2 = 80.0, 0.05                                                                     # in the continuation region the Black-Scholes equation holds with no time derivative
    v, v1, v2 = V(s0), (V(s0 + h2) - V(s0 - h2)) / (2 * h2), (V(s0 + h2) - 2 * V(s0) + V(s0 - h2)) / h2 ** 2
    assert 0.5 * sig ** 2 * s0 ** 2 * v2 + (r - q) * s0 * v1 - r * v == pytest.approx(0.0, abs=1e-5)
    assert V(120.0) > pr.bsm_price(120.0, 100.0, 50.0, r, q, sig, False) - 1e-9, "never worth less than a long-dated European put"


def test_a_perpetual_call_is_never_exercised_without_dividends_and_a_zero_rate_is_refused():
    assert exotics.perpetual_american(100.0, 100.0, 0.06, 0.0, 0.3, True) == {"price": 100.0, "boundary": float("inf"), "beta": 1.0}
    c = exotics.perpetual_american(100.0, 100.0, 0.06, 0.09, 0.3, True)
    assert c["beta"] > 1 and c["boundary"] > 100.0 and c["price"] > 0 and exotics.perpetual_american(250.0, 100.0, 0.06, 0.09, 0.3, True)["price"] == 150.0
    with pytest.raises(ValueError):
        exotics.perpetual_american(100.0, 100.0, 0.0, 0.0, 0.3, False)


# ------------------------------------------------------------------------------------------------------------------------------------ Asian
def test_the_geometric_average_option_is_a_black_formula_that_the_simulation_reproduces():
    assert exotics.asian_geometric(S, K, T, R, Q, SIG, True, 1) == pytest.approx(float(pr.bsm_price(S, K, T, R, Q, SIG, True)), abs=1e-10), "one fixing at expiry is the vanilla"
    assert exotics.asian_geometric(S, K, T, R, Q, SIG, True, 100_000) == pytest.approx(exotics.asian_geometric(S, K, T, R, Q, SIG, True, None), abs=1e-4), "continuous averaging is the limit"
    n, idx = 52, np.arange(1, 53)
    for call, strike in ((True, 100.0), (False, 105.0)):
        sign = 1.0 if call else -1.0
        mc = exotics.mc_price(BS, S, R, Q, T, lambda p: np.maximum(sign * (np.exp(np.log(p[:, idx]).mean(axis=1)) - strike), 0.0), steps=n, n_paths=150_000, seed=4)
        assert abs(mc["price"] - exotics.asian_geometric(S, strike, T, R, Q, SIG, call, n)) < 4.5 * mc["se"]
    m, v = exotics._geometric_moments(S, T, R, Q, SIG, n)
    c, p = exotics.asian_geometric(S, K, T, R, Q, SIG, True, n), exotics.asian_geometric(S, K, T, R, Q, SIG, False, n)
    assert c - p == pytest.approx(np.exp(-R * T) * (np.exp(m + 0.5 * v) - K), abs=1e-12), "parity: call less put is the discounted mean of the average less the strike"


def test_the_arithmetic_average_costs_more_than_the_geometric_one_for_a_call_and_levy_is_close_and_exact_where_it_can_be():
    n = 52
    geo_c, geo_p = exotics.asian_geometric(S, K, T, R, Q, SIG, True, n), exotics.asian_geometric(S, K, T, R, Q, SIG, False, n)
    ar = exotics.asian_arithmetic_mc(BS, S, K, T, R, Q, True, n, 60_000, seed=3)
    ar_put = exotics.asian_arithmetic_mc(BS, S, K, T, R, Q, False, n, 60_000, seed=3)
    assert ar["price"] > geo_c and ar_put["price"] < geo_p, "the arithmetic mean is never below the geometric mean"
    assert ar["control"] == "geometric-average option" and ar["variance_reduction"] > 100
    lv = exotics.asian_levy(S, K, T, R, Q, SIG, True, n)
    assert abs(lv - ar["price"]) < 0.02 * ar["price"], "Levy's lognormal approximation is within about a percent here"
    assert exotics.asian_levy(S, K, T, R, Q, SIG, True, 1) == pytest.approx(float(pr.bsm_price(S, K, T, R, Q, SIG, True)), abs=1e-10), "one fixing at expiry: the average is the lognormal price"
    assert exotics.asian_levy(S, K, T, R, Q, 1e-6, True, 12) == pytest.approx(np.exp(-R * T) * max(np.mean(S * np.exp((R - Q) * T * np.arange(1, 13) / 12)) - K, 0.0), abs=1e-4), "no volatility: the forward average"
    assert exotics.asian_levy(S, K, T, R, Q, SIG, True, None) == pytest.approx(exotics.asian_levy(S, K, T, R, Q, SIG, True, 20_000), rel=2e-3), "continuous averaging is the limit"
    for b_drift in (0.0, SIG ** 2, -SIG ** 2, -SIG ** 2 / 2):                      # the degenerate drifts of the continuous second moment
        v = exotics.asian_levy(S, K, T, b_drift + 0.0, 0.0, SIG, True, None)
        assert np.isfinite(v) and v > 0


def test_asian_options_under_another_model_use_the_average_price_as_control_and_stay_unbiased():
    merton = levy.Merton(sigma=0.15, lam=1.5, mu_j=-0.07, sigma_j=0.1)
    cv = exotics.asian_arithmetic_mc(merton, S, K, T, R, Q, True, 26, 80_000, seed=5)
    idx = np.arange(1, 27)
    plain = exotics.mc_price(merton, S, R, Q, T, lambda p: np.maximum(p[:, idx].mean(axis=1) - K, 0.0), steps=26, n_paths=240_000, seed=6)
    assert cv["control"] == "average price" and cv["variance_reduction"] > 2
    assert abs(cv["price"] - plain["price"]) < 4.5 * np.hypot(cv["se"], plain["se"])


def test_a_control_variate_leaves_the_price_unbiased_and_cuts_the_variance():
    payoff = lambda p: np.maximum(p[:, -1] - K, 0.0)                         # noqa: E731
    exact = float(pr.bsm_price(S, K, T, R, Q, SIG, True))
    plain = exotics.mc_price(BS, S, R, Q, T, payoff, steps=1, n_paths=80_000, seed=7)
    ctrl = exotics.mc_price(BS, S, R, Q, T, payoff, steps=1, n_paths=80_000, seed=7, control=(lambda p: p[:, -1], S * np.exp(-Q * T)))
    assert abs(plain["price"] - exact) < 4.5 * plain["se"] and abs(ctrl["price"] - exact) < 4.5 * ctrl["se"]
    assert ctrl["se"] < plain["se"] and ctrl["variance_reduction"] > 1.5 and plain["variance_reduction"] == 1.0
