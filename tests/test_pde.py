"""The finite-difference solver: second order against the Black-Scholes closed form, Greeks from the same grid, the Rannacher start and cell averaging that tame a payoff with a kink or a jump,
American options against the binomial tree (with both complementarity solvers and the early-exercise boundary against its known limits), barriers against the Reiner-Rubinstein formulas
and digitals against theirs."""

from __future__ import annotations

import numpy as np
import pytest

from src.derivatives import exotics, greeks, pde
from src.derivatives import pricing as pr

S, K, T, R, Q, SIG = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25


# ------------------------------------------------------------------------------------------------------------------------------ European
@pytest.mark.parametrize("call", [True, False])
def test_the_grid_reproduces_the_closed_form_price_and_greeks(call):
    res = pde.vanilla(S, K, T, R, Q, SIG, call, n_space=800, n_time=400)
    g = greeks.bsm_greeks(S, K, T, R, Q, SIG, call)
    assert res.price == pytest.approx(float(pr.bsm_price(S, K, T, R, Q, SIG, call)), abs=1e-4)
    assert res.delta == pytest.approx(float(g["delta"]), abs=1e-5) and res.gamma == pytest.approx(float(g["gamma"]), abs=1e-5) and res.theta == pytest.approx(float(g["theta"]), abs=2e-3)
    ref = pr.bsm_price(res.S, K, T, R, Q, SIG, call)
    assert np.abs(res.values - ref).max() < 3e-4, "the whole grid is right, boundary included (a wrongly applied boundary value spoils the nodes next to it)"


def test_the_error_falls_by_four_when_the_grid_is_refined_by_two():
    study = pde.convergence(lambda ns, nt: pde.vanilla(S, K, T, R, Q, SIG, True, n_space=ns, n_time=nt).price, float(pr.bsm_price(S, K, T, R, Q, SIG, True)))
    assert 1.8 < study["order"] < 2.2, study
    assert study["errors"][0] > 3.5 * study["errors"][1] > 12 * study["errors"][2]


def test_implicit_euler_is_first_order_in_time_and_the_rannacher_start_keeps_crank_nicolson_smooth_at_a_kink():
    ref = float(pr.bsm_price(S, K, 0.25, R, Q, SIG, True))
    eul = [abs(pde.vanilla(S, K, 0.25, R, Q, SIG, True, n_space=3200, n_time=nt, theta=1.0).price - ref) for nt in (10, 20, 40)]
    assert 1.7 < eul[0] / eul[1] < 2.3 and 1.7 < eul[1] / eul[2] < 2.3, eul
    # Crank-Nicolson started without implicit steps rings on the kink: the gamma it leaves near the strike oscillates, which the implicit half-steps at the start remove
    def gamma_error(res):
        x = np.log(res.S)
        vx = np.gradient(res.values, x)
        gamma = (np.gradient(vx, x) - vx) / res.S ** 2
        near = (res.S > 90) & (res.S < 110)
        return float(np.abs(gamma - greeks.bsm_greeks(res.S, K, 0.05, R, Q, SIG, True)["gamma"])[near].max())

    smooth = pde.vanilla(S, K, 0.05, R, Q, SIG, True, n_space=1600, n_time=10, rannacher=True)
    rough = pde.vanilla(S, K, 0.05, R, Q, SIG, True, n_space=1600, n_time=10, rannacher=False)
    assert gamma_error(smooth) < 0.2 * gamma_error(rough), (gamma_error(smooth), gamma_error(rough))


def test_cell_averaging_is_exact_for_a_kink_and_for_a_jump():
    x = np.linspace(np.log(60.0), np.log(160.0), 41)
    edges = np.concatenate([[x[0] - 0.5 * (x[1] - x[0])], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])
    k = float(np.log(K))
    call = pde._cell_average(lambda s: np.maximum(s - K, 0.0), edges, (k,))
    a, b = edges[:-1], edges[1:]
    exact = np.where(b <= k, 0.0, np.where(a >= k, np.exp(b) - np.exp(a) - K * (b - a), np.exp(b) - K - K * (b - k))) / (b - a)
    assert np.abs(call - exact).max() < 1e-10
    digital = pde._cell_average(lambda s: (s > K).astype(float), edges, (k,))
    frac = np.clip((b - k) / (b - a), 0.0, 1.0)
    assert np.abs(digital - frac).max() < 1e-12


def test_digitals_match_their_closed_forms_to_the_accuracy_of_the_kink_free_scheme():
    for call in (True, False):
        for kind in ("cash", "asset"):
            res = pde.digital(S, K, T, R, Q, SIG, call, kind, payout=1.0 if kind == "cash" else 1.0, n_space=800, n_time=400)
            ref = float(exotics.digital_price(S, K, T, R, Q, SIG, call, kind))
            assert res.price == pytest.approx(ref, rel=1e-4 if kind == "cash" else 2e-4), (call, kind)
    cash = pde.digital(S, K, T, R, Q, SIG, True, "cash", n_space=800, n_time=400)
    d2 = (np.log(S / K) + (R - Q - 0.5 * SIG ** 2) * T) / (SIG * np.sqrt(T))
    assert cash.delta == pytest.approx(float(np.exp(-R * T) * np.exp(-0.5 * d2 ** 2) / np.sqrt(2 * np.pi) / (S * SIG * np.sqrt(T))), rel=1e-4)


def test_the_cell_peclet_guard_refines_the_grid_when_the_drift_dominates_the_diffusion():
    res = pde.vanilla(S, K, 1.0, 0.5, 0.0, 0.05, True, n_space=100, n_time=50)               # r = 50%, sigma = 5%: the convection is 100 times the diffusion
    assert res.n_space > 100 and res.price == pytest.approx(float(pr.bsm_price(S, K, 1.0, 0.5, 0.0, 0.05, True)), abs=5e-3)


def test_expired_and_invalid_options():
    res = pde.vanilla(105.0, 100.0, 0.0, R, Q, SIG, True)
    assert res.price == 5.0 and pde.vanilla(95.0, 100.0, 0.0, R, Q, SIG, True).price == 0.0
    with pytest.raises(ValueError):
        pde.vanilla(S, K, -1.0, R, Q, SIG)
    with pytest.raises(ValueError):
        pde.vanilla(S, K, 1.0, R, Q, 0.0)
    with pytest.raises(ValueError):
        pde.digital(S, K, 1.0, R, Q, SIG, kind="binary")


# ----------------------------------------------------------------------------------------------------------------------------------- American
def test_the_american_put_matches_the_binomial_tree_and_both_complementarity_solvers_agree():
    tree = pr.binomial_price(S, K, T, R, Q, SIG, False, True, 4000)
    bs = pde.vanilla(S, K, T, R, Q, SIG, False, True, n_space=400, n_time=200, lcp="bs")
    psor = pde.vanilla(S, K, T, R, Q, SIG, False, True, n_space=400, n_time=200, lcp="psor")
    assert abs(bs.price - tree) < 1e-3 and abs(bs.price - psor.price) < 1e-8
    assert np.abs(bs.values - psor.values).max() < 1e-7
    fine = pde.vanilla(S, K, T, R, Q, SIG, False, True, n_space=1600, n_time=800)
    assert abs(fine.price - tree) < 5e-4 and abs(fine.price - bs.price) < 1e-3
    euro = float(pr.bsm_price(S, K, T, R, Q, SIG, False))
    assert fine.price > euro + 0.2, "the right to exercise early is worth something here (the rate exceeds the dividend yield)"
    assert np.all(bs.values >= np.maximum(K - bs.S, 0.0) - 1e-12), "never below the payoff"
    assert -1.0 < bs.delta < 0.0 and bs.gamma > 0


def test_an_american_call_is_a_european_call_without_dividends_and_worth_more_with_them():
    plain = pde.vanilla(S, K, T, R, 0.0, SIG, True, True, n_space=400, n_time=200)
    assert plain.price == pytest.approx(float(pr.bsm_price(S, K, T, R, 0.0, SIG, True)), abs=5e-4)
    rich = pde.vanilla(S, K, T, R, 0.08, SIG, True, True, n_space=800, n_time=400, boundary=True)
    tree = pr.binomial_price(S, K, T, R, 0.08, SIG, True, True, 4000)
    assert abs(rich.price - tree) < 1e-3 and rich.price > float(pr.bsm_price(S, K, T, R, 0.08, SIG, True)) + 0.1
    assert np.nanmin(rich.boundary) > K, "a call is exercised above the strike, at high prices"


def test_the_early_exercise_boundary_of_a_put_starts_at_the_strike_and_falls_towards_the_perpetual_one():
    res = pde.vanilla(S, K, T, R, Q, SIG, False, True, n_space=800, n_time=400, boundary=True)
    b = res.boundary[1:]
    assert np.all(np.isfinite(b)) and np.all(np.diff(b) < 1e-9), "the longer to expiry, the lower the price at which exercise is optimal"
    assert 0.85 * K < b[0] < K and b[-1] < 0.8 * K
    assert pde._free_boundary(np.log(np.linspace(50, 150, 101)), np.where(np.linspace(50, 150, 101) > 80.0, 0.5 * (np.linspace(50, 150, 101) - 80.0) ** 2 / 100, 0.0)
                              + np.maximum(100 - np.linspace(50, 150, 101), 0), np.maximum(100 - np.linspace(50, 150, 101), 0), "low") == pytest.approx(80.0, abs=1e-6), "smooth pasting: the boundary is recovered exactly from a quadratic"
    perp = exotics.perpetual_american(S, K, 0.06, 0.02, 0.3, False)
    far = [pde.vanilla(S, K, tt, 0.06, 0.02, 0.3, False, True, n_space=800, n_time=int(60 * tt), boundary=True, width=5.0) for tt in (10.0, 30.0)]
    assert far[0].boundary[-1] > far[1].boundary[-1] > perp["boundary"] and far[1].boundary[-1] < 1.012 * perp["boundary"]
    assert far[0].price < far[1].price < perp["price"] and perp["price"] - far[1].price < 0.02 * perp["price"], "the finite-maturity value rises towards the perpetual one"


# ------------------------------------------------------------------------------------------------------------------------------------ barrier
@pytest.mark.parametrize("call,kind,H", [(True, "down-out", 85.0), (True, "down-out", 95.0), (True, "up-out", 115.0), (True, "up-out", 130.0),
                                         (False, "down-out", 85.0), (False, "down-out", 95.0), (False, "up-out", 115.0), (False, "up-out", 130.0)])
@pytest.mark.parametrize("strike", [90.0, 110.0])
def test_knock_out_prices_match_the_reiner_rubinstein_formulas(call, kind, H, strike):
    res = pde.barrier(S, strike, H, T, R, Q, SIG, call, kind, n_space=800, n_time=400)
    assert res.price == pytest.approx(exotics.barrier_price(S, strike, H, T, R, Q, SIG, call, kind), abs=1e-4)


def test_a_barrier_option_that_has_already_knocked_out_is_worth_nothing_and_the_barrier_must_be_a_knock_out():
    assert pde.barrier(S, K, 95.0, T, R, Q, SIG, True, "up-out").price == 0.0 and pde.barrier(S, K, 105.0, T, R, Q, SIG, True, "down-out").price == 0.0     # spot already beyond the barrier
    with pytest.raises(ValueError):
        pde.barrier(S, K, 90.0, T, R, Q, SIG, True, "down-in")
    far = pde.barrier(S, K, 20.0, T, R, Q, SIG, True, "down-out", n_space=800, n_time=400)
    assert far.price == pytest.approx(float(pr.bsm_price(S, K, T, R, Q, SIG, True)), abs=5e-4), "a barrier far below changes nothing"
    near = pde.barrier(S, K, 99.5, T, R, Q, SIG, True, "down-out", n_space=1600, n_time=400)
    assert 0 <= near.price < 1.0, "a barrier at the money kills almost everything"
