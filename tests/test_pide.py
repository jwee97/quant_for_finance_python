"""The integro-differential solver for jump models: against the exact Fourier price for every model, second-order convergence where the theory promises it, the martingale property that the
drift is solved for, the properties of the discretised jump measure, and the stiffness control that keeps the explicit jump term accurate, and early exercise (an American option under each model against the Black-Scholes solver, a Longstaff-Schwartz simulation,
refinement and the properties early exercise must have)."""

from __future__ import annotations

import numpy as np
import pytest

from src.derivatives import fourier, levy, pde, pide

S, K, T, R, Q = 100.0, 100.0, 0.75, 0.03, 0.01
MODELS = {
    "bs": levy.BlackScholes(sigma=0.22),
    "merton": levy.Merton(sigma=0.15, lam=1.5, mu_j=-0.07, sigma_j=0.1),
    "kou": levy.Kou(sigma=0.15, lam=2.0, p=0.35, eta1=12.0, eta2=9.0),
    "vg": levy.VarianceGamma(sigma=0.18, theta=-0.12, nu=0.25),
    "nig": levy.NIG(sigma=0.18, theta=-0.12, kappa=0.25),
}
TOLERANCE = {"bs": 1.5e-4, "merton": 4e-4, "kou": 4e-4, "vg": 1.5e-3, "nig": 1.5e-3}


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("call", [True, False])
def test_the_pide_price_matches_the_exact_fourier_price(name, call):
    m = MODELS[name]
    for strike in (90.0, 100.0, 110.0):
        res = pide.european(m, S, strike, T, R, Q, call, n_space=1000, n_time=200)
        assert res.price == pytest.approx(float(fourier.price(m, S, strike, T, R, Q, call)), abs=TOLERANCE[name]), (name, strike)
        assert res.extras["neglected"] < 1e-9


@pytest.mark.parametrize("name", ["bs", "merton", "kou", "vg", "nig"])
def test_refining_the_grid_cuts_the_error(name):
    m = MODELS[name]
    ref = float(fourier.price(m, S, K, T, R, Q, True))
    errs = [abs(pide.european(m, S, K, T, R, Q, True, n_space=ns, n_time=nt).price - ref) for ns, nt in ((250, 50), (500, 100), (1000, 200))]
    if name in ("bs", "merton", "kou"):
        assert errs[0] > 3.0 * errs[1] and errs[1] > 3.0 * errs[2], ("second order for the models with finite activity", errs)
    else:
        assert errs[2] < 0.7 * errs[0] and errs[2] < 1.5e-3, ("the pure-jump models converge more slowly: the small jumps are only approximated", errs)


@pytest.mark.parametrize("name", list(MODELS))
def test_the_discrete_operator_keeps_the_discounted_price_a_martingale_so_call_minus_put_is_the_forward_whatever_the_grid(name):
    """The drift is the number that makes ``e^x`` a martingale for the discrete operator itself, so on a grid of only 100 cells (where the quadrature of the jump measure is poor) the
    solution for the linear payoff ``S - K`` is still the forward, up to the time stepping, which a finer step removes at second order. (The cell-averaged call minus the cell-averaged put is
    ``e^x sinh(dx/2)/(dx/2) - K``, not ``e^x - K``: that factor is the exact answer here.)"""
    m = MODELS[name]
    gaps = []
    for nt in (200, 800):
        c = pide.european(m, S, K, T, R, Q, True, n_space=100, n_time=nt)
        p = pide.european(m, S, K, T, R, Q, False, n_space=100, n_time=nt)
        dx = float(np.log(c.S[1] / c.S[0]))
        exact = S * np.exp(-Q * T) * np.sinh(dx / 2) / (dx / 2) - K * np.exp(-R * T)
        gaps.append(abs(c.price - p.price - exact))
    assert gaps[1] < 1e-4 and gaps[1] < gaps[0] / 3, (name, gaps)
    assert gaps[0] < 5e-4, "even at 200 steps the departure is a few parts in a million of the price"


def test_the_greeks_from_the_grid_agree_with_bumped_fourier_prices():
    m = MODELS["merton"]
    res = pide.european(m, S, K, T, R, Q, True, n_space=1500, n_time=300)
    h = 0.5
    up, mid, dn = (float(fourier.price(m, s, K, T, R, Q, True)) for s in (S + h, S, S - h))
    assert res.delta == pytest.approx((up - dn) / (2 * h), abs=2e-4) and res.gamma == pytest.approx((up - 2 * mid + dn) / h ** 2, abs=2e-4)
    dt = 1e-3
    theta = -(float(fourier.price(m, S, K, T + dt, R, Q, True)) - float(fourier.price(m, S, K, T - dt, R, Q, True))) / (2 * dt)
    assert res.theta == pytest.approx(theta, abs=2e-2)


def test_the_imex_scheme_is_second_order_in_time_and_euler_is_first_order():
    m = MODELS["merton"]
    ref = float(fourier.price(m, S, K, T, R, Q, True))
    cn = [abs(pide.european(m, S, K, T, R, Q, True, n_space=2000, n_time=nt, imex="cnab").price - ref) for nt in (10, 20, 40)]
    eu = [abs(pide.european(m, S, K, T, R, Q, True, n_space=2000, n_time=nt, imex="euler").price - ref) for nt in (10, 20, 40)]
    assert cn[1] < 0.35 * cn[0] and eu[1] > 0.4 * eu[0] and eu[0] > 5 * cn[0] and eu[2] > 3 * cn[2], (cn, eu)


# ------------------------------------------------------------------------------------------------------------------------ the jump measure
def test_the_discretised_jump_measure_has_the_mass_and_the_variance_of_the_levy_measure():
    m = MODELS["merton"]
    dx = 0.002
    k = pide.jump_kernel(m, dx, small_cells=0, match_variance=True)
    central = float(pide._cell_measure(m, np.array([-dx / 2]), np.array([dx / 2]))[0])               # the Merton density is smooth through zero, so the central cell can be integrated too
    assert k["nu"].sum() + central == pytest.approx(m.params["lam"], rel=1e-9), "the cells together carry exactly the intensity of a finite-activity measure"
    assert set(k["offsets"]) == set(range(-k["j_max"], k["j_max"] + 1)) - {0}
    z = k["offsets"] * dx
    p = m.params
    true_second = p["lam"] * (p["mu_j"] ** 2 + p["sigma_j"] ** 2)
    assert (k["nu"] * z ** 2).sum() + k["sigma_eps2"] + k["variance_defect"] == pytest.approx(true_second, rel=1e-9), "with the correction the discrete jumps and the central band have exactly the variance"
    assert k["variance_defect"] < 0 and abs(k["variance_defect"]) < 1e-3 * true_second
    plain = pide.jump_kernel(m, dx, small_cells=0, match_variance=False)
    assert plain["variance_defect"] == 0.0 and (plain["nu"] * z ** 2).sum() > (k["nu"] * z ** 2).sum() - 1e-12, "the midpoint rule overstates the variance"
    assert np.all(k["nu"] >= 0) and k["neglected"] < 1e-9 and k["j_max"] == int(np.abs(k["offsets"]).max())


def test_a_wider_band_of_small_jumps_leaves_fewer_explicit_cells_and_the_infinite_activity_models_need_one():
    vg, nig, merton = MODELS["vg"], MODELS["nig"], MODELS["merton"]
    dx, dt = 0.001, 0.75 / 400
    assert pide.choose_small_cells(merton, dx, dt, stiffness=0.03) == 0
    m_vg, m_nig = pide.choose_small_cells(vg, dx, dt, stiffness=0.03), pide.choose_small_cells(nig, dx, dt, stiffness=0.03)
    assert m_nig > m_vg >= 0 and m_nig >= 5, "the intensity of small jumps of the NIG law grows like 1/(dx) and needs a wide band"
    assert pide.choose_small_cells(nig, dx, dt / 10, stiffness=0.03) < m_nig, "a shorter time step tolerates a narrower band"
    narrow, wide = pide.jump_kernel(nig, dx, small_cells=1), pide.jump_kernel(nig, dx, small_cells=9)
    assert wide["nu"].sum() < narrow["nu"].sum() and wide["sigma_eps2"] > narrow["sigma_eps2"] > 0
    assert 0 not in wide["offsets"] and not np.any(np.abs(wide["offsets"]) <= 9)


def test_the_stiffness_control_is_what_makes_the_nig_accurate():
    nig = MODELS["nig"]
    ref = float(fourier.price(nig, S, K, T, R, Q, True))
    stiff = abs(pide.european(nig, S, K, T, R, Q, True, n_space=1000, n_time=100, small_cells=0).price - ref)
    tuned = abs(pide.european(nig, S, K, T, R, Q, True, n_space=1000, n_time=100).price - ref)
    assert tuned < 0.2 * stiff and tuned < 3e-3, (stiff, tuned)


# --------------------------------------------------------------------------------------------------------------------------- edges
def test_expired_options_and_unsuitable_models_and_schemes():
    assert pide.european(MODELS["vg"], 105.0, 100.0, 0.0, R, Q, True).price == 5.0
    with pytest.raises(ValueError, match="Levy"):
        pide.european(levy.Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.5), S, K, T, R, Q)
    with pytest.raises(ValueError):
        pide.european(MODELS["merton"], S, K, T, R, Q, imex="rk4")
    res = pide.european(MODELS["kou"], S, K, T, R, Q, n_space=400, n_time=100)
    assert set(res.summary()) >= {"price", "delta", "gamma", "theta", "neglected", "kernel_cells", "small_cells", "drift"} and res.n_space >= 400


# ------------------------------------------------------------------------------------------------------------------------ early exercise
def _lsmc_put(model, strike, steps=50, n=60_000, seed=1, degree=4):
    """A Longstaff-Schwartz estimate of a Bermudan put under ``model``: the independent check of the American PIDE (it is a lower bound that needs no grid)."""
    paths = levy.simulate_paths(model, S, R, Q, T, steps, n, seed, True)
    disc = np.exp(-R * T / steps)
    cash = np.maximum(strike - paths[:, -1], 0.0)
    for i in range(steps - 1, 0, -1):
        cash = cash * disc
        s = paths[:, i]
        ex = np.maximum(strike - s, 0.0)
        itm = ex > 0
        if itm.sum() > 50:
            basis = np.vander(s[itm] / strike, degree + 1)
            cont = basis @ np.linalg.lstsq(basis, cash[itm], rcond=None)[0]
            idx = np.flatnonzero(itm)[ex[itm] > cont]
            cash[idx] = ex[idx]
    pv = cash * disc
    return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(n))


@pytest.mark.parametrize("call", [False, True])
def test_the_american_pide_without_jumps_is_the_black_scholes_american_price(call):
    bs = MODELS["bs"]
    ref = pde.vanilla(S, K, T, R, Q, 0.22, call, american=True, n_space=1600, n_time=800).price
    assert pide.american(bs, S, K, T, R, Q, call).price == pytest.approx(ref, abs=1e-3)


@pytest.mark.parametrize("name", ["merton", "kou", "vg", "nig"])
def test_the_american_put_under_jump_models_agrees_with_simulation_and_refines(name):
    m = MODELS[name]
    coarse = pide.american(m, S, K, T, R, Q, False)
    fine = pide.american(m, S, K, T, R, Q, False, n_space=2000, n_time=400)
    assert abs(coarse.price - fine.price) < 2e-3, "the grid is fine enough to be converged to a fifth of a cent"
    est, se = _lsmc_put(m, K)
    assert abs(fine.price - est) < 4.0 * se + 0.02, (fine.price, est, se)       # the simulation's own bias (a Bermudan lower bound, a regression) is a couple of cents


@pytest.mark.parametrize("name", ["bs", "merton", "kou", "vg", "nig"])
def test_early_exercise_adds_value_to_a_put_and_none_to_a_call_without_dividends(name):
    m = MODELS[name]
    put_a, put_e = pide.american(m, S, K, T, R, Q, False).price, pide.european(m, S, K, T, R, Q, False).price
    assert put_a > put_e + 0.05 and put_a >= max(K - S, 0.0), "a put is worth more when it can be exercised early"
    call_a = pide.american(m, S, K, T, R, 0.0, True).price
    call_e = pide.european(m, S, K, T, R, 0.0, True).price
    assert call_a == pytest.approx(call_e, abs=2e-3), "with no dividend a call is never exercised early"
    assert pide.american(m, S, K, T, R, Q, True).price >= pide.european(m, S, K, T, R, Q, True).price - 1e-9


def test_the_early_exercise_boundary_of_a_put_lies_below_the_strike_and_falls_with_the_time_left():
    res = pide.american(MODELS["merton"], S, K, T, R, Q, False, boundary=True)
    edge = res.boundary[1:]
    assert np.all(np.isfinite(edge)) and np.all(edge < K) and np.all(edge > 0.4 * K)
    assert edge[-1] < edge[len(edge) // 4] < edge[2] <= K, "the more time is left, the lower the price at which exercising stops being optimal"
    assert res.tau.shape == res.boundary.shape and res.tau[-1] == pytest.approx(T)


def test_a_coarse_grid_is_refined_until_the_exercise_problem_is_well_posed():
    bs = levy.BlackScholes(sigma=0.05)                  # little diffusion against a large drift: the cell Peclet number of 20 cells is above one
    coarse = pide.american(bs, S, K, 1.0, 0.1, 0.0, False, n_space=20, n_time=100)
    fine = pide.american(bs, S, K, 1.0, 0.1, 0.0, False, n_space=200, n_time=100)
    assert coarse.n_space > 20 and fine.n_space < 300, "only the grid that was too coarse for the sweep to be exact is refined"
    ref = pde.vanilla(S, K, 1.0, 0.1, 0.0, 0.05, False, american=True, n_space=1600, n_time=800).price
    assert abs(fine.price - ref) < 0.01 and ref > pide.european(bs, S, K, 1.0, 0.1, 0.0, False, n_space=400).price + 0.3
    assert coarse.price > pide.european(bs, S, K, 1.0, 0.1, 0.0, False, n_space=400).price, "even refined only to well-posedness it is a valid American value, not an oscillating one"
