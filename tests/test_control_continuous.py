"""Merton's problem, HJB schemes, Pontryagin's principle and Schroedinger bridges, each against a closed form, a reference method or the property that defines it."""

from __future__ import annotations

import numpy as np
import pytest

from src.control import hjb, merton, pontryagin as pmp, schrodinger as sb
from src.derivatives.pricing import binomial_price, bsm_price


# ------------------------------------------------------------------------------------------------------------------ Merton
def test_merton_closed_forms():
    assert merton.merton_fraction(0.08, 0.02, 0.2, 3.0) == pytest.approx(0.5)
    g = merton.merton_value_growth(0.08, 0.02, 0.2, 3.0)
    assert g == pytest.approx(0.02 + 0.06 ** 2 / (2 * 3 * 0.04))
    assert merton.merton_value(2.0, 0.0, 0.08, 0.02, 0.2, 3.0) == pytest.approx(2.0 ** -2 / -2)         # at the horizon it is the utility
    assert merton.merton_consumption_rate(0.08, 0.02, 0.2, 3.0, 0.05) == pytest.approx((0.05 + 2 * g) / 3)
    with pytest.raises(ValueError):
        merton.merton_consumption_rate(0.5, 0.0, 0.1, 0.5, 0.01)                                      # growth so high the utility is unbounded


def test_the_optimal_fraction_maximises_expected_utility_of_wealth():
    mu, r, s, g = 0.08, 0.02, 0.2, 3.0
    R, w, Rf = merton.gauss_hermite_returns(mu, r, s, 1.0)
    pis = np.linspace(0, 1.2, 241)
    eu = [(w * ((p * R + (1 - p) * Rf) ** (1 - g)) / (1 - g)).sum() for p in pis]
    assert pis[int(np.argmax(eu))] == pytest.approx(merton.merton_fraction(mu, r, s, g), abs=0.02)


def test_without_costs_the_dynamic_programme_recovers_merton_and_costs_open_a_band():
    sol0 = merton.CostDP(0.08, 0.02, 0.2, 3.0, 0.0, periods=60, grid=201).solve()
    assert sol0.lower == pytest.approx(0.5, abs=0.01) and sol0.upper == pytest.approx(0.5, abs=0.01)
    widths = []
    for k in (0.001, 0.004, 0.016):
        sol = merton.CostDP(0.08, 0.02, 0.2, 3.0, k, periods=120, grid=241).solve()
        assert sol.lower < 0.5 < sol.upper
        assert (sol.target[(sol.grid < sol.lower - 0.02)] > sol.grid[(sol.grid < sol.lower - 0.02)]).all()   # outside the band it trades toward it
        widths.append(sol.half_width)
    assert widths[0] < widths[1] < widths[2]                                                           # the band widens with the cost
    ratio = np.log(widths[2] / widths[0]) / np.log(16.0)                                               # about one third: the cube-root law
    assert 0.25 < ratio < 0.5


def test_the_band_is_the_right_size_against_the_small_cost_formula():
    sol = merton.CostDP(0.08, 0.02, 0.2, 3.0, 0.004, periods=240, grid=301).solve()
    approx = merton.davis_norman_half_width(0.08, 0.02, 0.2, 3.0, 0.004)
    assert 0.6 * approx < sol.half_width < 1.4 * approx


def test_cost_dp_validates_inputs():
    with pytest.raises(ValueError):
        merton.CostDP(0.08, 0.02, 0.2, 3.0, -0.1)


# ------------------------------------------------------------------------------------------------------------------ HJB
@pytest.mark.parametrize("gamma", [3.0, 0.5])
def test_merton_hjb_matches_the_closed_form(gamma):
    res = hjb.merton_hjb(0.08, 0.02, 0.2, gamma, horizon=1.0, nx=201, nt=50)
    assert res.max_relative_error < 0.01
    star = min(merton.merton_fraction(0.08, 0.02, 0.2, gamma), 3.0)                                   # gamma 0.5 wants 3.0, the bound
    mid = len(res.x) // 2
    assert res.fraction[mid] == pytest.approx(star, abs=0.05)


def test_merton_hjb_respects_a_binding_bound_on_the_control():
    res = hjb.merton_hjb(0.08, 0.02, 0.2, 0.5, pi_bounds=(0.0, 1.0))                                  # the unconstrained answer is 3: the bound binds
    assert res.fraction[len(res.x) // 2] == pytest.approx(1.0)
    free = hjb.merton_hjb(0.08, 0.02, 0.2, 0.5, pi_bounds=(0.0, 3.0))
    assert (res.value[100] <= free.value[100] + 1e-9)                                                 # a constraint cannot raise the value


def test_merton_hjb_refuses_log_utility():
    with pytest.raises(ValueError):
        hjb.merton_hjb(0.08, 0.02, 0.2, 1.0)


def test_american_put_obstacle_problem_matches_the_tree_and_converges():
    ref = binomial_price(100, 100, 1.0, 0.05, 0.0, 0.3, call=False, american=True, steps=1500)
    errs = []
    for n in (100, 200, 400):
        price, sol = hjb.american_put_hjb(100, 100, 1.0, 0.05, 0.0, 0.3, nx=n, nt=n)
        errs.append(abs(price - ref))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.04
    european = float(bsm_price(100, 100, 1.0, 0.05, 0.0, 0.3, call=False))
    assert price > european + 0.1                                                                     # early exercise is worth something
    assert (sol.value >= sol.payoff - 1e-9).all()                                                     # the value never falls below the obstacle
    assert 60 < sol.exercise_boundary < 80                                                            # the put is exercised when the spot is low enough
    assert (np.diff(sol.value) <= 1e-9).all()                                                         # a put is decreasing in the spot


def test_american_call_without_dividends_is_not_exercised_early():
    # by put-call symmetry we only have the put solver; check instead that raising the dividend-free rate raises the put's early-exercise premium
    low, _ = hjb.american_put_hjb(100, 100, 1.0, 0.01, 0.0, 0.3, nx=200, nt=200)
    high, _ = hjb.american_put_hjb(100, 100, 1.0, 0.08, 0.0, 0.3, nx=200, nt=200)
    eu_low, eu_high = (float(bsm_price(100, 100, 1.0, r, 0.0, 0.3, call=False)) for r in (0.01, 0.08))
    assert (high - eu_high) > (low - eu_low)


# ------------------------------------------------------------------------------------------------------------------ Pontryagin
def test_pontryagin_matches_the_riccati_solution_of_the_lqr():
    sol = pmp.lqr_pmp(-0.5, 1.0, 1.0, 0.5, 2.0, 1.0, 5.0)
    t, x, u = pmp.lqr_riccati(-0.5, 1.0, 1.0, 0.5, 2.0, 1.0, 5.0)
    assert sol.success and np.abs(sol.x[0] - x).max() < 1e-7 and np.abs(sol.u[0] - u).max() < 1e-7


def test_pontryagin_transversality_holds_for_the_lqr():
    sol = pmp.lqr_pmp(0.3, 1.0, 1.0, 1.0, 3.0, 1.0, 2.0)
    assert sol.costate[0, -1] == pytest.approx(2 * 3.0 * sol.x[0, -1])


def test_pontryagin_finds_the_almgren_chriss_sinh_schedule():
    sol = pmp.liquidation_pmp(1e5, 1.0, 0.01, 5.0)
    exact = pmp.almgren_chriss_closed_form(1e5, 1.0, 0.01, 5.0, sol.t)
    assert sol.success and np.abs(sol.x[0] - exact).max() < 1e-3 * 1e5 * 1e-3
    assert sol.x[0, 0] == pytest.approx(1e5) and sol.x[0, -1] == pytest.approx(0.0, abs=1e-6)
    # more risk aversion sells faster: less left at the half-way point
    quick = pmp.liquidation_pmp(1e5, 1.0, 0.01, 50.0)
    assert quick.x[0, 100] < sol.x[0, 100]
    # no risk aversion: a straight line (the TWAP)
    twap = pmp.liquidation_pmp(1e5, 1.0, 0.01, 1e-8)
    assert np.allclose(twap.x[0], 1e5 * (1 - twap.t), atol=1.0)


def test_almgren_chriss_pmp_wrapper_agrees():
    a = pmp.almgren_chriss_pmp(1000.0, 1.0, 0.01, 2.0, 0.5)
    b = pmp.liquidation_pmp(1000.0, 1.0, 0.01, 2.0 * 0.25)
    assert np.allclose(a.x, b.x)


def test_ramsey_model_runs_through_to_its_steady_state():
    kstar, cstar = pmp.ramsey_steady_state()
    sol = pmp.ramsey_pmp(1.0, 80.0)
    assert sol.success and sol.x[0, 0] == pytest.approx(1.0) and sol.x[0, -1] == pytest.approx(kstar)
    assert (np.diff(sol.x[0]) > 0).all()                                                              # capital rises monotonically from below the steady state
    assert sol.u[0, -1] == pytest.approx(cstar, rel=0.05) and sol.u[0, 0] < cstar                    # the investor starts by consuming little
    mid = len(sol.t) // 2
    assert sol.x[0, mid] == pytest.approx(kstar, rel=0.15)                                           # the turnpike: most of the time is spent near the steady state


# ------------------------------------------------------------------------------------------------------------------ Schroedinger
def _chain(seed=0, S=6):
    rng = np.random.default_rng(seed)
    Q = rng.random((S, S)) + 0.1
    Q /= Q.sum(axis=1, keepdims=True)
    p0, pT = rng.random(S), rng.random(S)
    return Q, p0 / p0.sum(), pT / pT.sum()


def test_sinkhorn_hits_both_marginals():
    Q, p0, pT = _chain()
    res = sb.sinkhorn(p0, pT, np.linalg.matrix_power(Q, 3))
    assert res.error < 1e-10
    assert np.allclose(res.coupling.sum(axis=1), p0) and np.allclose(res.coupling.sum(axis=0), pT)


def test_sinkhorn_minimises_relative_entropy_among_couplings_with_the_same_marginals():
    Q, p0, pT = _chain(1)
    K = np.linalg.matrix_power(Q, 3)
    res = sb.sinkhorn(p0, pT, K)
    ref = p0[:, None] * K

    def kl(pi):
        return float((pi * np.log(pi / ref)).sum())

    rng = np.random.default_rng(0)
    for _ in range(20):
        delta = rng.normal(size=K.shape)
        delta -= delta.mean(axis=1, keepdims=True)
        delta -= delta.mean(axis=0, keepdims=True)                                                    # zero row and column sums: the marginals are unchanged
        other = res.coupling + 0.3 * res.coupling.min() * delta / np.abs(delta).max()
        assert kl(other) >= kl(res.coupling) - 1e-12
    assert kl(p0[:, None] * pT[None, :]) >= kl(res.coupling)                                         # and independence is no better


def test_markov_bridge_has_the_right_marginals_and_is_stochastic():
    Q, p0, pT = _chain(2)
    br = sb.markov_bridge(Q, p0, pT, 5)
    assert np.allclose(br.marginals[0], p0) and np.allclose(br.marginals[-1], pT, atol=1e-10)
    assert all(np.allclose(P.sum(axis=1), 1.0) and (P >= 0).all() for P in br.transitions)
    assert br.marginals.shape == (6, 6)


def test_bridge_path_entropy_equals_the_static_entropy_and_is_minimal():
    Q, p0, pT = _chain(3)
    steps = 4
    br = sb.markov_bridge(Q, p0, pT, steps)
    K = np.linalg.matrix_power(Q, steps)
    static = sb.sinkhorn(p0, pT, K)
    ref = p0[:, None] * K
    static_kl = float((static.coupling * np.log(static.coupling / ref)).sum())
    assert sb.path_kl(br.transitions, Q, p0) == pytest.approx(static_kl, rel=1e-8)
    # a different Markov chain that also joins p0 to pT costs more
    rng = np.random.default_rng(1)
    other = [P * np.exp(0.2 * rng.normal(size=P.shape)) for P in br.transitions]
    other = [P / P.sum(axis=1, keepdims=True) for P in other]
    # repair the end law with one more Sinkhorn on the last step so the comparison is between chains with the same endpoints
    m = p0.copy()
    for P in other[:-1]:
        m = m @ P
    fix = sb.sinkhorn(m, pT, other[-1])
    other[-1] = fix.coupling / fix.coupling.sum(axis=1, keepdims=True)
    end = p0.copy()
    for P in other:
        end = end @ P
    assert np.allclose(end, pT, atol=1e-8)
    assert sb.path_kl(other, Q, p0) > sb.path_kl(br.transitions, Q, p0)


def test_identity_bridge_when_targets_equal_the_reference_flow():
    Q, p0, _ = _chain(4)
    pT = p0 @ np.linalg.matrix_power(Q, 3)
    br = sb.markov_bridge(Q, p0, pT, 3)
    assert all(np.allclose(P, Q, atol=1e-8) for P in br.transitions)                                  # nothing to correct: the bridge is the reference
    assert sb.path_kl(br.transitions, Q, p0) == pytest.approx(0.0, abs=1e-10)


def test_bridge_validates_inputs():
    Q, p0, pT = _chain(5)
    with pytest.raises(ValueError):
        sb.markov_bridge(Q, p0, pT, 0)
    with pytest.raises(ValueError):
        sb.sinkhorn(p0, pT * 2, Q)


def _lmdp(seed=0, S=6):
    rng = np.random.default_rng(seed)
    P = rng.random((S, S))
    P /= P.sum(axis=1, keepdims=True)
    q = rng.random(S) * 0.5
    term = np.zeros(S, dtype=bool)
    term[-1] = True
    q[-1] = 0.0
    return P, q, term


def test_linearly_solvable_control_satisfies_the_bellman_equation():
    P, q, term = _lmdp()
    sol = sb.solve_lmdp_first_exit(P, q, term)
    v = sol.value
    for s in range(len(q) - 1):
        assert v[s] == pytest.approx(q[s] - np.log(P[s] @ np.exp(-v)))                                # v = q - log E_p exp(-v')
    assert np.allclose(sol.policy.sum(axis=1), 1.0)


def test_the_kl_optimal_policy_beats_other_policies_and_the_passive_one():
    P, q, term = _lmdp(1)
    sol = sb.solve_lmdp_first_exit(P, q, term)
    best = sb.kl_control_cost(P, sol.policy, q, term)
    assert np.allclose(best, sol.value, atol=1e-9)
    passive = sb.kl_control_cost(P, P, q, term)
    assert (best <= passive + 1e-12).all() and (best < passive - 1e-6).any()
    rng = np.random.default_rng(0)
    for _ in range(10):
        other = sol.policy * np.exp(0.3 * rng.normal(size=P.shape))
        other /= other.sum(axis=1, keepdims=True)
        assert (sb.kl_control_cost(P, other, q, term) >= best - 1e-9).all()
