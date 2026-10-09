"""Multi-period trading with costs: the closed-form dynamic programme against an independent solve of the whole problem, the trade-rate form of the policy, model-predictive control against the closed form,
constraints, and the no-trade region of a proportional cost."""

from __future__ import annotations

import numpy as np
import pytest

from src.equity import multiperiod as mp


def world(n=3, seed=0, T=6, decay=0.8, cost=0.4):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 2))
    S = (F @ F.T + np.diag(rng.uniform(0.5, 1.0, n))) * 0.02
    a0 = rng.normal(0.0, 0.02, n)
    path = mp.decaying_path(a0, decay, T)
    return S, path, mp.cost_matrix(S, cost, "risk")


def stacked_optimum(alpha, S, gamma, Lam, x0, beta):
    """The first-order conditions of the whole T-period problem as one linear system, solved directly: the oracle the recursion is checked against."""
    T, n = alpha.shape
    w = beta ** np.arange(T + 1)
    H = np.zeros((T * n, T * n))
    rhs = np.zeros(T * n)
    for s in range(T):
        sl = slice(s * n, (s + 1) * n)
        H[sl, sl] += w[s] * (gamma * S + Lam) + (w[s + 1] * Lam if s < T - 1 else 0.0)
        rhs[sl] += w[s] * alpha[s]
        if s > 0:
            H[sl, (s - 1) * n:s * n] -= w[s] * Lam
        if s < T - 1:
            H[sl, (s + 1) * n:(s + 2) * n] -= w[s + 1] * Lam
    rhs[:n] += w[0] * Lam @ x0
    return np.linalg.solve(H, rhs).reshape(T, n)


# ------------------------------------------------------------------------------------------------------------------ the closed form
@pytest.mark.parametrize("seed,beta", [(0, 1.0), (1, 0.97), (2, 0.9)])
def test_the_recursion_is_the_solution_of_the_whole_problem(seed, beta):
    S, path, Lam = world(seed=seed)
    x0 = np.random.default_rng(seed + 10).normal(0, 0.3, 3)
    sol = mp.lq_solve(path, S, 5.0, Lam, x0, beta)
    assert np.allclose(sol.holdings, stacked_optimum(path, S, 5.0, Lam, x0, beta), atol=1e-9)
    assert sol.value == pytest.approx(mp.utility(sol.holdings, path, S, 5.0, x0, Lam, None, beta), abs=1e-10)       # the value function is the utility of the path it prescribes
    assert sol.value == pytest.approx(sol.value_from(x0))


def test_nothing_nearby_does_better_and_the_policy_is_closed_loop():
    S, path, Lam = world(seed=3)
    x0 = np.zeros(3)
    sol = mp.lq_solve(path, S, 5.0, Lam, x0)
    rng = np.random.default_rng(0)
    for _ in range(25):
        other = sol.holdings + rng.normal(0, 0.02, sol.holdings.shape)
        assert mp.utility(other, path, S, 5.0, x0, Lam) < sol.value                                         # a concave problem: every perturbation is worse
    shock = np.array([0.4, -0.3, 0.2])                                                                        # started somewhere else, the same rule gives the optimal continuation
    again = mp.lq_solve(path, S, 5.0, Lam, shock)
    assert np.allclose(sol.next_holdings(0, shock), again.holdings[0], atol=1e-10)
    assert again.value == pytest.approx(sol.value_from(shock), abs=1e-10)


def test_the_policy_is_a_partial_adjustment_toward_an_aim():
    S, path, Lam = world(seed=4)
    x0 = np.array([0.1, -0.2, 0.3])
    sol = mp.lq_solve(path, S, 5.0, Lam, x0, 0.98)
    prev = x0
    for t in range(len(path)):
        assert np.allclose(sol.trades[t], sol.trade_rate[t] @ (sol.aim[t] - prev), atol=1e-10)
        eig = np.linalg.eigvals(sol.trade_rate[t]).real
        assert (eig > -1e-12).all() and (eig < 1 + 1e-12).all()                                              # it closes part of the gap: neither backwards nor past the aim
        prev = sol.holdings[t]


def test_free_trading_is_markowitz_and_prohibitive_costs_freeze_the_book():
    S, path, _ = world(seed=5)
    free = mp.lq_solve(path, S, 5.0, np.zeros((3, 3)), np.zeros(3))
    for t in range(len(path)):
        assert np.allclose(free.holdings[t], np.linalg.solve(5.0 * S, path[t]), atol=1e-9)                  # the single-period mean-variance portfolio each period
    x0 = np.array([0.3, 0.1, -0.2])
    frozen = mp.lq_solve(path, S, 5.0, mp.cost_matrix(S, 1e7, "risk"), x0)
    assert np.abs(frozen.holdings - x0).max() < 1e-4
    mild = mp.lq_solve(path, S, 5.0, mp.cost_matrix(S, 1.0, "risk"), np.zeros(3))
    assert np.abs(np.diff(mild.holdings, axis=0)).sum() < np.abs(np.diff(free.holdings, axis=0)).sum()       # costs smooth the path


def test_a_forecast_that_will_improve_or_fade_moves_the_position_now():
    S = np.array([[0.04]])
    Lam = np.array([[0.1]])
    rising = np.array([[0.01], [0.03], [0.05]])
    falling = rising[::-1].copy()
    up, down = mp.lq_solve(rising, S, 5.0, Lam), mp.lq_solve(falling, S, 5.0, Lam)
    myopic = lambda a: a / (5.0 * 0.04)
    assert up.holdings[0, 0] > myopic(0.01) + 0.005                                                          # buy ahead of the better days: the position looks forward
    assert down.holdings[0, 0] < myopic(0.05) - 0.05                                                         # and does not chase a forecast that is about to fade


def test_the_stationary_trade_rate_matches_the_scalar_closed_form_and_moves_the_right_way():
    gamma, sig2, beta = 4.0, 0.03, 0.97

    def closed(lam):
        k = gamma * sig2 + lam * (1 + beta)
        m = (k + np.sqrt(k * k - 4 * beta * lam * lam)) / 2                                                 # the stable root of  m^2 - k m + beta lam^2 = 0
        return 1 - lam / m

    rates = []
    for lam in (0.005, 0.02, 0.1, 0.5):
        Q, M, A = mp.stationary_trade_rate([[sig2]], gamma, [[lam]], beta)
        assert Q[0, 0] == pytest.approx(closed(lam), abs=1e-9)
        rates.append(Q[0, 0])
    assert rates == sorted(rates, reverse=True) and 0 < rates[-1] < rates[0] < 1                            # dearer trading, slower trading
    assert mp.stationary_trade_rate([[sig2]], 8.0, [[0.05]], beta)[0][0, 0] > mp.stationary_trade_rate([[sig2]], 2.0, [[0.05]], beta)[0][0, 0]   # a more risk-averse investor closes the gap faster


def test_cost_shapes_and_validation():
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert np.allclose(mp.cost_matrix(S, 2.0, "risk"), 2 * S) and np.allclose(mp.cost_matrix(S, 2.0, "identity"), 2 * np.eye(2))
    assert np.allclose(mp.cost_matrix(S, 2.0, "variance"), np.diag([0.08, 0.18])) and np.allclose(mp.cost_matrix(S, [1.0, 2.0]), np.diag([1.0, 2.0]))
    for bad in (lambda: mp.cost_matrix(S, -1.0), lambda: mp.cost_matrix(S, 1.0, "nope"), lambda: mp.lq_solve(np.zeros((3, 2)), S, 0.0, S), lambda: mp.lq_solve(np.zeros((3, 2)), S, 1.0, S, discount=1.5),
                lambda: mp.lq_solve(np.zeros((3, 3)), S, 1.0, S), lambda: mp.decaying_path([1.0], 1.5, 3)):
        with pytest.raises(ValueError):
            bad()


# ------------------------------------------------------------------------------------------------------------------ model-predictive control
def test_mpc_without_constraints_is_the_closed_form_whatever_the_horizon():
    S, path, Lam = world(seed=6, T=10)
    x0 = np.array([0.2, 0.0, -0.1])
    exact = mp.lq_solve(path, S, 5.0, Lam, x0, 0.99)
    for H in (1, 2, 4, 10):
        now, plan = mp.mpc_step(path, S, 5.0, x0, Lam, None, 0.99, plan_horizon=H)
        assert plan.ok and np.allclose(now, exact.holdings[0], atol=1e-6), H                                # with the exact terminal value, planning short loses nothing
        assert np.allclose(plan.holdings, exact.holdings[:H], atol=1e-6)
    short, _ = mp.mpc_step(path, S, 5.0, x0, Lam, None, 0.99, plan_horizon=2, terminal=False)
    assert np.abs(short - exact.holdings[0]).max() > 1e-4                                                    # without the terminal reward a short plan is myopic


def test_the_two_solvers_agree_on_a_constrained_plan():
    S, path, Lam = world(seed=7, T=4)
    x0 = np.zeros(3)
    kwargs = dict(lb=0.0, ub=0.5, net=(0.0, 0.6), max_gross=0.6, cost_linear=0.002)
    a = mp.mpc_plan(path * 4, S, 5.0, x0, Lam, solver="ipm", **kwargs)
    b = mp.mpc_plan(path * 4, S, 5.0, x0, Lam, solver="admm", **kwargs)
    assert a.ok and b.ok and np.allclose(a.holdings, b.holdings, atol=2e-4) and a.objective == pytest.approx(b.objective, abs=1e-6)


def test_constraints_are_respected_and_never_beat_the_unconstrained_bound():
    rng = np.random.default_rng(8)
    n, T = 4, 8
    F = rng.normal(size=(n, 2))
    S = (F @ F.T + np.eye(n)) * 0.02
    path = mp.decaying_path(rng.normal(0, 0.05, n), 0.85, T)
    Lam = mp.cost_matrix(S, 0.3)
    bound = mp.lq_solve(path, S, 3.0, Lam)
    for constraints in ({"lb": 0.0}, {"lb": -0.3, "ub": 0.3}, {"lb": -1.0, "ub": 1.0, "max_gross": 0.8, "net": (-0.1, 0.1)}, {"lb": 0.0, "ub": 0.4, "net": (0.0, 1.0), "max_gross": 1.0}):
        plan = mp.mpc_plan(path, S, 3.0, np.zeros(n), Lam, **constraints)
        assert plan.ok
        X = plan.holdings
        if "lb" in constraints:
            assert X.min() >= constraints["lb"] - 1e-7
        if "ub" in constraints:
            assert X.max() <= constraints["ub"] + 1e-7
        if "max_gross" in constraints:
            assert np.abs(X).sum(axis=1).max() <= constraints["max_gross"] + 1e-7
        if "net" in constraints:
            assert X.sum(axis=1).min() >= constraints["net"][0] - 1e-7 and X.sum(axis=1).max() <= constraints["net"][1] + 1e-7
        assert mp.utility(X, path, S, 3.0, np.zeros(n), Lam) <= bound.value + 1e-9                          # constraints only lower the best achievable utility
        assert plan.objective == pytest.approx(mp.utility(X, path, S, 3.0, np.zeros(n), Lam), abs=1e-7)    # and the plan reports the utility of what it holds


def test_slack_constraints_change_nothing_and_binding_ones_do():
    S, path, Lam = world(seed=9, T=5)
    free = mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam)
    slack = mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam, lb=-10.0, ub=10.0, max_gross=100.0, net=(-100.0, 100.0))
    assert np.allclose(free.holdings, slack.holdings, atol=1e-6)
    tight = mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam, lb=-0.01, ub=0.01)
    assert np.abs(tight.holdings).max() <= 0.01 + 1e-9 and tight.objective < free.objective - 1e-6


def test_a_loading_limit_holds_in_every_period():
    S, path, Lam = world(seed=10, T=4)
    beta = np.array([1.2, 0.8, 1.0])
    plan = mp.mpc_plan(path * 3, S, 5.0, np.zeros(3), Lam, exposures=beta[None, :], exposure_bounds=[[-0.05, 0.05]])
    assert plan.ok and np.abs(plan.holdings @ beta).max() <= 0.05 + 1e-7
    loose = mp.mpc_plan(path * 3, S, 5.0, np.zeros(3), Lam)
    assert np.abs(loose.holdings @ beta).max() > 0.05                                                       # the limit was doing something


def test_infeasible_plans_are_reported_not_returned():
    S, path, Lam = world(seed=11, T=3)
    plan = mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam, lb=0.5, ub=0.6, net=(0.0, 1.0))
    assert not plan.ok and np.isnan(plan.holdings).all()
    now, again = mp.mpc_step(path, S, 5.0, np.array([0.1, 0.1, 0.1]), Lam, lb=0.5, ub=0.6, net=(0.0, 1.0))
    assert np.allclose(now, [0.1, 0.1, 0.1]) and not again.ok                                                # the book is left as it was
    for bad in (lambda: mp.mpc_plan(path, S, 5.0, np.zeros(2), Lam), lambda: mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam, solver="nope"), lambda: mp.mpc_plan(path, S, 0.0, np.zeros(3), Lam),
                lambda: mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam, cost_linear=-0.1), lambda: mp.mpc_step(path, S, 5.0, np.zeros(3), Lam, plan_horizon=0)):
        with pytest.raises(ValueError):
            bad()


# ------------------------------------------------------------------------------------------------------------------ proportional costs and the no-trade region
def test_one_asset_with_a_proportional_cost_trades_to_the_edge_of_the_region_not_to_the_aim():
    gamma, sig2, alpha, kappa = 4.0, 0.05, 0.03, 0.01
    lo, hi = (alpha - kappa) / (gamma * sig2), (alpha + kappa) / (gamma * sig2)                           # the no-trade interval around the aim alpha / (gamma sigma^2)
    S = np.array([[sig2]])
    for start, expected in ((lo - 0.5, lo), (hi + 0.5, hi), ((lo + hi) / 2, (lo + hi) / 2), (lo + 1e-3, lo + 1e-3), (hi - 1e-3, hi - 1e-3)):
        out = mp.one_period_trade([start], [alpha], S, gamma, [kappa])
        assert out[0] == pytest.approx(expected, abs=1e-6), (start, expected)
    aim, half = mp.no_trade_region([alpha], S, gamma, [kappa])
    assert aim[0] == pytest.approx(alpha / (gamma * sig2)) and mp.inside_no_trade_region([lo + 1e-4], [alpha], S, gamma, [kappa]) and not mp.inside_no_trade_region([lo - 1e-3], [alpha], S, gamma, [kappa])


def test_many_assets_trade_only_the_ones_outside_the_region_and_to_its_boundary():
    rng = np.random.default_rng(12)
    n = 5
    F = rng.normal(size=(n, 2))
    S = (F @ F.T + np.eye(n)) * 0.03
    alpha = rng.normal(0, 0.03, n)
    kappa = np.full(n, 0.004)
    gamma = 4.0
    aim = np.linalg.solve(gamma * S, alpha)
    x0 = aim + rng.normal(0, 0.4, n)
    x1 = mp.one_period_trade(x0, alpha, S, gamma, kappa)
    grad = alpha - gamma * S @ x1
    traded = np.abs(x1 - x0) > 1e-6
    assert traded.any() and (~traded).any() or traded.all()
    assert np.allclose(np.abs(grad[traded]), kappa[traded], atol=1e-6)                                      # an asset that trades ends exactly on the boundary
    assert (np.abs(grad[~traded]) <= kappa[~traded] + 1e-7).all()                                           # one that does not was inside
    assert np.sign(grad[traded]).tolist() == np.sign(x1[traded] - x0[traded]).tolist()                       # and moved toward the aim, not away
    assert mp.inside_no_trade_region(x1, alpha, S, gamma, kappa, tol=1e-6)
    again = mp.one_period_trade(x1, alpha, S, gamma, kappa)
    assert np.allclose(again, x1, atol=1e-6)                                                                 # inside the region, nothing more to do


def brute_force_two_period(alpha, gamma, sig2, kappa, x_prev=0.0):
    grid = np.linspace(-1.5, 1.5, 601)
    x0, x1 = np.meshgrid(grid, grid, indexing="ij")
    j = (alpha[0] * x0 - 0.5 * gamma * sig2 * x0 ** 2 - kappa * np.abs(x0 - x_prev)) + (alpha[1] * x1 - 0.5 * gamma * sig2 * x1 ** 2 - kappa * np.abs(x1 - x0))
    k = np.unravel_index(np.argmax(j), j.shape)
    return grid[k[0]], grid[k[1]], j[k]


def test_the_plan_with_a_proportional_cost_is_the_global_optimum_and_waits_out_a_reversal():
    gamma, sig2, kappa = 1.0, 1.0, 0.6
    alpha = np.array([[1.0], [-1.0]])                                                                         # attractive now, but the forecast reverses next period
    S = np.array([[sig2]])
    plan = mp.mpc_plan(alpha, S, gamma, [0.0], None, [kappa])
    bx0, bx1, best = brute_force_two_period(alpha[:, 0], gamma, sig2, kappa)
    assert plan.ok and plan.objective == pytest.approx(best, abs=2e-4) and abs(plan.holdings[0, 0] - bx0) < 0.01 and abs(plan.holdings[1, 0] - bx1) < 0.01
    myopic = mp.one_period_trade([0.0], alpha[0], S, gamma, [kappa])[0]
    assert myopic == pytest.approx(0.4)                                                                       # (alpha - kappa) / (gamma sigma^2) when looking one period ahead
    assert plan.holdings[0, 0] < myopic - 0.05                                                                # looking ahead, it buys less now: the round trip is not worth the cost


def test_a_proportional_cost_and_the_quadratic_dynamic_programme_agree_when_it_is_zero():
    S, path, Lam = world(seed=13, T=5)
    exact = mp.lq_solve(path, S, 5.0, Lam, np.zeros(3), 0.99)
    plan = mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam, cost_linear=0.0, discount=0.99)
    assert np.allclose(plan.holdings, exact.holdings, atol=1e-6) and plan.objective == pytest.approx(exact.value, abs=1e-7)
    tiny = mp.mpc_plan(path, S, 5.0, np.zeros(3), Lam, cost_linear=1e-9, discount=0.99)                       # a vanishing proportional cost adds the auxiliary variables and changes nothing
    assert np.allclose(tiny.holdings, exact.holdings, atol=1e-5)
