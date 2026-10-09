"""The quadratic-programming solver behind the constrained portfolio and multi-period problems: closed forms, a general solver and the failure modes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from src.equity.qp import solve_qp as admm
from src.equity.qp import solve_qp_ipm as ipm


@pytest.fixture(params=[admm, ipm], ids=["admm", "ipm"])
def solve_qp(request):
    return request.param


def _cov(n, seed=0, ridge=0.05):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    return B @ B.T / n + ridge * np.eye(n)


def test_the_minimum_variance_portfolio_matches_its_closed_form(solve_qp):
    S = _cov(30)
    r = solve_qp(S, np.zeros(30), A=np.ones((1, 30)), l=[1.0], u=[1.0])
    w = np.linalg.solve(S, np.ones(30))
    w /= w.sum()
    assert r.ok and np.abs(r.x - w).max() < 1e-9 and abs(r.objective - 0.5 * w @ S @ w) < 1e-12


def test_a_box_and_budget_problem_agrees_with_a_general_solver(solve_qp):
    n, rng = 25, np.random.default_rng(1)
    S, mu = _cov(n, 2), rng.normal(0.05, 0.05, n)
    r = solve_qp(5.0 * S, -mu, A=np.ones((1, n)), l=[1.0], u=[1.0], lb=0.0, ub=0.15)
    ref = minimize(lambda x: 0.5 * x @ (5 * S) @ x - mu @ x, np.full(n, 1 / n), jac=lambda x: 5 * S @ x - mu, bounds=[(0, 0.15)] * n,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    assert r.ok and abs(r.objective - ref.fun) < 1e-8 and np.abs(r.x - ref.x).max() < 1e-5
    assert (r.x >= -1e-9).all() and (r.x <= 0.15 + 1e-9).all() and abs(r.x.sum() - 1) < 1e-9


def test_a_linear_program_is_solved_at_a_vertex(solve_qp):
    n, rng = 20, np.random.default_rng(3)
    c = rng.normal(size=n)
    r = solve_qp(np.zeros((n, n)), c, A=np.ones((1, n)), l=[1.0], u=[1.0], lb=0.0, ub=0.2)
    best = np.sort(c)[:5].sum() * 0.2                       # five names at the cap are the cheapest way to spend a budget of one
    assert r.ok and abs(r.objective - best) < 1e-8


def test_an_unconstrained_problem_solves_the_normal_equations(solve_qp):
    S, q = _cov(8, 4), np.arange(8.0)
    r = solve_qp(S, q)
    assert r.ok and np.abs(S @ r.x + q).max() < 1e-8


def test_equality_constraints_are_met_exactly_and_dependent_rows_do_not_break_it(solve_qp):
    n, rng = 12, np.random.default_rng(5)
    S, q = _cov(n, 6), rng.normal(size=n)
    A = np.vstack([np.ones(n), np.arange(n, dtype=float), np.ones(n)])          # the third row repeats the first
    b = np.array([0.0, 1.0, 0.0])
    r = solve_qp(S, q, A=A, l=b, u=b)
    assert r.ok and np.abs(A @ r.x - b).max() < 1e-8
    K = np.block([[S, A[:2].T], [A[:2], np.zeros((2, 2))]])
    ref = np.linalg.solve(K, np.concatenate([-q, b[:2]]))[:n]
    assert np.abs(r.x - ref).max() < 1e-7


def test_the_answer_does_not_depend_on_the_scale_of_the_objective(solve_qp):
    n, rng = 15, np.random.default_rng(7)
    S, mu = _cov(n, 8), rng.normal(0.0, 1.0, n)
    base = solve_qp(S, -mu, A=np.ones((1, n)), l=[0.0], u=[0.0], lb=-0.2, ub=0.2)
    ref = minimize(lambda x: 0.5 * x @ S @ x - mu @ x, np.zeros(n), jac=lambda x: S @ x - mu, bounds=[(-0.2, 0.2)] * n, constraints=[{"type": "eq", "fun": lambda x: x.sum()}],
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    assert base.ok and np.abs(base.x - ref.x).max() < 1e-6
    for k in (1e-9, 1e-6, 1e-3, 1e3, 1e6):                  # portfolio problems mix covariances near 1e-4 with alphas near 1e-2: the answer must not care
        scaled = solve_qp(k * S, -k * mu, A=np.ones((1, n)), l=[0.0], u=[0.0], lb=-0.2, ub=0.2)
        assert scaled.ok and np.abs(scaled.x - base.x).max() < 1e-8, k


def test_contradictory_bounds_and_infeasible_constraints_are_reported_not_returned_as_answers(solve_qp):
    r = solve_qp(np.eye(2), np.zeros(2), lb=[1.0, 0.0], ub=[0.0, 1.0])
    assert not r.ok and r.status == "infeasible bounds"
    n = 3
    r = solve_qp(np.eye(n), np.zeros(n), A=np.vstack([np.ones(n), np.ones(n)]), l=[1.0, -np.inf], u=[1.0, 0.0], max_iter=2000)       # the sum must be 1 and at most 0
    assert not r.ok and r.status == "max iterations"


def test_the_solver_handles_an_alpha_book_with_factor_neutrality_at_equity_size(solve_qp):
    n, k, rng = 300, 5, np.random.default_rng(9)
    F = rng.normal(size=(n, k))
    S = F @ F.T / k * 1e-4 + np.diag(rng.uniform(1e-4, 4e-4, n))
    alpha = rng.normal(0, 0.01, n)
    A = np.vstack([np.ones(n), F.T])
    lo, hi = np.concatenate([[0.0], np.full(k, -0.05)]), np.concatenate([[0.0], np.full(k, 0.05)])
    r = solve_qp(10 * S, -alpha, A=A, l=lo, u=hi, lb=-0.05, ub=0.05)
    assert r.ok and abs(r.x.sum()) < 1e-8 and (np.abs(F.T @ r.x) <= 0.05 + 1e-7).all() and (np.abs(r.x) <= 0.05 + 1e-9).all()
    assert alpha @ r.x > 0                                  # it holds what the alpha likes
    assert r.iterations < 5000


@pytest.mark.parametrize("seed", range(5))
def test_random_problems_satisfy_the_optimality_conditions(solve_qp, seed):
    n, m, rng = 20, 6, np.random.default_rng(100 + seed)
    S, q = _cov(n, seed), rng.normal(size=n)
    A = rng.normal(size=(m, n))
    x0 = rng.normal(size=n)
    l, u = A @ x0 - rng.uniform(0, 1, m), A @ x0 + rng.uniform(0, 1, m)              # feasible by construction
    r = solve_qp(S, q, A=A, l=l, u=u, lb=-3.0, ub=3.0)
    assert r.ok
    Ax = A @ r.x
    assert (Ax >= l - 1e-6).all() and (Ax <= u + 1e-6).all()
    # stationarity: the gradient is balanced by the multipliers of the constraints (the box rows come after the general ones)
    grad = S @ r.x + q
    nb = int(np.isfinite(r.y[m:]).sum())
    resid = grad + A.T @ r.y[:m] + r.y[m:m + n] if nb == n else None
    assert resid is not None and np.abs(resid).max() < 1e-6
