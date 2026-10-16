"""The linear solvers: each against numpy's dense solve, the classical convergence-rate relations of the iterations, and the complementarity problem against an exhaustive search."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import sparse

from src.numerics import linsolve as ls


def _laplacian(n: int):
    """The matrix of the one-dimensional Poisson problem (2 on the diagonal, -1 beside it): the textbook case with known convergence rates."""
    return np.full(n - 1, -1.0), np.full(n, 2.0), np.full(n - 1, -1.0)


def _dense(lower, diag, upper) -> np.ndarray:
    return np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)


def _dominant(n: int, seed: int):
    rng = np.random.default_rng(seed)
    lower, upper = rng.uniform(-1, 1, n - 1), rng.uniform(-1, 1, n - 1)
    diag = np.abs(np.concatenate([[0.0], lower])) + np.abs(np.concatenate([upper, [0.0]])) + rng.uniform(0.5, 2.0, n)
    return lower, diag, upper


# --------------------------------------------------------------------------------------------------------------------------------------- direct
@pytest.mark.parametrize("n", [1, 2, 3, 50, 400])
def test_thomas_and_the_lapack_factorisation_agree_with_the_dense_solve_for_one_and_many_right_hand_sides(n):
    lower, diag, upper = _dominant(n, n) if n > 1 else (np.zeros(0), np.array([2.5]), np.zeros(0))
    A = _dense(lower, diag, upper)
    rng = np.random.default_rng(n)
    b, B = rng.normal(size=n), rng.normal(size=(n, 4))
    assert np.allclose(ls.thomas(lower, diag, upper, b), np.linalg.solve(A, b), atol=1e-11)
    assert np.allclose(ls.thomas(lower, diag, upper, B), np.linalg.solve(A, B), atol=1e-11)
    lu = ls.tridiagonal_factor(lower, diag, upper)
    assert np.allclose(lu.solve(b), np.linalg.solve(A, b), atol=1e-11) and np.allclose(lu.solve(B), np.linalg.solve(A, B), atol=1e-11)
    assert np.allclose(ls.tridiagonal_solve(lower, diag, upper, b), np.linalg.solve(A, b), atol=1e-11)


def test_thomas_refuses_a_zero_pivot_and_the_pivoting_solver_does_not():
    lower, diag, upper = np.array([1.0]), np.array([0.0, 0.0]), np.array([1.0])         # [[0, 1], [1, 0]]
    with pytest.raises(ZeroDivisionError):
        ls.thomas(lower, diag, upper, [1.0, 2.0])
    assert np.allclose(ls.tridiagonal_solve(lower, diag, upper, [1.0, 2.0]), [2.0, 1.0])
    with pytest.raises(ZeroDivisionError):
        ls.tridiagonal_factor(np.array([0.0]), np.array([1.0, 0.0]), np.array([0.0]))        # singular


def test_the_tridiagonal_solvers_check_their_shapes():
    with pytest.raises(ValueError):
        ls.thomas([1.0, 1.0], [2.0, 2.0], [1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        ls.thomas([1.0], [2.0, 2.0], [1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        ls.tridiagonal_factor([1.0], [2.0, 2.0, 2.0], [1.0]).solve([1.0, 2.0, 3.0])


# ------------------------------------------------------------------------------------------------------------------------------------ iterations
def test_the_classical_iterations_converge_and_agree_with_the_direct_solution_on_dense_and_sparse_matrices():
    lower, diag, upper = _dominant(40, 3)
    A = _dense(lower, diag, upper)
    b = np.random.default_rng(1).normal(size=40)
    exact = np.linalg.solve(A, b)
    for solver in (ls.jacobi, ls.gauss_seidel, lambda A_, b_, **kw: ls.sor(A_, b_, 1.1, **kw)):
        for M in (A, sparse.csr_matrix(A)):
            r = solver(M, b, tol=1e-12)
            assert r.converged and np.allclose(r.x, exact, atol=1e-9) and r.residual < 1e-12 and len(r.history) == r.iterations


def test_gauss_seidel_needs_half_the_sweeps_of_jacobi_and_the_optimal_omega_far_fewer():
    """For the Poisson matrix the Jacobi spectral radius is cos(pi/(n+1)), Gauss-Seidel's is its square, and SOR with the optimal factor reaches (w - 1)."""
    n = 40
    lower, diag, upper = _laplacian(n)
    A = _dense(lower, diag, upper)
    b = np.ones(n)
    rho = np.cos(np.pi / (n + 1))
    omega = ls.optimal_omega(lower, diag, upper)
    assert omega == pytest.approx(2.0 / (1.0 + np.sin(np.pi / (n + 1))), rel=1e-9)
    jac, gs, opt = ls.jacobi(A, b, tol=1e-9), ls.gauss_seidel(A, b, tol=1e-9), ls.sor(A, b, omega, tol=1e-9)
    assert jac.converged and gs.converged and opt.converged
    assert jac.iterations / gs.iterations == pytest.approx(2.0, rel=0.1), "Gauss-Seidel converges twice as fast as Jacobi"
    assert opt.iterations < gs.iterations / 5
    # the observed error contraction per sweep matches the spectral radii
    assert (jac.history[-1] / jac.history[-21]) ** (1 / 20) == pytest.approx(rho, abs=2e-3)
    assert (gs.history[-1] / gs.history[-21]) ** (1 / 20) == pytest.approx(rho ** 2, abs=3e-3)
    assert ls.sor(A, b, 1.0, tol=1e-9).iterations == gs.iterations


def test_sor_tridiagonal_matches_dense_sor_and_the_direct_solution_and_reports_non_convergence():
    lower, diag, upper = _laplacian(30)
    b = np.random.default_rng(2).normal(size=30)
    exact = np.linalg.solve(_dense(lower, diag, upper), b)
    omega = ls.optimal_omega(lower, diag, upper)
    r = ls.sor_tridiagonal(lower, diag, upper, b, omega, tol=1e-12)
    assert r.converged and np.allclose(r.x, exact, atol=1e-9) and r.residual < 1e-8
    assert ls.sor(_dense(lower, diag, upper), b, omega, tol=1e-12).iterations == pytest.approx(r.iterations, abs=3)
    slow = ls.sor_tridiagonal(lower, diag, upper, b, 1.0, tol=1e-14, max_iter=5)
    assert not slow.converged and slow.iterations == 5


def test_the_iterations_check_their_arguments():
    A = np.eye(3)
    for bad in (0.0, 2.0, -0.5):
        with pytest.raises(ValueError):
            ls.sor(A, np.ones(3), bad)
        with pytest.raises(ValueError):
            ls.sor_tridiagonal([0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0], np.ones(3), bad)
    with pytest.raises(ZeroDivisionError):
        ls.jacobi(np.array([[0.0, 1.0], [1.0, 1.0]]), np.ones(2))
    with pytest.raises(ValueError):
        ls.jacobi(np.ones((2, 3)), np.ones(2))
    assert ls.optimal_omega(np.array([1.0]), np.array([1.0, 1.0]), np.array([-1.0])) == 1.0, "no real Jacobi spectrum: fall back to Gauss-Seidel"


# ------------------------------------------------------------------------------------------------------------------------ the complementarity problem
def _american_like(n: int = 60, alpha: float = 0.6):
    """An M-matrix, a right-hand side and a decreasing obstacle that the solution touches at the low end: the structure of an American put's time step."""
    lower, diag, upper = np.full(n - 1, -alpha), np.full(n, 1.0 + 2.0 * alpha), np.full(n - 1, -alpha)
    s = np.linspace(0.2, 2.0, n)
    obstacle = np.maximum(1.0 - s, 0.0)
    rhs = 0.85 * obstacle + 0.1 * np.exp(-((s - 1.0) / 0.4) ** 2)
    return lower, diag, upper, rhs, obstacle


def _is_solution(lower, diag, upper, rhs, obstacle, x, tol=1e-7) -> bool:
    w = _dense(lower, diag, upper) @ x - rhs
    return bool((x >= obstacle - tol).all() and (w >= -tol).all() and abs(float(w @ (x - obstacle))) < tol)


def test_projected_sor_and_brennan_schwartz_solve_the_complementarity_problem_and_agree():
    lower, diag, upper, rhs, g = _american_like()
    bs = ls.brennan_schwartz(lower, diag, upper, rhs, g, "low")
    ps = ls.psor_tridiagonal(lower, diag, upper, rhs, g, omega=1.3, tol=1e-13)
    assert ps.converged and _is_solution(lower, diag, upper, rhs, g, ps.x) and _is_solution(lower, diag, upper, rhs, g, bs)
    assert np.allclose(bs, ps.x, atol=1e-8)
    assert (bs <= g + 1e-12).sum() > 3 and (bs > g + 1e-6).sum() > 3, "the problem has an exercise region and a holding region"
    free = ls.thomas(lower, diag, upper, rhs)
    assert (bs >= free - 1e-12).all() and (bs > free + 1e-6).any(), "the constraint raises the solution above the unconstrained one"


def test_the_complementarity_solvers_match_an_exhaustive_search_over_the_active_sets():
    rng = np.random.default_rng(5)
    n = 8
    lower, upper = np.full(n - 1, -0.7), np.full(n - 1, -0.7)
    diag = np.full(n, 2.4)
    g = rng.uniform(-1, 1, n)
    d = rng.uniform(-2, 2, n)
    M = _dense(lower, diag, upper)
    found = None
    for active in itertools.product([False, True], repeat=n):
        a = np.array(active)
        x = g.copy()
        if (~a).any():
            F = ~a
            x[F] = np.linalg.solve(M[np.ix_(F, F)], d[F] - M[np.ix_(F, a)] @ g[a])
        if _is_solution(lower, diag, upper, d, g, x, 1e-9):
            found = x
            break
    assert found is not None, "an M-matrix complementarity problem has a unique solution"
    assert np.allclose(ls.psor_tridiagonal(lower, diag, upper, d, g, tol=1e-14).x, found, atol=1e-8)


def test_brennan_schwartz_handles_an_exercise_region_at_the_high_end_by_mirroring_the_grid():
    lower, diag, upper, rhs, g = _american_like()
    mirrored = ls.brennan_schwartz(upper[::-1], diag[::-1], lower[::-1], rhs[::-1], g[::-1], "high")[::-1]
    assert np.allclose(mirrored, ls.brennan_schwartz(lower, diag, upper, rhs, g, "low"))
    lower2, diag2, upper2 = np.array([-0.5, -0.9, -0.2]), np.array([2.0, 2.2, 2.4, 2.0]), np.array([-0.3, -0.6, -0.8])          # a non-symmetric M-matrix: exercise at the top
    rhs2, g2 = np.array([0.1, 0.2, 0.3, 0.4]), np.array([-5.0, -5.0, 0.2, 0.9])
    x = ls.brennan_schwartz(lower2, diag2, upper2, rhs2, g2, "high")
    assert _is_solution(lower2, diag2, upper2, rhs2, g2, x)
    with pytest.raises(ValueError):
        ls.brennan_schwartz(lower, diag, upper, rhs, g, "middle")
    with pytest.raises(ValueError):
        ls.psor_tridiagonal(lower, diag, upper, rhs, g[:-1])
    with pytest.raises(ValueError):
        ls.psor_tridiagonal(lower, diag, upper, rhs, g, omega=2.0)


def test_with_an_obstacle_below_the_solution_the_constrained_solution_is_the_unconstrained_one():
    lower, diag, upper, rhs, _ = _american_like()
    free = ls.thomas(lower, diag, upper, rhs)
    low = np.full(len(rhs), -10.0)
    assert np.allclose(ls.brennan_schwartz(lower, diag, upper, rhs, low), free)
    assert np.allclose(ls.psor_tridiagonal(lower, diag, upper, rhs, low, tol=1e-13).x, free, atol=1e-9)
