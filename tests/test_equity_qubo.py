"""Portfolio QUBOs: the matrices reproduce the objective exactly, annealing finds the true optimum that exhaustive search finds, and the decoded portfolios are what the continuous problem would pick."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.equity import qubo as qb
from src.equity.qp import solve_qp


def problem(n=6, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 3))
    cov = F @ F.T / 3 * 0.04 + np.diag(rng.uniform(0.01, 0.03, n))
    mu = rng.uniform(0.02, 0.12, n)
    return mu, cov


def test_the_weight_qubo_energy_is_the_penalised_mean_variance_objective_for_every_bit_pattern():
    mu, cov = problem(4, 1)
    q = qb.weight_qubo(mu, cov, risk_aversion=4.0, bits=2, w_max=0.6)
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.integers(0, 2, 8).astype(float)
        w = q.decode(x)
        direct = 0.5 * 4.0 * w @ cov @ w - mu @ w + q.penalty * (w.sum() - 1.0) ** 2
        assert q.energy(x)[0] == pytest.approx(direct, rel=1e-10, abs=1e-12)
    assert q.decode(np.ones(8)).tolist() == pytest.approx([0.6] * 4) and q.decode(np.zeros(8)).tolist() == [0.0] * 4        # all bits on is the maximum weight
    assert q.decode(np.array([1, 0, 0, 0, 0, 0, 0, 0.0]))[0] == pytest.approx(0.6 / 3)                                    # the lowest bit adds w_max / (2^bits - 1)


def test_the_cardinality_qubo_energy_is_the_equal_weighted_objective_plus_the_penalty():
    mu, cov = problem(7, 2)
    q = qb.cardinality_qubo(mu, cov, 3, risk_aversion=5.0)
    rng = np.random.default_rng(2)
    for _ in range(60):
        x = rng.integers(0, 2, 7).astype(float)
        k = x.sum()
        direct = (0.5 * 5.0 / 9.0) * x @ cov @ x - mu @ x / 3.0 + q.penalty * (k - 3.0) ** 2
        assert q.energy(x)[0] == pytest.approx(direct, rel=1e-10, abs=1e-12)
        if k == 3:
            assert q.energy(x)[0] == pytest.approx(qb.selection_objective(mu, cov, x > 0.5, 5.0), rel=1e-10, abs=1e-12)        # on the constraint the penalty vanishes


def test_brute_force_is_exact_and_annealing_finds_the_same_minimum_on_random_instances():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(14, 14))
        Q = np.triu(A)
        x_exact, e_exact = qb.brute_force(Q)
        x_sa, e_sa, energies = qb.simulated_annealing(Q, sweeps=300, restarts=24, seed=seed)
        M = 0.5 * (Q + Q.T)
        assert x_exact @ M @ x_exact == pytest.approx(e_exact) and e_sa == pytest.approx(x_sa @ M @ x_sa)
        assert e_sa == pytest.approx(e_exact, abs=1e-9), seed                                    # the annealer reaches the global minimum
        assert (energies >= e_exact - 1e-9).all()
    with pytest.raises(ValueError, match="26 bits"):
        qb.brute_force(np.zeros((27, 27)))


def test_annealing_finds_a_planted_ground_state_in_a_frustrated_landscape():
    # a ferromagnetic chain with a planted ground state of alternating bits: bit i prefers to differ from its neighbours
    N = 22
    Q = np.zeros((N, N))
    for i in range(N - 1):
        Q[i, i] += -1.0; Q[i + 1, i + 1] += -1.0; Q[i, i + 1] += 2.0                             # (x_i - x_{i+1})^2 reversed in sign: reward alternation
    Q = Q + np.diag(np.linspace(0, 0.01, N))                                                     # break the tie between the two alternating patterns
    x_sa, e_sa, _ = qb.simulated_annealing(Q, sweeps=500, restarts=32, seed=3)
    x_exact, e_exact = qb.brute_force(Q)
    assert e_sa == pytest.approx(e_exact, abs=1e-9) and np.array_equal(x_sa, x_exact)


def test_the_best_k_of_n_found_by_annealing_is_the_best_subset_by_exhaustive_search():
    hits = 0
    for seed in range(6):
        mu, cov = problem(14, seed)
        k = 4
        mask, objective = qb.anneal_selection(mu, cov, k, risk_aversion=6.0, sweeps=400, restarts=32, seed=seed)
        best = min(itertools.combinations(range(14), k), key=lambda c: qb.selection_objective(mu, cov, np.isin(np.arange(14), c), 6.0))
        best_obj = qb.selection_objective(mu, cov, np.isin(np.arange(14), best), 6.0)
        assert mask.sum() == k                                                                   # the penalty held the cardinality
        assert objective >= best_obj - 1e-12                                                     # nothing beats the exhaustive optimum
        hits += int(objective <= best_obj + 1e-12)
    assert hits >= 5                                                                             # and the annealer finds it almost every time


def test_annealed_weights_are_the_best_grid_point_and_close_to_the_continuous_solution():
    mu, cov = problem(5, 3)
    res = qb.anneal_weights(mu, cov, risk_aversion=8.0, bits=3, w_max=0.6, sweeps=400, restarts=32, seed=0)
    x_exact, e_exact = qb.brute_force(res.qubo.Q)
    assert res.energy == pytest.approx(e_exact + res.qubo.offset, abs=1e-9)                         # 15 bits: the exhaustive minimum
    assert abs(res.budget_error) <= 0.6 / 7 * 2.5                                               # within a couple of grid steps of fully invested
    assert (res.weights >= 0).all() and res.weights.max() <= 0.6 + 1e-12
    continuous = solve_qp(8.0 * cov, -mu, A=np.ones((1, 5)), l=[1.0], u=[1.0], lb=0.0, ub=0.6)
    assert np.abs(res.weights - continuous.x).max() < 0.6 / 7 * 1.6                              # each weight is within about a grid step of the continuous optimum
    finer = qb.anneal_weights(mu, cov, risk_aversion=8.0, bits=4, w_max=0.6, sweeps=500, restarts=48, seed=1)
    assert np.abs(finer.weights - continuous.x).max() <= np.abs(res.weights - continuous.x).max() + 1e-9


def test_a_larger_penalty_enforces_the_budget_more_tightly():
    mu, cov = problem(5, 4)
    tight = qb.weight_qubo(mu, cov, 5.0, 3, 0.6, penalty=50.0)
    x, e, _ = qb.simulated_annealing(tight.Q, sweeps=400, restarts=32, seed=0)
    assert abs(tight.decode(x).sum() - 1.0) <= 0.6 / 7
    with pytest.raises(ValueError):
        qb.weight_qubo(mu, cov[:3, :3])
    with pytest.raises(ValueError):
        qb.cardinality_qubo(mu, cov, 0)
    with pytest.raises(ValueError):
        qb.cardinality_qubo(mu, cov, 9)
    assert qb.selection_objective(mu, cov, np.zeros(5, bool)) == float("inf")
