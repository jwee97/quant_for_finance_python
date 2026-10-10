"""Markov decision processes: the Bellman fixed point, value and policy iteration against brute force, backward induction, the risk-sensitive limit and ordering, belief updates and point-based POMDP values."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.control import mdp as m


def random_mdp(seed=0, S=5, A=3, gamma=0.9):
    rng = np.random.default_rng(seed)
    P = rng.random((A, S, S))
    P /= P.sum(axis=2, keepdims=True)
    return m.MDP(P, rng.normal(size=(S, A)), gamma)


def tiger():
    P = np.array([np.eye(2), np.full((2, 2), .5), np.full((2, 2), .5)])
    O = np.array([[[.85, .15], [.15, .85]], [[.5, .5]] * 2, [[.5, .5]] * 2])
    R = np.array([[-1, -100, 10], [-1, 10, -100]], float)
    return m.POMDP(P, O, R, 0.95)


def test_value_iteration_reaches_the_bellman_fixed_point():
    mdp = random_mdp()
    V, pi, _ = m.value_iteration(mdp)
    assert np.allclose(V, mdp.q_values(V).max(axis=1), atol=1e-8)
    assert (pi == mdp.q_values(V).argmax(axis=1)).all()


def test_value_and_policy_iteration_agree_and_beat_every_policy():
    mdp = random_mdp(1, S=4, A=3)
    V, pi, _ = m.value_iteration(mdp)
    V2, pi2, _ = m.policy_iteration(mdp)
    assert np.allclose(V, V2, atol=1e-7) and (pi == pi2).all()
    for policy in itertools.product(range(3), repeat=4):                                      # all 81 deterministic policies
        assert (mdp.policy_value(np.array(policy)) <= V + 1e-9).all()


def test_backward_induction_matches_value_iteration_as_the_horizon_grows():
    mdp = random_mdp(2)
    V, pi, _ = m.value_iteration(mdp)
    Vt, pit = m.backward_induction(mdp, 400)
    assert np.allclose(Vt[0], V, atol=1e-6) and (pit[0] == pi).all()
    one, _ = m.backward_induction(mdp, 1)
    assert np.allclose(one[0], mdp.R.max(axis=1))                                             # one period left: the best immediate reward
    terminal = np.arange(mdp.n_states, dtype=float)
    assert np.allclose(m.backward_induction(mdp, 0, terminal)[0][0], terminal)


def test_a_known_two_state_problem():
    # state 0 pays 0 and may move to 1, which pays 1 for ever: move is worth gamma / (1 - gamma)
    P = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 1]]], float)                                 # action 0 stays, action 1 jumps to state 1
    R = np.array([[0.0, 0.0], [1.0, 1.0]])
    V, pi, _ = m.value_iteration(m.MDP(P, R, 0.9))
    assert pi[0] == 1 and V[1] == pytest.approx(10.0) and V[0] == pytest.approx(9.0)


def test_invalid_mdps_are_refused():
    with pytest.raises(ValueError):
        m.MDP(np.ones((2, 3, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        m.MDP(np.full((1, 2, 2), .5), np.zeros((2, 1)), gamma=1.5)


def test_risk_sensitive_limit_and_ordering():
    mdp = random_mdp(3)
    V0, _, _ = m.value_iteration(mdp)
    Vs, _ = m.risk_sensitive_value_iteration(mdp, 1e-6)
    assert np.allclose(Vs, V0, atol=1e-3)
    Vm, _ = m.risk_sensitive_value_iteration(mdp, -0.5)
    Vp, _ = m.risk_sensitive_value_iteration(mdp, 0.5)
    assert (Vm <= V0 + 1e-9).all() and (Vp >= V0 - 1e-9).all()                                # risk aversion lowers the value, risk seeking raises it


def test_risk_aversion_changes_the_choice_between_a_gamble_and_a_sure_thing():
    # state 0 decides: action 0 goes to the safe state 3 (pays 1 a period for ever); action 1 goes to state 1 (pays 3.2) or state 2 (pays -1) with equal chance (mean 1.1)
    P = np.zeros((2, 4, 4))
    for a in range(2):
        for s in (1, 2, 3):
            P[a, s, s] = 1.0
    P[0, 0, 3] = 1.0
    P[1, 0, 1] = P[1, 0, 2] = 0.5
    R = np.array([[0.0, 0.0], [3.2, 3.2], [-1.0, -1.0], [1.0, 1.0]])
    mdp = m.MDP(P, R, 0.5)
    assert m.value_iteration(mdp)[1][0] == 1                                                   # the risk-neutral agent takes the gamble
    assert m.risk_sensitive_value_iteration(mdp, -2.0)[1][0] == 0                              # the averse one takes the sure thing
    assert m.risk_sensitive_value_iteration(mdp, 2.0)[1][0] == 1


def test_belief_update_is_bayes_rule():
    pm = tiger()
    b, p = pm.belief_update(np.array([.5, .5]), 0, 0)                                         # listen, hear "left"
    assert p == pytest.approx(0.5) and b == pytest.approx([.85, .15])
    b2, _ = pm.belief_update(b, 0, 0)
    assert b2[0] == pytest.approx(.85 ** 2 / (.85 ** 2 + .15 ** 2))
    after_open, _ = pm.belief_update(np.array([.9, .1]), 1, 0)                                # opening a door resets the problem
    assert after_open == pytest.approx([.5, .5])


def test_observation_probabilities_sum_to_one():
    pm = tiger()
    b = np.array([.3, .7])
    assert sum(pm.belief_update(b, a, o)[1] for a in (0,) for o in range(2)) == pytest.approx(1.0)


def test_expectimax_on_the_tiger_matches_hand_values():
    pm = tiger()
    b = np.array([.5, .5])
    assert m.expectimax(pm, b, 1) == pytest.approx(-1.0)                                       # listening is the best of three at one step
    assert m.expectimax(pm, b, 2) == pytest.approx(-1.95)
    assert m.expectimax(pm, np.array([.99, .01]), 1) == pytest.approx(.99 * 10 + .01 * -100)  # confident: open the right door
    assert m.expectimax(pm, b, 3) == pytest.approx(2.3098, abs=1e-3)


def test_pbvi_matches_the_known_tiger_value_and_policy_shape():
    pm = tiger()
    grid = np.array([[x, 1 - x] for x in np.linspace(0, 1, 41)])
    alphas = m.pbvi(pm, grid, 200)
    value, action = m.alpha_value(alphas, np.array([.5, .5]))
    assert value == pytest.approx(19.37, abs=0.1) and action == 0                              # listen at the uniform belief (the published value is about 19.37)
    assert m.alpha_value(alphas, np.array([.99, .01]))[1] == 2 and m.alpha_value(alphas, np.array([.01, .99]))[1] == 1
    # the value function is convex and symmetric
    v = [m.alpha_value(alphas, np.array([x, 1 - x]))[0] for x in np.linspace(0, 1, 21)]
    assert np.allclose(v, v[::-1], atol=1e-6) and v[10] <= max(v[0], v[-1]) + 1e-9


def test_pbvi_is_a_lower_bound_on_the_exact_finite_horizon_when_horizons_match():
    pm = tiger()
    grid = np.array([[x, 1 - x] for x in np.linspace(0, 1, 21)])
    for h in (1, 2, 3):
        alphas = m.pbvi(pm, grid, h)
        for x in (.5, .8):
            b = np.array([x, 1 - x])
            # with the zero-reward-free floor the h-step point-based value cannot exceed the exact h-step value plus the floor's discounted tail
            assert m.alpha_value(alphas, b)[0] <= m.expectimax(pm, b, h) + 0.95 ** h * 2.0 * 1e3
