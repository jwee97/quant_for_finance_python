"""Tabular and linear reinforcement learning against exact dynamic programming: the temporal-difference methods reach the Bellman values, the policy-gradient methods reach an optimal policy, G-learning reaches
the soft Bellman fixed point and its inverse recovers a planted reward."""

from __future__ import annotations

import numpy as np
import pytest

from src.control.mdp import MDP, value_iteration
from src.rl import envs, glearning as gl, policy_gradient as pg, tabular as tb


def small_env(seed=3, S=4, A=2, gamma=0.7):
    return envs.random_mdp_env(S, A, gamma, seed=seed)


def epsilon_greedy_q(mdp, Qstar, eps):
    S, A = mdp.n_states, mdp.n_actions
    pol = np.full((S, A), eps / A)
    pol[np.arange(S), Qstar.argmax(1)] += 1 - eps
    P = np.einsum("sa,ast->st", pol, mdp.P)
    R = (pol * mdp.R).sum(axis=1)
    V = np.linalg.solve(np.eye(S) - mdp.gamma * P, R)
    return mdp.R + mdp.gamma * np.einsum("ast,t->sa", mdp.P, V)


# ------------------------------------------------------------------------------------------------------------------ environments
def test_mdp_env_follows_its_transition_matrix():
    env = small_env()
    counts = np.zeros(env.n_states)
    env.reset(seed=0)
    env.s = 0
    for _ in range(4000):
        env.s = 0
        s2, r, done = env.step(1)
        counts[s2] += 1
        assert r == pytest.approx(env.mdp.R[0, 1]) and not done
    assert counts / counts.sum() == pytest.approx(env.mdp.P[1, 0], abs=0.03)


def test_mdp_env_terminal_states_and_horizon():
    P = np.array([[[0, 1], [0, 1]]], float)
    env = envs.MDPEnv(MDP(P, np.ones((2, 1))), terminal=(1,))
    env.reset()
    assert env.step(0)[2] is True
    env = envs.MDPEnv(MDP(P, np.ones((2, 1))), horizon=3)
    env.reset()
    assert [env.step(0)[2] for _ in range(3)] == [False, False, True]


def test_cliff_walking_geometry_and_exact_solution():
    cw = envs.CliffWalking()
    mdp = cw.to_mdp(0.999)
    V, pi, _ = value_iteration(mdp, tol=1e-9)
    assert V[cw.start] == pytest.approx(-9.0, abs=0.05)                                                 # the shortest path: up, seven across, down
    cw.reset()
    for a in (1,):                                                                                      # stepping right from the start falls off the cliff
        s, r, done = cw.step(a)
    assert r == -100.0 and s == cw.start and not done
    assert cw.step(0)[1] == -1.0


def test_regime_env_has_a_known_optimum_that_differs_by_regime():
    env = envs.RegimeFractionEnv()
    f0, f1 = env.optimal_fraction(0), env.optimal_fraction(1)
    assert f0 > f1 > 0.3
    assert f0 == pytest.approx(0.12 / 0.25 ** 2, rel=0.15) and f1 == pytest.approx(0.04 / 0.25 ** 2, rel=0.15)
    assert env.expected_reward(0, f0) > env.expected_reward(0, f1) and env.expected_reward(1, f1) > env.expected_reward(1, f0)
    o = env.reset(seed=1)
    assert o.sum() == 1.0 and env.step(1.0)[0].sum() == 1.0


# ------------------------------------------------------------------------------------------------------------------ temporal-difference control
@pytest.mark.parametrize("method", ["q_learning", "double_q"])
def test_off_policy_methods_learn_the_optimal_action_values(method):
    env = small_env()
    mdp = env.to_mdp()
    V, pi, _ = value_iteration(mdp)
    res = tb.train(env, method, episodes=400, max_steps=100, gamma=mdp.gamma, alpha=0.5, omega=0.6, epsilon=0.1, seed=0)
    assert np.abs(res.Q - mdp.q_values(V)).max() < 0.15 and (res.policy == pi).all()


@pytest.mark.parametrize("method", ["sarsa", "expected_sarsa"])
def test_on_policy_methods_learn_the_values_of_the_policy_they_follow(method):
    env = small_env()
    mdp = env.to_mdp()
    V, pi, _ = value_iteration(mdp)
    res = tb.train(env, method, episodes=400, max_steps=100, gamma=mdp.gamma, alpha=0.5, omega=0.6, epsilon=0.1, seed=0)
    assert np.abs(res.Q - epsilon_greedy_q(mdp, mdp.q_values(V), 0.1)).max() < 0.15 and (res.policy == pi).all()


def test_sarsa_takes_the_safe_path_and_q_learning_the_edge_on_the_cliff():
    cw = envs.CliffWalking()
    sarsa = tb.train(cw, "sarsa", episodes=1000, gamma=1.0, alpha=0.5, omega=0.5, epsilon=0.1, seed=0)
    qlearn = tb.train(cw, "q_learning", episodes=1000, gamma=1.0, alpha=0.5, omega=0.5, epsilon=0.1, seed=0)
    assert np.mean(sarsa.returns[-100:]) > np.mean(qlearn.returns[-100:]) + 5                              # while exploring, SARSA falls off the cliff less
    assert tb.greedy_return(cw, qlearn.policy) == pytest.approx(-9.0)                                      # the greedy policy of Q-learning is the optimal edge path
    assert tb.greedy_return(cw, sarsa.policy) < -9.0                                                       # SARSA's is longer, further from the cliff
    assert tb.greedy_return(cw, sarsa.policy) > -20.0


def test_td_methods_reject_unknown_methods_and_epsilon_schedules_run():
    with pytest.raises(ValueError):
        tb.train(small_env(), "nope")
    res = tb.train(small_env(), "q_learning", episodes=20, max_steps=20, gamma=0.7, epsilon=(0.5, 0.0))
    assert len(res.returns) == 20 and res.visits.sum() > 0


# ------------------------------------------------------------------------------------------------------------------ policy gradients
def test_reinforce_and_a2c_find_an_optimal_policy():
    env = small_env()
    mdp = env.to_mdp()
    V, pi, _ = value_iteration(mdp)
    for res in (pg.reinforce(env, episodes=1500, gamma=0.7, lr=0.5, seed=0), pg.a2c(env, steps=40000, gamma=0.7, seed=0)):
        assert np.abs(mdp.policy_value(res.theta.argmax(axis=1)) - V).max() < 1e-6
        p = res.probabilities(pg.one_hot(4)(0))
        assert p.sum() == pytest.approx(1.0) and p.argmax() == pi[0]


def test_a2c_critic_learns_the_value_of_its_policy():
    env = small_env()
    mdp = env.to_mdp()
    res = pg.a2c(env, steps=60000, gamma=0.7, actor_lr=0.02, critic_lr=0.02, seed=1)
    soft = np.array([pg.softmax(res.theta[s]) for s in range(4)])
    P = np.einsum("sa,ast->st", soft, mdp.P)
    exact = np.linalg.solve(np.eye(4) - 0.7 * P, (soft * mdp.R).sum(axis=1))
    assert np.abs(res.critic - exact).max() < 0.5


def test_deterministic_policy_gradient_finds_the_regime_fractions():
    env = envs.RegimeFractionEnv()
    f = np.array([env.optimal_fraction(0), env.optimal_fraction(1)])
    fits = [pg.deterministic_policy_gradient(env, steps=40000, seed=s).theta for s in (0, 1, 2)]
    mean = np.mean(fits, axis=0)
    assert np.abs(mean - f).max() < 0.35 and mean[0] > mean[1] + 0.4


# ------------------------------------------------------------------------------------------------------------------ G-learning and GIRL
def test_soft_value_iteration_interpolates_between_the_prior_and_the_greedy_policy():
    env = small_env(2, 4, 3, 0.8)
    mdp = env.to_mdp()
    V, pi, _ = value_iteration(mdp)
    cold = gl.soft_value_iteration(mdp, None, 1e-4)
    hot = gl.soft_value_iteration(mdp, None, 500.0)
    assert np.allclose(cold.policy, 1 / 3, atol=1e-3)                                                    # too costly to deviate: the prior
    assert (hot.policy.argmax(axis=1) == pi).all() and np.abs(hot.F - V).max() < 0.02                    # free to deviate: ordinary dynamic programming
    prior = np.array([[0.7, 0.2, 0.1]] * 4)
    mid = gl.soft_value_iteration(mdp, prior, 2.0)
    assert np.allclose(mid.F, gl.soft_value(mid.G, prior, 2.0)) and np.allclose(mid.G, mdp.q_values(mid.F))
    assert np.allclose(mid.policy, prior * np.exp(2.0 * (mid.G - mid.F[:, None])))                       # pi = pi_0 exp(beta (G - F))


def test_soft_value_never_exceeds_the_unregularised_value():
    mdp = small_env(2, 4, 3, 0.8).to_mdp()
    V, _, _ = value_iteration(mdp)
    for beta in (0.5, 2.0, 10.0):
        assert (gl.soft_value_iteration(mdp, None, beta).F <= V + 1e-9).all()


def test_sample_g_learning_approaches_the_soft_fixed_point():
    env = small_env(2, 4, 3, 0.8)
    mdp = env.to_mdp()
    exact = gl.soft_value_iteration(mdp, None, 1.0)
    learned = gl.g_learning(env, None, 1.0, episodes=300, gamma=0.8, max_steps=100, omega=0.6, seed=0)
    assert np.abs(learned.policy - exact.policy).max() < 0.05 and np.abs(learned.F - exact.F).max() < 0.15
    stale = gl.g_learning(env, None, 1.0, episodes=300, gamma=0.8, max_steps=100, omega=0.6, explore=0.0, seed=0)
    assert np.abs(stale.F - exact.F).max() > 2 * np.abs(learned.F - exact.F).max()                      # without exploring, rarely taken actions keep stale values


def _girl_problem(seed=0, S=6, A=3, K=4, beta=2.0, gamma=0.9, episodes=300):
    rng = np.random.default_rng(seed)
    P = rng.random((A, S, S)) ** 2
    P /= P.sum(axis=2, keepdims=True)
    X = rng.normal(size=(S, A, K))
    theta = np.array([1.0, -0.5, 0.8, 0.3])
    mdp = MDP(P, X @ theta, gamma)
    policy = gl.soft_value_iteration(mdp, None, beta).policy
    env = envs.MDPEnv(mdp, seed=seed)
    trajs = []
    for _ in range(episodes):
        s = env.reset()
        tr = []
        for _ in range(30):
            a = int(rng.choice(A, p=policy[s]))
            tr.append((s, a))
            s, _, _ = env.step(a)
        trajs.append(tr)
    return P, X, theta, policy, gl.demonstration_counts(trajs, S, A), beta, gamma


def test_girl_recovers_a_planted_reward_and_the_policy_it_implies():
    P, X, theta, policy, counts, beta, gamma = _girl_problem()
    res = gl.girl(P, X, counts, gamma, beta)
    assert res.success and np.corrcoef(res.theta, theta)[0, 1] > 0.99 and np.abs(res.theta - theta).max() < 0.15
    assert gl.mean_policy_kl(policy, res.policy) < 0.01 < gl.mean_policy_kl(policy, np.full_like(policy, 1 / 3))   # far better than saying nothing


def test_girl_with_little_data_is_shrunk_toward_zero_and_counts_are_counted():
    P, X, theta, policy, counts, beta, gamma = _girl_problem(episodes=300)
    few = np.zeros_like(counts)
    few[0, 0] = 1
    small = gl.girl(P, X, few, gamma, beta, l2=1.0)
    assert np.linalg.norm(small.theta) < np.linalg.norm(theta)
    assert counts.sum() == 300 * 30
    assert gl.demonstration_counts([[(0, 1), (0, 1), (2, 0)]], 3, 2).tolist() == [[0, 2], [0, 0], [1, 0]]
