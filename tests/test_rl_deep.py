"""Deep reinforcement learning on a portfolio problem with a known answer: each method learns to hold a larger fraction of wealth in the risky asset in the regime where it pays more, close to the optimum
found by quadrature. (Torch only; the networks are small and the budgets short, so the tolerances are those of a sample of a few thousand noisy log returns.)"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from src.rl import deep, envs  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return envs.RegimeFractionEnv(horizon=50)


@pytest.fixture(scope="module")
def optimum(env):
    return np.array([env.optimal_fraction(0), env.optimal_fraction(1)])


def regret(env, table, optimum):
    """Expected log growth lost per period, averaged over the two regimes, against the optimum."""
    return float(np.mean([env.expected_reward(i, optimum[i]) - env.expected_reward(i, table[i]) for i in range(2)]))


@pytest.mark.parametrize("name,steps", [("ddpg", 5000), ("td3", 5000), ("sac", 4000), ("ppo", 16000)])
def test_continuous_action_methods_learn_the_regime_fractions(env, optimum, name, steps):
    agent = deep.AGENTS[name]().train(env, steps, seed=0)
    table = agent.policy_table(env)
    assert np.all((table >= 0) & (table <= env.max_fraction))
    assert np.abs(table - optimum).max() < 0.9
    random_regret = np.mean([regret(env, np.full(2, a), optimum) for a in np.linspace(0, env.max_fraction, 26)])
    assert regret(env, table, optimum) < 0.5 * random_regret                                           # far better than an arbitrary fraction


def test_double_dqn_learns_the_regime_fractions_on_a_grid(env, optimum):
    agent = deep.DQN(double=True).train(env, 6000, seed=0)
    table = agent.policy_table(env)
    assert set(np.round(table, 6)) <= set(np.round(agent.grid, 6))                                      # actions come from the grid
    assert table[0] > table[1] and np.abs(table - optimum).max() < 0.9


def test_agents_are_deterministic_given_a_seed(env):
    a = deep.DDPG().train(env, 600, seed=3).policy_table(env)
    b = deep.DDPG().train(env, 600, seed=3).policy_table(env)
    assert np.allclose(a, b)


def test_sac_adapts_its_temperature_and_acts_stochastically_when_asked(env):
    agent = deep.SAC().train(env, 800, seed=0)
    assert agent.temperature != pytest.approx(0.2)
    o = np.array([1.0, 0.0], dtype=np.float32)
    draws = {round(agent.act(o, deterministic=False), 5) for _ in range(5)}
    assert len(draws) > 1 and round(agent.act(o, True), 6) == round(agent.act(o, True), 6)


def test_td3_critic_prefers_the_better_action(env):
    agent = deep.TD3().train(env, 5000, seed=0)
    o = np.array([1.0, 0.0], dtype=np.float32)
    assert agent.q_value(o, 1.6) > agent.q_value(o, 0.0)                                                # levered long beats cash when the premium is high
