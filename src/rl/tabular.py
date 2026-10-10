"""Tabular temporal-difference control: SARSA, expected SARSA, Q-learning and double Q-learning.

All four learn the action-value table ``Q(s, a)`` from experience by moving it toward a one-step target,

    SARSA            r + gamma Q(s', a')                     a' is the action the behaviour policy actually takes next (on-policy)
    expected SARSA   r + gamma sum_a' pi(a'|s') Q(s', a')     the same in expectation over the behaviour policy: less noise
    Q-learning       r + gamma max_a' Q(s', a')               the greedy action, whatever was taken (off-policy)
    double Q         r + gamma Q_B(s', argmax_a' Q_A(s', a')) two tables, one picks the action and the other values it: removes the upward bias of the maximum of noisy estimates

with the step size decaying as ``alpha_0 / (1 + visits)^omega`` so the tables converge, and epsilon-greedy exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .envs import TabularEnv

METHODS = ("sarsa", "expected_sarsa", "q_learning", "double_q")


@dataclass
class TDResult:
    Q: np.ndarray
    returns: list = field(default_factory=list)       # undiscounted return of each episode
    visits: np.ndarray | None = None

    @property
    def policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)


def epsilon_greedy(Q: np.ndarray, s: int, epsilon: float, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] >= Q[s].max() - 1e-12)
    return int(rng.choice(best))


def train(env: TabularEnv, method: str = "q_learning", episodes: int = 2000, gamma: float = 0.95, alpha: float = 0.5, omega: float = 0.75, epsilon: float | tuple = 0.1, seed: int = 0,
          max_steps: int = 500) -> TDResult:
    """Learn ``Q`` by running ``episodes`` episodes. ``epsilon`` is a constant or ``(start, end)`` decayed linearly over the episodes."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rng = np.random.default_rng(seed)
    S, A = env.n_states, env.n_actions
    Q = np.zeros((S, A))
    QB = np.zeros((S, A)) if method == "double_q" else None
    n = np.zeros((S, A))
    returns = []
    for ep in range(episodes):
        eps = epsilon if np.isscalar(epsilon) else epsilon[0] + (epsilon[1] - epsilon[0]) * ep / max(episodes - 1, 1)
        s = env.reset(seed=seed if ep == 0 else None)
        total = 0.0
        Qsum = (Q + QB) if method == "double_q" else Q
        a = epsilon_greedy(Qsum, s, eps, rng)
        for _ in range(max_steps):
            s2, r, done = env.step(a)
            total += r
            Qsum = (Q + QB) if method == "double_q" else Q
            a2 = epsilon_greedy(Qsum, s2, eps, rng)
            lr = alpha / (1.0 + n[s, a]) ** omega
            n[s, a] += 1
            cont = 0.0 if done else 1.0
            if method == "sarsa":
                target = r + gamma * cont * Q[s2, a2]
                Q[s, a] += lr * (target - Q[s, a])
            elif method == "expected_sarsa":
                p = np.full(A, eps / A)
                p[np.flatnonzero(Q[s2] >= Q[s2].max() - 1e-12)[0]] += 1 - eps
                target = r + gamma * cont * float(p @ Q[s2])
                Q[s, a] += lr * (target - Q[s, a])
            elif method == "q_learning":
                target = r + gamma * cont * Q[s2].max()
                Q[s, a] += lr * (target - Q[s, a])
            else:
                if rng.random() < 0.5:
                    target = r + gamma * cont * QB[s2, int(Q[s2].argmax())]
                    Q[s, a] += lr * (target - Q[s, a])
                else:
                    target = r + gamma * cont * Q[s2, int(QB[s2].argmax())]
                    QB[s, a] += lr * (target - QB[s, a])
            s, a = s2, a2
            if done:
                break
        returns.append(total)
    return TDResult((Q + QB) / 2 if method == "double_q" else Q, returns, n)


def greedy_return(env: TabularEnv, policy: np.ndarray, episodes: int = 20, seed: int = 0, max_steps: int = 500) -> float:
    """Mean undiscounted return of a deterministic policy."""
    out = []
    for ep in range(episodes):
        s = env.reset(seed=seed + ep)
        total = 0.0
        for _ in range(max_steps):
            s, r, done = env.step(int(policy[s]))
            total += r
            if done:
                break
        out.append(total)
    return float(np.mean(out))
