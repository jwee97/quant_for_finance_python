"""Environments: a small interface (``reset``, ``step``) with tabular worlds that can be turned into an exact :class:`~src.control.mdp.MDP`, and a regime-dependent portfolio problem with a known optimum.

``step(action)`` returns ``(observation, reward, done)``. Tabular environments use integer states and actions and expose ``n_states``, ``n_actions`` and ``to_mdp()``, so what a learner finds can be compared with
the exact solution of the same world.
"""

from __future__ import annotations

import numpy as np

from ..control.mdp import MDP


class Env:
    def reset(self, seed: int | None = None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class TabularEnv(Env):
    n_states: int
    n_actions: int

    def to_mdp(self, gamma: float = 0.95) -> MDP:
        raise NotImplementedError


class MDPEnv(TabularEnv):
    """An :class:`MDP` as an environment: the reward is ``R[s, a]`` (plus optional noise), the next state is drawn from ``P[a, s]``; an episode ends in a terminal state or, if ``horizon`` is given, after that many
    steps. A learner that runs a fixed number of steps per episode (``max_steps``) treats the cut as a truncation, not a terminal state."""

    def __init__(self, mdp: MDP, horizon: int | None = None, start: int = 0, terminal: tuple = (), reward_noise: float = 0.0, seed: int = 0):
        self.mdp, self.horizon, self.start, self.terminal, self.noise = mdp, horizon, start, set(terminal), reward_noise
        self.n_states, self.n_actions = mdp.n_states, mdp.n_actions
        self.rng = np.random.default_rng(seed)
        self.s, self.t = start, 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s, self.t = self.start, 0
        return self.s

    def step(self, action):
        r = self.mdp.R[self.s, action] + (self.noise * self.rng.normal() if self.noise else 0.0)
        self.s = int(self.rng.choice(self.n_states, p=self.mdp.P[action, self.s]))
        self.t += 1
        return self.s, float(r), (self.horizon is not None and self.t >= self.horizon) or (self.s in self.terminal)

    def to_mdp(self, gamma: float | None = None) -> MDP:
        return self.mdp if gamma is None else MDP(self.mdp.P, self.mdp.R, gamma)


class CliffWalking(TabularEnv):
    """Sutton and Barto's cliff: a ``rows x cols`` grid, start at the bottom left, goal at the bottom right, the cells between them are a cliff (reward -100, back to the start); every step costs 1.
    The shortest path runs along the cliff edge; the safe path runs along the top. Q-learning learns the first (it is the optimal greedy policy) but, with an exploring policy, falls off sometimes; SARSA learns
    the safer path (it is optimal for the epsilon-greedy policy it follows). Actions: 0 up, 1 right, 2 down, 3 left. ``slip`` makes a move go in a random direction with that probability."""

    def __init__(self, rows: int = 4, cols: int = 8, slip: float = 0.0, seed: int = 0):
        self.rows, self.cols, self.slip = rows, cols, slip
        self.n_states, self.n_actions = rows * cols, 4
        self.start, self.goal = (rows - 1) * cols, rows * cols - 1
        self.cliff = {(rows - 1) * cols + c for c in range(1, cols - 1)}
        self.rng = np.random.default_rng(seed)
        self.s, self.t = self.start, 0

    def _move(self, s: int, a: int) -> int:
        r, c = divmod(s, self.cols)
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[a]
        return min(max(r + dr, 0), self.rows - 1) * self.cols + min(max(c + dc, 0), self.cols - 1)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s, self.t = self.start, 0
        return self.s

    def step(self, action):
        a = int(self.rng.integers(4)) if self.slip and self.rng.random() < self.slip else int(action)
        nxt = self._move(self.s, a)
        self.t += 1
        if nxt in self.cliff:
            self.s = self.start
            return self.s, -100.0, False
        self.s = nxt
        return self.s, -1.0, self.s == self.goal

    def to_mdp(self, gamma: float = 0.99) -> MDP:
        S, A = self.n_states, 4
        P, R = np.zeros((A, S, S)), np.full((S, A), -1.0)
        for s in range(S):
            for a in range(A):
                if s == self.goal:
                    P[a, s, s], R[s, a] = 1.0, 0.0
                    continue
                for b in range(A):
                    w = (1 - self.slip) * (b == a) + self.slip / 4
                    if w == 0:
                        continue
                    nxt = self._move(s, b)
                    if nxt in self.cliff:
                        P[a, s, self.start] += w
                        R[s, a] += w * (-99.0)
                    else:
                        P[a, s, nxt] += w
        return MDP(P, R, gamma)


def random_mdp_env(n_states: int = 6, n_actions: int = 3, gamma: float = 0.9, seed: int = 0) -> MDPEnv:
    rng = np.random.default_rng(seed)
    P = rng.random((n_actions, n_states, n_states)) ** 2
    P /= P.sum(axis=2, keepdims=True)
    return MDPEnv(MDP(P, rng.normal(size=(n_states, n_actions)), gamma), seed=seed)


class RegimeFractionEnv(Env):
    """A portfolio problem whose answer is known: each period the investor puts a fraction ``a`` of wealth in a risky asset (the rest earns ``rf``) and the reward is the log growth of wealth. The risky asset's
    expected excess return depends on a regime that follows a Markov chain and is observed (a one-hot vector); the return is lognormal with volatility ``sigma``. The regime does not depend on the action, so the
    optimal policy maximises each period's expected log growth and the best fraction in each regime can be found by quadrature (:meth:`optimal_fraction`; about ``mu / sigma^2`` for small periods, the Kelly or
    Merton fraction). The action is a single number in ``[0, max_fraction]``. An episode is ``horizon`` periods long; the end is a time limit, not a terminal state (learners keep bootstrapping through it)."""

    def __init__(self, mu_excess=(0.12, 0.04), sigma: float = 0.25, rf: float = 0.02, stay: float = 0.9, max_fraction: float = 2.5, horizon: int = 50, seed: int = 0):
        self.mu, self.sigma, self.rf, self.stay, self.max_fraction, self.horizon = np.asarray(mu_excess, dtype=float), sigma, rf, stay, max_fraction, horizon
        self.n_regimes = len(self.mu)
        self.obs_dim, self.act_dim = self.n_regimes, 1
        self.rng = np.random.default_rng(seed)
        self.regime, self.t = 0, 0

    def _obs(self):
        o = np.zeros(self.n_regimes)
        o[self.regime] = 1.0
        return o

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.regime, self.t = int(self.rng.integers(self.n_regimes)), 0
        return self._obs()

    def growth(self, regime: int, action: float, z: np.ndarray | float) -> np.ndarray | float:
        gross = np.exp(np.log1p(self.rf + self.mu[regime]) - 0.5 * self.sigma ** 2 + self.sigma * np.asarray(z))        # the risky asset's gross return
        return np.maximum(1.0 + self.rf + action * (gross - 1.0 - self.rf), 0.05)

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], 0.0, self.max_fraction))
        r = float(np.log(self.growth(self.regime, a, self.rng.normal())))
        if self.rng.random() > self.stay:
            self.regime = int(self.rng.integers(self.n_regimes))
        self.t += 1
        return self._obs(), r, self.t >= self.horizon

    def expected_reward(self, regime: int, action: float, n: int = 41) -> float:
        z, w = np.polynomial.hermite_e.hermegauss(n)
        return float((w / w.sum() * np.log(self.growth(regime, action, z))).sum())

    def optimal_fraction(self, regime: int) -> float:
        grid = np.linspace(0.0, self.max_fraction, 2501)
        return float(grid[np.argmax([self.expected_reward(regime, a) for a in grid])])
