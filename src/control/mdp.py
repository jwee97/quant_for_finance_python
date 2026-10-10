"""Markov decision processes: Bellman's equation solved exactly, by value iteration, policy iteration and backward induction, with a risk-sensitive version and a partially observed one.

A finite MDP is ``P[a, s, s']`` (the chance of going from ``s`` to ``s'`` under action ``a``), ``R[s, a]`` (the expected reward) and a discount ``gamma``. The optimal value satisfies Bellman's equation
``V(s) = max_a [R(s, a) + gamma sum_s' P(s' | s, a) V(s')]``; value iteration applies the right side repeatedly (a contraction with modulus ``gamma``, so it converges to the unique fixed point from anywhere),
policy iteration alternates solving for the value of a policy exactly with improving it, and backward induction handles a finite horizon.

**Risk-sensitive** control replaces the expectation by an *entropic* one, ``(1/theta) log E[exp(theta (r + gamma V'))]``: for ``theta < 0`` it penalises variance and bad outcomes (a Taylor expansion gives mean plus
``theta/2`` times the variance), for ``theta > 0`` it likes risk, and ``theta -> 0`` is the ordinary MDP.

**Partially observed** control (a POMDP) adds observations ``O[a, s', o]``. The agent cannot see the state, only a belief ``b`` (a probability vector) updated by Bayes' rule, and the problem is an MDP on beliefs.
The value function of a finite horizon is convex and piecewise linear in ``b``, the maximum of a set of ``alpha`` vectors; :func:`pbvi` improves a set of them at a grid of beliefs (point-based value iteration, Pineau,
Gordon and Thrun 2003) and :func:`expectimax` is the exact tree search the tests check it against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MDP:
    P: np.ndarray        # (A, S, S)
    R: np.ndarray        # (S, A)
    gamma: float = 0.95

    def __post_init__(self):
        self.P, self.R = np.asarray(self.P, dtype=float), np.asarray(self.R, dtype=float)
        A, S, S2 = self.P.shape
        if S != S2 or self.R.shape != (S, A) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("P must be (A, S, S), R (S, A), 0 <= gamma <= 1")
        if not np.allclose(self.P.sum(axis=2), 1.0, atol=1e-9) or (self.P < -1e-12).any():
            raise ValueError("every row of P must be a probability distribution")

    @property
    def n_states(self) -> int:
        return self.P.shape[1]

    @property
    def n_actions(self) -> int:
        return self.P.shape[0]

    def q_values(self, V: np.ndarray) -> np.ndarray:
        """``Q(s, a) = R(s, a) + gamma sum_s' P(s' | s, a) V(s')``."""
        return self.R + self.gamma * np.einsum("ast,t->sa", self.P, V)

    def policy_value(self, policy: np.ndarray) -> np.ndarray:
        """The exact value of a deterministic policy (an action per state): the solution of ``(I - gamma P_pi) V = R_pi``."""
        s = np.arange(self.n_states)
        P_pi, R_pi = self.P[policy, s, :], self.R[s, policy]
        return np.linalg.solve(np.eye(self.n_states) - self.gamma * P_pi, R_pi)


def value_iteration(mdp: MDP, tol: float = 1e-10, max_iter: int = 100000) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns ``(V, greedy policy, iterations)``."""
    V = np.zeros(mdp.n_states)
    for k in range(1, max_iter + 1):
        V_new = mdp.q_values(V).max(axis=1)
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    return V, mdp.q_values(V).argmax(axis=1), k


def policy_iteration(mdp: MDP, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray, int]:
    policy = np.zeros(mdp.n_states, dtype=int)
    for k in range(1, max_iter + 1):
        V = mdp.policy_value(policy)
        new = mdp.q_values(V).argmax(axis=1)
        new = np.where(np.isclose(mdp.q_values(V)[np.arange(mdp.n_states), new], mdp.q_values(V)[np.arange(mdp.n_states), policy]), policy, new)     # keep the old action on a tie
        if (new == policy).all():
            break
        policy = new
    return mdp.policy_value(policy), policy, k


def backward_induction(mdp: MDP, horizon: int, terminal: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Finite horizon: values ``V[t, s]`` and optimal actions ``pi[t, s]`` for ``t = 0 ... horizon - 1`` (``V[horizon]`` is the terminal reward, zero by default)."""
    V = np.zeros((horizon + 1, mdp.n_states))
    if terminal is not None:
        V[horizon] = terminal
    pi = np.zeros((horizon, mdp.n_states), dtype=int)
    for t in range(horizon - 1, -1, -1):
        Q = mdp.q_values(V[t + 1])
        V[t], pi[t] = Q.max(axis=1), Q.argmax(axis=1)
    return V, pi


def risk_sensitive_value_iteration(mdp: MDP, theta: float, tol: float = 1e-10, max_iter: int = 100000) -> tuple[np.ndarray, np.ndarray]:
    """Entropic risk-sensitive control with ``R(s, a)`` taken as the (deterministic) reward received on leaving ``s`` under ``a``:
    ``V(s) = max_a (1/theta) log sum_s' P(s'|s,a) exp(theta (R(s,a) + gamma V(s')))``. ``theta = 0`` is the ordinary problem; negative ``theta`` is risk-averse."""
    if abs(theta) < 1e-12:
        V, pi, _ = value_iteration(mdp, tol, max_iter)
        return V, pi
    V = np.zeros(mdp.n_states)
    for _ in range(max_iter):
        z = theta * (mdp.R[:, :, None] + mdp.gamma * V[None, None, :])                  # (S, A, S')
        m = z.max(axis=2, keepdims=True)
        Q = (m[:, :, 0] + np.log(np.einsum("ast,sat->sa", mdp.P, np.exp(z - m)) + 1e-300)) / theta
        V_new = Q.max(axis=1) if theta > 0 else Q.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    return V, Q.argmax(axis=1)


# ------------------------------------------------------------------------------------------------------------------ partial observability
@dataclass
class POMDP:
    P: np.ndarray        # (A, S, S)
    O: np.ndarray        # (A, S', O): the chance of observing o after action a lands in s'
    R: np.ndarray        # (S, A)
    gamma: float = 0.95

    def __post_init__(self):
        self.P, self.O, self.R = (np.asarray(x, dtype=float) for x in (self.P, self.O, self.R))
        A, S, _ = self.P.shape
        if self.O.shape[:2] != (A, S) or self.R.shape != (S, A):
            raise ValueError("P (A,S,S), O (A,S,O), R (S,A)")

    @property
    def n_states(self) -> int:
        return self.P.shape[1]

    @property
    def n_actions(self) -> int:
        return self.P.shape[0]

    @property
    def n_obs(self) -> int:
        return self.O.shape[2]

    def belief_update(self, b: np.ndarray, a: int, o: int) -> tuple[np.ndarray, float]:
        """Bayes' rule: the belief after action ``a`` and observation ``o``, and the probability of that observation."""
        pred = b @ self.P[a]
        joint = pred * self.O[a, :, o]
        p = float(joint.sum())
        return (joint / p if p > 0 else pred), p


def expectimax(pomdp: POMDP, b: np.ndarray, horizon: int) -> float:
    """The exact optimal value of a belief by searching the tree of actions and observations (exponential in the horizon: for checking small cases)."""
    if horizon == 0:
        return 0.0
    best = -np.inf
    for a in range(pomdp.n_actions):
        v = float(b @ pomdp.R[:, a])
        for o in range(pomdp.n_obs):
            nb, p = pomdp.belief_update(b, a, o)
            if p > 1e-14:
                v += pomdp.gamma * p * expectimax(pomdp, nb, horizon - 1)
        best = max(best, v)
    return best


def pbvi(pomdp: POMDP, beliefs: np.ndarray, iterations: int = 60) -> list[tuple[np.ndarray, int]]:
    """Point-based value iteration: a set of ``(alpha vector, action)`` pairs, improved at each belief of the grid in turn. The value of a belief is ``max alpha . b``."""
    S, A, O = pomdp.n_states, pomdp.n_actions, pomdp.n_obs
    floor = pomdp.R.min(axis=0) / (1.0 - pomdp.gamma) if pomdp.gamma < 1 else np.zeros(A)             # a lower bound: always take action a and get the worst reward forever
    alphas = [(np.full(S, float(floor.max())), int(floor.argmax()))]
    for _ in range(iterations):
        # for every action and observation, the vector gamma * sum_s' O[a,s',o] P[a,s,s'] alpha(s')
        proj = {(a, o, k): pomdp.gamma * pomdp.P[a] @ (pomdp.O[a][:, o] * alpha) for a in range(A) for o in range(O) for k, (alpha, _) in enumerate(alphas)}
        new = []
        for b in beliefs:
            best, best_val = None, -np.inf
            for a in range(A):
                vec = pomdp.R[:, a].copy()
                for o in range(O):
                    k = max(range(len(alphas)), key=lambda kk: float(b @ proj[(a, o, kk)]))
                    vec = vec + proj[(a, o, k)]
                val = float(b @ vec)
                if val > best_val:
                    best, best_val = (vec, a), val
            new.append(best)
        alphas = new
    return alphas


def alpha_value(alphas: list, b: np.ndarray) -> tuple[float, int]:
    """The value of belief ``b`` and the action of the best alpha vector."""
    vals = [float(b @ a) for a, _ in alphas]
    k = int(np.argmax(vals))
    return vals[k], alphas[k][1]
