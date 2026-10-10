"""Schroedinger bridges and linearly solvable control.

**The bridge.** Given a reference process (a Markov chain with transition matrix ``Q``, or a Gaussian kernel) and two distributions, ``p0`` today and ``pT`` at the horizon, the Schroedinger problem asks for the
process closest to the reference in relative entropy (KL divergence) that starts at ``p0`` and ends at ``pT``. Its solution is the reference reweighted by a function of the endpoints,
``P(x_0, ..., x_T) = Q(x_0, ..., x_T) a(x_0) b(x_T)``, and ``a`` and ``b`` are found by *Sinkhorn's iteration* (also called iterative proportional fitting): alternately rescale so the start marginal, then the
end marginal, is right. Equivalently the bridge is the Markov chain with transitions ``P_t(x, y) = Q(x, y) psi_{t+1}(y) / psi_t(x)`` where ``psi`` solves the backward (Schroedinger) equation ``psi_t = Q psi_{t+1}``.
As the reference becomes deterministic, the bridge approaches the optimal transport plan between ``p0`` and ``pT``; this is how it is used to move a portfolio's weight distribution, or a simulated scenario
distribution, to a target with the least change to the dynamics.

**Linearly solvable control.** With state cost ``q(s)`` and the control cost the KL divergence from the passive dynamics ``p(s' | s)``, the Bellman equation becomes linear in the *desirability*
``z = exp(-v)``: ``z(s) = exp(-q(s)) sum_s' p(s' | s) z(s')`` (Kappen 2005; Todorov 2007). The optimal controlled dynamics tilt the passive ones by ``z``: ``u*(s' | s) = p(s' | s) z(s') / sum p z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StaticBridge:
    coupling: np.ndarray    # (n0, nT): the joint distribution of start and end
    a: np.ndarray
    b: np.ndarray
    iterations: int
    error: float            # largest gap between the coupling's marginals and the targets


def sinkhorn(p0: np.ndarray, pT: np.ndarray, kernel: np.ndarray, tol: float = 1e-12, max_iter: int = 100000) -> StaticBridge:
    """Find ``a, b > 0`` so that ``coupling = diag(a) K diag(b)`` has row sums ``p0`` and column sums ``pT``; this coupling minimises ``KL(coupling || p0 x pT-free reference K)`` among joint laws with those
    marginals (the static Schroedinger problem). ``kernel`` is the reference's start-to-end transition weights, entries > 0."""
    p0, pT, K = (np.asarray(v, dtype=float) for v in (p0, pT, kernel))
    if K.shape != (len(p0), len(pT)) or (K < 0).any() or not np.isclose(p0.sum(), pT.sum()):
        raise ValueError("kernel must be (len(p0), len(pT)), non-negative, and both marginals must carry the same mass")
    a, b = np.ones(len(p0)), np.ones(len(pT))
    err = np.inf
    for it in range(1, max_iter + 1):
        a = p0 / (K @ b)
        b = pT / (K.T @ a)
        pi = a[:, None] * K * b[None, :]
        err = float(max(np.abs(pi.sum(axis=1) - p0).max(), np.abs(pi.sum(axis=0) - pT).max()))
        if err < tol:
            break
    return StaticBridge(a[:, None] * K * b[None, :], a, b, it, err)


@dataclass
class DynamicBridge:
    transitions: list           # P_t, t = 0 ... T-1: the bridge's transition matrices
    marginals: np.ndarray       # (T+1, S): the law of the bridge at each time
    a: np.ndarray
    b: np.ndarray
    iterations: int


def markov_bridge(Q: np.ndarray, p0: np.ndarray, pT: np.ndarray, steps: int, tol: float = 1e-12, max_iter: int = 100000) -> DynamicBridge:
    """The Schroedinger bridge of a time-homogeneous Markov chain ``Q`` over ``steps`` steps: the chain closest in relative entropy to ``Q`` (as a law on paths) with start law ``p0`` and end law ``pT``."""
    Q = np.asarray(Q, dtype=float)
    S = Q.shape[0]
    if Q.shape != (S, S) or not np.allclose(Q.sum(axis=1), 1.0) or steps < 1:
        raise ValueError("Q must be a stochastic matrix and steps >= 1")
    K = np.linalg.matrix_power(Q, steps)                                                            # start-to-end kernel of the reference
    static = sinkhorn(p0, pT, K, tol, max_iter)
    psi = [None] * (steps + 1)
    psi[steps] = static.b                                                                           # psi_T = b, psi_t = Q psi_{t+1}
    for t in range(steps - 1, -1, -1):
        psi[t] = Q @ psi[t + 1]
    P = [Q * psi[t + 1][None, :] / psi[t][:, None] for t in range(steps)]
    marg = np.empty((steps + 1, S))
    marg[0] = np.asarray(p0, dtype=float)
    for t in range(steps):
        marg[t + 1] = marg[t] @ P[t]
    return DynamicBridge(P, marg, static.a, static.b, static.iterations)


def path_kl(P: list, Q: np.ndarray, p0: np.ndarray, ref0: np.ndarray | None = None) -> float:
    """``KL`` between the path laws of the Markov chain ``(p0, P_t)`` and the reference ``(ref0, Q)``, summed from the chain rule: initial term plus the expected one-step KL at each time."""
    p0 = np.asarray(p0, dtype=float)
    ref0 = p0 if ref0 is None else np.asarray(ref0, dtype=float)
    total = float((p0[p0 > 0] * np.log(p0[p0 > 0] / ref0[p0 > 0])).sum())
    marg = p0.copy()
    for Pt in P:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(Pt > 0, Pt * np.log(Pt / Q), 0.0).sum(axis=1)
        total += float(marg @ term)
        marg = marg @ Pt
    return total


# ------------------------------------------------------------------------------------------------------------------ linearly solvable MDPs
@dataclass
class LinearlySolvable:
    z: np.ndarray               # desirability exp(-v)
    value: np.ndarray           # v = -log z (cost to go per step, discounted by the first-exit or average-cost normalisation of the problem)
    policy: np.ndarray          # u*(s' | s)


def solve_lmdp_first_exit(passive: np.ndarray, state_cost: np.ndarray, terminal: np.ndarray, tol: float = 1e-14, max_iter: int = 100000) -> LinearlySolvable:
    """First-exit problem: ``terminal[s]`` is True for absorbing states, where the cost to go is ``state_cost[s]``. For the others ``z = exp(-q) P z`` is solved by iteration from ``z = exp(-q)``;
    the desirability of a terminal state is ``exp(-q)``."""
    P, q, term = np.asarray(passive, dtype=float), np.asarray(state_cost, dtype=float), np.asarray(terminal, dtype=bool)
    z = np.exp(-q)
    for _ in range(max_iter):
        z_new = np.where(term, np.exp(-q), np.exp(-q) * (P @ z))
        if np.abs(z_new - z).max() < tol:
            z = z_new
            break
        z = z_new
    u = P * z[None, :]
    u = u / u.sum(axis=1, keepdims=True)
    u[term] = P[term]
    return LinearlySolvable(z, -np.log(z), u)


def kl_control_cost(passive: np.ndarray, policy: np.ndarray, state_cost: np.ndarray, terminal: np.ndarray, tol: float = 1e-13, max_iter: int = 100000) -> np.ndarray:
    """The expected total cost ``q + KL(u || p)`` to termination of a given controlled chain ``policy`` (to compare policies with the optimum): ``v = q + KL + u v`` on the non-terminal states."""
    P, U, q, term = (np.asarray(v) for v in (passive, policy, state_cost, terminal))
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(U > 0, U * np.log(U / P), 0.0).sum(axis=1)
    v = np.where(term, q, 0.0).astype(float)
    for _ in range(max_iter):
        v_new = np.where(term, q, q + kl + U @ v)
        if np.abs(v_new - v).max() < tol:
            return v_new
        v = v_new
    return v
