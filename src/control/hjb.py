"""Hamilton-Jacobi-Bellman equations by monotone finite differences and policy iteration, and the viscosity solution that makes them well posed.

A value function is rarely smooth: with a stopping decision (an American option) or a bound on the control it has kinks, and the HJB equation has no classical solution there. The *viscosity solution* (Crandall and
Lions 1983) is the one the right numerical scheme converges to: a scheme that is **monotone** (every neighbour enters with a non-negative weight), **stable** and **consistent** converges to it (Barles and
Souganidis 1991). The two schemes here are built that way.

* :func:`merton_hjb` solves ``V_t + max_pi [(r + pi(mu - r) - pi^2 sigma^2 / 2) V_x + pi^2 sigma^2 / 2 V_xx] = 0`` for ``x = log W`` with terminal power utility, by implicit time steps, an upwind difference for the
  drift and Howard's policy iteration at each step, and returns the value and the fraction of wealth in the risky asset to compare with Merton's closed form.
* :func:`american_put_hjb` solves the obstacle problem ``min(rV - V_tau + L V... )``: ``max(V_tau - L V, payoff - V) = 0`` for an American put by an implicit scheme with policy iteration on the stop-or-continue
  decision, to compare with a binomial tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

from .merton import merton_fraction, merton_value


@dataclass
class MertonHJB:
    x: np.ndarray           # log wealth grid
    value: np.ndarray       # V(0, x)
    fraction: np.ndarray    # optimal risky fraction at t = 0
    exact: np.ndarray       # Merton's V(0, x)
    iterations: int         # policy-iteration sweeps, summed over the time steps

    @property
    def max_relative_error(self) -> float:
        inner = slice(len(self.x) // 4, 3 * len(self.x) // 4)
        return float(np.max(np.abs(self.value[inner] / self.exact[inner] - 1.0)))


def merton_hjb(mu: float, r: float, sigma: float, gamma: float, horizon: float = 1.0, nx: int = 201, nt: int = 50, half_range: float = 3.0, pi_bounds: tuple = (0.0, 3.0), n_pi: int = 121) -> MertonHJB:
    """Terminal utility ``W^(1-gamma)/(1-gamma)`` (``gamma != 1``) of wealth ``horizon`` years ahead. The log wealth grid spans ``+-half_range`` around zero; the boundary condition is the homothety of power utility,
    ``V_x = (1 - gamma) V`` at both ends. The fraction is searched on ``n_pi`` values in ``pi_bounds``, so a bound that binds is respected (the scheme is the one for the constrained problem)."""
    if abs(gamma - 1.0) < 1e-12:
        raise ValueError("use gamma != 1 (the log case has the explicit solution V = log W + g tau)")
    x = np.linspace(-half_range, half_range, nx)
    dx, dt = x[1] - x[0], horizon / nt
    e = 1.0 - gamma
    V = np.exp(e * x) / e
    pis = np.linspace(pi_bounds[0], pi_bounds[1], n_pi)
    drift = r + pis * (mu - r) - 0.5 * pis ** 2 * sigma ** 2                                            # (n_pi,)
    diff = 0.5 * pis ** 2 * sigma ** 2
    ghost_up, ghost_dn = np.exp(e * dx), np.exp(-e * dx)                                                 # V at x +- dx beyond the ends is V * ghost
    sweeps = 0
    for _ in range(nt):
        prev, pol = V.copy(), np.full(nx, int(np.argmin(np.abs(pis - merton_fraction(mu, r, sigma, gamma)))))
        for it in range(50):
            sweeps += 1
            up, dn = np.empty(nx), np.empty(nx)
            up[:-1], up[-1] = V[1:], V[-1] * ghost_up
            dn[1:], dn[0] = V[:-1], V[0] * ghost_dn
            fwd, bwd = (up - V) / dx, (V - dn) / dx
            vxx = (up - 2.0 * V + dn) / dx ** 2
            vx = np.where(drift[None, :] > 0, fwd[:, None], bwd[:, None])
            H = drift[None, :] * vx + diff[None, :] * vxx[:, None]
            new = H.argmax(axis=1)
            if it and (new == pol).all():
                break
            pol = new
            d, s = drift[pol], diff[pol]
            pos = np.maximum(d, 0.0) / dx + s / dx ** 2                                               # weight on V(x + dx)
            neg = np.maximum(-d, 0.0) / dx + s / dx ** 2                                              # weight on V(x - dx)
            centre = -(np.maximum(d, 0.0) / dx + np.maximum(-d, 0.0) / dx + 2.0 * s / dx ** 2)
            lower, upper, mid = neg[1:].copy(), pos[:-1].copy(), centre.copy()
            mid[-1] += pos[-1] * ghost_up
            mid[0] += neg[0] * ghost_dn
            L = diags([lower, mid, upper], [-1, 0, 1], format="csc")
            V = spsolve((identity(nx, format="csc") - dt * L).tocsc(), prev)
    exact = merton_value(np.exp(x), horizon, mu, r, sigma, gamma)
    up, dn = np.empty(nx), np.empty(nx)
    up[:-1], up[-1] = V[1:], V[-1] * ghost_up
    dn[1:], dn[0] = V[:-1], V[0] * ghost_dn
    vx = np.where(drift[None, :] > 0, ((up - V) / dx)[:, None], ((V - dn) / dx)[:, None])
    H = drift[None, :] * vx + diff[None, :] * ((up - 2.0 * V + dn) / dx ** 2)[:, None]
    return MertonHJB(x, V, pis[H.argmax(axis=1)], exact, sweeps)


@dataclass
class ObstacleSolution:
    spot: np.ndarray
    value: np.ndarray
    payoff: np.ndarray
    exercise_boundary: float        # the highest spot at which stopping is optimal at t = 0
    iterations: int


def american_put_hjb(S0: float, K: float, T: float, r: float, q: float, sigma: float, nx: int = 400, nt: int = 400, s_max_factor: float = 4.0) -> tuple[float, ObstacleSolution]:
    """The price of an American put from the obstacle problem ``max(V_tau - L V, payoff - V) = 0`` (``tau`` is time to expiry, ``L`` the Black-Scholes operator ``sigma^2 S^2/2 V_SS + (r - q) S V_S - r V``).

    Implicit Euler in ``tau``, an upwind difference for the drift, and policy iteration (Forsyth and Vetzal 2002) on the exercise decision at every step: the policy is 'stop' where the payoff beats the continuation
    value, the linear system is solved with those nodes pinned to the payoff, and the loop repeats until the stopping set no longer changes. Returns ``(price at S0, the full solution)``."""
    S = np.linspace(0.0, s_max_factor * max(K, S0), nx + 1)
    dS, dt = S[1] - S[0], T / nt
    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()
    mu_ = (r - q) * S
    d2 = 0.5 * sigma ** 2 * S ** 2
    pos = np.maximum(mu_, 0.0) / dS + d2 / dS ** 2
    neg = np.maximum(-mu_, 0.0) / dS + d2 / dS ** 2
    centre = -(np.maximum(mu_, 0.0) / dS + np.maximum(-mu_, 0.0) / dS + 2.0 * d2 / dS ** 2) - r
    sweeps = 0
    n = len(S)
    L = diags([neg[1:], centre, pos[:-1]], [-1, 0, 1], format="lil")
    L[0, :], L[-1, :] = 0.0, 0.0                                                                       # boundaries handled below: V(0) = K, V(S_max) = 0 for a put
    A = (identity(n, format="lil") - dt * L).tocsr()
    A = A.tolil()
    A[0, :], A[-1, :] = 0.0, 0.0
    A[0, 0], A[-1, -1] = 1.0, 1.0
    A = A.tocsr()
    for _ in range(nt):
        rhs = V.copy()
        rhs[0], rhs[-1] = K, 0.0
        W = V.copy()
        stop = np.zeros(n, dtype=bool)
        for it in range(100):
            sweeps += 1
            # discrete problem: min(A V - rhs, V - payoff) = 0; the policy is the constraint with the smaller residual at the current iterate
            new_stop = (W - payoff) < (A @ W - rhs)
            new_stop[0] = new_stop[-1] = False
            if it and (new_stop == stop).all():
                break
            stop = new_stop
            M = A.tolil()
            b = rhs.copy()
            for i in np.flatnonzero(stop):
                M[i, :] = 0.0
                M[i, i] = 1.0
                b[i] = payoff[i]
            W = spsolve(M.tocsc(), b)
        V = np.maximum(W, payoff)
    price = float(np.interp(S0, S, V))
    stopped = np.flatnonzero(V <= payoff + 1e-9)
    stopped = stopped[payoff[stopped] > 0]
    boundary = float(S[stopped.max()]) if stopped.size else 0.0
    return price, ObstacleSolution(S, V, payoff, boundary, sweeps)
