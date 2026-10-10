"""Pontryagin's maximum principle: optimal control as a two-point boundary value problem.

For ``min  phi(x(T)) + int_0^T L(t, x, u) dt`` subject to ``x' = f(t, x, u)``, ``x(0) = x0``, define the Hamiltonian ``H = L + lambda . f``. An optimal control minimises ``H`` at every instant, and the *costate*
solves ``lambda' = -dH/dx`` with ``lambda(T) = d phi / dx (x(T))``. Substituting the minimiser ``u*(t, x, lambda)`` leaves ``x'`` and ``lambda'`` coupled, with ``x`` fixed at ``t = 0`` and ``lambda`` at ``T``:
a boundary value problem, solved here with ``scipy.integrate.solve_bvp`` (collocation).

The principle gives a *necessary* condition along one trajectory; the HJB equation gives the value function for all of them. For a linear-quadratic problem the two meet in the Riccati equation, which is the check used in
the tests, along with the Almgren-Chriss liquidation (``x(t) = X sinh(kappa (T - t)) / sinh(kappa T)``) and the Ramsey growth model, a nonlinear problem with a known steady state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp


@dataclass
class PMPSolution:
    t: np.ndarray
    x: np.ndarray          # (n_states, len(t))
    costate: np.ndarray    # (n_states, len(t))
    u: np.ndarray          # (n_controls, len(t))
    success: bool
    message: str


def solve_pmp(f: Callable, hamiltonian_x: Callable, u_star: Callable, x0, horizon: float, terminal_costate: Callable, n_nodes: int = 201, guess=None, tol: float = 1e-8,
              max_nodes: int = 50000) -> PMPSolution:
    """Solve the boundary value problem. ``f(t, x, u)`` is the dynamics, ``hamiltonian_x(t, x, u, lam)`` is ``dH/dx``, ``u_star(t, x, lam)`` the Hamiltonian's minimiser, ``terminal_costate(x_T)`` the transversality
    condition ``lambda(T) = dphi/dx``. Vectors are shaped ``(n_states, n_points)`` as in ``solve_bvp``. ``guess`` is an optional ``(x, lam)`` pair of arrays on the grid ``linspace(0, horizon, n_nodes)``."""
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x0)
    t = np.linspace(0.0, horizon, n_nodes)

    def rhs(tt, y):
        x, lam = y[:n], y[n:]
        u = u_star(tt, x, lam)
        return np.vstack([f(tt, x, u), -hamiltonian_x(tt, x, u, lam)])

    def bc(ya, yb):
        return np.concatenate([ya[:n] - x0, yb[n:] - np.atleast_1d(terminal_costate(yb[:n]))])

    if guess is None:
        y0 = np.zeros((2 * n, t.size))
        y0[:n] = x0[:, None]
    else:
        y0 = np.vstack(guess)
    sol = solve_bvp(rhs, bc, t, y0, tol=tol, max_nodes=max_nodes)
    tt = np.linspace(0.0, horizon, n_nodes)
    y = sol.sol(tt)
    u = np.atleast_2d(u_star(tt, y[:n], y[n:]))
    return PMPSolution(tt, y[:n], y[n:], u, bool(sol.success), sol.message)


# ------------------------------------------------------------------------------------------------------------------ the three reference problems
def lqr_pmp(a: float, b: float, q: float, r: float, s: float, x0: float, horizon: float, n_nodes: int = 201) -> PMPSolution:
    """Scalar linear-quadratic regulator ``min  s x(T)^2 + int (q x^2 + r u^2) dt``, ``x' = a x + b u``: ``u* = -b lambda / (2 r)``, ``lambda' = -(2 q x + a lambda)``, ``lambda(T) = 2 s x(T)``."""
    return solve_pmp(lambda t, x, u: a * x + b * u, lambda t, x, u, lam: 2 * q * x + a * lam, lambda t, x, lam: -b * lam / (2 * r), [x0], horizon, lambda xT: 2 * s * xT, n_nodes)


def lqr_riccati(a: float, b: float, q: float, r: float, s: float, x0: float, horizon: float, n_nodes: int = 201) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same problem by the Riccati equation ``P' = -(2 a P - b^2 P^2 / r + q)``, ``P(T) = s``, ``u = -b P x / r``. Returns ``(t, x, u)``."""
    tt = np.linspace(0.0, horizon, n_nodes)
    P = solve_ivp(lambda t, p: -(2 * a * p - b ** 2 * p ** 2 / r + q), [horizon, 0.0], [s], dense_output=True, rtol=1e-11, atol=1e-13).sol
    x = solve_ivp(lambda t, xx: a * xx - b ** 2 * P(t)[0] * xx / r, [0.0, horizon], [x0], t_eval=tt, rtol=1e-11, atol=1e-13).y[0]
    return tt, x, -b * P(tt)[0] * x / r


def almgren_chriss_pmp(shares: float, horizon: float, eta: float, risk_aversion: float, sigma: float, n_nodes: int = 201) -> PMPSolution:
    """Sell ``shares`` in ``horizon`` with temporary impact ``eta`` per unit of trading rate and risk aversion ``lambda`` against variance ``sigma^2 x^2`` of the shares still held: :func:`liquidation_pmp` with
    ``w = lambda sigma^2``."""
    return liquidation_pmp(shares, horizon, eta, risk_aversion * sigma ** 2, n_nodes)


def liquidation_pmp(shares: float, horizon: float, eta: float, risk_weight: float, n_nodes: int = 201) -> PMPSolution:
    """``min int (eta v^2 + w x^2) dt``, ``x' = -v``, ``x(0) = shares``, ``x(T) = 0``. Hamiltonian ``eta v^2 + w x^2 - lambda v``: ``v* = lambda / (2 eta)``, ``lambda' = -2 w x``."""
    t = np.linspace(0.0, horizon, n_nodes)

    def rhs(tt, y):
        x, lam = y[0], y[1]
        return np.vstack([-lam / (2 * eta), -2 * risk_weight * x])

    def bc(ya, yb):
        return np.array([ya[0] - shares, yb[0]])

    y0 = np.vstack([shares * (1 - t / horizon), np.zeros_like(t)])
    sol = solve_bvp(rhs, bc, t, y0, tol=1e-10, max_nodes=50000)
    y = sol.sol(t)
    return PMPSolution(t, y[:1], y[1:], y[1:] / (2 * eta), bool(sol.success), sol.message)


def almgren_chriss_closed_form(shares: float, horizon: float, eta: float, risk_weight: float, t: np.ndarray) -> np.ndarray:
    """``x(t) = X sinh(kappa (T - t)) / sinh(kappa T)`` with ``kappa = sqrt(w / eta)``."""
    kappa = np.sqrt(risk_weight / eta)
    return shares * np.sinh(kappa * (horizon - t)) / np.sinh(kappa * horizon)


def ramsey_pmp(k0: float, horizon: float, alpha: float = 0.33, A: float = 1.0, delta: float = 0.05, rho: float = 0.03, theta: float = 2.0, terminal_k: float | None = None,
               n_nodes: int = 401) -> PMPSolution:
    """The Ramsey-Cass-Koopmans model: ``max int e^{-rho t} c^(1-theta)/(1-theta) dt`` with ``k' = A k^alpha - delta k - c``. The current-value costate ``m`` gives ``c = m^{-1/theta}`` and
    ``m' = (rho + delta - alpha A k^(alpha-1)) m``. The end state is pinned at ``terminal_k`` (default the steady state), the usual turnpike boundary condition for a long horizon."""
    kstar = (alpha * A / (rho + delta)) ** (1.0 / (1.0 - alpha))
    kT = kstar if terminal_k is None else terminal_k
    t = np.linspace(0.0, horizon, n_nodes)

    def rhs(tt, y):
        k, m = y[0], y[1]
        c = np.power(np.maximum(m, 1e-12), -1.0 / theta)
        return np.vstack([A * np.maximum(k, 1e-9) ** alpha - delta * k - c, (rho + delta - alpha * A * np.maximum(k, 1e-9) ** (alpha - 1.0)) * m])

    def bc(ya, yb):
        return np.array([ya[0] - k0, yb[0] - kT])

    cstar = A * kstar ** alpha - delta * kstar
    y0 = np.vstack([k0 + (kT - k0) * t / horizon, np.full_like(t, cstar ** (-theta))])
    sol = solve_bvp(rhs, bc, t, y0, tol=1e-8, max_nodes=50000)
    y = sol.sol(t)
    c = np.power(y[1], -1.0 / theta)
    return PMPSolution(t, y[:1], y[1:], c[None, :], bool(sol.success), sol.message)


def ramsey_steady_state(alpha: float = 0.33, A: float = 1.0, delta: float = 0.05, rho: float = 0.03, theta: float = 2.0) -> tuple[float, float]:
    """The capital stock and consumption at which ``k' = c' = 0``: ``alpha A k^(alpha-1) = rho + delta``."""
    kstar = (alpha * A / (rho + delta)) ** (1.0 / (1.0 - alpha))
    return kstar, A * kstar ** alpha - delta * kstar
