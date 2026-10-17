"""Option prices by finite differences: the Black-Scholes equation on a grid of log prices.

With ``x = ln S`` and ``tau = T - t`` the value of an option solves ``V_tau = (sigma^2/2) V_xx + (r - q - sigma^2/2) V_x - r V``: a constant-coefficient diffusion, so the spatial
operator is the same tridiagonal matrix at every step and is factored once. The scheme is the theta method: ``theta = 1`` is implicit Euler (first order in time, never oscillates),
``theta = 1/2`` is Crank-Nicolson (second order, but it rings on a payoff with a kink or a jump). The ringing is removed by starting with a few implicit half-steps (Rannacher 1984),
and the initial condition is the *average of the payoff over each cell* rather than its value at the node (Pooley, Vetzal and Forsyth 2003), which keeps the second order for a kink
(call, put) and for a jump (digital). The result is second order in both ``dx`` and ``dt``, which ``convergence`` measures.

What the grid gives that a closed form does not:

``vanilla``    a European or American call or put. The American problem is a linear complementarity problem at every step (``V >= payoff``, ``V_tau >= L V``, one of them an equality),
               solved by Brennan and Schwartz's direct sweep or by projected SOR (``numerics.linsolve``); the early-exercise boundary comes with the price.
``barrier``    a knock-out option: the grid ends at the barrier, where the value is zero. A knock-in is the vanilla minus the knock-out.
``digital``    cash-or-nothing and asset-or-nothing.

Delta, gamma and theta come from the same grid at no cost (the cubic spline through the final values), which is also how the early-exercise boundary is read off.
Central differences keep the second order but the matrix stops being an M-matrix when the cell Peclet number ``|r - q - sigma^2/2| dx / sigma^2`` exceeds 1; the grid is refined until
it does not (a very small volatility with a large rate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from ..numerics import linsolve

_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(8)


@dataclass
class PDEResult:
    """A priced option and what the grid knows besides: the value, delta, gamma and theta at the spot, the final values on the grid and (American) the early-exercise boundary in time."""

    price: float
    delta: float
    gamma: float
    theta: float
    S: np.ndarray
    values: np.ndarray
    n_space: int
    n_time: int
    seconds: float
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))
    boundary: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extras: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"price": self.price, "delta": self.delta, "gamma": self.gamma, "theta": self.theta, "n_space": self.n_space, "n_time": self.n_time, "seconds": self.seconds, **self.extras}


# ------------------------------------------------------------------------------------------------------------------------------------------ grid
def _cell_average(payoff, edges: np.ndarray, kinks: tuple[float, ...]) -> np.ndarray:
    """The average of ``payoff(S)`` over each cell ``[edges[i], edges[i+1]]`` in ``x = ln S``, splitting a cell at a kink or jump (a log strike) inside it so that Gauss-Legendre is exact
    on each smooth piece."""
    a, b = edges[:-1], edges[1:]
    out = np.empty(len(a))
    smooth = np.ones(len(a), dtype=bool)
    for k in kinks:
        smooth &= ~((a < k) & (k < b))
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    pts = mid[:, None] + half[:, None] * _GAUSS_X[None, :]
    out[smooth] = (payoff(np.exp(pts[smooth])) * _GAUSS_W[None, :]).sum(axis=1) / 2.0
    for i in np.flatnonzero(~smooth):
        cuts = [a[i]] + sorted(k for k in kinks if a[i] < k < b[i]) + [b[i]]
        total = 0.0
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * _GAUSS_X
            total += 0.5 * (hi - lo) * float((payoff(np.exp(x)) * _GAUSS_W).sum())
        out[i] = total / (b[i] - a[i])
    return out


def _refine_for_peclet(sigma: float, mu: float, length: float, n_space: int, max_space: int = 40000) -> int:
    if mu == 0.0:
        return n_space
    need = int(np.ceil(length * abs(mu) / sigma ** 2 * 1.05))
    return int(min(max(n_space, need), max_space))


def _aligned_grid(x0: float, x_lo: float, x_hi: float, n_space: int, anchor_lo: bool = False, anchor_hi: bool = False) -> np.ndarray:
    """A uniform grid on ``[x_lo, x_hi]`` of about ``n_space`` cells. By default ``x0`` is a node (so the price at the spot needs no interpolation); with ``anchor_lo`` / ``anchor_hi`` the
    named end is kept exactly (a barrier sits on the edge of the grid)."""
    dx = (x_hi - x_lo) / n_space
    if anchor_lo:
        return x_lo + dx * np.arange(n_space + 1)
    if anchor_hi:
        return x_hi - dx * np.arange(n_space, -1, -1)
    n_lo, n_hi = int(np.ceil((x0 - x_lo) / dx)), int(np.ceil((x_hi - x0) / dx))
    return x0 + dx * np.arange(-n_lo, n_hi + 1)


def _theta_scheme(sigma, r, q, T, x, v0, bc_lo, bc_hi, n_time, theta, rannacher, obstacle=None, exercise="low", lcp="bs", record_boundary=False):
    """March ``v0`` (the cell-averaged payoff at the nodes ``x``) to ``tau = T``. ``bc_lo(tau)`` and ``bc_hi(tau)`` are the Dirichlet values at the two ends; ``obstacle`` (values at the nodes)
    turns each step into a linear complementarity problem. Returns the final values and, if asked, the boundary and the times."""
    dx = x[1] - x[0]
    n = len(x) - 2
    mu = r - q - 0.5 * sigma ** 2
    a, b = 0.5 * sigma ** 2 / dx ** 2, 0.5 * mu / dx
    c_lo, c_di, c_up = a - b, -2.0 * a - r, a + b
    dt = T / n_time
    v = v0.copy()
    v[0], v[-1] = bc_lo(0.0), bc_hi(0.0)                  # the boundary values at tau = 0 are the payoff's far limits
    cache: dict[tuple[float, float], tuple] = {}

    def operator(step: float, th: float):
        key = (round(step, 15), th)
        if key not in cache:
            lo, di, up = np.full(n - 1, -th * step * c_lo), np.full(n, 1.0 - th * step * c_di), np.full(n - 1, -th * step * c_up)
            cache[key] = (lo, di, up, linsolve.tridiagonal_factor(lo, di, up) if obstacle is None else None)
        return cache[key]

    boundary = np.full(n_time + 1, np.nan)
    tau = np.linspace(0.0, T, n_time + 1)

    def advance(v, tau0, step, th):
        lo, di, up, fact = operator(step, th)
        tau1 = tau0 + step
        explicit = v[1:-1] + (1.0 - th) * step * (c_lo * v[:-2] + c_di * v[1:-1] + c_up * v[2:])         # the old boundary values enter here through v[0] and v[-1]
        explicit[0] += step * c_lo * th * bc_lo(tau1)                                                    # the new ones are known and move to the right-hand side
        explicit[-1] += step * c_up * th * bc_hi(tau1)
        new = v.copy()
        new[0], new[-1] = bc_lo(tau1), bc_hi(tau1)
        if obstacle is None:
            new[1:-1] = fact.solve(explicit)
        elif lcp == "bs":
            new[1:-1] = linsolve.brennan_schwartz(lo, di, up, explicit, obstacle[1:-1], exercise)
        else:
            new[1:-1] = linsolve.psor_tridiagonal(lo, di, up, explicit, obstacle[1:-1], omega=1.3, x0=np.maximum(v[1:-1], obstacle[1:-1]), tol=1e-11, max_iter=2000).x
        return new

    n_start = 2 if (rannacher and theta != 1.0 and n_time >= 4) else 0
    for k in range(n_time):
        if k < n_start:                                   # two Crank-Nicolson steps replaced by four implicit half-steps
            v = advance(v, tau[k], dt / 2.0, 1.0)
            v = advance(v, tau[k] + dt / 2.0, dt / 2.0, 1.0)
        else:
            v = advance(v, tau[k], dt, theta)
        if record_boundary and obstacle is not None:
            boundary[k + 1] = _free_boundary(x, v, obstacle, exercise)
    return v, boundary, tau


def _free_boundary(x: np.ndarray, v: np.ndarray, g: np.ndarray, exercise: str) -> float:
    """The price at which exercising stops being optimal: where the value leaves the payoff, scanning from the exercised end of the grid (the far end, where the payoff is zero, is not part
    of the exercise region). The value and the payoff touch smoothly, so ``V - g`` is quadratic in the distance to the boundary and its square root is linear: the zero of a line
    through the square roots at the first continuation nodes (against the price, in which the relation is exactly quadratic for a smooth pasting) locates the boundary between the nodes."""
    gap = v - g
    cont = gap > 1e-10 * max(1.0, float(np.max(np.abs(g))))
    if not cont.any():
        return float("nan")
    if exercise == "low":
        j = int(np.argmax(cont))                                   # the first continuation node
        if j == 0:
            return float("nan")
        nodes, last = np.arange(j, min(j + 3, len(x))), j - 1
    else:
        j = len(cont) - 1 - int(np.argmax(cont[::-1]))
        if j == len(cont) - 1:
            return float("nan")
        nodes, last = np.arange(j, max(j - 3, -1), -1), j + 1
    price = np.exp(x)
    if len(nodes) < 2 or np.any(gap[nodes] <= 0):
        return float(price[last])
    slope, icpt = np.polyfit(price[nodes], np.sqrt(gap[nodes]), 1)
    if slope == 0.0:
        return float(price[last])
    lo, hi = sorted((price[last], price[nodes[0]]))
    return float(min(max(-icpt / slope, lo), hi))


def _finish(x, v, S, sigma, r, q, n_space, n_time, t0, tau=None, boundary=None) -> PDEResult:
    spline = CubicSpline(x, v)
    x0 = np.log(S)
    price = float(spline(x0))
    vx, vxx = float(spline(x0, 1)), float(spline(x0, 2))
    delta, gamma = vx / S, (vxx - vx) / S ** 2
    theta = -(0.5 * sigma ** 2 * S ** 2 * gamma + (r - q) * S * delta - r * price)
    return PDEResult(price, delta, gamma, theta, np.exp(x), v, n_space, n_time, time.perf_counter() - t0, np.zeros(0) if tau is None else tau, np.zeros(0) if boundary is None else boundary)


def _domain(S, K, T, sigma, r, q, width):
    mu = r - q - 0.5 * sigma ** 2
    w = width * sigma * np.sqrt(T) + abs(mu) * T
    return min(np.log(S), np.log(K)) - w, max(np.log(S), np.log(K)) + w


def _check(S, K, T, sigma):
    if S <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("spot, strike and volatility must be positive")
    if T < 0:
        raise ValueError("the maturity must not be negative")


def _intrinsic_result(S, payoff) -> PDEResult:
    v = float(payoff(np.array([S]))[0])
    return PDEResult(v, 0.0, 0.0, 0.0, np.array([S]), np.array([v]), 0, 0, 0.0)


# ------------------------------------------------------------------------------------------------------------------------------------ vanilla
def vanilla(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True, american: bool = False, n_space: int = 800, n_time: int = 400, theta: float = 0.5,
            rannacher: bool = True, width: float = 5.5, lcp: str = "bs", boundary: bool = False) -> PDEResult:
    """A European or American call or put by the theta scheme (default Crank-Nicolson with Rannacher start). ``lcp`` picks the American solver: ``'bs'`` (Brennan-Schwartz, direct) or ``'psor'``
    (projected SOR). ``boundary=True`` records the early-exercise boundary at every time step."""
    _check(S, K, T, sigma)
    sign = 1.0 if call else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    if T == 0:
        return _intrinsic_result(S, payoff)
    t0 = time.perf_counter()
    x_lo, x_hi = _domain(S, K, T, sigma, r, q, width)
    n_space = _refine_for_peclet(sigma, r - q - 0.5 * sigma ** 2, x_hi - x_lo, n_space)
    x = _aligned_grid(np.log(S), x_lo, x_hi, n_space)
    edges = np.concatenate([[x[0] - 0.5 * (x[1] - x[0])], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])
    v0 = _cell_average(payoff, edges, (float(np.log(K)),))
    s_lo, s_hi = np.exp(x[0]), np.exp(x[-1])
    if american:
        # exercise region at the low prices for a put and (with a dividend yield) at the high prices for a call: the far boundary is the payoff itself
        if call:
            bc_lo, bc_hi = (lambda tau: 0.0), (lambda tau: s_hi - K)
        else:
            bc_lo, bc_hi = (lambda tau: K - s_lo), (lambda tau: 0.0)
        exercise = "high" if call else "low"
        obstacle = payoff(np.exp(x))
        v, bdry, tau = _theta_scheme(sigma, r, q, T, x, np.maximum(v0, obstacle), bc_lo, bc_hi, n_time, theta, rannacher, obstacle, exercise, lcp, boundary)
    else:
        if call:
            bc_lo, bc_hi = (lambda tau: 0.0), (lambda tau: s_hi * np.exp(-q * tau) - K * np.exp(-r * tau))
        else:
            bc_lo, bc_hi = (lambda tau: K * np.exp(-r * tau) - s_lo * np.exp(-q * tau)), (lambda tau: 0.0)
        v, bdry, tau = _theta_scheme(sigma, r, q, T, x, v0, bc_lo, bc_hi, n_time, theta, rannacher)
    return _finish(x, v, S, sigma, r, q, len(x) - 1, n_time, t0, tau, bdry if boundary else None)


# ------------------------------------------------------------------------------------------------------------------------------------ barrier
def barrier(S: float, K: float, H: float, T: float, r: float, q: float, sigma: float, call: bool = True, kind: str = "down-out", n_space: int = 800, n_time: int = 400,
            theta: float = 0.5, rannacher: bool = True, width: float = 5.5) -> PDEResult:
    """A knock-out call or put with the barrier ``H`` monitored continuously (``kind`` is ``'down-out'`` or ``'up-out'``), no rebate. The grid ends at the barrier, where the value is zero.
    A knock-in is the vanilla minus this."""
    _check(S, K, T, sigma)
    if kind not in ("down-out", "up-out"):
        raise ValueError("kind must be 'down-out' or 'up-out' (a knock-in is the vanilla minus the knock-out)")
    down = kind == "down-out"
    if (down and S <= H) or (not down and S >= H):
        return PDEResult(0.0, 0.0, 0.0, 0.0, np.array([S]), np.array([0.0]), 0, 0, 0.0)
    sign = 1.0 if call else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    t0 = time.perf_counter()
    x_lo, x_hi = _domain(S, K, T, sigma, r, q, width)
    xh = np.log(H)
    if down:
        x_lo = xh
        x_hi = max(x_hi, np.log(S) + 1e-9)
    else:
        x_hi = xh
        x_lo = min(x_lo, np.log(S) - 1e-9)
    n_space = _refine_for_peclet(sigma, r - q - 0.5 * sigma ** 2, x_hi - x_lo, n_space)
    x = _aligned_grid(np.log(S), x_lo, x_hi, n_space, anchor_lo=down, anchor_hi=not down)
    edges = np.concatenate([[x[0] - 0.5 * (x[1] - x[0])], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])
    v0 = _cell_average(payoff, edges, (float(np.log(K)),))
    s_lo, s_hi = np.exp(x[0]), np.exp(x[-1])
    zero = (lambda tau: 0.0)
    if call and down:                                     # far from the barrier the knock-out is worth the vanilla: a call far above is S - K, a put far below is K - S
        far = (lambda tau: s_hi * np.exp(-q * tau) - K * np.exp(-r * tau))
    elif not call and not down:
        far = (lambda tau: K * np.exp(-r * tau) - s_lo * np.exp(-q * tau))
    else:                                                 # a call far below the barrier, a put far above it: worthless
        far = zero
    bc_lo, bc_hi = (zero, far) if down else (far, zero)   # the barrier end is worth nothing: the option is dead
    v0 = v0.copy()
    v0[0 if down else -1] = 0.0
    v, _, _ = _theta_scheme(sigma, r, q, T, x, v0, bc_lo, bc_hi, n_time, theta, rannacher)
    return _finish(x, v, S, sigma, r, q, len(x) - 1, n_time, t0)


# ------------------------------------------------------------------------------------------------------------------------------------ digital
def digital(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True, kind: str = "cash", payout: float = 1.0, n_space: int = 800, n_time: int = 400,
            theta: float = 0.5, rannacher: bool = True, width: float = 5.5) -> PDEResult:
    """A digital option: ``kind='cash'`` pays ``payout`` if the option finishes in the money, ``'asset'`` pays the asset."""
    _check(S, K, T, sigma)
    if kind not in ("cash", "asset"):
        raise ValueError("kind must be 'cash' or 'asset'")
    itm = (lambda s: s > K) if call else (lambda s: s < K)

    def payoff(s):
        return np.where(itm(s), payout if kind == "cash" else s, 0.0)

    if T == 0:
        return _intrinsic_result(S, payoff)
    t0 = time.perf_counter()
    x_lo, x_hi = _domain(S, K, T, sigma, r, q, width)
    n_space = _refine_for_peclet(sigma, r - q - 0.5 * sigma ** 2, x_hi - x_lo, n_space)
    x = _aligned_grid(np.log(S), x_lo, x_hi, n_space)
    edges = np.concatenate([[x[0] - 0.5 * (x[1] - x[0])], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])
    v0 = _cell_average(payoff, edges, (float(np.log(K)),))
    s_lo, s_hi = np.exp(x[0]), np.exp(x[-1])
    zero = (lambda tau: 0.0)
    if kind == "cash":                                    # deep in the money the option is a discounted payout, deep out of the money nothing
        inside = (lambda tau: payout * np.exp(-r * tau))
    else:                                                 # an asset-or-nothing is the asset itself (less its dividends) deep in the money
        inside = (lambda tau: (s_hi if call else s_lo) * np.exp(-q * tau))
    bc_lo, bc_hi = (zero, inside) if call else (inside, zero)
    v, _, _ = _theta_scheme(sigma, r, q, T, x, v0, bc_lo, bc_hi, n_time, theta, rannacher)
    return _finish(x, v, S, sigma, r, q, len(x) - 1, n_time, t0)


# ------------------------------------------------------------------------------------------------------------------------------ convergence
def convergence(price_fn, reference: float, levels=(1, 2, 4, 8), base_space: int = 100, base_time: int = 50) -> dict:
    """The error of ``price_fn(n_space, n_time)`` against ``reference`` as the grid is refined together (both halved at each level) and the observed order of accuracy (the slope of
    log error against log spacing). Second order means the error falls by four when the spacing halves."""
    spaces = [base_space * m for m in levels]
    errors = [abs(float(price_fn(n, base_time * m)) - reference) for n, m in zip(spaces, levels)]
    h = 1.0 / np.array(levels, dtype=float)
    ok = np.array(errors) > 1e-14
    order = float(np.polyfit(np.log(h[ok]), np.log(np.array(errors)[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
    return {"levels": list(levels), "n_space": spaces, "errors": errors, "order": order}
