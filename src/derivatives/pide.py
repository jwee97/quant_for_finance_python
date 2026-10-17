"""Option prices under a jump model by a partial integro-differential equation (PIDE).

For the exponential Levy models of ``derivatives.levy`` the value ``V(tau, x)`` of an option on ``S = e^x`` solves

    V_tau = (sigma^2 / 2) V_xx + b V_x - r V + Int [V(x + z) - V(x) - z V_x] nu(dz)

with ``nu`` the Levy measure of the jumps. The integral is non-local: the value at one price depends on the values at every other. The scheme follows Cont and Voltchkova (2005):

* the jumps smaller than ``eps`` (half a cell) are replaced by extra diffusion of variance ``sigma_eps^2 = Int_{|z| < eps} z^2 nu(dz)`` (Asmussen and Rosinski 2001), which is what makes an
  infinite-activity model such as the variance gamma computable: its Levy measure is not integrable at zero, and the small jumps are the part that cannot be summed;
* the jumps larger than ``eps`` are summed on the grid, ``Sum_j nu_j V(x + j dx)`` with ``nu_j`` the measure of the cell around ``j dx`` (a correlation with a fixed kernel). Putting a
  cell's mass at its centre overstates the variance of the jumps (by ``nu_j dx^2 / 12`` per cell, which adds up to ``O(dx)`` when the measure is like ``1/z^2``): the diffusion absorbs
  the difference, so that the discrete jumps and the Brownian part have exactly the variance of the model's Levy measure, which restores second order for the NIG;
* the diffusion, drift and discounting are treated implicitly (a tridiagonal solve, factored once) and the jump sum explicitly, in a Crank-Nicolson / Adams-Bashforth step: second order in
  time and unconditionally stable for the implicit part. The first steps are implicit Euler (Rannacher), so that a kink in the payoff does not ring.

The drift ``b`` is not taken from the model's formulas but *solved from the martingale condition on the grid itself*: it is the number that makes the discrete operator give
``A e^x = (r - q) e^x`` exactly, so the discounted price is a martingale for the scheme however coarse the grid, whatever the quadrature error in ``nu_j``. Prices far outside the grid
(which the jump sum reaches) are the option's asymptotic value: ``S e^{-q tau} - K e^{-r tau}`` for a call far above the strike, zero far below, and the reverse for a put.

Compared with the Fourier price of ``derivatives.fourier`` (the benchmark: it is exact to 1e-10), this solver is slower and less precise (1e-4 relative at the default grid). Its value is that
it prices what Fourier methods cannot: it is the same machinery that handles early exercise and barriers, and it prices payoffs with no characteristic-function representation.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline

from ..numerics import linsolve
from . import levy
from .pde import PDEResult, _aligned_grid, _cell_average, _check

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _cell_measure(model: levy.Model, a: np.ndarray, b: np.ndarray, power: int = 0) -> np.ndarray:
    """``Int z^power nu(dz)`` over each interval ``[a_i, b_i]`` (none containing zero) by 16-point Gauss-Legendre."""
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    z = mid[:, None] + half[:, None] * _GL_X[None, :]
    return half * (z ** power * model.levy_density(z) * _GL_W[None, :]).sum(axis=1)


def jump_kernel(model: levy.Model, dx: float, max_jump: float = 6.0, tail_tol: float = 1e-12, small_cells: int = 0, match_variance: bool = True) -> dict:
    """The discretised jump measure on a grid of spacing ``dx``: the offsets ``j`` (in cells) with ``|j| > small_cells`` and the measure ``nu_j`` of each cell ``[(j - 1/2) dx, (j + 1/2) dx]``,
    the variance ``sigma_eps^2`` of the jumps inside the central band ``|z| < (small_cells + 1/2) dx`` (which the scheme treats as diffusion), the (negative) ``variance_defect`` that
    corrects the midpoint rule's overstatement of the variance of the cells outside it (zero unless ``match_variance``), and the weighted measure left out beyond ``max_jump``. The kernel
    is cut where the cell measure times ``1 + e^z`` falls below ``tail_tol`` (or at ``max_jump``)."""
    m = int(small_cells)
    j_cap = int(np.ceil(max_jump / dx))
    j = np.arange(m + 1, j_cap + 1)
    pos = _cell_measure(model, (j - 0.5) * dx, (j + 0.5) * dx)
    neg = _cell_measure(model, -(j + 0.5) * dx, -(j - 0.5) * dx)
    weight_pos, weight_neg = pos * (1.0 + np.exp(j * dx)), neg * (1.0 + np.exp(-j * dx))
    live = np.flatnonzero((weight_pos > tail_tol) | (weight_neg > tail_tol))
    n_cells = max(min(int(live.max()) + 1 if len(live) else 0, len(j)), 0)           # cells kept: j = m+1 .. m+n_cells
    # the variance of the jumps inside the central band: z^2 nu(z) is smooth on each side of zero, so integrate cell by cell
    edges = np.arange(0, m + 2) * dx
    edges[1:] -= 0.5 * dx
    central = float(_cell_measure(model, edges[:-1], edges[1:], 2).sum() + _cell_measure(model, -edges[1:], -edges[:-1], 2).sum())
    offsets = np.concatenate([-np.arange(m + n_cells, m, -1), np.arange(m + 1, m + n_cells + 1)]) if n_cells else np.zeros(0, dtype=int)
    nu = np.concatenate([neg[:n_cells][::-1], pos[:n_cells]]) if n_cells else np.zeros(0)
    tail = float(weight_pos[n_cells:].sum() + weight_neg[n_cells:].sum()) if n_cells < len(j) else 0.0
    # the midpoint rule puts the whole mass of a cell at its centre and so overstates the variance of the jumps (by about nu_j dx^2 / 12 per cell); the diffusion can absorb the difference,
    # so that the discrete measure and the Brownian part together have exactly the variance of the model's Levy measure
    defect = 0.0
    if n_cells and match_variance:
        jj = j[:n_cells]
        true_m2 = _cell_measure(model, (jj - 0.5) * dx, (jj + 0.5) * dx, 2).sum() + _cell_measure(model, -(jj + 0.5) * dx, -(jj - 0.5) * dx, 2).sum()
        defect = float(true_m2 - (nu * (offsets * dx) ** 2).sum())
    return {"offsets": offsets, "nu": nu, "sigma_eps2": central, "variance_defect": defect, "j_max": int(np.abs(offsets).max()) if n_cells else 0, "small_cells": m, "neglected": tail}


def choose_small_cells(model: levy.Model, dx: float, dt: float, stiffness: float = 0.1, max_cells: int = 64) -> int:
    """The smallest half-width (in cells) of the band of jumps treated as diffusion such that the explicit jump term is not stiff: ``dt * nu(|z| > (m + 1/2) dx) <= stiffness``. The
    approximation error of replacing the band by a Gaussian is cubic in its width, so a wide band costs little; a narrow one makes the explicit term stiff (the intensity of small jumps
    of an infinite-activity model grows like ``1/dx``) and the Adams-Bashforth step inaccurate."""
    for m in range(0, max_cells + 1):
        probe = jump_kernel(model, dx, max_jump=2.0, small_cells=m, match_variance=False)
        if float(probe["nu"].sum()) * dt <= stiffness:
            return m
    return max_cells


def _asymptotics(call: bool, K: float, r: float, q: float):
    """The option's value far from the strike as functions of ``(tau, x)``: the two ends of the log-price axis."""
    if call:
        lo = (lambda tau, x: np.zeros_like(x))
        hi = (lambda tau, x: np.exp(x) * np.exp(-q * tau) - K * np.exp(-r * tau))
    else:
        lo = (lambda tau, x: K * np.exp(-r * tau) - np.exp(x) * np.exp(-q * tau))
        hi = (lambda tau, x: np.zeros_like(x))
    return lo, hi


def european(model: levy.Model, S: float, K: float, T: float, r: float, q: float, call: bool = True, n_space: int = 1000, n_time: int = 200, width: float = 6.0,
             max_jump: float = 6.0, imex: str = "cnab", small_cells: int | None = None, match_variance: bool = True, stiffness: float = 0.03) -> PDEResult:
    """A European call or put under a Levy ``model`` by the IMEX scheme described in the module docstring (``imex='cnab'`` second order in time, ``'euler'`` first order).
    ``extras['neglected']`` of the result is the jump measure (weighted by ``1 + e^z``) beyond ``max_jump``, which should be far below the accuracy wanted."""
    if not model.is_levy:
        raise ValueError("a PIDE needs a Levy model; the Heston model has a second state variable")
    if imex not in ("cnab", "euler"):
        raise ValueError("imex must be 'cnab' or 'euler'")
    sigma_d = model.diffusion()
    _check(S, K, T, max(sigma_d, 1e-8) if sigma_d > 0 else 1.0)
    sign = 1.0 if call else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    if T == 0:
        v = float(payoff(np.array([S]))[0])
        return PDEResult(v, 0.0, 0.0, 0.0, np.array([S]), np.array([v]), 0, 0, 0.0)
    t0 = time.perf_counter()
    sd = float(np.sqrt(model.cumulants()[1] * T))
    drift = abs((r - q - model.omega()) * T)
    x_lo = min(np.log(S), np.log(K)) - width * sd - drift
    x_hi = max(np.log(S), np.log(K)) + width * sd + drift
    x = _aligned_grid(np.log(S), x_lo, x_hi, n_space)
    dx = x[1] - x[0]
    n = len(x) - 2
    edges = np.concatenate([[x[0] - 0.5 * dx], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * dx]])
    v = _cell_average(payoff, edges, (float(np.log(K)),))
    ext_lo, ext_hi = _asymptotics(call, K, r, q)
    lo_val, hi_val = (lambda tau: float(ext_lo(tau, x[:1])[0])), (lambda tau: float(ext_hi(tau, x[-1:])[0]))
    v[0], v[-1] = lo_val(0.0), hi_val(0.0)

    dt = T / n_time
    m_small = choose_small_cells(model, dx, dt, stiffness) if small_cells is None else int(small_cells)
    kern = jump_kernel(model, dx, max_jump, small_cells=m_small, match_variance=match_variance)
    jmax, nu = kern["j_max"], kern["nu"]
    lam = float(nu.sum())
    sig2 = max(sigma_d ** 2 + kern["sigma_eps2"] + kern["variance_defect"], 0.0)
    g2, g1 = 2.0 * (np.cosh(dx) - 1.0) / dx ** 2, np.sinh(dx) / dx
    # the drift that makes exp(x) a martingale for the discrete operator: 1/2 sig2 g2 + b g1 + Sum nu_j (e^{z_j} - 1) = r - q
    sum_exp = float((nu * (np.exp(kern["offsets"] * dx) - 1.0)).sum()) if jmax else 0.0
    b = ((r - q) - 0.5 * sig2 * g2 - sum_exp) / g1
    full_kernel = np.zeros(2 * jmax + 1)                      # the weight of each offset -jmax .. jmax (zero in the band that is treated as diffusion)
    if jmax:
        full_kernel[kern["offsets"] + jmax] = nu
    a_c, b_c = 0.5 * sig2 / dx ** 2, 0.5 * b / dx
    c_lo, c_di, c_up = a_c - b_c, -2.0 * a_c - r - lam, a_c + b_c
    cache: dict[tuple[float, float], object] = {}

    def factor(step: float, th: float):
        key = (round(step, 15), th)
        if key not in cache:
            cache[key] = linsolve.tridiagonal_factor(np.full(n - 1, -th * step * c_lo), np.full(n, 1.0 - th * step * c_di), np.full(n - 1, -th * step * c_up))
        return cache[key]

    def jumps(vals: np.ndarray, tau: float) -> np.ndarray:
        """``Sum_j nu_j V(x_i + j dx)`` at the interior nodes, the values beyond the grid being the asymptotic ones."""
        if not jmax:
            return np.zeros(n)
        left = ext_lo(tau, x[0] - dx * np.arange(jmax, 0, -1))
        right = ext_hi(tau, x[-1] + dx * np.arange(1, jmax + 1))
        padded = np.concatenate([left, vals, right])
        return np.correlate(padded, full_kernel, mode="valid")[1:-1]

    def step(vals, tau0, h, th, prev_jump=None):
        """One step from ``tau0`` by ``h`` with the implicit weight ``th`` for the local operator; ``prev_jump`` (the jump term one step back) selects Adams-Bashforth."""
        tau1 = tau0 + h
        jn = jumps(vals, tau0)
        explicit_jump = jn if prev_jump is None else 1.5 * jn - 0.5 * prev_jump
        rhs = vals[1:-1] + (1.0 - th) * h * (c_lo * vals[:-2] + c_di * vals[1:-1] + c_up * vals[2:]) + h * explicit_jump
        rhs[0] += h * c_lo * th * lo_val(tau1)
        rhs[-1] += h * c_up * th * hi_val(tau1)
        new = vals.copy()
        new[0], new[-1] = lo_val(tau1), hi_val(tau1)
        new[1:-1] = factor(h, th).solve(rhs)
        return new, jn

    tau = np.linspace(0.0, T, n_time + 1)
    prev = None
    for k in range(n_time):
        if k < 2 or imex == "euler":                       # implicit Euler for the first two steps (in four half steps) removes the oscillation a kink would start
            if imex == "euler":
                v, prev = step(v, tau[k], dt, 1.0)
            else:
                v, _ = step(v, tau[k], dt / 2.0, 1.0)
                v, prev = step(v, tau[k] + dt / 2.0, dt / 2.0, 1.0)
        else:
            v, jn = step(v, tau[k], dt, 0.5, prev)
            prev = jn
    spline = CubicSpline(x, v)
    x0 = np.log(S)
    price = float(spline(x0))
    vx, vxx = float(spline(x0, 1)), float(spline(x0, 2))
    delta, gamma = vx / S, (vxx - vx) / S ** 2
    # theta = -V_tau = -(A V - r V) at the spot, from the discrete operator on the final values
    i0 = int(np.argmin(np.abs(x - x0)))
    i0 = min(max(i0, 1), n)
    av = c_lo * v[i0 - 1] + (c_di + r + lam) * v[i0] + c_up * v[i0 + 1] - lam * v[i0] + float(jumps(v, T)[i0 - 1])
    theta = -(av - r * v[i0])
    res = PDEResult(price, delta, gamma, theta, np.exp(x), v, len(x) - 1, n_time, time.perf_counter() - t0)
    res.extras.update(neglected=float(kern["neglected"]), kernel_cells=int(jmax), small_cells=int(m_small), drift=float(b), small_jump_variance=float(kern["sigma_eps2"]), variance_defect=float(kern["variance_defect"]))
    return res
