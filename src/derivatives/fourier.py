"""Pricing European options from a characteristic function.

For a model with no closed-form price the characteristic function of the log return is still known (``derivatives.levy``), and the price of a European option is an integral of it. Two
standard ways to do the integral:

``lewis_call``    Lewis (2001): ``C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi * Int_0^inf Re[e^{iuk} phi_c(u - i/2)] / (u^2 + 1/4) du`` with ``k = ln(F/K)`` and ``phi_c`` the
                  characteristic function of ``ln(S_T/F)``. One integral for the price itself (no damping parameter, no put/call split). The integrand is even in ``u`` and analytic in
                  the strip ``|Im u| < 1/2`` (the poles of ``1/(u^2 + 1/4)``), so the plain trapezoid rule converges geometrically, with an error of order ``exp(-pi / h)`` for a
                  step ``h``: a step of 0.1 gives 1e-13 and no higher-order rule does better (Simpson's rule, which weighs the points unevenly, loses exactly this property and is only
                  accurate to ``exp(-pi / 2h)``). It is the workhorse: ``O(N)`` characteristic-function evaluations for all the strikes of an expiry.
``carr_madan``    Carr and Madan (1999): damp the call with ``e^{alpha k}`` so that it has a Fourier transform, and invert that transform on a whole grid of log-strikes at once with the
                  FFT, ``O(N log N)``. The damping ``alpha`` must be positive and ``alpha + 1`` below the model's exponent limit (``Model.exponent_limit``), which is chosen automatically.
                  The prices at the strikes asked for are interpolated from the grid (cubic spline), which sets the accuracy (about 1e-8).

Both are checked against the Black-Scholes closed form, against each other and against Monte Carlo in the tests. ``implied_vols`` turns the prices into the model's smile, and
``risk_neutral_density`` inverts the characteristic function into the density of the price at expiry, to show what a fitted model believes.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, interpolate

from .iv import implied_vol


def _trapezoid_weights(n: int) -> np.ndarray:
    """Trapezoid weights ``(1/2, 1, 1, ..., 1, 1/2)`` for ``n`` intervals."""
    w = np.ones(n + 1)
    w[0] = w[-1] = 0.5
    return w


def default_upper(model, T: float, r: float = 0.0, q: float = 0.0, tol: float = 1e-14, start: float = 32.0, limit: float = 2e4) -> float:
    """An upper limit for the Lewis integral: the first ``u`` (found by doubling from ``start``, then bisection) at which the integrand's envelope ``|phi_c(u - i/2)| / (u^2 + 1/4)``
    is below ``tol``, or ``limit`` when it never is (the characteristic function of an infinite-activity model over a short horizon decays like a power of ``u``)."""

    def env(u: float) -> float:
        with np.errstate(all="ignore"):
            v = abs(np.asarray(model.cf_centered(np.array([u - 0.5j]), T, r, q))[0]) / (u ** 2 + 0.25)
        return v if np.isfinite(v) else 0.0

    hi = start
    while env(hi) >= tol:
        if hi >= limit:
            return limit
        hi = min(2.0 * hi, limit)
    lo = hi / 2.0 if hi > start else 0.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if env(mid) < tol:
            hi = mid
        else:
            lo = mid
    return float(hi)


def lewis_call(model, S: float, K, T: float, r: float = 0.0, q: float = 0.0, step: float = 0.1, upper: float | None = None, method: str = "trapezoid", max_upper: float = 2e4) -> np.ndarray:
    """European call prices at the strikes ``K`` (vector) by the Lewis (2001) formula, with the trapezoid rule at spacing ``step`` on ``[0, upper]`` (``upper`` defaults to ``default_upper``,
    capped at ``max_upper``: the integrand of an infinite-activity model over a few days decays so slowly that the cap, not the step, then limits the accuracy, to about 1e-5 in price
    at ``max_upper = 2e4``). ``method='quad'`` integrates each strike with adaptive quadrature instead (slow, for checking)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if T <= 0:
        return np.maximum(S - K, 0.0)
    if not 0.0 < step <= 0.5:
        raise ValueError("step must be in (0, 0.5]: the integrand has poles at +-i/2 and the trapezoid rule needs a step well below their distance")
    F = S * np.exp((r - q) * T)
    k = np.log(F / K)
    U = default_upper(model, T, r, q, limit=max_upper) if upper is None else float(upper)
    if method == "quad":
        out = np.empty(len(K))
        for i, ki in enumerate(k):
            f = lambda u: float(np.real(np.exp(1j * u * ki) * model.cf_centered(np.array([u - 0.5j]), T, r, q)[0]) / (u * u + 0.25))       # noqa: E731
            out[i] = integrate.quad(f, 0.0, U, limit=1000, epsabs=1e-13, epsrel=1e-12)[0]
        integral = out
    elif method == "trapezoid":
        n = max(int(np.ceil(U / step)), 8)
        u = np.linspace(0.0, U, n + 1)
        phi = model.cf_centered(u - 0.5j, T, r, q) / (u ** 2 + 0.25)
        integral = np.real(np.exp(1j * np.outer(k, u)) * phi) @ (_trapezoid_weights(n) * (U / n))
    else:
        raise ValueError("method must be 'trapezoid' or 'quad'")
    return S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-(r + q) * T / 2.0) / np.pi * integral


def carr_madan(model, S: float, T: float, r: float = 0.0, q: float = 0.0, alpha: float | None = None, n: int = 2 ** 12, eta: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The call price on a whole grid of strikes by the FFT (Carr and Madan 1999): returns ``(strikes, calls)``, ``n`` of each, with strikes ``S exp(-b + lambda j)``,
    ``lambda = 2 pi / (n eta)`` and ``b = n lambda / 2 = pi / eta``.

    The transform of the damped call has poles at ``v = i alpha`` and ``v = i (alpha + 1)``, so the trapezoid rule with spacing ``eta`` has an error of order ``exp(-2 pi alpha / eta)``:
    ``eta`` defaults to ``min(0.25, 2 pi alpha / 30)``, which keeps that near 1e-13 (a weakly damped model, one with a small exponent limit, needs a finer spacing and so covers a
    shorter range of ``v``)."""
    limit = float(model.exponent_limit()) if hasattr(model, "exponent_limit") else float("inf")
    if alpha is None:
        alpha = 1.5 if not np.isfinite(limit) else min(1.5, 0.4 * (limit - 1.0))
    if alpha <= 0:
        raise ValueError("the damping alpha must be positive: the model has no exponential moment of order above 1")
    if np.isfinite(limit) and alpha + 1.0 >= limit:
        raise ValueError(f"alpha + 1 = {alpha + 1:.3g} must be below the exponent limit {limit:.3g} of the model")
    if eta is None:
        eta = min(0.25, 2.0 * np.pi * alpha / 30.0)
    lam = 2.0 * np.pi / (n * eta)
    b = n * lam / 2.0
    j = np.arange(n)
    v = eta * j
    k = -b + lam * j                                           # ln(K / S)
    cf = model.cf(v - 1j * (alpha + 1.0), T, r, q)
    psi = np.exp(-r * T) * cf / (alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v)
    w = eta * np.where(j == 0, 0.5, 1.0)                       # trapezoid: the integrand is even in v, so the rule converges geometrically
    y = np.fft.fft(np.exp(1j * b * v) * psi * w)
    calls = S * np.exp(-alpha * k) / np.pi * np.real(y)
    return S * np.exp(k), calls


def price(model, S: float, K, T: float, r: float = 0.0, q: float = 0.0, call: bool = True, method: str = "lewis", **kw) -> np.ndarray:
    """European call or put prices at the strikes ``K`` under ``model``. ``method``: ``'lewis'`` (default), ``'fft'`` (Carr-Madan with cubic interpolation) or ``'quad'``. Puts follow from
    put-call parity, which holds exactly for a martingale price."""
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    if method == "fft":
        strikes, calls = carr_madan(model, S, T, r, q, **kw)
        spline = interpolate.CubicSpline(np.log(strikes / S), calls)
        c = spline(np.log(K_arr / S))
    elif method in ("lewis", "quad"):
        c = lewis_call(model, S, K_arr, T, r, q, method="trapezoid" if method == "lewis" else "quad", **kw)
    else:
        raise ValueError("method must be 'lewis', 'fft' or 'quad'")
    out = c if call else c - S * np.exp(-q * T) + K_arr * np.exp(-r * T)
    return out if np.ndim(K) else float(out[0])


def implied_vols(model, S: float, K, T: float, r: float = 0.0, q: float = 0.0, method: str = "lewis", **kw) -> np.ndarray:
    """The Black-Scholes implied volatility of the model's prices at the strikes ``K``: out-of-the-money options are priced (puts below the forward, calls above) because their prices
    carry the volatility information without the intrinsic value that swamps it."""
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    F = S * np.exp((r - q) * T)
    otm_call = K_arr >= F
    c = price(model, S, K_arr, T, r, q, True, method, **kw)
    p = c - S * np.exp(-q * T) + K_arr * np.exp(-r * T)
    px = np.where(otm_call, c, np.maximum(p, 0.0))
    iv = np.empty(len(K_arr))
    if otm_call.any():
        iv[otm_call] = implied_vol(px[otm_call], S, K_arr[otm_call], T, r, q, True)
    if (~otm_call).any():
        iv[~otm_call] = implied_vol(px[~otm_call], S, K_arr[~otm_call], T, r, q, False)
    return iv if np.ndim(K) else float(iv[0])


def risk_neutral_density(model, S: float, T: float, r: float = 0.0, q: float = 0.0, s_grid=None, points: int = 241, step: float = 0.05, limit: float = 1e3) -> dict:
    """The density of the price at expiry under the model, from the inversion ``p(x) = (1/pi) Int_0^inf Re[e^{-iux} phi(u)] du`` of the characteristic function of ``x = ln(S_T/S)``
    (trapezoid rule at spacing ``step``, up to where ``|phi|`` is negligible or ``limit``) and the change of variable ``p_S(s) = p_x(ln(s/S)) / s``. Returns the grid, the density and
    the mass it integrates to (1 when the grid and the truncation are fine; an infinite-activity model over a very short horizon has a cusp that a truncated inversion cannot resolve)."""
    h = 0.02
    sd = float(np.sqrt(max(-2.0 * np.log(abs(model.cf(h, T, r, q))), 1e-12)) / h)
    xs = np.linspace(-8.0 * sd, 8.0 * sd, points) + (r - q) * T if s_grid is None else np.log(np.asarray(s_grid, dtype=float) / S)
    upper = 8.0
    while upper < limit and abs(model.cf(upper, T, r, q)) > 1e-12:
        upper *= 2.0
    n = max(int(np.ceil(upper / step)), 8)
    u = np.linspace(0.0, upper, n + 1)
    phi = model.cf(u, T, r, q)
    px = np.real(np.exp(-1j * np.outer(xs, u)) * phi) @ (_trapezoid_weights(n) * (upper / n)) / np.pi
    s = S * np.exp(xs)
    dens = np.maximum(px, 0.0) / s
    return {"s": s, "density": dens, "mass": float(integrate.trapezoid(px, xs)), "x": xs}
