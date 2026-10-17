r"""Short-rate models: the Vasicek and Cox-Ingersoll-Ross prices of a zero-coupon bond, the yield curves they imply, and fits to a curve and to a history of rates.

A short-rate model says the whole term structure is a function of one number, today's instantaneous rate ``r``, and so prices every bond from it. Both models here pull the rate back to a
long-run level ``theta`` at speed ``kappa``::

    Vasicek   dr = kappa (theta - r) dt + sigma dW                   Gaussian: the rate can go negative (as rates have).
    CIR       dr = kappa (theta - r) dt + sigma sqrt(r) dW           the rate stays positive when ``2 kappa theta >= sigma^2``.

A zero-coupon bond of maturity ``tau`` is worth ``P = A(tau) exp(-B(tau) r)`` in both, with ``A`` and ``B`` in closed form (the solution of the bond equation
``P_tau = kappa (theta - r) P_r + (sigma^2 / 2) r^{2 gamma} P_rr - r P`` with ``gamma = 0`` or ``1/2``); ``B(tau)`` is also the bond's *duration to the short rate*: the percentage the
price falls for a unit rise in ``r``. The parameters here are the risk-neutral ones that price bonds; the history of rates follows the same equation with the physical long-run level
(``fit_history``), and the difference between the two levels is the market price of rate risk.

``bond_price``, ``yield_curve``   the closed forms.
``bond_price_mc``                 the same price as ``E[exp(-Int r dt)]`` over exact simulated paths: the check on the formula.
``fit_curve``                     a least-squares fit of the model to an observed curve of yields. One curve does not identify all of ``kappa, theta, sigma, r0``; fix the speed and volatility
                                  from history (``fit_history``) and fit the level and the starting rate, or let all four float and read the standard errors.
``fit_history``                   the maximum-likelihood fit to a history of the short rate (exact transition densities), through ``probability.ou`` and ``probability.sde``.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..probability import ou, sde

MODELS = ("vasicek", "cir")


def _check(model: str, kappa: float, theta: float, sigma: float) -> None:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if kappa <= 0 or sigma <= 0:
        raise ValueError("kappa and sigma must be positive")
    if model == "cir" and theta <= 0:
        raise ValueError("the CIR long-run level must be positive")


def _ab(model: str, tau, kappa: float, theta: float, sigma: float):
    """``A(tau)`` (as its log) and ``B(tau)`` for the two models."""
    tau = np.asarray(tau, dtype=float)
    if model == "vasicek":
        b = (1.0 - np.exp(-kappa * tau)) / kappa
        log_a = (theta - sigma ** 2 / (2.0 * kappa ** 2)) * (b - tau) - sigma ** 2 * b ** 2 / (4.0 * kappa)
        return log_a, b
    h = np.sqrt(kappa ** 2 + 2.0 * sigma ** 2)
    e = np.exp(-h * tau)
    denom = (h + kappa) * (1.0 - e) + 2.0 * h * e
    b = 2.0 * (1.0 - e) / denom
    log_a = 2.0 * kappa * theta / sigma ** 2 * np.log(2.0 * h * np.exp((kappa - h) * tau / 2.0) / denom)
    return log_a, b


def bond_price(model: str, r, tau, kappa: float, theta: float, sigma: float):
    """The price of a zero-coupon bond paying one in ``tau`` years when the short rate is ``r`` (arrays allowed for ``r`` and ``tau``)."""
    _check(model, kappa, theta, sigma)
    log_a, b = _ab(model, tau, kappa, theta, sigma)
    return np.exp(log_a - b * np.asarray(r, dtype=float))


def yield_curve(model: str, r: float, maturities, kappa: float, theta: float, sigma: float) -> np.ndarray:
    """The continuously compounded zero-coupon yields ``-ln P / tau`` at ``maturities``."""
    tau = np.asarray(maturities, dtype=float)
    return -np.log(bond_price(model, r, tau, kappa, theta, sigma)) / tau


def duration_to_short_rate(model: str, tau, kappa: float, theta: float, sigma: float) -> np.ndarray:
    """``B(tau)``: the proportional fall in a ``tau``-year bond's price per unit rise of the short rate (a sensitivity to the one factor, not a duration to a parallel shift of the curve)."""
    _check(model, kappa, theta, sigma)
    return _ab(model, tau, kappa, theta, sigma)[1]


def long_yield(model: str, kappa: float, theta: float, sigma: float) -> float:
    """The limit of the yield as the maturity grows: ``theta - sigma^2 / (2 kappa^2)`` for Vasicek, ``2 kappa theta / (kappa + h)`` for CIR (``h = sqrt(kappa^2 + 2 sigma^2)``): below ``theta``,
    because the convexity of the bond price makes long bonds worth more than the average path of the rate suggests."""
    _check(model, kappa, theta, sigma)
    if model == "vasicek":
        return float(theta - sigma ** 2 / (2.0 * kappa ** 2))
    return float(2.0 * kappa * theta / (kappa + np.sqrt(kappa ** 2 + 2.0 * sigma ** 2)))


def bond_price_mc(model: str, r0: float, tau: float, kappa: float, theta: float, sigma: float, steps: int = 100, n_paths: int = 100_000, seed: int = 0) -> dict:
    """``E[exp(-Int_0^tau r dt)]`` over ``n_paths`` exact paths (the Ornstein-Uhlenbeck transition for Vasicek, the non-central chi-square one for CIR), the integral by the trapezoid rule,
    with the standard error. For Vasicek the paths are mirrored (antithetic)."""
    _check(model, kappa, theta, sigma)
    if model == "vasicek":
        half = (n_paths + 1) // 2
        rng = np.random.default_rng(seed)
        dt = tau / steps
        phi = np.exp(-kappa * dt)
        sd = sigma * np.sqrt((1.0 - phi ** 2) / (2.0 * kappa))
        z = rng.standard_normal((half, steps))
        z = np.vstack([z, -z])[:n_paths]
        r = np.empty((n_paths, steps + 1))
        r[:, 0] = r0
        for i in range(steps):
            r[:, i + 1] = theta + (r[:, i] - theta) * phi + sd * z[:, i]
    else:
        r = sde.simulate_cir(r0, kappa, theta, sigma, tau, steps, n_paths, exact=True, seed=seed)
    integral = np.sum(0.5 * (r[:, 1:] + r[:, :-1]), axis=1) * (tau / steps)                  # the trapezoid rule
    disc = np.exp(-integral)
    return {"price": float(disc.mean()), "se": float(disc.std(ddof=1) / np.sqrt(len(disc))), "n_paths": int(len(disc))}


# ------------------------------------------------------------------------------------------------------------------------------------ fitting
def fit_curve(model: str, maturities, yields, kappa: float | None = None, sigma: float | None = None, r0: float | None = None) -> dict:
    """Fit the model to a curve of zero-coupon yields by least squares. The free parameters are the long-run level ``theta`` and the starting rate ``r0`` (unless given), and ``kappa`` and
    ``sigma`` unless fixed. Returns the parameters, the root-mean-square error in yield and the model's yields. A cross-section identifies the level and the starting rate well and the
    speed and volatility poorly (a faster pull to a lower level looks much like a slower pull to a higher one): fix those from ``fit_history`` where you can."""
    tau, y = np.asarray(maturities, dtype=float), np.asarray(yields, dtype=float)
    if len(tau) != len(y) or len(tau) < 4 or np.any(tau <= 0):
        raise ValueError("at least four maturities with their yields are needed")
    free = ["theta", "r0"] + (["kappa"] if kappa is None else []) + (["sigma"] if sigma is None else [])
    start = {"theta": float(y[-1]) + 0.005, "r0": float(y[0]) if r0 is None else float(r0), "kappa": 0.5 if kappa is None else kappa, "sigma": 0.01 if sigma is None else sigma}
    if r0 is not None:
        free.remove("r0")
    lows = {"theta": 1e-6 if model == "cir" else -0.05, "r0": 1e-6 if model == "cir" else -0.05, "kappa": 0.01, "sigma": 1e-4}
    highs = {"theta": 0.5, "r0": 0.5, "kappa": 10.0, "sigma": 0.5}

    def unpack(x):
        p = {"theta": 0.0, "r0": r0, "kappa": kappa, "sigma": sigma}
        p.update(dict(zip(free, x)))
        return p

    def resid(x):
        p = unpack(x)
        if model == "cir" and p["theta"] <= 0:
            return np.full(len(tau), 1.0)
        return yield_curve(model, p["r0"], tau, p["kappa"], p["theta"], p["sigma"]) - y

    sol = optimize.least_squares(resid, [start[k] for k in free], bounds=([lows[k] for k in free], [highs[k] for k in free]), xtol=1e-12, ftol=1e-12, x_scale=[max(abs(start[k]), 1e-2) for k in free])
    p = unpack(sol.x)
    fitted = yield_curve(model, p["r0"], tau, p["kappa"], p["theta"], p["sigma"])
    se = np.full(len(free), np.nan)
    dof = len(tau) - len(free)
    if dof > 0 and np.linalg.cond(sol.jac) < 1e12:
        se = np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac)) * 2.0 * sol.cost / dof)
    return {"model": model, "params": {k: float(v) for k, v in p.items()}, "std_errors": dict(zip(free, map(float, se))), "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
            "fitted": fitted, "success": bool(sol.success), "feller": bool(model != "cir" or 2.0 * p["kappa"] * p["theta"] >= p["sigma"] ** 2)}


def fit_history(model: str, rates, dt: float = 1.0 / 252.0) -> dict:
    """The maximum-likelihood fit of ``(kappa, theta, sigma)`` to a history of the short rate: for Vasicek the Ornstein-Uhlenbeck fit, for CIR the non-central chi-square one. These are the
    *physical* parameters. Returns the estimates, their standard errors and the log-likelihood."""
    x = np.asarray(rates, dtype=float)
    if model == "vasicek":
        f = ou.fit_ou(x, dt)
        names = {"theta": "kappa", "mu": "theta", "sigma": "sigma"}
        est = {names[k]: v for k, v in f.estimates.items()}
        se = {names[k]: v for k, v in f.std_errors.items()}
        return {"model": model, "params": est, "std_errors": se, "loglik": f.loglik, "n": f.n}
    if model == "cir":
        f = sde.fit_cir(x, dt)
        return {"model": model, "params": dict(f.estimates), "std_errors": dict(f.std_errors), "loglik": f.loglik, "n": f.n}
    raise ValueError(f"model must be one of {MODELS}")
