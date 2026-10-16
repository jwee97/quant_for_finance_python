"""Stochastic differential equations: discretisation schemes, their errors, exact and quasi-exact simulation of the square-root processes, and fitting them to data.

A stochastic differential equation ``dX = a(X) dt + b(X) dW`` has a solution only in a few cases; otherwise a path is built step by step and the question is how fast the
error falls as the step shrinks. Two errors matter. The *strong* error ``E|X_T - X_T^h|`` measures how close the simulated path is to the true one driven by the same
noise (it matters for hedging and for path-dependent payoffs); the *weak* error ``|E f(X_T) - E f(X_T^h)|`` measures how close the distribution is (it is all a European
price needs). Euler-Maruyama has strong order 1/2 and weak order 1; Milstein adds the term ``1/2 b b' (dW^2 - dt)`` and has strong order 1.

``euler_maruyama`` and ``milstein``   the two schemes for a scalar equation, with the option of returning the Brownian increments so that a run can be compared with the exact solution.
``strong_error``, ``weak_error``      measure the order of a scheme against a known solution.
``simulate_cir``                     the square-root (Cox-Ingersoll-Ross) process ``dv = kappa (theta - v) dt + xi sqrt(v) dW``: exact sampling of its transition (a scaled non-central
                                     chi-square), or the full-truncation Euler scheme that stays defined when the Feller condition ``2 kappa theta >= xi^2`` fails.
``simulate_heston``                  the Heston stochastic-volatility model by Andersen's (2008) quadratic-exponential (QE) scheme for the variance with a martingale-corrected
                                     log-price step, which stays accurate with the large steps (a day, a week) that full-truncation Euler needs very many substeps to match.
``fit_gbm``, ``fit_cir``             maximum-likelihood estimates with standard errors and confidence intervals, from the exact transition densities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize, stats


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


# --------------------------------------------------------------------------------------------------------------------------- generic schemes
def euler_maruyama(drift: Callable, diffusion: Callable, x0: float, horizon: float, steps: int, n_paths: int, seed=0, increments: np.ndarray | None = None,
                   return_noise: bool = False):
    """Euler-Maruyama: ``X_{k+1} = X_k + a(X_k) h + b(X_k) dW_k``. ``drift`` and ``diffusion`` take and return arrays. Returns paths of shape ``(n_paths, steps + 1)``; with
    ``return_noise`` also the Brownian increments used (shape ``(n_paths, steps)``), and ``increments`` supplies them instead of drawing, so that two schemes (or a scheme and an
    exact solution) can share one noise."""
    rng = _rng(seed)
    h = horizon / steps
    dw = np.sqrt(h) * rng.standard_normal((n_paths, steps)) if increments is None else np.asarray(increments, dtype=float)
    x = np.empty((n_paths, steps + 1))
    x[:, 0] = x0
    for k in range(steps):
        xk = x[:, k]
        x[:, k + 1] = xk + drift(xk) * h + diffusion(xk) * dw[:, k]
    return (x, dw) if return_noise else x


def milstein(drift: Callable, diffusion: Callable, diffusion_prime: Callable, x0: float, horizon: float, steps: int, n_paths: int, seed=0, increments: np.ndarray | None = None,
             return_noise: bool = False):
    """Milstein: Euler plus ``1/2 b(X) b'(X) (dW^2 - h)``, which lifts the strong order from 1/2 to 1 for a scalar equation. ``diffusion_prime`` is ``b'``."""
    rng = _rng(seed)
    h = horizon / steps
    dw = np.sqrt(h) * rng.standard_normal((n_paths, steps)) if increments is None else np.asarray(increments, dtype=float)
    x = np.empty((n_paths, steps + 1))
    x[:, 0] = x0
    for k in range(steps):
        xk = x[:, k]
        b = diffusion(xk)
        x[:, k + 1] = xk + drift(xk) * h + b * dw[:, k] + 0.5 * b * diffusion_prime(xk) * (dw[:, k] ** 2 - h)
    return (x, dw) if return_noise else x


def _gbm_exact(mu: float, sigma: float, x0: float, horizon: float, dw: np.ndarray) -> np.ndarray:
    """The exact geometric Brownian motion at the end of each path built from Brownian increments ``dw``."""
    return x0 * np.exp((mu - 0.5 * sigma ** 2) * horizon + sigma * dw.sum(axis=1))


def strong_error(scheme: str, mu: float, sigma: float, x0: float = 1.0, horizon: float = 1.0, steps_list=(4, 8, 16, 32, 64), n_paths: int = 20_000, seed: int = 0) -> dict:
    """The strong error ``E|X_T - X_T^h|`` of ``'euler'`` or ``'milstein'`` for geometric Brownian motion, whose exact solution is known, over a list of step counts, and the order
    estimated by the slope of ``log error`` on ``log h`` (about 0.5 for Euler and 1 for Milstein)."""
    if scheme not in ("euler", "milstein"):
        raise ValueError("scheme must be 'euler' or 'milstein'")
    rng = np.random.default_rng(seed)
    fine = max(steps_list) * 4
    dw_fine = np.sqrt(horizon / fine) * rng.standard_normal((n_paths, fine))
    exact = _gbm_exact(mu, sigma, x0, horizon, dw_fine)
    errors = []
    for steps in steps_list:
        group = fine // steps
        dw = dw_fine.reshape(n_paths, steps, group).sum(axis=2)           # the same noise, summed over each coarse step
        a, b = (lambda x: mu * x), (lambda x: sigma * x)
        if scheme == "euler":
            path = euler_maruyama(a, b, x0, horizon, steps, n_paths, increments=dw)
        else:
            path = milstein(a, b, lambda x: sigma * np.ones_like(x), x0, horizon, steps, n_paths, increments=dw)
        errors.append(float(np.mean(np.abs(path[:, -1] - exact))))
    h = horizon / np.asarray(steps_list, dtype=float)
    order = float(np.polyfit(np.log(h), np.log(errors), 1)[0])
    return {"scheme": scheme, "steps": list(steps_list), "errors": errors, "order": order}


def weak_error(scheme: str, mu: float, sigma: float, x0: float = 1.0, horizon: float = 1.0, steps_list=(2, 4, 8, 16), n_paths: int = 200_000, seed: int = 0) -> dict:
    """The weak error ``|E X_T^h - E X_T|`` of a scheme for geometric Brownian motion over a list of step counts, and the order estimated as for the strong error (about 1 for both
    schemes). The difference of the means is estimated from the scheme and the exact solution driven by the *same* noise, so the Monte Carlo error of the two cancels and the small
    bias of the scheme is not lost in it."""
    if scheme not in ("euler", "milstein"):
        raise ValueError("scheme must be 'euler' or 'milstein'")
    rng = np.random.default_rng(seed)
    fine = max(steps_list)
    dw_fine = np.sqrt(horizon / fine) * rng.standard_normal((n_paths, fine))
    exact = _gbm_exact(mu, sigma, x0, horizon, dw_fine)
    errors = []
    for steps in steps_list:
        dw = dw_fine.reshape(n_paths, steps, fine // steps).sum(axis=2)
        a, b = (lambda x: mu * x), (lambda x: sigma * x)
        if scheme == "euler":
            path = euler_maruyama(a, b, x0, horizon, steps, n_paths, increments=dw)
        else:
            path = milstein(a, b, lambda x: sigma * np.ones_like(x), x0, horizon, steps, n_paths, increments=dw)
        errors.append(float(abs(np.mean(path[:, -1] - exact))))
    h = horizon / np.asarray(steps_list, dtype=float)
    return {"scheme": scheme, "steps": list(steps_list), "errors": errors, "order": float(np.polyfit(np.log(h), np.log(errors), 1)[0])}


# ----------------------------------------------------------------------------------------------------------------------- square-root process
def cir_moments(v, kappa: float, theta: float, xi: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """The conditional mean and variance of the CIR process after ``dt`` given ``v`` (exact): ``theta + (v - theta) e^{-kappa dt}`` and
    ``v xi^2 e^{-kappa dt} (1 - e^{-kappa dt}) / kappa + theta xi^2 (1 - e^{-kappa dt})^2 / (2 kappa)``."""
    v = np.asarray(v, dtype=float)
    e = np.exp(-kappa * dt)
    mean = theta + (v - theta) * e
    var = v * xi ** 2 * e * (1.0 - e) / kappa + theta * xi ** 2 * (1.0 - e) ** 2 / (2.0 * kappa)
    return mean, var


def simulate_cir(v0: float, kappa: float, theta: float, xi: float, horizon: float, steps: int, n_paths: int, exact: bool = True, seed=0) -> np.ndarray:
    """Paths of ``dv = kappa (theta - v) dt + xi sqrt(v) dW``, shape ``(n_paths, steps + 1)``.

    ``exact=True`` samples the transition density: ``v_{t+h} = c * chi2'(d, v_t e^{-kappa h} / c)`` with ``c = xi^2 (1 - e^{-kappa h}) / (4 kappa)`` and ``d = 4 kappa theta / xi^2``
    degrees of freedom (a non-central chi-square), which is right for any step. ``exact=False`` is the full-truncation Euler scheme (Lord, Koekkoek and van Dijk 2010), which
    needs small steps when ``2 kappa theta < xi^2`` (variance reaches zero) but is the one that generalises to models with no known transition.
    """
    if min(v0, kappa, theta, xi) <= 0:
        raise ValueError("v0, kappa, theta and xi must be positive")
    rng = _rng(seed)
    h = horizon / steps
    v = np.empty((n_paths, steps + 1))
    v[:, 0] = v0
    if exact:
        e = np.exp(-kappa * h)
        c = xi ** 2 * (1.0 - e) / (4.0 * kappa)
        df = 4.0 * kappa * theta / xi ** 2
        for k in range(steps):
            v[:, k + 1] = c * rng.noncentral_chisquare(df, v[:, k] * e / c)
    else:
        for k in range(steps):
            vp = np.maximum(v[:, k], 0.0)
            v[:, k + 1] = v[:, k] + kappa * (theta - vp) * h + xi * np.sqrt(vp * h) * rng.standard_normal(n_paths)
    return v


# ------------------------------------------------------------------------------------------------------------------------------------ Heston
def simulate_heston(S0: float, v0: float, r: float, q: float, kappa: float, theta: float, xi: float, rho: float, horizon: float, steps: int, n_paths: int, scheme: str = "qe",
                    seed=0, psi_c: float = 1.5, antithetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Paths of the Heston model ``dS = (r - q) S dt + sqrt(v) S dW_1``, ``dv = kappa (theta - v) dt + xi sqrt(v) dW_2``, ``corr(dW_1, dW_2) = rho``; returns the spot and the
    variance, each ``(n_paths, steps + 1)``.

    ``scheme='qe'``: Andersen's (2008) quadratic-exponential scheme. The variance step matches the exact conditional mean ``m`` and variance ``s^2``: when ``psi = s^2 / m^2``
    is at most ``psi_c`` (variance far from zero) the next variance is ``a (b + Z)^2``, a scaled non-central chi-square approximated by a squared normal; otherwise it is zero with
    probability ``p`` and exponential otherwise, which puts the right mass at zero. The log-spot is stepped with the central discretisation ``gamma_1 = gamma_2 = 1/2`` and a
    drift correction ``K_0*`` that makes ``E[S_{t+h}] = S_t e^{(r-q) h}`` exactly, so the discounted price is a martingale however large the step.
    ``scheme='euler'``: full-truncation Euler (Lord, Koekkoek and van Dijk 2010), for comparison. ``antithetic`` pairs each path with its mirror image.
    """
    if scheme not in ("qe", "euler"):
        raise ValueError("scheme must be 'qe' or 'euler'")
    if min(v0, kappa, theta, xi) <= 0 or not -1.0 < rho < 1.0:
        raise ValueError("v0, kappa, theta and xi must be positive and rho strictly between -1 and 1")
    rng = _rng(seed)
    h = horizon / steps
    n = n_paths + (n_paths % 2) if antithetic else n_paths
    half = n // 2 if antithetic else n
    S = np.empty((n, steps + 1))
    V = np.empty((n, steps + 1))
    S[:, 0], V[:, 0] = S0, v0
    logS = np.full(n, np.log(S0))
    v = np.full(n, float(v0))
    e = np.exp(-kappa * h)
    k1, k2 = 0.5 * h * (kappa * rho / xi - 0.5) - rho / xi, 0.5 * h * (kappa * rho / xi - 0.5) + rho / xi        # the central discretisation of the integrated variance, gamma_1 = gamma_2 = 1/2
    k3 = k4 = 0.5 * h * (1.0 - rho ** 2)
    a_coef = k2 + 0.5 * k4
    for k in range(steps):
        zs = rng.standard_normal(half)
        zv = rng.standard_normal(half)
        us = rng.random(half)
        if antithetic:
            zs, zv, us = np.concatenate([zs, -zs]), np.concatenate([zv, -zv]), np.concatenate([us, 1.0 - us])
        if scheme == "euler":
            vp = np.maximum(v, 0.0)
            z1 = zs
            z2 = rho * zs + np.sqrt(1.0 - rho ** 2) * zv
            logS = logS + (r - q - 0.5 * vp) * h + np.sqrt(vp * h) * z1
            v = v + kappa * (theta - vp) * h + xi * np.sqrt(vp * h) * z2
        else:
            m = theta + (v - theta) * e
            s2 = v * xi ** 2 * e * (1.0 - e) / kappa + theta * xi ** 2 * (1.0 - e) ** 2 / (2.0 * kappa)
            psi = s2 / np.maximum(m ** 2, 1e-300)
            quad = psi <= psi_c
            vn = np.empty(n)
            corr = np.empty(n)                                                    # the log of E[exp(A v_next)], for the martingale correction
            if quad.any():
                p2 = 2.0 / psi[quad]
                b2 = p2 - 1.0 + np.sqrt(p2) * np.sqrt(p2 - 1.0)
                a = m[quad] / (1.0 + b2)
                if np.any(1.0 - 2.0 * a_coef * a <= 0.0):
                    raise ValueError("the step is too large for the martingale correction of the QE scheme: use more steps")
                vn[quad] = a * (np.sqrt(b2) + zv[quad]) ** 2
                corr[quad] = a_coef * b2 * a / (1.0 - 2.0 * a_coef * a) - 0.5 * np.log(1.0 - 2.0 * a_coef * a)
            if (~quad).any():
                pp = (psi[~quad] - 1.0) / (psi[~quad] + 1.0)
                beta = (1.0 - pp) / m[~quad]
                u = us[~quad]
                vn[~quad] = np.where(u <= pp, 0.0, np.log((1.0 - pp) / np.maximum(1.0 - u, 1e-300)) / beta)
                if np.any(beta - a_coef <= 0.0):
                    raise ValueError("the step is too large for the martingale correction of the QE scheme: use more steps")
                corr[~quad] = np.log(pp + beta * (1.0 - pp) / (beta - a_coef))
            k0_star = -corr - (k1 + 0.5 * k3) * v                                   # martingale-corrected drift of the log-spot
            logS = logS + (r - q) * h + k0_star + k1 * v + k2 * vn + np.sqrt(np.maximum(k3 * v + k4 * vn, 0.0)) * zs
            v = vn
        S[:, k + 1] = np.exp(logS)
        V[:, k + 1] = v
    return S[:n_paths], V[:n_paths]


# ----------------------------------------------------------------------------------------------------------------------------------- estimation
@dataclass
class FitResult:
    """A maximum-likelihood fit: parameter ``estimates``, their ``std_errors`` (from the observed information), 95% ``ci``, the log-likelihood and the sample size."""

    estimates: dict
    std_errors: dict
    ci: dict
    loglik: float
    n: int
    converged: bool = True


def _ci(est: dict, se: dict, level: float = 0.95) -> dict:
    z = stats.norm.ppf(0.5 + level / 2.0)
    return {k: (est[k] - z * se[k], est[k] + z * se[k]) for k in est}


def fit_gbm(prices=None, log_returns=None, dt: float = 1.0 / 252.0) -> FitResult:
    """Maximum-likelihood fit of geometric Brownian motion to prices (or to log returns): ``sigma^2 = Var(r) / dt`` (the divisor ``n``, which is the maximum-likelihood form),
    ``mu = mean(r) / dt + sigma^2 / 2``. The standard errors are the exact ones for a normal sample: ``se(mean log drift) = sigma / sqrt(T)`` and ``se(sigma) = sigma / sqrt(2 n)``;
    the standard error of ``mu`` adds the two in quadrature (the two estimates are independent). It makes plain how poorly the drift is known and how well the volatility is: a year
    of daily data pins down ``sigma`` to a few per cent and ``mu`` to several times its own size."""
    if (prices is None) == (log_returns is None):
        raise ValueError("give prices or log_returns")
    r = np.diff(np.log(np.asarray(prices, dtype=float))) if prices is not None else np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        raise ValueError("at least ten returns are needed")
    var = float(np.mean((r - r.mean()) ** 2))
    sigma = float(np.sqrt(var / dt))
    drift_log = float(r.mean() / dt)
    mu = drift_log + 0.5 * sigma ** 2
    total = n * dt
    se_drift, se_sigma = sigma / np.sqrt(total), sigma / np.sqrt(2.0 * n)
    se_mu = float(np.sqrt(se_drift ** 2 + (sigma * se_sigma) ** 2))
    ll = float(-0.5 * n * (np.log(2.0 * np.pi * var) + 1.0))
    est = {"mu": mu, "sigma": sigma, "log_drift": drift_log}
    se = {"mu": se_mu, "sigma": float(se_sigma), "log_drift": float(se_drift)}
    return FitResult(est, se, _ci(est, se), ll, n)


def cir_loglik(params, x: np.ndarray, dt: float) -> float:
    """The exact log-likelihood of a CIR sample ``x`` observed at steps ``dt``: ``x_{t+dt} / c`` is non-central chi-square with ``4 kappa theta / xi^2`` degrees of freedom and
    non-centrality ``x_t e^{-kappa dt} / c``."""
    kappa, theta, xi = params
    if min(kappa, theta, xi) <= 0:
        return -np.inf
    e = np.exp(-kappa * dt)
    c = xi ** 2 * (1.0 - e) / (4.0 * kappa)
    df = 4.0 * kappa * theta / xi ** 2
    nc = x[:-1] * e / c
    ll = stats.ncx2.logpdf(x[1:] / c, df, nc) - np.log(c)
    return float(np.sum(ll)) if np.all(np.isfinite(ll)) else -np.inf


def fit_cir(x, dt: float = 1.0 / 252.0, start=None) -> FitResult:
    """Maximum-likelihood fit of ``(kappa, theta, xi)`` of a square-root process to a positive series (a variance proxy such as a VIX squared, or an interest rate), from the exact
    non-central chi-square transition density. The start is the least-squares fit of the discretised mean reversion; the standard errors come from a numerical Hessian."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30 or np.any(x <= 0):
        raise ValueError("a CIR fit needs at least 30 strictly positive observations")
    if start is None:
        b, a = np.polyfit(x[:-1], x[1:], 1)
        b = float(np.clip(b, 1e-3, 0.999))
        kappa0 = -np.log(b) / dt
        theta0 = float(a / (1.0 - b))
        resid = x[1:] - (a + b * x[:-1])
        xi0 = float(np.sqrt(np.mean(resid ** 2 / np.maximum(x[:-1], 1e-12) * 2.0 * kappa0 / max(1.0 - b ** 2, 1e-6))))
        start = (kappa0, max(theta0, float(x.mean())), max(xi0, 1e-3))
    res = optimize.minimize(lambda p: -cir_loglik(np.exp(p), x, dt), np.log(np.maximum(start, 1e-6)), method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
    est = np.exp(res.x)
    # the standard errors from a finite-difference Hessian of the log-likelihood in the natural parameters
    se = np.full(3, np.nan)
    try:
        hess = np.empty((3, 3))
        steps = 1e-4 * np.maximum(est, 1e-3)
        f = lambda p: cir_loglik(p, x, dt)            # noqa: E731
        for i in range(3):
            for j in range(3):
                ei, ej = np.eye(3)[i] * steps[i], np.eye(3)[j] * steps[j]
                hess[i, j] = (f(est + ei + ej) - f(est + ei - ej) - f(est - ei + ej) + f(est - ei - ej)) / (4.0 * steps[i] * steps[j])
        cov = np.linalg.inv(-hess)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except (np.linalg.LinAlgError, FloatingPointError):
        pass
    names = ("kappa", "theta", "xi")
    est_d, se_d = dict(zip(names, map(float, est))), dict(zip(names, map(float, se)))
    return FitResult(est_d, se_d, _ci(est_d, se_d), float(-res.fun), len(x) - 1, bool(res.success))
