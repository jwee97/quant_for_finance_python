"""Option pricing models: Black-Scholes-Merton, Black-76, Bachelier, binomial trees (European and American), Longstaff-Schwartz, Heston, Merton jump diffusion and SABR.

Conventions. ``S`` spot, ``K`` strike, ``T`` years to expiry, ``r`` continuously compounded risk-free rate, ``q`` continuous dividend yield (for an index or an ETF),
``sigma`` Black volatility, ``call`` True for a call and False for a put. Every closed-form function is vectorised over numpy arrays (broadcasting).

* ``bsm_price``: ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)`` with ``d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T)``, ``d2 = d1 - sigma sqrt T``.
* ``black76_price``: the same for an option on a forward / futures price ``F`` (no cost of carry; discounting only).
* ``bachelier_price``: normal-model price, used for rates and for assets that can be negative (spreads, some commodities).
* ``binomial_price``: Cox-Ross-Rubinstein tree, European or American; converges to Black-Scholes for the European case.
* ``american_lsmc``: Longstaff & Schwartz (2001) least-squares Monte Carlo for early exercise.
* ``heston_price``: stochastic volatility (Heston 1993) by Fourier inversion of the characteristic function in the numerically stable form of Albrecher et al. (2007).
* ``merton_price``: jump diffusion (Merton 1976) as a Poisson-weighted sum of Black-Scholes prices.
* ``sabr_vol``: Hagan et al. (2002) lognormal implied-volatility approximation; ``sabr_calibrate`` fits ``alpha, rho, nu`` to a smile with ``beta`` fixed.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm, poisson


def _arr(*xs):
    return np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in xs])


def intrinsic(S, K, call=True):
    S, K = _arr(S, K)
    return np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)


def bsm_d1_d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sd
    return d1, d1 - sd


def bsm_price(S, K, T, r, q, sigma, call=True):
    """Black-Scholes-Merton price. At ``T = 0`` or ``sigma = 0`` the discounted forward intrinsic value is returned."""
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    d1, d2 = bsm_d1_d2(S, K, T, r, q, sigma)
    df, dq = np.exp(-r * T), np.exp(-q * T)
    if call:
        price = S * dq * norm.cdf(d1) - K * df * norm.cdf(d2)
    else:
        price = K * df * norm.cdf(-d2) - S * dq * norm.cdf(-d1)
    degenerate = (T <= 0) | (sigma <= 0)
    fwd_intrinsic = np.maximum(S * dq - K * df, 0.0) if call else np.maximum(K * df - S * dq, 0.0)
    return np.where(degenerate, fwd_intrinsic, price)


def black76_price(F, K, T, r, sigma, call=True):
    """Black (1976) price of an option on a forward or futures price: ``e^{-rT} [F N(d1) - K N(d2)]``."""
    return _black76(F, K, T, r, sigma, call)


def _black76(F, K, T, r, sigma, call):
    F, K, T, r, sigma = _arr(F, K, T, r, sigma)
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    price = df * (F * norm.cdf(d1) - K * norm.cdf(d2)) if call else df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    intrinsic_ = df * (np.maximum(F - K, 0.0) if call else np.maximum(K - F, 0.0))
    return np.where((T <= 0) | (sigma <= 0), intrinsic_, price)


def bachelier_price(F, K, T, r, sigma_n, call=True):
    """Normal-model price with absolute volatility ``sigma_n`` (price units per sqrt year): ``e^{-rT} [(F-K) N(d) + sigma sqrt T n(d)]``."""
    F, K, T, r, sigma_n = _arr(F, K, T, r, sigma_n)
    sd = sigma_n * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (F - K) / sd
    df = np.exp(-r * T)
    price = df * ((F - K) * norm.cdf(d) + sd * norm.pdf(d)) if call else df * ((K - F) * norm.cdf(-d) + sd * norm.pdf(d))
    intr = df * (np.maximum(F - K, 0.0) if call else np.maximum(K - F, 0.0))
    return np.where((T <= 0) | (sigma_n <= 0), intr, price)


def put_call_parity_gap(call_price, put_price, S, K, T, r, q=0.0):
    """``C - P - (S e^{-qT} - K e^{-rT})``: zero for European options without arbitrage; in real data it measures the bid-ask noise and the dividend / borrow cost."""
    S, K, T, r, q = _arr(S, K, T, r, q)
    return np.asarray(call_price) - np.asarray(put_price) - (S * np.exp(-q * T) - K * np.exp(-r * T))


# ----------------------------------------------------------------------------------------------------------------------- binomial
def binomial_price(S, K, T, r, q, sigma, call=True, american=False, steps=500):
    """Cox-Ross-Rubinstein tree with ``steps`` periods (scalar inputs). ``american=True`` allows exercise at every node."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    prices = S * u ** (steps - j) * d ** j
    values = np.maximum(prices - K, 0.0) if call else np.maximum(K - prices, 0.0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            nodes = S * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            values = np.maximum(values, np.maximum(nodes - K, 0.0) if call else np.maximum(K - nodes, 0.0))
    return float(values[0])


def american_lsmc(S, K, T, r, q, sigma, call=False, n_paths=50000, steps=50, degree=3, seed=0) -> dict:
    """Longstaff-Schwartz least-squares Monte Carlo for an American option under GBM. Regress the discounted continuation value on a polynomial of the spot over
    in-the-money paths only, exercise when the immediate payoff exceeds the fitted continuation. Returns the price, its standard error and the early-exercise premium
    over the European value (low-biased because the regression policy is sub-optimal; the binomial tree is the benchmark)."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    z = rng.standard_normal((n_paths // 2, steps))
    z = np.vstack([z, -z])
    paths = S * np.exp(np.cumsum((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z, axis=1))
    payoff = (lambda s: np.maximum(s - K, 0.0)) if call else (lambda s: np.maximum(K - s, 0.0))
    cash = payoff(paths[:, -1])
    disc = np.exp(-r * dt)
    for t in range(steps - 2, -1, -1):
        cash = cash * disc
        s = paths[:, t]
        itm = payoff(s) > 0
        if itm.sum() > degree + 2:
            coef = np.polynomial.polynomial.polyfit(s[itm] / S, cash[itm], degree)
            cont = np.polynomial.polynomial.polyval(s[itm] / S, coef)
            exercise = payoff(s[itm]) > cont
            idx = np.flatnonzero(itm)[exercise]
            cash[idx] = payoff(s[idx])
    cash = cash * disc
    price = float(cash.mean())
    return {"price": price, "se": float(cash.std(ddof=1) / np.sqrt(len(cash))), "european": float(bsm_price(S, K, T, r, q, sigma, call)),
            "early_exercise_premium": price - float(bsm_price(S, K, T, r, q, sigma, call))}


# ------------------------------------------------------------------------------------------------------------------------ Heston
def heston_characteristic(u, T, r, q, v0, kappa, theta, xi, rho, S=1.0):
    """Characteristic function of ``ln S_T`` under Heston, in the Albrecher et al. (2007) formulation (continuous in the complex plane, no branch-cut jumps)."""
    u = np.asarray(u, dtype=complex)
    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u ** 2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
    e = np.exp(-d * T)
    C = (r - q) * 1j * u * T + kappa * theta / xi ** 2 * ((kappa - rho * xi * 1j * u - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    D = (kappa - rho * xi * 1j * u - d) / xi ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0 + 1j * u * np.log(S))


def heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, call=True, upper=200.0):
    """Heston (1993) European price by numerical integration of ``P1`` and ``P2`` (Gil-Pelaez). ``v0`` initial variance, ``theta`` long-run variance, ``kappa`` mean
    reversion, ``xi`` volatility of variance, ``rho`` spot-variance correlation. Feller condition ``2 kappa theta >= xi^2`` keeps variance positive."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    out = np.empty(len(K))
    for i, k in enumerate(K):
        lk = np.log(k)
        cf = lambda u: heston_characteristic(u, T, r, q, v0, kappa, theta, xi, rho, S)       # noqa: E731

        def integrand1(u):
            return np.real(np.exp(-1j * u * lk) * cf(u - 1j) / (1j * u * cf(-1j))) if u > 1e-12 else 0.0

        def integrand2(u):
            return np.real(np.exp(-1j * u * lk) * cf(u) / (1j * u)) if u > 1e-12 else 0.0

        p1 = 0.5 + integrate.quad(integrand1, 1e-10, upper, limit=400)[0] / np.pi
        p2 = 0.5 + integrate.quad(integrand2, 1e-10, upper, limit=400)[0] / np.pi
        call_price = S * np.exp(-q * T) * p1 - k * np.exp(-r * T) * p2
        out[i] = call_price if call else call_price - S * np.exp(-q * T) + k * np.exp(-r * T)
    return out if out.size > 1 else float(out[0])


def heston_simulate(S, T, r, q, v0, kappa, theta, xi, rho, steps=252, n_paths=20000, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Full-truncation Euler scheme (Lord, Koekkoek & van Dijk 2010): paths of the spot and the instantaneous variance, shape ``(paths, steps + 1)``."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    s = np.empty((n_paths, steps + 1))
    v = np.empty((n_paths, steps + 1))
    s[:, 0], v[:, 0] = S, v0
    for t in range(steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
        vp = np.maximum(v[:, t], 0.0)
        s[:, t + 1] = s[:, t] * np.exp((r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
        v[:, t + 1] = v[:, t] + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    return s, v


def heston_calibrate(S, strikes, expiries, market_ivs, r=0.0, q=0.0, x0=(0.04, 2.0, 0.04, 0.5, -0.7), weights=None) -> dict:
    """Least-squares calibration of ``(v0, kappa, theta, xi, rho)`` to a grid of implied volatilities (``market_ivs[i, j]`` for ``expiries[i]``, ``strikes[j]``): the model
    prices are converted to implied vols and compared in vol space, which weights the smile evenly across strikes."""
    from .iv import implied_vol

    strikes, expiries = np.asarray(strikes, float), np.asarray(expiries, float)
    market = np.asarray(market_ivs, float)

    def model_ivs(p):
        v0, kappa, theta, xi, rho = p
        out = np.empty_like(market)
        for i, T in enumerate(expiries):
            px = heston_price(S, strikes, T, r, q, v0, kappa, theta, xi, rho, call=True)
            out[i] = implied_vol(px, S, strikes, T, r, q, True)
        return out

    def resid(p):
        m = model_ivs(p)
        res = (m - market)
        res = np.where(np.isfinite(res), res, 1.0)
        return (res * (1.0 if weights is None else weights)).ravel()

    lb, ub = [1e-4, 0.05, 1e-4, 0.05, -0.999], [4.0, 20.0, 4.0, 3.0, 0.999]
    sol = optimize.least_squares(resid, x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10)
    v0, kappa, theta, xi, rho = sol.x
    return {"v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "rmse_vol": float(np.sqrt(np.mean(sol.fun ** 2))), "feller": bool(2 * kappa * theta >= xi ** 2),
            "model_ivs": model_ivs(sol.x)}


# ------------------------------------------------------------------------------------------------------------------------- Merton
def merton_price(S, K, T, r, q, sigma, lam, jump_mean, jump_std, call=True, n_terms=60):
    """Merton (1976) jump diffusion: ``sum_n Poisson(n; lam' T) BS(S, K, T, r_n, q, sigma_n)`` with ``lam' = lam (1 + k)``, ``k = e^{m + s^2/2} - 1``,
    ``sigma_n^2 = sigma^2 + n s^2 / T`` and ``r_n = r - lam k + n (m + s^2/2) / T``."""
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    kbar = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1.0
    lam_p = lam * (1.0 + kbar)
    total = np.zeros_like(S, dtype=float)
    for n in range(n_terms):
        sig_n = np.sqrt(sigma ** 2 + n * jump_std ** 2 / T)
        r_n = r - lam * kbar + n * (jump_mean + 0.5 * jump_std ** 2) / T
        weight = poisson.pmf(n, lam_p * T)
        total = total + weight * bsm_price(S, K, T, r_n, q, sig_n, call) * np.exp((r_n - r) * T)       # grow at r_n, discount at r
    return total


# --------------------------------------------------------------------------------------------------------------------------- SABR
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan, Kumar, Lesniewski & Woodward (2002) lognormal implied volatility for the SABR model with forward ``F``."""
    F, K, T = _arr(F, K, T)
    out = np.empty_like(F, dtype=float)
    atm = np.abs(F - K) < 1e-12 * np.maximum(F, 1e-12)
    one_b = 1.0 - beta
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fk = np.log(F / K)
        fk_b = (F * K) ** (one_b / 2.0)
        z = nu / alpha * fk_b * log_fk
        x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))
        zx = np.where(np.abs(z) < 1e-8, 1.0, z / x)
        denom = fk_b * (1 + one_b ** 2 / 24 * log_fk ** 2 + one_b ** 4 / 1920 * log_fk ** 4)
        corr = 1 + (one_b ** 2 / 24 * alpha ** 2 / fk_b ** 2 + 0.25 * rho * beta * nu * alpha / fk_b + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
        vol = alpha / denom * zx * corr
        atm_vol = alpha / F ** one_b * (1 + (one_b ** 2 / 24 * alpha ** 2 / F ** (2 * one_b) + 0.25 * rho * beta * nu * alpha / F ** one_b + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T)
    out = np.where(atm, atm_vol, vol)
    return out


def sabr_calibrate(F, strikes, T, market_vols, beta=0.5, x0=None) -> dict:
    """Fit ``alpha, rho, nu`` of the SABR smile at one expiry by least squares on implied vols (``beta`` fixed, as is standard practice since it is not separately identified)."""
    strikes, mv = np.asarray(strikes, float), np.asarray(market_vols, float)
    atm_vol = float(np.interp(F, strikes, mv))
    a0 = atm_vol * F ** (1 - beta)
    x0 = x0 or [a0, -0.3, 0.5]

    def resid(p):
        return sabr_vol(F, strikes, T, p[0], beta, p[1], p[2]) - mv

    sol = optimize.least_squares(resid, x0, bounds=([1e-6, -0.999, 1e-4], [10.0, 0.999, 5.0]), xtol=1e-12, ftol=1e-12)
    return {"alpha": float(sol.x[0]), "beta": beta, "rho": float(sol.x[1]), "nu": float(sol.x[2]), "rmse": float(np.sqrt(np.mean(sol.fun ** 2)))}
