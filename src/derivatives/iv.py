"""Implied volatility: invert Black-Scholes-Merton (or Black-76 / Bachelier) for the volatility that reproduces an observed option price.

A price carries information only above the no-arbitrage lower bound ``max(S e^{-qT} - K e^{-rT}, 0)`` (call) and below the upper bound ``S e^{-qT}``; outside these bounds
no volatility exists and the result is NaN rather than a misleading number. So is a price within ``1e-9 S`` of a bound: the volatility is then not identified (a deep
in-the-money option is almost all intrinsic value). The solver is a safeguarded Newton method vectorised over arrays: it starts from the
Brenner-Subrahmanyam / Corrado-Miller guess, takes Newton steps while they stay inside a bracket, and bisects otherwise, so it converges for deep in- and out-of-the-money
options where plain Newton diverges (vega is almost zero there).
"""

from __future__ import annotations

import numpy as np

from .pricing import _arr, bachelier_price, black76_price, bsm_price


def _vega_bsm(S, K, T, r, q, sigma):
    from scipy.stats import norm

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r=0.0, q=0.0, call=True, tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """Implied Black-Scholes-Merton volatility. Vectorised; NaN where the price is outside the no-arbitrage bounds or ``T <= 0``."""
    price, S, K, T, r, q = _arr(price, S, K, T, r, q)
    shape = price.shape
    price, S, K, T, r, q = (np.array(x, dtype=float).ravel() for x in (price, S, K, T, r, q))
    df, dq = np.exp(-r * T), np.exp(-q * T)
    lower = np.maximum(S * dq - K * df, 0.0) if call else np.maximum(K * df - S * dq, 0.0)
    upper = S * dq if call else K * df
    eps = 1e-9 * S                                                # below this much time value the price carries no usable volatility information
    valid = np.isfinite(price) & (T > 0) & (price > lower + eps) & (price < upper - eps)
    lo = np.full(price.shape, 1e-8)
    hi = np.full(price.shape, 10.0)
    fwd = S * np.exp((r - q) * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.sqrt(2.0 * np.pi / T) * price / (S * dq)                      # Brenner-Subrahmanyam, exact at the money
        m = np.log(fwd / K)
        guess = np.where(np.abs(m) > 0.3, np.sqrt(2.0 * np.abs(m) / T), guess)
    sigma = np.clip(np.where(np.isfinite(guess), guess, 0.2), 0.01, 5.0)
    out = np.full(price.shape, np.nan)
    active = valid.copy()
    for _ in range(max_iter):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        s_i = sigma[idx]
        diff = bsm_price(S[idx], K[idx], T[idx], r[idx], q[idx], s_i, call) - price[idx]
        done = np.abs(diff) < tol * S[idx] * 1e-2                  # price tolerance 1e-12 of spot: tight enough to resolve options with almost no time value
        out[idx[done]] = s_i[done]
        # update the bracket: price is increasing in sigma
        too_high = diff > 0
        hi[idx] = np.where(too_high, s_i, hi[idx])
        lo[idx] = np.where(~too_high, s_i, lo[idx])
        vega = _vega_bsm(S[idx], K[idx], T[idx], r[idx], q[idx], s_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = s_i - diff / vega
        inside = np.isfinite(newton) & (newton > lo[idx]) & (newton < hi[idx])
        sigma[idx] = np.where(inside, newton, 0.5 * (lo[idx] + hi[idx]))
        active[idx[done]] = False
    leftover = active & valid
    out[leftover] = sigma[leftover]                              # bracket collapsed without meeting the tolerance (e.g. price within rounding of the bound)
    out[~valid] = np.nan
    return out.reshape(shape)


def implied_vol_black76(price, F, K, T, r=0.0, call=True) -> np.ndarray:
    """Implied Black-76 volatility for options on forwards / futures (reduces to the BSM solver on the discounted forward)."""
    price, F, K, T, r = _arr(price, F, K, T, r)
    df = np.exp(-r * T)
    # Black-76 price = e^{-rT} x (BSM with r = q = 0 on the forward): invert the undiscounted price
    del df
    return implied_vol(price * np.exp(r * T), F, K, T, 0.0, 0.0, call)


def implied_vol_bachelier(price, F, K, T, r=0.0, call=True, tol: float = 1e-12) -> np.ndarray:
    """Implied normal (Bachelier) volatility by bisection on the monotone price (vectorised)."""
    price, F, K, T, r = _arr(price, F, K, T, r)
    shape = price.shape
    price, F, K, T, r = (np.array(x, dtype=float).ravel() for x in (price, F, K, T, r))
    lo, hi = np.zeros(price.shape), np.full(price.shape, 10.0 * np.maximum(np.abs(F), 1.0))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        diff = bachelier_price(F, K, T, r, mid, call) - price
        hi = np.where(diff > 0, mid, hi)
        lo = np.where(diff <= 0, mid, lo)
        if np.max(hi - lo) < tol:
            break
    out = 0.5 * (lo + hi)
    intrinsic = np.exp(-r * T) * (np.maximum(F - K, 0.0) if call else np.maximum(K - F, 0.0))
    out[(price <= intrinsic + 1e-14) | (T <= 0)] = np.nan
    return out.reshape(shape)


def implied_vol_american(price, S, K, T, r, q, call=False, steps=200, tol=1e-6) -> float:
    """Implied volatility of an AMERICAN option by root-finding on the binomial tree (scalar). Use for equity-index / ETF puts where early exercise matters."""
    from scipy import optimize

    from .pricing import binomial_price

    f = lambda s: binomial_price(S, K, T, r, q, s, call, True, steps) - price         # noqa: E731
    lo, hi = 1e-4, 5.0
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(optimize.brentq(f, lo, hi, xtol=tol))


__all__ = ["implied_vol", "implied_vol_black76", "implied_vol_bachelier", "implied_vol_american", "black76_price", "bsm_price"]
