"""Option sensitivities (the Greeks): closed-form Black-Scholes-Merton and Black-76, central-difference Greeks for any pricer, and portfolio aggregation.

Closed-form BSM (``q`` dividend yield), per UNIT of the underlying; theta is per year (divide by 365 for a calendar day), vega per 1.00 of volatility (divide by 100
for one vol point), rho per 1.00 of rate:

* ``delta``  ``e^{-qT} N(d1)`` (call), ``-e^{-qT} N(-d1)`` (put): the hedge ratio.
* ``gamma``  ``e^{-qT} n(d1) / (S sigma sqrt T)``: how fast delta changes; the P&L of a delta-hedged position is ``~ 1/2 Gamma S^2 (realised^2 - implied^2) dt``.
* ``vega``   ``S e^{-qT} n(d1) sqrt T``;  ``theta`` the time decay;  ``rho`` the rate sensitivity.
* second order: ``vanna`` (``d delta / d sigma``), ``volga`` / vomma (``d vega / d sigma``), ``charm`` (``d delta / d t``), ``speed``.

``numerical_greeks`` differentiates any pricing function (a tree, Heston, a Monte Carlo with common random numbers) by central differences, which is how Greeks are
obtained when no formula exists; ``portfolio_greeks`` adds position-weighted Greeks and reports dollar gamma (the P&L from a 1% move) and the delta-neutral hedge.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .pricing import _arr, bsm_d1_d2


def bsm_greeks(S, K, T, r, q, sigma, call=True) -> dict:
    """All first- and second-order BSM Greeks as a dict of arrays (broadcast over the inputs)."""
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    d1, d2 = bsm_d1_d2(S, K, T, r, q, sigma)
    sq = np.sqrt(T)
    dq, df = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    sign = 1.0 if call else -1.0
    delta = dq * norm.cdf(d1) if call else -dq * norm.cdf(-d1)
    gamma = dq * pdf / (S * sigma * sq)
    vega = S * dq * pdf * sq
    theta_common = -S * dq * pdf * sigma / (2 * sq)
    theta = theta_common - sign * r * K * df * norm.cdf(sign * d2) + sign * q * S * dq * norm.cdf(sign * d1)
    rho = sign * K * T * df * norm.cdf(sign * d2)
    vanna = -dq * pdf * d2 / sigma
    volga = vega * d1 * d2 / sigma
    charm = (q * dq * norm.cdf(d1) if call else -q * dq * norm.cdf(-d1)) - dq * pdf * (2 * (r - q) * T - d2 * sigma * sq) / (2 * T * sigma * sq)     # d delta / d (calendar time)
    speed = -gamma / S * (d1 / (sigma * sq) + 1.0)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho, "vanna": vanna, "volga": volga, "charm": charm, "speed": speed}


def black76_greeks(F, K, T, r, sigma, call=True) -> dict:
    """Greeks of an option on a forward with respect to the FORWARD price (delta, gamma), volatility (vega) and time (theta)."""
    F, K, T, r, sigma = _arr(F, K, T, r, sigma)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    pdf = norm.pdf(d1)
    price = df * (F * norm.cdf(d1) - K * norm.cdf(d2)) if call else df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return {"delta": df * norm.cdf(d1) if call else -df * norm.cdf(-d1), "gamma": df * pdf / (F * sd), "vega": df * F * pdf * np.sqrt(T),
            "theta": -df * F * pdf * sigma / (2 * np.sqrt(T)) + r * price, "price": price}


def numerical_greeks(pricer: Callable[..., float], S: float, sigma: float, T: float, r: float, rel_bump: float = 1e-3, **fixed) -> dict:
    """Central-difference delta, gamma, vega, theta and rho of ``pricer(S=, sigma=, T=, r=, **fixed)``. Use common random numbers inside a Monte Carlo pricer."""
    p = lambda **kw: float(pricer(**{"S": S, "sigma": sigma, "T": T, "r": r, **fixed, **kw}))      # noqa: E731
    hS, hv, hT, hr = S * rel_bump, max(sigma * rel_bump, 1e-5), max(T * 1e-3, 1e-4), 1e-4
    up, mid, dn = p(S=S + hS), p(), p(S=S - hS)
    return {"price": mid, "delta": (up - dn) / (2 * hS), "gamma": (up - 2 * mid + dn) / hS ** 2, "vega": (p(sigma=sigma + hv) - p(sigma=sigma - hv)) / (2 * hv),
            "theta": -(p(T=T + hT) - p(T=max(T - hT, 1e-8))) / (hT + min(hT, T - 1e-8)), "rho": (p(r=r + hr) - p(r=r - hr)) / (2 * hr)}


def portfolio_greeks(positions: pd.DataFrame, spot: float, multiplier: float = 100.0) -> pd.Series:
    """Aggregate the Greeks of a book. ``positions`` needs ``quantity`` (contracts, signed), and per-option ``delta``, ``gamma``, ``vega``, ``theta`` (per unit of underlying,
    as returned by ``bsm_greeks``). Returns share-equivalent delta, gamma, dollar gamma (P&L of a 1% move in the underlying), vega per vol point, theta per calendar day,
    and the number of underlying units that would make the book delta-neutral."""
    q = positions["quantity"] * multiplier
    delta = float((q * positions["delta"]).sum())
    gamma = float((q * positions["gamma"]).sum())
    return pd.Series({"delta_shares": delta, "gamma": gamma, "dollar_gamma_1pct": 0.5 * gamma * (0.01 * spot) ** 2, "vega_per_volpoint": float((q * positions["vega"]).sum()) / 100.0,
                      "theta_per_day": float((q * positions["theta"]).sum()) / 365.0, "hedge_units": -delta, "delta_dollars": delta * spot})


def pnl_explain(delta, gamma, vega, theta, d_spot, d_vol, dt_days, rho=0.0, d_rate=0.0) -> dict:
    """Taylor expansion of an option position's P&L over one step: ``delta dS + 1/2 gamma dS^2 + vega d(sigma) + theta dt + rho dr``. The unexplained remainder is the
    higher-order and cross terms (vanna, volga, charm) plus anything the model misses."""
    parts = {"delta": delta * d_spot, "gamma": 0.5 * gamma * d_spot ** 2, "vega": vega * d_vol, "theta": theta * dt_days / 365.0, "rho": rho * d_rate}
    parts["total_explained"] = float(sum(parts.values()))
    return parts
