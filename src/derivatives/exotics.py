"""Exotic options: the closed forms that exist, and a Monte Carlo engine for the rest under any of the platform's models.

Closed forms under Black-Scholes (each is also a check on the grid and Monte Carlo prices in ``derivatives.pde`` and below):

``digital_price``       cash-or-nothing ``e^{-rT} N(d2)`` and asset-or-nothing ``S e^{-qT} N(d1)``.
``barrier_price``       the eight continuously monitored barrier options of Reiner and Rubinstein (1991), assembled from four terms ``A, B, C, D`` (no rebate). A knock-in plus the matching
                        knock-out is the vanilla, which the tests use as a check on the table.
``perpetual_american``  the American option with no expiry: the value function is a power ``S^beta`` on the continuation side, ``beta`` the root of ``sigma^2 beta (beta - 1) / 2 +
                        (r - q) beta - r = 0``, and smooth pasting gives the exercise boundary ``K beta / (beta - 1)``. It is the limit of the finite-maturity boundary as ``T -> infinity``.
``asian_geometric``     an average of lognormal prices is lognormal: the geometric-average option has a Black-Scholes-like price, for any number of fixings or continuously.
``asian_levy``          Levy's (1992) moment-matched lognormal approximation to the arithmetic average (the arithmetic average has no closed form).

``mc_price`` is the general tool: simulate paths under a model (``levy.simulate_paths``: exact increments for the Levy models, the quadratic-exponential scheme for Heston), apply a payoff to
the whole path, discount and report the standard error. A *control variate* (a quantity correlated with the payoff whose expected value is known, such as the geometric-average option for an
arithmetic one) removes most of the variance. Barriers are monitored at the simulated dates, so they are missed between dates: ``bridge`` applies the Brownian-bridge crossing probability
(exact for a diffusion between two dates, available for Black-Scholes), and for every model the bias shrinks like ``1 / sqrt(steps)``.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import norm

from . import levy
from .pricing import bsm_price


# ------------------------------------------------------------------------------------------------------------------------------------ digital
def digital_price(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True, kind: str = "cash", payout: float = 1.0):
    """A digital option under Black-Scholes: ``kind='cash'`` pays ``payout`` when it finishes in the money, ``'asset'`` pays the asset."""
    if kind not in ("cash", "asset"):
        raise ValueError("kind must be 'cash' or 'asset'")
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sd
    d2 = d1 - sd
    sign = 1.0 if call else -1.0
    out = payout * np.exp(-r * T) * norm.cdf(sign * d2) if kind == "cash" else S * np.exp(-q * T) * norm.cdf(sign * d1)
    return out if out.ndim else float(out)


# ------------------------------------------------------------------------------------------------------------------------------------ barrier
def barrier_price(S: float, K: float, H: float, T: float, r: float, q: float, sigma: float, call: bool = True, kind: str = "down-out") -> float:
    """A continuously monitored barrier option under Black-Scholes, no rebate (Reiner and Rubinstein 1991). ``kind`` is ``'down-out'``, ``'down-in'``, ``'up-out'`` or ``'up-in'``; a barrier
    already crossed at the start leaves a knock-out worthless and a knock-in a vanilla."""
    if kind not in ("down-out", "down-in", "up-out", "up-in"):
        raise ValueError("kind must be 'down-out', 'down-in', 'up-out' or 'up-in'")
    down, knock_in = kind.startswith("down"), kind.endswith("in")
    if (down and S <= H) or (not down and S >= H):
        return float(bsm_price(S, K, T, r, q, sigma, call)) if knock_in else 0.0
    if T <= 0:
        pay = max((S - K) if call else (K - S), 0.0)
        return pay if knock_in is False else 0.0
    b = r - q
    sd = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2
    phi, eta = (1.0 if call else -1.0), (1.0 if down else -1.0)
    x1 = np.log(S / K) / sd + (1.0 + mu) * sd
    x2 = np.log(S / H) / sd + (1.0 + mu) * sd
    y1 = np.log(H ** 2 / (S * K)) / sd + (1.0 + mu) * sd
    y2 = np.log(H / S) / sd + (1.0 + mu) * sd
    fwd, disc = S * np.exp((b - r) * T), K * np.exp(-r * T)
    A = phi * fwd * norm.cdf(phi * x1) - phi * disc * norm.cdf(phi * x1 - phi * sd)
    B = phi * fwd * norm.cdf(phi * x2) - phi * disc * norm.cdf(phi * x2 - phi * sd)
    C = phi * fwd * (H / S) ** (2.0 * (mu + 1.0)) * norm.cdf(eta * y1) - phi * disc * (H / S) ** (2.0 * mu) * norm.cdf(eta * y1 - eta * sd)
    D = phi * fwd * (H / S) ** (2.0 * (mu + 1.0)) * norm.cdf(eta * y2) - phi * disc * (H / S) ** (2.0 * mu) * norm.cdf(eta * y2 - eta * sd)
    above = K > H
    if knock_in:
        if down:
            value = (C if above else A - B + D) if call else (B - C + D if above else A)
        else:
            value = (A if above else B - C + D) if call else (A - B + D if above else C)
    else:
        if down:
            value = (A - C if above else B - D) if call else (A - B + C - D if above else 0.0)
        else:
            value = (0.0 if above else A - B + C - D) if call else (B - D if above else A - C)
    return float(max(value, 0.0))


# ----------------------------------------------------------------------------------------------------------------------------- perpetual American
def perpetual_american(S: float, K: float, r: float, q: float, sigma: float, call: bool = False) -> dict:
    """The American call or put that never expires. Returns the price, the exercise boundary (``inf`` for a call on a non-dividend payer: it is never exercised early) and the exponent ``beta``."""
    if r <= 0:
        raise ValueError("a perpetual option needs a positive rate (with r = 0 a put is worth the strike and a call the asset)")
    b = r - q
    m = b - 0.5 * sigma ** 2
    root = np.sqrt(m ** 2 + 2.0 * r * sigma ** 2)
    if call:
        beta = (-m + root) / sigma ** 2                          # > 1: the value rises with S
        if beta <= 1.0 + 1e-12:
            return {"price": float(S), "boundary": float("inf"), "beta": 1.0}
        star = K * beta / (beta - 1.0)
        price = float(S - K) if S >= star else float((star - K) * (S / star) ** beta)
    else:
        beta = (-m - root) / sigma ** 2                          # < 0: the value falls with S
        star = K * beta / (beta - 1.0)
        price = float(K - S) if S <= star else float((K - star) * (S / star) ** beta)
    return {"price": price, "boundary": float(star), "beta": float(beta)}


# ---------------------------------------------------------------------------------------------------------------------------------------- Asian
def _geometric_moments(S, T, r, q, sigma, n_fixings):
    b = r - q
    if n_fixings is None:                                            # continuous averaging
        return np.log(S) + (b - 0.5 * sigma ** 2) * T / 2.0, sigma ** 2 * T / 3.0
    n = int(n_fixings)
    return (np.log(S) + (b - 0.5 * sigma ** 2) * T * (n + 1) / (2.0 * n), sigma ** 2 * T * (n + 1) * (2 * n + 1) / (6.0 * n ** 2))


def asian_geometric(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True, n_fixings: int | None = None) -> float:
    """A fixed-strike option on the geometric average of the price at ``n_fixings`` equally spaced dates ``T/n, 2T/n, ..., T`` (``None``: averaged continuously). The average is lognormal with
    ``m = ln S + (r - q - sigma^2/2) T (n + 1) / 2n`` and ``v = sigma^2 T (n + 1)(2n + 1) / 6 n^2``, so the price is a Black formula."""
    m, v = _geometric_moments(S, T, r, q, sigma, n_fixings)
    sd = np.sqrt(v)
    d1 = (m + v - np.log(K)) / sd
    d2 = d1 - sd
    if call:
        return float(np.exp(-r * T) * (np.exp(m + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(np.exp(-r * T) * (K * norm.cdf(-d2) - np.exp(m + 0.5 * v) * norm.cdf(-d1)))


def asian_levy(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True, n_fixings: int | None = None) -> float:
    """Levy's approximation to a fixed-strike option on the arithmetic average: a lognormal with the exact first two moments of the average, priced with Black's formula. Fast and close for
    moderate volatility (within a percent or two), and exact in the limit of no volatility."""
    b = r - q
    if n_fixings is None:
        m1 = S * (np.exp(b * T) - 1.0) / (b * T) if abs(b) > 1e-12 else S
        m2 = _levy_second_moment_continuous(S, T, b, sigma)
    else:
        n = int(n_fixings)
        t = T * np.arange(1, n + 1) / n
        m1 = float(np.mean(S * np.exp(b * t)))
        # E[A^2] = S^2 / n^2 * (Sum_i e^{(2b + s^2) t_i} + 2 Sum_{i<j} e^{(b + s^2) t_i + b t_j}): the double sum is a running total, so the cost is O(n), not O(n^2)
        below = np.cumsum(np.exp((b + sigma ** 2) * t)) - np.exp((b + sigma ** 2) * t)
        m2 = float(S ** 2 / n ** 2 * (np.sum(np.exp((2.0 * b + sigma ** 2) * t)) + 2.0 * np.sum(np.exp(b * t) * below)))
    v = np.log(m2 / m1 ** 2)
    sd = np.sqrt(v)
    mean_log = np.log(m1) - 0.5 * v
    d1 = (mean_log + v - np.log(K)) / sd
    d2 = d1 - sd
    if call:
        return float(np.exp(-r * T) * (m1 * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(np.exp(-r * T) * (K * norm.cdf(-d2) - m1 * norm.cdf(-d1)))


def _levy_second_moment_continuous(S, T, b, sigma) -> float:
    """``E[A^2]`` for the continuous arithmetic average ``A = (1/T) Int_0^T S_t dt`` of a geometric Brownian motion with drift ``b``: ``(2 S^2 / T^2) Int_0^T Int_0^s e^{b (s + u) + sigma^2 u} du ds``."""
    s2 = sigma ** 2
    c = b + s2                                          # the inner integral over u in [0, s] is (e^{c s} - 1) / c, the outer one over s follows
    if abs(c) < 1e-12 or abs(2.0 * b + s2) < 1e-12 or abs(b) < 1e-12:               # a degenerate drift: integrate numerically
        f = lambda u_, s_: np.exp(b * (s_ + u_) + s2 * u_)          # noqa: E731
        val = integrate.dblquad(f, 0.0, T, lambda s_: 0.0, lambda s_: s_)[0]
        return float(2.0 * S ** 2 * val / T ** 2)
    val = (np.exp((2.0 * b + s2) * T) - 1.0) / ((2.0 * b + s2) * c) - (np.exp(b * T) - 1.0) / (b * c)
    return float(2.0 * S ** 2 * val / T ** 2)


# ------------------------------------------------------------------------------------------------------------------------------- Monte Carlo
def mc_price(model: levy.Model, S: float, r: float, q: float, T: float, payoff, steps: int = 252, n_paths: int = 100_000, seed: int = 0, antithetic: bool = True,
             control: tuple | None = None) -> dict:
    """The price of a path-dependent payoff under ``model`` by Monte Carlo. ``payoff(paths)`` maps the ``(n_paths, steps + 1)`` array of spot paths (column 0 is the start) to the payoff at
    expiry of each path. ``control`` is ``(fn, mean)``: ``fn(paths)`` is a control variate with the known expected discounted value ``mean``; the price is corrected by the regression
    coefficient estimated from the paths. Returns the price, its standard error and the variance reduction achieved."""
    paths = levy.simulate_paths(model, S, r, q, T, steps, n_paths, seed, antithetic)
    disc = np.exp(-r * T)
    raw = disc * np.asarray(payoff(paths), dtype=float)
    out = {"price": float(raw.mean()), "se": float(raw.std(ddof=1) / np.sqrt(len(raw))), "n_paths": int(len(raw)), "steps": int(steps), "variance_reduction": 1.0}
    if control is not None:
        fn, mean = control
        c = disc * np.asarray(fn(paths), dtype=float)
        cov = np.cov(raw, c)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        adj = raw - beta * (c - mean)
        out.update(price=float(adj.mean()), se=float(adj.std(ddof=1) / np.sqrt(len(adj))), variance_reduction=float(raw.var(ddof=1) / max(adj.var(ddof=1), 1e-300)), beta=float(beta))
    return out


def asian_arithmetic_mc(model: levy.Model, S: float, K: float, T: float, r: float, q: float, call: bool = True, n_fixings: int = 52, n_paths: int = 100_000, seed: int = 0,
                        sigma: float | None = None) -> dict:
    """A fixed-strike option on the arithmetic average of ``n_fixings`` equally spaced prices, by Monte Carlo. Under Black-Scholes (give ``sigma``, or use ``levy.BlackScholes``) the
    geometric-average option, whose price is known exactly, is the control variate, and it removes about 99.9% of the variance; under any other model the control is the average price itself
    (its mean is the average of the forwards)."""
    sign = 1.0 if call else -1.0
    idx = np.arange(1, n_fixings + 1) * 1                    # the fixings are the simulated dates 1..n

    def avg(paths):
        return paths[:, idx].mean(axis=1)

    def payoff(paths):
        return np.maximum(sign * (avg(paths) - K), 0.0)

    sig = sigma if sigma is not None else (model.params["sigma"] if isinstance(model, levy.BlackScholes) else None)
    if isinstance(model, levy.BlackScholes) and sig is not None:
        geo_price = asian_geometric(S, K, T, r, q, sig, call, n_fixings)
        control = (lambda p: np.maximum(sign * (np.exp(np.log(p[:, idx]).mean(axis=1)) - K), 0.0), geo_price)   # noqa: E731
    else:
        t = T * np.arange(1, n_fixings + 1) / n_fixings
        control = (lambda p: avg(p), float(np.exp(-r * T) * np.mean(S * np.exp((r - q) * t))))                   # noqa: E731
    res = mc_price(model, S, r, q, T, payoff, steps=n_fixings, n_paths=n_paths, seed=seed, antithetic=True, control=control)
    res["control"] = "geometric-average option" if isinstance(model, levy.BlackScholes) and sig is not None else "average price"
    return res


def barrier_mc(model: levy.Model, S: float, K: float, H: float, T: float, r: float, q: float, call: bool = True, kind: str = "down-out", steps: int = 252, n_paths: int = 100_000,
               seed: int = 0, bridge: bool = True) -> dict:
    """A barrier option by Monte Carlo, monitored at the ``steps`` simulated dates. For Black-Scholes ``bridge=True`` multiplies each path's survival by the Brownian-bridge probability
    ``1 - exp(-2 ln(S_i/H) ln(S_{i+1}/H) / (sigma^2 dt))`` of not touching the barrier between dates, which removes the bias of discrete monitoring; for other models (jumps, stochastic
    variance) only the monitored dates count, so a knock-out is priced a little high."""
    if kind not in ("down-out", "down-in", "up-out", "up-in"):
        raise ValueError("kind must be 'down-out', 'down-in', 'up-out' or 'up-in'")
    sign = 1.0 if call else -1.0
    down, knock_in = kind.startswith("down"), kind.endswith("in")
    use_bridge = bool(bridge and isinstance(model, levy.BlackScholes))
    sigma = model.params["sigma"] if isinstance(model, levy.BlackScholes) else None
    dt = T / steps

    def payoff(paths):
        vanilla = np.maximum(sign * (paths[:, -1] - K), 0.0)
        if down:
            alive = (paths.min(axis=1) > H) if S > H else np.zeros(len(paths), dtype=bool)
        else:
            alive = (paths.max(axis=1) < H) if S < H else np.zeros(len(paths), dtype=bool)
        weight = alive.astype(float)
        if use_bridge and alive.any():
            a, b = np.log(paths[:, :-1] / H), np.log(paths[:, 1:] / H)
            cross = np.exp(-2.0 * a * b / (sigma ** 2 * dt))                  # probability of touching in each step, when both ends are on the same side
            cross = np.where(a * b > 0, cross, 1.0)
            weight = weight * np.prod(1.0 - np.clip(cross, 0.0, 1.0), axis=1)
        return vanilla * ((1.0 - weight) if knock_in else weight)

    res = mc_price(model, S, r, q, T, payoff, steps=steps, n_paths=n_paths, seed=seed, antithetic=True)
    res["bridge"] = use_bridge
    return res
