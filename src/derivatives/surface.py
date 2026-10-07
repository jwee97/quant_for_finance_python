"""Implied-volatility surfaces: SVI slices, the arbitrage-free SSVI surface, local volatility, risk-neutral densities and variance swaps.

Work in **total implied variance** ``w(k, T) = sigma_imp(k, T)^2 T`` against **log-moneyness** ``k = ln(K / F_T)``: smiles in these coordinates are comparable across
expiries, and no-arbitrage conditions are simple.

* **SVI** (Gatheral 2004): ``w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + s^2)]``. ``fit_svi_slice`` calibrates one expiry by constrained least squares; ``svi_butterfly_density`` is
  Gatheral's ``g(k)`` whose non-negativity is the no-butterfly-arbitrage condition (``g(k) / sqrt(2 pi w) exp(-d2^2/2)`` is the risk-neutral density of ``k``).
* **SSVI** (Gatheral & Jacquier 2014): a whole surface from the ATM total-variance curve ``theta_t`` and three parameters,
  ``w(k, theta) = theta/2 [1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)]``, ``phi(theta) = eta / (theta^gamma (1 + theta)^{1 - gamma})``. It is free of butterfly arbitrage when
  ``theta phi (1 + |rho|) < 4`` and ``theta phi^2 (1 + |rho|) <= 4``, and of calendar arbitrage when ``theta`` is increasing and ``theta phi`` is non-decreasing.
* **Dupire local volatility** from the surface by Gatheral's formula in ``w``: ``sigma_loc^2 = (dw/dT) / [1 - (k/w) w_k + 1/4 (-1/4 - 1/w + k^2/w^2) w_k^2 + 1/2 w_kk]``.
* **Variance swap** fair strike by static replication over a strip of out-of-the-money options (Demeterfi, Derman, Kamal & Zou 1999) and the **VIX-style index** (the CBOE
  methodology applied to two expiries interpolated to a constant 30 days).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from .pricing import bsm_price


# ------------------------------------------------------------------------------------------------------------------------- SVI
def svi_total_variance(k, a, b, rho, m, s):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def svi_butterfly_density(k, a, b, rho, m, s):
    """Gatheral's ``g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2``. Non-negative everywhere iff the slice admits no butterfly arbitrage."""
    k = np.asarray(k, dtype=float)
    w = svi_total_variance(k, a, b, rho, m, s)
    root = np.sqrt((k - m) ** 2 + s ** 2)
    w1 = b * (rho + (k - m) / root)
    w2 = b * s ** 2 / root ** 3
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - w1 ** 2 / 4.0 * (1.0 / w + 0.25) + w2 / 2.0


@dataclass
class SVISlice:
    a: float
    b: float
    rho: float
    m: float
    s: float
    T: float
    rmse: float = float("nan")

    def w(self, k):
        return svi_total_variance(k, self.a, self.b, self.rho, self.m, self.s)

    def iv(self, k):
        return np.sqrt(np.maximum(self.w(k), 1e-12) / self.T)

    def density_g(self, k):
        return svi_butterfly_density(k, self.a, self.b, self.rho, self.m, self.s)

    def arbitrage_free(self, k_range=(-3.0, 3.0), n=601) -> bool:
        k = np.linspace(*k_range, n)
        return bool(np.all(self.density_g(k) >= -1e-8) and np.all(self.w(k) > 0))

    @property
    def params(self) -> tuple:
        return (self.a, self.b, self.rho, self.m, self.s)


def fit_svi_slice(k, iv, T, weights=None, n_starts: int = 12, seed: int = 0) -> SVISlice:
    """Least-squares SVI fit to ``(k, iv)`` at expiry ``T`` with the parameter constraints ``b >= 0``, ``|rho| < 1``, ``s > 0`` and ``a + b s sqrt(1 - rho^2) >= 0`` (minimum variance
    non-negative). Several starts and a penalty on butterfly arbitrage (negative ``g``) keep the result a valid slice."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    w_mkt = iv ** 2 * T
    wt = np.ones_like(k) if weights is None else np.asarray(weights, float)
    kmin, kmax = k.min(), k.max()
    grid = np.linspace(kmin - 0.5, kmax + 0.5, 121)

    def unpack(p):
        a, b, rho, m, s = p
        return a, b, rho, m, s

    def resid(p):
        a, b, rho, m, s = unpack(p)
        fit = svi_total_variance(k, a, b, rho, m, s)
        g = svi_butterfly_density(grid, a, b, rho, m, s)
        pen = np.clip(-g, 0, None).sum() * 0.05
        floor = max(0.0, -(a + b * s * np.sqrt(1 - rho ** 2))) * 10.0
        return np.concatenate([(fit - w_mkt) * np.sqrt(wt), [pen + floor]])

    rng = np.random.default_rng(seed)
    atm = float(np.interp(0.0, k, w_mkt)) if kmin < 0 < kmax else float(w_mkt.mean())
    best = None
    for i in range(n_starts):
        p0 = [atm * (0.5 + 0.2 * i / n_starts), 0.1 + 0.4 * rng.random(), -0.8 + 1.4 * rng.random(), (rng.random() - 0.5) * 0.2, 0.05 + 0.3 * rng.random()]
        sol = optimize.least_squares(resid, p0, bounds=([-1.0, 1e-6, -0.999, -1.5, 1e-4], [max(5 * w_mkt.max(), 1.0), 10.0, 0.999, 1.5, 3.0]), xtol=1e-12, ftol=1e-12, max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    a, b, rho, m, s = best.x
    err = svi_total_variance(k, a, b, rho, m, s) - w_mkt
    return SVISlice(a, b, rho, m, s, T, float(np.sqrt(np.mean((np.sqrt(np.maximum(svi_total_variance(k, a, b, rho, m, s), 1e-12) / T) - iv) ** 2))))


# ------------------------------------------------------------------------------------------------------------------------ SSVI
def ssvi_phi(theta, eta, gamma):
    theta = np.asarray(theta, dtype=float)
    return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_total_variance(k, theta, rho, eta, gamma):
    phi = ssvi_phi(theta, eta, gamma)
    k = np.asarray(k, dtype=float)
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2))


def ssvi_arbitrage_free(thetas, rho, eta, gamma) -> dict:
    """Gatheral-Jacquier sufficient conditions on the parameters and the ATM variance curve ``thetas`` (sorted by expiry)."""
    thetas = np.asarray(thetas, float)
    phi = ssvi_phi(thetas, eta, gamma)
    butterfly = bool(np.all(thetas * phi * (1 + abs(rho)) < 4.0) and np.all(thetas * phi ** 2 * (1 + abs(rho)) <= 4.0))
    calendar = bool(np.all(np.diff(thetas) >= 0) and np.all(np.diff(thetas * phi) >= -1e-12) and abs(rho) < 1)
    return {"butterfly": butterfly, "calendar": calendar, "free": butterfly and calendar}


class SSVISurface:
    """A whole surface from ``(rho, eta, gamma)`` and the ATM total-variance curve ``theta(T)`` (linear interpolation in total variance between the given expiries, flat
    forward variance beyond them), so ``iv(k, T)`` exists for every expiry."""

    def __init__(self, expiries, thetas, rho, eta, gamma):
        order = np.argsort(expiries)
        self.expiries = np.asarray(expiries, float)[order]
        self.thetas = np.asarray(thetas, float)[order]
        self.rho, self.eta, self.gamma = float(rho), float(eta), float(gamma)

    def theta(self, T):
        T = np.asarray(T, dtype=float)
        th = np.interp(T, np.r_[0.0, self.expiries], np.r_[0.0, self.thetas])
        slope_last = (self.thetas[-1] - self.thetas[-2]) / (self.expiries[-1] - self.expiries[-2]) if len(self.thetas) > 1 else self.thetas[-1] / self.expiries[-1]
        return np.where(T > self.expiries[-1], self.thetas[-1] + slope_last * (T - self.expiries[-1]), th)

    def w(self, k, T):
        return ssvi_total_variance(k, self.theta(T), self.rho, self.eta, self.gamma)

    def iv(self, k, T):
        T = np.asarray(T, dtype=float)
        return np.sqrt(np.maximum(self.w(k, T), 1e-14) / T)

    def arbitrage_free(self) -> dict:
        return ssvi_arbitrage_free(self.thetas, self.rho, self.eta, self.gamma)


def fit_ssvi(quotes: pd.DataFrame, seed: int = 0) -> SSVISurface:
    """Fit an SSVI surface to OTM quotes (``k``, ``iv``, ``T``). The ATM total variance at each expiry is taken from an SVI fit of that slice (forcing it to be increasing), then
    ``(rho, eta, gamma)`` are calibrated by least squares in implied-volatility space under the arbitrage-free parameter conditions."""
    expiries, thetas = [], []
    for T, g in quotes.groupby("T"):
        if len(g) < 5 or T <= 0:
            continue
        sl = fit_svi_slice(g["k"].to_numpy(), g["iv"].to_numpy(), float(T), n_starts=6, seed=seed)
        expiries.append(float(T))
        thetas.append(float(sl.w(0.0)))
    thetas = np.maximum.accumulate(np.asarray(thetas))
    expiries = np.asarray(expiries)
    theta_of = dict(zip(expiries, thetas))
    data = quotes[quotes["T"].isin(expiries)]
    K, IV, TT = data["k"].to_numpy(), data["iv"].to_numpy(), data["T"].to_numpy()
    TH = np.array([theta_of[t] for t in TT])

    def resid(p):
        rho, eta, gamma = p
        w = ssvi_total_variance(K, TH, rho, eta, gamma)
        iv = np.sqrt(np.maximum(w, 1e-14) / TT)
        phi = ssvi_phi(expiries * 0 + thetas, eta, gamma)
        pen = 10.0 * (np.clip(thetas * phi * (1 + abs(rho)) - 4.0, 0, None).sum() + np.clip(thetas * phi ** 2 * (1 + abs(rho)) - 4.0, 0, None).sum())
        pen += 10.0 * np.clip(-np.diff(thetas * phi), 0, None).sum() if len(thetas) > 1 else 0.0
        return np.concatenate([iv - IV, [pen]])

    best = None
    for x0 in ([-0.5, 0.8, 0.4], [-0.7, 1.5, 0.5], [-0.2, 0.4, 0.3], [-0.9, 2.5, 0.6]):
        sol = optimize.least_squares(resid, x0, bounds=([-0.999, 1e-3, 1e-3], [0.999, 8.0, 0.999]), xtol=1e-12, ftol=1e-12)
        if best is None or sol.cost < best.cost:
            best = sol
    rho, eta, gamma = best.x
    return SSVISurface(expiries, thetas, rho, eta, gamma)


# ------------------------------------------------------------------------------------------------------- surface-derived objects
def _derivatives(w_fun, k, T, hk=1e-3, hT=1e-3):
    w0 = w_fun(k, T)
    wk = (w_fun(k + hk, T) - w_fun(k - hk, T)) / (2 * hk)
    wkk = (w_fun(k + hk, T) - 2 * w0 + w_fun(k - hk, T)) / hk ** 2
    lo = np.maximum(T - hT, 1e-6)
    wT = (w_fun(k, T + hT) - w_fun(k, lo)) / (T + hT - lo)
    return w0, wk, wkk, wT


def local_volatility(w_fun, k, T) -> np.ndarray:
    """Dupire local volatility from a total-variance function ``w_fun(k, T)`` (Gatheral 2006): ``sigma_loc^2 = w_T / [1 - k w_k / w + 1/4 (-1/4 - 1/w + k^2/w^2) w_k^2 + w_kk / 2]``.
    NaN where the denominator is not positive (the surface admits butterfly arbitrage there)."""
    k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
    w, wk, wkk, wT = _derivatives(w_fun, k, T)
    denom = 1.0 - k * wk / w + 0.25 * (-0.25 - 1.0 / w + k ** 2 / w ** 2) * wk ** 2 + 0.5 * wkk
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(denom > 1e-10, wT / denom, np.nan)
    return np.sqrt(np.where(var > 0, var, np.nan))


def risk_neutral_density(w_fun, T: float, k_grid=None) -> pd.Series:
    """Risk-neutral density of the log-moneyness ``k`` implied by a surface: ``q(k) = g(k) / sqrt(2 pi w) exp(-d2^2 / 2)`` with ``d2 = -k / sqrt(w) - sqrt(w) / 2`` (the density of
    ``ln(S_T / F)``, in the same variable). Integrates to one for an arbitrage-free slice."""
    k = np.linspace(-1.5, 1.5, 601) if k_grid is None else np.asarray(k_grid, float)
    w, wk, wkk, _ = _derivatives(lambda kk, tt: w_fun(kk, tt), k, T)
    g = (1.0 - k * wk / (2 * w)) ** 2 - wk ** 2 / 4.0 * (1.0 / w + 0.25) + wkk / 2.0
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2.0
    return pd.Series(g / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * d2 ** 2), index=pd.Index(k, name="k"))


def variance_swap_strike(F: float, T: float, r: float, strikes, otm_prices) -> float:
    """Fair variance-swap strike (annualised VARIANCE) by static replication over a strip of OTM option prices (puts below the forward, calls above) at ONE expiry:
    ``K_var = (2 e^{rT} / T) sum Q(K_i) dK_i / K_i^2  -  (1/T) (F/K0 - 1)^2`` with ``K0`` the first strike below ``F`` (CBOE / Demeterfi et al.)."""
    K = np.asarray(strikes, float)
    Q = np.asarray(otm_prices, float)
    order = np.argsort(K)
    K, Q = K[order], Q[order]
    dK = np.gradient(K)
    k0 = K[K <= F].max()
    return float((2.0 * np.exp(r * T) / T) * np.sum(Q * dK / K ** 2) - (F / k0 - 1.0) ** 2 / T)


def vix_style_index(front: pd.DataFrame, nxt: pd.DataFrame, target_days: float = 30.0) -> float:
    """CBOE-methodology volatility index from two expiries' quotes (columns ``strike``, ``right``, ``bid``, ``ask``, ``T``, ``F``, ``rate``): the variance-swap strike of each
    expiry (using mid prices, stopping at two consecutive zero bids), time-interpolated to ``target_days`` and expressed as 100 x volatility."""
    def var_of(g):
        T, F, r = float(g["T"].iloc[0]), float(g["F"].iloc[0]), float(g["rate"].iloc[0])
        K0 = g.loc[g["strike"] <= F, "strike"].max()
        puts = g[(g["right"] == "P") & (g["strike"] < K0) & (g["bid"] > 0)].sort_values("strike", ascending=False)
        calls = g[(g["right"] == "C") & (g["strike"] > K0) & (g["bid"] > 0)].sort_values("strike")
        atm = g[g["strike"] == K0]
        q0 = atm["mid"].mean() if "mid" in atm else 0.5 * (atm["bid"] + atm["ask"]).mean()
        strikes = np.r_[puts["strike"].to_numpy()[::-1], K0, calls["strike"].to_numpy()]
        prices = np.r_[0.5 * (puts["bid"] + puts["ask"]).to_numpy()[::-1], q0, 0.5 * (calls["bid"] + calls["ask"]).to_numpy()]
        dK = np.gradient(strikes)
        return T, float((2.0 / T) * np.exp(r * T) * np.sum(prices * dK / strikes ** 2) - (F / K0 - 1.0) ** 2 / T)

    T1, v1 = var_of(front)
    T2, v2 = var_of(nxt)
    t = target_days / 365.0
    w1, w2 = (T2 - t) / (T2 - T1), (t - T1) / (T2 - T1)
    return float(100.0 * np.sqrt(max((T1 * v1 * w1 + T2 * v2 * w2) / t, 0.0)))


def smile_metrics(w_fun, T: float, delta_levels=(0.25,), F: float = 1.0, r: float = 0.0) -> pd.Series:
    """ATM vol, 25-delta risk reversal (call vol minus put vol) and butterfly (average of the wings minus ATM), skew slope ``d iv / d k`` at the money."""
    iv = lambda kk: np.sqrt(np.maximum(w_fun(kk, T), 1e-14) / T)         # noqa: E731
    atm = float(iv(0.0))
    out = {"atm_vol": atm, "skew_slope": float((iv(0.01) - iv(-0.01)) / 0.02)}
    for d in delta_levels:
        # strike with BS delta = d for the call: solve k by root-finding on the smile
        def call_delta_gap(kk, target=d):
            s = float(iv(kk))
            d1 = (-kk + 0.5 * s ** 2 * T) / (s * np.sqrt(T))
            return norm.cdf(d1) - target

        def put_delta_gap(kk, target=d):
            s = float(iv(kk))
            d1 = (-kk + 0.5 * s ** 2 * T) / (s * np.sqrt(T))
            return norm.cdf(d1) - 1.0 + target

        kc = optimize.brentq(call_delta_gap, -0.0, 3.0) if call_delta_gap(0.0) * call_delta_gap(3.0) < 0 else np.nan
        kp = optimize.brentq(put_delta_gap, -3.0, 0.0) if put_delta_gap(-3.0) * put_delta_gap(0.0) < 0 else np.nan
        if np.isfinite(kc) and np.isfinite(kp):
            vc, vp = float(iv(kc)), float(iv(kp))
            out[f"rr_{int(d * 100)}d"] = vc - vp
            out[f"bf_{int(d * 100)}d"] = 0.5 * (vc + vp) - atm
    return pd.Series(out)


def price_from_surface(S, K, T, r, q, w_fun, call=True):
    """Black-Scholes-Merton price with the surface's own implied vol at each strike and expiry (log-moneyness against the forward)."""
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    F = S * np.exp((r - q) * T)
    k = np.log(K / F)
    vol = np.sqrt(np.maximum(w_fun(k, T), 1e-14) / T)
    return bsm_price(S, K, T, r, q, vol, call)

