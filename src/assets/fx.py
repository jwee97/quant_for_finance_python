"""Foreign exchange: quoting conventions, forward points, carry, cross rates, a synthetic G10-style market and the standard FX factors.

Conventions. A pair ``EURUSD = 1.10`` is the number of USD per EUR (base = EUR, quote = USD). A position long the base currency funded in the quote currency earns the spot return
plus the carry ``(r_base - r_quote) dt`` (covered interest parity makes the forward ``F = S e^{(r_quote - r_base) T}``, so the forward discount is the interest differential).

* ``forward_rate`` / ``forward_points`` / ``implied_rate_differential``: covered interest parity.
* ``excess_return``: the daily total return on a long-base, short-quote position: ``S_t / S_{t-1} - 1 + (r_base - r_quote) dt``.
* ``cross_rate`` / ``triangular_gap``: synthetic crosses and the triangular-arbitrage check.
* ``fama_regression``: the forward-premium test (Fama 1984): regress the realised depreciation on the forward discount. Uncovered interest parity says the slope is 1; empirically it is
  near 0 or negative, which is why a carry trade (long high yield, short low yield) has earned a premium (Lustig, Roussanov & Verdelhan 2011).
* ``synthetic_fx_market``: currencies with mean-reverting interest rates, spot changes whose expected value offsets only a fraction ``uip_beta`` of the interest differential (so carry earns
  ``(1 - uip_beta)`` of the differential on average), and a crash factor: sudden depreciations of the high-yielders when global volatility spikes, the risk the carry premium pays for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats.regression import ols


def forward_rate(spot, r_quote, r_base, T):
    """Covered interest parity forward for quote-per-base spot: ``S e^{(r_quote - r_base) T}``."""
    return np.asarray(spot, float) * np.exp((np.asarray(r_quote, float) - np.asarray(r_base, float)) * np.asarray(T, float))


def forward_points(spot, r_quote, r_base, T, pip: float = 1e-4):
    """The forward minus the spot in pips (``pip`` = 1e-4, or 1e-2 for JPY pairs)."""
    return (forward_rate(spot, r_quote, r_base, T) - np.asarray(spot, float)) / pip


def implied_rate_differential(spot, forward, T):
    """``r_quote - r_base = ln(F / S) / T``: the interest differential an outright forward implies."""
    return np.log(np.asarray(forward, float) / np.asarray(spot, float)) / np.asarray(T, float)


def excess_return(spot: pd.Series, r_base: pd.Series, r_quote: pd.Series) -> pd.Series:
    """Daily return of the long-base position funded in the quote currency, including carry (rates are annual and continuously compounded, 252 trading days)."""
    carry = (r_base.reindex(spot.index).shift(1) - r_quote.reindex(spot.index).shift(1)) / 252.0
    return spot.pct_change() + carry


def cross_rate(a_per_usd: pd.Series, b_per_usd: pd.Series) -> pd.Series:
    """The A-per-B rate implied by two USD quotes (both quoted as units of the currency per 1 USD): ``a / b``."""
    return a_per_usd / b_per_usd


def triangular_gap(eurusd: pd.Series, usdjpy: pd.Series, eurjpy: pd.Series) -> pd.Series:
    """Log gap between a quoted EURJPY and the synthetic ``EURUSD x USDJPY``: zero without triangular arbitrage; in real data a few basis points of spread noise."""
    return np.log(eurjpy / (eurusd * usdjpy))


def fama_regression(spot: pd.Series, r_base: pd.Series, r_quote: pd.Series, horizon: int = 21) -> dict:
    """Fama (1984): ``ln(S_{t+h}/S_t) = a + b (r_quote - r_base) h/252 + e``. Uncovered interest parity: ``b = 1``. Overlapping observations, so HAC standard errors with ``h`` lags."""
    forward_discount = (r_quote - r_base).reindex(spot.index) * horizon / 252.0
    dep = np.log(spot.shift(-horizon) / spot)
    res = ols(dep, forward_discount.rename("fd").to_frame(), "HAC", lags=horizon)
    return {"beta": float(res.params["fd"]), "se": float(res.bse["fd"]), "t_vs_one": float((res.params["fd"] - 1.0) / res.bse["fd"]), "t_vs_zero": float(res.tvalues["fd"]), "nobs": res.nobs}


def synthetic_fx_market(n_days: int = 2500, n_currencies: int = 8, uip_beta: float = 0.3, crash_intensity: float = 1.5, seed: int = 0, start: str = "2010-01-04") -> dict:
    """Simulate ``n_currencies`` against USD. Rates ``r_i`` are Ornstein-Uhlenbeck around a currency-specific level; the spot of currency ``i`` (USD per unit) has expected log
    change ``-uip_beta (r_i - r_USD) dt``. Under uncovered interest parity (``uip_beta = 1``) the carry trade earns nothing; with ``uip_beta < 1`` it earns ``(1 - uip_beta)`` of the
    differential. A global-volatility factor ``v_t`` (CIR) scales the shocks, and a Poisson crash factor depreciates currencies in proportion to their rate level. Returns spot, rates
    and the excess returns, plus the true parameters."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    idx = pd.bdate_range(start, periods=n_days)
    names = [f"C{i + 1}" for i in range(n_currencies)]
    level = np.linspace(-0.01, 0.07, n_currencies) + rng.normal(0, 0.003, n_currencies)
    r_usd = np.zeros(n_days)
    r_usd[0] = 0.02
    rates = np.zeros((n_days, n_currencies))
    rates[0] = level
    for t in range(1, n_days):
        r_usd[t] = r_usd[t - 1] + 1.0 * (0.025 - r_usd[t - 1]) * dt + 0.004 * np.sqrt(dt) * rng.standard_normal()
        rates[t] = rates[t - 1] + 0.8 * (level - rates[t - 1]) * dt + 0.006 * np.sqrt(dt) * rng.standard_normal(n_currencies)
    v = np.empty(n_days)
    v[0] = 1.0
    for t in range(1, n_days):
        v[t] = max(v[t - 1] + 4.0 * (1.0 - v[t - 1]) * dt + 0.9 * np.sqrt(max(v[t - 1], 0) * dt) * rng.standard_normal(), 0.05)
    base_vol = 0.08
    usd_factor = rng.standard_normal(n_days)
    beta_usd = rng.uniform(0.6, 1.2, n_currencies)
    shocks = base_vol * np.sqrt(dt * v)[:, None] * (0.6 * usd_factor[:, None] * beta_usd + 0.8 * rng.standard_normal((n_days, n_currencies)))
    intensity = crash_intensity * (v / v.mean())                                                   # crashes cluster when global volatility is high
    crash = rng.random(n_days) < intensity * dt
    exposure = np.clip((rates - r_usd[:, None]) / 0.04, 0.0, 2.0)                                  # only high yielders crash against the dollar
    size = rng.uniform(0.015, 0.04, n_days)
    crash_size = -(crash * size)[:, None] * exposure
    compensator = (intensity * dt * 0.0275)[:, None] * exposure                                    # E[size] = 2.75%: the drift gives it back, so the expected depreciation is exactly uip_beta x differential
    drift = -uip_beta * (rates - r_usd[:, None]) * dt + compensator
    log_spot = np.cumsum(drift + shocks + crash_size, axis=0)
    spot = pd.DataFrame(np.exp(log_spot), index=idx, columns=names)
    rate_frame = pd.DataFrame(rates, index=idx, columns=names)
    usd = pd.Series(r_usd, index=idx, name="USD")
    excess = pd.DataFrame({c: excess_return(spot[c], rate_frame[c], usd) for c in names})
    return {"spot": spot, "rates": rate_frame, "usd_rate": usd, "excess_returns": excess, "uip_beta": uip_beta, "volatility_factor": pd.Series(v, index=idx)}


def carry_portfolio_returns(excess: pd.DataFrame, rates: pd.DataFrame, usd_rate: pd.Series, n_long: int = 2, n_short: int = 2) -> pd.Series:
    """The classic carry trade: each day long the ``n_long`` currencies with the highest rate differential and short the ``n_short`` lowest, equal-weighted, using the previous day's
    differential (known at the time). Returns the daily excess return of the (dollar-neutral) book."""
    diff = rates.sub(usd_rate, axis=0).shift(1)
    ranks = diff.rank(axis=1, ascending=False)
    n = diff.shape[1]
    long = (ranks <= n_long).astype(float) / n_long
    short = (ranks > n - n_short).astype(float) / n_short
    return (excess * (long - short)).sum(axis=1)
