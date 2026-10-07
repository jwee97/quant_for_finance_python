"""A synthetic option market: a reproducible stand-in for historical option-chain data, with known ground truth.

No free source of historical option chains exists, so the options machinery (surface fitting, Greeks, the backtester) is built and tested on a generated market whose
properties are known exactly, and every function that takes a chain also accepts a vendor file loaded with ``load_option_csv``.

Dynamics (physical measure P). The underlying follows a stochastic-volatility process with jumps::

    dS/S = (mu - q) dt + sqrt(v) dW_S + J dN,           dv = kappa_P (theta_P - v) dt + xi sqrt(v) dW_v,       corr(dW_S, dW_v) = rho_sv

Quotes (pricing measure Q). Implied volatilities come from an arbitrage-free SSVI surface whose at-the-money total variance is the Heston-style expectation of integrated
variance, ``theta(T) = (1 + premium) T [theta_Q + (v - theta_Q)(1 - e^{-kappa_Q T}) / (kappa_Q T)]``. The ``variance_premium`` makes implied variance exceed the variance that is
later realised on average: a built-in **variance risk premium** that short-volatility strategies collect, together with the **jump risk** (crashes) that is the reason
it exists. Quotes carry multiplicative noise and a bid-ask spread that widens away from the money and for short expiries.

This is a model of the market, not the market: it cannot tell you whether the premium exists in practice, only whether code that harvests it books costs, hedges and
settles correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pricing import bsm_price
from .schema import normalise_chain
from .surface import SSVISurface, ssvi_total_variance


@dataclass
class SyntheticMarketSpec:
    s0: float = 100.0
    rate: float = 0.02
    dividend: float = 0.015
    mu: float = 0.07
    kappa_p: float = 3.0
    theta_p: float = 0.030              # long-run variance under P (about 17% volatility)
    xi: float = 0.40                    # 2 kappa theta >= xi^2: the Feller condition holds, so variance stays positive
    rho_sv: float = -0.7
    jump_intensity: float = 0.4         # per year
    jump_mean: float = -0.05
    jump_std: float = 0.04
    kappa_q: float = 2.2
    theta_q: float = 0.035              # long-run variance under Q (the term structure of at-the-money variance reverts to it)
    variance_premium: float = 0.25      # implied total variance is this much above the model's expectation of integrated variance: the variance risk premium
    ssvi_rho: float = -0.7
    ssvi_eta: float = 0.9
    ssvi_gamma: float = 0.45
    quote_noise: float = 0.004          # multiplicative noise on implied volatilities
    min_spread: float = 0.01
    spread_pct: float = 0.006
    tick: float = 0.01
    strike_step: float = 1.0            # listed strikes are every ``strike_step`` near the money and every ``strike_step * far_multiple`` in the wings
    far_multiple: int = 5
    near_width: float = 1.2             # 'near the money' = within this many ATM standard deviations of the forward
    min_dte: int = 1
    max_dte: int = 400
    weekly_dte: int = 35


@dataclass
class SyntheticMarket:
    spec: SyntheticMarketSpec
    underlying: pd.DataFrame                     # index date; columns spot, var (instantaneous variance), ret
    quotes: pd.DataFrame                         # all normalised quotes
    _by_date: dict = field(default_factory=dict, repr=False)

    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.underlying.index)

    def chain(self, date) -> pd.DataFrame:
        return self._by_date[pd.Timestamp(date)]

    def true_iv(self, date, strike, expiry):
        """The noise-free implied volatility the generator used (for checking surface fits)."""
        date = pd.Timestamp(date)
        row = self.underlying.loc[date]
        T = (pd.Timestamp(expiry) - date).days / 365.0
        F = row["spot"] * np.exp((self.spec.rate - self.spec.dividend) * T)
        w = _surface_total_variance(self.spec, row["var"], np.log(np.asarray(strike, float) / F), T)
        return np.sqrt(w / T)


def simulate_underlying(spec: SyntheticMarketSpec, n_days: int, start: str = "2020-01-02", seed: int = 0) -> pd.DataFrame:
    """Daily spot and instantaneous variance paths under P (full-truncation Euler with the jump as a Poisson compound)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    spot = np.empty(n_days)
    var = np.empty(n_days)
    spot[0], var[0] = spec.s0, spec.theta_p
    jump_comp = spec.jump_intensity * (np.exp(spec.jump_mean + 0.5 * spec.jump_std ** 2) - 1.0)
    for t in range(1, n_days):
        vp = max(var[t - 1], 0.0)
        z1 = rng.standard_normal()
        z2 = spec.rho_sv * z1 + np.sqrt(1 - spec.rho_sv ** 2) * rng.standard_normal()
        n_j = rng.poisson(spec.jump_intensity * dt)
        jump = n_j * spec.jump_mean + np.sqrt(n_j) * spec.jump_std * rng.standard_normal() if n_j else 0.0
        spot[t] = spot[t - 1] * np.exp((spec.mu - spec.dividend - jump_comp - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1 + jump)
        var[t] = max(var[t - 1] + spec.kappa_p * (spec.theta_p - vp) * dt + spec.xi * np.sqrt(vp * dt) * z2, 1e-5)
    idx = pd.bdate_range(start, periods=n_days)
    out = pd.DataFrame({"spot": spot, "var": var}, index=idx)
    out["ret"] = out["spot"].pct_change()
    return out


def _atm_total_variance(spec: SyntheticMarketSpec, v: float, T):
    T = np.asarray(T, dtype=float)
    return (1.0 + spec.variance_premium) * T * (spec.theta_q + (v - spec.theta_q) * (1.0 - np.exp(-spec.kappa_q * T)) / (spec.kappa_q * T))


def _surface_total_variance(spec: SyntheticMarketSpec, v: float, k, T):
    return ssvi_total_variance(k, _atm_total_variance(spec, v, T), spec.ssvi_rho, spec.ssvi_eta, spec.ssvi_gamma)


def expiry_calendar(date: pd.Timestamp, spec: SyntheticMarketSpec) -> list[pd.Timestamp]:
    """Third-Friday monthly expiries within ``[min_dte, max_dte]`` days plus weekly Fridays within ``weekly_dte`` days."""
    out = set()
    horizon = date + pd.Timedelta(days=spec.max_dte)
    month = pd.Timestamp(date.year, date.month, 1)
    while month <= horizon:
        first_friday = month + pd.Timedelta(days=(4 - month.weekday()) % 7)
        out.add(first_friday + pd.Timedelta(days=14))
        month += pd.offsets.MonthBegin(1)
    d = date + pd.Timedelta(days=(4 - date.weekday()) % 7)
    while d <= date + pd.Timedelta(days=spec.weekly_dte):
        out.add(d)
        d += pd.Timedelta(days=7)
    return sorted(e for e in out if spec.min_dte <= (e - date).days <= spec.max_dte)


def chain_for_date(spec: SyntheticMarketSpec, date: pd.Timestamp, spot: float, var: float, rng: np.random.Generator, listed: dict | None = None) -> pd.DataFrame:
    """One date's quotes. ``listed`` (expiry -> strikes) is the exchange's listing state: strikes are added as the spot moves and are never delisted before expiry, so a contract that
    was quoted yesterday is quoted today."""
    rows = []
    listed = {} if listed is None else listed
    for expiry in expiry_calendar(date, spec):
        T = (expiry - date).days / 365.0
        F = spot * np.exp((spec.rate - spec.dividend) * T)
        atm_sd = np.sqrt(_atm_total_variance(spec, var, T))
        lo, hi = F * np.exp(-3.2 * atm_sd), F * np.exp(2.2 * atm_sd)
        lattice = np.arange(np.ceil(lo / spec.strike_step), np.floor(hi / spec.strike_step) + 1) * spec.strike_step     # a fixed lattice: a listed strike stays listed
        near = np.abs(np.log(lattice / F)) <= spec.near_width * atm_sd
        far_ok = np.isclose(lattice % (spec.strike_step * spec.far_multiple), 0.0) | np.isclose(lattice % (spec.strike_step * spec.far_multiple), spec.strike_step * spec.far_multiple)
        new = lattice[(near | far_ok) & (lattice > 0.2 * spot)]
        strikes = np.union1d(listed.get(expiry, np.array([])), new)
        listed[expiry] = strikes
        k = np.log(strikes / F)
        iv = np.sqrt(_surface_total_variance(spec, var, k, T) / T) * np.exp(spec.quote_noise * rng.standard_normal((2, len(k))))
        for j, right in enumerate(("C", "P")):
            mid = bsm_price(spot, strikes, T, spec.rate, spec.dividend, iv[j], right == "C")
            otm = np.abs(k) / atm_sd
            half = 0.5 * np.maximum(spec.min_spread, spec.spread_pct * mid * (1.0 + 0.6 * otm) * (1.0 + 0.3 / np.sqrt(max(T, 0.02) * 12.0)))
            bid = np.maximum(np.floor((mid - half) / spec.tick) * spec.tick, 0.0)
            ask = np.maximum(np.ceil((mid + half) / spec.tick) * spec.tick, bid + spec.tick)
            rows.append(pd.DataFrame({"date": date, "expiry": expiry, "strike": strikes, "right": right, "bid": bid, "ask": ask, "underlying": spot, "rate": spec.rate,
                                      "dividend": spec.dividend, "volume": np.maximum(0.0, 1000.0 * np.exp(-0.5 * otm ** 2) / (1 + 5 * T)).round(), "open_interest": 0.0,
                                      "iv": iv[j]}))
    return pd.concat(rows, ignore_index=True)


def generate_market(spec: SyntheticMarketSpec | None = None, n_days: int = 504, start: str = "2020-01-02", seed: int = 0, every: int = 1, warm: int = 0) -> SyntheticMarket:
    """Simulate the underlying and quote a full chain on every ``every``-th date (default: daily). ``warm`` extra leading days are simulated and discarded so the variance
    starts at its stationary distribution."""
    spec = spec or SyntheticMarketSpec()
    und = simulate_underlying(spec, n_days + warm, start, seed).iloc[warm:]
    rng = np.random.default_rng(seed + 12345)
    chains, by_date, listed = [], {}, {}
    for i, (date, row) in enumerate(und.iterrows()):
        if i % every:
            continue
        raw = chain_for_date(spec, date, row["spot"], row["var"], rng, listed)
        for gone in [e for e in listed if e <= date]:
            listed.pop(gone)
        norm_chain = normalise_chain(raw, "SYN")
        by_date[date] = norm_chain
        chains.append(norm_chain)
    return SyntheticMarket(spec, und, pd.concat(chains, ignore_index=True), by_date)


def surface_from_spec(spec: SyntheticMarketSpec, var: float, expiries) -> SSVISurface:
    """The exact SSVI surface the generator uses on a date with instantaneous variance ``var`` (for tests)."""
    thetas = _atm_total_variance(spec, var, np.asarray(expiries, float))
    return SSVISurface(expiries, thetas, spec.ssvi_rho, spec.ssvi_eta, spec.ssvi_gamma)


VENDOR_COLUMNS = {
    "date": ["date", "quote_date", "trade_date", "quotedate"],
    "expiry": ["expiry", "expiration", "exdate", "expiration_date"],
    "strike": ["strike", "strike_price"],
    "right": ["right", "cp_flag", "option_type", "type", "call_put"],
    "bid": ["bid", "best_bid", "bid_price"],
    "ask": ["ask", "best_offer", "ask_price", "offer"],
    "underlying": ["underlying", "underlying_price", "spot", "underlying_last", "close"],
    "rate": ["rate", "risk_free_rate", "r"],
    "dividend": ["dividend", "dividend_yield", "q"],
    "volume": ["volume", "trade_volume"],
    "open_interest": ["open_interest", "openinterest", "oi"],
}


def load_option_csv(path, column_map: dict | None = None, strike_scale: float = 1.0, default_rate: float = 0.0, default_dividend: float = 0.0,
                    underlying: str = "UND") -> pd.DataFrame:
    """Load a vendor option file into the standard chain. Column names are matched case-insensitively against common vendor spellings (OptionMetrics, CBOE DataShop,
    OPRA end-of-day files); override with ``column_map={'standard_name': 'your_column'}``. ``strike_scale`` rescales strikes (OptionMetrics quotes them x1000, so use ``0.001``).
    Missing ``rate`` / ``dividend`` columns take the given defaults; call/put flags may be 'C'/'P', 'call'/'put' or 'c'/'p'."""
    raw = pd.read_csv(path)
    lower = {c.lower(): c for c in raw.columns}
    picked = {}
    for std, candidates in VENDOR_COLUMNS.items():
        if column_map and std in column_map:
            picked[std] = column_map[std]
            continue
        for c in candidates:
            if c in lower:
                picked[std] = lower[c]
                break
    missing = [c for c in ("date", "expiry", "strike", "right", "bid", "ask", "underlying") if c not in picked]
    if missing:
        raise ValueError(f"could not find columns for {missing} in {list(raw.columns)}; pass column_map")
    df = pd.DataFrame({std: raw[col] for std, col in picked.items()})
    df["strike"] = df["strike"].astype(float) * strike_scale
    if "rate" not in df:
        df["rate"] = default_rate
    if "dividend" not in df:
        df["dividend"] = default_dividend
    return normalise_chain(df, underlying)
