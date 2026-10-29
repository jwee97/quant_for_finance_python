"""The options workspace: what an option chain says, what a position is exposed to, and what a strategy would have paid. Pure functions over pandas and numpy; the dashboard only asks for them.

**Where the chain comes from.** The repository holds no historical or live option quotes. For an ETF in the platform's universe the chain is therefore *generated*: ``build_model_chain`` quotes the
platform's arbitrage-free SSVI surface (``derivatives.synthetic``) anchored to the underlying's real last close and its real realised volatility, with an assumed rate and dividend yield, and adds
simulated volume, open interest and quote ages. Every number in such a chain is a model output and the payload says so (``source: simulated``). A chain you load yourself (a vendor CSV in
``data/user/options/``, read by ``derivatives.synthetic.load_option_csv``) is analysed exactly as it is (``source: user_file``). The analytics below never care which: they take the standard chain table of
``derivatives.schema`` (``date, expiry, strike, right, bid, ask, underlying, rate, dividend, volume, open_interest, iv, mid, T, F, k``).

Units. Implied volatilities are decimals (0.20 is 20%); the dashboard prints them in points. Greeks per contract are in dollars for the stated multiplier: ``delta_dollars`` is the dollar value of the
shares the position is equivalent to, ``gamma_pnl_1pct`` the P&L of a 1% move from convexity alone, ``vega_per_point`` the P&L of one volatility point, ``theta_per_day`` one calendar day, ``rho_per_1pct`` one
percentage point of rate.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..derivatives.greeks import bsm_greeks
from ..derivatives.iv import implied_vol
from ..derivatives.pricing import bsm_price
from ..derivatives.schema import normalise_chain
from ..derivatives.synthetic import SyntheticMarketSpec, chain_for_date
from .common import AssetSnapshot, add_flag, liquidity_score, nice_step, percentile_rank, range_rank, realized_vol

DEFAULT_RATE = 0.04
# Trailing dividend yields of the platform's ETFs, rounded and ASSUMED (they only move forward prices and put-call parity by a few points of a percent).
DIVIDEND_YIELDS = {"SPY": 0.013, "QQQ": 0.006, "IWM": 0.012, "EFA": 0.030, "EEM": 0.025, "VNQ": 0.040, "AGG": 0.035, "TLT": 0.038, "IEF": 0.035, "SHY": 0.040, "LQD": 0.040, "HYG": 0.060}
# A real implied-volatility index for some underlyings (series in the platform's macro panel) and the tenor of the index in days.
REFERENCE_INDEX = {"SPY": ("VIX", 30), "QQQ": ("VXN", 30), "GLD": ("GVZ", 30), "DBC": ("OVX", 30)}
MULTIPLIER = 100.0
STALE_AFTER_MIN = 15.0
KAPPA_Q = 2.2                                           # speed at which the model's implied variance reverts to its long-run level (the generator's ``kappa_q``)


# ------------------------------------------------------------------------------------------------------------------ the model chain
def model_spec(spot: float, vol: float, long_run_vol: float, rate: float = DEFAULT_RATE, dividend: float = 0.0, premium: float = 0.15) -> SyntheticMarketSpec:
    """The generator's settings for an underlying: strikes every 0.25% of the spot (rounded), listed out to three standard deviations of each expiry, long-run implied variance at ``long_run_vol`` squared, and ``premium`` as the implied-over-realised variance gap. No noise is added to the quotes (the generator's default adds some, which is
    realistic for a backtest but would make a displayed smile jagged and its static checks fail): what is left is the surface and the rounding of bids and asks to a cent."""
    if spot <= 0 or vol <= 0 or long_run_vol <= 0:
        raise ValueError("spot and volatilities must be positive")
    if not -0.5 <= premium <= 2.0:
        raise ValueError("variance premium must be between -0.5 and 2.0")
    return SyntheticMarketSpec(s0=spot, rate=rate, dividend=dividend, theta_q=long_run_vol ** 2, variance_premium=premium, strike_step=nice_step(spot * 0.0025), near_width=3.0, max_dte=400, weekly_dte=35,
                               quote_noise=0.0)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def model_atm_iv(vol, long_run_vol: float, tenor_days: float = 30.0, premium: float = 0.15, kappa: float = KAPPA_Q):
    """The model's at-the-money implied volatility at a tenor, given the current volatility ``vol`` (a number or a Series): ``sqrt((1 + premium) [theta + (v - theta)(1 - e^{-kT}) / (kT)])`` with ``v = vol^2``."""
    T = max(float(tenor_days), 1.0) / 365.0
    v = np.asarray(vol, dtype=float) ** 2
    theta = long_run_vol ** 2
    out = np.sqrt((1.0 + premium) * (theta + (v - theta) * (1.0 - math.exp(-kappa * T)) / (kappa * T)))
    return pd.Series(out, index=vol.index) if isinstance(vol, pd.Series) else (float(out) if np.ndim(out) == 0 else out)


def add_market_activity(chain: pd.DataFrame, seed: int, scale: float = 600.0) -> pd.DataFrame:
    """SIMULATED volume, open interest and quote age for a model chain (a real chain brings its own). Activity is highest at the money and for the nearer expiries, open interest builds in the downside puts
    and the upside calls, and the far wings are quoted rarely, which is what makes some of their quotes stale."""
    rng = np.random.default_rng(seed)
    out = chain.copy()
    n = len(out)
    atm_sd = np.maximum(out["iv"].to_numpy() * np.sqrt(np.maximum(out["T"].to_numpy(), 1.0 / 365.0)), 1e-3)
    m = np.abs(out["k"].to_numpy()) / atm_sd                                  # distance from the forward in standard deviations
    T = out["T"].to_numpy()
    noise = np.exp(0.35 * rng.standard_normal(n))
    hedge_side = np.where(out["right"].to_numpy() == "P", np.where(out["k"].to_numpy() < 0, 1.5, 0.6), np.where(out["k"].to_numpy() > 0, 1.2, 0.7))
    out["volume"] = np.round(scale * np.exp(-0.5 * (m / 0.9) ** 2) / (1.0 + 4.0 * np.sqrt(T)) * noise * (0.5 + 0.5 * hedge_side)).clip(min=0)
    out["open_interest"] = np.round(6.0 * scale * np.exp(-0.5 * (m / 1.2) ** 2) * (0.4 + np.sqrt(T)) * noise * hedge_side).clip(min=0)
    slow = float(np.clip(600.0 / max(scale, 1.0), 0.5, 8.0))                  # a thinly traded underlying is quoted less often
    out["quote_age_min"] = np.round(rng.exponential((0.5 + 0.8 * m ** 2.5) * slow, n), 1)
    return out


def build_model_chain(spot: float, as_of, vol: float, long_run_vol: float | None = None, rate: float = DEFAULT_RATE, dividend: float = 0.0, premium: float = 0.15,
                      symbol: str = "UND", activity_scale: float = 600.0) -> pd.DataFrame:
    """A full chain (all listed expiries, both rights) for ``as_of``, quoted from the model surface at ``spot`` with current volatility ``vol``. Deterministic in its inputs, so a page reload shows the same chain."""
    spec = model_spec(spot, vol, long_run_vol or vol, rate, dividend, premium)
    date = pd.Timestamp(as_of).normalize()
    rng = np.random.default_rng(_seed(symbol, date.date(), round(spot, 4), round(vol, 5)))
    raw = chain_for_date(spec, date, float(spot), float(vol) ** 2, rng)
    chain = normalise_chain(raw, symbol)
    return add_market_activity(chain, _seed("activity", symbol, date.date()), activity_scale)


def prepare_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """Fill what a displayed chain needs and a vendor file may lack: ``iv`` (inverted from the mid), ``volume``, ``open_interest``, ``dividend``; keep one quote date (the latest)."""
    out = chain.copy()
    if "date" in out and out["date"].nunique() > 1:
        out = out[out["date"] == out["date"].max()]
    for col in ("volume", "open_interest"):
        if col not in out:
            out[col] = 0.0
    if "mid" not in out:
        out["mid"] = 0.5 * (out["bid"] + out["ask"])
    if "T" not in out:
        out["T"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(out["date"])).dt.days / 365.0
    if "F" not in out:
        out["F"] = out["underlying"] * np.exp((out["rate"] - out.get("dividend", 0.0)) * out["T"])
    if "k" not in out:
        out["k"] = np.log(out["strike"] / out["F"])
    if "iv" not in out or out["iv"].isna().all():
        iv = np.full(len(out), np.nan)
        for right, call in (("C", True), ("P", False)):
            sel = (out["right"] == right).to_numpy()
            if sel.any():
                s = out[sel]
                iv[sel] = implied_vol(s["mid"].to_numpy(), s["underlying"].to_numpy(), s["strike"].to_numpy(), s["T"].to_numpy(), s["rate"].to_numpy(), s["dividend"].to_numpy(), call)
        out["iv"] = iv
    return out.reset_index(drop=True)


def quote_staleness(chain: pd.DataFrame, stale_after_min: float = STALE_AFTER_MIN) -> pd.Series:
    """True for a quote older than ``stale_after_min`` minutes (the ``quote_age_min`` column) or, when the file has no ages, dated before the newest quote date in the file."""
    if "quote_age_min" in chain:
        return chain["quote_age_min"].fillna(0.0) > stale_after_min
    if "date" in chain and chain["date"].nunique() > 1:
        return chain["date"] < chain["date"].max()
    return pd.Series(False, index=chain.index)


def expiry_list(chain: pd.DataFrame) -> list[dict]:
    """The expiries on the chain: date, days to expiry and number of strikes quoted."""
    out = []
    for expiry, g in chain.groupby("expiry"):
        out.append({"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(float(g["T"].iloc[0]) * 365)), "strikes": int(g["strike"].nunique())})
    return sorted(out, key=lambda r: r["dte"])


def nearest_expiry(chain: pd.DataFrame, min_dte: int = 1) -> pd.Timestamp:
    days = {e: int(round(float(g["T"].iloc[0]) * 365)) for e, g in chain.groupby("expiry")}
    ok = [e for e, d in days.items() if d >= min_dte]
    if not ok:
        raise ValueError("the chain has no expiry in the future")
    return min(ok, key=lambda e: days[e])


def _expiry_slice(chain: pd.DataFrame, expiry) -> pd.DataFrame:
    g = chain[chain["expiry"] == pd.Timestamp(expiry)]
    if g.empty:
        raise ValueError(f"expiry {pd.Timestamp(expiry).date()} is not on the chain; choose one of {[e['expiry'] for e in expiry_list(chain)]}")
    return g


# ------------------------------------------------------------------------------------------------------------------ smile, skew, term structure
def otm_smile(g: pd.DataFrame, min_mid: float = 0.05) -> pd.DataFrame:
    """The out-of-the-money quotes of one expiry with a usable implied volatility, sorted by strike: the informative side of the chain. ``min_mid`` drops quotes worth almost nothing, whose implied
    volatility is mostly rounding to the tick."""
    otm = g[((g["right"] == "P") & (g["strike"] <= g["F"])) | ((g["right"] == "C") & (g["strike"] > g["F"]))]
    otm = otm[np.isfinite(otm["iv"]) & (otm["iv"] > 0) & (otm["bid"] > 0) & (otm["mid"] >= min_mid)]
    return otm.sort_values("strike")


def atm_iv(g: pd.DataFrame) -> float:
    """At-the-money implied volatility of one expiry: the OTM smile interpolated at the forward (NaN with fewer than two usable quotes)."""
    s = otm_smile(g)
    if len(s) < 2:
        return float("nan")
    return float(np.interp(0.0, s["k"].to_numpy(), s["iv"].to_numpy()))


def smile(chain: pd.DataFrame, expiry) -> dict:
    """The volatility smile of one expiry: strike, moneyness (K / spot) and implied volatility of each OTM quote."""
    g = _expiry_slice(chain, expiry)
    s = otm_smile(g)
    spot = float(g["underlying"].iloc[0])
    return {"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(float(g["T"].iloc[0]) * 365)), "forward": float(g["F"].iloc[0]), "spot": spot, "atm_iv": atm_iv(g),
            "points": [{"strike": float(r.strike), "moneyness": float(r.strike / spot), "k": float(r.k), "iv": float(r.iv), "right": str(r.right)} for r in s.itertuples()]}


def iv_at_delta(g: pd.DataFrame, delta: float, put: bool = False) -> float:
    """Implied volatility at a Black-Scholes delta, read off the OTM smile of one expiry: the strike whose call delta (put: ``|put delta|``) equals ``delta``, found by interpolating the smile's own
    deltas. NaN if the delta lies outside the quoted range."""
    s = otm_smile(g)
    if len(s) < 4:
        return float("nan")
    T = float(g["T"].iloc[0])
    r, q = float(g["rate"].iloc[0]), float(g["dividend"].iloc[0]) if "dividend" in g else 0.0
    sig = s["iv"].to_numpy()
    d1 = (np.log(s["underlying"].to_numpy() / s["strike"].to_numpy()) + (r - q + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    call_delta = math.exp(-q * T) * norm.cdf(d1)                                 # falls as the strike rises
    target = math.exp(-q * T) - delta if put else delta                          # a put of |delta| d has call delta e^{-qT} - d at the same strike
    order = np.argsort(call_delta)
    xs, ys = call_delta[order], sig[order]
    if not xs[0] <= target <= xs[-1]:
        return float("nan")
    return float(np.interp(target, xs, ys))


def skew_metrics(chain: pd.DataFrame, expiry) -> dict:
    """ATM volatility, 25- and 10-delta risk reversals (call volatility minus put volatility: negative when downside protection is dear) and butterflies (the wings' average minus ATM), and the ATM skew slope
    ``d iv / d ln K`` per 1.00 of log-strike."""
    g = _expiry_slice(chain, expiry)
    atm = atm_iv(g)
    out = {"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(float(g["T"].iloc[0]) * 365)), "atm_iv": atm}
    for d in (25, 10):
        c, p = iv_at_delta(g, d / 100.0, put=False), iv_at_delta(g, d / 100.0, put=True)
        out[f"call_{d}d_iv"], out[f"put_{d}d_iv"] = c, p
        out[f"rr_{d}d"] = c - p if np.isfinite(c) and np.isfinite(p) else float("nan")
        out[f"bf_{d}d"] = 0.5 * (c + p) - atm if np.isfinite(c) and np.isfinite(p) and np.isfinite(atm) else float("nan")
    s = otm_smile(g)
    near = s[np.abs(s["k"]) <= 0.1]
    out["skew_slope"] = float(np.polyfit(near["k"], near["iv"], 1)[0]) if len(near) >= 3 else float("nan")
    return out


def term_structure(chain: pd.DataFrame) -> list[dict]:
    """ATM implied volatility by expiry (days, volatility, forward)."""
    rows = []
    for expiry, g in chain.groupby("expiry"):
        rows.append({"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(float(g["T"].iloc[0]) * 365)), "atm_iv": atm_iv(g), "forward": float(g["F"].iloc[0])})
    return [r for r in sorted(rows, key=lambda r: r["dte"]) if np.isfinite(r["atm_iv"])]


def interpolate_iv(term: list[dict], days: float) -> float:
    """Constant-maturity ATM volatility at ``days``: linear in TOTAL variance between the bracketing expiries (flat beyond the ends)."""
    if not term:
        return float("nan")
    d = np.array([max(t["dte"], 1) for t in term], dtype=float)
    w = np.array([t["atm_iv"] ** 2 * max(t["dte"], 1) / 365.0 for t in term])
    if days <= d[0]:
        return float(term[0]["atm_iv"])
    if days >= d[-1]:
        return float(term[-1]["atm_iv"])
    return float(math.sqrt(max(np.interp(days, d, w), 0.0) / (days / 365.0)))


def term_metrics(term: list[dict]) -> dict:
    """The 30- and 90-day ATM volatility, the slope between them in points, and whether the curve is inverted (the front above the back)."""
    f30, f90 = interpolate_iv(term, 30.0), interpolate_iv(term, 90.0)
    slope = f90 - f30 if np.isfinite(f30) and np.isfinite(f90) else float("nan")
    return {"iv_30d": f30, "iv_90d": f90, "term_slope": slope, "inverted": bool(np.isfinite(slope) and slope < -0.005)}


def surface_grid(chain: pd.DataFrame, moneyness: tuple[float, float] = (0.8, 1.2), points: int = 25, max_expiries: int = 14) -> dict:
    """The implied-volatility surface as a grid: one row per expiry, one column per strike as a fraction of the spot, interpolated along each expiry's OTM smile and left empty (``None``) outside the strikes
    actually quoted: the surface is not extrapolated."""
    grid = np.linspace(moneyness[0], moneyness[1], points)
    rows, labels = [], []
    for expiry, g in sorted(chain.groupby("expiry"), key=lambda kv: kv[0]):
        if len(labels) >= max_expiries:
            break
        s = otm_smile(g)
        if len(s) < 4:
            continue
        spot = float(g["underlying"].iloc[0])
        x, y = (s["strike"] / spot).to_numpy(), s["iv"].to_numpy()
        row = np.where((grid >= x.min()) & (grid <= x.max()), np.interp(grid, x, y), np.nan)
        rows.append([None if not np.isfinite(v) else round(float(v), 5) for v in row])
        labels.append({"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(float(g["T"].iloc[0]) * 365))})
    return {"moneyness": [round(float(x), 4) for x in grid], "expiries": labels, "iv": rows}


def nearest_expiry_to(chain: pd.DataFrame, dte: float) -> pd.Timestamp:
    """The listed expiry whose days to expiry are closest to ``dte``."""
    days = {e: float(g["T"].iloc[0]) * 365.0 for e, g in chain.groupby("expiry")}
    return min(days, key=lambda e: abs(days[e] - dte))


def iv_lookup(chain: pd.DataFrame, expiry, strike: float) -> float:
    """The implied volatility at ``strike`` on ``expiry`` read off the OTM smile by linear interpolation in strike (flat beyond the quoted strikes); the ATM volatility if the slice has too few quotes."""
    g = _expiry_slice(chain, expiry)
    s = otm_smile(g)
    if len(s) < 3:
        return atm_iv(g)
    return float(np.interp(float(strike), s["strike"].to_numpy(), s["iv"].to_numpy()))


def strike_step(chain: pd.DataFrame, expiry) -> float:
    """The strike spacing near the money on one expiry (the median gap between the 15 strikes nearest the forward)."""
    g = _expiry_slice(chain, expiry)
    strikes = np.sort(g["strike"].unique())
    near = strikes[np.argsort(np.abs(strikes - float(g["F"].iloc[0])))[:15]]
    gaps = np.diff(np.sort(near))
    return float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 1.0


def price_tick(chain: pd.DataFrame) -> float:
    """The smallest price increment the quotes use (a cent for a model chain; vendors often use five cents): the smallest positive gap between the distinct bids and asks, between one cent and a quarter."""
    levels = np.unique(np.round(np.r_[chain["bid"].to_numpy(), chain["ask"].to_numpy()], 4))
    gaps = np.diff(levels)
    return float(np.clip(gaps[gaps > 1e-4].min(), 0.01, 0.25)) if (gaps > 1e-4).any() else 0.01


def arbitrage_view(chain: pd.DataFrame, liquid_only: bool = False, band: float = 0.10) -> dict:
    """The static no-arbitrage checks of ``derivatives.schema.arbitrage_report``, with the tolerance a quote's rounding needs. Prices are rounded to a tick, so a slope between two strikes ``gap`` apart is wrong
    by up to ``tick / gap`` at each end, and a butterfly (a difference of two slopes) by twice that: a "violation" smaller than that is the rounding, not the market. ``tolerance`` is returned with the rows.

    ``liquid_only`` is for a chain of real quotes: a contract with no bid has a "mid" of half its offer, which is no price, and an in-the-money American option is worth more than parity says, so only
    contracts with a live bid within ``band`` of the forward (in log-moneyness) are checked. A model chain is checked whole."""
    from ..derivatives.schema import arbitrage_report

    if liquid_only:
        chain = chain[(chain["bid"] > 0) & (chain["k"].abs() <= band)]
        if chain.empty:
            return {"rows": [], "tick": None, "tolerance": None}
    tick = price_tick(chain)
    gaps = [float(np.diff(np.sort(g["strike"].unique())).min()) for _, g in chain.groupby("expiry") if g["strike"].nunique() > 1]
    tol = 2.0 * tick / max(min(gaps), 1e-6) if gaps else 1e-8
    return {"rows": arbitrage_report(chain, tol=tol).to_dict(orient="records"), "tick": tick, "tolerance": tol}


# ------------------------------------------------------------------------------------------------------------------ the chain table
def _side(g: pd.DataFrame, spot: float, stale_after_min: float) -> dict[float, dict]:
    """Per-strike quote, implied volatility, delta, gamma, volume, open interest, spread and staleness of ONE right at ONE expiry."""
    call = bool(g["right"].iloc[0] == "C") if len(g) else True
    T = float(g["T"].iloc[0]) if len(g) else 0.0
    sigma = g["iv"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        greeks = bsm_greeks(spot, g["strike"].to_numpy(), max(T, 1e-6), g["rate"].to_numpy(), g["dividend"].to_numpy() if "dividend" in g else 0.0, np.where(sigma > 0, sigma, np.nan), call)
    stale = quote_staleness(g, stale_after_min).to_numpy()
    out = {}
    for i, row in enumerate(g.itertuples()):
        mid = float(row.mid)
        spread = float(row.ask - row.bid)
        out[float(row.strike)] = {"bid": float(row.bid), "ask": float(row.ask), "mid": mid, "iv": float(row.iv) if np.isfinite(row.iv) else None,
                                  "delta": float(greeks["delta"][i]) if np.isfinite(greeks["delta"][i]) else None, "gamma": float(greeks["gamma"][i]) if np.isfinite(greeks["gamma"][i]) else None,
                                  "volume": float(row.volume), "oi": float(row.open_interest), "spread": spread, "spread_pct": spread / mid if mid > 0 else None, "stale": bool(stale[i]),
                                  "itm": bool((spot > row.strike) if call else (spot < row.strike))}
    return out


def chain_view(chain: pd.DataFrame, expiry=None, window: float = 0.15, stale_after_min: float = STALE_AFTER_MIN, max_rows: int = 81) -> dict:
    """The chain viewer for one expiry: calls and puts side by side per strike within ``window`` (a fraction of the forward) of the money, the strike nearest the forward marked, and the totals the header
    shows (volume and open interest by right, put/call ratios, quotes older than ``stale_after_min``, spreads wider than 10% of the mid)."""
    expiry = nearest_expiry(chain) if expiry in (None, "") else pd.Timestamp(expiry)
    g = _expiry_slice(chain, expiry)
    spot, forward, T = float(g["underlying"].iloc[0]), float(g["F"].iloc[0]), float(g["T"].iloc[0])
    keep = g[np.abs(g["strike"] / forward - 1.0) <= window]
    calls, puts = _side(keep[keep["right"] == "C"], spot, stale_after_min), _side(keep[keep["right"] == "P"], spot, stale_after_min)
    strikes = sorted(set(calls) | set(puts))
    if len(strikes) > max_rows:                                                  # keep the rows nearest the money
        strikes = sorted(sorted(strikes, key=lambda k: abs(k - forward))[:max_rows])
    atm_strike = min(strikes, key=lambda k: abs(k - forward)) if strikes else None
    rows = [{"strike": k, "moneyness": k / spot - 1.0, "atm": k == atm_strike, "call": calls.get(k), "put": puts.get(k)} for k in strikes]
    both = [r[side] for r in rows for side in ("call", "put") if r[side] is not None]
    cv, pv = sum(x["volume"] for x in calls.values()), sum(x["volume"] for x in puts.values())
    co, po = sum(x["oi"] for x in calls.values()), sum(x["oi"] for x in puts.values())
    return {"expiry": str(expiry.date()), "dte": int(round(T * 365)), "spot": spot, "forward": forward, "atm_strike": atm_strike, "atm_iv": atm_iv(g), "rows": rows,
            "totals": {"call_volume": cv, "put_volume": pv, "call_oi": co, "put_oi": po, "put_call_volume": pv / cv if cv > 0 else None, "put_call_oi": po / co if co > 0 else None},
            "stale_quotes": int(sum(x["stale"] for x in both)), "wide_quotes": int(sum(1 for x in both if x["spread_pct"] is not None and x["spread_pct"] > 0.10)), "quotes": len(both),
            "quote_ages": "quote_age_min" in chain.columns, "expiries": expiry_list(chain)}


# ------------------------------------------------------------------------------------------------------------------ overview
@dataclass
class UnderlyingContext:
    """What the workspace knows about the underlying's own price history."""

    symbol: str
    spot: float
    as_of: pd.Timestamp
    rv_21: float
    rv_63: float
    long_run_vol: float
    rv_history: pd.Series            # trailing 21-day realised volatility
    dollar_volume: float
    prices: pd.Series | None = None  # the closes the volatilities came from (the volatility cone is drawn from them)


def underlying_context(symbol: str, prices: pd.Series, volume: pd.Series | None = None) -> UnderlyingContext:
    """Spot, realised volatilities and the long-run volatility (the root-mean-square of the 21-day realised volatility over up to five years) from a price history."""
    p = prices.dropna()
    if len(p) < 130:
        raise ValueError(f"{symbol} has {len(p)} prices; the options workspace needs at least 130 days of history")
    rv = realized_vol(p, 21).dropna()
    rv63 = float(realized_vol(p, 63).dropna().iloc[-1])
    dollar_volume = float((p * volume.reindex(p.index)).tail(60).median()) if volume is not None else float("nan")
    return UnderlyingContext(symbol=symbol, spot=float(p.iloc[-1]), as_of=p.index[-1], rv_21=float(rv.iloc[-1]), rv_63=rv63, long_run_vol=float(np.sqrt((rv.tail(1260) ** 2).mean())), rv_history=rv,
                             dollar_volume=dollar_volume, prices=p)


def simulated_chain(symbol: str, prices: pd.Series, volume: pd.Series | None = None, rate: float | None = None, dividend: float | None = None, premium: float = 0.15):
    """``(chain, context, rate, dividend)``: the model chain of an ETF, an arbitrage-free SSVI surface quoted at its real last close and its real 21-day realised volatility (the platform holds no option quotes
    for it), with the activity columns scaled to its dollar volume. ``rate`` defaults to ``DEFAULT_RATE``, ``dividend`` to the ETF's yield in ``DIVIDEND_YIELDS``, ``premium`` is the implied-over-realised variance premium."""
    ctx = underlying_context(symbol, prices, volume)
    div = DIVIDEND_YIELDS.get(symbol, 0.0) if dividend is None else dividend
    r = DEFAULT_RATE if rate is None else rate
    scale = float(np.clip((ctx.dollar_volume if np.isfinite(ctx.dollar_volume) else 1e8) / 1e9, 0.05, 50.0)) * 600.0
    chain = prepare_chain(build_model_chain(ctx.spot, ctx.as_of, ctx.rv_21, ctx.long_run_vol, r, div, premium, symbol, scale))
    return chain, ctx, r, div


def reference_index_view(symbol: str, macro: pd.DataFrame | None, atm_at_tenor: float, as_of=None) -> dict | None:
    """The real implied-volatility index for the underlying, if the platform has one: its level, its one-year rank and percentile, and the chain's ATM volatility at the same tenor for comparison."""
    spec = REFERENCE_INDEX.get(symbol.upper())
    if spec is None or macro is None or spec[0] not in macro.columns:
        return None
    s = macro[spec[0]].dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]                                       # the index as it stood on the chain's date, not later
    if len(s) < 60:
        return None
    level = float(s.iloc[-1]) / 100.0
    year = s.tail(252) / 100.0
    return {"name": spec[0], "tenor_days": spec[1], "level": level, "as_of": str(s.index[-1].date()), "rank_1y": range_rank(year, level), "percentile_1y": percentile_rank(year, level),
            "chain_atm_iv": atm_at_tenor}


def overview(chain: pd.DataFrame, ctx: UnderlyingContext, premium: float = 0.15, macro: pd.DataFrame | None = None, simulated: bool = True, source: str = "user_file") -> dict:
    """The cards above the chain: spot, realised volatility, ATM implied volatility (30-day, constant maturity), IV rank and percentile over one year, put/call ratios, the nearest expiry, the term slope, and
    the risk flags worth reading. For a simulated chain the IV history is the model applied to the realised-volatility history, so the rank is the rank of realised volatility in a new coat and is labelled
    as such; the real index (when the platform has one) is shown beside it. ``source`` labels a real chain (``user_file`` or ``downloaded``); a real chain is one snapshot, so it has no
    implied-volatility history and no rank."""
    term = term_structure(chain)
    tm = term_metrics(term)
    near = nearest_expiry(chain)
    g_near = _expiry_slice(chain, near)
    iv30 = tm["iv_30d"]
    if simulated:
        hist = model_atm_iv(ctx.rv_history, ctx.long_run_vol, 30.0, premium).tail(252)
        basis = "model: the surface applied to the trailing realised-volatility history (not an observed implied-volatility history)"
    else:
        hist, basis = pd.Series(dtype=float), "a real chain is one day's snapshot: it has no history of implied volatility, so the one-year rank is unavailable"
    rank, pct = (range_rank(hist, iv30), percentile_rank(hist, iv30)) if len(hist) else (float("nan"), float("nan"))
    call_v, put_v = float(chain.loc[chain["right"] == "C", "volume"].sum()), float(chain.loc[chain["right"] == "P", "volume"].sum())
    call_o, put_o = float(chain.loc[chain["right"] == "C", "open_interest"].sum()), float(chain.loc[chain["right"] == "P", "open_interest"].sum())
    atm = g_near.iloc[(g_near["strike"] - g_near["F"]).abs().argsort().iloc[:2]]
    atm_spread_pct = float(((atm["ask"] - atm["bid"]) / atm["mid"].where(atm["mid"] > 0)).mean())
    stale_share = float(quote_staleness(chain).mean())
    days_of = {e["expiry"]: e["dte"] for e in expiry_list(chain)}
    skew_pool = [e for e, d in days_of.items() if d >= 7] or list(days_of)
    skew_expiry = min(skew_pool, key=lambda e: abs(days_of[e] - 30))                   # the expiry nearest 30 days (at least a week out): a one-day smile is mostly noise
    skew = skew_metrics(chain, skew_expiry) if len(otm_smile(_expiry_slice(chain, skew_expiry))) >= 4 else {}
    flags: list[str] = []
    add_flag(flags, np.isfinite(atm_spread_pct) and atm_spread_pct > 0.10, f"at-the-money spread is {atm_spread_pct:.0%} of the mid: expect to pay it twice")
    add_flag(flags, stale_share > 0.10, f"{stale_share:.0%} of the quotes are stale")
    add_flag(flags, tm["inverted"], "the volatility term structure is inverted: the front is above the back, as in stress")
    add_flag(flags, np.isfinite(rank) and rank > 0.8, "implied volatility is near the top of its one-year range: long-volatility positions are dear")
    add_flag(flags, np.isfinite(rank) and rank < 0.2, "implied volatility is near the bottom of its one-year range: short-volatility positions pay little")
    add_flag(flags, np.isfinite(skew.get("rr_25d", float("nan"))) and skew.get("rr_25d", 0.0) < -0.06, "steep put skew: the market pays a lot for downside protection")
    add_flag(flags, np.isfinite(iv30) and iv30 < ctx.rv_21 * 0.9, "implied volatility is below recent realised volatility: options are cheap against what the stock has done")
    liquid = liquidity_score(atm_spread_pct * 1e4 if np.isfinite(atm_spread_pct) else float("inf"), ctx.dollar_volume)
    snap = AssetSnapshot(ctx.symbol, "option", ctx.spot, float(iv30) if np.isfinite(iv30) else ctx.rv_21, liquid, flags, str(ctx.as_of.date()), "simulated" if simulated else source)
    ref = reference_index_view(ctx.symbol, macro, interpolate_iv(term, REFERENCE_INDEX.get(ctx.symbol.upper(), ("", 30))[1]), ctx.as_of)
    return {"symbol": ctx.symbol, "spot": ctx.spot, "as_of": str(ctx.as_of.date()), "realized_vol_21d": ctx.rv_21, "realized_vol_63d": ctx.rv_63, "long_run_vol": ctx.long_run_vol,
            "atm_iv_30d": iv30, "atm_iv_90d": tm["iv_90d"], "iv_premium": iv30 - ctx.rv_21 if np.isfinite(iv30) else None, "iv_rank": rank, "iv_percentile": pct, "iv_rank_basis": basis,
            "put_call_volume": put_v / call_v if call_v > 0 else None, "put_call_oi": put_o / call_o if call_o > 0 else None, "nearest_expiry": str(pd.Timestamp(near).date()),
            "nearest_dte": int(round(float(g_near["T"].iloc[0]) * 365)), "term_slope": tm["term_slope"], "term_inverted": tm["inverted"], "skew": skew, "reference_index": ref,
            "atm_spread_pct": atm_spread_pct, "stale_share": stale_share, "quote_ages": "quote_age_min" in chain.columns, "snapshot": snap.to_dict(), "risk_flags": flags}


# ------------------------------------------------------------------------------------------------------------------ what the quotes imply
def _integrate(y, x) -> float:
    """The trapezoid rule (``numpy.trapz`` was renamed between numpy releases)."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _cumulative(y, x) -> np.ndarray:
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    return np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))]


TAIL_MOVES = (0.05, 0.10, 0.15, 0.20)


def implied_distribution(chain: pd.DataFrame, expiry, points: int = 801) -> dict:
    """The distribution of the price at one expiry that the option quotes imply (the risk-neutral density of Breeden and Litzenberger 1978): the second derivative of the call price with respect to
    the strike, taken here from a smooth volatility smile rather than from the noisy quotes themselves. An SVI curve (Gatheral) is fitted to the expiry's out-of-the-money implied volatilities
    (``derivatives.surface.fit_svi_slice``, weighted by the inverse of each quote's total variance so that the wings do not dominate) and its density is read from the fitted total variance
    (``risk_neutral_density``, the density of ``ln(S_T / F)``). A negative density (a butterfly arbitrage in the fit) is set to zero and its share reported; the density is rescaled to one over the grid
    (``area`` says how much of the mass the grid held). Returned: the price, the density per dollar and the cumulative probability on a grid, the lognormal density with the at-the-money volatility
    for comparison, quantiles, moments of the log return, the probability of a fall or a rise of 5, 10, 15 and 20% against the lognormal's, the move the at-the-money straddle prices in, and how well the fit
    reproduced the quotes. It is the distribution the options are PRICED on (the market's risk-neutral view, with the risk premium inside it), not a forecast of the price."""
    from ..derivatives.surface import fit_svi_slice, risk_neutral_density

    g = _expiry_slice(chain, expiry)
    quotes = otm_smile(g)
    if len(quotes) < 6:
        raise ValueError(f"the {pd.Timestamp(expiry).date()} expiry has {len(quotes)} usable out-of-the-money quotes; the implied distribution needs at least 6")
    T, F, spot = float(g["T"].iloc[0]), float(g["F"].iloc[0]), float(g["underlying"].iloc[0])
    k, iv = quotes["k"].to_numpy(), quotes["iv"].to_numpy()
    sigma = atm_iv(g)
    fit = fit_svi_slice(k, iv, T, weights=1.0 / (iv ** 2 * T))
    half = float(min(1.5, max(0.15, 6.0 * sigma * math.sqrt(T))))
    grid = np.linspace(-half, half, points)
    raw = risk_neutral_density(lambda kk, tt: fit.w(kk), T, grid).to_numpy()
    negative = _integrate(np.clip(-raw, 0.0, None), grid) / max(_integrate(np.abs(raw), grid), 1e-300)
    dens = np.clip(raw, 0.0, None)
    area = _integrate(dens, grid)
    dens = dens / area
    cdf = _cumulative(dens, grid)
    cdf = cdf / cdf[-1]
    sd = sigma * math.sqrt(T)
    lognormal = norm.pdf(grid, loc=-0.5 * sigma ** 2 * T, scale=sd)
    price = F * np.exp(grid)
    m1 = _integrate(grid * dens, grid)
    var = _integrate((grid - m1) ** 2 * dens, grid)
    moments = {"mean_log_return": m1, "sd_log_return": math.sqrt(var), "skewness": _integrate((grid - m1) ** 3 * dens, grid) / var ** 1.5, "excess_kurtosis": _integrate((grid - m1) ** 4 * dens, grid) / var ** 2 - 3.0,
               "mean_price_over_forward": _integrate(np.exp(grid) * dens, grid)}
    quantiles = {f"{p:g}": float(np.interp(p, cdf, price)) for p in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)}
    tails = []
    for x in TAIL_MOVES:
        for side, level in (("fall", spot * (1.0 - x)), ("rise", spot * (1.0 + x))):
            kk = math.log(level / F)
            if not -half < kk < half:
                continue
            below = float(np.interp(kk, grid, cdf))
            lognormal_below = float(norm.cdf((kk + 0.5 * sigma ** 2 * T) / sd))
            tails.append({"side": side, "move": x, "level": float(level), "implied": below if side == "fall" else 1.0 - below, "lognormal": lognormal_below if side == "fall" else 1.0 - lognormal_below})
    # the move the at-the-money straddle prices in: its price over the spot (about 0.8 standard deviations of the move, for a normal)
    atm_strike = float(g.loc[(g["strike"] - F).abs().idxmin(), "strike"])
    at = g[g["strike"] == atm_strike]
    call, put = at.loc[at["right"] == "C", "mid"], at.loc[at["right"] == "P", "mid"]
    straddle = float(call.iloc[0] + put.iloc[0]) if len(call) and len(put) else float("nan")
    lo_k, hi_k = np.interp([0.16, 0.84], cdf, grid)
    shown = np.abs(grid) <= min(half, 4.5 * sd)
    strike = quotes["strike"].to_numpy()
    return {"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(T * 365)), "forward": F, "spot": spot, "atm_iv": sigma, "one_sd_move": sd, "area": area,
            "smile": {"strike": [round(float(x), 4) for x in strike], "quote_iv": [round(float(x), 6) for x in iv], "fit_iv": [round(float(x), 6) for x in fit.iv(k)]},
            "price": [round(float(x), 4) for x in price[shown]], "density": [float(f"{x:.6g}") for x in (dens / price)[shown]], "lognormal": [float(f"{x:.6g}") for x in (lognormal / price)[shown]],
            "cdf": [round(float(x), 6) for x in cdf[shown]], "quantiles": quantiles, "moments": moments, "tails": tails,
            "expected_move": {"straddle": straddle, "straddle_pct": straddle / spot if np.isfinite(straddle) else float("nan"), "one_sd_pct": sd, "band_low": float(F * math.exp(lo_k)), "band_high": float(F * math.exp(hi_k))},
            "fit": {"rmse_vol": float(fit.rmse), "max_error_vol": float(np.max(np.abs(np.sqrt(fit.w(k) / T) - iv))), "quotes": int(len(quotes)), "arbitrage_free": bool(fit.arbitrage_free((-half, half))),
                    "negative_share": float(negative), "params": {"a": fit.a, "b": fit.b, "rho": fit.rho, "m": fit.m, "s": fit.s}}}


def expected_moves(chain: pd.DataFrame, min_dte: int = 2) -> list[dict]:
    """For every expiry of at least ``min_dte`` days: the at-the-money straddle's price (the strike nearest the forward) as a share of the spot, and one standard deviation of the move, the at-the-money
    volatility times the square root of the time. The straddle is worth about 0.8 of a standard deviation for a normal move, so the two disagree when the smile does."""
    rows = []
    for expiry, g in chain.groupby("expiry"):
        T = float(g["T"].iloc[0])
        if T * 365 < min_dte:
            continue
        F, spot, sigma = float(g["F"].iloc[0]), float(g["underlying"].iloc[0]), atm_iv(g)
        if not np.isfinite(sigma):
            continue
        strike = float(g.loc[(g["strike"] - F).abs().idxmin(), "strike"])
        at = g[g["strike"] == strike]
        call, put = at.loc[at["right"] == "C", "mid"], at.loc[at["right"] == "P", "mid"]
        if not (len(call) and len(put)):
            continue
        straddle = float(call.iloc[0] + put.iloc[0])
        rows.append({"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(T * 365)), "atm_iv": sigma, "strike": strike, "straddle": straddle, "straddle_pct": straddle / spot, "one_sd_pct": sigma * math.sqrt(T)})
    return sorted(rows, key=lambda r: r["dte"])


def model_free_variance(chain: pd.DataFrame, expiry) -> dict:
    """The model-free implied variance of one expiry, by the Cboe VIX rules (Cboe 2025; Demeterfi, Derman, Kamal and Zou 1999; Britten-Jones and Neuberger 2000): the out-of-the-money options of
    every strike, weighted by 1 / K^2, replicate a variance swap, whatever the smile, so no pricing model is needed. With ``K0`` the strike at or just below the forward, puts are used below it and
    calls above (the average of the two at ``K0``), each at the mid, walking outwards from ``K0`` and stopping after two strikes in a row without a bid or an ask; ``sigma^2 = (2 / T) e^{rT} sum
    (dK / K^2) Q(K) - (1 / T)(F / K0 - 1)^2`` with ``dK`` half the gap between the neighbouring strikes. Returns the annualised variance and volatility, ``K0`` and the strikes used. ``None``
    where the expiry has no quote at ``K0`` or fewer than three strikes qualify."""
    from ..derivatives.surface import variance_swap_strike

    g = _expiry_slice(chain, expiry)
    T, F, r = float(g["T"].iloc[0]), float(g["F"].iloc[0]), float(g["rate"].iloc[0])
    strikes = np.sort(g["strike"].unique())
    below = strikes[strikes <= F]
    if not len(below):
        return None
    k0 = float(below.max())
    mids = {}
    for right in ("C", "P"):
        side = g[g["right"] == right].drop_duplicates("strike").set_index("strike")
        mids[right] = side
    if k0 not in mids["C"].index or k0 not in mids["P"].index:
        return None

    def usable(right: str, strike: float) -> bool:
        row = mids[right]
        return strike in row.index and float(row.at[strike, "bid"]) > 0.0 and float(row.at[strike, "ask"]) > 0.0

    def walk(candidates, right: str) -> list[float]:
        chosen, misses = [], 0
        for strike in candidates:
            if usable(right, float(strike)):
                chosen.append(float(strike))
                misses = 0
            else:
                misses += 1
                if misses == 2:
                    break
        return chosen

    puts = walk(strikes[strikes < k0][::-1], "P")
    calls = walk(strikes[strikes > k0], "C")
    if len(puts) + len(calls) < 2:
        return None
    used = np.r_[puts[::-1], k0, calls]
    mid = lambda right, strike: 0.5 * float(mids[right].at[strike, "bid"] + mids[right].at[strike, "ask"])             # noqa: E731
    prices = np.r_[[mid("P", x) for x in puts[::-1]], 0.5 * (mid("C", k0) + mid("P", k0)), [mid("C", x) for x in calls]]
    variance = variance_swap_strike(F, T, r, used, prices)
    return {"expiry": str(pd.Timestamp(expiry).date()), "dte": int(round(T * 365)), "variance": float(variance), "vol": float(math.sqrt(variance)) if variance > 0 else float("nan"), "k0": k0, "forward": F,
            "strikes": int(len(used)), "lowest": float(used[0]), "highest": float(used[-1])}


def model_free_term(chain: pd.DataFrame, target_days: float = 30.0) -> dict:
    """``model_free_variance`` for every expiry (with the at-the-money volatility beside it, so that the difference the smile makes is visible) and the constant-maturity figure at ``target_days``: the
    total variance (variance times time) of the two expiries around it, interpolated in time, as the VIX does with its near and next terms, flat in volatility beyond the ends."""
    rows = []
    for expiry in sorted(chain["expiry"].unique()):
        mf = model_free_variance(chain, expiry)
        atm = atm_iv(_expiry_slice(chain, expiry))
        if mf is None or not np.isfinite(mf["vol"]) or mf["dte"] < 2:
            continue
        rows.append({**mf, "atm_iv": atm, "difference": mf["vol"] - atm if np.isfinite(atm) else float("nan")})
    out = {"rows": rows, "target_days": float(target_days), "iv_30d": float("nan"), "atm_30d": interpolate_iv(term_structure(chain), target_days)}
    if rows:
        days = np.array([r["dte"] for r in rows], dtype=float)
        total = np.array([r["variance"] * r["dte"] / 365.0 for r in rows])
        if target_days <= days[0]:
            out["iv_30d"] = rows[0]["vol"]
        elif target_days >= days[-1]:
            out["iv_30d"] = rows[-1]["vol"]
        else:
            out["iv_30d"] = float(math.sqrt(np.interp(target_days, days, total) / (target_days / 365.0)))
    return out


def volatility_cone(prices: pd.Series, windows: tuple = (5, 10, 21, 42, 63, 126, 252), periods: float = 252.0, min_windows: int = 60) -> dict:
    """The volatility cone of Burghardt and Lane (1990): for each window of ``h`` trading days, the lowest, 10th, 25th, 50th, 75th, 90th percentile and highest realised volatility of the underlying over every
    ``h``-day window of its history, and today's, with the share of windows at or below it. Each row also carries ``calendar_days`` (``h`` trading days in calendar days) so that the at-the-money implied
    volatility of an expiry that far out can be set against it. A window the history is too short to give ``min_windows`` observations of is left out."""
    r = np.log(prices.dropna().astype(float)).diff().dropna()
    rows = []
    for h in windows:
        vol = (r.rolling(h).std(ddof=1) * math.sqrt(periods)).dropna()
        if len(vol) < min_windows:
            continue
        q = vol.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        rows.append({"days": int(h), "calendar_days": int(round(h * 365.0 / periods)), "now": float(vol.iloc[-1]), "min": float(vol.min()), "p10": float(q[0.1]), "p25": float(q[0.25]), "median": float(q[0.5]),
                     "p75": float(q[0.75]), "p90": float(q[0.9]), "max": float(vol.max()), "percentile": float((vol <= vol.iloc[-1]).mean()), "windows": int(len(vol))})
    return {"rows": rows, "history_days": int(len(r))}


def positioning(chain: pd.DataFrame, expiry=None, grid_points: int = 41, span: float = 0.15) -> dict:
    """Where the open interest sits, from the chain's ``open_interest`` and ``volume`` columns: by strike, the open interest of calls and puts (one expiry, or all together when ``expiry`` is None); the walls
    (the call strike above the price and the put strike below it with the most open interest); the *max pain* strike of a single expiry (the settlement price at which option holders would be paid the
    least, ``argmin_S sum_calls OI max(S - K, 0) + sum_puts OI max(K - S, 0)``, which is a folk rule: the evidence is that prices tend to cluster at strikes with large open interest near expiry, Ni,
    Pearson and Poteshman 2005); and the *gamma exposure* (GEX): the change in the dollar delta of all the open contracts for a 1% move in the underlying, ``Gamma x OI x 100 x S^2 / 100``, counted
    positive for calls and negative for puts, the sign that follows if the people who hold the other side (the dealers) are long the calls customers write and short the puts they buy. Open interest
    has no sign, so that assumption is not in the data: it is the convention of the dealer-gamma literature (Garleanu, Pedersen and Poteshman 2009 show the demand pressure on prices; Barbon and Buraschi 2020 build a
    gamma imbalance from signed data). The net GEX is also recomputed at spot levels from ``1 - span`` to ``1 + span`` times the price, with each contract's volatility held fixed, and the *flip* is where it
    crosses zero nearest the price: below it dealers' hedging would add to the move, above it damp it."""
    if expiry is not None:
        g = _expiry_slice(chain, expiry)
        label = str(pd.Timestamp(expiry).date())
    else:
        g, label = chain, "all expiries"
    spot = float(g["underlying"].iloc[0])
    oi = g.pivot_table(index="strike", columns="right", values="open_interest", aggfunc="sum").reindex(columns=["C", "P"]).fillna(0.0)
    vol = g.pivot_table(index="strike", columns="right", values="volume", aggfunc="sum").reindex(columns=["C", "P"]).fillna(0.0)
    book = g[(g["open_interest"] > 0) & np.isfinite(g["iv"]) & (g["iv"] > 0)]

    def gamma_dollars(level: float) -> pd.Series:
        """Per row of ``book``: signed dollar gamma for a 1% move with the underlying at ``level``."""
        gam = bsm_greeks(level, book["strike"].to_numpy(), np.maximum(book["T"].to_numpy(), 1.0 / 365.0), book["rate"].to_numpy(), book["dividend"].to_numpy(), book["iv"].to_numpy(), True)["gamma"]
        sign = np.where(book["right"].to_numpy() == "C", 1.0, -1.0)
        return pd.Series(sign * gam * book["open_interest"].to_numpy() * MULTIPLIER * level ** 2 / 100.0, index=book.index)

    now = gamma_dollars(spot)
    by_row = pd.DataFrame({"strike": book["strike"], "right": book["right"], "gex": now}).groupby(["strike", "right"])["gex"].sum().unstack("right").reindex(columns=["C", "P"]).fillna(0.0)
    levels = spot * np.linspace(1.0 - span, 1.0 + span, grid_points)
    net = np.array([float(gamma_dollars(x).sum()) for x in levels])
    flip = None
    for i in np.flatnonzero(net[:-1] * net[1:] < 0):
        crossing = float(levels[i] - net[i] * (levels[i + 1] - levels[i]) / (net[i + 1] - net[i]))
        if flip is None or abs(crossing - spot) < abs(flip - spot):
            flip = crossing
    strikes = sorted(set(oi.index) | set(by_row.index))
    rows = [{"strike": float(k), "call_oi": float(oi["C"].get(k, 0.0)), "put_oi": float(oi["P"].get(k, 0.0)), "call_volume": float(vol["C"].get(k, 0.0)), "put_volume": float(vol["P"].get(k, 0.0)),
             "call_gex": float(by_row["C"].get(k, 0.0)), "put_gex": float(by_row["P"].get(k, 0.0))} for k in strikes]
    call_wall = oi["C"][oi.index >= spot].idxmax() if (oi["C"][oi.index >= spot] > 0).any() else None
    put_wall = oi["P"][oi.index <= spot].idxmax() if (oi["P"][oi.index <= spot] > 0).any() else None
    pain = max_pain(oi["C"], oi["P"]) if expiry is not None and len(oi) else None
    total_c, total_p = float(oi["C"].sum()), float(oi["P"].sum())
    return {"scope": label, "spot": spot, "rows": rows, "call_oi": total_c, "put_oi": total_p, "put_call_oi": total_p / total_c if total_c > 0 else None, "call_volume": float(vol["C"].sum()), "put_volume": float(vol["P"].sum()),
            "call_wall": None if call_wall is None else float(call_wall), "put_wall": None if put_wall is None else float(put_wall), "max_pain": pain,
            "gex": {"net": float(now.sum()), "calls": float(now[book["right"] == "C"].sum()), "puts": float(now[book["right"] == "P"].sum()), "flip": flip,
                    "curve": {"spot": [round(float(x), 4) for x in levels], "net": [round(float(x), 2) for x in net]}}}


def max_pain(call_oi: pd.Series, put_oi: pd.Series) -> float:
    """The strike at which the open interest, if the price settled exactly there, would pay its holders the least: ``argmin_S [sum_K OI_call(K) max(S - K, 0) + sum_K OI_put(K) max(K - S, 0)]`` over the
    listed strikes (the first, the lowest, on a tie). The payout is a convex piecewise-linear function of the settlement, so its minimum is at a strike."""
    strikes = np.sort(np.union1d(call_oi.index.to_numpy(dtype=float), put_oi.index.to_numpy(dtype=float)))
    calls = call_oi.reindex(strikes).fillna(0.0).to_numpy()
    puts = put_oi.reindex(strikes).fillna(0.0).to_numpy()
    payout = np.array([float((calls * np.maximum(s - strikes, 0.0)).sum() + (puts * np.maximum(strikes - s, 0.0)).sum()) for s in strikes])
    return float(strikes[int(np.argmin(payout))])


# ------------------------------------------------------------------------------------------------------------------ positions and portfolio risk
def _years(position: dict, as_of: pd.Timestamp) -> float:
    if position.get("dte") not in (None, ""):
        return max(float(position["dte"]), 0.0) / 365.0
    return max((pd.Timestamp(position["expiry"]) - as_of).days, 0) / 365.0


def position_values(positions: list[dict], spot: float, rate: float, dividend: float, as_of, shock_spot: float = 0.0, shock_vol: float = 0.0, days: float = 0.0) -> pd.DataFrame:
    """Value and Greeks of each position at a shocked market: spot x (1 + ``shock_spot``), every implied volatility + ``shock_vol`` (decimal points), ``days`` of time passed. ``quantity`` is in contracts
    (option) or shares (``type: "stock"``). Dollar values include the multiplier (default 100)."""
    as_of = pd.Timestamp(as_of)
    rows = []
    s = spot * (1.0 + shock_spot)
    for i, pos in enumerate(positions):
        qty = float(pos.get("quantity", 0.0))
        if str(pos.get("type", "option")) == "stock":
            rows.append({"i": i, "kind": "stock", "label": "shares", "expiry": None, "dte": None, "strike": None, "right": None, "quantity": qty, "iv": None, "value": qty * s, "delta": qty, "gamma": 0.0,
                         "vega": 0.0, "theta": 0.0, "rho": 0.0})
            continue
        mult = float(pos.get("multiplier", MULTIPLIER))
        right = str(pos.get("right", "C")).upper()[:1]
        if right not in ("C", "P"):
            raise ValueError("right must be C or P")
        K = float(pos["strike"])
        if K <= 0:
            raise ValueError("strike must be positive")
        T = max(_years(pos, as_of) - days / 365.0, 0.0)
        iv = float(pos.get("iv") or 0.0)
        if iv <= 0:
            raise ValueError("every option position needs an implied volatility (iv)")
        sigma = max(iv + shock_vol, 0.01)
        call = right == "C"
        if T <= 0:
            price = float(max(s - K, 0.0) if call else max(K - s, 0.0))
            g = {k: 0.0 for k in ("delta", "gamma", "vega", "theta", "rho")}
            g["delta"] = float((1.0 if s > K else 0.0) if call else (-1.0 if s < K else 0.0))
        else:
            price = float(bsm_price(s, K, T, rate, dividend, sigma, call))
            raw = bsm_greeks(s, K, T, rate, dividend, sigma, call)
            g = {k: float(raw[k]) for k in ("delta", "gamma", "vega", "theta", "rho")}
        e = pos.get("expiry") or (as_of + pd.Timedelta(days=int(pos["dte"]))).date()
        rows.append({"i": i, "kind": "option", "label": f"{right} {K:g}", "expiry": str(pd.Timestamp(e).date()), "dte": int(round(_years(pos, as_of) * 365)), "strike": K, "right": right, "quantity": qty,
                     "iv": iv, "value": qty * mult * price, "delta": qty * mult * g["delta"], "gamma": qty * mult * g["gamma"], "vega": qty * mult * g["vega"] / 100.0, "theta": qty * mult * g["theta"] / 365.0,
                     "rho": qty * mult * g["rho"] / 100.0, "price": price})
    return pd.DataFrame(rows)


def portfolio_risk(positions: list[dict], spot: float, rate: float = DEFAULT_RATE, dividend: float = 0.0, as_of=None, spot_shocks=(-0.05, -0.02, 0.0, 0.02, 0.05),
                   vol_shocks=(-0.05, 0.0, 0.05), days: float = 0.0) -> dict:
    """Greeks and stress of a book of options (and shares). Returns the position table, the totals (dollar delta, share-equivalent delta, gamma P&L of a 1% move, vega per point, theta per day, rho per
    1 pt) and by expiry, and a stress grid of spot and volatility shocks: the full revaluation P&L next to the Taylor approximation from the Greeks and the part the Greeks miss."""
    if not positions:
        raise ValueError("add at least one position")
    if len(positions) > 60:
        raise ValueError("at most 60 positions")
    as_of = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today().normalize())
    base = position_values(positions, spot, rate, dividend, as_of)
    tot = {"value": float(base["value"].sum()), "delta_shares": float(base["delta"].sum()), "delta_dollars": float(base["delta"].sum() * spot), "gamma": float(base["gamma"].sum()),
           "gamma_pnl_1pct": float(0.5 * base["gamma"].sum() * (0.01 * spot) ** 2), "vega_per_point": float(base["vega"].sum()), "theta_per_day": float(base["theta"].sum()), "rho_per_1pct": float(base["rho"].sum())}
    by_expiry = []
    for expiry, grp in base[base["kind"] == "option"].groupby("expiry"):
        by_expiry.append({"expiry": expiry, "dte": int(grp["dte"].iloc[0]), "delta_dollars": float(grp["delta"].sum() * spot), "gamma_pnl_1pct": float(0.5 * grp["gamma"].sum() * (0.01 * spot) ** 2),
                          "vega_per_point": float(grp["vega"].sum()), "theta_per_day": float(grp["theta"].sum())})
    by_expiry.sort(key=lambda r: r["dte"])
    stress = []
    for ds in spot_shocks:
        for dv in vol_shocks:
            shocked = position_values(positions, spot, rate, dividend, as_of, ds, dv, days)
            pnl = float(shocked["value"].sum() - base["value"].sum())
            d_spot = spot * ds
            taylor = tot["delta_shares"] * d_spot + 0.5 * tot["gamma"] * d_spot ** 2 + tot["vega_per_point"] * dv * 100.0 + tot["theta_per_day"] * days
            stress.append({"spot_shock": float(ds), "vol_shock": float(dv), "pnl": pnl, "taylor": float(taylor), "residual": pnl - float(taylor)})
    worst = min(stress, key=lambda r: r["pnl"])
    flags: list[str] = []
    add_flag(flags, tot["gamma"] < 0 and abs(tot["gamma_pnl_1pct"]) > 0.01 * abs(tot["value"]) + 1.0, "net short gamma: losses accelerate in a large move")
    add_flag(flags, tot["vega_per_point"] < 0, "net short vega: a rise in implied volatility loses money")
    add_flag(flags, tot["theta_per_day"] > 0 and tot["gamma"] < 0, "collecting time decay while short gamma: the classic premium-selling profile")
    add_flag(flags, worst["pnl"] < -0.5 * max(abs(tot["delta_dollars"]), 1.0) and worst["pnl"] < 0, f"the worst stress scenario loses {worst['pnl']:,.0f}")
    positions_out = [{k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()} for r in base.to_dict("records")]
    return {"spot": float(spot), "positions": positions_out, "totals": tot, "by_expiry": by_expiry, "stress": stress, "spot_shocks": [float(x) for x in spot_shocks], "vol_shocks": [float(x) for x in vol_shocks],
            "worst": worst, "risk_flags": flags}


# ------------------------------------------------------------------------------------------------------------------ strategies
def _round_strike(x: float, step: float) -> float:
    return round(round(x / step) * step, 6)


STRATEGY_INFO = {
    "long_call": "Buy a call: profit from a rise, loss limited to the premium",
    "long_put": "Buy a put: profit from a fall, loss limited to the premium",
    "covered_call": "Own 100 shares and sell a call above the market: income now for giving up gains above the strike",
    "protective_put": "Own 100 shares and buy a put below the market: insurance with a floor at the strike",
    "bull_call_spread": "Buy a call and sell a higher call (a vertical): a cheaper bet on a rise with capped profit",
    "bear_put_spread": "Buy a put and sell a lower put (a vertical): a cheaper bet on a fall with capped profit",
    "iron_condor": "Sell an out-of-the-money put spread and call spread: collect premium if the price stays between the short strikes",
    "long_straddle": "Buy a call and a put at the same strike: profit from a big move in either direction",
    "long_strangle": "Buy an out-of-the-money put and call: a cheaper bet on a big move",
    "short_straddle": "Sell a call and a put at the same strike: collect premium if the price stays put; losses are unlimited",
    "short_strangle": "Sell an out-of-the-money put and call: collect premium over a range; losses are unlimited",
    "calendar_spread": "Sell the near call and buy the same-strike far call: profit from the near option decaying faster",
    "butterfly": "Buy the wings and sell two at the middle: profit if the price ends near the middle strike, loss limited",
    "ratio_spread": "Buy one call and sell two higher calls: profit from a modest rise; losses are unlimited above the break-even",
    "collar": "Own 100 shares, buy a put below the market and sell a call above it: the call pays for (some of) the put, so the shares are held between a floor and a cap",
    "cash_secured_put": "Sell a put below the market and hold the cash to buy the shares if they are put to you: paid to wait for a lower price",
    "iron_butterfly": "Sell the at-the-money call and put and buy wings either side: the most premium for a price that stays put, loss limited to the wing width",
    "long_condor": "Buy the outer calls and sell the two inner ones: profit across a band between the inner strikes, loss limited to the debit",
    "risk_reversal": "Buy a call above the market and sell a put below it: a bet on a rise that costs little, with the loss of the shares on a fall",
    "strap": "Buy two calls and one put at the same strike: a bet on a big move, more on the upside",
    "strip": "Buy one call and two puts at the same strike: a bet on a big move, more on the downside",
    "call_backspread": "Sell one call at the money and buy two higher: a free or cheap bet on a big rise that loses a little if the price drifts to the higher strike",
    "put_backspread": "Sell one put at the money and buy two lower: a free or cheap hedge against a crash that loses a little in a modest fall",
    "diagonal_spread": "Buy a longer-dated at-the-money call and sell a nearer-dated higher call against it: cheaper than the long call, with the near option's decay working for you",
    "tail_hedge": "Buy a put far below the market: costs little and pays only in a crash",
}
STRATEGY_PARAMS = {"offset": 0.05, "width": 0.05, "dte": 30, "far_dte": 60}
STRATEGY_GROUPS = {
    "Bet on direction": ["long_call", "long_put", "bull_call_spread", "bear_put_spread", "risk_reversal", "ratio_spread", "call_backspread"],
    "Own the shares": ["covered_call", "protective_put", "collar"],
    "Sell premium": ["cash_secured_put", "short_straddle", "short_strangle", "iron_condor", "iron_butterfly"],
    "Buy volatility": ["long_straddle", "long_strangle", "strap", "strip", "put_backspread", "tail_hedge"],
    "Bet on a range": ["butterfly", "long_condor"],
    "Bet on time": ["calendar_spread", "diagonal_spread"],
}


def strategy_legs(name: str, spot: float, forward: float, step: float, dte: int = 30, offset: float = 0.05, width: float = 0.05, far_dte: int = 60, quantity: int = 1) -> list[dict]:
    """The legs of a named strategy around the forward: ``offset`` is the distance of the short or protective strike from the money, ``width`` the wing or spread width, both as a fraction of the forward; strikes
    are rounded to ``step``. A leg is ``{type, right, strike, quantity, dte}`` (``type: "stock"`` for the shares, quantity in shares)."""
    if name not in STRATEGY_INFO:
        raise ValueError(f"unknown strategy '{name}'; choose from {sorted(STRATEGY_INFO)}")
    if not (dte >= 1 and far_dte > dte and 0.0 <= offset <= 0.5 and 0.005 <= width <= 0.5 and quantity >= 1):
        raise ValueError("dte >= 1, far_dte > dte, offset in [0, 0.5], width in [0.005, 0.5], quantity >= 1")
    r = lambda x: _round_strike(x, step)                                                                                  # noqa: E731
    atm, up, dn = r(forward), r(forward * (1.0 + offset)), r(forward * (1.0 - offset))
    wide_up, wide_dn = r(forward * (1.0 + offset + width)), r(forward * (1.0 - offset - width))
    q = quantity
    leg = lambda right, k, n, d=dte: {"type": "option", "right": right, "strike": k, "quantity": n * q, "dte": d}          # noqa: E731
    shares = {"type": "stock", "quantity": 100 * q}
    layouts = {
        "long_call": [leg("C", atm, 1)], "long_put": [leg("P", atm, 1)],
        "covered_call": [shares, leg("C", up, -1)], "protective_put": [shares, leg("P", dn, 1)],
        "bull_call_spread": [leg("C", atm, 1), leg("C", r(forward * (1.0 + width)), -1)], "bear_put_spread": [leg("P", atm, 1), leg("P", r(forward * (1.0 - width)), -1)],
        "iron_condor": [leg("P", wide_dn, 1), leg("P", dn, -1), leg("C", up, -1), leg("C", wide_up, 1)],
        "long_straddle": [leg("C", atm, 1), leg("P", atm, 1)], "long_strangle": [leg("P", dn, 1), leg("C", up, 1)],
        "short_straddle": [leg("C", atm, -1), leg("P", atm, -1)], "short_strangle": [leg("P", dn, -1), leg("C", up, -1)],
        "calendar_spread": [leg("C", atm, -1), leg("C", atm, 1, far_dte)],
        "butterfly": [leg("C", r(forward * (1.0 - width)), 1), leg("C", atm, -2), leg("C", r(forward * (1.0 + width)), 1)],
        "ratio_spread": [leg("C", atm, 1), leg("C", r(forward * (1.0 + width)), -2)],
        "collar": [shares, leg("P", dn, 1), leg("C", up, -1)], "cash_secured_put": [leg("P", dn, -1)],
        "iron_butterfly": [leg("P", r(forward * (1.0 - width)), 1), leg("P", atm, -1), leg("C", atm, -1), leg("C", r(forward * (1.0 + width)), 1)],
        "long_condor": [leg("C", wide_dn, 1), leg("C", dn, -1), leg("C", up, -1), leg("C", wide_up, 1)],
        "risk_reversal": [leg("C", up, 1), leg("P", dn, -1)], "strap": [leg("C", atm, 2), leg("P", atm, 1)], "strip": [leg("C", atm, 1), leg("P", atm, 2)],
        "call_backspread": [leg("C", atm, -1), leg("C", r(forward * (1.0 + width)), 2)], "put_backspread": [leg("P", atm, -1), leg("P", r(forward * (1.0 - width)), 2)],
        "diagonal_spread": [leg("C", atm, 1, far_dte), leg("C", up, -1)], "tail_hedge": [leg("P", wide_dn, 1)],
    }
    legs = layouts[name]
    for L in legs:
        if L["type"] == "option" and L["strike"] <= 0:
            raise ValueError("a strike came out at or below zero: reduce the offset or width")
    return legs


def chain_legs(chain: pd.DataFrame, legs: list[dict]) -> list[dict]:
    """The legs with each option's days moved to the nearest listed expiry and its implied volatility read from the chain at its strike (shares are unchanged)."""
    out = []
    for leg in legs:
        if leg["type"] == "option":
            expiry = nearest_expiry_to(chain, leg["dte"])
            leg = {**leg, "iv": iv_lookup(chain, expiry, leg["strike"]), "dte": int(round(float(chain.loc[chain["expiry"] == expiry, "T"].iloc[0]) * 365))}
        out.append(leg)
    return out


def structure_context(chain: pd.DataFrame, dte: int = 30, far_dte: int | None = None) -> dict:
    """What a structure is built around: the expiries nearest ``dte`` and ``far_dte`` (default about twice ``dte``), the forward, the at-the-money volatility and the strike step of the first, and the
    days to both."""
    e1, e2 = nearest_expiry_to(chain, dte), nearest_expiry_to(chain, far_dte if far_dte else max(dte * 2, dte + 14))
    g1 = chain[chain["expiry"] == e1]
    return {"expiry": e1, "far_expiry": e2, "forward": float(g1["F"].iloc[0]), "atm_iv": atm_iv(g1), "step": strike_step(chain, e1), "dte": int(round(float(g1["T"].iloc[0]) * 365)),
            "far_dte": int(round(float(chain.loc[chain["expiry"] == e2, "T"].iloc[0]) * 365))}


def named_structure(chain: pd.DataFrame, spot: float, name: str, dte: int = 30, offset: float = 0.05, width: float = 0.05, far_dte: int | None = None, quantity: int = 1) -> tuple[list[dict], dict]:
    """``(legs, context)`` for a named structure on this chain (``structure_context`` is the second): the strikes are set around the forward of the expiry nearest ``dte`` and rounded to its strike step,
    and each option takes the chain's own volatility at its strike."""
    ctx = structure_context(chain, dte, far_dte)
    legs = strategy_legs(name, spot, ctx["forward"], ctx["step"], ctx["dte"], offset, width, max(ctx["far_dte"], ctx["dte"] + 1), quantity)
    return chain_legs(chain, legs), ctx


def compare_structures(chain: pd.DataFrame, spot: float, names: list[str], prices, **kwargs) -> list[dict]:
    """Other structures priced on the same chain and the same settings, each with the headline numbers of ``strategy_analysis`` and its profit at the horizon at the given ``prices`` (by linear
    interpolation of its own exact curve, so the lines can be drawn on one axis). ``kwargs`` are the arguments of ``named_structure`` (``dte``, ``offset``, ``width``, ``far_dte``, ``quantity``) and,
    under ``analysis``, those of ``strategy_analysis``."""
    analysis = kwargs.pop("analysis", {})
    out = []
    for name in names:
        legs, meta = named_structure(chain, spot, name, **kwargs)
        a = strategy_analysis(legs, spot, meta["forward"], meta["atm_iv"], cash_secured=(name == "cash_secured_put"), **analysis)
        curve = np.interp(np.asarray(prices, dtype=float), a["curve"]["spot"], a["curve"]["expiry"])
        out.append({"strategy": name, "group": group_of(name), "net_debit": a["net_debit"], "net_credit": a["net_credit"], "max_profit": a["max_profit"], "max_loss": a["max_loss"], "unlimited_profit": a["unlimited_profit"],
                    "unlimited_loss": a["unlimited_loss"], "breakevens": a["breakevens"], "prob_profit": a["prob_profit"], "expected_value": a["expected_value"], "margin": a["margin"]["estimate"],
                    "curve": [round(float(x), 2) for x in curve]})
    return out


def group_of(name: str) -> str:
    """The family a named structure belongs to (``STRATEGY_GROUPS``)."""
    return next((g for g, members in STRATEGY_GROUPS.items() if name in members), "Other")


def _leg_value(leg: dict, s, horizon: float, sigma_of, rate: float, dividend: float, mult: float):
    """Value per leg (dollars, signed) at the horizon (years from now) for the spot array ``s``: intrinsic if the leg expires by the horizon, Black-Scholes with its remaining life otherwise."""
    qty = leg["quantity"]
    if leg["type"] == "stock":
        return qty * s
    rem = leg["dte"] / 365.0 - horizon
    call = leg["right"] == "C"
    if rem <= 1e-9:
        return qty * mult * (np.maximum(s - leg["strike"], 0.0) if call else np.maximum(leg["strike"] - s, 0.0))
    return qty * mult * bsm_price(s, leg["strike"], rem, rate, dividend, sigma_of(leg), call)


def _entry_cost(legs: list[dict], spot: float, rate: float, dividend: float, sigma_of, mult: float, fill: str, spread_pct: float, commission: float) -> tuple[float, list[float]]:
    """The cash paid (positive) to open the legs: the model mid, moved against you by ``spread_pct`` (full spread to cross) when ``fill`` is "worst", plus commission per contract. Returns the total and per-leg prices."""
    total, prices = 0.0, []
    for leg in legs:
        if leg["type"] == "stock":
            prices.append(spot)
            total += leg["quantity"] * spot
            continue
        mid = float(bsm_price(spot, leg["strike"], leg["dte"] / 365.0, rate, dividend, sigma_of(leg), leg["right"] == "C"))
        half = 0.5 * spread_pct * mid if fill == "worst" else 0.0
        px = mid + half if leg["quantity"] > 0 else max(mid - half, 0.0)
        prices.append(px)
        total += leg["quantity"] * mult * px + commission * abs(leg["quantity"])
    return total, prices


def _breakevens(s: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Spot values where the P&L curve crosses zero, by linear interpolation between grid points."""
    out = []
    sign = np.sign(pnl)
    for i in np.flatnonzero(sign[:-1] * sign[1:] < 0):
        out.append(float(s[i] - pnl[i] * (s[i + 1] - s[i]) / (pnl[i + 1] - pnl[i])))
    for i in np.flatnonzero(sign == 0):
        if 0 < i < len(s) - 1 and sign[i - 1] != 0 and sign[i + 1] != 0:
            out.append(float(s[i]))
    return sorted(out)


def margin_estimate(legs: list[dict], spot: float, premium: float, mult: float = MULTIPLIER, cash_secured: bool = False) -> dict:
    """An INDICATIVE initial margin. Positions with a bounded worst case (long options, spreads, condors, butterflies) need their maximum loss; naked short options need the Reg-T style
    ``max(20% x spot - out-of-the-money amount, 10% x spot)`` (puts: 10% of the strike) per contract plus the premium, taking the larger of the call and put side for a straddle or strangle; shares need 50%
    of their value. With ``cash_secured`` the short puts are held against the cash to buy the shares at the strike (strike x contracts x multiplier, less the premium): the requirement of the Cboe PutWrite
    index, which is the whole strike. Your broker's rule and exchange house margins differ: this is for comparing structures, not for sizing an account."""
    shares = sum(l["quantity"] for l in legs if l["type"] == "stock")
    stock_margin = 0.5 * max(shares, 0.0) * spot
    opts = [l for l in legs if l["type"] == "option"]
    calls = [l for l in opts if l["right"] == "C"]
    puts = [l for l in opts if l["right"] == "P"]
    covered_by_stock = max(shares, 0.0) / mult
    net_calls = sum(l["quantity"] for l in calls) + covered_by_stock                       # negative: uncovered short calls (a long call or shares covers one)
    net_puts = sum(l["quantity"] for l in puts)
    req = {"call": 0.0, "put": 0.0}
    if net_calls < 0:
        shorts = [l for l in calls if l["quantity"] < 0]
        k = min(l["strike"] for l in shorts)
        req["call"] = -net_calls * mult * max(0.2 * spot - max(k - spot, 0.0), 0.1 * spot)
    if net_puts < 0:
        shorts = [l for l in puts if l["quantity"] < 0]
        k = max(l["strike"] for l in shorts)
        req["put"] = -net_puts * mult * (k if cash_secured and not calls else max(0.2 * spot - max(spot - k, 0.0), 0.1 * k))
    naked = max(req.values())
    bounded_hint = naked == 0.0
    secured = bool(cash_secured and not calls and net_puts < 0)
    return {"stock_margin": float(stock_margin), "naked_requirement": float(naked), "defined_risk": bool(bounded_hint), "premium_received": float(max(-premium, 0.0)), "cash_secured": secured,
            "basis": "max loss of the option legs" if bounded_hint else "the strike held in cash against the put" if secured else "Reg-T style naked-option rule (indicative)"}


def _price_density(spot: float, vol: float, drift: float, horizon: float, bins: int = 56, width: float = 3.5) -> dict:
    """The probability of each price band at the horizon when the log price is normal with the given drift and volatility (the lognormal the structure's odds use): ``edges`` (``bins`` + 1 prices,
    ``width`` standard deviations either side of the median) and ``mass`` (the probability in each band, exact from the normal distribution function, so they sum to the probability inside the range)."""
    sd = vol * math.sqrt(horizon)
    centre = math.log(spot) + (drift - 0.5 * vol ** 2) * horizon
    edges = np.exp(centre + sd * np.linspace(-width, width, bins + 1))
    cdf = norm.cdf((np.log(edges) - centre) / sd)
    return {"edges": [round(float(x), 4) for x in edges], "mass": [float(x) for x in np.diff(cdf)], "median": float(math.exp(centre))}


def _pnl_grid(value_on, cost: float, spot: float, vol: float, horizon: float, horizon_days: int, prices: int = 21, steps: int = 8) -> dict:
    """The profit or loss of the legs on a coarse grid: ``prices`` spot values spread over +-2.5 standard deviations of the horizon and the days from now to the horizon (at most ``steps`` + 1 of them,
    including both ends). ``value_on(years_passed, prices)`` is the value of all the legs; ``pnl[i][j]`` is the profit at price ``spot[i]`` after ``days[j]`` days."""
    sd = vol * math.sqrt(horizon)
    s = spot * np.exp(np.linspace(-2.5, 2.5, prices) * sd)
    days = sorted({int(round(x)) for x in np.linspace(0, horizon_days, min(horizon_days, steps) + 1)})
    pnl = np.array([value_on(d / 365.0, s) - cost for d in days])
    return {"spot": [round(float(x), 4) for x in s], "days": days, "pnl": [[round(float(pnl[j][i]), 2) for j in range(len(days))] for i in range(prices)]}


def _greeks_curve(legs: list[dict], prices: list[float], rate: float, dividend: float, sigma_of, mult: float) -> dict:
    """Delta (shares), gamma (shares per dollar), vega (dollars per volatility point) and theta (dollars per day) of the legs together today, at each of ``prices``."""
    s = np.asarray(prices, dtype=float)
    out = {"delta": np.zeros_like(s), "gamma": np.zeros_like(s), "vega": np.zeros_like(s), "theta": np.zeros_like(s)}
    for leg in legs:
        if leg["type"] == "stock":
            out["delta"] += leg["quantity"]
            continue
        g = bsm_greeks(s, leg["strike"], leg["dte"] / 365.0, rate, dividend, sigma_of(leg), leg["right"] == "C")
        q = leg["quantity"] * mult
        out["delta"] += q * g["delta"]
        out["gamma"] += q * g["gamma"]
        out["vega"] += q * g["vega"] / 100.0
        out["theta"] += q * g["theta"] / 365.0
    return {"spot": [round(float(x), 4) for x in s], **{k: [round(float(x), 5) for x in v] for k, v in out.items()}}


def strategy_analysis(legs: list[dict], spot: float, forward: float, atm_vol: float, rate: float = DEFAULT_RATE, dividend: float = 0.0, skew_slope: float = 0.0, view_vol: float | None = None,
                      view_drift: float | None = None, horizon_days: int | None = None, fill: str = "mid", spread_pct: float = 0.05, commission: float = 0.65, mult: float = MULTIPLIER,
                      points: int = 241, cash_secured: bool = False) -> dict:
    """Everything the strategy builder shows for a set of legs.

    Each option leg is priced with its own ``iv`` if it has one (the dashboard takes it from the chain), else with the ATM volatility plus the skew slope times its log-strike (``skew_slope`` in volatility per unit of ln K / F, negative for equities). The P&L at the horizon (default the
    earliest expiry) is drawn at expiry and, for the curved lines, now ("T+0") and half way. Maximum profit and loss come from the expiry curve and the slope beyond the last strike (unlimited if it is not
    flat), break-evens by interpolation, the Greeks are the position's today, ``prob_profit`` and ``expected_value`` integrate the lognormal distribution of the spot at the horizon with volatility
    ``view_vol`` (default the ATM volatility, i.e. the market's own view) and drift ``view_drift`` (default the risk-neutral ``r - q``): at the market's own view the expected value is the costs, and a view
    that differs from the market's is what makes it nonzero. The distribution is lognormal with a single volatility, so with a skewed smile the expected value of a put-heavy or call-heavy
    structure is not zero even at the market's own volatility.

    Three further views are returned for the picture: ``density`` (the probability of each price band at the horizon under that same lognormal view), ``grid`` (the profit or loss at a coarse grid of prices
    and days passed, from today to the horizon) and ``greeks_curve`` (the structure's delta, gamma, vega and theta today at each price of the grid; each leg keeps its own volatility, so a leg's vega does not
    move with the price)."""
    if not legs:
        raise ValueError("add at least one leg")
    if fill not in ("mid", "worst"):
        raise ValueError("fill must be mid or worst")
    opts = [l for l in legs if l["type"] == "option"]
    if not opts:
        raise ValueError("a strategy needs at least one option leg")
    sigma_of = lambda leg: float(leg["iv"]) if leg.get("iv") else float(max(atm_vol + skew_slope * math.log(leg["strike"] / forward), 0.05))        # noqa: E731
    horizon_d = int(horizon_days if horizon_days is not None else min(l["dte"] for l in opts))
    if horizon_d < 1 or horizon_d > max(l["dte"] for l in opts):
        raise ValueError("the horizon must be between one day and the last expiry")
    H = horizon_d / 365.0
    cost, prices = _entry_cost(legs, spot, rate, dividend, sigma_of, mult, fill, spread_pct, commission)
    lo, hi = max(spot * 0.5, 1e-6), spot * 1.5
    sd = atm_vol * math.sqrt(H)
    lo, hi = min(lo, spot * math.exp(-4.5 * sd)), max(hi, spot * math.exp(4.5 * sd))
    s = np.unique(np.r_[0.0, np.linspace(lo, hi, points), [l["strike"] for l in opts]])
    value_on = lambda h, prices: sum(_leg_value(l, prices, h, sigma_of, rate, dividend, mult) for l in legs)             # noqa: E731
    value_at = lambda h: value_on(h, s)                                                                                      # noqa: E731
    exp_pnl = value_at(H) - cost
    now_pnl = value_at(0.0) - cost
    mid_pnl = value_at(H / 2.0) - cost
    # extremes: finite where the curve is flat beyond the strikes; unlimited where its slope is not
    top_slope = sum(l["quantity"] * (mult if l["right"] == "C" else 0.0) for l in opts) + sum(l["quantity"] for l in legs if l["type"] == "stock") if not any(l["dte"] > horizon_d for l in opts) else None
    if top_slope is None:                                                           # later legs still have value at the horizon: read the slope from the curve instead
        top_slope = float((exp_pnl[-1] - exp_pnl[-2]) / (s[-1] - s[-2]))
    flat = 0.01 * mult * max(abs(l["quantity"]) for l in opts)                       # a slope within 1% of one contract's delta is flat: a far leg's delta is never exactly 1
    unlimited_profit, unlimited_loss = bool(top_slope > flat), bool(top_slope < -flat)
    max_profit = None if unlimited_profit else float(exp_pnl.max())
    max_loss = None if unlimited_loss else float(exp_pnl.min())
    stock_cost = sum(l["quantity"] for l in legs if l["type"] == "stock") * spot
    option_cost = cost - stock_cost
    margin = margin_estimate(legs, spot, option_cost if option_cost < 0 else 0.0, mult, cash_secured)
    if margin["cash_secured"]:                                                      # the strike is held in cash and the premium received reduces what has to be put up
        margin_total = max(margin["naked_requirement"] - margin["premium_received"], 0.0) + margin["stock_margin"]
    elif not margin["defined_risk"]:                                                # naked shorts: the requirement already counts the premium received, which stays on deposit
        margin_total = margin["naked_requirement"] + margin["premium_received"] + margin["stock_margin"]
    elif stock_cost:                                                                # covered calls and protective puts: half the shares' value plus any option premium paid
        margin_total = margin["stock_margin"] + max(option_cost, 0.0)
    else:                                                                           # spreads, condors, butterflies and long options: the worst loss
        margin_total = max(-(max_loss or 0.0), max(option_cost, 0.0), 0.0)
    # probability of profit and expected value under a lognormal spot at the horizon
    vv = float(view_vol) if view_vol not in (None, "") else atm_vol
    mu = (rate - dividend) if view_drift in (None, "") else float(view_drift)
    if vv <= 0:
        raise ValueError("the view volatility must be positive")
    z = np.linspace(-6.0, 6.0, 2401)
    w = norm.pdf(z)
    w = w / w.sum()
    s_T = spot * np.exp((mu - 0.5 * vv ** 2) * H + vv * math.sqrt(H) * z)
    pnl_T = sum(_leg_value(l, s_T, H, sigma_of, rate, dividend, mult) for l in legs) - cost
    pop = float(w[pnl_T > 0].sum())
    ev = float((w * pnl_T).sum())
    greeks = position_values([{**l, "iv": sigma_of(l)} if l["type"] == "option" else l for l in legs], spot, rate, dividend, pd.Timestamp.today().normalize())
    g = {"delta_shares": float(greeks["delta"].sum()), "delta_dollars": float(greeks["delta"].sum() * spot), "gamma": float(greeks["gamma"].sum()), "gamma_pnl_1pct": float(0.5 * greeks["gamma"].sum() * (0.01 * spot) ** 2),
         "vega_per_point": float(greeks["vega"].sum()), "theta_per_day": float(greeks["theta"].sum()), "rho_per_1pct": float(greeks["rho"].sum())}
    risk = abs(max_loss) if max_loss is not None else None
    density = _price_density(spot, vv, mu, H)
    grid = _pnl_grid(value_on, cost, spot, vv, H, horizon_d)
    gcurve = _greeks_curve(legs, grid["spot"], rate, dividend, sigma_of, mult)
    legs_out = [{**l, "price": (round(p, 4) if l["type"] == "option" else None), "iv": (round(sigma_of(l), 5) if l["type"] == "option" else None)} for l, p in zip(legs, prices)]
    flags: list[str] = []
    add_flag(flags, unlimited_loss, "unlimited loss on the upside" if top_slope < 0 else "unlimited loss")
    add_flag(flags, not margin["defined_risk"] and not margin["cash_secured"], "naked short options: margin and gap risk are large, and the estimate is indicative only")
    return {"legs": legs_out, "cost": float(cost), "net_debit": float(max(cost, 0.0)), "net_credit": float(max(-cost, 0.0)), "horizon_days": horizon_d, "spot": float(spot), "forward": float(forward),
            "max_profit": max_profit, "max_loss": max_loss, "unlimited_profit": unlimited_profit, "unlimited_loss": unlimited_loss, "breakevens": _breakevens(s, exp_pnl), "greeks": g,
            "margin": {**margin, "estimate": float(margin_total)}, "prob_profit": pop, "expected_value": ev, "view_vol": vv, "view_drift": mu,
            "reward_risk": (max_profit / risk if (max_profit is not None and risk) else None),
            "curve": {"spot": [round(float(x), 4) for x in s], "expiry": [round(float(x), 2) for x in exp_pnl], "now": [round(float(x), 2) for x in now_pnl], "half": [round(float(x), 2) for x in mid_pnl]},
            "density": density, "grid": grid, "greeks_curve": gcurve, "risk_flags": flags}


# ------------------------------------------------------------------------------------------------------------------ strategy backtests
BACKTEST_STRATEGIES = ("short_put", "put_spread", "covered_call", "collar", "protective_put", "short_strangle", "iron_condor", "iron_butterfly", "delta_hedged_straddle", "risk_reversal", "calendar_spread")


BACKTEST_TEXT = {
    "short_put": "Sell a cash-secured put of about 25 delta, hold it to a week before expiry and roll (the premium-selling idea behind the Cboe PUT index).",
    "covered_call": "Own the underlying and sell a call of about 30 delta each month (Cboe BXM-style).",
    "short_strangle": "Sell a put and a call of about 16 delta each; optionally delta-hedge daily so that only volatility is traded.",
    "iron_condor": "The short strangle with bought wings at 5 delta: the same premium sale with a defined worst case.",
    "delta_hedged_straddle": "Sell (or buy) the at-the-money straddle and hedge the delta daily: a pure bet on realised against implied volatility.",
    "risk_reversal": "Buy a 25-delta call and sell a 25-delta put, delta-hedged: the skew premium.",
    "calendar_spread": "Sell the front-month at-the-money option and buy the back month: long the term structure.",
    "put_spread": "Sell a 25-delta put and buy a 10-delta put against it each period: the short put's premium with the loss limited to the width of the spread.",
    "collar": "Own the underlying, buy a put 5% below the market and sell a call 5% above it each period (the Cboe CLL index buys its put at 95% and sells its call at 110%).",
    "protective_put": "Own the underlying and buy a put 5% below the market each period (the Cboe PPUT index): what insurance costs.",
    "iron_butterfly": "Sell the at-the-money put and call and buy wings 5% out of the money each period (the Cboe BFLY index): the most premium for a price that stays put.",
}


def _param(default, lo=None, hi=None, kind: str = "float", choices=None, text: str = "") -> dict:
    return {"default": default, "lo": lo, "hi": hi, "kind": kind, "choices": choices, "text": text}


_ROLL = {"dte": _param(45, 7, 400, "int", text="days to expiry when the package is opened"), "close_dte": _param(7, 0, 60, "int", text="close this many days before expiry and roll"),
         "size": _param(1.0, 0.1, 100.0, text="contracts per leg"), "min_open_dte": _param(20, 1, 200, "int", text="do not open a package with fewer days than this")}
# What each rolling option strategy accepts (the page builds its form from this, the API refuses anything else, and ``derivatives.strategies`` receives the validated values).
BACKTEST_PARAMS: dict[str, dict[str, dict]] = {
    "short_put": {"delta": _param(0.25, 0.02, 0.6, text="absolute delta of the put sold"), **_ROLL},
    "covered_call": {"delta": _param(0.30, 0.02, 0.6, text="delta of the call sold against the shares"), **_ROLL},
    "short_strangle": {"delta": _param(0.16, 0.02, 0.6, text="delta of the put and the call sold"), "hedge": _param(False, kind="bool", text="delta-hedge the position daily"), **_ROLL},
    "iron_condor": {"short_delta": _param(0.16, 0.02, 0.6, text="delta of the short put and call"), "wing_delta": _param(0.05, 0.01, 0.4, text="delta of the protective wings"), **_ROLL},
    "delta_hedged_straddle": {"side": _param(-1, kind="choice", choices=[-1, 1], text="-1 sells the straddle (short volatility), +1 buys it"), "hedge_threshold": _param(0.0, 0.0, 1000.0, text="hedge when the net delta exceeds this"), **_ROLL},
    "risk_reversal": {"delta": _param(0.25, 0.02, 0.6, text="delta of the call and the put"), "side": _param(1, kind="choice", choices=[-1, 1], text="+1 is long the call and short the put"), **_ROLL},
    "calendar_spread": {"front_dte": _param(30, 7, 200, "int", text="days to expiry of the option sold"), "back_dte": _param(90, 21, 400, "int", text="days to expiry of the option bought"),
                        **{k: v for k, v in _ROLL.items() if k != "dte"}},
    "put_spread": {"short_delta": _param(0.25, 0.05, 0.6, text="absolute delta of the put sold"), "long_delta": _param(0.10, 0.01, 0.4, text="absolute delta of the put bought against it (further out of the money)"), **_ROLL},
    "iron_butterfly": {"wing": _param(0.05, 0.01, 0.3, text="distance of the bought put and call from the price, as a fraction of it"), **_ROLL},
    "collar": {"put_moneyness": _param(0.95, 0.6, 0.99, text="strike of the put bought, as a fraction of the price"), "call_moneyness": _param(1.05, 1.01, 1.4, text="strike of the call sold, as a fraction of the price"), **_ROLL},
    "protective_put": {"put_moneyness": _param(0.95, 0.6, 0.99, text="strike of the put bought, as a fraction of the price"), **_ROLL},
}


def backtest_params(name: str) -> list[dict]:
    """The settings of a rolling option strategy as form fields: ``name, default, lo, hi, kind, choices, text``."""
    if name not in BACKTEST_PARAMS:
        raise ValueError(f"unknown strategy '{name}'; choose from {list(BACKTEST_PARAMS)}")
    return [{"name": k, **v} for k, v in BACKTEST_PARAMS[name].items()]


def parse_backtest_params(name: str, given: dict | None) -> dict:
    """The settings in ``given`` checked against ``BACKTEST_PARAMS`` and cast to the right types; a name the strategy does not take, or a value out of range, raises ``ValueError``."""
    if name not in BACKTEST_PARAMS:
        raise ValueError(f"unknown strategy '{name}'; choose from {list(BACKTEST_PARAMS)}")
    if given is not None and not isinstance(given, dict):
        raise ValueError("params must be an object")
    spec, out = BACKTEST_PARAMS[name], {}
    for key, raw in (given or {}).items():
        if key not in spec:
            raise ValueError(f"{name} has no setting '{key}'; it takes {list(spec)}")
        p = spec[key]
        if raw is None or raw == "":
            continue
        if p["kind"] == "bool":
            if isinstance(raw, str):
                raw = raw.strip().lower() in ("1", "true", "yes", "on")
            out[key] = bool(raw)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number") from None
        if p["kind"] == "choice":
            if value not in p["choices"]:
                raise ValueError(f"{key} must be one of {p['choices']}")
            out[key] = type(p["default"])(value)
        else:
            if not math.isfinite(value) or not p["lo"] <= value <= p["hi"]:
                raise ValueError(f"{key} must be between {p['lo']:g} and {p['hi']:g}")
            out[key] = int(round(value)) if p["kind"] == "int" else value
    if name == "calendar_spread" and out.get("back_dte", spec["back_dte"]["default"]) <= out.get("front_dte", spec["front_dte"]["default"]) + 7:
        raise ValueError("back_dte must be at least a week longer than front_dte")
    if name == "iron_condor" and out.get("wing_delta", spec["wing_delta"]["default"]) >= out.get("short_delta", spec["short_delta"]["default"]):
        raise ValueError("wing_delta must be smaller than short_delta (the wings are further out of the money)")
    if name == "put_spread" and out.get("long_delta", spec["long_delta"]["default"]) >= out.get("short_delta", spec["short_delta"]["default"]):
        raise ValueError("long_delta must be smaller than short_delta (the put bought is further out of the money)")
    return out


TIMING_DEFAULTS = {"window": 21, "threshold": 0.0}


def parse_timing(given: dict | None) -> dict | None:
    """The variance-premium filter of a backtest, checked: ``{"window": days of realised volatility, "threshold": volatility points}``. The strategy is traded only while the implied volatility of the
    30-day expiry exceeds the trailing realised volatility by more than the threshold (Bollerslev, Tauchen and Zhou 2009; Carr and Wu 2009), and closed while it does not. ``None`` or ``{}``: always trade."""
    if not given:
        return None
    if not isinstance(given, dict) or set(given) - set(TIMING_DEFAULTS):
        raise ValueError(f"timing takes {sorted(TIMING_DEFAULTS)}")
    try:
        window, threshold = int(round(float(given.get("window", 21)))), float(given.get("threshold", 0.0))
    except (TypeError, ValueError):
        raise ValueError("timing window and threshold must be numbers") from None
    if not 5 <= window <= 126 or not -0.2 <= threshold <= 0.5:
        raise ValueError("timing window must be between 5 and 126 days and the threshold between -0.20 and 0.50 (volatility, as a fraction)")
    return {"window": window, "threshold": threshold}


_MARKETS: "OrderedDict[tuple, object]" = OrderedDict()
_MARKET_LOCK = threading.Lock()


def simulated_market(n_days: int = 252, seed: int = 0, premium: float = 0.25):
    """A reproducible synthetic option market (daily chains for ``n_days``), cached because building one takes seconds: ``(n_days, seed, premium)`` keys the cache of the last three."""
    from ..derivatives.synthetic import generate_market

    key = (int(n_days), int(seed), float(premium))
    with _MARKET_LOCK:
        if key in _MARKETS:
            _MARKETS.move_to_end(key)
            return _MARKETS[key]
    market = generate_market(SyntheticMarketSpec(variance_premium=premium, max_dte=150, weekly_dte=21), n_days=int(n_days), seed=int(seed))
    with _MARKET_LOCK:
        _MARKETS[key] = market
        while len(_MARKETS) > 3:
            _MARKETS.popitem(last=False)
    return market


def _timed(strategy, timing: dict | None):
    """The strategy as given, or trading only while the variance premium is positive (``VRPTimed``)."""
    if not timing:
        return strategy
    from ..derivatives.strategies import VRPTimed

    return VRPTimed(strategy, window=timing["window"], threshold=timing["threshold"])


def _backtest_result(name: str, params: dict | None, n_days: int, seed: int, capital: float, premium: float, commission: float, fill_fraction: float, timing: dict | None):
    """``(result, market)``: the strategy run on the simulated market (``OptionBacktester``), possibly behind the variance-premium filter."""
    from ..derivatives.backtest import OptionBacktester
    from ..derivatives.strategies import make_strategy

    if name not in BACKTEST_STRATEGIES:
        raise ValueError(f"unknown strategy '{name}'; choose from {list(BACKTEST_STRATEGIES)}")
    if not 60 <= n_days <= 756:
        raise ValueError("days must be between 60 and 756")
    market = simulated_market(n_days, seed, premium)
    spot = market.underlying["spot"]
    bt = OptionBacktester(market._by_date, spot, capital=float(capital), commission=float(commission), fill_fraction=float(fill_fraction))
    return bt.run(_timed(make_strategy(name, **(params or {})), parse_timing(timing))), spot


def run_strategy_backtest(name: str, params: dict | None = None, n_days: int = 252, seed: int = 0, capital: float = 100_000.0, premium: float = 0.25, commission: float = 0.65,
                          fill_fraction: float = 1.0, timing: dict | None = None) -> dict:
    """Backtest one of the rolling option strategies on the simulated option market and summarise it with the platform's own performance and tail-risk measures. The market is a MODEL: it pays the
    variance risk premium it was built with, so the result says whether the machinery books costs, hedges and settles correctly, not whether the premium exists. ``timing`` (see ``parse_timing``) trades the
    strategy only while the trailing variance premium is positive. Beside the equity curve the result carries the views the other workspaces show for a strategy: the deepest falls, the volatility
    through time, the years, months and distribution of daily returns, and every risk measure (``risk.performance``)."""
    from ..backtest.metrics import conditional_value_at_risk, performance_summary, value_at_risk
    from ..risk import performance

    timing = parse_timing(timing)
    result, spot = _backtest_result(name, params, n_days, seed, capital, premium, commission, fill_fraction, timing)
    returns = result.returns.dropna()
    summary = performance_summary(returns)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    bench = spot / spot.iloc[0]
    greeks = result.greeks.reindex(equity.index)
    every = slice(None, None, 2)
    rolling = performance.rolling_volatility(returns, 63).reindex(equity.index)
    years = [{"year": int(y), "value": float((1.0 + r).prod() - 1.0), "days": int(len(r))} for y, r in returns.groupby(returns.index.year)]
    held = result.positions.reindex(equity.index)
    in_market = float(((held["open_contracts"] > 0) | (held["underlying_units"].abs() > 0)).mean()) if len(held) else float("nan")
    return {"strategy": name, "params": dict(params or {}), "timing": timing, "n_days": int(n_days), "seed": int(seed), "capital": float(capital), "summary": {k: float(v) for k, v in summary.items() if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)},
            "start": summary.get("start"), "end": summary.get("end"),
            "option_summary": {k: float(v) for k, v in result.summary().items()}, "var_95": float(value_at_risk(returns, 0.95)), "cvar_95": float(conditional_value_at_risk(returns, 0.95)),
            "dates": [str(d.date()) for d in equity.index[every]], "equity": [round(float(v), 5) for v in equity.to_numpy()[every]], "benchmark": [round(float(v), 5) for v in bench.reindex(equity.index).to_numpy()[every]],
            "drawdown": [round(float(v), 5) for v in drawdown.to_numpy()[every]], "drawdown_episodes": performance.drawdown_episodes(returns, 5), "rolling_vol": [None if not np.isfinite(v) else round(float(v), 4) for v in rolling.to_numpy()[every]],
            "years": years, "calendar_months": performance.calendar_month_profile(returns), "monthly": performance.monthly_returns(returns), "distribution": performance.return_distribution(returns),
            "risk": performance.risk_metrics(returns), "share_in_market": in_market,
            "net_delta": [None if not np.isfinite(v) else round(float(v), 4) for v in greeks["delta"].to_numpy()[every]] if "delta" in greeks else [],
            "attribution": {c: float(result.attribution[c].sum()) for c in result.attribution.columns}, "trades": int(len(result.trades)), "skipped_orders": int(result.skipped_orders),
            "recent_trades": [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))) for k, v in row.items()}
                              for row in result.trades.tail(12).to_dict("records")],
            "notes": ["The option market is simulated (an SSVI surface with a built-in variance risk premium and jump risk); it is not historical data.",
                      "Returns are daily P&L over a fixed capital: margin and funding are not modelled."]}


SWEEP_POINTS = (2, 9)


def _headline(out: dict) -> dict:
    """The few numbers a sweep or a comparison tabulates for a finished backtest."""
    s = out["summary"]
    return {"sharpe": s.get("sharpe"), "cagr": s.get("cagr"), "ann_vol": s.get("ann_vol"), "max_drawdown": s.get("max_drawdown"), "total_return": s.get("total_return"), "worst_day": s.get("worst_day"),
            "cvar_95": out["cvar_95"], "trades": out["trades"], "costs": out["option_summary"]["costs_total"], "share_in_market": out["share_in_market"]}


def check_sweep(name: str, param: str, values, params: dict | None = None) -> list[dict]:
    """The settings of each run of a sweep, checked: ``param`` must be a numeric setting of ``name``, there must be 2 to 9 values, and each must pass ``parse_backtest_params`` together with the strategy's
    other settings. Returns the full checked settings of each run."""
    if name not in BACKTEST_PARAMS:
        raise ValueError(f"unknown strategy '{name}'; choose from {list(BACKTEST_PARAMS)}")
    spec = BACKTEST_PARAMS[name].get(param)
    if spec is None or spec["kind"] not in ("float", "int"):
        raise ValueError(f"{name} has no numeric setting '{param}' to sweep; it takes {[k for k, v in BACKTEST_PARAMS[name].items() if v['kind'] in ('float', 'int')]}")
    if not isinstance(values, (list, tuple)) or not SWEEP_POINTS[0] <= len(values) <= SWEEP_POINTS[1]:
        raise ValueError(f"give between {SWEEP_POINTS[0]} and {SWEEP_POINTS[1]} values to sweep")
    return [parse_backtest_params(name, {**(params or {}), param: v}) for v in values]


def run_strategy_sweep(name: str, param: str, values: list, params: dict | None = None, progress=None, **kwargs) -> dict:
    """The same backtest at each of ``values`` (2 to 9) of one numeric setting ``param`` of the strategy, everything else as in ``params``: how the result moves as the setting does, which is how a
    setting is fine-tuned and, as importantly, how fragile the choice is (a Sharpe ratio that peaks at one value and falls away either side is a fit to this market, not a finding). ``kwargs`` are those of
    ``run_strategy_backtest``. Every value is checked against the setting's own range and against the strategy's other settings (``check_sweep``)."""
    runs = check_sweep(name, param, values, params)
    rows = []
    for i, checked in enumerate(runs):
        if progress:
            progress(f"{param.replace('_', ' ')} = {checked[param]:g} ({i + 1} of {len(runs)})")
        rows.append({"value": checked[param], **_headline(run_strategy_backtest(name, checked, **kwargs))})
    default = BACKTEST_PARAMS[name][param]["default"]
    return {"strategy": name, "param": param, "rows": rows, "base": parse_backtest_params(name, params).get(param, default), "default": default, "params": dict(params or {})}


def run_strategy_comparison(names: list | None = None, progress=None, **kwargs) -> dict:
    """Every rolling strategy (or ``names``) at its default settings on the SAME simulated market, so that they can be set side by side: their headline numbers and their equity curves. ``kwargs`` are
    those of ``run_strategy_backtest`` (days, seed, capital, premium, commission, fill and timing)."""
    names = list(names or BACKTEST_STRATEGIES)
    unknown = [n for n in names if n not in BACKTEST_STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategy '{unknown[0]}'; choose from {list(BACKTEST_STRATEGIES)}")
    if len(names) < 2:
        raise ValueError("compare at least two strategies")
    rows, curves, dates, bench = [], {}, None, None
    for i, name in enumerate(names):
        if progress:
            progress(f"{name.replace('_', ' ')} ({i + 1} of {len(names)})")
        out = run_strategy_backtest(name, None, **kwargs)
        rows.append({"strategy": name, **_headline(out)})
        curves[name] = out["equity"]
        dates, bench = out["dates"], out["benchmark"]
    return {"rows": rows, "dates": dates, "equity": curves, "benchmark": bench, "params": {k: v for k, v in kwargs.items() if k in ("n_days", "seed", "capital", "premium", "commission", "fill_fraction", "timing")}}
