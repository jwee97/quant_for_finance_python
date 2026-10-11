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
    """The generator's settings for an underlying: strikes every 0.25% of the spot (rounded), listed out to three standard deviations of each expiry, long-run implied variance at ``long_run_vol`` squared, and ``premium`` as the implied-over-realised variance gap."""
    if spot <= 0 or vol <= 0 or long_run_vol <= 0:
        raise ValueError("spot and volatilities must be positive")
    if not -0.5 <= premium <= 2.0:
        raise ValueError("variance premium must be between -0.5 and 2.0")
    return SyntheticMarketSpec(s0=spot, rate=rate, dividend=dividend, theta_q=long_run_vol ** 2, variance_premium=premium, strike_step=nice_step(spot * 0.0025), near_width=3.0, max_dte=400, weekly_dte=35)


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
            "expiries": expiry_list(chain)}


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


def underlying_context(symbol: str, prices: pd.Series, volume: pd.Series | None = None) -> UnderlyingContext:
    """Spot, realised volatilities and the long-run volatility (the root-mean-square of the 21-day realised volatility over up to five years) from a price history."""
    p = prices.dropna()
    if len(p) < 130:
        raise ValueError(f"{symbol} has {len(p)} prices; the options workspace needs at least 130 days of history")
    rv = realized_vol(p, 21).dropna()
    rv63 = float(realized_vol(p, 63).dropna().iloc[-1])
    dollar_volume = float((p * volume.reindex(p.index)).tail(60).median()) if volume is not None else float("nan")
    return UnderlyingContext(symbol=symbol, spot=float(p.iloc[-1]), as_of=p.index[-1], rv_21=float(rv.iloc[-1]), rv_63=rv63, long_run_vol=float(np.sqrt((rv.tail(1260) ** 2).mean())), rv_history=rv,
                             dollar_volume=dollar_volume)


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


def overview(chain: pd.DataFrame, ctx: UnderlyingContext, premium: float = 0.15, macro: pd.DataFrame | None = None, simulated: bool = True) -> dict:
    """The cards above the chain: spot, realised volatility, ATM implied volatility (30-day, constant maturity), IV rank and percentile over one year, put/call ratios, the nearest expiry, the term slope, and
    the risk flags worth reading. For a simulated chain the IV history is the model applied to the realised-volatility history, so the rank is the rank of realised volatility in a new coat and is labelled
    as such; the real index (when the platform has one) is shown beside it."""
    term = term_structure(chain)
    tm = term_metrics(term)
    near = nearest_expiry(chain)
    g_near = _expiry_slice(chain, near)
    iv30 = tm["iv_30d"]
    if simulated:
        hist = model_atm_iv(ctx.rv_history, ctx.long_run_vol, 30.0, premium).tail(252)
        basis = "model: the surface applied to the trailing realised-volatility history (not an observed implied-volatility history)"
    else:
        hist, basis = pd.Series(dtype=float), "history of ATM implied volatility is not in the file; rank unavailable"
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
    snap = AssetSnapshot(ctx.symbol, "option", ctx.spot, float(iv30) if np.isfinite(iv30) else ctx.rv_21, liquid, flags, str(ctx.as_of.date()), "simulated" if simulated else "user_file")
    ref = reference_index_view(ctx.symbol, macro, interpolate_iv(term, REFERENCE_INDEX.get(ctx.symbol.upper(), ("", 30))[1]), ctx.as_of)
    return {"symbol": ctx.symbol, "spot": ctx.spot, "as_of": str(ctx.as_of.date()), "realized_vol_21d": ctx.rv_21, "realized_vol_63d": ctx.rv_63, "long_run_vol": ctx.long_run_vol,
            "atm_iv_30d": iv30, "atm_iv_90d": tm["iv_90d"], "iv_premium": iv30 - ctx.rv_21 if np.isfinite(iv30) else None, "iv_rank": rank, "iv_percentile": pct, "iv_rank_basis": basis,
            "put_call_volume": put_v / call_v if call_v > 0 else None, "put_call_oi": put_o / call_o if call_o > 0 else None, "nearest_expiry": str(pd.Timestamp(near).date()),
            "nearest_dte": int(round(float(g_near["T"].iloc[0]) * 365)), "term_slope": tm["term_slope"], "term_inverted": tm["inverted"], "skew": skew, "reference_index": ref,
            "atm_spread_pct": atm_spread_pct, "stale_share": stale_share, "snapshot": snap.to_dict(), "risk_flags": flags}


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
}
STRATEGY_PARAMS = {"offset": 0.05, "width": 0.05, "dte": 30, "far_dte": 60}


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
    }
    legs = layouts[name]
    for L in legs:
        if L["type"] == "option" and L["strike"] <= 0:
            raise ValueError("a strike came out at or below zero: reduce the offset or width")
    return legs


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


def margin_estimate(legs: list[dict], spot: float, premium: float, mult: float = MULTIPLIER) -> dict:
    """An INDICATIVE initial margin. Positions with a bounded worst case (long options, spreads, condors, butterflies) need their maximum loss; naked short options need the Reg-T style
    ``max(20% x spot - out-of-the-money amount, 10% x spot)`` (puts: 10% of the strike) per contract plus the premium, taking the larger of the call and put side for a straddle or strangle; shares need 50%
    of their value. Your broker's rule and exchange house margins differ: this is for comparing structures, not for sizing an account."""
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
        req["put"] = -net_puts * mult * max(0.2 * spot - max(spot - k, 0.0), 0.1 * k)
    naked = max(req.values())
    bounded_hint = naked == 0.0
    return {"stock_margin": float(stock_margin), "naked_requirement": float(naked), "defined_risk": bool(bounded_hint), "premium_received": float(max(-premium, 0.0)),
            "basis": "max loss of the option legs" if bounded_hint else "Reg-T style naked-option rule (indicative)"}


def strategy_analysis(legs: list[dict], spot: float, forward: float, atm_vol: float, rate: float = DEFAULT_RATE, dividend: float = 0.0, skew_slope: float = 0.0, view_vol: float | None = None,
                      view_drift: float | None = None, horizon_days: int | None = None, fill: str = "mid", spread_pct: float = 0.05, commission: float = 0.65, mult: float = MULTIPLIER,
                      points: int = 241) -> dict:
    """Everything the strategy builder shows for a set of legs.

    Each option leg is priced with its own ``iv`` if it has one (the dashboard takes it from the chain), else with the ATM volatility plus the skew slope times its log-strike (``skew_slope`` in volatility per unit of ln K / F, negative for equities). The P&L at the horizon (default the
    earliest expiry) is drawn at expiry and, for the curved lines, now ("T+0") and half way. Maximum profit and loss come from the expiry curve and the slope beyond the last strike (unlimited if it is not
    flat), break-evens by interpolation, the Greeks are the position's today, ``prob_profit`` and ``expected_value`` integrate the lognormal distribution of the spot at the horizon with volatility
    ``view_vol`` (default the ATM volatility, i.e. the market's own view) and drift ``view_drift`` (default the risk-neutral ``r - q``): at the market's own view the expected value is the costs, and a view
    that differs from the market's is what makes it nonzero. The distribution is lognormal with a single volatility, so with a skewed smile the expected value of a put-heavy or call-heavy
    structure is not zero even at the market's own volatility."""
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
    value_at = lambda h: sum(_leg_value(l, s, h, sigma_of, rate, dividend, mult) for l in legs)                          # noqa: E731
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
    margin = margin_estimate(legs, spot, option_cost if option_cost < 0 else 0.0, mult)
    if not margin["defined_risk"]:                                                  # naked shorts: the requirement already counts the premium received, which stays on deposit
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
    legs_out = [{**l, "price": (round(p, 4) if l["type"] == "option" else None), "iv": (round(sigma_of(l), 5) if l["type"] == "option" else None)} for l, p in zip(legs, prices)]
    flags: list[str] = []
    add_flag(flags, unlimited_loss, "unlimited loss on the upside" if top_slope < 0 else "unlimited loss")
    add_flag(flags, not margin["defined_risk"], "naked short options: margin and gap risk are large, and the estimate is indicative only")
    return {"legs": legs_out, "cost": float(cost), "net_debit": float(max(cost, 0.0)), "net_credit": float(max(-cost, 0.0)), "horizon_days": horizon_d, "spot": float(spot), "forward": float(forward),
            "max_profit": max_profit, "max_loss": max_loss, "unlimited_profit": unlimited_profit, "unlimited_loss": unlimited_loss, "breakevens": _breakevens(s, exp_pnl), "greeks": g,
            "margin": {**margin, "estimate": float(margin_total)}, "prob_profit": pop, "expected_value": ev, "view_vol": vv, "view_drift": mu,
            "reward_risk": (max_profit / risk if (max_profit is not None and risk) else None),
            "curve": {"spot": [round(float(x), 4) for x in s], "expiry": [round(float(x), 2) for x in exp_pnl], "now": [round(float(x), 2) for x in now_pnl], "half": [round(float(x), 2) for x in mid_pnl]},
            "risk_flags": flags}


# ------------------------------------------------------------------------------------------------------------------ strategy backtests
BACKTEST_STRATEGIES = ("short_put", "covered_call", "short_strangle", "iron_condor", "delta_hedged_straddle", "risk_reversal", "calendar_spread")


BACKTEST_TEXT = {
    "short_put": "Sell a cash-secured put of about 25 delta, hold it to a week before expiry and roll (the premium-selling idea behind the Cboe PUT index).",
    "covered_call": "Own the underlying and sell a call of about 30 delta each month (Cboe BXM-style).",
    "short_strangle": "Sell a put and a call of about 16 delta each; optionally delta-hedge daily so that only volatility is traded.",
    "iron_condor": "The short strangle with bought wings at 5 delta: the same premium sale with a defined worst case.",
    "delta_hedged_straddle": "Sell (or buy) the at-the-money straddle and hedge the delta daily: a pure bet on realised against implied volatility.",
    "risk_reversal": "Buy a 25-delta call and sell a 25-delta put, delta-hedged: the skew premium.",
    "calendar_spread": "Sell the front-month at-the-money option and buy the back month: long the term structure.",
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
    return out


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


def run_strategy_backtest(name: str, params: dict | None = None, n_days: int = 252, seed: int = 0, capital: float = 100_000.0, premium: float = 0.25, commission: float = 0.65,
                          fill_fraction: float = 1.0) -> dict:
    """Backtest one of the rolling option strategies on the simulated option market and summarise it with the platform's own performance and tail-risk measures. The market is a MODEL: it pays the
    variance risk premium it was built with, so the result says whether the machinery books costs, hedges and settles correctly, not whether the premium exists."""
    from ..backtest.metrics import conditional_value_at_risk, performance_summary, value_at_risk
    from ..derivatives.backtest import OptionBacktester
    from ..derivatives.strategies import make_strategy

    if name not in BACKTEST_STRATEGIES:
        raise ValueError(f"unknown strategy '{name}'; choose from {list(BACKTEST_STRATEGIES)}")
    if not 60 <= n_days <= 756:
        raise ValueError("days must be between 60 and 756")
    market = simulated_market(n_days, seed, premium)
    spot = market.underlying["spot"]
    bt = OptionBacktester(market._by_date, spot, capital=float(capital), commission=float(commission), fill_fraction=float(fill_fraction))
    result = bt.run(make_strategy(name, **(params or {})))
    returns = result.returns.dropna()
    summary = performance_summary(returns)
    equity = (1.0 + returns).cumprod()
    bench = spot / spot.iloc[0]
    greeks = result.greeks.reindex(equity.index)
    return {"strategy": name, "params": dict(params or {}), "n_days": int(n_days), "seed": int(seed), "capital": float(capital), "summary": {k: float(v) for k, v in summary.items() if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)},
            "start": summary.get("start"), "end": summary.get("end"),
            "option_summary": {k: float(v) for k, v in result.summary().items()}, "var_95": float(value_at_risk(returns, 0.95)), "cvar_95": float(conditional_value_at_risk(returns, 0.95)),
            "dates": [str(d.date()) for d in equity.index[::2]], "equity": [round(float(v), 5) for v in equity.to_numpy()[::2]], "benchmark": [round(float(v), 5) for v in bench.reindex(equity.index).to_numpy()[::2]],
            "net_delta": [None if not np.isfinite(v) else round(float(v), 4) for v in greeks["delta"].to_numpy()[::2]] if "delta" in greeks else [],
            "attribution": {c: float(result.attribution[c].sum()) for c in result.attribution.columns}, "trades": int(len(result.trades)), "skipped_orders": int(result.skipped_orders),
            "recent_trades": [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))) for k, v in row.items()}
                              for row in result.trades.tail(12).to_dict("records")],
            "notes": ["The option market is simulated (an SSVI surface with a built-in variance risk premium and jump risk); it is not historical data.",
                      "Returns are daily P&L over a fixed capital: margin and funding are not modelled."]}
