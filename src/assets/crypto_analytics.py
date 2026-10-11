"""The crypto workspace: trend, funding, basis, cross-venue spreads and liquidity for spot and perpetual markets. Pure functions over daily series.

**What is real and what is not.** The repository holds the Deribit perpetual's daily closes, hourly funding and index price for BTC and ETH (2018/19 to date) and DefiLlama's total stablecoin supply. Those
are ``platform`` data and are used as they are. It holds NO open interest, exchange net flows, order books or quotes from other venues, and no history for other coins. Where the workspace needs them it
SIMULATES them and says so (``source: simulated``): other venues' funding and quotes are Deribit's with venue-specific offsets and noise; SOL, BNB and XRP are generated from BTC with a beta and their own noise;
the order book is a parametric depth profile. A simulated number is a way to exercise the page and the code, never a quote. A file in ``data/user/crypto/<SYMBOL>.csv`` (``time, spot`` and optionally ``perp``,
``funding_daily`` or ``funding_8h``) replaces the simulation for that symbol (``source: user_file``).

Free data fetched for you (``assets.free_data.crypto``, ``source: downloaded``) replaces the simulation the same way: a coin's history (spot, perpetual and funding from OKX), OKX's funding beside Deribit's
(``CryptoSeries.venue_funding``) and a snapshot of the real quotes and order books of the coin on up to six venues (``snapshot``), from which the spreads, depth and the cost of size are measured on the real
books and the cross-venue view compares real prices.

Conventions. Crypto trades every day, so annualisation here is by 365 (the strategy backtests use weekday marks and 252, like the rest of the platform). ``funding`` is the daily funding rate EARNED BY A SHORT
perpetual position (positive when longs pay shorts, the usual state); annualised funding is that times 365. ``basis`` is ``perp / spot - 1``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..instruments.crypto import VENUES
from .common import AssetSnapshot, add_flag, last_zscore, liquidity_score, percentile_rank, realized_vol
from .free_data.crypto import book_summary

ANN = 365.0
VENUE_ORDER = ("DERIBIT", "BINANCE", "BYBIT", "OKX")
REAL_SYMBOLS = ("BTC", "ETH")
SIMULATED_ALTS = {"SOL": {"beta": 1.5, "idio": 0.030, "level": 140.0, "funding_beta": 1.1}, "BNB": {"beta": 1.0, "idio": 0.020, "level": 600.0, "funding_beta": 0.7},
                  "XRP": {"beta": 1.2, "idio": 0.028, "level": 0.6, "funding_beta": 0.8}}
# Depth within 10 basis points of the mid, per side, in dollars, for BTC on each venue (ASSUMED round numbers for the simulated book), and the scale of other coins against BTC.
BOOK_DEPTH_10BP = {"BINANCE": 15e6, "OKX": 7e6, "BYBIT": 6.5e6, "DERIBIT": 2e6}
ASSET_DEPTH_SCALE = {"BTC": 1.0, "ETH": 0.55, "SOL": 0.12, "BNB": 0.10, "XRP": 0.08}
ASSET_SPREAD_BPS = {"BTC": 0.6, "ETH": 0.9, "SOL": 2.5, "BNB": 2.0, "XRP": 3.0}


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


@dataclass
class CryptoSeries:
    """One coin's daily history with its provenance."""

    symbol: str
    spot: pd.Series                       # index price at the daily 08:00 UTC mark
    perp: pd.Series | None                # perpetual close, same marks
    funding: pd.Series | None             # daily funding rate earned by a short
    source: str                           # platform | simulated | user_file | downloaded
    notes: list[str] = field(default_factory=list)
    funding_venue: str = "DERIBIT"        # the venue ``funding`` belongs to (OKX for a downloaded coin)
    venue_funding: dict = field(default_factory=dict)      # other venues' REAL funding series, by venue (downloaded)

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.spot.dropna().index[-1])


def _naive(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None) if pd.DatetimeIndex(out.index).tz is not None else pd.DatetimeIndex(out.index)
    return out


def platform_series(frames: dict, symbol: str) -> CryptoSeries:
    """BTC or ETH from the Deribit frames of ``data.crypto.build_crypto_frames``."""
    if symbol not in REAL_SYMBOLS:
        raise ValueError(f"the platform holds only {REAL_SYMBOLS}")
    spot, perp, funding = _naive(frames["spot"][symbol]).dropna(), _naive(frames["perp"][symbol]).dropna(), _naive(frames["funding"][symbol]).dropna()
    idx = spot.index
    return CryptoSeries(symbol, spot, perp.reindex(idx), funding.reindex(idx), "platform",
                        ["Deribit index price, perpetual close and funding (data/raw/crypto); the perpetual is inverse-margined and treated as linear."])


def simulated_series(symbol: str, btc: CryptoSeries) -> CryptoSeries:
    """SIMULATED history for a coin the platform does not hold: log returns ``beta x BTC + idiosyncratic noise``, funding ``funding_beta x BTC funding`` plus a persistent shock, and a perpetual that trades
    at a premium following that funding. Deterministic in the symbol and the BTC data."""
    if symbol not in SIMULATED_ALTS:
        raise ValueError(f"no simulation for '{symbol}'; the platform has {REAL_SYMBOLS} and can simulate {sorted(SIMULATED_ALTS)}")
    p = SIMULATED_ALTS[symbol]
    rng = np.random.default_rng(_seed(symbol, btc.spot.index[0].date(), btc.spot.index[-1].date()))
    idx = btc.spot.index
    r = np.log(btc.spot).diff().fillna(0.0).to_numpy()
    lr = p["beta"] * r + p["idio"] * rng.standard_normal(len(idx))
    spot = pd.Series(p["level"] * np.exp(lr.cumsum() - lr.cumsum()[-1]), index=idx, name=symbol)         # ends at the nominal level
    shock = np.zeros(len(idx))
    for t in range(1, len(idx)):
        shock[t] = 0.92 * shock[t - 1] + 0.00012 * rng.standard_normal()
    fund = btc.funding.reindex(idx).fillna(0.0).to_numpy() * p["funding_beta"] + shock
    basis = np.zeros(len(idx))
    for t in range(1, len(idx)):
        basis[t] = 0.85 * basis[t - 1] + 0.15 * (12.0 * fund[t]) + 0.0004 * rng.standard_normal()
    return CryptoSeries(symbol, spot, spot * (1.0 + basis), pd.Series(fund, index=idx, name=symbol), "simulated",
                        [f"{symbol} is SIMULATED from BTC (beta {p['beta']}); the platform has no {symbol} data."])


def user_series(path, symbol: str, source: str = "user_file", notes: list[str] | None = None, funding_venue: str = "DERIBIT") -> CryptoSeries:
    """A coin from ``data/user/crypto/<SYMBOL>.csv`` (or a free-data download of the same layout, ``source="downloaded"``): columns ``time, spot`` and optionally ``perp``, ``funding_daily`` (decimal per day)
    or ``funding_8h`` (decimal per 8 hours). ``funding_venue`` names the venue the funding is from."""
    raw = pd.read_csv(path, parse_dates=["time"])
    if "spot" not in raw.columns:
        raise ValueError("the crypto file needs the columns time and spot")
    raw = raw.dropna(subset=["spot"]).drop_duplicates("time").set_index("time").sort_index()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None) if pd.DatetimeIndex(raw.index).tz is not None else pd.DatetimeIndex(raw.index)
    funding = raw["funding_daily"] if "funding_daily" in raw.columns else (raw["funding_8h"] * 3.0 if "funding_8h" in raw.columns else None)
    return CryptoSeries(symbol, raw["spot"].astype(float), raw["perp"].astype(float) if "perp" in raw.columns else None, None if funding is None else funding.astype(float), source,
                        notes if notes is not None else ["Prices are from your file."], funding_venue.upper())


# ------------------------------------------------------------------------------------------------------------------ the order book (simulated)
def order_book_summary(symbol: str, venue: str, mid: float, notional: tuple = (1e5, 1e6, 1e7), seed: int = 0) -> dict:
    """A PARAMETRIC order book, not a feed: depth within ``d`` basis points of the mid grows as ``depth_10bp x (d / 10)^0.8`` per side, the half-spread depends on the coin, and the imbalance is a small random
    tilt. From it the spread, the depth at 10/25/50 bps and the impact (in basis points from the mid) of a market order of each notional."""
    venue = venue.upper()
    if venue not in BOOK_DEPTH_10BP:
        raise ValueError(f"unknown venue '{venue}'; choose from {sorted(BOOK_DEPTH_10BP)}")
    rng = np.random.default_rng(_seed("book", symbol, venue, seed))
    base = BOOK_DEPTH_10BP[venue] * ASSET_DEPTH_SCALE.get(symbol, 0.05)
    tilt = float(np.clip(0.12 * rng.standard_normal(), -0.3, 0.3))
    depth10 = {"bid": base * (1.0 + tilt), "ask": base * (1.0 - tilt)}
    spread = ASSET_SPREAD_BPS.get(symbol, 4.0) * (1.0 + 0.2 * (venue == "DERIBIT"))
    depth_at = lambda side, d: depth10[side] * (d / 10.0) ** 0.8                                                       # noqa: E731

    def impact(n: float, side: str) -> float:
        return float(spread / 2.0 + 10.0 * (n / depth10[side]) ** (1.0 / 0.8))

    return {"symbol": symbol, "venue": venue, "mid": float(mid), "spread_bps": float(spread), "depth": {f"{d}bp": {"bid": float(depth_at("bid", d)), "ask": float(depth_at("ask", d))} for d in (10, 25, 50)},
            "imbalance": float((depth10["bid"] - depth10["ask"]) / (depth10["bid"] + depth10["ask"])), "impact_bps": {f"{n:,.0f}": {"buy": impact(n, "ask"), "sell": impact(n, "bid")} for n in notional},
            "source": "simulated", "note": "A parametric depth profile with assumed round numbers; no order book is stored in the platform."}


# ------------------------------------------------------------------------------------------------------------------ overview
def _card(name: str, title: str, state: str, value, detail: str, source: str) -> dict:
    return {"name": name, "title": title, "state": state, "value": value, "detail": detail, "source": source}


def overview(s: CryptoSeries, stable: pd.Series | None = None, book: dict | None = None) -> dict:
    """The cards of one coin: price and returns, realised volatility, funding, basis and the six read-outs (trend, funding, momentum, volatility, liquidity, on-chain) each with a state and the source. Open
    interest and exchange net flows are not in the data and are reported as unavailable, not estimated."""
    spot = s.spot.dropna()
    if len(spot) < 250:
        raise ValueError(f"{s.symbol} has {len(spot)} daily prices; the workspace needs at least 250")
    price, ret = float(spot.iloc[-1]), spot.pct_change()
    ma50, ma200 = float(spot.tail(50).mean()), float(spot.tail(200).mean())
    r30, r90 = float(spot.iloc[-1] / spot.iloc[-31] - 1.0), float(spot.iloc[-1] / spot.iloc[-91] - 1.0)
    rv = realized_vol(spot, 21, ANN).dropna()
    rv21, rv_pct = float(rv.iloc[-1]), percentile_rank(rv.tail(365), float(rv.iloc[-1]))
    trend = "uptrend" if price > ma50 > ma200 else "downtrend" if price < ma50 < ma200 else "mixed"
    cards = [_card("trend", "Trend", trend, price / ma200 - 1.0, f"price {price / ma50 - 1.0:+.1%} vs the 50-day and {price / ma200 - 1.0:+.1%} vs the 200-day average", s.source),
             _card("momentum", "Momentum", "positive" if r30 > 0 and r90 > 0 else "negative" if r30 < 0 and r90 < 0 else "mixed", r30, f"{r30:+.1%} over 30 days, {r90:+.1%} over 90 days", s.source),
             _card("volatility", "Volatility", "high" if rv_pct > 0.8 else "low" if rv_pct < 0.2 else "normal", rv21, f"{rv21:.0%} annualised (21 days), {rv_pct:.0%} percentile of the last year", s.source)]
    fund_ann = fund_z = float("nan")
    funding_state = "unavailable"
    if s.funding is not None and s.funding.notna().sum() > 100:
        f = s.funding.dropna()
        fund_ann, fund_z = float(f.tail(7).mean() * ANN), last_zscore(f, 90)
        funding_state = "crowded longs" if fund_ann > 0.20 or fund_z > 2 else "crowded shorts" if fund_ann < -0.02 or fund_z < -2 else "balanced"
        cards.insert(1, _card("funding", "Funding", funding_state, fund_ann, f"{fund_ann:+.1%} a year earned by a short (7-day average), z-score {fund_z:+.1f} over 90 days", s.source))
    else:
        cards.insert(1, _card("funding", "Funding", "unavailable", None, "no funding history for this coin", s.source))
    liq = liquidity_score(book["spread_bps"], book["depth"]["10bp"]["bid"] * 40.0) if book else 0.0          # dollar volume taken as 40 times the depth within 10 bps (a convention for the score)
    if book:
        real = book.get("source") == "downloaded"
        cards.append(_card("liquidity", "Liquidity", "deep" if liq > 70 else "thin" if liq < 40 else "moderate", liq,
                           f"spread {book['spread_bps']:.2f} bps, ${book['depth']['25bp']['ask'] / 1e6:.1f}M within 25 bps ({'real book on ' + str(book['venue']).title() if real else 'simulated book'})", "downloaded" if real else "simulated"))
    else:
        cards.append(_card("liquidity", "Liquidity", "unavailable", None, "no order book in the data", "simulated"))
    if stable is not None and len(stable.dropna()) > 60:
        st = stable.dropna()
        g30 = float(st.iloc[-1] / st.iloc[-31] - 1.0)
        cards.append(_card("onchain", "On-chain", "inflow" if g30 > 0.01 else "outflow" if g30 < -0.01 else "flat", g30, f"stablecoin supply {g30:+.1%} over 30 days (DefiLlama); exchange net flows are not in the data", "platform"))
    else:
        cards.append(_card("onchain", "On-chain", "unavailable", None, "no stablecoin supply in the data", "platform"))
    basis = (s.perp / s.spot - 1.0).dropna() if s.perp is not None else pd.Series(dtype=float)
    flags: list[str] = []
    add_flag(flags, np.isfinite(fund_z) and abs(fund_z) > 2.0, f"funding is {fund_z:+.1f} standard deviations from its 90-day mean: a crowded trade")
    add_flag(flags, rv_pct > 0.9, "realised volatility is in its top decile of the last year")
    add_flag(flags, drawdown_from_peak(spot) < -0.3, f"{drawdown_from_peak(spot):.0%} below the all-time high in the data")
    basis_last = float(basis.iloc[-1]) if len(basis) else float("nan")
    add_flag(flags, np.isfinite(basis_last) and abs(basis_last) > 0.01, f"the perpetual trades {basis_last:+.2%} from spot")
    snap = AssetSnapshot(s.symbol, "crypto", price, rv21, liq, flags, str(s.as_of.date()), s.source)
    return {"symbol": s.symbol, "price": price, "as_of": str(s.as_of.date()), "return_24h": float(ret.iloc[-1]), "return_7d": float(spot.iloc[-1] / spot.iloc[-8] - 1.0), "return_30d": r30, "return_90d": r90,
            "realized_vol_21d": rv21, "realized_vol_63d": float(realized_vol(spot, 63, ANN).dropna().iloc[-1]), "vol_percentile_1y": rv_pct, "funding_annualised": None if not np.isfinite(fund_ann) else fund_ann,
            "funding_zscore": None if not np.isfinite(fund_z) else fund_z, "basis": None if not np.isfinite(basis_last) else basis_last, "open_interest": None, "exchange_net_flow": None,
            "cards": cards, "snapshot": snap.to_dict(), "risk_flags": flags, "source": s.source, "notes": s.notes + ["Open interest and exchange net flows are not in the platform's data and are not estimated."]}


def drawdown_from_peak(p: pd.Series) -> float:
    p = p.dropna()
    return float(p.iloc[-1] / p.cummax().iloc[-1] - 1.0)


# ------------------------------------------------------------------------------------------------------------------ spot and perpetual
def spot_perp_view(s: CryptoSeries, days: int = 365) -> dict:
    """Spot and perpetual prices, the basis (perp / spot - 1) and annualised funding through the last ``days`` days, with the correlation of daily returns and the basis statistics."""
    if s.perp is None:
        raise ValueError(f"{s.symbol} has no perpetual price series")
    frame = pd.DataFrame({"spot": s.spot, "perp": s.perp}).dropna().tail(days)
    frame["basis"] = frame["perp"] / frame["spot"] - 1.0
    funding = (s.funding.reindex(frame.index) * ANN) if s.funding is not None else pd.Series(np.nan, index=frame.index)
    rets = frame[["spot", "perp"]].pct_change().dropna()
    return {"symbol": s.symbol, "source": s.source, "dates": [str(d.date()) for d in frame.index], "spot": [round(float(v), 4) for v in frame["spot"]], "perp": [round(float(v), 4) for v in frame["perp"]],
            "basis": [round(float(v), 6) for v in frame["basis"]], "funding_annualised": [None if not np.isfinite(v) else round(float(v), 5) for v in funding.rolling(7, min_periods=3).mean()],
            "stats": {"basis_mean": float(frame["basis"].mean()), "basis_std": float(frame["basis"].std(ddof=1)), "basis_last": float(frame["basis"].iloc[-1]), "basis_zscore": last_zscore(frame["basis"], 90),
                      "return_correlation": float(rets["spot"].corr(rets["perp"])), "tracking_error_annual": float((rets["perp"] - rets["spot"]).std(ddof=1) * math.sqrt(ANN))}}


# ------------------------------------------------------------------------------------------------------------------ funding by venue
def venue_funding(s: CryptoSeries, venue: str) -> tuple[pd.Series, str]:
    """The funding series on a venue and its source. The series of the coin itself belongs to ``s.funding_venue`` (Deribit's, the platform's real series, for BTC and ETH; OKX's for a downloaded coin); a venue
    whose real series was downloaded (``s.venue_funding``) returns it; every other venue is SIMULATED as the coin's rate times a venue factor plus an offset and a persistent shock (the platforms' funding
    rates are strongly but not perfectly correlated), because the platform holds one venue's history."""
    venue = venue.upper()
    if venue not in VENUES:
        raise ValueError(f"unknown venue '{venue}'; choose from {sorted(VENUES)}")
    if s.funding is None:
        raise ValueError(f"{s.symbol} has no funding history")
    if venue == s.funding_venue:
        return s.funding, s.source
    if venue in s.venue_funding:
        return s.venue_funding[venue], "downloaded"
    rng = np.random.default_rng(_seed("funding", s.symbol, venue))
    idx = s.funding.index
    factor = {"BINANCE": 1.00, "BYBIT": 1.04, "OKX": 0.97}.get(venue, 1.0)
    offset = {"BINANCE": 0.00002, "BYBIT": 0.00001, "OKX": -0.00001}.get(venue, 0.0)
    shock = np.zeros(len(idx))
    for t in range(1, len(idx)):
        shock[t] = 0.8 * shock[t - 1] + 0.00004 * rng.standard_normal()
    return pd.Series(s.funding.to_numpy() * factor + offset + shock, index=idx, name=f"{s.symbol}@{venue}"), "simulated"


def funding_dashboard(series: dict[str, CryptoSeries], venues: tuple = VENUE_ORDER, days: int = 120) -> dict:
    """Funding by coin and venue: the rate per funding interval, annualised, the 7- and 30-day averages, the 90-day z-score and one-year percentile, and a series for the chart. Flags crowded funding and
    large gaps between venues (a funding arbitrage, before costs)."""
    rows, flags, charts = [], [], {}
    for sym, s in series.items():
        if s.funding is None:
            flags.append(f"{sym}: no funding history")
            continue
        by_venue = {}
        for v in venues:
            f, src = venue_funding(s, v)
            f = f.dropna()
            hours = VENUES[v]["funding_hours"]
            now = float(f.tail(3).mean())                                         # the last few days smooth a single reading
            rows.append({"symbol": sym, "venue": v, "interval_hours": hours, "rate_per_interval": float(f.iloc[-1] * hours / 24.0), "annualised": float(f.iloc[-1] * ANN), "avg_7d": float(f.tail(7).mean() * ANN),
                         "avg_30d": float(f.tail(30).mean() * ANN), "zscore_90d": last_zscore(f, 90), "percentile_1y": percentile_rank(f.tail(365), float(f.iloc[-1])), "source": src})
            by_venue[v] = (f, src)
            charts.setdefault(sym, {})[v] = {"source": src, "dates": [str(d.date()) for d in f.index[-days:]], "annualised": [round(float(x), 5) for x in (f.tail(days).rolling(3, min_periods=1).mean() * ANN)]}
        latest = {v: float(f.tail(7).mean() * ANN) for v, (f, _) in by_venue.items()}
        gap = max(latest.values()) - min(latest.values())
        add_flag(flags, gap > 0.05, f"{sym}: funding differs by {gap:.1%} a year between venues" + (" (some of the venues are simulated)" if any(src == "simulated" for _, src in by_venue.values()) else ""))
        z = [r["zscore_90d"] for r in rows if r["symbol"] == sym and np.isfinite(r["zscore_90d"])]
        add_flag(flags, bool(z) and max(abs(x) for x in z) > 2.0, f"{sym}: funding is more than two standard deviations from its 90-day mean")
    real = sorted({f"{r['symbol']} on {r['venue']}" for r in rows if r["source"] != "simulated"})
    fake = sorted({f"{r['symbol']} on {r['venue']}" for r in rows if r["source"] == "simulated"})
    notes = ["Funding that is data: " + (", ".join(real) if real else "none") + ". Funding that is simulated: " + (", ".join(fake) if fake else "none")
             + ". A simulated venue is the coin's own series with offsets and noise; download the coin from the Free data card to replace it with real history."]
    return {"rows": rows, "series": charts, "risk_flags": flags, "venues": list(venues), "notes": notes}


# ------------------------------------------------------------------------------------------------------------------ basis
def basis_dashboard(s: CryptoSeries, fee_bps: float = 5.0, days: int = 365) -> dict:
    """The perpetual's basis through time and today, its z-score, and what a cash-and-carry (long spot, short perpetual) earned: the funding collected, the basis change, and the break-even number of days to
    recover round-trip fees of ``fee_bps`` per leg."""
    if s.perp is None or s.funding is None:
        raise ValueError(f"{s.symbol} needs a perpetual and funding series")
    frame = pd.DataFrame({"spot": s.spot, "perp": s.perp, "funding": s.funding}).dropna().tail(days + 30)
    frame["basis"] = frame["perp"] / frame["spot"] - 1.0
    carry_daily = frame["funding"] + frame["spot"].pct_change() - frame["perp"].pct_change()                   # long spot, short perpetual (linear approximation, as in the crypto bundle)
    carry_daily = carry_daily.dropna().tail(days)
    funding_ann = float(frame["funding"].tail(30).mean() * ANN)
    round_trip = 4.0 * fee_bps / 1e4                                                                               # open and close both legs
    daily_edge = float(frame["funding"].tail(30).mean())
    sd = float(carry_daily.std(ddof=1))
    flags: list[str] = []
    add_flag(flags, funding_ann < 0.02, "funding is below 2% a year: the cash-and-carry earns little over fees")
    add_flag(flags, abs(float(frame["basis"].iloc[-1])) > 0.01, "the basis is wide: mark-to-market risk on the short leg until it converges")
    return {"symbol": s.symbol, "source": s.source, "basis": float(frame["basis"].iloc[-1]), "basis_zscore": last_zscore(frame["basis"], 90), "basis_percentile_1y": percentile_rank(frame["basis"].tail(365), float(frame["basis"].iloc[-1])),
            "funding_annualised_30d": funding_ann, "carry_annualised": float(carry_daily.mean() * ANN), "carry_vol": float(sd * math.sqrt(ANN)), "carry_sharpe": float(carry_daily.mean() / sd * math.sqrt(ANN)) if sd > 0 else None,
            "carry_max_drawdown": float((np.cumprod(1.0 + carry_daily) / np.cumprod(1.0 + carry_daily).cummax() - 1.0).min()), "breakeven_days": float(round_trip / daily_edge) if daily_edge > 0 else None, "fee_bps": float(fee_bps),
            "dates": [str(d.date()) for d in frame.index[-days:]], "basis_series": [round(float(v), 6) for v in frame["basis"].tail(days)], "carry_cumulative": [round(float(v), 5) for v in (np.cumprod(1.0 + carry_daily) - 1.0)],
            "risk_flags": flags, "note": "A linear approximation of the inverse perpetual; funding and the basis are as the data states them."}


# ------------------------------------------------------------------------------------------------------------------ cross-exchange
def snapshot_quotes(snapshot: dict) -> list[dict]:
    """The REAL quotes of a venue snapshot (``assets.free_data.crypto.fetch_venues``): each venue's best bid and ask when the snapshot was taken, its spread, the assumed taker fee, and whether it is a spot or a
    perpetual market (only markets of one kind are compared with each other: a perpetual trades at a basis to spot)."""
    out = []
    for v in snapshot["venues"]:
        bid, ask = float(v["bid"]), float(v["ask"])
        mid = (bid + ask) / 2.0
        out.append({"venue": v["venue"], "label": v.get("label", v["venue"]), "kind": v["kind"], "pair": v["pair"], "quote_asset": v["quote_asset"], "bid": bid, "ask": ask, "mid": mid,
                    "spread_bps": (ask - bid) / mid * 1e4, "taker_bps": float(v["taker_bps"]), "source": "downloaded"})
    return out


def snapshot_books(snapshot: dict, symbol: str) -> dict[str, dict]:
    """The order-book summary of every venue in a snapshot, measured on the real levels (same shape as ``order_book_summary``)."""
    return {v["venue"]: book_summary(v, symbol) for v in snapshot["venues"]}


def venue_quotes(s: CryptoSeries, venues: tuple = VENUE_ORDER, snapshot: dict | None = None) -> list[dict]:
    """The perpetual's bid and ask on each venue at the latest mark. Deribit's mid is the platform's last perpetual close (real, but a daily mark, not a live quote); every other venue's mid is that price times
    a SIMULATED venue offset, and every spread is the venue's simulated book spread. With a downloaded ``snapshot`` the quotes are the real ones of that moment instead."""
    if snapshot is not None:
        return snapshot_quotes(snapshot)
    if s.perp is None:
        raise ValueError(f"{s.symbol} has no perpetual price")
    base = float(s.perp.dropna().iloc[-1])
    rng = np.random.default_rng(_seed("quotes", s.symbol, str(s.as_of.date())))
    out = []
    for v in venues:
        book = order_book_summary(s.symbol, v, base, seed=int(s.as_of.toordinal()))
        offset = 0.0 if v == "DERIBIT" else float(np.clip(1.5e-4 * rng.standard_normal(), -4e-4, 4e-4))
        mid = base * (1.0 + offset)
        half = mid * book["spread_bps"] / 2e4
        out.append({"venue": v, "label": v, "kind": "perp", "bid": mid - half, "ask": mid + half, "mid": mid, "spread_bps": book["spread_bps"], "taker_bps": float(VENUES[v]["taker_bps"]),
                    "source": "platform (daily mark)" if (v == "DERIBIT" and s.source == "platform") else "simulated"})
    return out


def cross_exchange(s: CryptoSeries, slippage_bps: float = 1.0, notional: float = 1e6, venues: tuple = VENUE_ORDER, snapshot: dict | None = None) -> dict:
    """Every ordered pair of venues of the same kind: buy at the first venue's ask, sell at the second's bid. Gross spread in basis points, less both taker fees and ``slippage_bps`` of slippage per side, is the
    net edge; the dollar edge is that on ``notional``. Warnings say when an edge does not cover its costs and what the number does not include (transfers, funding, margin, latency). A downloaded
    ``snapshot`` supplies real quotes; without one the quotes are modelled."""
    quotes = venue_quotes(s, venues, snapshot)
    rows = []
    for a in quotes:
        for b in quotes:
            if a["venue"] == b["venue"] or a["kind"] != b["kind"]:
                continue
            gross = (b["bid"] / a["ask"] - 1.0) * 1e4
            fees = a["taker_bps"] + b["taker_bps"]
            net = gross - fees - 2.0 * slippage_bps
            rows.append({"buy": a["venue"], "sell": b["venue"], "kind": a["kind"], "gross_bps": float(gross), "fees_bps": float(fees), "slippage_bps": float(2.0 * slippage_bps), "net_bps": float(net),
                         "net_dollars": float(net / 1e4 * notional), "profitable": bool(net > 0)})
    rows.sort(key=lambda r: r["net_bps"], reverse=True)
    flags = []
    add_flag(flags, not any(r["profitable"] for r in rows), "no pair covers its fees: the spreads here are smaller than the cost of crossing them")
    if snapshot is None:
        add_flag(flags, any(q["source"] == "simulated" for q in quotes), "all venues except Deribit are simulated, and Deribit's price is a daily mark, so these spreads are illustrative")
    else:
        add_flag(flags, any(q["quote_asset"] == "USDT" for q in quotes) and any(q["quote_asset"] == "USD" for q in quotes),
                 "OKX quotes in USDT, not dollars: part of its gap to the dollar venues is the USDT/USD rate, not an opportunity")
    notes = ["Net edge excludes transfer time and fees between venues, funding paid while the legs are open, margin, borrowing and latency."]
    if snapshot is not None:
        notes.append(f"Real quotes of {snapshot.get('fetched_at', 'one moment')}, taken from each venue's public order book; a quote is gone in seconds. Taker fees are entry-level assumptions, not your tier. "
                     "Spot venues are compared with spot venues and perpetuals with perpetuals.")
    return {"symbol": s.symbol, "quotes": quotes, "pairs": rows, "notional": float(notional), "risk_flags": flags, "notes": notes}


# ------------------------------------------------------------------------------------------------------------------ strategies
CRYPTO_STRATEGIES = {
    "spot_momentum": {"label": "Spot momentum", "model": "tsmom", "params": {"lookback": 90}, "assets": "spot", "text": "Long what has risen over the lookback, short what has fallen, sized by volatility (time-series momentum)."},
    "funding_carry": {"label": "Funding carry", "model": "funding_carry", "params": {}, "assets": "carry", "text": "Hold long-spot / short-perpetual when trailing funding is high and reverse it when negative."},
    "basis_trade": {"label": "Basis trade", "model": "basis_reversion", "params": {}, "assets": "carry", "text": "Hold the carry position when the perpetual trades rich to spot against its own history."},
    "vol_breakout": {"label": "Volatility breakout", "model": "vol_breakout", "params": {"window": 20, "k": 2.0}, "assets": "spot", "text": "Long above the moving average plus k ATRs, short below it minus k ATRs."},
    "stablecoin_flow": {"label": "Stablecoin flow", "model": "stablecoin_flow", "params": {"window": 30}, "assets": "spot", "text": "Long BTC and ETH when the total stablecoin supply has been growing quickly."},
    "regime_trend": {"label": "Regime-filtered trend", "model": "ma_crossover", "params": {"fast": 20, "slow": 100}, "assets": "spot", "regime": "vol_state", "text": "A moving-average crossover whose volatility target depends on the volatility regime."},
    "exchange_flow": {"label": "Exchange-flow", "model": None, "params": {}, "assets": "spot", "text": "Needs exchange net-flow data, which the platform does not have."},
}


def run_crypto_strategy(config, strategy: str, assets: list[str] | None = None, frequency: str | None = None, lookback: int | None = None, target_vol: float | None = None,
                        fees_bps: float = 5.0, slippage_bps: float = 2.0, regime: str | None = None) -> dict:
    """Run one of ``CRYPTO_STRATEGIES`` on the Deribit-based crypto bundle through the platform pipeline. ``assets`` picks BTC and/or ETH (the carry strategies trade the matching ``_CARRY`` index);
    ``fees_bps`` plus ``slippage_bps`` is the one-way cost of trading the whole book; ``regime`` overrides the strategy's detector. Weekday marks and 252-day annualisation, like every other backtest."""
    from ..framework.crypto_bundle import load_crypto_bundle
    from .strategy_runs import run_strategy, subset_bundle

    if strategy not in CRYPTO_STRATEGIES:
        raise ValueError(f"unknown crypto strategy '{strategy}'; choose from {sorted(CRYPTO_STRATEGIES)}")
    spec = CRYPTO_STRATEGIES[strategy]
    if spec["model"] is None:
        raise ValueError("the exchange-flow strategy needs exchange net-flow data, which the platform does not hold; use the stablecoin-flow strategy (the on-chain signal that is available) or load your own data")
    chosen = [a.upper() for a in (assets or ["BTC", "ETH"])]
    bad = [a for a in chosen if a not in REAL_SYMBOLS]
    if bad or not chosen:
        raise ValueError(f"the strategy backtests use the platform's data, which covers {list(REAL_SYMBOLS)} (asked for {bad or 'nothing'})")
    bundle = load_crypto_bundle(config)
    columns = [f"{a}_CARRY" for a in chosen] if spec["assets"] == "carry" else chosen
    params = dict(spec["params"])
    if lookback is not None and strategy == "spot_momentum":
        params["lookback"] = int(lookback)
    if lookback is not None and strategy == "stablecoin_flow":
        params["window"] = int(lookback)
    if lookback is not None and strategy == "regime_trend":
        params["slow"] = int(lookback)
        params["fast"] = max(5, int(lookback) // 5)
    if lookback is not None and strategy == "vol_breakout":
        params["window"] = int(lookback)
    out = run_strategy(subset_bundle(bundle, columns), config, spec["model"], params, rebalance=frequency, target_vol=target_vol, regime=regime or spec.get("regime"),
                       cost_bps=float(fees_bps) + float(slippage_bps), label=spec["label"])
    out.update({"strategy": strategy, "label": spec["label"], "text": spec["text"], "assets": chosen, "fees_bps": float(fees_bps), "slippage_bps": float(slippage_bps), "source": "platform",
                "notes": ["Deribit BTC and ETH daily data, weekday marks; the carry strategies trade a synthetic delta-neutral index (long spot, short perpetual, linear approximation).",
                          "Fees and slippage are one-way basis points on the traded notional; funding is earned inside the carry index."]})
    return out
