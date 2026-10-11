"""Real crypto data from free public exchange feeds, in the formats the Crypto workspace reads.

**A coin's history** (``fetch_history``): daily spot, perpetual and funding from OKX (the spot ``COIN-USDT``, the perpetual ``COIN-USDT-SWAP`` and its funding-rate history, so that the basis compares prices
quoted in the same unit); a coin OKX does not list falls back to a spot-only history from Kraken, Bitstamp or Coinbase. The file has the columns ``time, spot`` and, where known, ``perp`` and
``funding_daily`` (the daily funding earned by a short: what the workspace reads as a user file).

**The venues** (``fetch_venues``): at one moment, the best bid and ask and the order book of the same coin on Coinbase, Kraken, Bitstamp and OKX (spot) and on OKX and Deribit (perpetuals), saved as one
document. From the book levels the workspace measures spread, depth and the cost of a market order, and the cross-venue view compares real quotes. Prices on OKX are in USDT, not dollars: a
difference of a few basis points between OKX and the dollar venues is partly the USDT/USD rate, and the page says so. One snapshot is a moment; it does not become history unless you download again.

**Funding by venue** (``fetch_funding``): OKX's funding-rate history as a daily series, to sit beside Deribit's in the funding view.

Binance and Bybit are not used: they refuse requests from many countries (HTTP 451 and 403 from the United States). Taker fees below are entry-level assumptions, not your tier.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

import pandas as pd

from .http import FetchError, Http
from .store import Download, utc_now

COINBASE = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com/0/public"
BITSTAMP = "https://www.bitstamp.net/api/v2"
OKX = "https://www.okx.com/api/v5"
DERIBIT = "https://www.deribit.com/api/v2/public"
COIN = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
KRAKEN_NAMES = {"BTC": "XBT", "DOGE": "XDG"}
MAX_LEVELS = 400
MIN_DAYS = 250                                                            # the Crypto tab needs at least this many daily prices of a coin
BAND_BPS = 500.0                                                          # book levels farther than this from the mid are not kept
# Entry-level taker fees in basis points. ASSUMPTIONS: the schedules change and depend on your volume.
TAKER_BPS = {"COINBASE": 60.0, "KRAKEN": 40.0, "BITSTAMP": 40.0, "OKX": 10.0, "OKX-PERP": 5.0, "DERIBIT-PERP": 5.0}
LABELS = {"COINBASE": "Coinbase Exchange", "KRAKEN": "Kraken", "BITSTAMP": "Bitstamp", "OKX": "OKX spot", "OKX-PERP": "OKX perpetual", "DERIBIT-PERP": "Deribit perpetual"}
DEFAULT_COINS = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "LTC")


def check_coin(coin: str) -> str:
    c = str(coin or "").strip().upper()
    if not COIN.match(c):
        raise ValueError(f"'{coin}' is not a coin symbol (letters and digits, such as BTC)")
    return c


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _levels(raw, price_index: int = 0, size_index: int = 1, size_scale=None) -> list[list[float]]:
    out = []
    for row in raw or []:
        p, s = _num(row[price_index]), _num(row[size_index])
        if p > 0 and s > 0:
            out.append([p, s / p if size_scale == "usd" else s])
    return out


def _trim(bids: list, asks: list) -> tuple[list, list]:
    """Best-first levels within ``BAND_BPS`` of the mid, at most ``MAX_LEVELS`` a side."""
    if not bids or not asks:
        return [], []
    bids, asks = sorted(bids, key=lambda x: -x[0]), sorted(asks, key=lambda x: x[0])
    mid = (bids[0][0] + asks[0][0]) / 2.0
    lo, hi = mid * (1 - BAND_BPS / 1e4), mid * (1 + BAND_BPS / 1e4)
    return [[round(p, 8), round(s, 8)] for p, s in bids if p >= lo][:MAX_LEVELS], [[round(p, 8), round(s, 8)] for p, s in asks if p <= hi][:MAX_LEVELS]


# ------------------------------------------------------------------------------------------------------------------ one venue's book
def coinbase_book(http: Http, coin: str) -> dict:
    pair = f"{coin}-USD"
    d = http.get_json(f"{COINBASE}/products/{pair}/book", {"level": 2})
    bids, asks = _trim(_levels(d.get("bids")), _levels(d.get("asks")))
    return {"venue": "COINBASE", "kind": "spot", "pair": pair, "quote_asset": "USD", "bids": bids, "asks": asks}


def kraken_book(http: Http, coin: str) -> dict:
    pair = f"{KRAKEN_NAMES.get(coin, coin)}USD"
    d = http.get_json(f"{KRAKEN}/Depth", {"pair": pair, "count": 500})
    if d.get("error"):
        raise FetchError(f"Kraken: {'; '.join(map(str, d['error']))}", status=404)
    book = next(iter((d.get("result") or {}).values()), None)
    if not isinstance(book, dict):
        raise FetchError("Kraken's answer has no order book")
    bids, asks = _trim(_levels(book.get("bids")), _levels(book.get("asks")))
    return {"venue": "KRAKEN", "kind": "spot", "pair": pair, "quote_asset": "USD", "bids": bids, "asks": asks}


def bitstamp_book(http: Http, coin: str) -> dict:
    pair = f"{coin.lower()}usd"
    d = http.get_json(f"{BITSTAMP}/order_book/{pair}/")
    if not isinstance(d, dict) or "bids" not in d:
        raise FetchError("Bitstamp's answer has no order book", status=404)
    bids, asks = _trim(_levels(d.get("bids")), _levels(d.get("asks")))
    return {"venue": "BITSTAMP", "kind": "spot", "pair": pair, "quote_asset": "USD", "bids": bids, "asks": asks}


def okx_book(http: Http, coin: str, swap: bool = False) -> dict:
    inst = f"{coin}-USDT-SWAP" if swap else f"{coin}-USDT"
    d = http.get_json(f"{OKX}/market/books", {"instId": inst, "sz": 400})
    data = (d.get("data") or [None])[0] if isinstance(d, dict) and str(d.get("code", "0")) == "0" else None
    if not data:
        raise FetchError(f"OKX has no order book for {inst} ({(d or {}).get('msg') or 'no data'})", status=404)
    value = _contract_value(http, inst) if swap else 1.0                  # perpetual sizes are contracts of ctVal coins each
    bids, asks = _trim([[p, s * value] for p, s in _levels(data.get("bids"))], [[p, s * value] for p, s in _levels(data.get("asks"))])
    return {"venue": "OKX-PERP" if swap else "OKX", "kind": "perp" if swap else "spot", "pair": inst, "quote_asset": "USDT", "bids": bids, "asks": asks}


def _contract_value(http: Http, inst: str) -> float:
    """The coins in one OKX contract (``ctVal``): 0.01 for BTC, 0.1 for ETH, and so on."""
    d = http.get_json(f"{OKX}/public/instruments", {"instType": "SWAP", "instId": inst})
    rows = d.get("data") if isinstance(d, dict) else None
    if not rows:
        raise FetchError(f"OKX does not describe {inst}", status=404)
    v = _num(rows[0].get("ctVal"))
    if not v > 0:
        raise FetchError(f"OKX gave no contract size for {inst}")
    return v


def deribit_book(http: Http, coin: str) -> dict:
    inst = f"{coin}-PERPETUAL"
    d = http.get_json(f"{DERIBIT}/get_order_book", {"instrument_name": inst, "depth": 1000})
    r = d.get("result") if isinstance(d, dict) else None
    if not isinstance(r, dict):
        raise FetchError(f"Deribit has no order book for {inst}", status=404)
    bids, asks = _trim(_levels(r.get("bids"), size_scale="usd"), _levels(r.get("asks"), size_scale="usd"))
    return {"venue": "DERIBIT-PERP", "kind": "perp", "pair": inst, "quote_asset": "USD", "bids": bids, "asks": asks, "funding_8h": r.get("funding_8h"), "mark": r.get("mark_price")}


VENUE_FETCHERS = (("COINBASE", coinbase_book), ("KRAKEN", kraken_book), ("BITSTAMP", bitstamp_book), ("OKX", lambda h, c: okx_book(h, c, False)), ("OKX-PERP", lambda h, c: okx_book(h, c, True)),
                  ("DERIBIT-PERP", deribit_book))


def book_summary(venue: dict, coin: str, notionals=(1e5, 1e6, 1e7)) -> dict:
    """The spread, the depth within 10, 25 and 50 basis points of the mid (in dollars, per side), the bid/ask imbalance within 10 bps and the cost, in basis points from the mid, of a market buy and sell
    of each notional that walks the REAL levels. A notional the saved levels cannot fill has no cost (``None``) and is listed in ``unfilled``. Same shape as the workspace's modelled book."""
    bids, asks = venue["bids"], venue["asks"]
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread = (best_ask - best_bid) / mid * 1e4

    def depth(levels, side, bps):
        edge = mid * (1 - bps / 1e4) if side == "bid" else mid * (1 + bps / 1e4)
        return float(sum(p * s for p, s in levels if (p >= edge if side == "bid" else p <= edge)))

    def walk(levels, notional, side):
        """Basis points from the mid of the average price paid (buy) or received (sell) for ``notional`` dollars, or ``None`` when the levels run out first."""
        remaining, coins = notional, 0.0
        for p, s in levels:
            take = min(remaining, p * s)
            coins += take / p
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-6 * notional:
            return None
        average = notional / coins
        return float((average / mid - 1.0) * 1e4 if side == "buy" else (1.0 - average / mid) * 1e4)

    depth_out = {f"{d}bp": {"bid": depth(bids, "bid", d), "ask": depth(asks, "ask", d)} for d in (10, 25, 50)}
    near_b, near_a = depth_out["10bp"]["bid"], depth_out["10bp"]["ask"]
    impact, unfilled = {}, []
    for n in notionals:
        buy, sell = walk(asks, n, "buy"), walk(bids, n, "sell")
        impact[f"{n:,.0f}"] = {"buy": buy, "sell": sell}
        if buy is None or sell is None:
            unfilled.append(f"{n:,.0f}")
    return {"symbol": coin, "venue": venue["venue"], "mid": float(mid), "spread_bps": float(spread), "depth": depth_out, "imbalance": float((near_b - near_a) / (near_b + near_a)) if near_b + near_a > 0 else 0.0,
            "impact_bps": impact, "unfilled": unfilled, "levels": {"bids": len(bids), "asks": len(asks)}, "source": "downloaded",
            "note": f"The real order book of {venue['pair']} on {LABELS.get(venue['venue'], venue['venue'])} at the moment of the download."
                    + (f" The saved levels cannot fill an order of {', '.join(unfilled)}." if unfilled else "")}


def fetch_venues(http: Http, coin: str, progress=None) -> Download:
    """One snapshot of the coin on every venue that has it. A venue that cannot be reached or does not list the coin is reported, not fatal; with no venue at all the download fails."""
    coin = check_coin(coin)
    venues, failed = [], {}
    for name, fetch in VENUE_FETCHERS:
        try:
            v = fetch(http, coin)
        except (FetchError, ValueError, KeyError, TypeError, IndexError) as exc:
            failed[name] = str(exc)
            continue
        if progress:
            progress(f"{coin} on {LABELS[v['venue']]}")
        if not v["bids"] or not v["asks"] or v["bids"][0][0] >= v["asks"][0][0]:
            failed[v["venue"]] = "an empty or crossed order book"
            continue
        v["bid"], v["ask"] = v["bids"][0][0], v["asks"][0][0]
        v["label"], v["taker_bps"] = LABELS[v["venue"]], TAKER_BPS[v["venue"]]
        venues.append(v)
    if not venues:
        raise FetchError(f"no venue gave an order book for {coin}: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))
    when = utc_now().isoformat().replace("+00:00", "Z")
    payload = {"coin": coin, "fetched_at": when, "venues": venues, "failed": failed, "taker_bps_assumed": {v["venue"]: v["taker_bps"] for v in venues}}
    notes = [f"One moment ({when}): the best bid and ask and the order book of {coin} on {len(venues)} venues: {', '.join(v['label'] for v in venues)}.",
             "Prices on OKX are in USDT, not dollars: a difference of a few basis points to the dollar venues is partly the USDT/USD rate. Perpetuals are not spot: their prices carry the basis.",
             "Taker fees are entry-level assumptions (Coinbase 60, Kraken 40, Bitstamp 40, OKX 10, perpetuals 5 basis points), not your tier."]
    if failed:
        notes.append("Not available: " + "; ".join(f"{LABELS.get(k, k)} ({v})" for k, v in failed.items()) + ".")
    meta = {"source_id": "exchanges", "source": ", ".join(v["label"] for v in venues), "url": "public order-book endpoints", "as_of": when[:10], "coin": coin, "venues": [v["venue"] for v in venues], "notes": notes}
    return Download("crypto", f"venues_{coin}", "venues", payload=payload, meta=meta)


# ------------------------------------------------------------------------------------------------------------------ histories
def _day(ms) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).date())


def okx_daily(http: Http, inst: str, days: int) -> pd.Series:
    """Daily closes of an OKX instrument, newest page first, ``days`` of them at most (100 a request), the unfinished day dropped."""
    out: dict[pd.Timestamp, float] = {}
    after = None
    for _ in range(math.ceil(days / 100) + 1):
        params = {"instId": inst, "bar": "1D", "limit": 100}
        if after:
            params["after"] = after
        d = http.get_json(f"{OKX}/market/history-candles", params)
        if not isinstance(d, dict) or str(d.get("code", "0")) != "0":
            raise FetchError(f"OKX has no candles for {inst} ({(d or {}).get('msg') or 'no data'})", status=404)
        rows = d.get("data") or []
        if not rows:
            break
        for r in rows:
            if str(r[8]) == "1":                                              # a confirmed (finished) candle
                out[_day(r[0])] = _num(r[4])
        after = rows[-1][0]
        if len(out) >= days:
            break
    s = pd.Series(out, dtype=float).sort_index().dropna()
    if s.empty:
        raise FetchError(f"OKX returned no finished daily candles for {inst}")
    return s.tail(days)


def okx_funding_daily(http: Http, inst: str, days: int) -> pd.Series:
    """The funding earned by a SHORT per UTC day (the sum of the day's settlements; positive when longs pay shorts), from OKX's funding-rate history."""
    out: dict[pd.Timestamp, float] = {}
    after = None
    for _ in range(math.ceil(days * 3 / 100) + 2):
        params = {"instId": inst, "limit": 100}
        if after:
            params["after"] = after
        d = http.get_json(f"{OKX}/public/funding-rate-history", params)
        if not isinstance(d, dict) or str(d.get("code", "0")) != "0":
            raise FetchError(f"OKX has no funding history for {inst} ({(d or {}).get('msg') or 'no data'})", status=404)
        rows = d.get("data") or []
        if not rows:
            break
        for r in rows:
            rate = _num(r.get("realizedRate") if r.get("realizedRate") not in (None, "") else r.get("fundingRate"))
            if math.isfinite(rate):
                day = _day(r["fundingTime"])
                out[day] = out.get(day, 0.0) + rate
        after = rows[-1]["fundingTime"]
        if len(out) >= days + 1:
            break
    s = pd.Series(out, dtype=float).sort_index()
    if len(s) < 3:
        raise FetchError(f"OKX returned almost no funding history for {inst}")
    return s.iloc[1:].tail(days)                                              # the first day is usually a part day


def _spot_fallback(http: Http, coin: str, days: int) -> tuple[pd.Series, str]:
    """Daily spot closes in US dollars from the first of Kraken, Bitstamp and Coinbase that has the coin. The unfinished day is dropped."""
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    problems = []
    for name, fn in (("Kraken", _kraken_daily), ("Bitstamp", _bitstamp_daily), ("Coinbase", _coinbase_daily)):
        try:
            s = fn(http, coin)
        except (FetchError, KeyError, ValueError, TypeError, IndexError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        s = s[s.index < today].dropna()
        if len(s) >= 30:
            return s.tail(days), name
        problems.append(f"{name}: only {len(s)} days")
    raise FetchError(f"no exchange gave a daily history for {coin} ({'; '.join(problems)})")


def _kraken_daily(http: Http, coin: str) -> pd.Series:
    d = http.get_json(f"{KRAKEN}/OHLC", {"pair": f"{KRAKEN_NAMES.get(coin, coin)}USD", "interval": 1440})
    if d.get("error"):
        raise FetchError(f"Kraken: {'; '.join(map(str, d['error']))}", status=404)
    rows = next(v for k, v in d["result"].items() if k != "last")
    return pd.Series({_day(r[0] * 1000): _num(r[4]) for r in rows}, dtype=float).sort_index()


def _bitstamp_daily(http: Http, coin: str) -> pd.Series:
    d = http.get_json(f"{BITSTAMP}/ohlc/{coin.lower()}usd/", {"step": 86400, "limit": 1000})
    rows = (d.get("data") or {}).get("ohlc") if isinstance(d, dict) else None
    if not rows:
        raise FetchError("Bitstamp has no candles", status=404)
    return pd.Series({_day(int(r["timestamp"]) * 1000): _num(r["close"]) for r in rows}, dtype=float).sort_index()


def _coinbase_daily(http: Http, coin: str) -> pd.Series:
    rows = http.get_json(f"{COINBASE}/products/{coin}-USD/candles", {"granularity": 86400})
    if not isinstance(rows, list) or not rows:
        raise FetchError("Coinbase has no candles", status=404)
    return pd.Series({_day(r[0] * 1000): _num(r[4]) for r in rows}, dtype=float).sort_index()


def fetch_history(http: Http, coin: str, days: int = 400, progress=None) -> Download:
    """A coin's daily history: spot, perpetual and funding from OKX when it lists the coin, else spot alone from Kraken, Bitstamp or Coinbase."""
    coin = check_coin(coin)
    days = int(days)
    if not MIN_DAYS <= days <= 1500:
        raise ValueError(f"days must be between {MIN_DAYS} and 1500 (the Crypto tab needs at least {MIN_DAYS} daily prices)")
    sources, notes = {}, []
    spot = perp = funding = None
    try:
        if progress:
            progress(f"{coin}: spot prices from OKX")
        spot = okx_daily(http, f"{coin}-USDT", days)
        sources["spot"] = "OKX (USDT)"
        try:
            if progress:
                progress(f"{coin}: perpetual prices from OKX")
            perp = okx_daily(http, f"{coin}-USDT-SWAP", days)
            sources["perp"] = "OKX (USDT)"
            if progress:
                progress(f"{coin}: funding history from OKX")
            funding = okx_funding_daily(http, f"{coin}-USDT-SWAP", days)
            sources["funding"] = "OKX"
        except FetchError as exc:
            notes.append(f"OKX has no perpetual or funding history for {coin} ({exc}); the file has spot prices only.")
    except FetchError as exc:
        if exc.blocked:
            notes.append(f"OKX did not answer ({exc}).")
        if progress:
            progress(f"{coin}: spot prices from Kraken, Bitstamp or Coinbase")
        spot, name = _spot_fallback(http, coin, days)
        sources["spot"] = f"{name} (USD)"
    frame = pd.DataFrame({"time": spot.index, "spot": spot.to_numpy()})
    if perp is not None:
        frame["perp"] = frame["time"].map(perp)
    if funding is not None:
        frame["funding_daily"] = frame["time"].map(funding)
    frame = frame.dropna(subset=["spot"]).reset_index(drop=True)
    if len(frame) < MIN_DAYS:
        raise FetchError(f"only {len(frame)} days of {coin} exist at the exchange; the Crypto tab needs at least {MIN_DAYS} daily prices")
    notes = [f"Daily closes of {coin}: spot from {sources['spot']}" + (f", perpetual from {sources['perp']}" if "perp" in sources else "") + (f", funding from {sources['funding']}" if "funding" in sources else "")
             + f"; {len(frame)} days to {frame['time'].iloc[-1].date()}. The day in progress is left out.", "Daily closes are at 00:00 UTC; the platform's own BTC and ETH series use the 08:00 UTC mark, so the two are not on the same clock."] + notes
    if "OKX (USDT)" in sources.values():
        notes.append("OKX quotes in USDT, not dollars: a few basis points of the basis can be the USDT/USD rate.")
    meta = {"source_id": "okx" if "perp" in sources else "exchange", "source": ", ".join(dict.fromkeys(sources.values())), "url": "public candle endpoints", "as_of": str(frame["time"].iloc[-1].date()), "coin": coin,
            "sources": sources, "first_date": str(frame["time"].iloc[0].date()), "notes": notes}
    return Download("crypto", coin, "history", frame=frame, meta=meta)


def fetch_funding(http: Http, coin: str, days: int = 400, progress=None) -> Download:
    """OKX's funding history for ``COIN-USDT-SWAP`` as a daily series, saved as ``funding_<COIN>_OKX`` (for the funding view's OKX row)."""
    coin = check_coin(coin)
    if progress:
        progress(f"{coin}: funding history from OKX")
    s = okx_funding_daily(http, f"{coin}-USDT-SWAP", int(days))
    frame = pd.DataFrame({"time": s.index, "funding_daily": s.to_numpy()})
    meta = {"source_id": "okx", "source": "OKX", "url": f"{OKX}/public/funding-rate-history", "as_of": str(frame["time"].iloc[-1].date()), "coin": coin, "venue": "OKX",
            "notes": [f"OKX's funding settlements of {coin}-USDT-SWAP summed per UTC day: the funding earned by a short (positive when longs pay shorts). {len(frame)} days to {frame['time'].iloc[-1].date()}."]}
    return Download("crypto", f"funding_{coin}_OKX", "funding", frame=frame, meta=meta)
