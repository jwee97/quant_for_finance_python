"""Payloads in the exact shape of the free public feeds, built from a known model so that a test can check what a parser recovers, and a transport that answers from a table instead of the network.

The shapes were taken from live answers of each feed (Cboe delayed quotes, Deribit book summaries, Yahoo's chart endpoint, Coinbase, Kraken, Bitstamp and OKX); the numbers are generated.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np

from src.derivatives.pricing import bsm_price


class FakeFeeds:
    """A transport ``(url, headers, timeout) -> (status, headers, body)`` that answers from ``routes``: ``{url prefix: payload | (status, payload) | callable(url) -> payload}``. The longest prefix that
    matches wins; an address nobody registered answers 404. ``calls`` records every address asked."""

    def __init__(self, routes: dict | None = None):
        self.routes = dict(routes or {})
        self.calls: list[str] = []

    def __call__(self, url: str, headers: dict, timeout: float):
        self.calls.append(url)
        for prefix in sorted(self.routes, key=len, reverse=True):
            if url.startswith(prefix):
                answer = self.routes[prefix]
                if callable(answer):
                    answer = answer(url)
                status, body = answer if isinstance(answer, tuple) and len(answer) == 2 and isinstance(answer[0], int) else (200, answer)
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                return status, {}, body if isinstance(body, bytes) else str(body).encode()
        return 404, {}, b"{}"


def cboe_payload(symbol: str = "SPY", spot: float = 500.0, quote_date: str = "2026-10-09", expiry_days=(5, 12, 33, 96, 187), vol: float = 0.20, rate: float = 0.04, div: float = 0.013,
                 strikes_pct=(-0.15, -0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10, 0.15), spread: float = 0.04, step: float = 5.0) -> dict:
    """A Cboe delayed-quote document for ``symbol``: Black-Scholes mids at one volatility, a rate and a dividend yield, a fixed relative spread, OCC contract names."""
    day = datetime.strptime(quote_date, "%Y-%m-%d")
    options = []
    for days in expiry_days:
        expiry = day + timedelta(days=days)
        T = days / 365.0
        for pct in strikes_pct:
            strike = round(spot * (1 + pct) / step) * step
            for right in ("C", "P"):
                mid = float(bsm_price(spot, strike, T, rate, div, vol, right == "C"))
                half = max(0.01, mid * spread / 2)
                options.append({"option": f"{symbol}{expiry:%y%m%d}{right}{int(round(strike * 1000)):08d}", "bid": round(max(mid - half, 0.0), 2), "ask": round(mid + half, 2), "iv": round(vol, 4),
                                "open_interest": 100.0 + days, "volume": 10.0, "delta": 0.5, "gamma": 0.01, "vega": 0.1, "theta": -0.01, "rho": 0.0, "last_trade_price": round(mid, 2),
                                "last_trade_time": f"{quote_date}T15:59:00"})
    return {"timestamp": f"{quote_date} 20:00:00", "symbol": symbol, "data": {"symbol": symbol, "current_price": spot, "last_trade_time": f"{quote_date}T15:59:59", "options": options}}


def deribit_options_payload(currency: str = "BTC", index: float = 80000.0, quote_date: str = "2026-10-11", expiry_days=(5, 19, 47, 110), vol: float = 0.50, basis: float = 0.05,
                            strikes_pct=(-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)) -> dict:
    """Deribit ``get_book_summary_by_currency`` for options: prices in coin units, ``underlying_price`` the forward of each expiry, ``estimated_delivery_price`` the index."""
    day = datetime.strptime(quote_date, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=14)
    out = []
    for days in expiry_days:
        expiry = day + timedelta(days=days)
        T = days / 365.0
        forward = index * float(np.exp(basis * T))
        for pct in strikes_pct:
            strike = round(index * (1 + pct) / 500.0) * 500.0
            for right in ("C", "P"):
                usd = float(bsm_price(forward, strike, T, 0.0, 0.0, vol, right == "C"))              # Black-76 on the forward
                coin = usd / index
                out.append({"instrument_name": f"{currency}-{expiry:%d%b%y}".upper() + f"-{int(strike)}-{right}", "bid_price": round(coin * 0.97, 4), "ask_price": round(coin * 1.03 + 0.0005, 4),
                            "mark_price": coin, "mark_iv": vol * 100.0, "underlying_price": forward, "estimated_delivery_price": index, "open_interest": 12.0, "volume": 3.0,
                            "creation_timestamp": int(day.timestamp() * 1000), "base_currency": currency, "quote_currency": currency})
    return {"jsonrpc": "2.0", "result": out}


def yahoo_chart(symbol: str, dates, closes, volumes=None, gmtoffset: int = -14400) -> dict:
    """Yahoo's ``v8/finance/chart`` answer for one symbol: a bar per date, stamped at the exchange's local midnight like the live feed (04:00 UTC in the US summer)."""
    stamps = [int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) - gmtoffset for d in dates]
    return {"chart": {"result": [{"meta": {"symbol": symbol, "gmtoffset": gmtoffset, "instrumentType": "FUTURE"}, "timestamp": stamps,
                                  "indicators": {"quote": [{"close": list(closes), "volume": list(volumes if volumes is not None else [1000] * len(dates)), "open": list(closes), "high": list(closes), "low": list(closes)}]}}],
                      "error": None}}


def yahoo_future_doc(symbol: str, name: str, exchange: str = "CME", kind: str = "future") -> dict:
    """One document of Yahoo's lookup service (``/v1/finance/lookup?type=future``): a name that carries the delivery month and is cut at about thirty characters, as the live one is."""
    return {"exchange": exchange, "quoteType": kind, "shortName": name[:31], "symbol": symbol, "rank": 1000, "regularMarketPrice": 100.0}


def yahoo_lookup(docs: list[dict]) -> dict:
    return {"finance": {"result": [{"count": len(docs), "start": 0, "total": len(docs), "documents": docs}], "error": None}}


YAHOO_MISSING = {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"}}}


# ------------------------------------------------------------------------------------------------------------------ crypto venues
def _ladder(mid: float, n: int, tick: float, size: float, side: int, fmt=float):
    """``n`` price levels away from the mid (``side`` +1 asks, -1 bids), each ``size`` coins, ``tick`` apart, starting half a tick from the mid."""
    return [(fmt(mid + side * (i + 0.5) * tick), size) for i in range(n)]


def coinbase_book(mid=100.0, n=30, tick=0.01, size=5.0) -> dict:
    return {"bids": [[str(p), str(s), 3] for p, s in _ladder(mid, n, tick, size, -1)], "asks": [[str(p), str(s), 2] for p, s in _ladder(mid, n, tick, size, +1)], "sequence": 1, "auction_mode": False}


def kraken_depth(pair_key="XXBTZUSD", mid=100.0, n=30, tick=0.01, size=5.0) -> dict:
    return {"error": [], "result": {pair_key: {"asks": [[f"{p:.5f}", f"{s:.3f}", 1791729274] for p, s in _ladder(mid, n, tick, size, +1)], "bids": [[f"{p:.5f}", f"{s:.3f}", 1791729270] for p, s in _ladder(mid, n, tick, size, -1)]}}}


def bitstamp_book(mid=100.0, n=30, tick=0.01, size=5.0) -> dict:
    return {"timestamp": "1791729277", "microtimestamp": "1791729277000000", "bids": [[f"{p:.4f}", f"{s:.8f}"] for p, s in _ladder(mid, n, tick, size, -1)],
            "asks": [[f"{p:.4f}", f"{s:.8f}"] for p, s in _ladder(mid, n, tick, size, +1)]}


def okx_book(mid=100.0, n=30, tick=0.01, size=5.0, contracts=False) -> dict:
    """OKX's ``market/books``; for a perpetual the sizes are contracts (here ``size`` / 0.01 of them, so 1 contract = 0.01 coin)."""
    scale = 100.0 if contracts else 1.0
    return {"code": "0", "msg": "", "data": [{"asks": [[str(p), str(s * scale), "0", "3"] for p, s in _ladder(mid, n, tick, size, +1)], "bids": [[str(p), str(s * scale), "0", "2"] for p, s in _ladder(mid, n, tick, size, -1)],
                                              "ts": "1791729278906", "seqId": 1}]}


def okx_instrument(ct_val="0.01") -> dict:
    return {"code": "0", "msg": "", "data": [{"instId": "X-USDT-SWAP", "ctVal": ct_val, "ctMult": "1", "instType": "SWAP"}]}


def deribit_book(mid=100.0, n=30, tick=0.01, usd_per_level=500.0) -> dict:
    """Deribit's inverse perpetual: the size of a level is in US dollars."""
    return {"result": {"bids": [[p, usd_per_level] for p, _ in _ladder(mid, n, tick, 1.0, -1)], "asks": [[p, usd_per_level] for p, _ in _ladder(mid, n, tick, 1.0, +1)], "mark_price": mid, "funding_8h": 0.0001,
                       "timestamp": 1791729278000}}


def okx_candles(days: list[str], closes: list[float]):
    """A route function for ``market/history-candles``: pages of 100 candles, newest first, older than the ``after`` stamp; the newest candle is unconfirmed (the day in progress)."""
    stamps = [int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) for d in days]
    rows = [[str(t), str(c), str(c), str(c), str(c), "1", "1", "1", "1"] for t, c in zip(stamps, closes)][::-1]
    rows[0][8] = "0"

    def route(url: str):
        after = None
        if "after=" in url:
            after = int(url.split("after=")[1].split("&")[0])
        page = [r for r in rows if after is None or int(r[0]) < after][:100]
        return {"code": "0", "msg": "", "data": page}

    return route


def okx_funding(days: list[str], rate_per_settlement: float = 0.0001):
    """A route function for ``public/funding-rate-history``: three settlements a day at 00:00, 08:00 and 16:00 UTC, newest first, 100 a page, older than ``after``."""
    stamps = []
    for d in days:
        base = int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        stamps += [base + h * 3600 * 1000 for h in (0, 8, 16)]
    rows = [{"fundingTime": str(t), "fundingRate": str(rate_per_settlement), "realizedRate": str(rate_per_settlement), "instId": "X-USDT-SWAP"} for t in sorted(stamps, reverse=True)]

    def route(url: str):
        after = int(url.split("after=")[1].split("&")[0]) if "after=" in url else None
        return {"code": "0", "msg": "", "data": [r for r in rows if after is None or int(r["fundingTime"]) < after][:100]}

    return route


def kraken_ohlc(days: list[str], closes: list[float], pair_key="XXBTZUSD") -> dict:
    stamps = [int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for d in days]
    return {"error": [], "result": {pair_key: [[t, str(c), str(c), str(c), str(c), str(c), "1.0", 10] for t, c in zip(stamps, closes)], "last": stamps[-1]}}


def bitstamp_ohlc(days: list[str], closes: list[float]) -> dict:
    stamps = [int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for d in days]
    return {"data": {"pair": "BTC/USD", "ohlc": [{"timestamp": str(t), "open": str(c), "high": str(c), "low": str(c), "close": f"{c:.2f}", "volume": "1"} for t, c in zip(stamps, closes)]}}


def coinbase_candles(days: list[str], closes: list[float]) -> list:
    stamps = [int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for d in days]
    return [[t, c, c, c, c, 1.0] for t, c in zip(stamps, closes)][::-1]


OKX_UNKNOWN = {"code": "51001", "msg": "Instrument ID, Instrument ID code, or Spread ID doesn't exist.", "data": []}
