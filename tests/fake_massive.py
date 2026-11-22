"""A stand-in for api.massive.com that follows the response shapes in Massive's documentation. It records every request (path, params, headers) so tests can check what was sent."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd

BASE = "https://api.massive.com"


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = self.content.decode()

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _bdays(a, b):
    return pd.bdate_range(a, b)


class FakeMassive:
    """``deny`` is a set of path prefixes answered with HTTP 403; ``flaky`` a list of statuses to answer first; ``page`` the page size used to exercise next_url."""

    def __init__(self, deny=(), flaky=(), page=3, today="2024-06-14", splits=None, gaps=()):
        self.calls: list[dict] = []
        self.deny, self.flaky, self.page, self.today = set(deny), list(flaky), page, today
        self.splits = splits or []
        self.gaps = set(gaps)
        self.headers = {}
        self.factor = 1.0                                  # a split's adjustment applied to every stock bar

    def mount(self, *a, **k):
        pass

    # --- the session interface the client uses
    def get(self, url, params=None, headers=None, timeout=None):
        parts = urlsplit(url)
        q = dict(parse_qsl(parts.query))
        q.update({k: str(v) for k, v in (params or {}).items()})
        self.calls.append({"path": parts.path, "params": q, "headers": dict(headers or {}), "url": url})
        if self.flaky:
            st = self.flaky.pop(0)
            return Resp(st, {"status": "ERROR", "message": "slow down"}, {"Retry-After": "0"})
        if (headers or {}).get("Authorization", "") != "Bearer TESTKEY123456":
            return Resp(401, {"status": "ERROR", "message": "Unknown API Key"})
        if any(parts.path.startswith(d) for d in self.deny):
            return Resp(403, {"status": "NOT_AUTHORIZED", "message": "You are not entitled to this data."})
        return self._route(parts.path, q)

    # --- routes
    def _page(self, rows, path, q):
        start = int(q.get("cursor", 0))
        chunk = rows[start:start + self.page]
        body = {"status": "OK", "count": len(chunk), "results": chunk, "request_id": "x"}
        if start + self.page < len(rows):
            body["next_url"] = f"{BASE}{path}?cursor={start + self.page}&" + "&".join(f"{k}={v}" for k, v in q.items() if k != "cursor")
        return Resp(200, body)

    def _route(self, path, q):
        m = re.match(r"^/v2/aggs/ticker/([^/]+)/range/(\d+)/(\w+)/([^/]+)/([^/]+)$", path)
        if m:
            tk, _, span, a, b = m.groups()
            idx = _bdays(a, b) if tk.startswith(("X:",)) is False else pd.date_range(a, b)
            idx = [d for d in idx if str(d.date()) not in self.gaps and str(d.date()) <= self.today]
            base = 100.0 if not tk.startswith("O:") else 5.0
            rows = []
            for d in idx:
                ms = int(pd.Timestamp(d).tz_localize("America/New_York" if not tk.startswith("X:") else "UTC").timestamp() * 1000)
                px = (base + (d.dayofyear % 17) * 0.1) * (self.factor if not tk.startswith(('O:', 'X:')) else 1.0)
                rows.append({"o": px, "h": px + 1, "l": px - 1, "c": px + 0.5, "v": 1000 + d.day, "vw": px, "n": 50, "t": ms})
            body = {"ticker": tk, "adjusted": q.get("adjusted") == "true", "resultsCount": len(rows), "status": "OK", "results": rows}
            return self._page(rows, path, q) if len(rows) > self.page else Resp(200, body)
        m = re.match(r"^/futures/v1/aggs/(\w+)$", path)
        if m:
            lo, hi = q.get("window_start.gte") or str((pd.Timestamp(self.today) - pd.Timedelta(days=10)).date()), q.get("window_start.lte") or self.today
            idx = [d for d in _bdays(lo, hi) if str(d.date()) <= self.today]
            rows = [{"window_start": int((pd.Timestamp(d) - pd.Timedelta(days=1)).value), "session_end_date": str(d.date()), "open": 70.0, "high": 71.0, "low": 69.0, "close": 70.5, "volume": 900, "transactions": 10,
                     "dollar_volume": 1.0, "settlement_price": 70.4 + 0.01 * i} for i, d in enumerate(idx)]
            return self._page(rows, path, q)
        if path == "/v3/reference/tickers":
            rows = [{"ticker": "AAPL", "name": "Apple", "market": q.get("market", "stocks"), "type": "CS", "active": True}, {"ticker": "MSFT", "name": "Microsoft", "market": "stocks", "type": "CS", "active": True}]
            return self._page(rows, path, q)
        if re.match(r"^/v3/reference/tickers/\w+$", path):
            return Resp(200, {"status": "OK", "results": {"ticker": "AAPL", "name": "Apple", "market_cap": 3e12}})
        if path == "/stocks/v1/splits":
            return self._page(self.splits, path, q)
        if path == "/stocks/v1/dividends":
            return self._page([{"ticker": "AAPL", "ex_dividend_date": "2024-02-09", "cash_amount": 0.24, "frequency": 4}], path, q)
        if path == "/v3/reference/options/contracts":
            exp, spot = "2024-07-19", 100.0
            rows = []
            for k in (90, 95, 100, 105, 110):
                for ct in ("call", "put"):
                    rows.append({"ticker": f"O:SPY240719{'C' if ct == 'call' else 'P'}{int(k * 1000):08d}", "underlying_ticker": q.get("underlying_ticker"), "contract_type": ct, "expiration_date": exp,
                                 "strike_price": float(k), "exercise_style": "american", "shares_per_contract": 100})
            return self._page(rows, path, q)
        if path.startswith("/v3/snapshot/options/"):
            rows = [{"details": {"ticker": "O:SPY240719C00100000", "contract_type": "call", "expiration_date": "2024-07-19", "strike_price": 100, "exercise_style": "american", "shares_per_contract": 100},
                     "greeks": {"delta": 0.5, "gamma": 0.05, "theta": -0.1, "vega": 0.2}, "implied_volatility": 0.22, "open_interest": 1234, "day": {"close": 3.1, "volume": 50},
                     "last_quote": {"bid": 3.0, "ask": 3.2, "bid_size": 5, "ask_size": 6}, "last_trade": {"price": 3.1}, "underlying_asset": {"price": 100.2}},
                    {"details": {"ticker": "O:SPY240719P00100000", "contract_type": "put", "expiration_date": "2024-07-19", "strike_price": 100}, "open_interest": 10, "day": {"close": 2.9},
                     "last_quote": {"bid": 2.8, "ask": 3.0}, "underlying_asset": {"price": 100.2}}]
            return self._page(rows, path, q)
        if path == "/futures/v1/contracts":
            date = q.get("date")
            rows = []
            for i, (code, last) in enumerate([("CLN4", "2024-06-20"), ("CLQ4", "2024-07-22"), ("CLU4", "2024-08-20"), ("CLV4", "2024-09-20"), ("CLG4", "2024-02-20")]):
                if date and not (pd.Timestamp("2024-01-01") <= pd.Timestamp(date) <= pd.Timestamp(last)):
                    continue
                rows.append({"ticker": code, "product_code": q.get("product_code") or "CL", "first_trade_date": "2023-01-02", "last_trade_date": last, "settlement_date": last, "active": True, "type": "single"})
            return self._page(rows, path, q)
        if path == "/futures/v1/products":
            return self._page([{"product_code": "CL", "name": "Crude Oil", "trading_venue": "XNYM", "sector": "energy"}], path, q)
        m = re.match(r"^/v3/(quotes|trades)/(.+)$", path)
        if m:
            kind = m.group(1)
            rows = [{"bid_price": 1.0, "ask_price": 1.1, "sip_timestamp": 1_700_000_000_000_000_000 + i} if kind == "quotes" else {"price": 1.0, "size": 1, "sip_timestamp": 1_700_000_000_000_000_000 + i} for i in range(4)]
            return self._page(rows, path, q)
        m = re.match(r"^/futures/v1/trades/(\w+)$", path)
        if m:
            return self._page([{"timestamp": 1_700_000_000_000_000_000 + i, "price": 70.0, "size": 1} for i in range(4)], path, q)
        return Resp(404, {"status": "ERROR", "message": "not found"})
