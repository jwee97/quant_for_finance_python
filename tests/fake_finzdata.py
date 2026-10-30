"""A stand-in for Finzdata's futures endpoints, for tests (the real service needs an account and the network).

``FakeFinzdata`` is a transport: the client calls it as ``transport(url, headers, timeout) -> (status, headers, body)``. It serves ``/v1/futures/roots``, ``/v1/futures/contracts`` and ``/v1/futures/daily``
from a small deterministic world (E-mini S&P 500 with quarterly contracts, WTI crude with monthly ones, the euro future under Finzdata's own symbol, and the DAX, which the platform has no terms for), checks the ``X-API-Key`` header, clamps the period to the
last 30 days the way the free plan does (with a ``history_note``) and has a switch for each behaviour the code has to survive:

* ``per_contract_only``  asking for a root without a contract is a 422 (the service may insist on one);
* ``anonymous_rows``     the rows of a root-wide answer do not say which contract they are of;
* ``alt_names``          other spellings of the columns (``symbol``, ``trade_date``, ``settle``, ``oi``, ``expiration``);
* ``no_expiry``          the contract list has no expiry date;
* ``forbidden``          roots a plan does not include (HTTP 403);
* ``quota``              the number of calls after which the day's allowance is over (HTTP 429);
* ``full_list``          the list of markets is the 139 of the public page (the four above with their contracts, the rest a name and an exchange only), as the smoke check expects of the real service;
* ``plan_excludes``      parts of a path (``/v1/crypto``) that the plan does not include: HTTP 403.

The other endpoints (``/v1/me``, ``/v1/datasets``, prices, crypto, options, fundamentals, positioning) answer a few rows in the shape that seemed likeliest, for the smoke check that reports their columns.

The OpenAPI document of the real service leaves the rows undefined, so the shapes here are the likeliest ones, not a copy of the service's.
"""

from __future__ import annotations

import json
import zlib
from urllib import parse

import numpy as np
import pandas as pd

KEY = "fz_TESTKEY0123456789abcdefABCDEF"
MONTH_CODES = "FGHJKMNQUVXZ"
TODAY = "2026-10-29"


def _table(*rows: dict) -> dict:
    return {"data": list(rows)}


OTHER = {
    "/v1/me": lambda q: {"plan": "free", "history_days": 30, "daily_limit": 1000, "calls_today": 7, "email": "someone@example.com", "api_key_prefix": "fz_TEST", "customer": {"id": "cus_123", "name": "A Person"}},
    "/v1/datasets": lambda q: _table({"key": "futures"}, {"key": "prices"}, {"key": "crypto"}, {"key": "gex"}),
    "/v1/prices/daily": lambda q: _table({"date": "2026-10-28", "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 10}, {"date": "2026-10-29", "open": 1.1, "high": 1.3, "low": 1.0, "close": 1.2, "volume": 12}),
    "/v1/crypto/pairs": lambda q: _table({"symbol": "ETHUSDT", "base": "ETH", "quote": "USDT"}, {"symbol": "BTCUSDT", "base": "BTC", "quote": "USDT"}),
    "/v1/crypto/bars": lambda q: _table({"ts": "2026-10-29T00:00:00Z", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 3}),
    "/v1/gex/daily": lambda q: _table({"date": "2026-10-29", "ticker": "SPY", "net_gex": 1.5e9, "flip": 5900.0}),
    "/v1/options/eod": lambda q: _table({"date": "2026-10-29", "expiry": "2026-11-20", "right": "C", "strike": 600.0, "bid": 1.0, "ask": 1.1, "iv": 0.2}),
    "/v1/fundamentals": lambda q: _table({"ticker": "AAPL", "concept": "Revenues", "value": 4.0e11, "period_end": "2025-09-27"}),
    "/v1/cot": lambda q: _table({"report_date": "2026-10-27", "market": "E-MINI S&P 500", "noncomm_long": 1, "noncomm_short": 2}),
}

WORLD = {
    "ES": {"name": "E-mini S&P 500", "exchange": "CME", "asset_class": "Equity index", "months": (3, 6, 9, 12), "base": 6000.0, "since": 2008, "expiry": "third_friday"},
    "CL": {"name": "Crude Oil WTI", "exchange": "NYMEX", "asset_class": "Energy", "months": tuple(range(1, 13)), "base": 70.0, "since": 2008, "expiry": "twentieth"},
    "E6": {"name": "Euro FX", "exchange": "CME", "asset_class": "Currency", "months": (3, 6, 9, 12), "base": 1.08, "since": 2008, "expiry": "third_friday"},
    "FDAX": {"name": "DAX", "exchange": "Eurex", "asset_class": "Equity index", "months": (3, 6, 9, 12), "base": 24000.0, "since": 2008, "expiry": "third_friday"},
}


def expiry_of(root: str, year: int, month: int) -> pd.Timestamp:
    if WORLD[root]["expiry"] == "third_friday":
        first = pd.Timestamp(year, month, 1)
        return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
    return pd.Timestamp(year, month, 20)


class FakeFinzdata:
    def __init__(self, key: str = KEY, today: str = TODAY, window: int = 30, per_contract_only: bool = False, anonymous_rows: bool = False, alt_names: bool = False, no_expiry: bool = False,
                 forbidden: tuple = (), quota: int | None = None, plan_note: bool = True, full_list: bool = False, plan_excludes: tuple = ()):
        self.key, self.today, self.window = key, pd.Timestamp(today), window
        self.per_contract_only, self.anonymous_rows, self.alt_names, self.no_expiry = per_contract_only, anonymous_rows, alt_names, no_expiry
        self.forbidden, self.quota, self.plan_note, self.full_list, self.plan_excludes = set(forbidden), quota, plan_note, full_list, tuple(plan_excludes)
        self.requests: list[tuple[str, dict]] = []          # path and query of every call that reached a route
        self.urls: list[str] = []
        self.keys: list[str | None] = []                    # the X-API-Key header of every call
        self.script: list[tuple[int, dict, bytes]] = []     # answers to give, in order, before behaving normally

    # ------------------------------------------------------------------------------------------------------------ the world
    def contracts(self, root: str) -> list[dict]:
        spec, out = WORLD[root], []
        for year in range(self.today.year - 3, self.today.year + 3):
            for month in spec["months"]:
                expiry = expiry_of(root, year, month)
                first = expiry - pd.Timedelta(days=700)
                out.append({"code": f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}", "year": year, "month": month, "expiry": expiry, "first": first, "last": min(expiry, self.today)})
        return out

    def bars(self, root: str, c: dict, start: pd.Timestamp) -> list[dict]:
        days = pd.bdate_range(max(start, c["first"]), min(c["last"], self.today))
        rng = np.random.default_rng(zlib.crc32(c["code"].encode()))
        walk = np.cumsum(rng.normal(0, 0.004, len(days)))
        base = WORLD[root]["base"] * (1 + 0.002 * ((c["year"] - self.today.year) * 12 + c["month"]))
        rows = []
        for i, day in enumerate(days):
            price = float(base * np.exp(walk[i]))
            rows.append({"date": day, "price": round(price, 4), "volume": int(1000 + 50 * rng.integers(0, 20)), "oi": int(50_000 + 100 * rng.integers(0, 50)), "code": c["code"]})
        return rows

    # ------------------------------------------------------------------------------------------------------------ the rows, spelt either way
    def _roots_rows(self) -> list[dict]:
        out = []
        for root, spec in WORLD.items():
            row = {"root": root, "name": spec["name"], "exchange": spec["exchange"], "asset_class": spec["asset_class"], "first_date": f"{spec['since']}-01-02", "last_date": str(self.today.date()),
                   "contracts": len(self.contracts(root))}
            if self.alt_names:
                row = {"symbol": f"/{root}", "description": spec["name"], "venue": spec["exchange"], "class": spec["asset_class"], "start": f"{spec['since']}-01-02", "end": str(self.today.date())}
            out.append(row)
        if self.full_list:
            from src.assets.free_data import finzdata_catalog as catalog

            for m in catalog.bundled():
                if m.root in WORLD:
                    continue
                out.append({"symbol": f"/{m.root}", "description": m.name, "venue": m.exchange, "class": m.group} if self.alt_names else {"root": m.root, "name": m.name, "exchange": m.exchange, "asset_class": m.group})
        return out

    def _contract_rows(self, root: str) -> list[dict]:
        out = []
        for c in self.contracts(root):
            row = {"contract": c["code"], "root": root, "expiry": str(c["expiry"].date()), "first_date": str(c["first"].date()), "last_date": str(c["last"].date())}
            if self.alt_names:
                row = {"symbol": c["code"], "expiration": str(c["expiry"].date()), "first": str(c["first"].date()), "last": str(c["last"].date())}
            if self.no_expiry:
                row.pop("expiry", None), row.pop("expiration", None)
            out.append(row)
        return out

    def _bar_rows(self, root: str, contract: str | None, start: pd.Timestamp, with_code: bool = True) -> list[dict]:
        rows = []
        for c in self.contracts(root):
            if contract is not None and c["code"] != contract:
                continue
            for r in self.bars(root, c, start):
                if self.alt_names:
                    row = {"trade_date": str(r["date"].date()), "contract_code": r["code"], "settle": r["price"], "volume": r["volume"], "oi": r["oi"]}
                else:
                    row = {"date": str(r["date"].date()), "contract": r["code"], "open": r["price"], "high": r["price"], "low": r["price"], "close": r["price"], "volume": r["volume"],
                           "open_interest": r["oi"]}
                if not with_code:
                    row.pop("contract", None), row.pop("contract_code", None)
                rows.append(row)
        return rows

    # ------------------------------------------------------------------------------------------------------------ the transport
    def __call__(self, url: str, headers: dict, timeout: float):
        self.urls.append(url)
        self.keys.append(headers.get("X-API-Key"))
        if self.script:
            return self.script.pop(0)
        parts = parse.urlsplit(url)
        query = {k: v[0] for k, v in parse.parse_qs(parts.query).items()}
        if headers.get("X-API-Key") != self.key:
            return 401, {}, json.dumps({"detail": "invalid or missing API key. Get a free key with just an email at https://finzdata.com (full history: https://finzdata.com/shop)"}).encode()
        self.requests.append((parts.path, query))
        if self.quota is not None and len(self.requests) > self.quota:
            return 429, {}, json.dumps({"detail": "daily call limit reached"}).encode()
        if parts.path in OTHER:
            if any(part in parts.path for part in self.plan_excludes):
                return 403, {}, json.dumps({"detail": "plan does not include this dataset"}).encode()
            return 200, {}, json.dumps(OTHER[parts.path](query)).encode()
        root = query.get("root", "").upper()
        if parts.path == "/v1/futures/roots":
            return 200, {}, json.dumps({"data": self._roots_rows()}).encode()
        if root not in WORLD:
            return 200, {}, json.dumps({"data": []}).encode()
        if root in self.forbidden:
            return 403, {}, json.dumps({"detail": "plan does not include this data"}).encode()
        if parts.path == "/v1/futures/contracts":
            return 200, {}, json.dumps({"data": self._contract_rows(root)}).encode()
        if parts.path == "/v1/futures/daily":
            contract = query.get("contract")
            if contract is None and self.per_contract_only:
                return 422, {}, json.dumps({"detail": [{"type": "missing", "loc": ["query", "contract"], "msg": "Field required"}]}).encode()
            earliest = self.today - pd.Timedelta(days=self.window)
            asked = pd.Timestamp(query["start"]) if "start" in query else pd.Timestamp("2000-01-01")
            start = max(asked, earliest)
            body: dict = {"data": self._bar_rows(root, contract, start, with_code=not (self.anonymous_rows and contract is None))}
            if self.plan_note and asked < earliest:
                body["history_note"] = f"Your plan includes the last {self.window} days of futures data, so the bars start on {start.date()}."
                body["upgrade"] = {"message": "Full history of one market is a one-time $49 purchase.", "url": "https://finzdata.com/shop"}
            return 200, {}, json.dumps(body).encode()
        return 404, {}, json.dumps({"detail": "Not Found"}).encode()
