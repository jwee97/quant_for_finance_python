"""The Massive adapter: every endpoint this app uses, as documented at https://massive.com/docs/rest (read 2026-11-22), and nothing about options maths or curves.

Each asset class has its own paths and shapes and none is assumed to follow another:

=========  ================================================================  ===========================================================================================
stocks     ``/v2/aggs/ticker/{t}/range/{m}/{ts}/{from}/{to}``                ``o h l c v vw n t`` (t in ms); ``adjusted`` defaults to true
           ``/v3/reference/tickers``, ``/v3/reference/tickers/{t}``           discovery and overview
           ``/stocks/v1/splits``, ``/stocks/v1/dividends``                    (the ``/v3/reference`` ones are deprecated); ``historical_adjustment_factor``
           ``/v3/quotes/{t}``, ``/v3/trades/{t}``                             nanosecond ``sip_timestamp``
options    ``/v3/reference/options/contracts``                                ``expired`` is false unless asked; ``as_of`` gives the chain of a past date
           ``/v3/snapshot/options/{underlying}``                              ``greeks`` (delta gamma theta vega), ``implied_volatility``, ``open_interest``: provider values, may be absent
           ``/v2/aggs/ticker/O:.../range/...``, ``/v3/quotes|trades/O:...``   same shapes as stocks
futures    ``/futures/v1/products``, ``/futures/v1/contracts``                ``last_trade_date`` (expiry), ``settlement_date``; no open interest in any futures endpoint read
           ``/futures/v1/aggs/{ticker}``                                      ``open high low close volume``, ``settlement_price`` on session candles; ``window_start`` in ns
           ``/futures/v1/trades/{ticker}``, ``/futures/v1/snapshot``          (futures quotes were not read in the documentation: not enabled)
crypto     ``/v2/aggs/ticker/X:BTCUSD/range/...``, ``/v3/reference/tickers``  spot pairs (``market=crypto``): never a futures price
=========  ================================================================  ===========================================================================================

What a key can reach depends on the plan; the documentation says which plan has which dataset but not what a refusal looks like, so :meth:`MassiveProvider.probe` asks once per dataset and records
what was answered, refused or empty (``entitlements.json``). Nothing is reported ``available`` that was not asked.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .client import MassiveClient
from .models import Bars, BarsRequest, Capability, MarketDataError, Provenance, Table

PROVIDER = "massive"
NY = "America/New_York"
FUTURES_RESOLUTION = {"second": "sec", "minute": "min", "hour": "hour", "day": "session", "week": "week", "month": "month", "quarter": "quarter", "year": "year"}

# (asset class, dataset, endpoint, the plan note in the documentation)
DOCUMENTED = [
    ("stocks", "bars", "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}", "all Stocks plans; Basic end-of-day 2 years, Starter 5 years delayed, Developer 10 years, Advanced all history"),
    ("stocks", "tickers", "/v3/reference/tickers", "all Stocks plans"),
    ("stocks", "overview", "/v3/reference/tickers/{ticker}", "all Stocks plans"),
    ("stocks", "splits", "/stocks/v1/splits", "all Stocks plans"),
    ("stocks", "dividends", "/stocks/v1/dividends", "all Stocks plans"),
    ("stocks", "trades", "/v3/trades/{ticker}", "Developer and Advanced"),
    ("stocks", "quotes", "/v3/quotes/{ticker}", "Advanced"),
    ("options", "contracts", "/v3/reference/options/contracts", "all Options plans"),
    ("options", "chain_snapshot", "/v3/snapshot/options/{underlying}", "Starter, Developer, Advanced"),
    ("options", "bars", "/v2/aggs/ticker/{option}/range/{multiplier}/{timespan}/{from}/{to}", "all Options plans; Basic end-of-day 2 years"),
    ("options", "trades", "/v3/trades/{option}", "Developer and Advanced"),
    ("options", "quotes", "/v3/quotes/{option}", "Advanced"),
    ("futures", "products", "/futures/v1/products", "all Futures plans"),
    ("futures", "contracts", "/futures/v1/contracts", "all Futures plans"),
    ("futures", "bars", "/futures/v1/aggs/{ticker}", "all Futures plans; Basic 8-hour historical, Starter and Developer 10-minute delayed"),
    ("futures", "snapshot", "/futures/v1/snapshot", "Starter, Developer, Advanced"),
    ("futures", "trades", "/futures/v1/trades/{ticker}", "Developer and Advanced"),
    ("crypto", "bars", "/v2/aggs/ticker/{pair}/range/{multiplier}/{timespan}/{from}/{to}", "Currencies plans (as the page states)"),
    ("crypto", "tickers", "/v3/reference/tickers?market=crypto", "Currencies plans (as the page states)"),
]
LIMITS = {"bars": 50000, "contracts_options": 1000, "contracts_futures": 1000, "tickers": 1000, "quotes": 50000, "trades": 50000, "snapshot_options": 250, "actions": 5000}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _day(value) -> str:
    return str(pd.Timestamp(value).date())


def _gate(name):
    return name


class MassiveProvider:
    name = PROVIDER
    label = "Massive"

    def __init__(self, client: MassiveClient | None = None, root: Path | None = None):
        self.client = client or MassiveClient(root=root)
        self.root = Path(root) if root else self.client.root
        self._caps: dict[str, Capability] = {}
        self._load_caps()

    # ------------------------------------------------------------------------------------------------------------ capabilities and entitlements
    def _caps_path(self) -> Path | None:
        return (self.root / "data" / "user" / "massive" / "entitlements.json") if self.root else None

    def _load_caps(self) -> None:
        for c in DOCUMENTED:
            self._caps[f"{c[0]}.{c[1]}"] = Capability(c[0], c[1], c[2], "documented", "", "", c[3])
        p = self._caps_path()
        try:
            if p and p.is_file():
                for k, v in json.loads(p.read_text()).items():
                    if k in self._caps:
                        self._caps[k] = Capability(**{**self._caps[k].as_dict(), **{x: v.get(x, "") for x in ("status", "detail", "checked_at")}})
        except (OSError, ValueError, TypeError):
            pass

    def _save_caps(self) -> None:
        p = self._caps_path()
        if not p:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v.as_dict() for k, v in self._caps.items()}))
            tmp.replace(p)
        except OSError:
            pass

    def _record(self, key: str, status: str, detail: str = "") -> None:
        if key in self._caps:
            c = self._caps[key]
            c.status, c.detail, c.checked_at = status, detail[:300], _now()
            self._save_caps()

    def get_provider_capabilities(self) -> list[Capability]:
        """Every documented dataset with what is known about this key: ``documented`` until a probe or a real request said otherwise."""
        return list(self._caps.values())

    def _need(self, key: str) -> None:
        """Refuse up front a dataset a recent request showed the plan does not include, with the plan note: no request is spent on a known refusal."""
        c = self._caps.get(key)
        if c and c.status == "not_entitled":
            raise MarketDataError(f"Massive refused this dataset for your key ({c.asset_class} {c.dataset}); documentation: {c.plan_note}. Checked {c.checked_at}. Run the entitlement check again after changing plan.", "entitlement", 403, c.endpoint)

    def _call(self, key: str, fn):
        """Run ``fn`` and keep what the answer says about the entitlement of ``key``."""
        self._need(key)
        try:
            out = fn()
        except MarketDataError as exc:
            if exc.kind == "entitlement":
                self._record(key, "not_entitled", str(exc))
            elif exc.kind not in ("cancelled", "rate_limit", "network", "not_configured"):
                self._record(key, "error", str(exc))
            raise
        if self._caps.get(key) and self._caps[key].status in ("documented", "not_entitled", "error"):
            self._record(key, "available", "answered a request")
        return out

    def probe(self, asset_classes=("stocks", "options", "futures", "crypto"), cancel=None) -> list[Capability]:
        """Ask once per dataset (one result each) and record the answers. About 15 requests; a probe never reads more than one row."""
        probes = {
            "stocks.bars": lambda: self.get_historical_bars(BarsRequest("AAPL", "stocks", _day(datetime.now(timezone.utc) - timedelta(days=10)), _day(datetime.now(timezone.utc)), limit=5)),
            "stocks.tickers": lambda: self.get_instruments("stocks", limit=1, max_pages=1),
            "stocks.overview": lambda: self.client.get("/v3/reference/tickers/AAPL", cancel=cancel),
            "stocks.splits": lambda: self.client.get("/stocks/v1/splits", {"limit": 1}, cancel=cancel),
            "stocks.dividends": lambda: self.client.get("/stocks/v1/dividends", {"limit": 1}, cancel=cancel),
            "stocks.trades": lambda: self.client.get("/v3/trades/AAPL", {"limit": 1}, cancel=cancel),
            "stocks.quotes": lambda: self.client.get("/v3/quotes/AAPL", {"limit": 1}, cancel=cancel),
            "options.contracts": lambda: self.client.get("/v3/reference/options/contracts", {"underlying_ticker": "SPY", "limit": 1}, cancel=cancel),
            "options.chain_snapshot": lambda: self.client.get("/v3/snapshot/options/SPY", {"limit": 1}, cancel=cancel),
            "futures.products": lambda: self.client.get("/futures/v1/products", {"limit": 1}, cancel=cancel),
            "futures.contracts": lambda: self.client.get("/futures/v1/contracts", {"limit": 1, "active": "true"}, cancel=cancel),
            "futures.snapshot": lambda: self.client.get("/futures/v1/snapshot", {"limit": 1}, cancel=cancel),
            "crypto.bars": lambda: self.get_historical_bars(BarsRequest("X:BTCUSD", "crypto", _day(datetime.now(timezone.utc) - timedelta(days=10)), _day(datetime.now(timezone.utc)), limit=5)),
            "crypto.tickers": lambda: self.client.get("/v3/reference/tickers", {"market": "crypto", "limit": 1}, cancel=cancel),
        }
        for key, run in probes.items():
            if key.split(".")[0] not in asset_classes:
                continue
            self._caps[key].status = "documented" if self._caps[key].status == "not_entitled" else self._caps[key].status
            try:
                out = run()
                n = len(out.frame) if isinstance(out, Bars) else len((out or {}).get("results") or [] if isinstance(out, dict) else [])
                if isinstance(out, dict) and isinstance(out.get("results"), dict):
                    n = 1
                self._record(key, "available" if n else "no_data", "answered" if n else "answered with no rows")
            except MarketDataError as exc:
                if exc.kind in ("cancelled", "not_configured", "auth", "network"):
                    raise
                self._record(key, "not_entitled" if exc.kind == "entitlement" else "error", str(exc))
        # futures bars and trades need a ticker: take the first active contract
        if "futures" in asset_classes and self._caps["futures.contracts"].status == "available":
            try:
                rows = (self.client.get("/futures/v1/contracts", {"limit": 1, "active": "true", "type": "single"}).get("results") or [])
                if rows:
                    t = rows[0]["ticker"]
                    for key, path, params in (("futures.bars", f"/futures/v1/aggs/{t}", {"resolution": "1session", "limit": 1}), ("futures.trades", f"/futures/v1/trades/{t}", {"limit": 1})):
                        try:
                            body = self.client.get(path, params, cancel=cancel)
                            self._record(key, "available" if body.get("results") else "no_data", "answered" if body.get("results") else "answered with no rows")
                        except MarketDataError as exc:
                            self._record(key, "not_entitled" if exc.kind == "entitlement" else "error", str(exc))
            except MarketDataError:
                pass
        if "options" in asset_classes and self._caps["options.contracts"].status == "available":
            try:
                rows = (self.client.get("/v3/reference/options/contracts", {"underlying_ticker": "SPY", "limit": 1}).get("results") or [])
                if rows:
                    t = rows[0]["ticker"]
                    end = _day(datetime.now(timezone.utc))
                    for key, path, params in (("options.bars", f"/v2/aggs/ticker/{t}/range/1/day/{_day(datetime.now(timezone.utc) - timedelta(days=30))}/{end}", {"limit": 1}),
                                              ("options.trades", f"/v3/trades/{t}", {"limit": 1}), ("options.quotes", f"/v3/quotes/{t}", {"limit": 1})):
                        try:
                            body = self.client.get(path, params, cancel=cancel)
                            self._record(key, "available" if body.get("results") else "no_data", "answered" if body.get("results") else "answered with no rows")
                        except MarketDataError as exc:
                            self._record(key, "not_entitled" if exc.kind == "entitlement" else "error", str(exc))
            except MarketDataError:
                pass
        return self.get_provider_capabilities()

    # ------------------------------------------------------------------------------------------------------------ helpers
    def _prov(self, endpoint: str, asset_class: str, adjusted=None, pages: int = 1, notes=()) -> Provenance:
        return Provenance(PROVIDER, endpoint, _now(), asset_class, adjusted, pages, tuple(notes))

    def _collect(self, path: str, params: dict, max_pages: int, cancel=None, on_page=None) -> tuple[list, int]:
        rows, pages = [], 0
        for results, _body in self.client.paginate(path, params, max_pages=max_pages, cancel=cancel, on_page=on_page):
            rows.extend(results)
            pages += 1
        return rows, pages

    # ------------------------------------------------------------------------------------------------------------ discovery
    def get_instruments(self, asset_class: str = "stocks", search: str | None = None, active: bool = True, ticker: str | None = None, type_: str | None = None, limit: int = 100, max_pages: int = 5,
                        cancel=None) -> Table:
        """Symbols: stocks, ETFs and crypto pairs from ``/v3/reference/tickers`` (``market`` stocks/crypto/fx/otc/indices); futures products from ``/futures/v1/products``."""
        if asset_class == "futures":
            key = "futures.products"
            rows, pages = self._call(key, lambda: self._collect("/futures/v1/products", {"limit": min(limit, 1000), "name": search}, max_pages, cancel))
            frame = pd.DataFrame(rows)
            return Table(frame, self._prov("/futures/v1/products", "futures", None, pages))
        market = {"stocks": "stocks", "crypto": "crypto", "indices": "indices"}.get(asset_class)
        if market is None:
            raise MarketDataError(f"instrument discovery for '{asset_class}' goes through contracts (options) or products (futures)", "unsupported")
        key = "crypto.tickers" if asset_class == "crypto" else "stocks.tickers"
        params = {"market": market, "active": "true" if active else "false", "search": search, "ticker": ticker, "type": type_, "limit": min(limit, LIMITS["tickers"]), "order": "asc", "sort": "ticker"}
        rows, pages = self._call(key, lambda: self._collect("/v3/reference/tickers", params, max_pages, cancel))
        frame = pd.DataFrame(rows)
        return Table(frame, self._prov("/v3/reference/tickers", asset_class, None, pages, ["Massive does not return CUSIP numbers"] if asset_class == "stocks" else []))

    def get_instrument(self, ticker: str, date: str | None = None) -> dict:
        """The overview of one stock or ETF (name, exchange, type, listing date, shares outstanding, ...)."""
        body = self._call("stocks.overview", lambda: self.client.get(f"/v3/reference/tickers/{ticker}", {"date": date}))
        return body.get("results") or {}

    def get_contracts(self, asset_class: str, underlying: str | None = None, as_of: str | None = None, expired: bool | None = None, contract_type: str | None = None,
                      expiration_from: str | None = None, expiration_to: str | None = None, strike_from: float | None = None, strike_to: float | None = None, active: bool | None = None,
                      limit: int = 1000, max_pages: int = 20, cancel=None) -> Table:
        """Option contracts (``/v3/reference/options/contracts``; ``as_of`` is the date the chain is read for, ``expired=True`` includes the expired ones) or futures contracts
        (``/futures/v1/contracts``; ``underlying`` is the product code, ``as_of`` the ``date`` the contract list is read for)."""
        if asset_class == "options":
            params = {"underlying_ticker": underlying, "as_of": as_of, "expired": None if expired is None else str(bool(expired)).lower(), "contract_type": contract_type, "expiration_date.gte": expiration_from,
                      "expiration_date.lte": expiration_to, "strike_price.gte": strike_from, "strike_price.lte": strike_to, "limit": min(limit, LIMITS["contracts_options"]), "sort": "expiration_date", "order": "asc"}
            rows, pages = self._call("options.contracts", lambda: self._collect("/v3/reference/options/contracts", params, max_pages, cancel))
            frame = pd.DataFrame(rows)
            if len(frame):
                frame["expiration_date"] = pd.to_datetime(frame["expiration_date"])
            return Table(frame, self._prov("/v3/reference/options/contracts", "options", None, pages))
        if asset_class == "futures":
            params = {"product_code": underlying, "date": as_of, "active": None if active is None else str(bool(active)).lower(), "type": "single", "last_trade_date.gte": expiration_from,
                      "last_trade_date.lte": expiration_to, "limit": min(limit, LIMITS["contracts_futures"]), "sort": "last_trade_date.asc"}
            rows, pages = self._call("futures.contracts", lambda: self._collect("/futures/v1/contracts", params, max_pages, cancel))
            frame = pd.DataFrame(rows)
            for c in ("first_trade_date", "last_trade_date", "settlement_date", "date"):
                if c in frame:
                    frame[c] = pd.to_datetime(frame[c], errors="coerce")
            return Table(frame, self._prov("/futures/v1/contracts", "futures", None, pages, ["open interest is not in any Massive futures endpoint that was read"]))
        raise MarketDataError(f"'{asset_class}' has no contracts: use get_instruments", "unsupported")

    # ------------------------------------------------------------------------------------------------------------ bars
    def get_historical_bars(self, req: BarsRequest, cancel=None, on_page=None) -> Bars:
        """OHLCV bars. Stocks, options and crypto share ``/v2/aggs``; futures use ``/futures/v1/aggs``. A daily bar of a stock or option is dated in New York time, of crypto in UTC, of a futures
        session by its ``session_end_date``."""
        ac, sym = req.asset_class, req.symbol
        end = req.end or _day(datetime.now(timezone.utc))
        start = req.start or _day(pd.Timestamp(end) - pd.Timedelta(days=365))
        if req.timespan not in FUTURES_RESOLUTION:
            raise MarketDataError(f"timespan must be one of {sorted(FUTURES_RESOLUTION)}", "bad_request")
        if ac == "futures":
            res = f"{req.multiplier}{FUTURES_RESOLUTION[req.timespan]}"
            params = {"resolution": res, "window_start.gte": start, "window_start.lte": end, "limit": min(req.limit, LIMITS["bars"]), "sort": "window_start.asc"}
            rows, pages = self._call("futures.bars", lambda: self._collect(f"/futures/v1/aggs/{sym}", params, req.max_pages, cancel, on_page))
            frame = self._futures_frame(rows, req.timespan)
            return Bars(req, frame, self._prov(f"/futures/v1/aggs/{{ticker}}?resolution={res}", ac, None, pages, ["session bars carry the exchange settlement price in settlement_price"]))
        if ac not in ("stocks", "options", "crypto", "indices"):
            raise MarketDataError(f"no bars for asset class '{ac}'", "unsupported")
        key = {"stocks": "stocks.bars", "options": "options.bars", "crypto": "crypto.bars"}.get(ac, "stocks.bars")
        path = f"/v2/aggs/ticker/{sym}/range/{req.multiplier}/{req.timespan}/{start}/{end}"
        params = {"adjusted": str(bool(req.adjusted)).lower(), "sort": "asc", "limit": min(req.limit, LIMITS["bars"])}
        rows, pages = self._call(key, lambda: self._collect(path, params, req.max_pages, cancel, on_page))
        frame = self._agg_frame(rows, req.timespan, "UTC" if ac == "crypto" else NY)
        note = ["adjusted flag is the provider's own (splits); dividends are not in the price"] if ac == "stocks" and req.adjusted else []
        return Bars(req, frame, self._prov("/v2/aggs/ticker/{ticker}/range", ac, req.adjusted, pages, note))

    @staticmethod
    def _agg_frame(rows: list, timespan: str, tz: str) -> pd.DataFrame:
        cols = ["open", "high", "low", "close", "volume", "vwap", "transactions"]
        if not rows:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
        df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "transactions"})
        ts = pd.to_datetime(df["t"], unit="ms", utc=True)
        if timespan in ("day", "week", "month", "quarter", "year"):
            idx = pd.DatetimeIndex(ts.dt.tz_convert(tz).dt.tz_localize(None).dt.normalize(), name="date")
        else:
            idx = pd.DatetimeIndex(ts, name="timestamp")
        for c in cols:
            if c not in df:
                df[c] = np.nan
        out = df[cols].astype(float)
        out.index = idx
        return out[~out.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _futures_frame(rows: list, timespan: str) -> pd.DataFrame:
        cols = ["open", "high", "low", "close", "volume", "vwap", "transactions", "settlement_price"]
        if not rows:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
        df = pd.DataFrame(rows)
        if timespan == "day" and "session_end_date" in df:
            idx = pd.DatetimeIndex(pd.to_datetime(df["session_end_date"]), name="date")
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(df["window_start"], unit="ns", utc=True), name="timestamp")
        for c in cols:
            if c not in df:
                df[c] = np.nan
        if "dollar_volume" in df:
            df["vwap"] = np.nan                                                     # dollar_volume has no contract multiplier: a price per unit cannot be recovered from it
        out = df[cols].astype(float)
        out.index = idx
        return out[~out.index.duplicated(keep="last")].sort_index()

    # ------------------------------------------------------------------------------------------------------------ quotes, trades
    def _ticks(self, kind: str, symbol: str, asset_class: str, start, end, limit: int, max_pages: int, cancel) -> Table:
        if asset_class == "futures":
            if kind == "quotes":
                raise MarketDataError("futures quotes are not enabled: Massive's futures quotes documentation was not verified", "unsupported")
            params = {"timestamp.gte": start, "timestamp.lte": end, "limit": min(limit, LIMITS["trades"]), "sort": "timestamp.asc"}
            rows, pages = self._call("futures.trades", lambda: self._collect(f"/futures/v1/trades/{symbol}", params, max_pages, cancel))
            frame = pd.DataFrame(rows)
            if len(frame):
                frame.index = pd.to_datetime(frame["timestamp"], unit="ns", utc=True)
            return Table(frame, self._prov("/futures/v1/trades/{ticker}", "futures", None, pages))
        if asset_class not in ("stocks", "options"):
            raise MarketDataError(f"{kind} are not enabled for '{asset_class}'", "unsupported")
        params = {"timestamp.gte": start, "timestamp.lte": end, "limit": min(limit, LIMITS[kind]), "order": "asc", "sort": "timestamp"}
        rows, pages = self._call(f"{asset_class}.{kind}", lambda: self._collect(f"/v3/{kind}/{symbol}", params, max_pages, cancel))
        frame = pd.DataFrame(rows)
        if len(frame) and "sip_timestamp" in frame:
            frame.index = pd.to_datetime(frame["sip_timestamp"], unit="ns", utc=True)
        notes = ["an option's price is per underlying share: the dollar value is price x size x the contract multiplier (usually 100)"] if asset_class == "options" and kind == "trades" else []
        return Table(frame, self._prov(f"/v3/{kind}/{{ticker}}", asset_class, None, pages, notes))

    def get_quotes(self, symbol: str, asset_class: str = "stocks", start=None, end=None, limit: int = 1000, max_pages: int = 5, cancel=None) -> Table:
        return self._ticks("quotes", symbol, asset_class, start, end, limit, max_pages, cancel)

    def get_trades(self, symbol: str, asset_class: str = "stocks", start=None, end=None, limit: int = 1000, max_pages: int = 5, cancel=None) -> Table:
        return self._ticks("trades", symbol, asset_class, start, end, limit, max_pages, cancel)

    # ------------------------------------------------------------------------------------------------------------ corporate actions
    def get_corporate_actions(self, ticker: str, max_pages: int = 10, cancel=None) -> Table:
        """Splits (with ``historical_adjustment_factor``) and dividends of a stock, one table with a ``kind`` column. Price for date D: multiply the unadjusted price by the factor of the first split whose
        execution date is after D (Massive's own rule)."""
        splits, p1 = self._call("stocks.splits", lambda: self._collect("/stocks/v1/splits", {"ticker": ticker, "limit": LIMITS["actions"], "sort": "execution_date.desc"}, max_pages, cancel))
        divs, p2 = self._call("stocks.dividends", lambda: self._collect("/stocks/v1/dividends", {"ticker": ticker, "limit": LIMITS["actions"], "sort": "ex_dividend_date.desc"}, max_pages, cancel))
        s = pd.DataFrame(splits)
        d = pd.DataFrame(divs)
        if len(s):
            s["kind"], s["date"] = "split", pd.to_datetime(s["execution_date"])
        if len(d):
            d["kind"], d["date"] = "dividend", pd.to_datetime(d["ex_dividend_date"])
        frame = pd.concat([x for x in (s, d) if len(x)], ignore_index=True, sort=False).sort_values("date") if (len(s) or len(d)) else pd.DataFrame(columns=["kind", "date"])
        return Table(frame, self._prov("/stocks/v1/splits + /stocks/v1/dividends", "stocks", None, p1 + p2))

    # ------------------------------------------------------------------------------------------------------------ option chain (provider snapshot)
    def get_option_chain(self, underlying: str, expiration_from: str | None = None, expiration_to: str | None = None, contract_type: str | None = None, max_pages: int = 40, cancel=None) -> Table:
        """The provider's chain snapshot (``/v3/snapshot/options``): quotes, last trade, day bar, open interest, and the provider's own implied volatility and Greeks where it sends them. These are
        CURRENT values (delayed 15 minutes on Starter and Developer); a chain of a past date is rebuilt by ``service.reconstruct_option_chain`` from contracts and bars, with no bid or ask."""
        params = {"expiration_date.gte": expiration_from, "expiration_date.lte": expiration_to, "contract_type": contract_type, "limit": LIMITS["snapshot_options"], "sort": "expiration_date", "order": "asc"}
        rows, pages = self._call("options.chain_snapshot", lambda: self._collect(f"/v3/snapshot/options/{underlying}", params, max_pages, cancel))
        recs, warns = [], []
        for r in rows:
            det, g, day, q, lt, ua = (r.get(k) or {} for k in ("details", "greeks", "day", "last_quote", "last_trade", "underlying_asset"))
            recs.append({"ticker": det.get("ticker"), "expiry": det.get("expiration_date"), "strike": det.get("strike_price"), "right": "C" if det.get("contract_type") == "call" else "P" if det.get("contract_type") == "put" else None,
                         "exercise_style": det.get("exercise_style"), "shares_per_contract": det.get("shares_per_contract"), "bid": q.get("bid"), "ask": q.get("ask"), "bid_size": q.get("bid_size"),
                         "ask_size": q.get("ask_size"), "last": lt.get("price"), "day_close": day.get("close"), "volume": day.get("volume"), "open_interest": r.get("open_interest"),
                         "iv": r.get("implied_volatility"), "delta": g.get("delta"), "gamma": g.get("gamma"), "theta": g.get("theta"), "vega": g.get("vega"), "underlying": ua.get("price"),
                         "quote_time": q.get("last_updated") or q.get("sip_timestamp")})
        frame = pd.DataFrame(recs)
        if len(frame):
            frame["expiry"] = pd.to_datetime(frame["expiry"])
            for c in ("strike", "bid", "ask", "iv", "delta", "gamma", "theta", "vega", "underlying", "open_interest", "volume", "last", "day_close"):
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
            frame["iv_source"] = np.where(frame["iv"].notna(), "provider", "missing")
            frame["greeks_source"] = np.where(frame["delta"].notna(), "provider", "missing")
            miss = int(frame["iv"].isna().sum())
            if miss:
                warns.append(f"{miss} of {len(frame)} contracts have no provider implied volatility (Massive omits it, for example for deep in-the-money contracts); compute it internally and label it so")
        return Table(frame, self._prov("/v3/snapshot/options/{underlying}", "options", None, pages, ["current snapshot: not a historical chain"]), warns)

    # ------------------------------------------------------------------------------------------------------------ futures curve
    def get_futures_curve(self, product_code: str, as_of: str | None = None, max_contracts: int = 12, cancel=None) -> Table:
        """The curve of a futures product on a date: the contracts alive on it (``/futures/v1/contracts`` with ``date``) and for each the last session bar up to that date. The price is the exchange
        settlement price where the session bar has one, else the close, and ``price_source`` says which. Open interest is not available from Massive (``open_interest`` is NaN)."""
        day = _day(as_of or datetime.now(timezone.utc))
        contracts = self.get_contracts("futures", product_code, as_of=day, active=True, limit=200, max_pages=2, cancel=cancel)
        c = contracts.frame
        if not len(c):
            return Table(pd.DataFrame(), contracts.provenance, [f"Massive lists no active {product_code} contracts on {day}"])
        c = c.sort_values("last_trade_date").head(max_contracts)
        rows, warns = [], []
        for _, k in c.iterrows():
            try:
                bars = self.get_historical_bars(BarsRequest(k["ticker"], "futures", _day(pd.Timestamp(day) - pd.Timedelta(days=10)), day, timespan="day", limit=20), cancel=cancel).frame
            except MarketDataError as exc:
                if exc.kind in ("entitlement", "cancelled", "auth"):
                    raise
                warns.append(f"{k['ticker']}: {exc}")
                continue
            bars = bars[bars.index <= pd.Timestamp(day)]
            if not len(bars):
                warns.append(f"{k['ticker']}: no session bar up to {day}")
                continue
            last = bars.iloc[-1]
            settle = last.get("settlement_price")
            rows.append({"date": bars.index[-1], "contract": k["ticker"], "expiry": k["last_trade_date"], "settlement_date": k.get("settlement_date"), "price": float(settle if pd.notna(settle) else last["close"]),
                         "price_source": "settlement" if pd.notna(settle) else "close", "volume": float(last["volume"]), "open_interest": np.nan})
        frame = pd.DataFrame(rows)
        if len(frame):
            frame["ttm"] = ((frame["expiry"] - pd.Timestamp(day)).dt.days / 365.0).clip(lower=0.0)
        return Table(frame, self._prov("/futures/v1/contracts + /futures/v1/aggs/{ticker}", "futures", None, 1, ["open interest is not available from Massive"]), warns)

    # ------------------------------------------------------------------------------------------------------------ status
    def describe(self) -> dict:
        ok, why = self.client.available()
        return {"configured": ok, "problem": why if not ok else "", "base_url": self.client.base_url, "key_notes": self.client.key_notes() if self.client.key else [], "usage": self.client.usage()}
