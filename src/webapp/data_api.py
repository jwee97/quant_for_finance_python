"""The Free data endpoints of the Options, Futures and Crypto tabs: download real data from free public sources so that nothing has to be found and formatted by hand.

``GET  /api/data/status``   what is downloaded (with where from and when), what each tab can ask for, and which downloads are running
``GET  /api/data/search``   the futures products or the crypto pairs Massive lists that match a search (one call); with ``source=yahoo``, the futures Yahoo lists; with ``source=finzdata``,
                            Finzdata's 139 markets (no call)
``GET  /api/market/status`` Massive's connection, verified entitlements, coverage, freshness and warnings (never the key)
``POST /api/market/probe``  (a job) ask Massive once per dataset what this key can reach and remember the answers
``POST /api/market/sync``   (a job) bring stored bars of symbols up to date incrementally
``POST /api/market/migrate`` import the downloads earlier versions made from iTick into the normalised store, as provider ``itick``
``GET  /api/data/contracts`` every contract Finzdata lists for a root, with what is stored of each (``root=ES``; one call, or none within a day of the last)
``POST /api/data/lookup``  ask Yahoo's lookup service for the futures matching a word, and keep what it finds (one to three calls); with ``source=finzdata``, fetch Finzdata's own list of markets (one call)
``POST /api/data/fetch``    (a job) download for one tab: ``{"kind": "options" | "futures" | "crypto", "items": [...], ...settings}``
``POST /api/data/delete``   remove one download: ``{"kind", "name", "confirm": true}``

The browser names what it wants (a ticker, a futures root, a coin) and a few bounded settings; every address, file name and format is fixed here, so a request cannot reach another host or another file. A
download runs on its own worker, apart from the backtests, because a slow feed must not make a backtest wait nor the reverse. One download per tab at a time. The files and their records are kept by
``assets.free_data.store`` under ``data/user/downloads/``; the workspaces read them and label what they show ``downloaded``.

**Massive** (``MASSIVE_API_KEY``) is the keyed source for the three tabs, through ``marketdata.MarketDataService`` (one client, one limiter, one normalised store): contract-level futures curves with settlement
prices, a coin's daily SPOT history, the daily prices of an option underlying, Massive's chain snapshot and chains rebuilt for past dates; see ``assets.free_data.massive``. What the key may use is
verified by the entitlement probe, not assumed. Downloads that earlier versions made from iTick stay readable and are imported, as provider ``itick``, by ``marketdata.legacy``.

**Finzdata** is a futures source with a key of its own (the ``FINZDATA_API_KEY`` secret): every individual contract of 139 markets with open interest, which the free plan shows for the last 30 days
only. A download is merged into the file of the last one, so the history grows; see ``data_finzdata``.
"""

from __future__ import annotations

import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..assets.free_data import cboe_futures as cb
from ..assets.free_data import crypto as fc
from ..assets.free_data import finzdata_catalog as fzc
from ..assets.free_data import futures as ff
from ..assets.free_data import massive as fm
from ..assets.free_data import options as fo
from ..assets.free_data import yahoo_futures as yf
from ..assets.free_data.http import FetchError, Http
from ..assets.free_data.sources import describe
from ..assets.free_data.store import KINDS, DownloadStore
from ..assets.options_analytics import DEFAULT_RATE
from . import data_finzdata as dfz
from ..marketdata import legacy as market_legacy
from ..marketdata.models import MarketDataError
from ..marketdata.service import MarketDataService
from .api import ApiError, Job, _number
from .finzdata import FinzdataError
from .payload import clean
from .providers import RateLimitTimeout

MAX_ITEMS = 15
OPTION_PRESETS = ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "SPX")
WHAT = ("history", "venues", "funding")
WHAT_ALL = WHAT + ("openinterest", "liquidations")                                      # the last two are chosen on purpose: they ask three exchanges and keep only what those answer
SOURCE_IDS = {"options": ("massive", "cboe", "yahoo", "deribit"), "futures": ("massive", "yahoo", "finzdata", "cboe_cfe", "deribit"), "crypto": ("massive", "okx", "coinbase", "kraken", "bitstamp", "deribit")}
FUTURES_SOURCES = ("yahoo", "massive", "yahoo_series", "finzdata")   # the curve of each contract month from Yahoo or Massive, the price history of any future Yahoo lists, or every contract of a Finzdata market
CRYPTO_SOURCES = ("exchanges", "massive")
PRICE_SOURCES = ("massive", "yahoo")                         # where an option underlying's daily prices can come from
HOSTS = {"query1.finance.yahoo.com": 0.5, "www.okx.com": 0.12, "api.kraken.com": 0.3, "cdn.cboe.com": 0.15, "www-api.cboe.com": 0.15, "www.deribit.com": 0.15}
DERIBIT_MODES = ("curve", "history")                       # Bitcoin and Ether futures: today's strip with open interest, or every monthly and quarterly future since 2019 from the daily charts
FUTURES_HISTORY = ("6mo", "1y", "2y", "5y", "10y", "max")


def _items(body: dict, check, limit: int = MAX_ITEMS) -> list[str]:
    raw = body.get("items")
    if isinstance(raw, str):
        raw = [x for x in raw.replace(",", " ").split() if x]
    if not isinstance(raw, list) or not 1 <= len(raw) <= limit:
        raise ApiError(f"items must be a list of 1 to {limit} names")
    out = []
    for x in raw:
        try:
            name = check(str(x))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        if name not in out:
            out.append(name)
    return out


def _items_of(value, allowed: tuple) -> list[str]:
    """The members of ``value`` (a list) that are in ``allowed``; all of ``allowed`` when it is empty. An unknown member is an error."""
    if not value:
        return list(allowed)
    if not isinstance(value, list) or any(v not in allowed for v in value):
        raise ApiError(f"must be a list from {list(allowed)}")
    return list(value)


def _check_option_item(x: str) -> str:
    x = x.strip()
    if x.lower().startswith("deribit:"):
        coin = x.split(":", 1)[1].strip().upper()
        if coin not in fo.DERIBIT_CURRENCIES:
            raise ValueError(f"Deribit options are available for {', '.join(fo.DERIBIT_CURRENCIES)}")
        return f"deribit:{coin}"
    return fo.check_symbol(x)


def _check_root(x: str) -> str:
    root = x.strip().upper()
    if root not in ff.FREE_ROOTS and root not in ff.CRYPTO_ROOTS and root not in cb.CBOE_ROOTS:
        listed = list(ff.FREE_ROOTS)
        raise ValueError(f"free curves are available for {len(listed)} roots ({', '.join(listed[:8])}, …), {', '.join(cb.CBOE_ROOTS)} and {', '.join(ff.CRYPTO_ROOTS)}: pick one from the list, or choose iTick for any other futures product")
    return root


def _check_yahoo_future(x: str) -> str:
    symbol = x.strip().upper()
    if not yf.SYMBOL.match(symbol):
        raise ValueError(f"{x.strip()[:20]} is not a futures symbol Yahoo uses: ES=F (the front month) or CLZ26.NYM (one contract). Search the list in the card to find one")
    return symbol


def _check_symbol(x: str) -> str:
    """A Massive ticker: a stock (AAPL), an option (O:SPY240719C00100000), a crypto pair (X:BTCUSD), an index (I:SPX) or a futures contract (CLN4)."""
    t = x.strip()
    if not re.fullmatch(r"(?:[OXIC]:)?[A-Za-z0-9._-]{1,30}", t):
        raise ValueError(f"'{x}' is not a Massive ticker")
    return t


def _check_product(x: str) -> str:
    """A Massive futures product code (``CL``, ``ES``, ``6E``): letters and digits."""
    code = x.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{1,8}", code):
        raise ValueError(f"'{x}' is not a futures product code (letters and digits, such as CL or ES)")
    return code


def _choice(body: dict, key: str, default: str, allowed: tuple) -> str:
    value = str(body.get(key) or default)
    if value not in allowed:
        raise ApiError(f"{key} must be one of {list(allowed)}")
    return value


class DataApi:
    def __init__(self, app):
        self.app = app
        self.root = Path(app.root)
        self.store = DownloadStore(self.root)
        shared = getattr(app.builder.sources.get("massive"), "provider", None)
        self.market = getattr(shared, "service", None) or MarketDataService(self.root)      # one client, limiter and store for the Backtest tab and the data cards; a test replaces ``app.data.market.provider.client.session`` with a fake transport
        try:
            market_legacy.migrate(self.root, self.market.store)           # earlier iTick downloads become provider ``itick`` datasets: read only, once, harmless to repeat
        except Exception:                                                # noqa: BLE001 - an import problem must never stop the app; /api/market/migrate shows it
            traceback.print_exc()
        self.finz = dfz.FinzdataData(self.root, self.store)             # a test sets ``app.data.finz.client`` to a client with a fake transport
        self.http: Http | None = None                      # a test (or an app that must go through its own proxy) can set a client with another transport
        self.yfinance = None                               # likewise the ``yfinance`` module the option-chain fallback uses
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._pool = ThreadPoolExecutor(max_workers=1)     # downloads wait for each other, never for a backtest

    def get_routes(self) -> dict:
        return {"/api/data/status": self.status, "/api/data/search": self.search, "/api/data/contracts": self.contracts, "/api/market/status": self.market_status}

    def post_routes(self) -> dict:
        return {"/api/data/fetch": self.fetch, "/api/data/delete": self.delete, "/api/data/lookup": self.lookup, "/api/market/probe": self.market_probe, "/api/market/sync": self.market_sync,
                "/api/market/migrate": self.market_migrate}

    # ------------------------------------------------------------------------------------------------------------ Massive
    def _massive_status(self) -> dict:
        """Whether Massive can be used and what is known about the key's plan: only whether a key exists is ever reported, never the key."""
        try:
            return self.market.status()
        except Exception as exc:                                                       # noqa: BLE001 - a broken store must not take the whole page down
            return {"configured": False, "problem": f"{type(exc).__name__}: {exc}", "capabilities": [], "coverage": [], "warnings": [str(exc)]}

    def _need_massive(self) -> None:
        ok, why = self.market.provider.client.available()
        if not ok:
            raise ApiError(why)

    def market_status(self, q: dict | None = None) -> dict:
        """``GET /api/market/status``: connection, verified entitlements (what was asked and answered), coverage and freshness of the stored data, usage, integrity and warnings."""
        return clean(self._massive_status())

    def market_probe(self, body: dict) -> dict:
        """``POST /api/market/probe`` (a job): about fifteen one-row requests, one per dataset, recording what the key may use. Spends requests; run it after changing plan."""
        self._need_massive()
        classes = tuple(c for c in _items_of(body.get("asset_classes"), ("stocks", "options", "futures", "crypto")))

        def work(progress):
            progress("asking Massive what this key can reach")
            try:
                self.market.provider.probe(classes)
            except MarketDataError as exc:
                raise ApiError(str(exc)) from None
            return clean(self._massive_status())
        return self.app.submit_task("Massive: check entitlements", work, kind="data", lane="data")

    def market_sync(self, body: dict) -> dict:
        """``POST /api/market/sync`` (a job): ``{"symbols": [...], "asset_class": "stocks", "start": "2020-01-01", "adjusted": true}``: incremental bars into the normalised store."""
        self._need_massive()
        ac = _choice(body, "asset_class", "stocks", ("stocks", "options", "futures", "crypto", "indices"))
        symbols = _items({"items": body.get("symbols")}, lambda x: _check_symbol(x), limit=40)
        start, force = (str(body.get("start") or "") or None), bool(body.get("force"))
        adjusted = body.get("adjusted", True) is not False

        def work(progress):
            out = []
            for i, sym in enumerate(symbols):
                progress(f"{sym} ({i + 1} of {len(symbols)})")
                try:
                    r = self.market.sync_bars(sym, ac, start, adjusted=adjusted, progress=progress, force=force)
                    out.append({"symbol": sym, "ok": True, "added": r["added"], "revised": r["revised"], "skipped": r["skipped"], "warnings": r.get("warnings", []), "info": r.get("info")})
                except MarketDataError as exc:
                    out.append({"symbol": sym, "ok": False, "message": str(exc), "kind": exc.kind})
                    if exc.kind in ("auth", "not_configured", "cancelled"):
                        break
            if not any(o["ok"] for o in out):
                raise ApiError("Nothing could be synchronised. " + " | ".join(dict.fromkeys(o["message"] for o in out))[:600])
            return clean({"results": out, "status": self._massive_status()})
        return self.app.submit_task(f"Massive: synchronise {len(symbols)} {ac} symbol(s)", work, kind="data", lane="data")

    def market_migrate(self, body: dict) -> dict:
        """``POST /api/market/migrate``: import the earlier iTick downloads (read only) into the normalised store as provider ``itick``; repeating it imports nothing twice."""
        return clean(market_legacy.migrate(self.root, self.market.store))

    # ------------------------------------------------------------------------------------------------------------ status
    def _presets(self) -> dict:
        roots = list(ff.FREE_ROOTS.values())
        return {
            "options": {"symbols": list(OPTION_PRESETS), "crypto": list(fo.DERIBIT_CURRENCIES), "max_dte": [30, 90, 180, 365, 730], "defaults": {"max_dte": 365},
                        "prices": {"sources": [{"id": "massive", "text": "Massive (needs your key)"}, {"id": "yahoo", "text": "Yahoo Finance"}], "years": [1, 2, 3, 5, 10], "defaults": {"years": 3, "source": "yahoo"}},
                        "chains": {"sources": [{"id": "free", "text": "Cboe, then Yahoo (free, live)"}, {"id": "massive", "text": "Massive: current snapshot (needs your key)"}, {"id": "massive_past", "text": "Massive: a past date, rebuilt from daily closes"}]}},
            "futures": {"roots": [{"id": r.root, "name": r.name, "exchange": r.exchange, "group": r.group} for r in roots + list(cb.CBOE_ROOTS.values())], "groups": list(dict.fromkeys(r.group for r in roots + list(cb.CBOE_ROOTS.values()))),
                        "crypto": [{"id": k, "name": v} for k, v in ff.CRYPTO_ROOTS.items()], "history": list(FUTURES_HISTORY), "max_contracts": ff.MAX_CONTRACTS, "deribit": list(DERIBIT_MODES),
                        "defaults": {"contracts": 10, "history": "1y", "deribit": "curve"},
                        "massive": {"years": [1, 2, 3, 5], "defaults": {"years": 1}},
                        "yahoo": {"groups": yf.groups(self._yahoo_list()), "years": [1, 2, 3, 5, 10, 20], "products": len(yf.catalog(self._yahoo_list())), "defaults": {"years": 5}},
                        "finzdata": {"groups": fzc.groups(self.finz.live_roots()), "markets": len(fzc.markets(self.finz.live_roots())[0]), "max_calls_per_root": dfz.MAX_CALLS_PER_ROOT}},
            "crypto": {"coins": list(fc.DEFAULT_COINS), "what": [{"id": "history", "text": "Daily history: spot, perpetual and funding"}, {"id": "venues", "text": "Real quotes and order books on six venues, now"},
                                                                  {"id": "funding", "text": "OKX funding history, for the funding view"},
                                                                  {"id": "openinterest", "text": "Open interest by exchange (OKX, Bybit, Binance)"}, {"id": "liquidations", "text": "Recent liquidations (OKX, a short window)"}], "defaults": {"days": 400, "what": list(WHAT)},
                       "massive": {"note": "daily SPOT closes against the US dollar; Deribit and the exchanges supply derivatives"}},
        }

    def status(self, q: dict | None = None) -> dict:
        """``GET /api/data/status``: for each tab, its downloads (newest first, each with its source, address, dates, row count and notes), the sources it can use and what it offers to download."""
        presets = self._presets()
        massive = self._massive_status()
        kinds = {k: {"downloads": self.store.list(k), "sources": describe(SOURCE_IDS[k]), "massive_status": massive, **presets[k]} for k in KINDS}
        finzdata = self.finz.status()
        kinds["futures"]["finzdata_status"] = finzdata
        return clean({"kinds": kinds, "running": sorted(self._running), "directory": "data/user/downloads/", "limits": {"items": MAX_ITEMS}, "massive": clean(massive), "finzdata": finzdata,
                      "note": "Free public feeds: no account and no key, but delayed, unaudited and somebody else's service. A provider can refuse an address (common on shared cloud hosts) or change its format. "
                              "Massive and Finzdata are the exceptions: each needs your key (the MASSIVE_API_KEY and FINZDATA_API_KEY secrets), and what your plan includes decides what they return."})

    def search(self, q: dict) -> dict:
        """``GET /api/data/search?kind=futures&q=crude&group=energy&limit=30``: futures products from Massive (one call); ``kind=crypto`` searches its spot pairs against the dollar;
        ``source=yahoo`` searches the futures Yahoo lists (the bundled snapshot and what earlier lookups added)."""
        kind = str(q.get("kind") or "futures")
        text = str(q.get("q") or "")[:60]
        limit = _number(q, "limit", 30, 1, 100, integer=True)
        if kind == "futures" and str(q.get("source") or "") == "finzdata":
            group = str(q.get("group") or "") or None
            if group is not None and group not in fzc.groups(self.finz.live_roots()):
                raise ApiError(f"group must be one of {fzc.groups(self.finz.live_roots())}")
            return clean({"kind": kind, **self.finz.search(text, group, limit)})
        if kind == "futures" and str(q.get("source") or "") == "yahoo":
            group = str(q.get("group") or "") or None
            if group is not None and group not in yf.GROUPS:
                raise ApiError(f"group must be one of {list(yf.GROUPS)}")
            return clean({"kind": kind, "source": "yahoo", **yf.search(text, group, None, limit, self._yahoo_list()), "groups": yf.groups(self._yahoo_list())})
        if kind == "futures":
            self._need_massive()
            try:
                t = self.market.provider.get_instruments("futures", search=text or None, limit=min(max(limit, 20), 200), max_pages=1)
            except MarketDataError as exc:
                raise ApiError(str(exc)) from None
            f = t.frame
            rows = [{"item": str(r.get("product_code")), "code": str(r.get("product_code")), "name": str(r.get("name") or r.get("product_code")), "exchange": str(r.get("trading_venue") or ""), "region": "US",
                     "group": str(r.get("sector") or r.get("asset_class") or "")} for r in f.to_dict("records")] if len(f) else []
            group = str(q.get("group") or "")
            if group:
                rows = [r for r in rows if r["group"] == group]
            return clean({"kind": kind, "source": "massive", "results": rows[:limit], "total": len(rows), "shown": min(limit, len(rows)), "groups": sorted({r["group"] for r in rows if r["group"]})})
        if kind == "crypto":
            self._need_massive()
            try:
                t = self.market.provider.get_instruments("crypto", search=text or None, limit=min(max(limit, 20), 200), max_pages=1)
            except MarketDataError as exc:
                raise ApiError(str(exc)) from None
            rows = []
            for r in t.frame.to_dict("records") if len(t.frame) else []:
                m = str(r.get("ticker", ""))
                if m.startswith("X:") and m.endswith("USD") and not m.endswith("USDT"):
                    rows.append({"coin": m[2:-3], "item": m[2:-3], "name": str(r.get("name") or m)})
            return clean({"kind": kind, "source": "massive", "results": rows[:limit], "total": len(rows), "shown": min(limit, len(rows))})
        raise ApiError("kind must be futures or crypto")

    def contracts(self, q: dict) -> dict:
        """``GET /api/data/contracts?root=ES[&refresh=1]``: every contract Finzdata lists for a root, soonest expiry first, with what is stored of each. One call to Finzdata, or none when the list was
        fetched within the last day."""
        try:
            root = dfz.check_root(str(q.get("root") or ""))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        refresh = str(q.get("refresh") or "").lower() in ("1", "true", "yes")
        cached = (self.finz.user / dfz.CONTRACTS_FOLDER / f"{root}.json").exists()
        if refresh or not cached:
            ok, why = self.finz.client.available()
            if not ok:
                raise ApiError(why)
        try:
            return clean({"source": "finzdata", **self.finz.contracts(root, refresh)})
        except (FetchError, ValueError) as exc:
            raise ApiError(str(exc)) from None

    def _yahoo_list(self) -> Path:
        return self.root / "data" / "user" / "yahoo_futures.json"

    def lookup(self, body: dict) -> dict:
        """``POST /api/data/lookup`` with ``{"kind": "futures", "source": "yahoo", "q": "lumber"}``: ask Yahoo's lookup service for the futures that match (one to three calls, a few seconds), keep what it
        found so that the search has it from now on, and return it in the shape of a search. Yahoo has no page that lists its futures; this is how a name you type is found. With ``"source": "finzdata"`` it
        fetches Finzdata's own list of markets instead (one call) and keeps it for the search."""
        if str(body.get("kind") or "futures") == "futures" and str(body.get("source") or "") == "finzdata":
            ok, why = self.finz.client.available()
            if not ok:
                raise ApiError(why)
            try:
                done = self.finz.refresh_roots()
            except (FetchError, ValueError) as exc:
                raise ApiError(str(exc)) from None
            return clean({"kind": "futures", "source": "finzdata", **done, **self.finz.search(str(body.get("q") or ""), None, 30)})
        if str(body.get("kind") or "futures") != "futures" or str(body.get("source") or "yahoo") != "yahoo":
            raise ApiError("only the futures of Yahoo Finance can be looked up this way; Massive's products are searched with source=massive")
        http = self.http or Http(pauses=HOSTS, retries=1, backoff=1.5, cooldown=60.0)
        try:
            found = yf.lookup(http, str(body.get("q") or ""), self._yahoo_list())
        except (ValueError, FetchError) as exc:
            raise ApiError(str(exc)) from None
        return clean({"kind": "futures", "source": "yahoo", **found, "total": found["found"], "shown": len(found["results"]), "groups": yf.groups(self._yahoo_list())})

    # ------------------------------------------------------------------------------------------------------------ one item of each kind
    def _client(self) -> Http:
        return self.http or Http(pauses=HOSTS, retries=3, backoff=3.0)

    def _save(self, d, summary: str, refresh: dict, item: str | None = None) -> dict:
        """Write a download with the request that would repeat it (``refresh``: the page's Update button posts it back as it is). ``item`` is what was asked for when the file's name is not it (a coin's
        order books are saved as ``venues_SOL``)."""
        d.meta = {**d.meta, "refresh": refresh}
        rec = self.store.save(d)
        return {"item": item or d.name, "what": d.item, "ok": True, "name": d.name, "message": summary, "source": rec.get("source"), "as_of": rec.get("as_of"), "rows": rec.get("rows")}

    def _fail(self, item: str, what: str, exc: Exception) -> dict:
        if isinstance(exc, (FetchError, ValueError, ApiError, MarketDataError, RateLimitTimeout)):          # FinzdataError is a FetchError
            message = str(exc)
        else:
            traceback.print_exc()
            message = f"{type(exc).__name__}: {exc}"
        out = {"item": item, "what": what, "ok": False, "message": message}
        if isinstance(exc, RateLimitTimeout) or (isinstance(exc, FinzdataError) and exc.fatal) or (isinstance(exc, MarketDataError) and exc.kind in ("auth", "not_configured", "entitlement")):
            out["fatal"] = True                                                   # nothing else asked of this provider in this run can work: the loop stops spending requests
        return out

    def _known(self, kind: str, name: str, source_id: str, columns: dict) -> pd.DataFrame | None:
        """An earlier download of ``name`` from ``source_id`` as a frame indexed by date, or ``None``. A file from another source is never extended with this source's bars."""
        record = self.store.meta(kind, name)
        if not record or record.get("source_id") != source_id or not str(record.get("file", "")).endswith(".csv"):
            return None
        try:
            frame = pd.read_csv(self.store.path(kind, name)).rename(columns=columns)
            frame["date"] = pd.to_datetime(frame["date"])
            return frame.set_index("date").sort_index()
        except (OSError, ValueError, KeyError):
            return None

    def _one_option(self, http: Http, item: str, s: dict, progress=None) -> list[dict]:
        if "chain" not in s["what"]:
            return []
        try:
            if item.startswith("deribit:"):
                d = fo.fetch_deribit_chain(http, item.split(":", 1)[1], s["max_dte"])
            elif s["chain_source"] == "massive":
                d = fm.chain(self.market, item, s["rate"], s["max_dte"], progress)
            elif s["chain_source"] == "massive_past":
                d = fm.historical_chain(self.market, item, s["chain_date"], s["rate"], min(s["max_dte"], 120), progress=progress)
            else:
                d = fo.fetch_chain(http, item, s["rate"], s["max_dte"], self.yfinance)
            return [self._save(d, f"{d.meta['contracts']:,} contracts on {d.meta['expiries']} expiries, quotes of {d.meta['as_of']} ({d.meta['source']})", {"items": [item], "max_dte": s["max_dte"], "rate": s["rate"]})]
        except Exception as exc:                                              # noqa: BLE001 - one symbol failing must not lose the others
            return [self._fail(item, "chain", exc)]

    def _one_prices(self, http: Http, item: str, s: dict, progress) -> list[dict]:
        """The daily prices of an option's underlying, for the realised volatility beside the implied one (a Deribit chain's underlying is already in the platform's data)."""
        if "prices" not in s["what"]:
            return []
        if item.startswith("deribit:"):
            return [{"item": item, "what": "prices", "ok": True, "skipped": True, "message": f"left alone: the platform already holds the {item.split(':', 1)[1]} price history"}]
        try:
            if s["prices_source"] == "massive":
                d = fm.underlying_prices(self.market, item, s["years"], progress)
            else:
                d = fo.fetch_underlying_prices(http, item, s["years"])
            return [self._save(d, f"{d.meta['bars']:,} daily prices, {d.meta['first_date']} to {d.meta['as_of']} ({d.meta['source']})", {"items": [item], "what": ["prices"], "prices_source": s["prices_source"], "years": s["years"]}, item)]
        except Exception as exc:                                              # noqa: BLE001
            return [self._fail(item, "prices", exc)]

    def _one_root(self, http: Http, item: str, s: dict, progress) -> list[dict]:
        try:
            path = self.store.path("futures", item)
            earlier = pd.read_csv(path) if path.exists() and (self.store.meta("futures", item) or {}).get("item") == "curve" else None
            if item in cb.CBOE_ROOTS:
                d = cb.fetch_vx_curve(http, earlier=earlier, progress=progress)
            elif item in ff.CRYPTO_ROOTS:
                d = ff.fetch_deribit_history(http, item, progress=progress) if s["deribit"] == "history" else ff.fetch_deribit_curve(http, item)
            else:
                d = ff.fetch_yahoo_curve(http, item, s["contracts"], s["history"], progress=progress)
            if earlier is not None:
                d.frame = ff.merge_history(earlier, d.frame)
                d.meta["first_date"] = str(d.frame["date"].min().date())
                d.meta["notes"] = list(d.meta["notes"]) + [f"Merged with the earlier download: the file now runs from {d.meta['first_date']} and holds {d.frame['contract'].nunique()} contracts."]
            held = d.frame["contract"].nunique()
            msg = f"{held} contracts, {d.frame['date'].nunique()} days to {d.meta['as_of']} ({d.meta['source']})" + (" · incomplete: the feed stopped answering" if d.meta.get("incomplete") else "")
            return [self._save(d, msg, {"items": [item], "contracts": s["contracts"] or 0, "history": s["history"], "deribit": s["deribit"]})]      # 0: every contract Yahoo lists
        except Exception as exc:                                              # noqa: BLE001
            return [self._fail(item, "curve", exc)]

    def _one_finzdata(self, item: str, s: dict, progress) -> list[dict]:
        """Every contract of one Finzdata market, merged into the stored file (see ``data_finzdata``)."""
        try:
            return [self.finz.update_and_save(item, {"source": "finzdata", "items": [item]}, progress=progress, max_calls=dfz.MAX_CALLS_PER_ROOT)]
        except Exception as exc:                                              # noqa: BLE001
            return [self._fail(item, "curve", exc)]

    def _one_massive_curve(self, item: str, s: dict, progress) -> list[dict]:
        """A futures product's contract-level history from Massive, saved as the curve the Futures tab reads; the stored bars make a repeat ask only for the newest sessions."""
        try:
            d = fm.futures_curve(self.market, item, s["years"], progress)
            return [self._save(d, f"{d.meta['contracts']} contracts, {len(d.frame):,} rows to {d.meta['as_of']} (Massive)", {"source": "massive", "items": [item], "years": s["years"]}, item)]
        except Exception as exc:                                              # noqa: BLE001
            return [self._fail(item, "curve", exc)]

    def _one_yahoo_series(self, http: Http, item: str, s: dict, progress) -> list[dict]:
        """The daily price history of one future Yahoo lists (a continuous series such as ES=F, or one contract): one call, saved for the History view."""
        try:
            d = yf.fetch_series(http, item, s["years"], self._yahoo_list())
            return [self._save(d, f"{len(d.frame):,} daily bars, {d.meta['first_date']} to {d.meta['as_of']} (Yahoo Finance)", {"source": "yahoo_series", "items": [item], "years": s["years"]}, item)]
        except Exception as exc:                                              # noqa: BLE001
            return [self._fail(item, "series", exc)]

    def _one_coin(self, http: Http, coin: str, s: dict, progress) -> list[dict]:
        out = []
        for what in s["what"]:
            try:
                if what == "history":
                    if coin in ("BTC", "ETH"):
                        out.append({"item": coin, "what": "history", "ok": True, "skipped": True, "message": f"left alone: the platform already holds Deribit's {coin} history, which is longer than a download"})
                        continue
                    if s["source"] == "massive":
                        d = fm.coin_history(self.market, coin, s["days"], progress)
                        out.append(self._save(d, f"{d.meta['first_date']} to {d.meta['as_of']} ({d.meta['source']})", {"source": "massive", "items": [coin], "what": ["history"], "days": s["days"]}, coin))
                        continue
                    d = fc.fetch_history(http, coin, s["days"], progress)
                    out.append(self._save(d, f"{d.meta['first_date']} to {d.meta['as_of']} ({d.meta['source']})", {"items": [coin], "what": ["history"], "days": s["days"]}, coin))
                elif what == "venues":
                    d = fc.fetch_venues(http, coin, progress)
                    out.append(self._save(d, f"quotes and books on {len(d.payload['venues'])} venues" + (f", {len(d.payload['failed'])} unavailable" if d.payload["failed"] else ""), {"items": [coin], "what": ["venues"], "days": s["days"]}, coin))
                elif what == "openinterest":
                    d = fc.fetch_open_interest(http, coin, min(s["days"], 365), progress)
                    out.append(self._save(d, f"{d.meta['as_of']}: open interest on {len(d.meta['venues'])} venues", {"items": [coin], "what": ["openinterest"], "days": s["days"]}, coin))
                elif what == "liquidations":
                    d = fc.fetch_liquidations(http, coin, progress)
                    out.append(self._save(d, f"{len(d.payload['events'])} recent liquidations on OKX", {"items": [coin], "what": ["liquidations"], "days": s["days"]}, coin))
                else:
                    d = fc.fetch_funding(http, coin, s["days"], progress)
                    out.append(self._save(d, f"{d.meta['as_of']}: {len(d.frame)} days of OKX funding", {"items": [coin], "what": ["funding"], "days": s["days"]}, coin))
            except Exception as exc:                                          # noqa: BLE001
                out.append(self._fail(coin, what, exc))
        return out

    # ------------------------------------------------------------------------------------------------------------ the request
    def _settings(self, kind: str, body: dict) -> tuple[list[str], dict]:
        if kind == "options":
            what = body.get("what") or ["chain"]
            if not isinstance(what, list) or not what or any(w not in ("chain", "prices") for w in what):
                raise ApiError("what must be a list from ['chain', 'prices']")
            return _items(body, _check_option_item), {"max_dte": _number(body, "max_dte", 365, 7, fo.MAX_DTE, integer=True), "rate": _number(body, "rate", DEFAULT_RATE, -0.02, 0.2),
                                                       "what": [w for w in ("chain", "prices") if w in what], "prices_source": _choice(body, "prices_source", "yahoo", PRICE_SOURCES),
                                                       "years": _number(body, "years", 3, 1, 10, integer=True), "chain_source": _choice(body, "chain_source", "free", ("free", "massive", "massive_past")),
                                                       "chain_date": self._chain_date(body)}
        if kind == "futures":
            source = _choice(body, "source", "yahoo", FUTURES_SOURCES)
            if source == "finzdata":
                return _items(body, dfz.check_root, limit=40), {"source": "finzdata"}
            if source == "massive":
                return _items(body, _check_product), {"source": "massive", "years": _number(body, "years", 1, 1, 5, integer=True)}
            if source == "yahoo_series":
                return _items(body, _check_yahoo_future), {"source": "yahoo_series", "years": _number(body, "years", 5, 1, 20, integer=True)}
            history = str(body.get("history") or "1y")
            if history not in FUTURES_HISTORY:
                raise ApiError(f"history must be one of {', '.join(FUTURES_HISTORY)}")
            raw = body.get("contracts")
            contracts = None if str(raw).lower() in ("0", "all") else _number(body, "contracts", 10, 2, ff.MAX_CONTRACTS, integer=True)        # None: every contract Yahoo lists
            return _items(body, _check_root), {"source": "yahoo", "contracts": contracts, "history": history, "deribit": _choice(body, "deribit", "curve", DERIBIT_MODES)}
        source = _choice(body, "source", "exchanges", CRYPTO_SOURCES)
        what = body.get("what") or (["history"] if source == "massive" else list(WHAT))
        if not isinstance(what, list) or not what or any(w not in WHAT_ALL for w in what):
            raise ApiError(f"what must be a list from {list(WHAT_ALL)}")
        if source == "massive" and what != ["history"]:
            raise ApiError("Massive supplies the daily spot history of a coin; the order books, perpetuals and funding come from the exchanges and Deribit")
        return _items(body, fc.check_coin), {"source": source, "days": _number(body, "days", 400, fc.MIN_DAYS, 1500, integer=True), "what": [w for w in WHAT_ALL if w in what]}

    @staticmethod
    def _chain_date(body: dict) -> str | None:
        if str(body.get("chain_source") or "") != "massive_past":
            return None
        try:
            day = pd.Timestamp(str(body.get("chain_date") or ""))
        except (ValueError, TypeError):
            raise ApiError("chain_date must be a date (YYYY-MM-DD) for a chain rebuilt from a past day") from None
        if pd.isna(day) or day >= pd.Timestamp.now().normalize():
            raise ApiError("chain_date must be a past date")
        return str(day.date())

    def _uses_massive(self, kind: str, s: dict) -> bool:
        return (kind in ("futures", "crypto") and s.get("source") == "massive") or (kind == "options" and (s.get("chain_source", "free") != "free" or ("prices" in s["what"] and s["prices_source"] == "massive")))

    def fetch(self, body: dict) -> dict:
        """``POST /api/data/fetch`` (a job): download for one tab. The job's result lists every item with whether it worked and why not; a download where nothing worked is an error whose message says why."""
        kind = str(body.get("kind") or "")
        if kind not in KINDS:
            raise ApiError(f"kind must be one of {list(KINDS)}")
        items, settings = self._settings(kind, body)
        if kind == "futures" and settings.get("source") == "finzdata":
            ok, why = self.finz.client.available()
            if not ok:
                raise ApiError(why)
            left = self.finz.client.describe()["calls_left"]
            if 2 * len(items) > left:
                raise ApiError(f"this download needs at least {2 * len(items)} Finzdata calls (two for each market) and {left} are left today: choose fewer markets, or try tomorrow")
        if self._uses_massive(kind, settings):
            self._need_massive()
        with self._lock:
            if kind in self._running:
                raise ApiError(f"a {kind} download is already running: wait for it to finish")
            self._running.add(kind)

        def work(progress):
            http = self._client()
            results = []
            try:
                for i, item in enumerate(items):
                    progress(f"{kind}: {item} ({i + 1} of {len(items)})")
                    if kind == "options":
                        results += self._one_option(http, item, settings, progress) + self._one_prices(http, item, settings, progress)
                    elif kind == "futures":
                        if settings["source"] == "finzdata":
                            results += self._one_finzdata(item, settings, progress)
                        elif settings["source"] == "massive":
                            results += self._one_massive_curve(item, settings, progress)
                        elif settings["source"] == "yahoo_series":
                            results += self._one_yahoo_series(http, item, settings, progress)
                        else:
                            results += self._one_root(http, item, settings, progress)
                    else:
                        results += self._one_coin(http, item, settings, progress)
                    if results and results[-1].get("fatal"):                  # the provider refused the key or the plan: asking again for the rest would only spend requests
                        results += [{"item": rest, "what": results[-1]["what"], "ok": False, "message": results[-1]["message"]} for rest in items[i + 1:]]
                        break
            finally:
                with self._lock:
                    self._running.discard(kind)
            for r in results:
                r.pop("fatal", None)
            worked = [r for r in results if r["ok"]]
            if not worked:
                reasons = list(dict.fromkeys(r["message"] for r in results))
                raise ApiError(("Nothing could be downloaded. " + " | ".join(reasons[:3]) + (f" (and {len(reasons) - 3} other reasons)" if len(reasons) > 3 else "")).strip())
            return {"kind": kind, "results": results, "ok": len(worked), "failed": len(results) - len(worked), "downloads": self.store.list(kind)}

        try:
            job = self.app._register(Job(f"free data: {kind} ({len(items)})", "data"))
            self._pool.submit(self.app._run_task, job, work)
        except BaseException:
            with self._lock:
                self._running.discard(kind)
            raise
        return {"job": job.id}

    def delete(self, body: dict) -> dict:
        """``POST /api/data/delete``: remove one download and its record (the files in ``data/user/<tab>/`` are yours and are never touched)."""
        kind, name = str(body.get("kind") or ""), str(body.get("name") or "")
        if kind not in KINDS:
            raise ApiError(f"kind must be one of {list(KINDS)}")
        if body.get("confirm") is not True:
            raise ApiError("deleting a download needs confirm: true")
        try:
            gone = self.store.delete(kind, name)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        if not gone:
            raise ApiError(f"there is no download called '{name}' for {kind}")
        return {"deleted": name, "kind": kind, "downloads": self.store.list(kind)}
