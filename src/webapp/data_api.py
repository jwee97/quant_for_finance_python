"""The Free data endpoints of the Options, Futures and Crypto tabs: download real data from free public sources so that nothing has to be found and formatted by hand.

``GET  /api/data/status``   what is downloaded (with where from and when), what each tab can ask for, and which downloads are running
``GET  /api/data/search``   the products of iTick's bundled list that match a search (futures) or the coins it quotes (crypto); with ``source=yahoo``, the futures Yahoo lists; with ``source=finzdata``,
                            Finzdata's 139 markets (no call)
``GET  /api/data/contracts`` every contract Finzdata lists for a root, with what is stored of each (``root=ES``; one call, or none within a day of the last)
``POST /api/data/lookup``  ask Yahoo's lookup service for the futures matching a word, and keep what it finds (one to three calls); with ``source=finzdata``, fetch Finzdata's own list of markets (one call)
``POST /api/data/fetch``    (a job) download for one tab: ``{"kind": "options" | "futures" | "crypto", "items": [...], ...settings}``
``POST /api/data/delete``   remove one download: ``{"kind", "name", "confirm": true}``

The browser names what it wants (a ticker, a futures root, a coin) and a few bounded settings; every address, file name and format is fixed here, so a request cannot reach another host or another file. A
download runs on its own worker, apart from the backtests, because a slow feed must not make a backtest wait nor the reverse. One download per tab at a time. The files and their records are kept by
``assets.free_data.store`` under ``data/user/downloads/``; the workspaces read them and label what they show ``downloaded``.

**iTick** is one more source for the three tabs (the key is the ``ITICK_API_KEY`` secret of the Backtest tab's iTick source, whose provider and call allowance are shared): the daily price history of any
futures product, a coin's daily history, and the daily prices of an option underlying. It has no option data and no per-contract futures prices; see ``data_itick``.

**Finzdata** is a futures source with a key of its own (the ``FINZDATA_API_KEY`` secret): every individual contract of 139 markets with open interest, which the free plan shows for the last 30 days
only. A download is merged into the file of the last one, so the history grows; see ``data_finzdata``.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..assets.free_data import cboe_futures as cb
from ..assets.free_data import crypto as fc
from ..assets.free_data import finzdata_catalog as fzc
from ..assets.free_data import futures as ff
from ..assets.free_data import itick_catalog as ic
from ..assets.free_data import options as fo
from ..assets.free_data import yahoo_futures as yf
from ..assets.free_data.http import FetchError, Http
from ..assets.free_data.sources import describe
from ..assets.free_data.store import KINDS, DownloadStore
from ..assets.options_analytics import DEFAULT_RATE
from . import data_finzdata as dfz
from . import data_itick as di
from .api import ApiError, Job, _number
from .finzdata import FinzdataError
from .payload import clean
from .providers import ITickError, RateLimitTimeout

MAX_ITEMS = 15
OPTION_PRESETS = ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "SPX")
WHAT = ("history", "venues", "funding")
SOURCE_IDS = {"options": ("cboe", "yahoo", "deribit", "itick"), "futures": ("yahoo", "finzdata", "cboe_cfe", "deribit", "itick"), "crypto": ("okx", "coinbase", "kraken", "bitstamp", "deribit", "itick")}
FUTURES_SOURCES = ("yahoo", "itick", "yahoo_series", "finzdata")   # the curve of each contract month, the price history of a product from iTick, or of any future Yahoo lists, or every contract of a Finzdata market
CRYPTO_SOURCES = ("exchanges", "itick")
PRICE_SOURCES = ("itick", "yahoo")                         # where an option underlying's daily prices can come from
BARS_PER_YEAR = {"future": 260, "crypto": 365, "stock": 252}
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


def _check_product(x: str) -> str:
    region, code = ic.parse_item(x)
    return f"{region}:{code}"


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
        self.finz = dfz.FinzdataData(self.root, self.store)             # a test sets ``app.data.finz.client`` to a client with a fake transport
        self.http: Http | None = None                      # a test (or an app that must go through its own proxy) can set a client with another transport
        self.yfinance = None                               # likewise the ``yfinance`` module the option-chain fallback uses
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._pool = ThreadPoolExecutor(max_workers=1)     # downloads wait for each other, never for a backtest

    def get_routes(self) -> dict:
        return {"/api/data/status": self.status, "/api/data/search": self.search, "/api/data/contracts": self.contracts}

    def post_routes(self) -> dict:
        return {"/api/data/fetch": self.fetch, "/api/data/delete": self.delete, "/api/data/lookup": self.lookup}

    # ------------------------------------------------------------------------------------------------------------ iTick
    def _itick(self):
        """The Backtest tab's iTick provider: one key, one rate limiter and one call allowance for the whole app."""
        source = self.app.builder.sources.get("itick")
        provider = getattr(source, "provider", None)
        if provider is None:
            raise ApiError("iTick is not part of this app")
        return provider

    def _itick_status(self) -> dict:
        """Whether iTick can be used and what its plan allows right now: only whether a key exists is ever reported, never the key."""
        try:
            d = self._itick().describe()
        except ApiError as exc:
            return {"configured": False, "problem": str(exc), "setup": "", "calls_per_minute": 0, "calls_available_now": 0, "page_size": 1000, "bars_per_year": BARS_PER_YEAR}
        keep = {k: d[k] for k in ("configured", "problem", "setup", "calls_per_minute", "calls_available_now", "page_size", "host") if k in d}
        return {**keep, "bars_per_year": BARS_PER_YEAR, "max_calls_per_run": getattr(self._itick(), "max_calls_per_run", 60)}

    def _need_itick(self) -> None:
        ok, why = self._itick().available()
        if not ok:
            raise ApiError(why)

    # ------------------------------------------------------------------------------------------------------------ status
    def _presets(self) -> dict:
        roots = list(ff.FREE_ROOTS.values())
        return {
            "options": {"symbols": list(OPTION_PRESETS), "crypto": list(fo.DERIBIT_CURRENCIES), "max_dte": [30, 90, 180, 365, 730], "defaults": {"max_dte": 365},
                        "prices": {"sources": [{"id": "itick", "text": "iTick (needs your key)"}, {"id": "yahoo", "text": "Yahoo Finance"}], "years": [1, 2, 3, 5, 10], "defaults": {"years": 3, "source": "yahoo"}}},
            "futures": {"roots": [{"id": r.root, "name": r.name, "exchange": r.exchange, "group": r.group} for r in roots + list(cb.CBOE_ROOTS.values())], "groups": list(dict.fromkeys(r.group for r in roots + list(cb.CBOE_ROOTS.values()))),
                        "crypto": [{"id": k, "name": v} for k, v in ff.CRYPTO_ROOTS.items()], "history": list(FUTURES_HISTORY), "max_contracts": ff.MAX_CONTRACTS, "deribit": list(DERIBIT_MODES),
                        "defaults": {"contracts": 10, "history": "1y", "deribit": "curve"},
                        "itick": {"regions": [{"id": k, "name": v} for k, v in ic.REGION_NAMES.items()], "groups": ic.groups(), "years": [1, 2, 3, 5, 10, 20], "products": len(ic.futures_products()), "defaults": {"years": 5}},
                        "yahoo": {"groups": yf.groups(self._yahoo_list()), "years": [1, 2, 3, 5, 10, 20], "products": len(yf.catalog(self._yahoo_list())), "defaults": {"years": 5}},
                        "finzdata": {"groups": fzc.groups(self.finz.live_roots()), "markets": len(fzc.markets(self.finz.live_roots())[0]), "max_calls_per_root": dfz.MAX_CALLS_PER_ROOT}},
            "crypto": {"coins": list(fc.DEFAULT_COINS), "what": [{"id": "history", "text": "Daily history: spot, perpetual and funding"}, {"id": "venues", "text": "Real quotes and order books on six venues, now"},
                                                                  {"id": "funding", "text": "OKX funding history, for the funding view"}], "defaults": {"days": 400, "what": list(WHAT)},
                       "itick": {"coins": len(ic.coins())}},
        }

    def status(self, q: dict | None = None) -> dict:
        """``GET /api/data/status``: for each tab, its downloads (newest first, each with its source, address, dates, row count and notes), the sources it can use and what it offers to download."""
        presets = self._presets()
        itick = self._itick_status()
        kinds = {k: {"downloads": self.store.list(k), "sources": describe(SOURCE_IDS[k]), "itick_status": itick, **presets[k]} for k in KINDS}
        finzdata = self.finz.status()
        kinds["futures"]["finzdata_status"] = finzdata
        return clean({"kinds": kinds, "running": sorted(self._running), "directory": "data/user/downloads/", "limits": {"items": MAX_ITEMS}, "itick": itick, "finzdata": finzdata,
                      "note": "Free public feeds: no account and no key, but delayed, unaudited and somebody else's service. A provider can refuse an address (common on shared cloud hosts) or change its format. "
                              "iTick and Finzdata are the exceptions: each needs your key (the ITICK_API_KEY and FINZDATA_API_KEY secrets) and counts every call against your plan."})

    def search(self, q: dict) -> dict:
        """``GET /api/data/search?kind=futures&q=nasdaq&region=US&group=Energy&limit=30``: products from iTick's bundled list (no call is made), best match first; ``kind=crypto`` searches its coins;
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
            region = str(q.get("region") or "").upper() or None
            if region is not None and region not in ic.REGIONS:
                raise ApiError(f"region must be one of {list(ic.REGIONS)}")
            group = str(q.get("group") or "") or None
            found = ic.search(text, region, group, limit)
            try:
                seen = self._itick().access("future")                              # what iTick returned or refused for this key lately: the list is iTick's, the plan is yours
            except ApiError:
                seen = {}                                                          # an app without the iTick source has no marks to show
            for row in found["results"]:
                if row["item"] in seen:
                    row["access"] = seen[row["item"]]
            return clean({"kind": kind, **found, "groups": ic.groups(region)})
        if kind == "crypto":
            return clean({"kind": kind, **ic.search_coins(text, limit)})
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
            raise ApiError("only the futures of Yahoo Finance can be looked up this way; iTick's list is searched with no call")
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
        if isinstance(exc, (FetchError, ValueError, ApiError, ITickError, RateLimitTimeout)):          # FinzdataError is a FetchError
            message = str(exc)
        else:
            traceback.print_exc()
            message = f"{type(exc).__name__}: {exc}"
        out = {"item": item, "what": what, "ok": False, "message": message}
        if isinstance(exc, RateLimitTimeout) or (isinstance(exc, (ITickError, FinzdataError)) and exc.fatal):
            out["fatal"] = True                                                   # nothing else asked of iTick or Finzdata in this run can work: the loop stops spending calls
        return out

    def _known(self, kind: str, name: str, source_id: str, columns: dict) -> pd.DataFrame | None:
        """An earlier iTick download of ``name`` as a frame indexed by date (so only the newest bars are fetched), or ``None``. A file from another source is never extended with iTick's bars."""
        record = self.store.meta(kind, name)
        if not record or record.get("source_id") != source_id or not str(record.get("file", "")).endswith(".csv"):
            return None
        try:
            frame = pd.read_csv(self.store.path(kind, name)).rename(columns=columns)
            frame["date"] = pd.to_datetime(frame["date"])
            return frame.set_index("date").sort_index()
        except (OSError, ValueError, KeyError):
            return None

    def _one_option(self, http: Http, item: str, s: dict) -> list[dict]:
        if "chain" not in s["what"]:
            return []
        try:
            if item.startswith("deribit:"):
                d = fo.fetch_deribit_chain(http, item.split(":", 1)[1], s["max_dte"])
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
            if s["prices_source"] == "itick":
                d = di.underlying_prices(self._itick(), item, s["years"], progress)
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

    def _one_series(self, item: str, s: dict, progress) -> list[dict]:
        """A futures product's daily price history from iTick; an earlier download of it is only topped up."""
        try:
            region, code = ic.parse_item(item)
            known = self._known("futures", di.series_name(region, code), "itick", {})
            d = di.futures_series(self._itick(), region, code, s["years"], known, progress)
            return [self._save(d, f"{len(d.frame):,} daily bars, {d.meta['first_date']} to {d.meta['as_of']} (iTick)", {"source": "itick", "items": [item], "years": s["years"]}, item)]
        except Exception as exc:                                              # noqa: BLE001
            return [self._fail(item, "series", exc)]

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
                    if s["source"] == "itick":
                        known = self._known("crypto", coin, "itick", {"time": "date", "spot": "close"})
                        d = di.coin_history(self._itick(), coin, s["days"], known, progress)
                        out.append(self._save(d, f"{d.meta['first_date']} to {d.meta['as_of']} ({d.meta['source']})", {"source": "itick", "items": [coin], "what": ["history"], "days": s["days"]}, coin))
                        continue
                    d = fc.fetch_history(http, coin, s["days"], progress)
                    out.append(self._save(d, f"{d.meta['first_date']} to {d.meta['as_of']} ({d.meta['source']})", {"items": [coin], "what": ["history"], "days": s["days"]}, coin))
                elif what == "venues":
                    d = fc.fetch_venues(http, coin, progress)
                    out.append(self._save(d, f"quotes and books on {len(d.payload['venues'])} venues" + (f", {len(d.payload['failed'])} unavailable" if d.payload["failed"] else ""), {"items": [coin], "what": ["venues"], "days": s["days"]}, coin))
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
                                                       "years": _number(body, "years", 3, 1, 10, integer=True)}
        if kind == "futures":
            source = _choice(body, "source", "yahoo", FUTURES_SOURCES)
            if source == "finzdata":
                return _items(body, dfz.check_root, limit=40), {"source": "finzdata"}
            if source == "itick":
                return _items(body, _check_product), {"source": "itick", "years": _number(body, "years", 5, 1, 20, integer=True)}
            if source == "yahoo_series":
                return _items(body, _check_yahoo_future), {"source": "yahoo_series", "years": _number(body, "years", 5, 1, 20, integer=True)}
            history = str(body.get("history") or "1y")
            if history not in FUTURES_HISTORY:
                raise ApiError(f"history must be one of {', '.join(FUTURES_HISTORY)}")
            raw = body.get("contracts")
            contracts = None if str(raw).lower() in ("0", "all") else _number(body, "contracts", 10, 2, ff.MAX_CONTRACTS, integer=True)        # None: every contract Yahoo lists
            return _items(body, _check_root), {"source": "yahoo", "contracts": contracts, "history": history, "deribit": _choice(body, "deribit", "curve", DERIBIT_MODES)}
        source = _choice(body, "source", "exchanges", CRYPTO_SOURCES)
        what = body.get("what") or (["history"] if source == "itick" else list(WHAT))
        if not isinstance(what, list) or not what or any(w not in WHAT for w in what):
            raise ApiError(f"what must be a list from {list(WHAT)}")
        if source == "itick" and what != ["history"]:
            raise ApiError("iTick has only the daily history of a coin; the order books and the funding come from the exchanges")
        return _items(body, fc.check_coin), {"source": source, "days": _number(body, "days", 400, fc.MIN_DAYS, 1500, integer=True), "what": [w for w in WHAT if w in what]}

    def _itick_calls(self, kind: str, items: list[str], s: dict) -> int:
        """About how many iTick calls a request takes (zero when it does not use iTick): one per page of the provider's page size, at least one per item."""
        page = max(int(self._itick_status().get("page_size") or 1000), 10)
        if kind == "futures" and s.get("source") == "itick":
            return sum(max(1, -(-int(s["years"] * BARS_PER_YEAR["future"]) // page)) for _ in items)
        if kind == "crypto" and s.get("source") == "itick":
            return sum(max(1, -(-int(s["days"]) // page)) for coin in items if coin not in ("BTC", "ETH"))
        if kind == "options" and "prices" in s["what"] and s["prices_source"] == "itick":
            return sum(max(1, -(-int(s["years"] * BARS_PER_YEAR["stock"]) // page)) for item in items if not item.startswith("deribit:"))
        return 0

    def _uses_itick(self, kind: str, s: dict) -> bool:
        return (kind in ("futures", "crypto") and s.get("source") == "itick") or (kind == "options" and "prices" in s["what"] and s["prices_source"] == "itick")

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
        if self._uses_itick(kind, settings):
            self._need_itick()
            calls, provider = self._itick_calls(kind, items, settings), self._itick()
            if calls > provider.max_calls_per_run:
                raise ApiError(f"this download needs about {calls} iTick calls (around {int(provider.limiter.estimate_seconds(calls) / 60) + 1} minutes at {provider.limiter.calls} calls a minute), more than the "
                               f"{provider.max_calls_per_run} allowed in one run: choose fewer products or fewer years")
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
                        results += self._one_option(http, item, settings) + self._one_prices(http, item, settings, progress)
                    elif kind == "futures":
                        if settings["source"] == "finzdata":
                            results += self._one_finzdata(item, settings, progress)
                        elif settings["source"] == "itick":
                            results += self._one_series(item, settings, progress)
                        elif settings["source"] == "yahoo_series":
                            results += self._one_yahoo_series(http, item, settings, progress)
                        else:
                            results += self._one_root(http, item, settings, progress)
                    else:
                        results += self._one_coin(http, item, settings, progress)
                    if results and results[-1].get("fatal"):                  # iTick refused the key or the plan: asking again for the rest would only spend calls
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
