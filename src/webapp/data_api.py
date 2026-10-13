"""The Free data endpoints of the Options, Futures and Crypto tabs: download real data from free public sources so that nothing has to be found and formatted by hand.

``GET  /api/data/status``   what is downloaded (with where from and when), what each tab can ask for, and which downloads are running
``GET  /api/data/search``   the products of iTick's bundled list that match a search (futures) or the coins it quotes (crypto)
``POST /api/data/fetch``    (a job) download for one tab: ``{"kind": "options" | "futures" | "crypto", "items": [...], ...settings}``
``POST /api/data/delete``   remove one download: ``{"kind", "name", "confirm": true}``

The browser names what it wants (a ticker, a futures root, a coin) and a few bounded settings; every address, file name and format is fixed here, so a request cannot reach another host or another file. A
download runs on its own worker, apart from the backtests, because a slow feed must not make a backtest wait nor the reverse. One download per tab at a time. The files and their records are kept by
``assets.free_data.store`` under ``data/user/downloads/``; the workspaces read them and label what they show ``downloaded``.

**iTick** is one more source for the three tabs (the key is the ``ITICK_API_KEY`` secret of the Backtest tab's iTick source, whose provider and call allowance are shared): the daily price history of any
futures product, a coin's daily history, and the daily prices of an option underlying. It has no option data and no per-contract futures prices; see ``data_itick``.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..assets.free_data import crypto as fc
from ..assets.free_data import futures as ff
from ..assets.free_data import itick_catalog as ic
from ..assets.free_data import options as fo
from ..assets.free_data.http import FetchError, Http
from ..assets.free_data.sources import describe
from ..assets.free_data.store import KINDS, DownloadStore
from ..assets.options_analytics import DEFAULT_RATE
from . import data_itick as di
from .api import ApiError, Job, _number
from .payload import clean
from .providers import ITickError, RateLimitTimeout

MAX_ITEMS = 15
OPTION_PRESETS = ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "SPX")
WHAT = ("history", "venues", "funding")
SOURCE_IDS = {"options": ("cboe", "yahoo", "deribit", "itick"), "futures": ("yahoo", "deribit", "itick"), "crypto": ("okx", "coinbase", "kraken", "bitstamp", "deribit", "itick")}
FUTURES_SOURCES = ("yahoo", "itick")                       # the curve of each contract month, or the price history of a product
CRYPTO_SOURCES = ("exchanges", "itick")
PRICE_SOURCES = ("itick", "yahoo")                         # where an option underlying's daily prices can come from
BARS_PER_YEAR = {"future": 260, "crypto": 365, "stock": 252}
HOSTS = {"query1.finance.yahoo.com": 0.5, "www.okx.com": 0.12, "api.kraken.com": 0.3}


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
    if root not in ff.FREE_ROOTS and root not in ff.CRYPTO_ROOTS:
        listed = list(ff.FREE_ROOTS)
        raise ValueError(f"free curves are available for {len(listed)} roots ({', '.join(listed[:8])}, …) and {', '.join(ff.CRYPTO_ROOTS)}: pick one from the list, or choose iTick for any other futures product")
    return root


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
        self.http: Http | None = None                      # a test (or an app that must go through its own proxy) can set a client with another transport
        self.yfinance = None                               # likewise the ``yfinance`` module the option-chain fallback uses
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._pool = ThreadPoolExecutor(max_workers=1)     # downloads wait for each other, never for a backtest

    def get_routes(self) -> dict:
        return {"/api/data/status": self.status, "/api/data/search": self.search}

    def post_routes(self) -> dict:
        return {"/api/data/fetch": self.fetch, "/api/data/delete": self.delete}

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
            "futures": {"roots": [{"id": r.root, "name": r.name, "exchange": r.exchange, "group": r.group} for r in roots], "groups": list(dict.fromkeys(r.group for r in roots)),
                        "crypto": [{"id": k, "name": v} for k, v in ff.CRYPTO_ROOTS.items()], "history": ["6mo", "1y", "2y", "5y"], "defaults": {"contracts": 10, "history": "1y"},
                        "itick": {"regions": [{"id": k, "name": v} for k, v in ic.REGION_NAMES.items()], "groups": ic.groups(), "years": [1, 2, 3, 5, 10, 20], "products": len(ic.futures_products()), "defaults": {"years": 5}}},
            "crypto": {"coins": list(fc.DEFAULT_COINS), "what": [{"id": "history", "text": "Daily history: spot, perpetual and funding"}, {"id": "venues", "text": "Real quotes and order books on six venues, now"},
                                                                  {"id": "funding", "text": "OKX funding history, for the funding view"}], "defaults": {"days": 400, "what": list(WHAT)},
                       "itick": {"coins": len(ic.coins())}},
        }

    def status(self, q: dict | None = None) -> dict:
        """``GET /api/data/status``: for each tab, its downloads (newest first, each with its source, address, dates, row count and notes), the sources it can use and what it offers to download."""
        presets = self._presets()
        itick = self._itick_status()
        kinds = {k: {"downloads": self.store.list(k), "sources": describe(SOURCE_IDS[k]), "itick_status": itick, **presets[k]} for k in KINDS}
        return clean({"kinds": kinds, "running": sorted(self._running), "directory": "data/user/downloads/", "limits": {"items": MAX_ITEMS}, "itick": itick,
                      "note": "Free public feeds: no account and no key, but delayed, unaudited and somebody else's service. A provider can refuse an address (common on shared cloud hosts) or change its format. "
                              "iTick is the exception: it needs your key (the ITICK_API_KEY secret) and counts every call against your plan."})

    def search(self, q: dict) -> dict:
        """``GET /api/data/search?kind=futures&q=nasdaq&region=US&group=Energy&limit=30``: products from iTick's bundled list (no call is made), best match first; ``kind=crypto`` searches its coins."""
        kind = str(q.get("kind") or "futures")
        text = str(q.get("q") or "")[:60]
        limit = _number(q, "limit", 30, 1, 100, integer=True)
        if kind == "futures":
            region = str(q.get("region") or "").upper() or None
            if region is not None and region not in ic.REGIONS:
                raise ApiError(f"region must be one of {list(ic.REGIONS)}")
            group = str(q.get("group") or "") or None
            return clean({"kind": kind, **ic.search(text, region, group, limit), "groups": ic.groups(region)})
        if kind == "crypto":
            return clean({"kind": kind, **ic.search_coins(text, limit)})
        raise ApiError("kind must be futures or crypto")

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
        if isinstance(exc, (FetchError, ValueError, ApiError, ITickError, RateLimitTimeout)):
            message = str(exc)
        else:
            traceback.print_exc()
            message = f"{type(exc).__name__}: {exc}"
        out = {"item": item, "what": what, "ok": False, "message": message}
        if isinstance(exc, RateLimitTimeout) or (isinstance(exc, ITickError) and exc.fatal):
            out["fatal"] = True                                                   # nothing else asked of iTick in this run can work: the loop stops spending calls
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
            d = ff.fetch_deribit_curve(http, item) if item in ff.CRYPTO_ROOTS else ff.fetch_yahoo_curve(http, item, s["contracts"], s["history"], progress=progress)
            path = self.store.path("futures", d.name)
            earlier = pd.read_csv(path) if path.exists() and (self.store.meta("futures", d.name) or {}).get("item") == "curve" else None
            if earlier is not None:
                d.frame = ff.merge_history(earlier, d.frame)
                d.meta["first_date"] = str(d.frame["date"].min().date())
                d.meta["notes"] = list(d.meta["notes"]) + [f"Merged with the earlier download: the file now runs from {d.meta['first_date']} and holds {d.frame['contract'].nunique()} contracts."]
            held = d.frame["contract"].nunique()
            msg = f"{held} contracts, {d.frame['date'].nunique()} days to {d.meta['as_of']} ({d.meta['source']})" + (" · incomplete: the feed stopped answering" if d.meta.get("incomplete") else "")
            return [self._save(d, msg, {"items": [item], "contracts": s["contracts"], "history": s["history"]})]
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
            if source == "itick":
                return _items(body, _check_product), {"source": "itick", "years": _number(body, "years", 5, 1, 20, integer=True)}
            history = str(body.get("history") or "1y")
            if history not in ("6mo", "1y", "2y", "5y"):
                raise ApiError("history must be 6mo, 1y, 2y or 5y")
            return _items(body, _check_root), {"source": "yahoo", "contracts": _number(body, "contracts", 10, 2, 16, integer=True), "history": history}
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
                        results += self._one_series(item, settings, progress) if settings["source"] == "itick" else self._one_root(http, item, settings, progress)
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
            job = self.app._register(Job(f"free data: {kind} ({len(items)})"))
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
