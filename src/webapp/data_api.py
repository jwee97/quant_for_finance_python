"""The Free data endpoints of the Options, Futures and Crypto tabs: download real data from free public sources so that nothing has to be found and formatted by hand.

``GET  /api/data/status``   what is downloaded (with where from and when), what each tab can ask for, and which downloads are running
``POST /api/data/fetch``    (a job) download for one tab: ``{"kind": "options" | "futures" | "crypto", "items": [...], ...settings}``
``POST /api/data/delete``   remove one download: ``{"kind", "name", "confirm": true}``

The browser names what it wants (a ticker, a futures root, a coin) and a few bounded settings; every address, file name and format is fixed here, so a request cannot reach another host or another file. A
download runs on its own worker, apart from the backtests, because a slow feed must not make a backtest wait nor the reverse. One download per tab at a time. The files and their records are kept by
``assets.free_data.store`` under ``data/user/downloads/``; the workspaces read them and label what they show ``downloaded``.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..assets.free_data import crypto as fc
from ..assets.free_data import futures as ff
from ..assets.free_data import options as fo
from ..assets.free_data.http import FetchError, Http
from ..assets.free_data.sources import describe
from ..assets.free_data.store import KINDS, DownloadStore
from ..assets.options_analytics import DEFAULT_RATE
from .api import ApiError, Job, _number
from .payload import clean

MAX_ITEMS = 15
OPTION_PRESETS = ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "SPX")
WHAT = ("history", "venues", "funding")
SOURCE_IDS = {"options": ("cboe", "yahoo", "deribit"), "futures": ("yahoo", "deribit"), "crypto": ("okx", "coinbase", "kraken", "bitstamp", "deribit")}
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
        raise ValueError(f"free curves are available for {', '.join(list(ff.FREE_ROOTS) + list(ff.CRYPTO_ROOTS))}")
    return root


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
        return {"/api/data/status": self.status}

    def post_routes(self) -> dict:
        return {"/api/data/fetch": self.fetch, "/api/data/delete": self.delete}

    # ------------------------------------------------------------------------------------------------------------ status
    def _presets(self) -> dict:
        return {
            "options": {"symbols": list(OPTION_PRESETS), "crypto": list(fo.DERIBIT_CURRENCIES), "max_dte": [30, 90, 180, 365, 730], "defaults": {"max_dte": 365}},
            "futures": {"roots": [{"id": r.root, "name": r.name, "exchange": r.exchange} for r in ff.FREE_ROOTS.values()], "crypto": [{"id": k, "name": v} for k, v in ff.CRYPTO_ROOTS.items()],
                        "history": ["6mo", "1y", "2y", "5y"], "defaults": {"contracts": 10, "history": "1y"}},
            "crypto": {"coins": list(fc.DEFAULT_COINS), "what": [{"id": "history", "text": "Daily history: spot, perpetual and funding"}, {"id": "venues", "text": "Real quotes and order books on six venues, now"},
                                                                  {"id": "funding", "text": "OKX funding history, for the funding view"}], "defaults": {"days": 400, "what": list(WHAT)}},
        }

    def status(self, q: dict | None = None) -> dict:
        """``GET /api/data/status``: for each tab, its downloads (newest first, each with its source, address, dates, row count and notes), the sources it can use and what it offers to download."""
        presets = self._presets()
        kinds = {k: {"downloads": self.store.list(k), "sources": describe(SOURCE_IDS[k]), **presets[k]} for k in KINDS}
        return clean({"kinds": kinds, "running": sorted(self._running), "directory": "data/user/downloads/", "limits": {"items": MAX_ITEMS},
                      "note": "Free public feeds: no account and no key, but delayed, unaudited and somebody else's service. A provider can refuse an address (common on shared cloud hosts) or change its format."})

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
        if isinstance(exc, (FetchError, ValueError, ApiError)):
            message = str(exc)
        else:
            traceback.print_exc()
            message = f"{type(exc).__name__}: {exc}"
        return {"item": item, "what": what, "ok": False, "message": message}

    def _one_option(self, http: Http, item: str, s: dict) -> list[dict]:
        try:
            if item.startswith("deribit:"):
                d = fo.fetch_deribit_chain(http, item.split(":", 1)[1], s["max_dte"])
            else:
                d = fo.fetch_chain(http, item, s["rate"], s["max_dte"], self.yfinance)
            return [self._save(d, f"{d.meta['contracts']:,} contracts on {d.meta['expiries']} expiries, quotes of {d.meta['as_of']} ({d.meta['source']})", {"items": [item], "max_dte": s["max_dte"], "rate": s["rate"]})]
        except Exception as exc:                                              # noqa: BLE001 - one symbol failing must not lose the others
            return [self._fail(item, "chain", exc)]

    def _one_root(self, http: Http, item: str, s: dict, progress) -> list[dict]:
        try:
            d = ff.fetch_deribit_curve(http, item) if item in ff.CRYPTO_ROOTS else ff.fetch_yahoo_curve(http, item, s["contracts"], s["history"], progress=progress)
            path = self.store.path("futures", d.name)
            earlier = pd.read_csv(path) if path.exists() else None
            if earlier is not None:
                d.frame = ff.merge_history(earlier, d.frame)
                d.meta["first_date"] = str(d.frame["date"].min().date())
                d.meta["notes"] = list(d.meta["notes"]) + [f"Merged with the earlier download: the file now runs from {d.meta['first_date']} and holds {d.frame['contract'].nunique()} contracts."]
            held = d.frame["contract"].nunique()
            msg = f"{held} contracts, {d.frame['date'].nunique()} days to {d.meta['as_of']} ({d.meta['source']})" + (" · incomplete: the feed stopped answering" if d.meta.get("incomplete") else "")
            return [self._save(d, msg, {"items": [item], "contracts": s["contracts"], "history": s["history"]})]
        except Exception as exc:                                              # noqa: BLE001
            return [self._fail(item, "curve", exc)]

    def _one_coin(self, http: Http, coin: str, s: dict, progress) -> list[dict]:
        out = []
        for what in s["what"]:
            try:
                if what == "history":
                    if coin in ("BTC", "ETH"):
                        out.append({"item": coin, "what": "history", "ok": True, "skipped": True, "message": f"left alone: the platform already holds Deribit's {coin} history, which is longer than a download"})
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
            return _items(body, _check_option_item), {"max_dte": _number(body, "max_dte", 365, 7, fo.MAX_DTE, integer=True), "rate": _number(body, "rate", DEFAULT_RATE, -0.02, 0.2)}
        if kind == "futures":
            history = str(body.get("history") or "1y")
            if history not in ("6mo", "1y", "2y", "5y"):
                raise ApiError("history must be 6mo, 1y, 2y or 5y")
            return _items(body, _check_root), {"contracts": _number(body, "contracts", 10, 2, 16, integer=True), "history": history}
        what = body.get("what") or list(WHAT)
        if not isinstance(what, list) or not what or any(w not in WHAT for w in what):
            raise ApiError(f"what must be a list from {list(WHAT)}")
        return _items(body, fc.check_coin), {"days": _number(body, "days", 400, fc.MIN_DAYS, 1500, integer=True), "what": [w for w in WHAT if w in what]}

    def fetch(self, body: dict) -> dict:
        """``POST /api/data/fetch`` (a job): download for one tab. The job's result lists every item with whether it worked and why not; a download where nothing worked is an error whose message says why."""
        kind = str(body.get("kind") or "")
        if kind not in KINDS:
            raise ApiError(f"kind must be one of {list(KINDS)}")
        items, settings = self._settings(kind, body)
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
                        results += self._one_option(http, item, settings)
                    elif kind == "futures":
                        results += self._one_root(http, item, settings, progress)
                    else:
                        results += self._one_coin(http, item, settings, progress)
            finally:
                with self._lock:
                    self._running.discard(kind)
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
