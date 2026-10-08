"""Turning a list of tickers into a ``MarketBundle``.

The platform's 15 ETFs come from the cleaned, versioned dataset (with point-in-time macro series). Any other ticker is downloaded from the data source you choose (Yahoo Finance, or iTick
with an API key; see ``providers``) and cached under ``data/user`` for a day, separately for each source. A mixed universe takes the platform's prices for the platform's tickers and the
downloads for the rest, keeps weekdays on which at least 60% of the tickers traded, and never fills a missing price. One ticker is a valid universe; whether a strategy can use it is
for ``framework.requirements`` to say.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..framework.data import MarketBundle, bundle_from_prices, load_default_bundle
from .providers import ITickProvider

TICKER = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,11}$")
MAX_TICKERS, MIN_TICKERS, MIN_OBSERVATIONS, TTL_SECONDS = 40, 1, 300, 24 * 3600
CLASSES = ("equity", "rates", "fixed_income", "credit", "commodity", "real_estate", "crypto", "unknown")
FIELDS = ("close", "high", "low", "volume")


class UniverseError(ValueError):
    """The request cannot be turned into a usable universe; the message is shown to the user."""


def normalise(tickers) -> list[str]:
    """Upper-case, de-duplicated, validated tickers in the order given."""
    if isinstance(tickers, str):
        tickers = re.split(r"[\s,;]+", tickers)
    out: list[str] = []
    for raw in tickers or []:
        t = str(raw).strip().upper()
        if not t:
            continue
        if not TICKER.match(t):
            raise UniverseError(f"'{raw}' is not a valid ticker (letters, digits and . ^ = - only, up to 12 characters)")
        if t not in out:
            out.append(t)
    if len(out) > MAX_TICKERS:
        raise UniverseError(f"at most {MAX_TICKERS} tickers at a time")
    return out


def yahoo_fetch(tickers: list[str], start: str) -> dict[str, pd.DataFrame | str]:
    """Adjusted close, high, low and volume per ticker (or an error string), from Yahoo Finance."""
    import yfinance as yf

    try:
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=False)
    except Exception as error:                                   # network, rate limit, provider change
        return {t: f"download failed: {type(error).__name__}: {str(error)[:120]}" for t in tickers}
    out: dict[str, pd.DataFrame | str] = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                level = next(i for i, name in enumerate(raw.columns.names) if raw.columns.get_level_values(i).isin(tickers).any())
                frame = raw.xs(t, axis=1, level=level)
            else:
                frame = raw
            frame = frame.rename(columns=str.lower)[[c for c in ("close", "high", "low", "volume") if c in frame.rename(columns=str.lower).columns]]
            frame = frame.dropna(subset=["close"])
            out[t] = frame if len(frame) else "no data found: check the symbol"
        except (KeyError, StopIteration):
            out[t] = "no data found: check the symbol"
    return out


class TickerStore:
    """Downloaded prices on disk, one file per ticker, refreshed when older than a day. Each data source has its own store (its own folder), so two sources never mix.

    ``fetch(tickers, start)`` returns ``{ticker: DataFrame | error string}``. A fetcher that sets ``supports_context = True`` is called as ``fetch(tickers, start, known=..., progress=...)``
    instead, where ``known`` holds the frames already on disk (so it can ask only for the newest bars) and ``progress(text)`` reports what it is doing; it must return complete frames.
    """

    def __init__(self, directory: Path, fetch=yahoo_fetch, ttl: float = TTL_SECONDS, provider: str = "yahoo"):
        self.directory, self.fetch, self.ttl, self.provider = Path(directory), fetch, ttl, provider
        self.directory.mkdir(parents=True, exist_ok=True)
        self._io = threading.Lock()                              # one download at a time per source: a second caller finds the first one's files fresh

    def _path(self, ticker: str) -> Path:
        return self.directory / (re.sub(r"[^A-Z0-9]", lambda m: f"_{ord(m.group()):02x}", ticker) + ".csv")

    def _meta(self, ticker: str) -> dict | None:
        path = self._path(ticker).with_suffix(".json")
        if not (self._path(ticker).exists() and path.exists()):
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def _fresh(self, ticker: str, start: str) -> bool:
        info = self._meta(ticker)
        return bool(info) and time.time() - info["fetched"] < self.ttl and info["start"] <= start

    def _read(self, ticker: str) -> pd.DataFrame:
        return pd.read_csv(self._path(ticker), parse_dates=["date"]).set_index("date").sort_index()

    def notes(self, tickers: list[str]) -> dict[str, list[str]]:
        """Warnings the downloader recorded about each ticker's data (for example a suspected unadjusted split)."""
        return {t: list((self._meta(t) or {}).get("warnings") or []) for t in tickers if (self._meta(t) or {}).get("warnings")}

    def peek(self, tickers: list[str], start: str) -> tuple[dict[str, dict], list[str]]:
        """``(summaries of the tickers already on disk and fresh, the rest)`` without touching the network."""
        have, missing = {}, []
        for t in tickers:
            info = self._meta(t)
            if info and self._fresh(t, start):
                if "days" not in info:                           # a cache written before summaries were kept
                    frame = self._read(t)
                    info = {**info, "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()), "days": int(len(frame))}
                have[t] = info
            else:
                missing.append(t)
        return have, missing

    def _call(self, todo: list[str], start: str, known: dict, progress) -> dict:
        if getattr(self.fetch, "supports_context", False):
            return self.fetch(todo, start, known=known, progress=progress)
        return self.fetch(todo, start)

    def get(self, tickers: list[str], start: str, refresh: bool = False, progress=None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """``(frames, errors)``; only the stale or missing tickers go to the network."""
        with self._io:
            todo = [t for t in tickers if refresh or not self._fresh(t, start)]
            fetched: dict = {}
            if todo:
                known = {}
                if getattr(self.fetch, "supports_context", False):
                    for t in todo:
                        info = self._meta(t)
                        if info and info.get("start", "9999") <= start:      # the saved file reaches back as far as this request wants
                            try:
                                known[t] = self._read(t)
                            except (OSError, ValueError, KeyError):
                                pass
                fetched = self._call(todo, start, known, progress)
                for t, value in fetched.items():
                    if isinstance(value, pd.DataFrame):
                        value.to_csv(self._path(t), index_label="date")
                        self._path(t).with_suffix(".json").write_text(json.dumps({
                            "fetched": time.time(), "start": start, "provider": self.provider, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "first": str(value.index[0].date()), "last": str(value.index[-1].date()), "days": int(len(value)), "warnings": list(value.attrs.get("warnings") or [])}))
            frames, errors = {}, {}
            for t in tickers:
                value = fetched.get(t)
                if self._path(t).exists():                       # a file on disk is used even when refreshing it failed; the failure is kept as a note on the ticker
                    frames[t] = self._read(t)
                    if isinstance(value, str):
                        self._note_failed_refresh(t, value)
                else:
                    errors[t] = value if isinstance(value, str) else "no data found: check the symbol"
            return frames, errors

    def _note_failed_refresh(self, ticker: str, reason: str) -> None:
        path = self._path(ticker).with_suffix(".json")
        info = self._meta(ticker)
        if info is None:
            return
        kept = [w for w in info.get("warnings") or [] if not w.startswith("could not refresh")]
        saved = str(info.get("fetched_at", ""))[:10] or "an earlier day"
        info["warnings"] = kept + [f"could not refresh ({reason[:160]}); using the copy saved on {saved}"]
        try:
            path.write_text(json.dumps(info))
        except OSError:
            pass


@dataclass
class Source:
    """A place prices can be downloaded from: its name, the store that caches its downloads and, for a source with a key or a quota, the provider that knows how to use it."""

    name: str
    label: str
    store: TickerStore
    provider: object | None = None
    note: str = ""
    downloads_on_check: bool = True            # False for a rate-limited source: checking a ticker must not spend the quota (the run downloads it)
    default_years: int | None = None           # how far back to download when no start date is chosen (None: from 2005); a rate-limited source asks for less

    def status(self) -> dict:
        info = {"name": self.name, "label": self.label, "available": True, "problem": "", "note": self.note, "downloads_on_check": self.downloads_on_check, "default_years": self.default_years}
        describe = getattr(self.provider, "describe", None)
        if describe is not None:
            detail = describe()
            info.update(available=bool(detail.pop("configured")), problem=detail.pop("problem", ""), **detail)
        return info


def itick_source(root: Path, provider: ITickProvider | None = None, directory: Path | None = None) -> Source:
    """The iTick source: its own cache folder, a provider that reads the key from the environment, and no downloading while a ticker is merely being checked."""
    provider = provider or ITickProvider(root=root)
    store = TickerStore(directory or Path(root) / "data" / "user" / "prices_itick", fetch=provider, provider="itick")
    return Source("itick", "iTick", store, provider, downloads_on_check=False, default_years=10,
                  note="Daily prices as iTick delivers them (not documented as adjusted for splits or dividends). Needs the ITICK_API_KEY secret; every call counts against your plan's rate limit. "
                       "Without a start date it downloads the last ten years; choose an earlier date for more.")


class UniverseBuilder:
    def __init__(self, config, store: TickerStore, default_loader=load_default_bundle, sources: dict[str, Source] | None = None):
        self.config, self.store, self._load_default = config, store, default_loader
        self.sources: dict[str, Source] = {"yahoo": Source("yahoo", "Yahoo Finance", store, note="Prices adjusted for splits and dividends. No key needed.")}
        self.sources.update(sources or {})
        self._default: MarketBundle | None = None
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()

    def default_bundle(self) -> MarketBundle:
        with self._lock:
            if self._default is None:
                self._default = self._load_default(self.config)
            return self._default

    def default_tickers(self) -> list[str]:
        return list(self.default_bundle().assets)

    @staticmethod
    def start_for(src: Source, start, has_downloads: bool = True) -> str:
        """The first date to download: the one asked for, else (when something has to be downloaded) the source's default depth from the start of its year, else 2005. The start of the
        year, not today's date, so the default does not move every day."""
        if start:
            return str(pd.Timestamp(start).date())
        if has_downloads and src.default_years:
            return f"{pd.Timestamp.now().year - int(src.default_years)}-01-01"
        return "2005-01-01"

    def source(self, name: str | None) -> Source:
        key = str(name or "yahoo").strip().lower()
        if key not in self.sources:
            raise UniverseError(f"unknown data source '{name}'; choose from {sorted(self.sources)}")
        return self.sources[key]

    def waiting(self, tickers, start=None, source: str | None = None) -> list[str]:
        """The tickers a rate-limited source has not saved yet (always empty for a source that downloads freely): a quick check should not wait minutes for them."""
        tickers, src = normalise(tickers), self.source(source)
        if src.downloads_on_check:
            return []
        default = self.default_bundle()
        extra = [t for t in tickers if t not in default.prices.columns]
        return src.store.peek(extra, self.start_for(src, start, True))[1] if extra else []

    def preflight(self, tickers, start=None, source: str | None = None) -> Source:
        """Refuse early (with the reason) a run that could not even start: a source that is not set up while some ticker still has to be downloaded from it."""
        tickers, src = normalise(tickers), self.source(source)
        default = self.default_bundle()
        extra = [t for t in tickers if t not in default.prices.columns]
        status = src.status()
        if extra and not status["available"]:
            if src.store.peek(extra, self.start_for(src, start))[1]:
                raise UniverseError(f"{src.label} is selected but {status['problem'] or 'not available'}")
        return src

    def check(self, tickers, start=None, refresh: bool = False, source: str | None = None, progress=None) -> dict:
        """What the data look like for each ticker: coverage and first/last date, or why it failed. A rate-limited source is not asked to download here: a ticker it has not saved yet
        is checked only for spelling and for the key, and comes back ``pending`` (it is downloaded when you run)."""
        tickers, src = normalise(tickers), self.source(source)
        default = self.default_bundle()
        rows, extra = {}, [t for t in tickers if t not in default.prices.columns]
        start = self.start_for(src, start, bool(extra))
        for t in tickers:
            if t in default.prices.columns:
                s = default.prices[t].dropna()
                rows[t] = {"ok": True, "source": "platform", "first": str(s.index[0].date()), "last": str(s.index[-1].date()), "days": int(len(s)), "class": default.asset_class.get(t, "unknown")}
        if extra and not src.downloads_on_check:
            have, missing = src.store.peek(extra, start)
            for t, info in have.items():
                days = int(info["days"])
                rows[t] = {"ok": days >= MIN_OBSERVATIONS, "source": src.name, "first": info["first"], "last": info["last"], "days": days, "class": "unknown", "cached": True,
                           **({"warnings": info["warnings"]} if info.get("warnings") else {}),
                           **({} if days >= MIN_OBSERVATIONS else {"error": f"only {days} days of history; strategies need at least {MIN_OBSERVATIONS}"})}
            preflight = getattr(src.provider, "preflight", None)
            for t in missing:
                problem = preflight(t) if preflight else None
                if problem:
                    rows[t] = {"ok": False, "source": src.name, "error": problem}
                else:
                    calls = src.provider.calls_for(t, start) if hasattr(src.provider, "calls_for") else 1
                    rows[t] = {"ok": True, "pending": True, "source": src.name, "class": "unknown", "calls": calls,
                               "note": f"not downloaded yet: about {calls} {src.label} call{'s' if calls != 1 else ''} when you run"}
            return {"tickers": {t: rows[t] for t in tickers}, "ok": all(r["ok"] for r in rows.values())}
        frames, errors = src.store.get(extra, start, refresh, progress) if extra else ({}, {})
        notes = src.store.notes(list(frames))
        for t, frame in frames.items():
            days = int(len(frame))
            rows[t] = {"ok": days >= MIN_OBSERVATIONS, "source": src.name, "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()), "days": days, "class": "unknown",
                       **({"warnings": notes[t]} if t in notes else {}),
                       **({} if days >= MIN_OBSERVATIONS else {"error": f"only {days} days of history; strategies need at least {MIN_OBSERVATIONS}"})}
        for t, message in errors.items():
            rows[t] = {"ok": False, "error": message}
        return {"tickers": {t: rows[t] for t in tickers}, "ok": all(r["ok"] for r in rows.values())}

    def resolve(self, tickers, classes: dict | None = None, start: str | None = None, refresh: bool = False, source: str | None = None, progress=None) -> tuple[MarketBundle, dict]:
        """``(bundle, info)``. Raises ``UniverseError`` with a user-facing message when the universe cannot be built. ``progress(text)`` is told what a slow download is doing."""
        tickers, classes = normalise(tickers), {k.upper(): v for k, v in (classes or {}).items()}
        if len(tickers) < MIN_TICKERS:
            raise UniverseError("choose at least one ticker")
        bad = {t: c for t, c in classes.items() if c not in CLASSES}
        if bad:
            raise UniverseError(f"unknown asset class {sorted(set(bad.values()))}; choose from {list(CLASSES)}")
        src = self.source(source)
        default = self.default_bundle()
        extra = [t for t in tickers if t not in default.prices.columns]
        start = self.start_for(src, start, bool(extra))
        key = (tuple(tickers), tuple(sorted(classes.items())), start, src.name)
        with self._lock:
            if key in self._cache and not refresh:
                self._cache.move_to_end(key)
                return self._cache[key]
        frames, errors = src.store.get(extra, start, refresh, progress) if extra else ({}, {})      # no lock held: a rate-limited download can take minutes
        if errors:
            raise UniverseError("; ".join(f"{t}: {m}" for t, m in errors.items()))
        short = [f"{t} ({len(f)} days)" for t, f in frames.items() if len(f) < MIN_OBSERVATIONS]
        if short:
            raise UniverseError(f"too little history: {', '.join(short)}; at least {MIN_OBSERVATIONS} trading days are needed")
        if not extra and set(tickers) == set(default.assets) and start <= str(default.index[0].date()):
            bundle, label = default, "platform dataset"
        else:
            bundle = self._combine(default, tickers, frames, classes, start)
            label = src.label if extra and len(extra) == len(tickers) else (f"platform dataset + {src.label}" if extra else "platform dataset")
        info = {"tickers": list(bundle.assets), "classes": dict(bundle.asset_class), "first": str(bundle.index[0].date()), "last": str(bundle.index[-1].date()),
                "days": int(len(bundle.index)), "source": label, "has_macro": bool(not bundle.macro.empty and bundle.macro.notna().any().any()), "name": bundle.name}
        if extra:
            info["provider"] = src.name
            notes = src.store.notes(extra)
            if notes:
                info["data_notes"] = notes
        with self._lock:
            self._cache[key] = (bundle, info)
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
        return bundle, info

    def _combine(self, default: MarketBundle, tickers: list[str], frames: dict, classes: dict, start: str) -> MarketBundle:
        columns = {}
        for field in FIELDS:
            parts = {}
            for t in tickers:
                if t in frames and field in frames[t]:
                    parts[t] = frames[t][field]
                elif t in default.prices.columns:
                    source = {"close": default.prices, "high": default.high, "low": default.low, "volume": default.volume}[field]
                    if source is not None:
                        parts[t] = source[t]
            columns[field] = pd.DataFrame(parts).reindex(columns=[t for t in tickers if t in parts])
        prices = columns["close"].sort_index()
        prices = prices[(prices.index.dayofweek < 5) & (prices.index >= pd.Timestamp(start))]
        prices = prices[prices.notna().mean(axis=1) >= 0.6]
        if len(prices) < MIN_OBSERVATIONS:
            raise UniverseError(f"the tickers share only {len(prices)} trading days from {start}; pick tickers with overlapping history or an earlier start")
        side = {f: (columns[f].reindex(index=prices.index, columns=prices.columns) if len(columns[f].columns) == len(prices.columns) else None) for f in ("high", "low", "volume")}
        asset_class = {t: classes.get(t) or default.asset_class.get(t, "unknown") for t in prices.columns}
        macro = default.macro.reindex(prices.index).ffill(limit=5) if not default.macro.empty else None
        label = "+".join(prices.columns[:4]) + (f"+{len(prices.columns) - 4}" if len(prices.columns) > 4 else "")
        return bundle_from_prices(prices, volume=side["volume"], high=side["high"], low=side["low"], asset_class=asset_class, macro=macro, name=f"user:{label}")
