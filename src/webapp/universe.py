"""Turning a list of tickers into a ``MarketBundle``.

The platform's 15 ETFs come from the cleaned, versioned dataset (with point-in-time macro series). Any other ticker is downloaded from Yahoo Finance as
split- and dividend-adjusted prices and cached under ``data/user/prices`` for a day. A mixed universe takes the platform's prices for the platform's
tickers and the downloads for the rest, keeps weekdays on which at least 60% of the tickers traded, and never fills a missing price.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..framework.data import MarketBundle, bundle_from_prices, load_default_bundle

TICKER = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,11}$")
MAX_TICKERS, MIN_TICKERS, MIN_OBSERVATIONS, TTL_SECONDS = 40, 2, 300, 24 * 3600
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
    """Downloaded prices on disk, one file per ticker, refreshed when older than a day."""

    def __init__(self, directory: Path, fetch=yahoo_fetch, ttl: float = TTL_SECONDS):
        self.directory, self.fetch, self.ttl = Path(directory), fetch, ttl
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        return self.directory / (re.sub(r"[^A-Z0-9]", lambda m: f"_{ord(m.group()):02x}", ticker) + ".csv")

    def _fresh(self, ticker: str, start: str) -> bool:
        meta = self._path(ticker).with_suffix(".json")
        if not (self._path(ticker).exists() and meta.exists()):
            return False
        info = json.loads(meta.read_text())
        return time.time() - info["fetched"] < self.ttl and info["start"] <= start

    def get(self, tickers: list[str], start: str, refresh: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """``(frames, errors)``; only the stale or missing tickers go to the network."""
        todo = [t for t in tickers if refresh or not self._fresh(t, start)]
        if todo:
            fetched = self.fetch(todo, start)
            for t, value in fetched.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(self._path(t), index_label="date")
                    self._path(t).with_suffix(".json").write_text(json.dumps({"fetched": time.time(), "start": start, "provider": "yahoo",
                                                                              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
        frames, errors = {}, {}
        for t in tickers:
            if self._path(t).exists():
                frames[t] = pd.read_csv(self._path(t), parse_dates=["date"]).set_index("date").sort_index()
            else:
                value = (fetched if todo else {}).get(t)
                errors[t] = value if isinstance(value, str) else "no data found: check the symbol"
        return frames, errors


class UniverseBuilder:
    def __init__(self, config, store: TickerStore, default_loader=load_default_bundle):
        self.config, self.store, self._load_default = config, store, default_loader
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

    def check(self, tickers, start: str = "2005-01-01", refresh: bool = False) -> dict:
        """What the data look like for each ticker: coverage and first/last date, or why it failed."""
        tickers = normalise(tickers)
        default = self.default_bundle()
        rows, extra = {}, [t for t in tickers if t not in default.prices.columns]
        for t in tickers:
            if t in default.prices.columns:
                s = default.prices[t].dropna()
                rows[t] = {"ok": True, "source": "platform", "first": str(s.index[0].date()), "last": str(s.index[-1].date()), "days": int(len(s)), "class": default.asset_class.get(t, "unknown")}
        frames, errors = self.store.get(extra, start, refresh) if extra else ({}, {})
        for t, frame in frames.items():
            days = int(len(frame))
            rows[t] = {"ok": days >= MIN_OBSERVATIONS, "source": "yahoo", "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()), "days": days, "class": "unknown",
                       **({} if days >= MIN_OBSERVATIONS else {"error": f"only {days} days of history; strategies need at least {MIN_OBSERVATIONS}"})}
        for t, message in errors.items():
            rows[t] = {"ok": False, "error": message}
        return {"tickers": {t: rows[t] for t in tickers}, "ok": all(r["ok"] for r in rows.values())}

    def resolve(self, tickers, classes: dict | None = None, start: str | None = None, refresh: bool = False) -> tuple[MarketBundle, dict]:
        """``(bundle, info)``. Raises ``UniverseError`` with a user-facing message when the universe cannot be built."""
        tickers, classes = normalise(tickers), {k.upper(): v for k, v in (classes or {}).items()}
        if len(tickers) < MIN_TICKERS:
            raise UniverseError(f"choose at least {MIN_TICKERS} tickers: cross-sectional strategies compare assets with each other")
        bad = {t: c for t, c in classes.items() if c not in CLASSES}
        if bad:
            raise UniverseError(f"unknown asset class {sorted(set(bad.values()))}; choose from {list(CLASSES)}")
        start = str(pd.Timestamp(start).date()) if start else "2005-01-01"
        key = (tuple(tickers), tuple(sorted(classes.items())), start)
        with self._lock:
            if key in self._cache and not refresh:
                self._cache.move_to_end(key)
                return self._cache[key]
            default = self.default_bundle()
            extra = [t for t in tickers if t not in default.prices.columns]
            frames, errors = self.store.get(extra, start, refresh) if extra else ({}, {})
            if errors:
                raise UniverseError("; ".join(f"{t}: {m}" for t, m in errors.items()))
            short = [f"{t} ({len(f)} days)" for t, f in frames.items() if len(f) < MIN_OBSERVATIONS]
            if short:
                raise UniverseError(f"too little history: {', '.join(short)}; at least {MIN_OBSERVATIONS} trading days are needed")
            if not extra and set(tickers) == set(default.assets) and start <= str(default.index[0].date()):
                bundle, source = default, "platform dataset"
            else:
                bundle, source = self._combine(default, tickers, frames, classes, start), "platform dataset + Yahoo Finance" if extra and len(extra) < len(tickers) else (
                    "platform dataset" if not extra else "Yahoo Finance")
            info = {"tickers": list(bundle.assets), "classes": dict(bundle.asset_class), "first": str(bundle.index[0].date()), "last": str(bundle.index[-1].date()),
                    "days": int(len(bundle.index)), "source": source, "has_macro": bool(not bundle.macro.empty and bundle.macro.notna().any().any()), "name": bundle.name}
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
