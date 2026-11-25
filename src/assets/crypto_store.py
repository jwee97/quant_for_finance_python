"""Historical crypto observations on disk: order-book snapshots and updates, funding settlements, open interest and dated-futures contracts, as compressed Parquet under ``data/user/crypto_history/``.

What is stored is what was RECEIVED. Every row carries the exchange's own event time (when it gave one), the local receipt time, the sequence identifiers the exchange supplies, whether the row is a snapshot or an
incremental update, and a quality flag. Nothing is reconstructed: a book whose updates skipped a sequence number is marked ``gap`` and stays invalid until the next snapshot, and history before the recorder was first
run does not exist. Layout::

    books/<VENUE>/<INSTRUMENT>/<YYYY-MM-DD>/part-<receipt ms>.parquet      bids_px, bids_qty, asks_px, asks_qty (best first, ``depth`` levels), sequence, prev_sequence, kind, quality, exchange_ts_ms, recv_ts_ms
    funding/<VENUE>/<INSTRUMENT>.parquet                                   funding_time_ms, rate, interval_hours, recv_ts_ms, source
    open_interest/<VENUE>/<KEY>.parquet                                    ts_ms, oi, unit, recv_ts_ms, source
    futures/<VENUE>/<COIN>.parquet                                         observation time, instrument, expiry, mark, index, open interest, contract metadata

The recorder here samples REST snapshots. The reconciler applies incremental updates for the exchanges whose update messages carry sequence identifiers (OKX ``seqId``/``prevSeqId``, Deribit ``change_id``/
``prev_change_id``, Coinbase ``sequence``); the logic is tested on constructed sequences, and no long-running websocket collector is shipped.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NAME = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
COMPRESSION = "zstd"


@dataclass
class StoreConfig:
    depth: int = 50                    # levels kept per side
    min_interval_s: float = 5.0        # at most one snapshot of an instrument per interval
    retention_days: int = 30           # book partitions older than this are deleted by ``prune``
    max_bytes: int = 2_000_000_000     # ``prune`` deletes the oldest book partitions until the books fit


def _now_ms() -> int:
    return int(time.time() * 1000)


def _check(*names: str) -> None:
    for n in names:
        if not NAME.match(str(n)):
            raise ValueError(f"invalid name '{n}'")


class CryptoStore:
    def __init__(self, root, config: StoreConfig | None = None):
        self.dir = Path(root) / "data" / "user" / "crypto_history"
        self.cfg = config or StoreConfig()
        self._last: dict[tuple, int] = {}

    # ------------------------------------------------------------------------------------------------------------ books
    def _book_dir(self, venue: str, instrument: str) -> Path:
        _check(venue, instrument)
        return self.dir / "books" / venue.upper() / instrument.upper()

    def add_book(self, venue: str, instrument: str, bids, asks, exchange_ts_ms=None, recv_ts_ms=None, sequence=None, prev_sequence=None, kind: str = "snapshot", source: str = "rest", force: bool = False) -> dict | None:
        """Keep one book row. ``bids``/``asks`` are ``[[price, quantity], ...]`` best first. Returns the row, or ``None`` when the sampling interval says to skip it. The quality flag is set from the book itself: ``empty`` (a
        side has no level), ``crossed`` (best bid at or above best ask) or ``ok``; a ``gap`` is set only by the reconciler."""
        if kind not in ("snapshot", "update"):
            raise ValueError("kind must be snapshot or update")
        recv = int(recv_ts_ms if recv_ts_ms is not None else _now_ms())
        key = (venue.upper(), instrument.upper())
        if not force and key in self._last and recv - self._last[key] < self.cfg.min_interval_s * 1000:
            return None
        b, a = [(float(p), float(q)) for p, q in bids[: self.cfg.depth]], [(float(p), float(q)) for p, q in asks[: self.cfg.depth]]
        quality = "empty" if not b or not a else "crossed" if b[0][0] >= a[0][0] else "ok"
        row = {"venue": key[0], "instrument": key[1], "kind": kind, "quality": quality, "exchange_ts_ms": None if exchange_ts_ms in (None, "") else int(exchange_ts_ms), "recv_ts_ms": recv,
               "sequence": None if sequence in (None, "") else int(sequence), "prev_sequence": None if prev_sequence in (None, "") else int(prev_sequence), "source": source,
               "bids_px": [p for p, _ in b], "bids_qty": [q for _, q in b], "asks_px": [p for p, _ in a], "asks_qty": [q for _, q in a]}
        day = pd.Timestamp(recv, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        d = self._book_dir(*key) / day
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(d / f"part-{recv}.parquet", compression=COMPRESSION, index=False)
        self._last[key] = recv
        return row

    def books(self, venue: str, instrument: str, start=None, end=None) -> pd.DataFrame:
        """Every stored book row of an instrument between ``start`` and ``end`` (UTC, by receipt time), oldest first."""
        d = self._book_dir(venue, instrument)
        files = sorted(d.glob("*/part-*.parquet")) if d.is_dir() else []
        if start is not None:
            lo = pd.Timestamp(start).tz_localize("UTC").strftime("%Y-%m-%d") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC").strftime("%Y-%m-%d")
            files = [f for f in files if f.parent.name >= lo]
        if not files:
            return pd.DataFrame()
        out = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("recv_ts_ms").reset_index(drop=True)
        if start is not None:
            out = out[out["recv_ts_ms"] >= _ms(start)]
        if end is not None:
            out = out[out["recv_ts_ms"] <= _ms(end)]
        return out.reset_index(drop=True)

    def book_asof(self, venue: str, instrument: str, ts, max_age_s: float = 5.0) -> dict | None:
        """The latest VALID book received at or before ``ts`` and no older than ``max_age_s`` before it, else ``None``: a stale or invalid book is never returned, and nothing received after ``ts`` can be seen."""
        t = _ms(ts)
        df = self.books(venue, instrument, end=t)
        df = df[(df["quality"] == "ok")] if len(df) else df
        if df.empty:
            return None
        r = df.iloc[-1]
        if t - int(r["recv_ts_ms"]) > max_age_s * 1000:
            return None
        return {**{k: r[k] for k in ("venue", "instrument", "kind", "exchange_ts_ms", "recv_ts_ms", "sequence")}, "bids": list(zip(r["bids_px"], r["bids_qty"])), "asks": list(zip(r["asks_px"], r["asks_qty"])), "age_s": (t - int(r["recv_ts_ms"])) / 1000.0}

    # ------------------------------------------------------------------------------------------------------------ derivatives observations
    def _append(self, path: Path, frame: pd.DataFrame, key_cols: list[str]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        old = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
        both = pd.concat([old, frame], ignore_index=True).drop_duplicates(key_cols, keep="last").sort_values(key_cols[0]).reset_index(drop=True)
        both.to_parquet(path, compression=COMPRESSION, index=False)
        return int(len(both) - len(old))

    def add_funding(self, venue: str, instrument: str, rows: list[dict], source: str = "rest") -> int:
        """``rows``: ``{funding_time_ms, rate, interval_hours}``. A settlement already stored is not duplicated. Returns the number of new rows."""
        _check(venue, instrument)
        recv = _now_ms()
        f = pd.DataFrame([{"funding_time_ms": int(r["funding_time_ms"]), "rate": float(r["rate"]), "interval_hours": float(r["interval_hours"]), "recv_ts_ms": recv, "source": source} for r in rows])
        return self._append(self.dir / "funding" / venue.upper() / f"{instrument.upper()}.parquet", f, ["funding_time_ms"]) if len(f) else 0

    def funding(self, venue: str, instrument: str) -> pd.DataFrame:
        p = self.dir / "funding" / venue.upper() / f"{instrument.upper()}.parquet"
        return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

    def add_open_interest(self, venue: str, key: str, rows: list[dict], unit: str, source: str = "rest") -> int:
        _check(venue, key)
        recv = _now_ms()
        f = pd.DataFrame([{"ts_ms": int(r["ts_ms"]), "oi": float(r["oi"]), "unit": unit, "recv_ts_ms": recv, "source": source} for r in rows])
        return self._append(self.dir / "open_interest" / venue.upper() / f"{key.upper()}.parquet", f, ["ts_ms"]) if len(f) else 0

    def open_interest(self, venue: str, key: str) -> pd.DataFrame:
        p = self.dir / "open_interest" / venue.upper() / f"{key.upper()}.parquet"
        return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

    def add_futures(self, venue: str, coin: str, contracts: list[dict], observed_ms: int | None = None, source: str = "rest") -> int:
        """One observation of the dated futures curve: ``instrument``, ``expiry``, ``mark_price``, ``underlying_price`` and optionally ``open_interest``, ``settlement_currency``, ``contract_size``, ``open_interest_unit``."""
        _check(venue, coin)
        obs = int(observed_ms or _now_ms())
        f = pd.DataFrame([{"obs_ts_ms": obs, "instrument": c["instrument"], "expiry": c["expiry"], "mark_price": float(c["mark_price"]), "index_price": float(c["underlying_price"]), "open_interest": c.get("open_interest"),
                           "open_interest_unit": c.get("open_interest_unit"), "settlement_currency": c.get("settlement_currency"), "contract_size": c.get("contract_size"), "recv_ts_ms": _now_ms(), "source": source} for c in contracts])
        return self._append(self.dir / "futures" / venue.upper() / f"{coin.upper()}.parquet", f, ["obs_ts_ms", "instrument"]) if len(f) else 0

    def futures(self, venue: str, coin: str) -> pd.DataFrame:
        p = self.dir / "futures" / venue.upper() / f"{coin.upper()}.parquet"
        return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

    # ------------------------------------------------------------------------------------------------------------ coverage and limits
    def coverage(self, venue: str, instrument: str) -> dict:
        """What is held of an instrument's books: the first and last receipt, the number of rows by quality, the gaps in time longer than twice the median spacing. History before the first row does not exist."""
        df = self.books(venue, instrument)
        if df.empty:
            return {"state": "empty", "reason": "no book of this instrument has been recorded; history before recording began does not exist"}
        t = df["recv_ts_ms"].to_numpy()
        step = np.diff(t)
        med = float(np.median(step)) if len(step) else 0.0
        gaps = [{"from": int(t[i]), "to": int(t[i + 1])} for i in np.flatnonzero(step > 2 * max(med, 1))][:50]
        return {"state": "ok", "rows": int(len(df)), "first_ms": int(t[0]), "last_ms": int(t[-1]), "by_quality": {k: int(v) for k, v in df["quality"].value_counts().items()}, "median_spacing_s": med / 1000.0, "time_gaps": gaps,
                "sequence_gaps": int((df["quality"] == "gap").sum())}

    def size_bytes(self) -> int:
        return int(sum(f.stat().st_size for f in (self.dir / "books").rglob("*.parquet"))) if (self.dir / "books").is_dir() else 0

    def prune(self, now=None) -> dict:
        """Delete book partitions older than the retention, then the oldest ones until the books fit ``max_bytes``. Funding, open-interest and futures files are not touched."""
        now = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
        cutoff = (now.tz_localize("UTC") if now.tzinfo is None else now) - pd.Timedelta(days=self.cfg.retention_days)
        removed = 0
        parts = sorted((p for p in (self.dir / "books").glob("*/*/*") if p.is_dir()), key=lambda p: p.name) if (self.dir / "books").is_dir() else []
        for p in parts[:]:
            if pd.Timestamp(p.name, tz="UTC") < cutoff.normalize():
                shutil.rmtree(p)
                parts.remove(p)
                removed += 1
        while parts and self.size_bytes() > self.cfg.max_bytes:
            shutil.rmtree(parts.pop(0))
            removed += 1
        return {"partitions_removed": removed, "bytes": self.size_bytes()}


def _ms(ts) -> int:
    if isinstance(ts, (int, float, np.integer, np.floating)) and not isinstance(ts, bool):
        return int(ts)                                                                       # a number is milliseconds since the epoch
    t = pd.Timestamp(ts)
    return int((t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).timestamp() * 1000)


# ------------------------------------------------------------------------------------------------------------------ incremental-update reconciliation
PROFILES = {"OKX": "prev", "DERIBIT": "prev", "COINBASE": "consecutive"}


class BookReconciler:
    """Keep a book from a snapshot and its incremental updates, for exchanges whose updates carry sequence identifiers. ``prev`` profiles (OKX, Deribit): an update's ``prev_sequence`` must equal the last applied
    sequence; ``consecutive`` (Coinbase): the sequence must be last + 1. Any other order marks the book ``gap`` and INVALID: updates are then ignored until the next snapshot. A missing update is never filled in.
    An update is ``{sequence, prev_sequence, bids: [[price, qty]], asks: [[price, qty]]}``; a quantity of 0 removes the level."""

    def __init__(self, venue: str, snapshot: dict):
        if venue.upper() not in PROFILES:
            raise ValueError(f"no reconciliation profile for {venue}; supported: {sorted(PROFILES)}")
        self.venue, self.profile = venue.upper(), PROFILES[venue.upper()]
        self.reset(snapshot)

    def reset(self, snapshot: dict) -> None:
        self.bids = {float(p): float(q) for p, q in snapshot["bids"]}
        self.asks = {float(p): float(q) for p, q in snapshot["asks"]}
        self.sequence = None if snapshot.get("sequence") in (None, "") else int(snapshot["sequence"])
        self.valid, self.reason, self.applied = self.sequence is not None, "" if self.sequence is not None else "the snapshot carries no sequence: updates cannot be checked", 0

    def apply(self, update: dict) -> bool:
        if not self.valid:
            return False
        seq = int(update["sequence"])
        ok = (update.get("prev_sequence") not in (None, "") and int(update["prev_sequence"]) == self.sequence) if self.profile == "prev" else seq == self.sequence + 1
        if not ok:
            self.valid, self.reason = False, f"sequence gap: expected {'prev ' if self.profile == 'prev' else 'sequence '}{self.sequence if self.profile == 'prev' else self.sequence + 1}, got {update.get('prev_sequence') if self.profile == 'prev' else seq}"
            return False
        for side, book in (("bids", self.bids), ("asks", self.asks)):
            for p, q in update.get(side, []):
                if float(q) == 0.0:
                    book.pop(float(p), None)
                else:
                    book[float(p)] = float(q)
        self.sequence, self.applied = seq, self.applied + 1
        if self.bids and self.asks and max(self.bids) >= min(self.asks):
            self.valid, self.reason = False, "crossed book after the update"
            return False
        return True

    def book(self) -> dict:
        return {"valid": self.valid, "reason": self.reason, "sequence": self.sequence, "bids": sorted(self.bids.items(), reverse=True), "asks": sorted(self.asks.items())}


# ------------------------------------------------------------------------------------------------------------------ the REST sampler
def record_once(http, store: CryptoStore, coin: str = "BTC", venues: tuple | None = None) -> dict:
    """One sampling pass: the books of every venue that answers (spot and perpetual), the latest funding settlements of OKX, daily open interest of OKX and Deribit's dated futures. A venue that fails is reported, not
    fatal. Everything is stored with its own event and receipt times."""
    from .free_data import crypto as fc

    report = {"books": {}, "failed": {}}
    for name, fetch in fc.VENUE_FETCHERS:
        if venues and name not in venues:
            continue
        try:
            v = fetch(http, coin)
            row = store.add_book(v["venue"], v["pair"], v["bids"], v["asks"], v.get("exchange_ts_ms"), None, v.get("sequence"), None, "snapshot", "rest")
            report["books"][v["venue"]] = "stored" if row else "skipped (sampling interval)"
        except Exception as exc:                                                    # noqa: BLE001 - one venue failing must not stop the pass
            report["failed"][name] = str(exc)[:160]
    try:
        d = http.get_json(f"{fc.OKX}/public/funding-rate-history", {"instId": f"{coin}-USDT-SWAP", "limit": 20})
        rows = [{"funding_time_ms": int(r["fundingTime"]), "rate": float(r["realizedRate"] or r["fundingRate"]), "interval_hours": 8.0} for r in (d.get("data") or [])]
        report["funding_new"] = store.add_funding("OKX", f"{coin}-USDT-SWAP", rows, "okx public/funding-rate-history")
    except Exception as exc:                                                        # noqa: BLE001
        report["failed"]["OKX funding"] = str(exc)[:160]
    try:
        oi = fc.okx_open_interest(http, coin, 400)
        report["open_interest_new"] = store.add_open_interest("OKX", coin, [{"ts_ms": int(t.timestamp() * 1000), "oi": v} for t, v in oi.items()], "usd", "okx rubik/open-interest-volume")
    except Exception as exc:                                                        # noqa: BLE001
        report["failed"]["OKX open interest"] = str(exc)[:160]
    try:
        f = fc.fetch_deribit_futures(http, coin)
        report["futures_new"] = store.add_futures("DERIBIT", coin, f.payload["contracts"], None, "deribit get_book_summary_by_currency")
    except Exception as exc:                                                        # noqa: BLE001
        report["failed"]["Deribit futures"] = str(exc)[:160]
    return report
