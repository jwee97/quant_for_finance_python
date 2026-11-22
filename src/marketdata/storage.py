"""The normalised market-data store: one SQLite file, ``data/user/marketdata/marketdata.db``.

Every observation is keyed by **provider** as well as symbol, bar size and adjustment, so two providers' prices for the same instrument sit side by side and nothing one provider sends can overwrite
another's. Within one provider a re-download that differs (a revised bar, a split re-adjusting history) is applied and counted as ``revised`` in the dataset's version log; a bar that is the same is
left alone. Tables:

* ``bars``: the observations, with the time the provider's answer was fetched.
* ``series``: one row per dataset (provider, asset class, symbol, bar size, multiplier, adjusted): first and last timestamp, rows, current version, status, last synchronisation, endpoint, origin
  (``download`` or ``legacy_import``) and notes.
* ``versions``: every change to a dataset (rows added, rows revised, range, reason).
* ``actions``: splits and dividends as the provider sent them.
* ``quality``: the findings of the last data-quality run of each dataset.
* ``jobs``: download status (queued, running, done, failed, cancelled) so an interrupted download can be resumed from what is stored.
* ``migrations``: legacy files already imported (path and sha256), so an import runs once.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars(provider TEXT, asset_class TEXT, symbol TEXT, timespan TEXT, multiplier INTEGER, adjusted INTEGER, ts TEXT,
  open REAL, high REAL, low REAL, close REAL, volume REAL, vwap REAL, transactions REAL, settlement_price REAL, fetched_at TEXT,
  PRIMARY KEY(provider, asset_class, symbol, timespan, multiplier, adjusted, ts));
CREATE TABLE IF NOT EXISTS series(provider TEXT, asset_class TEXT, symbol TEXT, timespan TEXT, multiplier INTEGER, adjusted INTEGER, first_ts TEXT, last_ts TEXT, rows INTEGER,
  version INTEGER, status TEXT, last_sync TEXT, endpoint TEXT, origin TEXT, notes TEXT, covered_from TEXT, covered_to TEXT,
  PRIMARY KEY(provider, asset_class, symbol, timespan, multiplier, adjusted));
CREATE TABLE IF NOT EXISTS versions(provider TEXT, asset_class TEXT, symbol TEXT, timespan TEXT, multiplier INTEGER, adjusted INTEGER, version INTEGER, at TEXT, added INTEGER, revised INTEGER,
  first_ts TEXT, last_ts TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS actions(provider TEXT, symbol TEXT, kind TEXT, date TEXT, payload TEXT, fetched_at TEXT, PRIMARY KEY(provider, symbol, kind, date, payload));
CREATE TABLE IF NOT EXISTS quality(provider TEXT, asset_class TEXT, symbol TEXT, timespan TEXT, multiplier INTEGER, adjusted INTEGER, run_at TEXT, level TEXT, check_name TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, target TEXT, status TEXT, started TEXT, finished TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS migrations(source TEXT PRIMARY KEY, sha256 TEXT, imported_at TEXT, provider TEXT, symbol TEXT, rows INTEGER);
"""
KEY = ("provider", "asset_class", "symbol", "timespan", "multiplier", "adjusted")
VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "transactions", "settlement_price"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ts(index: pd.DatetimeIndex, daily: bool) -> list[str]:
    if daily:
        return [str(pd.Timestamp(t).date()) for t in index]
    return [pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if pd.Timestamp(t).tzinfo else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" for t in index]


def is_daily(timespan: str) -> bool:
    return timespan in ("day", "week", "month", "quarter", "year")


class MarketStore:
    def __init__(self, root: Path | str):
        self.dir = Path(root) / "data" / "user" / "marketdata"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "marketdata.db"
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    # ------------------------------------------------------------------------------------------------------------ bars
    @staticmethod
    def key(provider: str, asset_class: str, symbol: str, timespan: str = "day", multiplier: int = 1, adjusted: bool = True) -> tuple:
        return (provider, asset_class, symbol, timespan, int(multiplier), int(bool(adjusted)))

    def upsert_bars(self, key: tuple, frame: pd.DataFrame, endpoint: str = "", origin: str = "download", reason: str = "sync", notes: str = "", status: str = "ok") -> dict:
        """Store ``frame`` under ``key``. New timestamps are added; a timestamp that exists with other values is revised (counted); identical rows are left alone. Returns the counts and the new version."""
        provider, asset_class, symbol, timespan, mult, adjusted = key
        daily = is_daily(timespan)
        f = frame.copy()
        f = f[~f.index.duplicated(keep="last")].sort_index()
        for col in VALUE_COLUMNS:
            if col not in f:
                f[col] = np.nan
        stamps = _ts(f.index, daily)
        fetched = now()
        added = revised = 0
        with self._lock, self._conn() as c:
            existing = {}
            if stamps:
                lo, hi = stamps[0], stamps[-1]
                for row in c.execute("SELECT ts, open, high, low, close, volume, vwap, transactions, settlement_price FROM bars WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=? AND ts>=? AND ts<=?", (*key, lo, hi)):
                    existing[row[0]] = row[1:]
            ins, upd = [], []
            for ts, vals in zip(stamps, f[VALUE_COLUMNS].itertuples(index=False, name=None)):
                vals = tuple(None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else float(v) for v in vals)
                old = existing.get(ts)
                if old is None:
                    ins.append((*key, ts, *vals, fetched))
                    added += 1
                elif any((a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9 * max(1.0, abs(a))) for a, b in zip(old, vals)):
                    upd.append((*vals, fetched, *key, ts))
                    revised += 1
            if ins:
                c.executemany(f"INSERT OR IGNORE INTO bars VALUES({','.join('?' * 16)})", ins)
            if upd:
                c.executemany("UPDATE bars SET open=?, high=?, low=?, close=?, volume=?, vwap=?, transactions=?, settlement_price=?, fetched_at=? WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=? AND ts=?", upd)
            span = c.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM bars WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", key).fetchone()
            prev = c.execute("SELECT version FROM series WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", key).fetchone()
            version = (prev[0] if prev else 0) + (1 if (added or revised or not prev) else 0)
            old = c.execute("SELECT covered_from, covered_to FROM series WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", key).fetchone() or (None, None)
            c.execute("INSERT OR REPLACE INTO series VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (*key, span[0], span[1], span[2], version, status, fetched, endpoint, origin, notes, old[0] or span[0], old[1] or span[1]))
            if added or revised or not prev:
                c.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (*key, version, fetched, added, revised, stamps[0] if stamps else None, stamps[-1] if stamps else None, reason))
        return {"added": added, "revised": revised, "version": version, "rows": span[2], "first": span[0], "last": span[1]}

    def read_bars(self, key: tuple, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        provider, asset_class, symbol, timespan, mult, adjusted = key
        q, args = "SELECT ts, open, high, low, close, volume, vwap, transactions, settlement_price FROM bars WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", list(key)
        if start:
            q += " AND ts>=?"
            args.append(str(start))
        if end:
            q += " AND ts<=?"
            args.append(str(end) + ("T99" if is_daily(timespan) and len(str(end)) == 10 else ""))
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY ts", args).fetchall()
        df = pd.DataFrame(rows, columns=["ts"] + VALUE_COLUMNS)
        if not len(df):
            return pd.DataFrame(columns=VALUE_COLUMNS, index=pd.DatetimeIndex([], name="date"))
        idx = pd.to_datetime(df.pop("ts"), utc=not is_daily(timespan))
        df.index = pd.DatetimeIndex(idx, name="date" if is_daily(timespan) else "timestamp")
        return df

    def series_info(self, key: tuple) -> dict | None:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM series WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", key).fetchone()
        return dict(r) if r else None

    def list_series(self, provider: str | None = None, asset_class: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM series WHERE 1=1", []
        if provider:
            q += " AND provider=?"
            args.append(provider)
        if asset_class:
            q += " AND asset_class=?"
            args.append(asset_class)
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(q + " ORDER BY provider, asset_class, symbol", args)]

    def versions(self, key: tuple) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT version, at, added, revised, first_ts, last_ts, reason FROM versions WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=? ORDER BY version", key)]

    def mark_covered(self, key: tuple, lo: str, hi: str) -> None:
        """Record that the provider was asked, and answered, for ``lo`` to ``hi``: a request whose answer begins later (a listing date, a plan's history depth, a weekend) is not repeated."""
        with self._lock, self._conn() as c:
            r = c.execute("SELECT covered_from, covered_to FROM series WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", key).fetchone()
            if r is None:
                return
            c.execute("UPDATE series SET covered_from=?, covered_to=? WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?",
                      (min(filter(None, [r[0], lo])), max(filter(None, [r[1], hi])), *key))

    def set_status(self, key: tuple, status: str, notes: str | None = None) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE series SET status=?, notes=COALESCE(?, notes) WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", (status, notes, *key))

    # ------------------------------------------------------------------------------------------------------------ gaps
    def missing_ranges(self, key: tuple, start: str | None = None, min_run: int = 3) -> list[tuple[str, str, int]]:
        """Runs of at least ``min_run`` expected sessions with no bar inside the stored range (calendar days for crypto, business days otherwise). Single holidays are not reported as gaps. Intraday series are not
        checked."""
        provider, asset_class, symbol, timespan, mult, adjusted = key
        if timespan != "day":
            return []
        df = self.read_bars(key, start)
        if len(df) < 2:
            return []
        freq = "D" if asset_class == "crypto" else "B"
        expected = pd.date_range(df.index[0], df.index[-1], freq=freq)
        gone = expected.difference(df.index)
        if not len(gone):
            return []
        runs, run = [], [gone[0]]
        for d in gone[1:]:
            if (d - run[-1]) <= pd.Timedelta(days=3 if freq == "B" else 1) and (freq == "D" or len(pd.bdate_range(run[-1], d)) <= 2):
                run.append(d)
            else:
                runs.append(run)
                run = [d]
        runs.append(run)
        return [(str(r[0].date()), str(r[-1].date()), len(r)) for r in runs if len(r) >= (1 if freq == "D" else min_run)]

    # ------------------------------------------------------------------------------------------------------------ quality
    def quality(self, key: tuple, jump: float = 0.4) -> list[dict]:
        """Checks of the stored bars: non-positive prices, high below low, close outside the high-low range, negative volume, one-day moves beyond ``jump`` (a split or a bad print), and gaps. Findings are stored."""
        df = self.read_bars(key)
        out: list[dict] = []

        def add(level, name, detail):
            out.append({"level": level, "check": name, "detail": detail})
        if not len(df):
            add("error", "empty", "the dataset has no bars")
        else:
            px = df[["open", "high", "low", "close"]]
            bad = int((px <= 0).any(axis=1).sum())
            if bad:
                add("error", "non_positive_price", f"{bad} bars have a price at or below zero")
            hl = int((df["high"] < df["low"]).sum())
            if hl:
                add("error", "high_below_low", f"{hl} bars have a high below the low")
            oc = int(((df["close"] > df["high"] * (1 + 1e-9)) | (df["close"] < df["low"] * (1 - 1e-9))).sum())
            if oc:
                add("warning", "close_outside_range", f"{oc} bars close outside their high-low range")
            if (df["volume"].dropna() < 0).any():
                add("error", "negative_volume", "some bars have a negative volume")
            if len(df) > 1 and key[3] == "day":
                r = np.log(df["close"].where(df["close"] > 0)).diff().abs()
                big = r[r > np.log1p(jump)]
                if len(big):
                    add("warning", "large_move", f"{len(big)} one-day moves above {jump:.0%} (first {big.index[0].date()}): a split, a bad print or a real move; check the corporate actions")
            nan = int(df["close"].isna().sum())
            if nan:
                add("warning", "missing_close", f"{nan} bars have no close")
            for a, b, n in self.missing_ranges(key)[:5]:
                add("warning", "gap", f"{n} expected sessions missing between {a} and {b}")
        stamp = now()
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM quality WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", key)
            c.executemany("INSERT INTO quality VALUES(?,?,?,?,?,?,?,?,?,?)", [(*key, stamp, o["level"], o["check"], o["detail"]) for o in out])
        return out

    # ------------------------------------------------------------------------------------------------------------ corporate actions
    def save_actions(self, provider: str, symbol: str, frame: pd.DataFrame) -> int:
        n = 0
        stamp = now()
        with self._lock, self._conn() as c:
            for _, r in frame.iterrows():
                payload = json.dumps({k: (None if (isinstance(v, float) and not np.isfinite(v)) else (str(v.date()) if isinstance(v, pd.Timestamp) else v)) for k, v in r.items() if k not in ("kind", "date")}, sort_keys=True, default=str)
                cur = c.execute("INSERT OR IGNORE INTO actions VALUES(?,?,?,?,?,?)", (provider, symbol, r["kind"], str(pd.Timestamp(r["date"]).date()), payload, stamp))
                n += cur.rowcount
        return n

    def actions(self, provider: str, symbol: str) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT kind, date, payload, fetched_at FROM actions WHERE provider=? AND symbol=? ORDER BY date", (provider, symbol)).fetchall()
        recs = [{"kind": k, "date": pd.Timestamp(d), **json.loads(p), "fetched_at": f} for k, d, p, f in rows]
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------------------------------------------------ jobs and migrations
    def job_start(self, kind: str, target: str) -> int:
        with self._lock, self._conn() as c:
            return c.execute("INSERT INTO jobs(kind, target, status, started) VALUES(?,?,?,?)", (kind, target, "running", now())).lastrowid

    def job_end(self, job_id: int, status: str, detail: str = "") -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET status=?, finished=?, detail=? WHERE id=?", (status, now(), detail[:500], job_id))

    def jobs(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    def migrated(self, source: str, sha: str) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT sha256 FROM migrations WHERE source=?", (source,)).fetchone()
        return bool(r) and r[0] == sha

    def mark_migrated(self, source: str, sha: str, provider: str, symbol: str, rows: int) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO migrations VALUES(?,?,?,?,?,?)", (source, sha, now(), provider, symbol, rows))

    def migrations(self) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM migrations ORDER BY imported_at")]

    def integrity(self) -> dict:
        """SQLite's own check and a reconciliation of the ``series`` summary with the bars."""
        with self._conn() as c:
            ok = c.execute("PRAGMA integrity_check").fetchone()[0]
            bad = []
            for s in c.execute("SELECT provider, asset_class, symbol, timespan, multiplier, adjusted, rows FROM series"):
                n = c.execute("SELECT COUNT(*) FROM bars WHERE provider=? AND asset_class=? AND symbol=? AND timespan=? AND multiplier=? AND adjusted=?", s[:6]).fetchone()[0]
                if n != s[6]:
                    bad.append({"series": list(s[:6]), "summary_rows": s[6], "stored_rows": n})
        return {"sqlite": ok, "series_mismatches": bad, "ok": ok == "ok" and not bad}
