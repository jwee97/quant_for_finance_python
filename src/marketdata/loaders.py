"""Loaders and writers for normalised events: CSV, Parquet, Arrow (IPC/Feather), JSON Lines, SQL, REST polling and streaming adapters.

Every loader returns the canonical frame (``schema.normalise_events``), optionally passed through a :class:`~src.marketdata.contracts.DataContract`. Structured payloads
(``curve_values``, ``reference_values``) travel as JSON text in the flat formats (CSV, Parquet, Arrow, SQL) and as nested objects in JSON Lines, and come back as dictionaries (curve
tenors as float keys). A ``mapping`` renames vendor columns to the standard ones (``{"Date": "timestamp", "Symbol": "instrument_id", "Close": "close"}``) so a vendor file needs no
preprocessing.

Live-style sources stamp ``available_at`` with the time of ARRIVAL, the only honest availability for data you did not have before it reached you:

* :class:`RestPollingAdapter` polls a callable (or an HTTP endpoint through :func:`http_fetcher`) and returns only events not seen before;
* :class:`WebSocketAdapter` collects messages from a socket for a bounded count or time;
* :class:`ReplayStream` replays stored events in delivery order, optionally paced by a simulated clock.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from .contracts import DataContract, apply_contract
from .events import COLUMNS
from .schema import normalise_events

PAYLOADS = ("curve_values", "reference_values")


def _loads(value, curve: bool):
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        if not value.strip():
            return None
        value = json.loads(value)
    if hasattr(value, "items") and not isinstance(value, dict):
        value = dict(value.items())
    if isinstance(value, dict) and curve:
        return {float(k): float(v) for k, v in value.items()}
    return value


def _dumps(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    return json.dumps({str(k): v for k, v in value.items()} if isinstance(value, dict) else value, sort_keys=True, default=float)


def decode_payloads(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PAYLOADS:
        if col in out.columns:
            out[col] = [_loads(v, col == "curve_values") for v in out[col]]
    return out


def encode_payloads(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PAYLOADS:
        if col in out.columns:
            out[col] = [_dumps(v) for v in out[col]]
    return out


def _finish(df: pd.DataFrame, mapping, lag, contract: DataContract | None, source: str | None) -> pd.DataFrame:
    if mapping:
        df = df.rename(columns=mapping)
    df = decode_payloads(df)
    if contract is not None:
        out, _ = apply_contract(df, contract)
        return out
    return normalise_events(df, lag=lag, source=source)


# ---------------------------------------------------------------------------------------------------------------------------------- files
def load_csv(path, mapping=None, lag=None, contract=None, source=None, **read_kwargs) -> pd.DataFrame:
    return _finish(pd.read_csv(path, **read_kwargs), mapping, lag, contract, source or Path(path).name)


def load_parquet(path, mapping=None, lag=None, contract=None, source=None) -> pd.DataFrame:
    return _finish(pd.read_parquet(path), mapping, lag, contract, source or Path(path).name)


def load_arrow(source_, mapping=None, lag=None, contract=None, source=None) -> pd.DataFrame:
    """From a ``pyarrow.Table``, or an Arrow IPC / Feather file path."""
    import pyarrow as pa
    import pyarrow.feather as feather

    if isinstance(source_, pa.Table):
        df = source_.to_pandas()
        name = source or "arrow"
    else:
        df = feather.read_table(str(source_)).to_pandas()
        name = source or Path(source_).name
    return _finish(df, mapping, lag, contract, name)


def load_jsonl(path, mapping=None, lag=None, contract=None, source=None) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return _finish(pd.DataFrame(rows), mapping, lag, contract, source or Path(path).name)


def load_sql(connection, query: str, params=None, mapping=None, lag=None, contract=None, source: str = "sql") -> pd.DataFrame:
    """From a DB-API connection, a SQLAlchemy engine/connection or a path to a SQLite file."""
    if isinstance(connection, (str, Path)):
        with sqlite3.connect(str(connection)) as con:
            df = pd.read_sql_query(query, con, params=params)
    else:
        df = pd.read_sql_query(query, connection, params=params)
    return _finish(df, mapping, lag, contract, source)


def load_events(path, fmt: str | None = None, **kwargs) -> pd.DataFrame:
    """Dispatch on the file extension: ``.csv``, ``.parquet``/``.pq``, ``.arrow``/``.feather``/``.ipc``, ``.jsonl``/``.ndjson``, ``.db``/``.sqlite`` (needs ``query=``)."""
    fmt = fmt or Path(path).suffix.lower().lstrip(".")
    table = {"csv": load_csv, "parquet": load_parquet, "pq": load_parquet, "arrow": load_arrow, "feather": load_arrow, "ipc": load_arrow, "jsonl": load_jsonl, "ndjson": load_jsonl,
             "db": load_sql, "sqlite": load_sql}
    if fmt not in table:
        raise ValueError(f"unsupported format '{fmt}'; use one of {sorted(table)}")
    return table[fmt](path, **kwargs)


def write_csv(df: pd.DataFrame, path) -> None:
    encode_payloads(df).to_csv(path, index=False)


def write_parquet(df: pd.DataFrame, path) -> None:
    encode_payloads(df).to_parquet(path, index=False)


def write_arrow(df: pd.DataFrame, path) -> None:
    import pyarrow as pa
    import pyarrow.feather as feather

    feather.write_feather(pa.Table.from_pandas(encode_payloads(df), preserve_index=False), str(path))


def write_jsonl(df: pd.DataFrame, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in df.to_dict("records"):
            row = {}
            for k, v in rec.items():
                if isinstance(v, pd.Timestamp):
                    v = v.isoformat()
                elif v is pd.NaT:
                    v = None
                elif isinstance(v, (float, np.floating)) and v != v:
                    v = None
                elif isinstance(v, dict):
                    v = {str(a): b for a, b in v.items()}
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                row[k] = v
            fh.write(json.dumps(row) + "\n")


def write_sql(df: pd.DataFrame, connection, table: str = "events", if_exists: str = "replace") -> None:
    if isinstance(connection, (str, Path)):
        with sqlite3.connect(str(connection)) as con:
            encode_payloads(df).to_sql(table, con, if_exists=if_exists, index=False)
    else:
        encode_payloads(df).to_sql(table, connection, if_exists=if_exists, index=False)


# -------------------------------------------------------------------------------------------------------------------------- live-style
def http_fetcher(url: str, params: dict | None = None, headers: dict | None = None, records_path: str | None = None, since_param: str | None = None, timeout: float = 10.0) -> Callable:
    """A ``fetch(cursor)`` callable for :class:`RestPollingAdapter` that GETs a JSON endpoint. ``records_path`` is a dotted path to the list of records in the response; ``since_param``
    names the query parameter that receives the cursor (the latest observation time already seen)."""
    import requests

    def fetch(cursor):
        q = dict(params or {})
        if since_param and cursor is not None:
            q[since_param] = pd.Timestamp(cursor).isoformat()
        r = requests.get(url, params=q, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        for part in (records_path.split(".") if records_path else []):
            data = data[part]
        return data
    return fetch


class RestPollingAdapter:
    """Poll a source for event records and keep only what is new. ``available_at`` is the poll time (arrival), regardless of any earlier vendor timestamp."""

    def __init__(self, fetch: Callable, parse: Callable[[dict], dict] | None = None, mapping: dict | None = None, source: str = "rest"):
        self.fetch, self.parse, self.mapping, self.source = fetch, parse, mapping, source
        self.cursor: pd.Timestamp | None = None
        self._seen: set = set()

    def poll(self, now) -> pd.DataFrame:
        now = pd.Timestamp(now)
        records = list(self.fetch(self.cursor))
        if self.parse is not None:
            records = [self.parse(r) for r in records]
        fresh = []
        for r in records:
            key = (r.get("instrument_id"), r.get("event_type"), str(r.get("timestamp")), r.get("revision", 0))
            if key not in self._seen:
                self._seen.add(key)
                fresh.append(r)
        if not fresh:
            return normalise_events(pd.DataFrame(columns=["timestamp", "instrument_id", "event_type"]), lag="0s")
        df = pd.DataFrame(fresh)
        if self.mapping:
            df = df.rename(columns=self.mapping)
        df["available_at"] = now
        out = normalise_events(decode_payloads(df), source=self.source)
        latest = out["timestamp"].max()
        self.cursor = latest if self.cursor is None else max(self.cursor, latest)
        return out

    def run(self, times: Iterable) -> pd.DataFrame:
        parts = [self.poll(t) for t in times]
        parts = [p for p in parts if len(p)]
        return normalise_events(pd.concat(parts, ignore_index=True), lag="0s") if parts else normalise_events(pd.DataFrame(columns=["timestamp", "instrument_id", "event_type"]), lag="0s")


class StreamAdapter:
    """Base class for streaming sources: iterate to receive event dictionaries as they arrive."""

    def __iter__(self) -> Iterator[dict]:
        raise NotImplementedError

    def collect(self, max_events: int | None = None) -> pd.DataFrame:
        rows = []
        for i, rec in enumerate(self):
            rows.append(rec)
            if max_events is not None and i + 1 >= max_events:
                break
        return normalise_events(pd.DataFrame(rows), lag="0s") if rows else normalise_events(pd.DataFrame(columns=["timestamp", "instrument_id", "event_type"]), lag="0s")


class ReplayStream(StreamAdapter):
    """Replay stored events in delivery order (``available_at``); ``speed`` is not wall-clock pacing, it only labels the replay: the engine's simulated clock does the pacing."""

    def __init__(self, events: pd.DataFrame, start=None, end=None):
        df = events
        if start is not None:
            df = df[df["available_at"] > pd.Timestamp(start)]
        if end is not None:
            df = df[df["available_at"] <= pd.Timestamp(end)]
        self.events = df

    def __iter__(self):
        for rec in self.events.to_dict("records"):
            yield rec


class WebSocketAdapter(StreamAdapter):
    """Collect messages from a WebSocket (``websockets`` package). ``subscribe`` is sent on connection, ``parse`` turns each text message into an event dict or ``None`` to skip it;
    ``available_at`` is the receipt time from ``clock`` (UTC now by default)."""

    def __init__(self, url: str, parse: Callable[[str], dict | None], subscribe: str | None = None, max_messages: int | None = None, timeout: float = 5.0, clock: Callable | None = None):
        self.url, self.parse, self.subscribe, self.max_messages, self.timeout = url, parse, subscribe, max_messages, timeout
        self.clock = clock or (lambda: pd.Timestamp.now("UTC").tz_localize(None))

    def __iter__(self):
        from websockets.sync.client import connect

        n = 0
        with connect(self.url, open_timeout=self.timeout) as ws:
            if self.subscribe is not None:
                ws.send(self.subscribe)
            while self.max_messages is None or n < self.max_messages:
                try:
                    message = ws.recv(timeout=self.timeout)
                except TimeoutError:
                    return
                received = self.clock()
                rec = self.parse(message)
                if rec is None:
                    continue
                rec = dict(rec)
                rec["available_at"] = received
                n += 1
                yield rec


__all__ = ["load_csv", "load_parquet", "load_arrow", "load_jsonl", "load_sql", "load_events", "write_csv", "write_parquet", "write_arrow", "write_jsonl", "write_sql", "http_fetcher",
           "RestPollingAdapter", "StreamAdapter", "ReplayStream", "WebSocketAdapter", "decode_payloads", "encode_payloads", "COLUMNS"]
