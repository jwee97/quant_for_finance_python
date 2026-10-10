"""A stand-in for iTick's ``/{kind}/kline`` endpoint, for tests (the real service needs an account and the network).

``FakeITick`` is a transport: the provider calls it as ``transport(url, headers, timeout) -> (status, headers, body)``. It serves deterministic daily bars per code (a seeded random walk),
honours ``limit`` and ``et`` the way the documentation describes, checks the ``token`` header, and has switches for the behaviours the client has to survive: a server that clamps ``limit``,
an unknown symbol, rate-limit and server errors, bars stamped at midnight UTC or at the exchange's open, an unadjusted split, and a bar for today.
"""

from __future__ import annotations

import json
import zlib
from urllib import parse

import numpy as np
import pandas as pd

KEY = "TESTKEY-0123456789abcdef"


class FakeClock:
    """A clock that only moves when something sleeps, so rate-limit waits cost no real time."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.t, self.slept = start, []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def new_york_today() -> pd.Timestamp:
    """The date in New York, which is what the provider calls today for US stocks (the local clock's date differs for several hours a day)."""
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


class FakeITick:
    def __init__(self, bars: int = 2600, last: str | None = None, key: str = KEY, cap: int | None = None, stamp: str = "midnight_utc", split: tuple | None = None,
                 include_today: bool = False, daily: bool = True):
        self.n, self.key, self.cap, self.stamp, self.split, self.include_today, self.daily = bars, key, cap, stamp, split, include_today, daily
        self.last = pd.Timestamp(last) if last else new_york_today() - pd.Timedelta(days=1)
        self.calls: list[dict] = []                     # the query parameters of every call
        self.tokens: list[str | None] = []              # the token header of every call
        self.urls: list[str] = []
        self.script: list[tuple[int, dict, bytes]] = []  # answers to give, in order, before behaving normally
        self.unknown_prefix = "NOPE"
        self.history_start: dict[str, pd.Timestamp] = {}  # per code: the first date that exists (a short history)
        self.restate: float | None = None               # multiply every close by this factor (iTick restating its history)

    # ------------------------------------------------------------------------------------------------------------------------- data
    def series(self, code: str) -> pd.DataFrame:
        days = pd.bdate_range(end=self.last, periods=self.n)
        if self.include_today:
            days = days.append(pd.DatetimeIndex([new_york_today()]))
        if code in self.history_start:
            days = days[days >= self.history_start[code]]
        rng = np.random.default_rng(zlib.crc32(code.encode()))
        close = 50.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, len(days)))
        if self.split is not None:                       # a ``(date, ratio)`` split that is NOT adjusted for: the close falls by ``ratio`` from that date on
            when, ratio = pd.Timestamp(self.split[0]), float(self.split[1])
            close = np.where(days >= when, close / ratio, close)
        if self.restate is not None:
            close = close * self.restate
        return pd.DataFrame({"c": close, "o": close * 0.999, "h": close * 1.01, "l": close * 0.99, "v": 1_000_000.0, "tu": close * 1e6}, index=days)

    def _stamp(self, day: pd.Timestamp) -> int:
        if self.stamp == "open_local":
            return int((day + pd.Timedelta(hours=13, minutes=30)).value // 10**6)       # 09:30 New York in winter, UTC
        return int(day.value // 10**6)

    # ------------------------------------------------------------------------------------------------------------------------ the call
    def __call__(self, url: str, headers: dict, timeout: float):
        query = dict(parse.parse_qsl(parse.urlparse(url).query))
        self.calls.append(query)
        self.tokens.append(headers.get("token"))
        self.urls.append(url)
        if self.script:
            return self.script.pop(0)
        if headers.get("token") != self.key:
            return 200, {}, json.dumps({"code": "E002", "msg": "auth failed", "data": None}).encode()
        code = query.get("code", "")
        if code.startswith(self.unknown_prefix):
            return 200, {}, json.dumps({"code": "E001", "msg": "not found this produce", "data": None}).encode()
        frame = self.series(code)
        end = int(query["et"]) if "et" in query else None
        rows = []
        for day, row in frame.iterrows():
            if not self.daily:                                                              # an account that returns hourly bars instead of daily ones
                for hour in range(6):
                    rows.append({"t": int((day + pd.Timedelta(hours=9 + hour)).value // 10**6), "o": row.o, "h": row.h, "l": row.l, "c": row.c, "v": row.v, "tu": row.tu})
            else:
                rows.append({"t": self._stamp(day), "o": row.o, "h": row.h, "l": row.l, "c": row.c, "v": row.v, "tu": row.tu})
        if end is not None:
            rows = [r for r in rows if r["t"] <= end]
        limit = int(query.get("limit", 100))
        if self.cap:
            limit = min(limit, self.cap)
        rows = rows[-limit:]
        return 200, {}, json.dumps({"code": 0, "msg": None, "data": rows}).encode()
