"""Engine events, their ordering guarantees, the simulation clock and the replay log.

**Ordering.** The engine is a discrete-event simulation. Every event has a time, a PRIORITY and a sequence number; events are processed in ``(time, priority, sequence)`` order, so
what happens at one instant is always the same and never depends on the order things were inserted by accident. At a given timestamp the order is:

=====  ================  ===============================================================================================================================
prio   kind              what happens
=====  ================  ===============================================================================================================================
10     ``market``        a data event is delivered (it is known from ``available_at`` on); working orders for that instrument are tested against it
20     ``corporate``     splits and dividends take effect
30     ``lifecycle``     expiries, exercise and assignment, perpetual funding, swap and bond cashflows, forward settlement
40     ``accrual``       interest on cash, borrow fees
50     ``mark``          positions are revalued, variation margin settles, equity is recorded
60     ``margin``        margin is checked; a breach can liquidate
65     ``roll``          roll rules are evaluated and rolls generated
70     ``schedule``      strategies decide (``on_schedule``)
80     ``arrival``       orders submitted with a latency reach the venue
90     ``close``         day orders expire, end-of-day housekeeping
=====  ================  ===============================================================================================================================

A decision at ``t`` therefore sees all data available at ``t`` and the marks and margin of ``t``, and its orders cannot be filled before ``t`` plus the latency (or the next data event).

**Clock.** :class:`SimulationClock` only moves forward; trying to move it backward raises. **Replay.** :class:`EventLog` keeps a running SHA-256 over every processed event, fill and
journal entry, so two runs with the same inputs and configuration can be compared by one hash (deterministic replay).
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
from dataclasses import dataclass, field

import pandas as pd

PRIORITY = {"market": 10, "corporate": 20, "lifecycle": 30, "accrual": 40, "mark": 50, "margin": 60, "roll": 65, "schedule": 70, "arrival": 80, "close": 90}


@dataclass(order=True)
class Event:
    ts_ns: int
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)

    @property
    def ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.ts_ns)


class EventQueue:
    """A heap of future events with the engine's deterministic ordering."""

    def __init__(self):
        self._heap: list[Event] = []
        self._counter = itertools.count()

    def push(self, ts, kind: str, payload: dict | None = None, priority: int | None = None) -> Event:
        ev = Event(pd.Timestamp(ts).value, PRIORITY[kind] if priority is None else priority, next(self._counter), kind, payload or {})
        heapq.heappush(self._heap, ev)
        return ev

    def peek(self) -> Event | None:
        return self._heap[0] if self._heap else None

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)


class SimulationClock:
    def __init__(self, start=None):
        self._now: pd.Timestamp | None = None if start is None else pd.Timestamp(start)

    @property
    def now(self) -> pd.Timestamp:
        if self._now is None:
            raise RuntimeError("the clock has not started")
        return self._now

    def advance(self, ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if self._now is not None and ts < self._now:
            raise RuntimeError(f"the clock cannot move backward: {self._now} -> {ts}")
        self._now = ts
        return ts


class EventLog:
    """A running hash of everything that happened, plus counts by kind (cheap enough to leave on)."""

    def __init__(self, keep_records: bool = False):
        self._h = hashlib.sha256()
        self.counts: dict[str, int] = {}
        self.records: list[tuple] | None = [] if keep_records else None

    def add(self, ts, kind: str, text: str = "") -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        line = f"{pd.Timestamp(ts).value}|{kind}|{text}"
        self._h.update(line.encode())
        if self.records is not None:
            self.records.append((pd.Timestamp(ts), kind, text))

    @property
    def digest(self) -> str:
        return self._h.hexdigest()
