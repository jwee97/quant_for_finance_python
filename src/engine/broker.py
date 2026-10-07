"""Paper trading and the broker interface: the same simulator, ledger and strategies, driven by a streaming feed instead of a finished data set.

A backtest hands the engine all its data at once; a paper (or live) session hands it data as it ARRIVES. :class:`PaperSession` wraps an :class:`~src.engine.engine.Engine` in streaming mode:
``feed`` appends the events that arrived (their ``available_at`` is the arrival time), ``advance(until)`` lets the engine process everything up to that time. The strategies, the cost
models, the margin and lifecycle handlers, the ledger and the reconciliation are exactly the backtest's, so a strategy that runs in research runs in paper trading unchanged, and a session fed
the same events a backtest saw produces the same replay digest (this is tested).

The feeds shipped here (:class:`ReplayFeed` for stored data in delivery order, :class:`PollingFeed` over a :class:`~src.marketdata.loaders.RestPollingAdapter`, :class:`StreamFeed` over any
:class:`~src.marketdata.loaders.StreamAdapter` such as the WebSocket adapter) all yield ``(until, events)`` batches; :func:`run_session` pumps them through a session.

:class:`Broker` is the interface a strategy host needs from an execution venue: submit and cancel orders, read positions, balances and fills. :class:`PaperBroker` implements it on the
simulated venue (market orders fill at the next tradeable event with spread, impact, slippage and fees, limit and stop orders rest). :class:`LiveBroker` is the abstract adapter for a real
broker or exchange: it defines what an implementation must provide and refuses to run without one, because a real adapter needs credentials and a venue's own sandbox that this repository
does not have; :func:`reconcile_positions` compares the internal ledger with a venue's reported positions, which is the check a live deployment must run every cycle.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, runtime_checkable

import pandas as pd

from ..marketdata.schema import normalise_events
from ..marketdata.store import PointInTimeStore
from .engine import Engine, EngineConfig
from .orders import Order


def empty_events() -> pd.DataFrame:
    return normalise_events(pd.DataFrame(columns=["timestamp", "instrument_id", "event_type"]), lag="0s")


# ------------------------------------------------------------------------------------------------------------------------------------- the broker
@runtime_checkable
class Broker(Protocol):
    def submit(self, instrument_id: str, quantity: float, **kwargs) -> Order: ...

    def cancel(self, order_id: int) -> None: ...

    def open_orders(self) -> list[Order]: ...

    def positions(self) -> dict[str, float]: ...

    def balances(self) -> dict[str, float]: ...

    def fills(self, since=None) -> pd.DataFrame: ...


class PaperBroker:
    """The simulated venue behind a :class:`PaperSession` (or any running engine). Orders submitted here go through the same validation, fill simulation, costs and ledger as strategy orders."""

    def __init__(self, engine: Engine, strategy: str = "manual"):
        self.engine, self.strategy = engine, strategy

    def submit(self, instrument_id: str, quantity: float, order_type: str = "market", limit_price: float | None = None, stop_price: float | None = None, tif: str = "GTC",
               tags: tuple = (), reduce_only: bool = False) -> Order:
        return self.engine.submit_order(Order(instrument_id, quantity, order_type, limit_price, stop_price, tif, self.strategy, tuple(tags), reduce_only))

    def cancel(self, order_id: int) -> None:
        self.engine.cancel_order(order_id)

    def open_orders(self) -> list[Order]:
        return [o for o in self.engine.orders.values() if o.is_open]

    def positions(self) -> dict[str, float]:
        return {i: p.quantity for i, p in self.engine.ledger.positions.items() if not p.is_flat}

    def balances(self) -> dict[str, float]:
        return {c: v for c, v in self.engine.ledger.cash.items() if abs(v) > 1e-12}

    def fills(self, since=None) -> pd.DataFrame:
        f = self.engine.ledger.fills_frame()
        return f if since is None or f.empty else f[f["ts"] > pd.Timestamp(since)].reset_index(drop=True)

    def equity(self) -> float:
        return self.engine.ledger.equity()


class LiveBroker:
    """The contract for a real broker or exchange adapter. Subclass it, implement the methods against the venue's API and pass the adapter wherever a :class:`Broker` is expected.

    It is deliberately not implemented: a real adapter needs the venue's credentials, its sandbox for testing and its own error handling (rejects, partial fills, reconnects), none of which
    can be exercised honestly offline. Instantiating the base class raises :class:`BrokerUnavailable`."""

    def __init__(self, *args, **kwargs):
        raise BrokerUnavailable(f"{type(self).__name__} is an interface: implement connect/submit/cancel/open_orders/positions/balances/fills against a real venue (credentials required)")

    def connect(self) -> None:
        raise NotImplementedError

    def submit(self, instrument_id: str, quantity: float, **kwargs) -> Order:
        raise NotImplementedError

    def cancel(self, order_id: int) -> None:
        raise NotImplementedError

    def open_orders(self) -> list[Order]:
        raise NotImplementedError

    def positions(self) -> dict[str, float]:
        raise NotImplementedError

    def balances(self) -> dict[str, float]:
        raise NotImplementedError

    def fills(self, since=None) -> pd.DataFrame:
        raise NotImplementedError


class BrokerUnavailable(RuntimeError):
    pass


def reconcile_positions(internal: dict[str, float], external: dict[str, float], tolerance: float = 1e-9) -> pd.DataFrame:
    """Differences between the internal ledger and a venue's reported positions: one row per instrument that disagrees (``internal``, ``external``, ``difference``). An empty frame means
    they agree; a live deployment halts trading on a non-empty one."""
    ids = sorted(set(internal) | set(external))
    rows = [{"instrument_id": i, "internal": internal.get(i, 0.0), "external": external.get(i, 0.0), "difference": internal.get(i, 0.0) - external.get(i, 0.0)} for i in ids
            if abs(internal.get(i, 0.0) - external.get(i, 0.0)) > tolerance]
    return pd.DataFrame(rows, columns=["instrument_id", "internal", "external", "difference"])


# ------------------------------------------------------------------------------------------------------------------------------------ the session
class PaperSession:
    """A streaming engine. ``history`` (events already known at ``config.start``) warms up the strategies; ``config.end`` bounds the session and ``config.session_days`` (or the default
    weekdays between start and end) says on which days to mark, accrue and check rolls, since a stream cannot reveal tomorrow's data days in advance."""

    def __init__(self, registry, strategies, config: EngineConfig, history: pd.DataFrame | None = None, **engine_kwargs):
        if config.start is None or config.end is None:
            raise ValueError("a PaperSession needs config.start and config.end")
        if config.session_days is None:
            config.session_days = pd.bdate_range(pd.Timestamp(config.start).normalize(), pd.Timestamp(config.end).normalize())
        store = PointInTimeStore(history if history is not None and len(history) else empty_events(), config.revision_policy, config.max_mark_age)
        self.engine = Engine(registry, store, strategies, config, **engine_kwargs)
        self.engine.begin()
        self.broker = PaperBroker(self.engine)
        self.fed = 0

    @property
    def now(self) -> pd.Timestamp:
        return self.engine.clock.now

    def feed(self, events: pd.DataFrame) -> int:
        n = self.engine.store.extend(events)
        self.fed += n
        return n

    def advance(self, until) -> None:
        self.engine.advance(until)

    def result(self):
        """Finish the session (final marks, ``on_end``) and return the result; the session cannot be advanced afterwards."""
        return self.engine.finish()


# --------------------------------------------------------------------------------------------------------------------------------------- feeds
class ReplayFeed:
    """Stored events in delivery order, in windows of ``step`` of availability time: yields ``(until, events)``. The windows are half-open, ``(previous until, until]``."""

    def __init__(self, events: pd.DataFrame, start, end, step: str = "1D", offset: str = "23:59:59"):
        self.events = events.sort_values("available_at", kind="stable")
        self.start, self.end, self.step, self.offset = pd.Timestamp(start), pd.Timestamp(end), pd.Timedelta(step), pd.Timedelta(offset)

    def __iter__(self) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
        prev = self.start
        until = self.start.normalize() + self.offset
        while prev < self.end:
            until = min(until, self.end)
            if until > prev:
                a = self.events["available_at"]
                yield until, self.events[(a > prev) & (a <= until)]
                prev = until
            until = until + self.step


class PollingFeed:
    """Poll a :class:`~src.marketdata.loaders.RestPollingAdapter` at the given times (a simulated or a wall clock): each poll's new events arrive at the poll time."""

    def __init__(self, adapter, times: Iterable):
        self.adapter, self.times = adapter, list(times)

    def __iter__(self):
        for t in self.times:
            yield pd.Timestamp(t), self.adapter.poll(t)


class StreamFeed:
    """Batch a :class:`~src.marketdata.loaders.StreamAdapter` (WebSocket, replay) into windows: every ``batch`` messages become one ``(until, events)`` step with ``until`` the last
    message's arrival time."""

    def __init__(self, stream, batch: int = 100):
        self.stream, self.batch = stream, batch

    def __iter__(self):
        rows = []
        for rec in self.stream:
            rows.append(rec)
            if len(rows) >= self.batch:
                yield self._flush(rows)
                rows = []
        if rows:
            yield self._flush(rows)

    @staticmethod
    def _flush(rows):
        df = normalise_events(pd.DataFrame(rows), lag="0s")
        return pd.Timestamp(df["available_at"].max()), df


def run_session(session: PaperSession, feed, on_step=None):
    """Pump a feed through a session: for each ``(until, events)`` add the events and advance to ``until``. Returns the session (call ``result()`` to finish)."""
    for until, events in feed:
        if len(events):
            session.feed(events)
        session.advance(until)
        if on_step is not None:
            on_step(session, until)
    return session


__all__ = ["Broker", "PaperBroker", "LiveBroker", "BrokerUnavailable", "reconcile_positions", "PaperSession", "ReplayFeed", "PollingFeed", "StreamFeed", "run_session", "empty_events"]
