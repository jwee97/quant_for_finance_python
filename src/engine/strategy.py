"""The common strategy API: one interface for every signal type and every instrument.

A strategy emits one of three things, at whichever level suits it:

1. **Signals** (:class:`Signal`): a view on an instrument (a forecast, a z-score, a carry) with a confidence. ``generate_signals`` produces them.
2. **Targets** (:class:`Target`): what the portfolio should hold: a quantity, a signed base-currency notional, or a fraction of equity (the strategy's own capital). ``map_to_targets``
   turns signals into targets; ``ctx.set_targets`` sends them to the engine, which applies the constraints, sizes in whole contracts, nets against the current position and trades the
   difference.
3. **Orders** (:class:`~src.engine.orders.Order`): explicit instructions through ``ctx.submit``.

The hooks (all optional) are called by the engine at the right moments, always with a :class:`StrategyContext` that exposes only what is known at the engine's current time:

``on_start`` / ``on_end``; ``on_market`` (for instruments in ``subscriptions``); ``on_instrument_event`` (a contract rolled, expired, was exercised or assigned, a perpetual paid
funding, a margin call, a corporate action); ``on_portfolio_update`` (a fill); ``on_risk_update`` (after each mark); ``on_schedule`` (the decision time defined by the strategy's
:class:`Schedule`; by default signals -> targets -> orders).

A target on a FUTURE CHAIN id (``"ES"``) is resolved to the contract the chain's roll rule currently designates, so a strategy can think in markets and leave contracts to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ..instruments.calendars import get_calendar
from .orders import Order


@dataclass(frozen=True)
class Signal:
    instrument_id: str
    value: float
    confidence: float = 1.0
    horizon: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Target:
    """The desired holding of one instrument (or future chain) for the strategy. Exactly one of ``quantity`` (contracts, signed), ``notional`` (signed, base currency) or ``weight``
    (signed fraction of the strategy's capital) must be set."""

    instrument_id: str
    quantity: float | None = None
    notional: float | None = None
    weight: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if sum(x is not None for x in (self.quantity, self.notional, self.weight)) != 1:
            raise ValueError("a Target needs exactly one of quantity, notional or weight")


@dataclass(frozen=True)
class InstrumentEvent:
    kind: str                       # roll, expiry, exercise, assignment, funding, margin_call, liquidation, corporate_action, cashflow, rejected
    instrument_id: str
    ts: pd.Timestamp
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Schedule:
    """When a strategy decides. ``freq``: ``daily`` (every business day of ``calendar``), ``weekly`` (on ``weekday``, Monday = 0, or the previous business day if a holiday),
    ``monthly`` (the last business day) or ``every`` (a fixed grid such as ``15min``); ``time`` is the local clock time of the decision (UTC in the engine)."""

    freq: str = "daily"
    time: str = "16:30"
    weekday: int = 4
    calendar: str = "WEEKDAY"
    every: str | None = None
    times: tuple = ()

    def timestamps(self, start, end) -> list[pd.Timestamp]:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if self.times:
            return [pd.Timestamp(t) for t in sorted(self.times) if start <= pd.Timestamp(t) <= end]
        h, m = (int(x) for x in self.time.split(":"))
        offset = pd.Timedelta(hours=h, minutes=m)
        if self.freq == "every":
            return list(pd.date_range(start, end, freq=self.every or "1h"))
        cal = get_calendar(self.calendar)
        days = cal.business_days(start.normalize() - pd.Timedelta(days=7), end.normalize())
        if self.freq == "daily":
            chosen = list(days)
        elif self.freq == "weekly":
            groups: dict = {}
            for d in days:
                iso = d.isocalendar()
                groups.setdefault((iso.year, iso.week), []).append(d)
            chosen = [[d for d in ds if d.weekday() <= self.weekday][-1] for ds in groups.values() if any(d.weekday() <= self.weekday for d in ds)]
        elif self.freq == "monthly":
            chosen = [d for d in days if cal.next_business_day(d, include_self=False).month != d.month]
        else:
            raise ValueError("freq must be daily, weekly, monthly or every")
        return [d + offset for d in chosen if start <= d + offset <= end]


class PortfolioView:
    """A read-only view of the account for strategies."""

    def __init__(self, engine, strategy: str):
        self._e, self._s = engine, strategy

    @property
    def equity(self) -> float:
        return self._e.ledger.equity()

    @property
    def capital(self) -> float:
        """The equity assigned to this strategy: its share of the account (``capital_share``)."""
        return self._e.ledger.equity() * self._e.capital_share(self._s)

    @property
    def cash(self) -> dict:
        return dict(self._e.ledger.cash)

    def position(self, instrument_id: str) -> float:
        """The strategy's OWN position in an instrument (what it has traded), in contracts."""
        return self._e.strategy_qty.get(self._s, {}).get(instrument_id, 0.0)

    def total_position(self, instrument_id: str) -> float:
        return self._e.ledger.quantity(instrument_id)

    def positions(self) -> dict[str, float]:
        return {k: v for k, v in self._e.strategy_qty.get(self._s, {}).items() if abs(v) > 1e-12}

    def exposures(self) -> pd.DataFrame:
        return self._e.ledger.exposures(self._e.current_marks)

    @property
    def gross(self) -> float:
        return self._e.ledger.gross_exposure(self._e.current_marks)

    @property
    def net(self) -> float:
        return self._e.ledger.net_exposure(self._e.current_marks)

    @property
    def margin(self):
        return self._e.last_margin

    @property
    def drawdown(self) -> float:
        return self._e.current_drawdown()


class StrategyContext:
    def __init__(self, engine, strategy: "Strategy"):
        self._e, self._strategy = engine, strategy
        self.name = strategy.name
        self.data = engine.data
        self.portfolio = PortfolioView(engine, strategy.name)
        self.costs = engine.costs
        self.registry = engine.registry
        self.store: dict = {}                       # a scratch space that survives between calls (state for the strategy)

    @property
    def ts(self) -> pd.Timestamp:
        return self._e.clock.now

    @property
    def risk(self):
        return self._e.last_risk

    def submit(self, order: Order) -> Order:
        order.strategy = order.strategy or self.name
        return self._e.submit_order(order)

    def order(self, instrument_id: str, quantity: float, **kwargs) -> Order:
        return self.submit(Order(instrument_id, quantity, **kwargs))

    def buy(self, instrument_id: str, quantity: float, **kwargs) -> Order:
        return self.order(instrument_id, abs(quantity), **kwargs)

    def sell(self, instrument_id: str, quantity: float, **kwargs) -> Order:
        return self.order(instrument_id, -abs(quantity), **kwargs)

    def cancel(self, order_id: int) -> None:
        self._e.cancel_order(order_id)

    def cancel_all(self, instrument_id: str | None = None) -> None:
        self._e.cancel_strategy_orders(self.name, instrument_id)

    def set_targets(self, targets: Iterable[Target] | dict, **kwargs):
        """Hand the desired holdings to the engine (see :class:`Target`). A dict maps instrument id to a Target or to a quantity."""
        if isinstance(targets, dict):
            targets = [t if isinstance(t, Target) else Target(k, quantity=float(t)) for k, t in targets.items()]
        return self._e.apply_targets(self.name, list(targets), **kwargs)

    def close_all(self, instrument_id: str | None = None) -> None:
        pos = self.portfolio.positions()
        for iid, q in pos.items():
            if instrument_id is None or iid == instrument_id:
                self.order(iid, -q, tags=("close",))

    def exercise(self, instrument_id: str, quantity: float | None = None) -> None:
        """Ask to exercise a long American option now (all of it if ``quantity`` is None)."""
        self._e.request_exercise(self.name, instrument_id, quantity)

    def register_instrument(self, instrument):
        """Add an OTC contract (a swap, a forward) to the registry so it can be traded."""
        return self._e.register_instrument(instrument)

    def log(self, message: str) -> None:
        self._e.log.add(self.ts, "strategy_log", f"{self.name}:{message}")


class Strategy:
    """Base class. Override what you need; everything has a harmless default."""

    name: str = "strategy"
    schedule: Schedule = Schedule()
    subscriptions: tuple = ()
    capital_share: float = 1.0                    # fraction of the account's equity this strategy manages

    def on_start(self, ctx: StrategyContext) -> None:
        pass

    def on_end(self, ctx: StrategyContext) -> None:
        pass

    def on_market(self, ctx: StrategyContext, obs) -> None:
        pass

    def on_instrument_event(self, ctx: StrategyContext, event: InstrumentEvent) -> None:
        pass

    def on_portfolio_update(self, ctx: StrategyContext, fill) -> None:
        pass

    def on_risk_update(self, ctx: StrategyContext, risk) -> None:
        pass

    def generate_signals(self, ctx: StrategyContext) -> list[Signal]:
        return []

    def map_to_targets(self, ctx: StrategyContext, signals: list[Signal]) -> list[Target]:
        """Default mapping: each signal's value is the target weight of the strategy's capital (clipped to +-1)."""
        return [Target(s.instrument_id, weight=float(np.clip(s.value * s.confidence, -1.0, 1.0))) for s in signals]

    def on_schedule(self, ctx: StrategyContext) -> None:
        signals = self.generate_signals(ctx)
        targets = self.map_to_targets(ctx, signals)
        if targets or signals:
            ctx.set_targets(targets)
