"""The event-driven engine: one simulation loop for every instrument, one ledger for every portfolio, one strategy interface for every signal.

``Engine.run`` replays market events in the order they became AVAILABLE, interleaved with the engine's own events (strategy decisions, order arrivals, daily marking, expiries, funding,
cashflows, rolls, margin checks) under the ordering guarantees in ``events``. Orders become fills through the execution simulator and the cost models, fills become journal entries in the
ledger, and every lifecycle event (a future expiring, an option exercised, a perpetual paying funding, a swap paying a coupon) is processed by a handler that posts its cash and
profit to the same ledger. Nothing in a strategy's reach can see data before it became available (``PITData``), and the whole run is deterministic: the same inputs and configuration give
the same ``EventLog`` digest.

Fill policy (``EngineConfig.fill_policy``): ``next_event`` (default) fills an order at the first tradeable data event for its instrument AFTER the decision, which for daily bars is
the next day's price (no trading at a price you used to decide); ``latency`` fills after a fixed delay at the market as then known. OTC instruments priced by models (swaps, forwards)
have no events of their own and use ``otc_latency``.

The independent quantity book (``qty_book``) is updated from fills and lifecycle events separately from the ledger and compared with it at the end (position conservation).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ..instruments.registry import InstrumentRegistry
from ..ledger import CurrencyConverter, Fill, Ledger, MarginModel
from ..ledger.reconcile import reconcile
from ..marketdata.store import PointInTimeStore
from .constraints import ConstraintInputs, ConstraintSet
from .costs import CostSchedule, FinancingModel, LiquidityModel, realised_sigma
from .data import PITData
from .events import EventLog, EventQueue, SimulationClock
from .execution import Quote, simulate_fill
from .marks import MarkProvider
from .orders import Order, validate_order
from .strategy import InstrumentEvent, Strategy, StrategyContext, Target

TRADEABLE = ("quote", "trade", "bar", "settlement")


def _depth(row):
    """The displayed depth of a quote event (``reference_values = {"bids": [[p, size], ...], "asks": [...]}``) as ``(bids, asks)`` tuples, or None."""
    rv = row.reference_values
    if isinstance(rv, dict) and rv.get("bids") is not None and rv.get("asks") is not None:
        return tuple((float(p), float(q)) for p, q in rv["bids"]), tuple((float(p), float(q)) for p, q in rv["asks"])
    return None


@dataclass
class EngineConfig:
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    base_currency: str = "USD"
    initial_cash: dict = field(default_factory=lambda: {"USD": 1_000_000.0})
    pegs: dict = field(default_factory=dict)                  # currency -> rate against the base for currencies with no observed pair (e.g. {"USDT": 1.0})
    fill_policy: str = "next_event"                           # next_event | latency
    latency: str = "0s"
    otc_latency: str = "1D"
    snapshot_time: str = "23:59"
    roll_time: str = "16:20"
    max_mark_age: str | None = "10D"
    revision_policy: str = "latest_known"
    accrue: bool = True
    margin_check: bool = True
    liquidate: bool = True
    liquidation_penalty_bps: float = 50.0
    liquidation_fraction: float = 0.5
    liquidation_buffer: float = 0.05
    isolated_leverage: float = 5.0
    managed_chains: tuple | None = None                       # chains whose positions the engine rolls (None = all); () disables automatic rolling
    physical_expiry: str = "cash_settle"                      # cash_settle | error: what to do with a physically settled future still held at expiry
    early_assignment: bool = False
    assignment_extrinsic: float = 0.0                         # a short American option is assigned when its time value is below this (per unit) and it is in the money
    min_trade_fraction: float = 0.0                           # ignore target changes smaller than this fraction of capital
    keep_event_records: bool = False
    tolerance: float = 1e-6
    risk_every_snapshot: bool = False
    curve_ids: dict = field(default_factory=dict)             # currency -> discount curve id; overrides the "{ccy}-DISCOUNT" convention
    session_days: object = None                               # days on which to mark, accrue and check rolls (default: the days that have data; a streaming session needs it explicit)


class Engine:
    def __init__(self, registry: InstrumentRegistry, events: pd.DataFrame | PointInTimeStore, strategies: Iterable[Strategy], config: EngineConfig | None = None,
                 costs: CostSchedule | None = None, financing: FinancingModel | None = None, margin: MarginModel | None = None, liquidity: LiquidityModel | None = None,
                 constraints: ConstraintSet | dict | None = None, risk_model=None):
        self.config = cfg = config or EngineConfig()
        self.registry = registry
        self.store = events if isinstance(events, PointInTimeStore) else PointInTimeStore(events, cfg.revision_policy, cfg.max_mark_age)
        self.strategies = list(strategies)
        if len({s.name for s in self.strategies}) != len(self.strategies):
            raise ValueError("strategy names must be unique")
        self.costs = costs or CostSchedule()
        self.financing = financing or FinancingModel()
        self.margin_model = margin or MarginModel()
        self.liquidity = liquidity or LiquidityModel()
        self.constraints = constraints if isinstance(constraints, dict) or constraints is None else {"*": constraints}
        self.risk_model = risk_model
        lo, hi = self.store.span if len(self.store.events) else (pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))
        self.start = pd.Timestamp(cfg.start) if cfg.start is not None else lo
        self.end = pd.Timestamp(cfg.end) if cfg.end is not None else hi
        self.clock = SimulationClock()
        self.queue = EventQueue()
        self.log = EventLog(cfg.keep_event_records)
        self.data = PITData(self.store, registry, lambda: self.clock.now, cfg.max_mark_age)
        self.marks = MarkProvider(self.store, registry, cfg.max_mark_age, cfg.curve_ids)
        self.converter = CurrencyConverter(cfg.base_currency, cfg.pegs)
        self.ledger = Ledger(registry, cfg.base_currency, None, self.converter, cfg.tolerance)
        self.contexts = {s.name: StrategyContext(self, s) for s in self.strategies}
        self.orders: dict[int, Order] = {}
        self._order_seq = 0
        self.working: dict[str, list[Order]] = {}
        self.pending_next: dict[str, list[Order]] = {}
        self.strategy_qty: dict[str, dict[str, float]] = {s.name: {} for s in self.strategies}
        self.qty_book: dict[str, float] = {}
        self.current_marks: dict[str, float] = {}
        self.mark_ages: dict[str, pd.Timedelta] = {}
        self.equity_rows: list[dict] = []
        self.diagnostics: Counter = Counter()
        self.instrument_events: list[InstrumentEvent] = []
        self.last_margin = None
        self.last_risk = None
        self.risk_rows: list[dict] = []
        self.peak_equity = 0.0
        self.last_accrual: pd.Timestamp | None = None
        self.chain_current: dict[str, str] = {}
        self.exercise_requests: list[tuple] = []
        self.started = False
        from . import lifecycle
        from .. import swaps  # noqa: F401  registers the curve-based mark and cashflow models
        self.lifecycle = lifecycle

    # ------------------------------------------------------------------------------------------------------------------------------ helpers
    def capital_share(self, strategy: str) -> float:
        for s in self.strategies:
            if s.name == strategy:
                return s.capital_share
        return 1.0

    def current_drawdown(self) -> float:
        eq = self.ledger.equity() if self.started else 0.0
        return max(0.0, 1.0 - eq / self.peak_equity) if self.peak_equity > 0 else 0.0

    def register_instrument(self, instrument):
        self.registry.add(instrument)
        if self.started:
            self.lifecycle.schedule_instrument(self, instrument, self.clock.now, self.end)
        return instrument

    def _has_events(self, iid: str) -> bool:
        return any(self.store.has(iid, et) for et in TRADEABLE)

    def notify(self, ev: InstrumentEvent, strategy: str | None = None) -> None:
        self.instrument_events.append(ev)
        self.log.add(ev.ts, f"ievent:{ev.kind}", ev.instrument_id)
        for s in self.strategies:
            if strategy is None or s.name == strategy:
                s.on_instrument_event(self.contexts[s.name], ev)

    def unit_notional_base(self, inst, price: float) -> float:
        """Base-currency notional of ONE contract at ``price``."""
        return inst.settlement_notional(price, 1.0) * self.ledger.rate(inst.currency)

    # ------------------------------------------------------------------------------------------------------------------------------ orders
    def submit_order(self, order: Order) -> Order:
        ts = self.clock.now
        self._order_seq += 1
        order.id, order.submitted_at = self._order_seq, ts
        self.orders[order.id] = order
        if not self.registry.has_instrument(order.instrument_id):
            order.status, order.reason = "rejected", f"unknown instrument '{order.instrument_id}'"
        else:
            inst = self.registry.get(order.instrument_id)
            pos = self.strategy_qty.get(order.strategy, {}).get(order.instrument_id, 0.0) if order.strategy in self.strategy_qty else self.ledger.quantity(order.instrument_id)
            reason = validate_order(order, inst, pos, ts)
            if reason:
                order.status, order.reason = "rejected", reason
        self.log.add(ts, "order", f"{order.id}|{order.instrument_id}|{order.quantity:.10g}|{order.type}|{order.status}")
        if order.status == "rejected":
            self.diagnostics["orders_rejected"] += 1
            self.notify(InstrumentEvent("rejected", order.instrument_id, ts, {"order": order.id, "reason": order.reason}), order.strategy if order.strategy in self.contexts else None)
            return order
        order.status = "working"
        inst = self.registry.get(order.instrument_id)
        if self.config.fill_policy == "next_event" and self._has_events(order.instrument_id):
            self.pending_next.setdefault(order.instrument_id, []).append(order)
        else:
            delay = pd.Timedelta(self.config.otc_latency if not self._has_events(order.instrument_id) else self.config.latency)
            order.arrival_at = ts + delay
            self.queue.push(order.arrival_at, "arrival", {"order": order.id})
        return order

    def cancel_order(self, order_id: int) -> None:
        o = self.orders.get(order_id)
        if o is not None and o.is_open:
            o.status = "cancelled"
            self.log.add(self.clock.now, "cancel", str(order_id))

    def cancel_strategy_orders(self, strategy: str, instrument_id: str | None = None) -> None:
        for o in list(self.orders.values()):
            if o.is_open and o.strategy == strategy and (instrument_id is None or o.instrument_id == instrument_id):
                self.cancel_order(o.id)

    def _quote_for(self, inst, ts, event_row=None) -> Quote | None:
        """The tradeable market state for ``inst`` at ``ts``: from the triggering event, or the latest known data / model mark."""
        cs = self.costs.models_for(inst)
        needs_stats = cs["impact"].kind != "none" or self.liquidity.max_participation is not None
        adv = sigma = float("nan")
        if needs_stats:
            vols = self.data.volume(inst.instrument_id, self.liquidity.adv_window)
            adv = float(vols.mean()) if len(vols) else float("nan")
            sigma = realised_sigma(self.data.history(inst.instrument_id, 22))
        if event_row is not None:
            et = event_row.event_type
            if et == "quote":
                return Quote((event_row.bid + event_row.ask) / 2.0, float(event_row.bid), float(event_row.ask), float(event_row.trade), float(event_row.volume), adv=adv, sigma=sigma,
                             depth=_depth(event_row))
            if et == "bar":
                return Quote(float(event_row.close), last=float(event_row.close), volume=float(event_row.volume), low=float(event_row.low), high=float(event_row.high), adv=adv, sigma=sigma)
            if et == "trade":
                return Quote(float(event_row.trade), last=float(event_row.trade), volume=float(event_row.volume), adv=adv, sigma=sigma)
            return Quote(float(event_row.settlement), last=float(event_row.settlement), adv=adv, sigma=sigma)
        r = self.marks.mark(inst, ts)
        if r is None:
            return None
        bid, ask = getattr(r, "bid", float("nan")), getattr(r, "ask", float("nan"))
        return Quote(r.price, bid, ask, adv=adv, sigma=sigma)

    def _attempt(self, order: Order, ts, event_row=None) -> None:
        if not order.is_open:
            return
        inst = self.registry.get(order.instrument_id)
        if inst.expiry is not None and ts > inst.expiry + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1):
            order.status, order.reason = "expired", "instrument expired"
            return
        quote = self._quote_for(inst, ts, event_row)
        if quote is None or not np.isfinite(quote.mid):
            self.diagnostics["no_quote_for_order"] += 1
            if order.tif in ("IOC", "FOK"):
                order.status, order.reason = "cancelled", "no market"
            return
        res = simulate_fill(order, inst, quote, self.costs, self.liquidity)
        if res.triggered:
            order.triggered = True
        if res.quantity != 0.0:
            self._book_fill(order, inst, res, ts)
        if order.is_open and order.tif in ("IOC", "FOK"):
            order.status = "cancelled" if order.filled == 0 else order.status
            if order.filled != 0 and abs(order.remaining) > 1e-12:
                order.status, order.reason = "cancelled", "IOC remainder cancelled"

    def _book_fill(self, order: Order, inst, res, ts) -> None:
        fill = Fill(pd.Timestamp(ts), inst.instrument_id, res.quantity, res.price, order.id, res.fee, res.fee_currency, res.spread_price, res.impact_price, tuple(order.tags), order.strategy)
        self.ledger.apply_fill(ts, fill, res.mid)
        self.qty_book[inst.instrument_id] = self.qty_book.get(inst.instrument_id, 0.0) + res.quantity
        if order.strategy in self.strategy_qty:
            book = self.strategy_qty[order.strategy]
            book[inst.instrument_id] = book.get(inst.instrument_id, 0.0) + res.quantity
        prev = order.filled
        order.filled += res.quantity
        order.avg_price = (order.avg_price * abs(prev) + res.price * abs(res.quantity)) / abs(order.filled)
        order.fees += res.fee
        order.status = "filled" if abs(order.remaining) < 1e-12 else "partial"
        self.log.add(ts, "fill", f"{order.id}|{inst.instrument_id}|{res.quantity:.10g}|{res.price:.10g}|{res.fee:.8g}")
        self.diagnostics["fills"] += 1
        if order.strategy in self.contexts:
            for s in self.strategies:
                if s.name == order.strategy:
                    s.on_portfolio_update(self.contexts[s.name], fill)

    # ------------------------------------------------------------------------------------------------------------------------------ targets
    def _resolve_chain(self, chain_id: str, ts) -> str | None:
        cur = self.chain_current.get(chain_id)
        if cur is not None and cur in self.registry and not self.registry.get(cur).has_expired_by(ts):
            return cur
        f = self.registry.chain(chain_id).front(ts)
        if f is not None:
            self.chain_current[chain_id] = f.instrument_id
            return f.instrument_id
        return None

    def apply_targets(self, strategy: str, targets: list[Target], **kwargs) -> dict:
        """Turn targets into orders: resolve chains, size in whole contracts, apply constraints, net against the strategy's own holdings and trade the difference."""
        ts = self.clock.now
        cfg = self.config
        equity = self.ledger.equity()
        capital = equity * self.capital_share(strategy)
        own = self.strategy_qty.setdefault(strategy, {})
        wanted: dict[str, float] = {}                     # instrument id -> target quantity (contracts)
        notionals: dict[str, float] = {}
        unit: dict[str, float] = {}
        chain_of: dict[str, str] = {}
        for t in targets:
            iid = t.instrument_id
            if self.registry.is_chain(iid):
                chain_id = iid
                iid = self._resolve_chain(chain_id, ts)
                if iid is None:
                    self.diagnostics["target_no_contract"] += 1
                    continue
                chain_of[iid] = chain_id
            if not self.registry.has_instrument(iid):
                self.diagnostics["target_unknown_instrument"] += 1
                continue
            inst = self.registry.get(iid)
            r = self.marks.mark(inst, ts)
            if r is None or not np.isfinite(r.price) or (r.price <= 0 and inst.cash_style not in ("otc_mtm", "variation_margin")):
                self.diagnostics["target_no_price"] += 1
                continue
            u = self.unit_notional_base(inst, r.price)
            if not u > 0:
                self.diagnostics["target_no_price"] += 1
                continue
            unit[iid] = u
            if t.quantity is not None:
                notionals[iid] = t.quantity * u
            elif t.notional is not None:
                notionals[iid] = t.notional
            else:
                notionals[iid] = t.weight * capital
        constraint = (self.constraints or {}).get(strategy) or (self.constraints or {}).get("*")
        bound: list[str] = []
        if constraint is not None and notionals:
            inputs = self._constraint_inputs(strategy, capital, list(notionals), unit)
            notionals, bound = constraint.apply(notionals, inputs)
            for b in bound:
                self.diagnostics[f"constraint:{b}"] += 1
        for iid, n in notionals.items():
            inst = self.registry.get(iid)
            q = inst.round_quantity(n / unit[iid])
            wanted[iid] = q
        orders = {}
        # chain-level current holdings: all contracts of the chain count toward the chain target
        for iid, q in wanted.items():
            chain_id = chain_of.get(iid)
            if chain_id:
                held = sum(own.get(c.instrument_id, 0.0) for c in self.registry.chain(chain_id).contracts)
            else:
                held = own.get(iid, 0.0)
            delta = q - held
            inst = self.registry.get(iid)
            if abs(delta) < inst.lot_size - 1e-12:
                continue
            if cfg.min_trade_fraction > 0 and abs(delta) * unit[iid] < cfg.min_trade_fraction * capital:
                continue
            self.cancel_strategy_orders(strategy, iid)
            o = self.submit_order(Order(iid, delta, strategy=strategy, tags=tuple(kwargs.get("tags", ()))))
            orders[iid] = o
        self.log.add(ts, "targets", f"{strategy}|{len(targets)}|{len(orders)}|{','.join(bound)}")
        return {"orders": orders, "constraints": bound, "wanted": wanted}

    def _constraint_inputs(self, strategy: str, capital: float, ids: list[str], unit: dict) -> ConstraintInputs:
        own = self.strategy_qty.get(strategy, {})
        current = {i: own.get(i, 0.0) * unit.get(i, 0.0) for i in ids}
        asset_class = {i: self.registry.get(i).asset_class for i in ids}
        ccy_exp = {}
        margin_rate = {}
        for i in ids:
            inst = self.registry.get(i)
            if hasattr(inst, "base_currency") and hasattr(inst, "quote_currency") and inst.cash_style == "currency_exchange":
                ccy_exp[i] = {inst.base_currency: 1.0, inst.quote_currency: -1.0}
            else:
                ccy_exp[i] = {inst.currency: 1.0}
            if inst.margin_type == "percent":
                margin_rate[i] = inst.initial_margin
            elif inst.margin_type == "fixed" and unit.get(i, 0) > 0:
                margin_rate[i] = inst.initial_margin / unit[i]
        adv = {}
        for i in ids:
            vols = self.data.volume(i, 20)
            px = self.data.mid(i)
            if len(vols) and np.isfinite(px):
                adv[i] = float(vols.mean()) * unit[i]
        cov = None
        if any(c.vol_target is not None for c in (self.constraints or {}).values()):
            rets = pd.DataFrame({i: self.data.returns(i, 126) for i in ids}).dropna(how="all").fillna(0.0)
            if len(rets) > 20:
                cov = rets.ewm(halflife=40).cov().iloc[-len(ids):].droplevel(0) * 252.0
        return ConstraintInputs(capital, current, asset_class, ccy_exp, adv, cov, {}, self.current_drawdown(), margin_rate)

    # ----------------------------------------------------------------------------------------------------------------------------- exercise
    def request_exercise(self, strategy: str, instrument_id: str, quantity: float | None) -> None:
        self.exercise_requests.append((strategy, instrument_id, quantity))
        self.queue.push(self.clock.now, "lifecycle", {"op": "exercise_request"})

    # ----------------------------------------------------------------------------------------------------------------------------- the loop
    def _setup(self) -> None:
        cfg = self.config
        self.clock.advance(self.start)
        for ccy, amt in cfg.initial_cash.items():
            self.ledger.deposit(self.start, ccy, amt)
        self.peak_equity = self.ledger.equity()
        self.start_equity = self.peak_equity
        for s in self.strategies:
            for t in s.schedule.timestamps(self.start, self.end):
                self.queue.push(t, "schedule", {"strategy": s.name})
        self.lifecycle.setup(self)
        self.started = True
        for s in self.strategies:
            s.on_start(self.contexts[s.name])
        self._equity_record(self.start)

    def begin(self) -> None:
        """Set up a run: initial cash, strategy schedules, lifecycle events and the ``on_start`` hooks. ``run`` calls it; a live or paper session calls it once and then ``advance``."""
        self._setup()
        self._cursor = int(np.searchsorted(self.store._avail_all, self.start.value, side="right"))
        self._subs = {s.name: set(s.subscriptions) for s in self.strategies}

    def advance(self, until=None) -> None:
        """Process everything that happens up to and including ``until`` (default: the end): market events in the order they became available, interleaved with the engine's own events.

        With a streaming feed the contract is: add every event available up to ``until`` to the store (``store.extend``) BEFORE advancing to ``until``; the engine then never decides on
        a clock that is ahead of its data."""
        until_ns = self.end.value if until is None else min(pd.Timestamp(until).value, self.end.value)
        avail = self.store._avail_all
        hi = int(np.searchsorted(avail, until_ns, side="right"))
        i = self._cursor
        while True:
            nm = int(avail[i]) if i < hi else None
            top = self.queue.peek()
            if top is not None and top.ts_ns > self.end.value:
                self.queue.pop()
                continue
            if top is not None and top.ts_ns > until_ns:
                top = None
            if nm is None and top is None:
                break
            if top is not None and (nm is None or top.ts_ns < nm):
                ev = self.queue.pop()
                self.clock.advance(ev.ts)
                self._dispatch(ev)
                continue
            ts = pd.Timestamp(nm)
            self.clock.advance(ts)
            while i < hi and int(avail[i]) == nm:
                self._on_market(i, self._subs)
                i += 1
            self._cursor = i
        self._cursor = i

    def finish(self):
        """Close the run (final marks, ``on_end`` hooks) and return the :class:`~src.engine.analysis.BacktestResult`."""
        from .analysis import BacktestResult

        self._finish()
        return BacktestResult.from_engine(self)

    def run(self):
        self.begin()
        self.advance(self.end)
        return self.finish()

    def _on_market(self, i: int, subs: dict) -> None:
        from ..marketdata.store import Obs

        obs = Obs(self.store, i)
        ts = self.clock.now
        iid, et = obs.instrument_id, obs.event_type
        self.log.add(ts, "market", f"{iid}|{et}")
        if et == "corporate_action":
            self.queue.push(ts, "corporate", {"row": i})
        if et in TRADEABLE and iid in self.registry:
            queue = self.pending_next.pop(iid, [])
            for o in queue:
                if o.is_open and o.submitted_at < ts:
                    self._attempt(o, ts, obs)
                    if o.is_open and o.tif != "IOC":
                        self.working.setdefault(iid, []).append(o)
                elif o.is_open:
                    self.pending_next.setdefault(iid, []).append(o)
            still = []
            for o in self.working.get(iid, []):
                if o.is_open and o.submitted_at < ts:
                    self._attempt(o, ts, obs)
                if o.is_open:
                    still.append(o)
            if iid in self.working:
                self.working[iid] = still
        for s in self.strategies:
            if iid in subs[s.name]:
                s.on_market(self.contexts[s.name], obs)

    def _dispatch(self, ev) -> None:
        kind = ev.kind
        if kind == "schedule":
            for s in self.strategies:
                if s.name == ev.payload["strategy"]:
                    self.log.add(ev.ts, "schedule", s.name)
                    s.on_schedule(self.contexts[s.name])
        elif kind == "arrival":
            o = self.orders[ev.payload["order"]]
            self.log.add(ev.ts, "arrival", str(o.id))
            if o.is_open:
                self._attempt(o, ev.ts)
                if o.is_open and o.tif not in ("IOC", "FOK"):
                    self.working.setdefault(o.instrument_id, []).append(o)
        elif kind == "mark":
            self.lifecycle.on_mark(self, ev)
        elif kind == "accrual":
            self.lifecycle.on_accrual(self, ev)
        elif kind in ("lifecycle", "corporate", "margin", "roll", "close"):
            self.lifecycle.handle(self, ev)
        else:
            raise RuntimeError(f"unknown event kind '{kind}'")

    # ------------------------------------------------------------------------------------------------------------------------------ marking
    def mark_positions(self, ts, only=None) -> None:
        held = [i for i, p in self.ledger.positions.items() if not p.is_flat and (only is None or i in only)]
        fx_ids = [i.instrument_id for i in self.registry if i.cash_style == "currency_exchange" and self._has_events(i.instrument_id)]
        marks, ages = self.marks.mark_all(ts, sorted(set(held) | set(fx_ids)))
        limit = pd.Timedelta(self.config.max_mark_age) if self.config.max_mark_age else None
        for iid, age in list(ages.items()):
            if limit is not None and age > limit and iid in held:
                self.diagnostics["stale_marks"] += 1
                marks.pop(iid, None)
        for iid in held:
            if iid not in marks:
                self.diagnostics["missing_marks"] += 1
        self.ledger.revalue(ts, marks)
        self.current_marks.update(marks)
        self.mark_ages.update(ages)

    def _equity_record(self, ts) -> dict:
        eq = self.ledger.equity()
        self.peak_equity = max(self.peak_equity, eq)
        row = self.ledger.snapshot(ts, self.current_marks)
        row["drawdown"] = 0.0 if self.peak_equity <= 0 else 1.0 - eq / self.peak_equity
        if self.last_margin is not None:
            row["initial_margin"] = self.last_margin.initial_margin
            row["maintenance_margin"] = self.last_margin.maintenance_margin
            row["margin_utilization"] = self.last_margin.utilization
        self.equity_rows.append(row)
        return row

    def _finish(self) -> None:
        end = self.end
        self.clock.advance(max(self.clock.now, end))
        self.lifecycle.finalize(self)
        for s in self.strategies:
            s.on_end(self.contexts[s.name])
        self.log.add(self.clock.now, "end", f"{self.ledger.equity():.6f}")
        for e in self.ledger.journal:
            self.log.add(e.ts, "journal", f"{e.category}|{e.instrument_id}|{e.pnl:.8g}")

    def reconciliation(self):
        return reconcile(self.ledger, start_equity=0.0, expected_quantities=self.qty_book, ts=self.clock.now, tolerance=self.config.tolerance)
