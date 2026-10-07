"""Contract lifecycle: everything that happens to a position because of the CONTRACT rather than because of a trade.

=========================  =====================================================================================================================================
event                      what the engine does
=========================  =====================================================================================================================================
daily mark (``on_mark``)   revalue all positions from point-in-time marks, settle variation margin, translate foreign balances, check margin, record equity
accrual (``on_accrual``)   interest on cash balances, financing on negative balances, borrow fees on short positions
futures expiry             cash-settle at the final settlement price (a physically settled contract can instead raise: ``physical_expiry='error'``)
futures roll               the roll rule of the chain selects the contract to hold; positions in earlier contracts are rolled with a pair of trades tagged ``roll``
perpetual funding          every funding interval: ``-position notional at the MARK price x funding rate``, paid to or from cash
option expiry              in-the-money long options are exercised and short ones assigned (cash or physical delivery into the underlying), the rest expire worthless
early exercise/assignment  a long American option is exercised on request; a short one is assigned when its time value is gone (``early_assignment``)
swap, bond, forward flows  coupons and net swap payments on their payment dates (from the registered cashflow model), the final exchange of a forward, maturity removal
corporate actions          splits rescale positions; dividends pay holders and charge shorts
margin                     margin call, liquidation of the largest requirement at the mark plus a penalty, isolated-margin liquidation of perpetuals
=========================  =====================================================================================================================================

Instrument types register their cashflow dates and amounts in ``CASHFLOW_DATES`` and ``CASHFLOW_MODELS`` (the swap module does so for swaps, bonds and basis swaps).
Every posting goes through the ledger, so each one is an explained journal entry and the position quantities are mirrored in the engine's independent ``qty_book``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..instruments.crypto import CryptoFuture, CryptoPerp
from ..instruments.futures import Future
from ..instruments.fx import FXForward
from ..instruments.options import Option
from ..ledger import Fill
from .orders import Order
from .strategy import InstrumentEvent


class LifecycleError(RuntimeError):
    pass


CASHFLOW_DATES: dict[str, Callable] = {}      # instrument_type -> fn(inst) -> list of payment timestamps
CASHFLOW_MODELS: dict[str, Callable] = {}     # instrument_type -> fn(engine, inst, ts) -> {currency: amount per unit of position}


def register_cashflows(instrument_type: str, dates: Callable, amounts: Callable) -> None:
    CASHFLOW_DATES[instrument_type] = dates
    CASHFLOW_MODELS[instrument_type] = amounts


# --------------------------------------------------------------------------------------------------------------------------------- setup
def _day_end(engine, ts: pd.Timestamp, minutes_before: int = 1) -> pd.Timestamp:
    h, m = (int(x) for x in engine.config.snapshot_time.split(":"))
    return ts.normalize() + pd.Timedelta(hours=h, minutes=m) - pd.Timedelta(minutes=minutes_before)


def expiry_time(engine, inst) -> pd.Timestamp:
    """When the engine processes an expiry: an expiry given as a date means the END of that day (after the day's closing data), a timestamp means that instant."""
    e = inst.expiry
    return _day_end(engine, e) if e == e.normalize() else e


def setup(engine) -> None:
    cfg = engine.config
    if cfg.session_days is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(list(cfg.session_days))).normalize().unique().sort_values()
    else:
        dates = pd.DatetimeIndex(engine.store.events["available_at"].dt.normalize().unique()).sort_values()
    dates = dates[(dates >= engine.start.normalize()) & (dates <= engine.end.normalize())]
    h, m = (int(x) for x in cfg.snapshot_time.split(":"))
    rh, rm = (int(x) for x in cfg.roll_time.split(":"))
    for d in dates:
        ts = d + pd.Timedelta(hours=h, minutes=m)
        if ts > engine.end:
            ts = engine.end
        engine.queue.push(ts, "accrual", {})
        engine.queue.push(ts, "mark", {})
        engine.queue.push(ts, "close", {"op": "close_day"})
        if engine.registry.chains:
            engine.queue.push(max(d + pd.Timedelta(hours=rh, minutes=rm), engine.start), "roll", {"op": "roll_check"})
    if engine.end > engine.start:
        engine.queue.push(engine.end, "mark", {"final": True})
    for inst in list(engine.registry):
        schedule_instrument(engine, inst, engine.start, engine.end)
    engine.last_accrual = engine.start


def schedule_instrument(engine, inst, start: pd.Timestamp, end: pd.Timestamp) -> None:
    last_flow = None
    dates_fn = CASHFLOW_DATES.get(inst.instrument_type)
    if dates_fn is not None:
        for d in dates_fn(inst):
            d = pd.Timestamp(d)
            if start < d <= end:
                engine.queue.push(_day_end(engine, d, 2) if d == d.normalize() else d, "lifecycle", {"op": "cashflow", "iid": inst.instrument_id, "date": d})
            last_flow = d if last_flow is None or d > last_flow else last_flow
    if isinstance(inst, CryptoPerp):
        for t in inst.funding_times(start, end):
            engine.queue.push(t, "lifecycle", {"op": "funding", "iid": inst.instrument_id})
    if inst.expiry is not None:
        t = expiry_time(engine, inst)
        if last_flow is not None:
            t = max(t, (_day_end(engine, last_flow, 1) if last_flow == last_flow.normalize() else last_flow + pd.Timedelta(minutes=1)))
        if start < t <= end:
            engine.queue.push(t, "lifecycle", {"op": "expiry", "iid": inst.instrument_id})


def finalize(engine) -> None:
    ts = engine.clock.now
    engine.mark_positions(ts)
    engine._equity_record(ts)


# -------------------------------------------------------------------------------------------------------------------------------- helpers
def strategy_slices(engine, iid: str) -> list[tuple[str, float]]:
    """``(strategy, quantity)`` pairs for an instrument, with any quantity not attributed to a strategy (an engine-side liquidation, for example) under ``''``."""
    out, total = [], 0.0
    for name, book in engine.strategy_qty.items():
        q = book.get(iid, 0.0)
        if abs(q) > 1e-12:
            out.append((name, q))
            total += q
    rest = engine.ledger.quantity(iid) - total
    if abs(rest) > 1e-9:
        out.append(("", rest))
    return out


def _zero_quantities(engine, iid: str) -> None:
    engine.qty_book[iid] = 0.0
    for book in engine.strategy_qty.values():
        if iid in book:
            book[iid] = 0.0
    for o in engine.orders.values():
        if o.instrument_id == iid and o.is_open:
            o.status, o.reason = "cancelled", "instrument settled"
    engine.pending_next.pop(iid, None)
    engine.working.pop(iid, None)


def settlement_price(engine, inst, ts) -> float:
    """The final settlement price of an expiring instrument: the official settlement if published, else the last price known (a stale price is counted, a missing one is an error)."""
    snap = engine.store.price(inst.instrument_id, ts, order=("settlement", "bar", "trade", "quote", "mark"))
    if snap is None:
        raise LifecycleError(f"no price to settle {inst.instrument_id} at {ts}: provide a settlement or closing price for the expiry date")
    if snap.timestamp < inst.expiry.normalize() - pd.Timedelta(days=5):
        engine.diagnostics["stale_settlement"] += 1
    return snap.mid


def underlying_mark(engine, inst, ts) -> float:
    u = engine.registry.get(inst.underlying_id)
    r = engine.marks.mark(u, ts)
    if r is None:
        snap = engine.store.price(u.instrument_id, ts, order=("settlement", "bar", "trade", "quote", "mark"))
        if snap is None:
            raise LifecycleError(f"no price for the underlying {u.instrument_id} of {inst.instrument_id} at {ts}")
        return snap.mid
    return r.price


# ------------------------------------------------------------------------------------------------------------------------------- handlers
def handle(engine, ev) -> None:
    op = ev.payload.get("op")
    ts = ev.ts
    if ev.kind == "corporate":
        return corporate_action(engine, ev)
    if ev.kind == "roll":
        return roll_check(engine, ts)
    if ev.kind == "close":
        for o in engine.orders.values():
            if o.is_open and o.tif == "DAY" and o.submitted_at.normalize() == ts.normalize():
                o.status, o.reason = "expired", "day order expired"
        return
    engine.log.add(ts, f"lifecycle:{op}", ev.payload.get("iid", ""))
    if op == "expiry":
        return expiry(engine, engine.registry.get(ev.payload["iid"]), ts)
    if op == "funding":
        return funding(engine, engine.registry.get(ev.payload["iid"]), ts)
    if op == "cashflow":
        return cashflow(engine, engine.registry.get(ev.payload["iid"]), ts, ev.payload["date"])
    if op == "exercise_request":
        return process_exercise_requests(engine, ts)
    raise LifecycleError(f"unknown lifecycle operation '{op}'")


def on_accrual(engine, ev) -> None:
    cfg = engine.config
    ts = ev.ts
    if not cfg.accrue or engine.last_accrual is None:
        engine.last_accrual = ts
        return
    dt = (ts - engine.last_accrual).total_seconds() / (engine.financing.day_count * 86400.0)
    if dt > 0:
        led = engine.ledger
        borrow = {}
        for ccy in led.cash:
            r = engine.financing.deposit_rates.get(ccy, 0.0)
            borrow[ccy] = engine.financing.borrow_rates.get(ccy, r + engine.financing.borrow_spread)
        led.accrue_interest(ts, dt, engine.financing.deposit_rates, borrow)
        short_rates = engine.financing.short_rates_for(list(led.positions))
        if short_rates:
            led.accrue_borrow(ts, dt, short_rates, engine.current_marks)
        if engine.financing.margin_interest_spread and engine.last_margin is not None and engine.last_margin.initial_margin > 0:
            amt = -engine.last_margin.initial_margin * engine.financing.margin_interest_spread * dt
            led.post_cashflow(ts, "financing", engine.config.base_currency, amt, note="margin interest")
    engine.last_accrual = ts


def on_mark(engine, ev) -> None:
    ts = ev.ts
    engine.mark_positions(ts)
    if engine.config.early_assignment:
        early_assignment(engine, ts)
    if engine.config.margin_check:
        check_margin(engine, ts)
    row = engine._equity_record(ts)
    need_risk = engine.risk_model is not None and (engine.config.risk_every_snapshot or any(type(s).on_risk_update.__qualname__ != "Strategy.on_risk_update" for s in engine.strategies))
    if need_risk:
        engine.last_risk = engine.risk_model.report(engine)
        if engine.config.risk_every_snapshot:
            engine.risk_rows.append({"ts": ts, **engine.last_risk.summary().to_dict()})
        for s in engine.strategies:
            s.on_risk_update(engine.contexts[s.name], engine.last_risk)
    engine.log.add(ts, "mark", f"{row['equity']:.6f}")


def expiry(engine, inst, ts) -> None:
    iid = inst.instrument_id
    led = engine.ledger
    q = led.quantity(iid)
    if abs(q) < 1e-12:
        _zero_quantities(engine, iid)
        return
    slices = strategy_slices(engine, iid)
    if isinstance(inst, Option):
        option_expiry(engine, inst, ts, slices)
    elif isinstance(inst, (Future, CryptoFuture)):
        if isinstance(inst, Future) and inst.settlement_type == "physical" and engine.config.physical_expiry == "error":
            raise LifecycleError(f"{iid} is physically settled and still held at expiry")
        px = settlement_price(engine, inst, ts)
        led.close_position(ts, iid, px, note="final settlement")
        _zero_quantities(engine, iid)
        engine.notify(InstrumentEvent("expiry", iid, ts, {"price": px, "quantity": q}))
    elif isinstance(inst, FXForward):
        fixing = None if inst.deliverable else _spot_for_forward(engine, inst, ts)
        cash = inst.settlement_amounts(q, fixing)
        spot = _spot_for_forward(engine, inst, ts)
        mark = (spot - inst.strike) if inst.deliverable else (fixing - inst.strike) if fixing is not None else 0.0
        led.settle_with_cash(ts, iid, cash, mark, note="forward settlement")
        _zero_quantities(engine, iid)
        engine.notify(InstrumentEvent("expiry", iid, ts, {"cash": cash}))
    else:
        led.settle_with_cash(ts, iid, {}, 0.0, note="maturity")
        _zero_quantities(engine, iid)
        engine.notify(InstrumentEvent("expiry", iid, ts, {"quantity": q}))
    engine.mark_positions(ts)


def _spot_for_forward(engine, inst, ts) -> float:
    u = inst.underlying_id
    snap = engine.store.price(u, ts, order=("quote", "trade", "bar", "mark", "settlement"))
    if snap is None:
        raise LifecycleError(f"no spot price for {u} to settle {inst.instrument_id}")
    return snap.mid


def option_expiry(engine, inst: Option, ts, slices) -> None:
    led = engine.ledger
    iid = inst.instrument_id
    S = underlying_mark(engine, inst, ts)
    q = led.quantity(iid)
    intrinsic = inst.intrinsic(S)
    if intrinsic > inst.auto_exercise_threshold:
        kind = "exercise" if q > 0 else "assignment"
        _settle_option(engine, inst, ts, S, intrinsic, slices, kind)
    else:
        led.close_position(ts, iid, 0.0, note="expired worthless")
        _zero_quantities(engine, iid)
        engine.notify(InstrumentEvent("expiry", iid, ts, {"underlying": S, "quantity": q}))


def _settle_option(engine, inst: Option, ts, S: float, intrinsic: float, slices, kind: str) -> None:
    led = engine.ledger
    iid = inst.instrument_id
    q = led.quantity(iid)
    if inst.settlement_type == "cash" or inst.underlying_kind == "index":
        led.close_position(ts, iid, intrinsic, note=f"cash {kind}")
        _zero_quantities(engine, iid)
        engine.notify(InstrumentEvent(kind, iid, ts, {"underlying": S, "intrinsic": intrinsic, "quantity": q, "delivery": None}))
        return
    spec = inst.exercise_settlement(q, S)
    under_id, dq_option_units, price = spec["delivery"]
    if under_id not in engine.registry:
        raise LifecycleError(f"{iid} delivers {under_id}, which is not a registered instrument")
    under = engine.registry.get(under_id)
    dq = dq_option_units if inst.underlying_kind == "future" else dq_option_units / under.contract_multiplier          # shares -> underlying contracts
    led.remove_position_value(ts, iid, intrinsic, note=f"physical {kind}")
    fill = Fill(ts, under_id, dq, price, order_id=-1, tags=(kind,), settlement=True)
    led.apply_fill(ts, fill, mid=S, strategy="")
    engine.qty_book[under_id] = engine.qty_book.get(under_id, 0.0) + dq
    total_q = sum(s for _, s in slices) or q
    for name, sq in slices:                                              # the delivered position goes to the strategies that held the option, pro rata
        if name in engine.strategy_qty:
            b = engine.strategy_qty[name]
            b[under_id] = b.get(under_id, 0.0) + dq * (sq / total_q)
    _zero_quantities(engine, iid)
    engine.current_marks[under_id] = S
    engine.notify(InstrumentEvent(kind, iid, ts, {"underlying": S, "intrinsic": intrinsic, "quantity": q, "delivery": (under_id, dq, price)}))


def early_assignment(engine, ts) -> None:
    cfg = engine.config
    for iid, pos in list(engine.ledger.positions.items()):
        inst = engine.registry.get(iid)
        if not isinstance(inst, Option) or inst.exercise_style != "american" or pos.quantity >= 0:
            continue
        try:
            S = underlying_mark(engine, inst, ts)
        except LifecycleError:
            continue
        intrinsic = inst.intrinsic(S)
        mark = engine.current_marks.get(iid, pos.last_mark)
        if intrinsic > inst.auto_exercise_threshold and mark == mark and mark - intrinsic <= cfg.assignment_extrinsic:
            _settle_option(engine, inst, ts, S, intrinsic, strategy_slices(engine, iid), "assignment")


def process_exercise_requests(engine, ts) -> None:
    requests, engine.exercise_requests = engine.exercise_requests, []
    for strategy, iid, quantity in requests:
        inst = engine.registry.get(iid)
        pos = engine.ledger.quantity(iid)
        if not isinstance(inst, Option) or pos <= 0:
            engine.diagnostics["exercise_rejected"] += 1
            continue
        if inst.exercise_style != "american" and ts < inst.expiry.normalize():
            engine.diagnostics["exercise_rejected"] += 1
            continue
        own = engine.strategy_qty.get(strategy, {}).get(iid, 0.0)
        if quantity is not None and abs(quantity - pos) > 1e-9 and abs(quantity - own) > 1e-9:
            raise ValueError("partial exercise is not supported: exercise the whole position")
        S = underlying_mark(engine, inst, ts)
        intrinsic = inst.intrinsic(S)
        if intrinsic <= 0:
            engine.diagnostics["exercise_rejected"] += 1
            continue
        _settle_option(engine, inst, ts, S, intrinsic, strategy_slices(engine, iid), "exercise")


def funding(engine, inst: CryptoPerp, ts) -> None:
    q_total = engine.ledger.quantity(inst.instrument_id)
    if abs(q_total) < 1e-12:
        return
    o = engine.store.latest(inst.instrument_id, "funding", ts)
    if o is None or abs(pd.Timestamp(o.timestamp) - ts) > pd.Timedelta(hours=inst.funding_interval_hours) / 2:
        engine.diagnostics["missing_funding"] += 1
        return
    mark = engine.marks.mark(inst, ts)
    if mark is None:
        engine.diagnostics["missing_funding"] += 1
        return
    rate = float(o.funding)
    for name, q in strategy_slices(engine, inst.instrument_id):
        amount = inst.funding_payment(q, mark.price, rate)
        engine.ledger.post_cashflow(ts, "funding", inst.currency, amount, inst.instrument_id, strategy=name, note=f"rate {rate:.6%}")
        engine.ledger.positions[inst.instrument_id].funding_paid += amount
    engine.notify(InstrumentEvent("funding", inst.instrument_id, ts, {"rate": rate, "mark": mark.price, "quantity": q_total}))


def cashflow(engine, inst, ts, date) -> None:
    q_total = engine.ledger.quantity(inst.instrument_id)
    if abs(q_total) < 1e-12:
        return
    model = CASHFLOW_MODELS[inst.instrument_type]
    per_unit = model(engine, inst, date)
    for name, q in strategy_slices(engine, inst.instrument_id):
        for ccy, amount in per_unit.items():
            engine.ledger.post_cashflow(ts, "coupon", ccy, amount * q, inst.instrument_id, strategy=name, note=f"payment {pd.Timestamp(date).date()}")
    engine.mark_positions(ts, only={inst.instrument_id})
    engine.notify(InstrumentEvent("cashflow", inst.instrument_id, ts, {"per_unit": per_unit, "quantity": q_total}))


def corporate_action(engine, ev) -> None:
    store = engine.store
    row = ev.payload["row"]
    iid = store._cols["instrument_id"][row]
    spec = store._cols["reference_values"][row] or {}
    if iid not in engine.registry:
        return
    kind = spec.get("type")
    ts = ev.ts
    q = engine.ledger.quantity(iid)
    if kind == "split":
        ratio = float(spec["ratio"])
        engine.ledger.split(ts, iid, ratio)
        if iid in engine.qty_book:
            engine.qty_book[iid] *= ratio
        for book in engine.strategy_qty.values():
            if iid in book:
                book[iid] *= ratio
        for o in engine.orders.values():
            if o.instrument_id == iid and o.is_open:
                o.status, o.reason = "cancelled", "corporate action"
        engine.notify(InstrumentEvent("corporate_action", iid, ts, {"type": "split", "ratio": ratio}))
    elif kind == "dividend" and abs(q) > 1e-12:
        inst = engine.registry.get(iid)
        amount = float(spec["amount"])
        for name, sq in strategy_slices(engine, iid):
            engine.ledger.post_cashflow(ts, "dividend", spec.get("currency", inst.currency), sq * inst.contract_multiplier * amount, iid, strategy=name)
        engine.notify(InstrumentEvent("corporate_action", iid, ts, {"type": "dividend", "amount": amount, "quantity": q}))
    else:
        engine.diagnostics["corporate_action_ignored"] += 1


# ----------------------------------------------------------------------------------------------------------------------------------- roll
def desired_contract(engine, chain, ts):
    """The contract the chain's roll rule says to hold at ``ts``."""
    front = chain.front(ts)
    if front is None:
        return None
    nxt = chain.after(front)
    if nxt is None:
        return front
    spec = chain.roll
    calendar_due = ts.normalize() >= chain.roll_date(front)
    if spec.method in ("calendar", "fixed_days"):
        return nxt if calendar_due else front
    if spec.method == "volume":
        a = engine.data.volume(front.instrument_id, 1)
        b = engine.data.volume(nxt.instrument_id, 1)
    else:
        a = engine.data.open_interest(front.instrument_id, 1)
        b = engine.data.open_interest(nxt.instrument_id, 1)
    if len(a) and len(b) and b.iloc[-1] >= spec.ratio * a.iloc[-1]:
        return nxt
    return nxt if calendar_due else front


def roll_check(engine, ts) -> None:
    cfg = engine.config
    chains = engine.registry.chains
    names = chains if cfg.managed_chains is None else [c for c in chains if c in cfg.managed_chains]
    for cid in names:
        chain = chains[cid]
        want = desired_contract(engine, chain, ts)
        if want is None:
            continue
        engine.chain_current[cid] = want.instrument_id
        for strat, book in engine.strategy_qty.items():
            for c in chain.contracts:
                held = book.get(c.instrument_id, 0.0)
                if abs(held) > 1e-12 and c.expiry < want.expiry and not c.has_expired_by(ts):
                    already = any(o.is_open and o.strategy == strat and o.instrument_id == c.instrument_id and "roll" in o.tags for o in engine.orders.values())
                    if already:
                        continue
                    engine.submit_order(Order(c.instrument_id, -held, strategy=strat, tags=("roll",)))
                    engine.submit_order(Order(want.instrument_id, held, strategy=strat, tags=("roll",)))
                    engine.diagnostics["rolls"] += 1
                    engine.notify(InstrumentEvent("roll", c.instrument_id, ts, {"from": c.instrument_id, "to": want.instrument_id, "quantity": held}), strat)


# ------------------------------------------------------------------------------------------------------------------------------- margin
def risk_records_for(engine, ts) -> tuple[dict, dict]:
    """Greek records for portfolio-margin instruments and underlying prices for option margin (both from current marks)."""
    records, under = {}, {}
    for iid, pos in engine.ledger.positions.items():
        inst = engine.registry.get(iid)
        if inst.underlying_id is None or inst.underlying_id not in engine.registry:
            continue
        try:
            S = underlying_mark(engine, inst, ts)
        except LifecycleError:
            continue
        under[inst.underlying_id] = S
        if inst.margin_type == "portfolio" and isinstance(inst, Option):
            mark = engine.current_marks.get(iid)
            if mark is None or not mark > 0:
                continue
            from ..derivatives.iv import implied_vol
            T = inst.time_to_expiry(ts)
            if T <= 0:
                continue
            iv = float(np.asarray(implied_vol(mark, S, inst.strike, T, 0.0, 0.0, inst.is_call)).ravel()[0])
            if not np.isfinite(iv):
                continue
            g = inst.greeks(S, iv, ts)
            m = inst.contract_multiplier
            records[iid] = {"delta": g["delta"] * m, "gamma": g["gamma"] * m, "vega": g["vega"] * m / 100.0 * 100.0, "spot": S, "vol": iv, "underlying": inst.underlying_id}
    return records, under


def check_margin(engine, ts) -> None:
    led = engine.ledger
    if not led.positions:
        engine.last_margin = engine.margin_model.state(led, engine.current_marks)
        return
    records, under = risk_records_for(engine, ts)
    state = engine.margin_model.state(led, engine.current_marks, records, under)
    engine.last_margin = state
    if engine.config.liquidate:
        _isolated_liquidations(engine, ts)
    if state.margin_call:
        engine.diagnostics["margin_calls"] += 1
        engine.notify(InstrumentEvent("margin_call", "", ts, {"equity": state.equity, "maintenance": state.maintenance_margin}))
        if engine.config.liquidate:
            _liquidate(engine, ts, state, records, under)


def _forced_fill(engine, ts, iid: str, qty: float, mark: float, tag: str) -> None:
    inst = engine.registry.get(iid)
    penalty = engine.config.liquidation_penalty_bps * 1e-4 * mark
    sign = 1.0 if qty > 0 else -1.0
    price = mark + sign * penalty
    fill = Fill(ts, iid, qty, price, order_id=-1, spread_price=penalty, tags=(tag,))
    engine.ledger.apply_fill(ts, fill, mid=mark, strategy="")
    engine.qty_book[iid] = engine.qty_book.get(iid, 0.0) + qty
    held = {s: b.get(iid, 0.0) for s, b in engine.strategy_qty.items()}
    total = sum(held.values())
    for s, hq in held.items():                                           # reduce every strategy's holding in proportion
        if total != 0 and hq != 0:
            engine.strategy_qty[s][iid] = hq + qty * (hq / total)
    engine.diagnostics["liquidation_fills"] += 1
    engine.log.add(ts, "liquidation", f"{iid}|{qty:.10g}|{price:.10g}")
    engine.notify(InstrumentEvent("liquidation", iid, ts, {"quantity": qty, "price": price}))


def _liquidate(engine, ts, state, records, under) -> None:
    cfg = engine.config
    led = engine.ledger
    for _ in range(50):
        if state.equity >= state.maintenance_margin * (1.0 + cfg.liquidation_buffer) or not led.positions:
            break
        scores = {k: v[1] for k, v in state.by_instrument.items() if k in led.positions}
        if not scores:
            scores = {i: abs(p.last_value) + abs(p.quantity) for i, p in led.positions.items()}
        iid = max(scores, key=scores.get)
        pos = led.positions[iid]
        inst = engine.registry.get(iid)
        qty = -pos.quantity * cfg.liquidation_fraction
        qty = inst.round_quantity(qty) or -pos.quantity
        mark = engine.current_marks.get(iid, pos.last_mark)
        if not mark == mark:
            break
        _forced_fill(engine, ts, iid, qty, mark, "liquidation")
        engine.mark_positions(ts, only={iid})
        records, under = risk_records_for(engine, ts)
        state = engine.margin_model.state(led, engine.current_marks, records, under)
    engine.last_margin = state


def _isolated_liquidations(engine, ts) -> None:
    lev = engine.config.isolated_leverage
    for iid, pos in list(engine.ledger.positions.items()):
        inst = engine.registry.get(iid)
        if not isinstance(inst, CryptoPerp) or inst.margin_mode != "isolated" or pos.quantity == 0:
            continue
        mark = engine.current_marks.get(iid, pos.last_mark)
        liq = inst.liquidation_price(pos.avg_price, pos.quantity, lev)
        breached = (pos.quantity > 0 and mark <= liq) or (pos.quantity < 0 and mark >= liq)
        if breached:
            _forced_fill(engine, ts, iid, -pos.quantity, mark, "liquidation")
            engine.mark_positions(ts, only={iid})
