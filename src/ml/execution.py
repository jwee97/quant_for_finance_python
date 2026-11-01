"""The evaluation-mode adapter: target weights in, orders, fills, a ledger and a reconciliation out (spec v3, Release 4, P1-04 to P1-11).

The research engine (``src/backtest/engine.py``) turns weights into returns with a cost per unit of turnover. That is the declared mode ``vectorized_legacy``, and it stays for backward-compatible research. A
claim that depends on order lifecycles, partial fills, liquidity limits, borrow, corporate actions or cash needs the mode ``realistic_cash_equity``, which this module provides by driving the platform's own
accounting and cost components from the same weights:

* every fill is booked in ``src/ledger`` (principal at the mid, spread, impact, slippage and fees as separate journal entries; dividends, splits and borrow as cashflows), so the ledger's identities hold at
  every entry and ``reconcile`` can check the final state;
* the fill price, the order states and the participation limits are the ones in ``realism.py``; nothing is a free input (no volume record means no capacity, no borrow record means no new short);
* the result carries a gross P&L computed independently from the share holdings and the marks, so that "gross less itemized costs equals net" is a check and not an identity.

It is one driver loop over the existing components, not a third engine: order lifecycle, cost models and accounting are those of the event-driven stack. What it does not model is listed in ``limitations``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from ..backtest.execution import apply_execution_lag, restrict_to_rebalances, vol_target_scaling
from ..instruments import Instrument, InstrumentRegistry
from ..ledger import Fill, Ledger, reconcile
from . import provenance as pv
from . import realism as rz

MODES = ("vectorized_legacy", "realistic_cash_equity")


@dataclass
class ExecutionSpec:
    """The assumptions of one realistic evaluation. Every number here is an assumption of the run and is reported with it."""

    capital: float = 1_000_000.0
    convention: str = "next_close"
    delay: int = 1                                                    # sessions between the decision close and the fill: the platform's signal lag, applied ONCE, here
    sizing: str = "decision_close"                                    # "decision_close": shares fixed when the order is decided; "fill_time": re-sized at the fill session (the vectorized engine's convention)
    lot: int = 1
    max_participation: float | None = 0.10
    adv_window: int = 20
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    impact: rz.ImpactSpec = field(default_factory=rz.ImpactSpec)
    sigma_window: int = 20
    min_trade_weight: float = 0.002
    expiry_sessions: int = 2
    cash_rate: float = 0.0
    financing_spread: float = 0.005
    periods_per_year: int = 252

    def describe(self) -> dict:
        d = asdict(self)
        d["impact_label"] = self.impact.label()
        return d


@dataclass
class ExecutionResult:
    mode: str
    equity: pd.Series
    returns: pd.Series
    shares: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    costs: pd.Series
    gross_pnl: float
    net_pnl: float
    turnover: pd.Series
    reconciliation: dict
    spec: dict
    limitations: list
    ledger: Ledger
    book: rz.OrderBook
    daily_costs: pd.DataFrame = field(default_factory=pd.DataFrame)      # date x component, positive amounts in the base currency


def _sigma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return np.log(prices.where(prices > 0)).diff().rolling(window, min_periods=max(5, window // 2)).std().shift(1)


def execute_weights(weights: pd.DataFrame, prices: pd.DataFrame, spec: ExecutionSpec | None = None, *, volume: pd.DataFrame | None = None, open_prices: pd.DataFrame | None = None,
                    eligible: pd.DataFrame | None = None, actions: list[rz.CorporateAction] | None = None, borrow: rz.BorrowBook | None = None, closed: pd.DataFrame | None = None,
                    halted: pd.DataFrame | None = None, data_status: str = "unknown") -> ExecutionResult:
    """Execute a table of target weights (date x asset, the book wanted at the close of each date; NaN rows mean no decision) on unadjusted ``prices``.

    A decision at the close of day t is sent after that close and filled in the next session (``next_close`` at its close; ``next_open`` at its open with ``open_prices``). An order is capped by the
    participation limit of the prior average volume, rounded to whole lots, and the unfilled remainder waits for ``expiry_sessions`` sessions and then expires. A short is limited by ``borrow``; a recall forces
    a cover. ``eligible`` (date x asset, known before the order) stops new buys of an ineligible asset and forces the exit of one already held. ``closed``/``halted`` frames mark days on which an asset cannot
    be traded (it is valued at its last good mark). Corporate ``actions`` are applied at their ex-date from the previous close."""
    spec = spec or ExecutionSpec()
    if spec.delay < 1:
        raise ValueError("delay must be at least one session: an order decided at a close cannot fill at that close")
    if spec.sizing not in ("decision_close", "fill_time"):
        raise ValueError("sizing must be 'decision_close' or 'fill_time'")
    if spec.convention == "next_open" and open_prices is None:
        raise ValueError("the next_open convention needs open_prices; there is no opening price to fill at")
    if spec.convention == "same_close_auction":
        raise ValueError("a same-bar fill needs an explicit auction model; use next_close or next_open")
    dates, assets = pd.DatetimeIndex(prices.index), list(prices.columns)
    reg = InstrumentRegistry([Instrument(instrument_id=a, asset_class="equity", currency="USD", lot_size=1.0) for a in assets])
    ledger = Ledger(reg, "USD", {"USD": spec.capital}, start=dates[0])
    book = rz.OrderBook()
    adv_sh = volume.rolling(spec.adv_window, min_periods=spec.adv_window).mean().shift(1) if volume is not None else None
    sigma = _sigma(prices, spec.sigma_window)
    closed_f = closed.reindex(index=dates, columns=assets).fillna(False).astype(bool) if closed is not None else pd.DataFrame(False, index=dates, columns=assets)
    halted_f = halted.reindex(index=dates, columns=assets).fillna(False).astype(bool) if halted is not None else pd.DataFrame(False, index=dates, columns=assets)
    by_date: dict = {}
    for a in rz.latest_versions(actions or []):
        by_date.setdefault(a.ex_date, []).append(a)
    shares = pd.DataFrame(0.0, index=dates, columns=assets)
    equity = pd.Series(np.nan, index=dates)
    last_mark: dict[str, float] = {}
    pending: dict[str, list] = {a: [] for a in assets}                # asset -> its decisions in flight: {order, born, first, shares (target position), weight}
    order_rows, gross, traded_notional = [], 0.0, pd.Series(0.0, index=dates)
    div_by_date: dict = {}
    prev_shares = pd.Series(0.0, index=assets)
    prev_px: dict[str, float] = {}
    for i, t in enumerate(dates):
        px = prices.loc[t]
        # --- corporate actions at the ex-date, valued from the previous close
        split_ratio = {a: 1.0 for a in assets}
        for act in by_date.get(t, []):
            a = act.security_id
            if act.type == "split":
                split_ratio[a] = float(act.payload["ratio"])
            if act.type == "dividend":
                div_by_date[t] = div_by_date.get(t, 0.0) + ledger.quantity(a) * float(act.payload["amount"])
            rz.apply_action(ledger, act, last_mark.get(a, float(px.get(a, np.nan))), float(px.get(act.payload.get("successor") or act.payload.get("new_security"), np.nan)) if act.type in ("merger_stock", "spinoff") else None)
        # --- marks, then financing and borrow on what was held overnight (before today's fills)
        for a in assets:
            if np.isfinite(px[a]) and not closed_f.loc[t, a] and not halted_f.loc[t, a]:
                last_mark[a] = float(px[a])
        marks = {a: last_mark[a] for a in assets if a in last_mark}
        ledger.revalue(t, marks)
        if i > 0:
            dt = 1.0 / spec.periods_per_year
            ledger.accrue_interest(t, dt, {"USD": spec.cash_rate}, {"USD": spec.cash_rate + spec.financing_spread})
            if borrow is not None:
                rates = {a: borrow.rate(a, t) for a in assets if ledger.quantity(a) < 0}
                ledger.accrue_borrow(t, dt, {k: v for k, v in rates.items() if np.isfinite(v)}, marks)
        # --- fills of the orders decided on an earlier close. An order carries a TARGET POSITION; when its lag has elapsed it is the gap to that target that is traded, so decisions that overlap in time
        # (a lag of two sessions with a decision every day) each execute at their own session, and a newer decision replaces what an older one left unfilled.
        for a in assets:
            live = [q for q in pending[a] if q["order"].state in ("pending", "partial")]
            active = [q for q in live if q["first"] <= i]
            if not active:
                pending[a] = live
                continue
            cur = max(active, key=lambda q: q["born"])                  # the newest decision whose lag has elapsed
            for q in live:
                if q is not cur and q["born"] < cur["born"]:
                    book.cancel(q["order"].order_id, t, "replaced by a newer decision")
            pending[a] = [q for q in live if q is cur or q["born"] > cur["born"]]
            order = cur["order"]
            waited = i - cur["first"] + 1                               # fill opportunities used so far, this one included

            def drop(expire: bool, cur=cur, a=a):
                if expire:
                    book.expire(cur["order"].order_id, t)
                pending[a] = [q for q in pending[a] if q is not cur]

            fill_px = (open_prices.loc[t, a] if spec.convention == "next_open" else px[a])
            tradable = np.isfinite(fill_px) and not closed_f.loc[t, a] and not halted_f.loc[t, a]
            if not tradable:
                if waited >= spec.expiry_sessions:
                    drop(True)
                continue
            if order.state == "pending" and order.filled == 0.0:
                tgt_now = cur["shares"] if spec.sizing == "decision_close" else float(np.floor(abs(cur["weight"]) * ledger.equity() / float(px[a]) / spec.lot) * spec.lot * np.sign(cur["weight"]))
                gap = float(tgt_now - ledger.quantity(a))
                if gap == 0.0:
                    book.cancel(order.order_id, t, "the target position is already reached")
                    drop(False)
                    continue
                if np.sign(gap) != np.sign(order.quantity):                                   # the gap changed side while the order waited: the same decision, a new order
                    book.cancel(order.order_id, t, "the gap to the target changed side before the lag elapsed; re-issued")
                    order = book.submit(a, gap, t)
                    cur["order"] = order
                elif gap != order.quantity:
                    book.amend(order.order_id, gap, t, "re-sized to the gap to the target position when the lag elapsed")
            want = order.remaining
            adv = float(adv_sh.loc[t, a]) if adv_sh is not None else float("nan")
            lim = rz.fill_with_limits(want, adv if adv_sh is not None else float("inf"), spec.max_participation if adv_sh is not None else None, spec.lot)
            qty = lim["filled"]
            note = lim.get("reason", "")
            if qty < 0 and borrow is not None:
                held = ledger.quantity(a)
                new_short = max(-(held + qty), 0.0) - max(-held, 0.0)             # the extra shares that would be sold short
                if new_short > 0:
                    allowed, why = borrow.allowed_short(a, t, max(-held, 0.0), new_short)
                    if allowed < new_short:
                        qty = qty + (new_short - allowed)                          # sell less (qty is negative)
                        note = why
            if qty == 0:
                if waited >= spec.expiry_sessions:
                    drop(True)
                continue
            mid = float(fill_px)
            sgn = 1.0 if qty > 0 else -1.0
            imp = rz.impact_bps(spec.impact, abs(qty), adv if adv_sh is not None else float("nan"), float(sigma.loc[t, a]) if np.isfinite(sigma.loc[t, a]) else float("nan"))
            imp_bps = 0.0 if not np.isfinite(imp["bps"]) else imp["bps"]
            spread_p, impact_p, slip_p = mid * spec.half_spread_bps / 1e4, mid * imp_bps / 1e4, mid * spec.slippage_bps / 1e4
            price = mid + sgn * (spread_p + impact_p + slip_p)
            fee = abs(qty) * price * spec.commission_bps / 1e4
            book.fill(order.order_id, f"{order.order_id}@{t.date()}", qty, t)
            ledger.apply_fill(t, Fill(t, a, qty, price, order_id=order.order_id, fee=fee, spread_price=spread_p, impact_price=impact_p, strategy="ml"), mid)
            traded_notional[t] += abs(qty) * mid
            if order.state == "filled":
                drop(False)
            elif waited >= spec.expiry_sessions:
                drop(True)
            if note:
                order.reason = note
        equity.iloc[i] = ledger.equity()
        shares.loc[t] = [ledger.quantity(a) for a in assets]
        # --- independent gross P&L from the holdings and the marks
        for a in assets:
            q0 = prev_shares[a] * split_ratio[a]
            if q0 and a in prev_px and np.isfinite(px[a]) and not closed_f.loc[t, a] and not halted_f.loc[t, a]:
                gross += q0 * float(px[a]) - prev_shares[a] * prev_px[a]
        # --- recalls force a cover
        if borrow is not None:
            for a in assets:
                held = ledger.quantity(a)
                if held < 0:
                    over = -held - max(borrow.available(a, t) - borrow.recalled(a, t), 0.0) if np.isfinite(borrow.available(a, t)) else 0.0
                    if over > 0 and np.isfinite(px[a]):
                        cover = float(np.ceil(over))
                        mid = float(px[a])
                        ledger.apply_fill(t, Fill(t, a, cover, mid * (1 + (spec.half_spread_bps + spec.slippage_bps) / 1e4), fee=cover * mid * spec.commission_bps / 1e4,
                                                  spread_price=mid * spec.half_spread_bps / 1e4, tags=("recall",), strategy="ml"), mid)
                        order_rows.append({"order_id": -1, "asset": a, "decided": t, "quantity": cover, "state": "forced_recall", "reason": f"borrow recalled: {over:g} shares returned"})
                        traded_notional[t] += cover * mid
                        shares.loc[t] = [ledger.quantity(x) for x in assets]
                        equity.iloc[i] = ledger.equity()
        prev_shares = shares.loc[t].copy()
        prev_px = {a: last_mark[a] for a in last_mark}
        # --- the decision at this close
        if t in weights.index and weights.loc[t].notna().any():
            w = weights.loc[t].reindex(assets).fillna(0.0)
            eq = ledger.equity()
            for a in assets:
                if not np.isfinite(px[a]) or px[a] <= 0:
                    continue
                ok = True if eligible is None else bool(eligible.reindex(index=dates, columns=assets).fillna(False).loc[t, a])
                coming = sum(q["order"].remaining for q in pending[a] if q["order"].state in ("pending", "partial") and q["first"] > i)      # orders of earlier decisions that fill before this one can
                target_w = w[a]
                if not ok:                                                     # an ineligible asset is exited, never added to: what the strategy asked for beyond the book is rejected on the record
                    wanted = float(np.floor(abs(target_w) * eq / px[a] / spec.lot) * spec.lot * np.sign(target_w))
                    have = ledger.quantity(a) + coming
                    ask = wanted - have
                    if abs(wanted) > abs(have) + 1e-12 and abs(ask) * px[a] >= spec.min_trade_weight * eq:
                        r = book.submit(a, float(ask), t)
                        book.reject(r.order_id, t, "not eligible on this date (membership, liquidity or listing age)")
                        order_rows.append({"order_id": r.order_id, "asset": a, "decided": t, "quantity": float(ask), "state": "rejected", "reason": r.reason})
                    target_w = 0.0
                tgt = float(np.floor(abs(target_w) * eq / px[a] / spec.lot) * spec.lot * np.sign(target_w))
                delta = tgt - (ledger.quantity(a) + coming)
                if delta == 0.0 or abs(delta) * px[a] < spec.min_trade_weight * eq:
                    continue
                o = book.submit(a, float(delta), t)
                pending[a].append({"order": o, "born": i, "first": i + spec.delay, "shares": tgt, "weight": float(target_w)})
                order_rows.append({"order_id": o.order_id, "asset": a, "decided": t, "quantity": float(delta), "state": "pending", "reason": ""})
    # --- results
    rets = equity.pct_change().fillna(0.0)
    costs = rz.itemize(ledger)
    daily = rz.daily_costs(ledger)
    total_pnl = float(equity.iloc[-1] - spec.capital)
    divs = float(sum(div_by_date.values()))
    gross_total = gross + divs
    rec = rz.reconcile_net(gross_total + _other_flows(ledger), costs, total_pnl, tolerance=max(1e-6 * spec.capital, 1e-4))
    led = reconcile(ledger)
    rec.update({"ledger_cash_gap": {k: float(v) for k, v in led.cash_gap.items()}, "ledger_equity_gap": float(led.equity_gap), "ledger_identity_violations": int(led.identity_violations),
                "ledger_ok": bool(led.ok) if hasattr(led, "ok") else bool(abs(led.equity_gap) < 1e-6 and led.identity_violations == 0)})
    orders = book.frame()
    if order_rows:
        extra = pd.DataFrame([r for r in order_rows if r["order_id"] == -1])
        orders = pd.concat([orders, extra], ignore_index=True) if len(extra) else orders
    limits = []
    if "simulated" in data_status or "synthetic" in data_status:
        limits.append(f"the data is {data_status}: this exercises the execution machinery, not a market")
    if spec.convention == "next_close":
        limits.append("fills are at the next session's closing price with a modelled spread; the opening auction and intraday liquidity are not represented")
    if volume is None:
        limits.append("no volume: participation limits and impact were not applied (capacity is NOT established)")
    elif spec.max_participation is None:
        limits.append("no participation limit was applied: every order was assumed to fill in full at its session's close (capacity is NOT established)")
    if spec.impact.label() != "measured":
        limits.append("the impact coefficient is an assumption, not a measured cost")
    if borrow is None:
        limits.append("no borrow records: shorts were not constrained by availability and pay no borrow fee")
    return ExecutionResult("realistic_cash_equity", equity, rets, shares, orders, ledger.fills_frame(), costs, gross_total, total_pnl, traded_notional / equity.shift(1).fillna(spec.capital), rec, spec.describe(), limits, ledger,
                           book, daily.reindex(dates).fillna(0.0))


def _other_flows(ledger) -> float:
    """P&L from flows that are not price moves on held shares and not costs: interest earned on cash and the corporate-action entries other than dividends (already in the gross)."""
    by = ledger.pnl_by("category")
    return float(by.get("interest", 0.0) + by.get("corporate_action", 0.0) + by.get("lifecycle_settlement", 0.0))


def trading_diagnostics(result: ExecutionResult, prices: pd.DataFrame | None = None, volume: pd.DataFrame | None = None) -> dict:
    """The block that goes into a report: the mode and assumptions, gross and net, costs by component, turnover, fill states, reconciliation, and the limitations that qualify the result."""
    fills = result.fills
    states = result.orders["state"].value_counts().to_dict() if len(result.orders) else {}
    return {"mode": result.mode, "assumptions": result.spec, "gross_pnl": result.gross_pnl, "net_pnl": result.net_pnl, "costs": {k: float(v) for k, v in result.costs.items()},
            "cost_total": float(result.costs.sum()), "ann_turnover": float(result.turnover.mean() * result.spec["periods_per_year"]), "n_fills": int(len(fills)), "order_states": {k: int(v) for k, v in states.items()},
            "reconciliation": result.reconciliation, "limitations": result.limitations, "end_equity": float(result.equity.iloc[-1])}


# ------------------------------------------------------------------------------------------------------------------ the evaluation-mode adapter used by the experiment pipeline
SETTING_KEYS = {"mode", "capital", "convention", "lot", "max_participation", "adv_window", "commission_bps", "half_spread_bps", "slippage_bps", "impact", "sigma_window", "min_trade_weight", "expiry_sessions",
                "cash_rate", "financing_spread", "periods_per_year", "sizing", "capital_levels", "min_adv_dollars", "min_listing_days", "shorts"}
IMPACT_KEYS = {"coefficient", "max_participation", "calibration", "fallback"}


def validate_settings(settings: dict) -> None:
    """Refuse an ``execution`` section that cannot mean what it says. The lag is not a setting: it is the dataset's lag plus the backtest's extra lag, applied once, here."""
    if not isinstance(settings, dict):
        raise ValueError("execution must be a mapping")
    unknown = set(settings) - SETTING_KEYS
    if "delay" in unknown or "lag" in unknown:
        raise ValueError("the execution delay is not a setting: it is dataset.lag + backtest.extra_lag, applied once, by the adapter (setting it here would lag the order twice)")
    if unknown:
        raise ValueError(f"unknown execution settings {sorted(unknown)}; allowed: {sorted(SETTING_KEYS)}")
    mode = settings.get("mode", "realistic_cash_equity")
    if mode not in MODES:
        raise ValueError(f"execution.mode must be one of {MODES}")
    if mode == "vectorized_legacy":
        return
    if settings.get("convention", "next_close") != "next_close":
        raise ValueError("execution.convention must be next_close here: there are no opening prices in the dataset, and a same-bar fill needs an explicit auction model")
    if settings.get("sizing", "decision_close") not in ("decision_close", "fill_time"):
        raise ValueError("execution.sizing must be decision_close or fill_time")
    if settings.get("shorts", "unconstrained") not in ("unconstrained", "refuse"):
        raise ValueError("execution.shorts must be unconstrained (no borrow model: said in the report and blocking the claim) or refuse (strict validation: a new short without a borrow record is refused)")
    for k in ("capital", "adv_window", "sigma_window", "expiry_sessions", "lot", "periods_per_year"):
        if k in settings and not float(settings[k]) > 0:
            raise ValueError(f"execution.{k} must be positive")
    for k in ("commission_bps", "half_spread_bps", "slippage_bps", "min_trade_weight", "cash_rate", "financing_spread", "min_adv_dollars", "min_listing_days"):
        if settings.get(k) is not None and float(settings[k]) < 0:
            raise ValueError(f"execution.{k} must not be negative")
    mp = settings.get("max_participation")
    if mp is not None and not 0 < float(mp) <= 1:
        raise ValueError("execution.max_participation must be in (0, 1], or null for no limit (which is then stated as a limitation)")
    imp = settings.get("impact", {})
    if not isinstance(imp, dict) or set(imp) - IMPACT_KEYS:
        raise ValueError(f"execution.impact allows only {sorted(IMPACT_KEYS)}")
    if imp.get("fallback", "block") not in ("block", "conservative"):
        raise ValueError("execution.impact.fallback must be block or conservative")
    if "capital_levels" in settings and not (isinstance(settings["capital_levels"], (list, tuple)) and settings["capital_levels"] and all(float(c) > 0 for c in settings["capital_levels"])):
        raise ValueError("execution.capital_levels must be a non-empty list of positive amounts, declared before the run")


def spec_from_settings(settings: dict, bt_cfg: dict, delay: int) -> ExecutionSpec:
    """The adapter's spec for a configuration: the costs default to the backtest section's own (``cost_bps`` is the commission, ``slippage_bps`` the price concession), so that one specification feeds both
    engines; everything else has the adapter's defaults and is reported with the result."""
    validate_settings(settings)
    imp = settings.get("impact", {})
    impact = rz.ImpactSpec(**{k: v for k, v in imp.items() if k in IMPACT_KEYS})
    kw = {k: settings[k] for k in ("capital", "lot", "max_participation", "adv_window", "sigma_window", "min_trade_weight", "expiry_sessions", "cash_rate", "financing_spread", "periods_per_year", "sizing")
          if k in settings}
    return ExecutionSpec(**kw, convention="next_close", delay=int(delay), commission_bps=float(settings.get("commission_bps", bt_cfg.get("cost_bps", 0.0))),
                         half_spread_bps=float(settings.get("half_spread_bps", 0.0)), slippage_bps=float(settings.get("slippage_bps", bt_cfg.get("slippage_bps", 0.0))), impact=impact)


def decision_weights(weights: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, bt_cfg: dict, lag_total: int) -> pd.DataFrame:
    """The engine's pre-processing of target weights, with the lag left to the adapter: weights outside the investable set are zero, the book is scaled to the volatility target the engine would use, and
    only the rebalance dates are kept. Row ``d`` holds the target DECIDED at the close of ``d`` for a rebalance ``lag_total`` sessions later (NaN where nothing is scheduled)."""
    if lag_total < 1:
        raise ValueError("realistic execution needs a lag of at least one session (dataset.lag + backtest.extra_lag >= 1): a decision made from a close cannot trade at that close without an auction model")
    w = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    if investable is not None:
        mask = investable.reindex_like(w).fillna(False)
        w = w.where(mask, 0.0).where(mask.sum(axis=1) >= 1, 0.0)
    tv = bt_cfg.get("target_vol")
    if tv not in (None, 0):
        w, _ = vol_target_scaling(w, returns.reindex(w.index).reindex(columns=w.columns), float(tv), 63, float(bt_cfg.get("max_leverage", 1.5)), 252)
    scheduled = restrict_to_rebalances(apply_execution_lag(w, lag_total), w.index, str(bt_cfg["rebalance"]))                   # the engine's own lag and schedule helpers
    return scheduled.shift(-lag_total)                                                                                         # back to the date of the decision


def to_backtest_result(res: ExecutionResult, name: str, prices: pd.DataFrame) -> BacktestResult:
    """The ledger's result in the form the rest of the pipeline reads: net and gross daily returns, the cost as a fraction of the previous equity, the book held at each close and the turnover."""
    prev_eq = res.equity.shift(1)
    cost_rate = (res.daily_costs.sum(axis=1) / prev_eq).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    net = res.returns.copy()
    held = (res.shares * prices.reindex(index=res.shares.index, columns=res.shares.columns)).div(res.equity, axis=0).fillna(0.0)
    cfg = {"mode": res.mode, "signal_lag": res.spec["delay"], "rebalance": "per the backtest section", "weight_drift": False, "cost_model": "itemised in the ledger (see report['execution'])"}
    return BacktestResult(name=name, gross_returns=net + cost_rate, net_returns=net, costs=cost_rate, weights=held, trades=held.diff().fillna(held), turnover=res.turnover.copy(), vol_scalar=None, config=cfg)


def _cost_cum(r: pd.Series) -> float:
    return float((1.0 + r.fillna(0.0)).prod() - 1.0)


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1) if len(r) > 2 else float("nan")
    return float(r.mean() / sd * np.sqrt(252.0)) if sd and sd > 0 else float("nan")


COMMON_SUBSET = ("daily-or-scheduled rebalance dates, the same signal lag applied once, the same target weights (investable mask and volatility target included), close-to-close prices, "
                 "costs proportional to traded notional")
OUTSIDE_COMMON_SUBSET = ("between rebalances the vectorized engine renormalises the drifted book to constant gross exposure while the ledger lets the shares drift", "whole shares and lots, and the cash they leave",
                         "participation limits and unfilled or expired orders", "impact beyond the linear cost, when ADV is available", "financing of borrowed cash and interest on cash",
                         "an order sized in shares at the decision close when sizing is decision_close")


def legacy_block(cfg: dict, recorded: bool = True) -> dict:
    """What a vectorized_legacy result assumes and cannot support: the block of every run that did not ask for the realistic mode, and what an experiment saved before the mode existed is read as."""
    bt, lag = cfg["backtest"], int(cfg["dataset"]["lag"]) + int(cfg["backtest"].get("extra_lag", 0))
    return {"mode": "vectorized_legacy", "recorded": bool(recorded), "engine": "src/backtest/engine.py (BacktestEngine)",
            "assumptions": {"signal_lag_sessions": lag, "rebalance": bt["rebalance"], "weight_drift": "yes, renormalised to constant gross exposure", "cost_model": f"linear: {float(bt['cost_bps']) + float(bt['slippage_bps']):g} bps one way on traded weight",
                            "target_vol": bt.get("target_vol"), "fills": "the book is held from the close of the decision day plus the lag; there are no orders"},
            "price_convention": "research closes (vendor-adjusted where the source adjusts them); no separate executable prices",
            "not_modelled": ["order lifecycle and partial fills", "participation and volume limits", "market impact", "cash, financing and borrow", "corporate actions as events", "whole shares and lots", "margin, funding and settlement"],
            "cannot_support": ["capacity", "short-selling feasibility", "executable-cost realism"],
            "note": None if recorded else "this experiment was saved before execution modes were recorded; it is read as vectorized_legacy"}


def _capacity(decisions: pd.DataFrame, prices: pd.DataFrame, volume: pd.DataFrame | None, spec: ExecutionSpec, levels) -> dict:
    if volume is None:
        return {"status": "not_established", "reason": "no volume in the data: capacity and impact are not assumed", "levels": []}
    held = decisions.dropna(how="all")
    held = held[held.abs().sum(axis=1) > 0]                                        # a flat target (the days at the end that have no scores) says nothing about capacity
    if held.empty:
        return {"status": "not_established", "reason": "no decision held a position", "levels": []}
    day = held.index[-1]
    adv = rz.pit_adv(volume, prices, spec.adv_window)
    px = prices.loc[day]
    sig = _sigma(prices, spec.sigma_window).loc[day]
    cur = rz.capacity_curve(held.iloc[-1].abs(), px, adv.loc[day], levels, spec.max_participation if spec.max_participation is not None else 1.0, spec.impact, sig.fillna(0.01))
    return {"status": "measured_on_last_decision", "date": str(pd.Timestamp(day).date()), "levels": cur.to_dict("records"),
            "note": "the share of the last target that one session's participation limit lets through at each predeclared capital, and its modelled impact; the impact coefficient is an assumption unless calibrated"}


@dataclass
class RealisticRun:
    result: ExecutionResult
    backtest: BacktestResult
    block: dict
    decisions: pd.DataFrame
    inputs: dict


def realistic_run(weights: pd.DataFrame, prices: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, bt_cfg: dict, lag_total: int, settings: dict, *, volume: pd.DataFrame | None = None,
                  data_status: str = "unknown", name: str = "ml_strategy", legacy: BacktestResult | None = None, membership: rz.Membership | None = None, actions: list | None = None,
                  borrow: rz.BorrowBook | None = None, closed: pd.DataFrame | None = None, halted: pd.DataFrame | None = None) -> RealisticRun:
    """Run target weights through the adapter on the SAME inputs the vectorized engine got, and describe the result: mode, timeline, assumptions, gross and net, itemised costs, turnover, capacity,
    reconciliation, the comparison with the vectorized run on the shared subset, the capability report and the limitations."""
    spec = spec_from_settings(settings, bt_cfg, lag_total)
    decisions = decision_weights(weights, returns, investable, bt_cfg, lag_total)
    px = prices.reindex(index=returns.index, columns=returns.columns)
    vol = None if volume is None else volume.reindex(index=px.index, columns=px.columns)
    eligible, screens = None, None
    min_adv, min_age = settings.get("min_adv_dollars"), settings.get("min_listing_days")
    member_ok = membership.eligible_tickers(px.index, list(px.columns)) if membership is not None else None
    if min_adv is not None or min_age is not None or member_ok is not None:         # point-in-time screens: everything they use was known before the order
        eligible, why = rz.pit_screens(px, vol, member_ok, window=spec.adv_window, min_adv=None if min_adv is None else float(min_adv), min_age=int(min_age or 0), halted=halted)
        reasons = why.to_numpy().ravel()
        screens = {"min_adv_dollars": min_adv, "min_listing_days": min_age, "adv_window": spec.adv_window, "membership": membership is not None,
                   "ineligible_asset_days": {r: int((reasons == r).sum()) for r in sorted({x for x in reasons if x})}, "asset_days": int(eligible.size)}
    refused_shorts = borrow is None and settings.get("shorts") == "refuse"
    if refused_shorts:                                                                  # strict validation: with no record of availability a new short cannot be opened
        borrow = rz.BorrowBook(pd.DataFrame(np.nan, index=px.index, columns=px.columns), strict=True)
    res = execute_weights(decisions, px, spec, volume=vol, eligible=eligible, actions=actions, borrow=borrow, closed=closed, halted=halted, data_status=data_status)
    bt = to_backtest_result(res, name, px)
    diag = trading_diagnostics(res, px, vol)
    block = {**diag, "mode": res.mode, "recorded": True,
             "timeline": {"convention": spec.convention, "decision": "the close of the decision day", "submission": "after that close", "fill": f"the close of the session {spec.delay} session(s) later, at a modelled price", "delay_sessions": spec.delay,
                          "delay_source": "dataset.lag + backtest.extra_lag, applied once, by the adapter (the vectorized engine's own lag is not applied on top)", "sizing": spec.sizing},
             "price_convention": "research closes (vendor-adjusted where the source adjusts them) are the marks and the fill reference; no separate executable series or corporate-action events were supplied",
             "gross_net": {"gross_pnl": res.gross_pnl, "net_pnl": res.net_pnl, "gross_return": res.gross_pnl / spec.capital, "net_return": res.net_pnl / spec.capital, "costs_total": float(res.costs.sum()),
                           "costs_pct_of_capital": float(res.costs.sum() / spec.capital)},
             "leverage": {"mean_gross": float(bt.weights.abs().sum(axis=1).mean()), "max_gross": float(bt.weights.abs().sum(axis=1).max()), "end_gross": float(bt.weights.abs().sum(axis=1).iloc[-1])},
             "equity": {"start": spec.capital, "before_costs": float(spec.capital + res.gross_pnl), "after_costs": float(res.equity.iloc[-1])},
             "capacity": _capacity(decisions, px, vol, spec, settings.get("capital_levels", [1e5, 1e6, 1e7, 1e8, 1e9])),
             "common_subset": {"shared": COMMON_SUBSET, "outside": list(OUTSIDE_COMMON_SUBSET)}}
    if legacy is not None:
        both = bt.net_returns.index.intersection(legacy.net_returns.index)
        both = both[(bt.weights.abs().sum(axis=1).reindex(both) > 0).cumsum() > 0] if len(both) else both
        block["comparison_with_vectorized"] = {"n_days": int(len(both)), "vectorized_total_return": _cost_cum(legacy.net_returns.reindex(both)), "realistic_total_return": _cost_cum(bt.net_returns.reindex(both)),
                                               "vectorized_sharpe": _sharpe(legacy.net_returns.reindex(both)), "realistic_sharpe": _sharpe(bt.net_returns.reindex(both)),
                                               "note": "the two are measured on the same weights and dates; they differ by the assumptions listed under common_subset.outside, not by an error in either"}
    caps = rz.capability_report(None, data_status=data_status, membership=membership, actions=actions, borrow=borrow, adapter_modes=MODES)
    caps["execution_fields"] = {"close": True, "open": False, "high": False, "low": False, "volume": vol is not None, "bid_ask": False, "marks": False}
    shorts = bool((res.shares < -1e-9).any().any())
    blockers = [b for b in caps["blockers"] if not b.startswith(("no volume", "no borrow"))]
    if borrow is not None and borrow.strict is False:
        caps["limitations"] = caps["limitations"] + [f"borrow availability and fees follow a labelled scenario (scenario fee {borrow.scenario_fee}), not records"]
    if vol is None:
        blockers.append("no volume: participation limits, capacity and impact cannot be measured (they are not assumed)")
    if shorts and (borrow is None or borrow.empty):
        blockers.append("no borrow records: short positions were held and their availability and fee are not evidenced (only a labelled scenario)")
    if refused_shorts:
        caps["limitations"] = caps["limitations"] + ["shorts were refused under strict validation (no borrow records): the book is long-only where the strategy asked for shorts"]
    else:
        caps["limitations"] = caps["limitations"] + ["no short position was held, so borrow availability and fees were not needed"]
    caps["blockers"] = blockers
    caps["realistic_execution_claim"] = "blocked" if blockers else ("limited" if caps["limitations"] else "supported")
    block["capability"], block["short_positions"], block["screens"] = caps, shorts, screens
    block["limitations"] = list(dict.fromkeys(res.limitations + caps["limitations"] + blockers))
    inputs = {"weights": pv.frame_hash(decisions), "prices": pv.frame_hash(px), "volume": pv.frame_hash(vol), "spec_hash": pv.digest(res.spec, 32)}
    return RealisticRun(res, bt, block, decisions, inputs)


def write_artifacts(run: RealisticRun, directory) -> dict:
    """Write the order, fill and journal tables, the equity path and a manifest to ``directory``, and return the manifest: every file with its row count and sha256, the input hashes, the specification hash
    and the reconciliation. The manifest's own hash is returned beside it so that a report can cite it."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    res = run.result
    journal = res.ledger.journal_frame()
    if len(journal) and "cash" in journal:
        journal = journal.assign(cash=journal["cash"].map(lambda c: json.dumps(c, sort_keys=True)))
    tables = {"orders": res.orders, "fills": res.fills, "journal": journal,
              "equity": pd.DataFrame({"equity": res.equity, "return": res.returns, "turnover": res.turnover}).join(res.daily_costs.add_prefix("cost_"), how="left")}
    files = {}
    for key, frame in tables.items():
        path = d / f"execution_{key}.parquet"
        frame.to_parquet(path)
        files[key] = {"file": path.name, "rows": int(len(frame)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"version": 1, "mode": res.mode, "files": files, "inputs": run.inputs, "specification": res.spec, "reconciliation": {k: res.reconciliation[k] for k in ("gross", "net", "gap", "reconciled", "ledger_ok") if k in res.reconciliation}}
    text = json.dumps(manifest, sort_keys=True, default=str, indent=1)
    (d / "execution_manifest.json").write_text(text, encoding="utf-8")
    return {**manifest, "manifest_sha256": hashlib.sha256(text.encode()).hexdigest()}
