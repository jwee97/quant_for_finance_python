"""The components of executable trading (spec v3, Release 4, P1-01 to P1-10, P1-12 to P1-14): who may be traded, what happened to them, when an order fills, what it costs, what limits it, and what the
data allows anyone to claim.

Nothing here is a new engine. The accounting is the platform's own ledger (``src/ledger``), the cost and liquidity models are ``src/engine/costs.py``, and ``execution.py`` drives them from a table of target
weights; this module holds the parts the vectorized research engine and the event-driven engine did not state: dated membership by persistent security id, versioned corporate actions applied to the ledger,
point-in-time liquidity, an explicit decision-to-fill timeline, itemized costs that cannot be counted twice, impact with its calibration status, order states with legal transitions, borrow with recalls,
capacity, mark policies for closed or halted markets, timestamped currency conversion, futures and crypto-derivative conventions, and the capability report that turns the actual data into the limits on a claim.

A missing input never becomes a free one: no liquidity is not infinite capacity, no borrow record is not free borrow, no quote is not a zero return, and no currency rate is not 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------ P1-01: dated membership by persistent security id
STATUSES = ("active", "delisted", "failed", "merged", "suspended")


@dataclass(frozen=True)
class Listing:
    """One security's life. ``security_id`` persists across ticker changes; a ticker is valid for ``[start, end)``. ``end`` None means still listed. ``status`` and ``successor`` say how it ended."""

    security_id: str
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp | None = None
    status: str = "active"
    successor: str | None = None
    note: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        if self.end is not None:
            object.__setattr__(self, "end", pd.Timestamp(self.end))
            if self.end <= self.start:
                raise ValueError(f"{self.security_id}: a listing must end after it starts")


class Membership:
    """Dated listings and index membership keyed by security id. ``coverage`` is ``complete`` only when the history includes the securities that left; otherwise ``survivorship`` states the limitation and no
    missing security is invented."""

    def __init__(self, listings: list[Listing], coverage: str = "partial", members: Mapping[str, list[tuple]] | None = None):
        if coverage not in ("complete", "partial", "none"):
            raise ValueError("coverage is complete, partial or none")
        self.listings, self.coverage, self.members = list(listings), coverage, dict(members or {})
        self._by_id: dict[str, list[Listing]] = {}
        for lst in self.listings:
            self._by_id.setdefault(lst.security_id, []).append(lst)

    def security_of(self, ticker: str, date) -> str | None:
        """The security a ticker meant on ``date`` (a reused or changed ticker does not point at the wrong company)."""
        d = pd.Timestamp(date)
        for lst in self.listings:
            if lst.ticker == ticker and lst.start <= d and (lst.end is None or d < lst.end):
                return lst.security_id
        return None

    def ticker_of(self, security_id: str, date) -> str | None:
        d = pd.Timestamp(date)
        for lst in self._by_id.get(security_id, []):
            if lst.start <= d and (lst.end is None or d < lst.end):
                return lst.ticker
        return None

    def listed(self, security_id: str, date) -> bool:
        return self.ticker_of(security_id, date) is not None

    def member(self, security_id: str, date) -> bool:
        """Listed, and (when membership periods are given for the security) inside one of them."""
        if not self.listed(security_id, date):
            return False
        periods = self.members.get(security_id)
        if periods is None:
            return True
        d = pd.Timestamp(date)
        return any(pd.Timestamp(a) <= d and (b is None or d < pd.Timestamp(b)) for a, b in periods)

    def eligible(self, dates, security_ids) -> pd.DataFrame:
        """A date x security boolean frame of who could be held on each date."""
        return pd.DataFrame({s: [self.member(s, d) for d in dates] for s in security_ids}, index=pd.DatetimeIndex(dates))

    def eligible_tickers(self, dates, tickers) -> pd.DataFrame:
        """A date x ticker boolean frame: the ticker meant a security that was listed and in the universe on that date. A ticker that never meant anything on a date is not eligible then."""
        rows = []
        for d in dates:
            row = {}
            for t in tickers:
                sid = self.security_of(t, d)
                row[t] = bool(sid is not None and self.member(sid, d))
            rows.append(row)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=list(tickers))

    def ending(self, security_id: str) -> Listing | None:
        """The listing that ended the security's life (delisted, failed, merged), or None."""
        ended = [lst for lst in self._by_id.get(security_id, []) if lst.status in ("delisted", "failed", "merged") and lst.end is not None]
        return max(ended, key=lambda lst: lst.end) if ended else None

    def survivorship(self) -> dict:
        n_end = sum(1 for lst in self.listings if lst.status in ("delisted", "failed", "merged"))
        limitation = None
        if self.coverage != "complete":
            limitation = ("the history of securities that left the universe is " + ("partial" if self.coverage == "partial" else "absent") +
                          ": results are conditional on survivors (survivorship bias) and are not corrected here, since missing securities are not invented")
        return {"coverage": self.coverage, "securities": len({lst.security_id for lst in self.listings}), "ended": n_end, "limitation": limitation}


# ------------------------------------------------------------------------------------------------------------------ P1-02: corporate actions
SUPPORTED_ACTIONS = ("split", "dividend", "merger_cash", "merger_stock", "spinoff", "delisting")


class UnsupportedCorporateAction(ValueError):
    """An action type the platform cannot account for: the claims that depend on it are blocked, not approximated."""


@dataclass(frozen=True)
class CorporateAction:
    """A versioned event. ``version`` rises when a vendor corrects it; the ledger applies the highest version of each ``(security_id, type, ex_date)``.

    ``payload`` by type: split ``{"ratio": 2.0}`` (new shares per old); dividend ``{"amount": 0.5}`` per share; merger_cash ``{"price": 60.0}``; merger_stock ``{"successor": "B", "ratio": 1.5}`` (successor shares per
    share); spinoff ``{"new_security": "C", "ratio": 0.1, "value_fraction": 0.2}`` (new shares per share and the share of the parent's value they carry); delisting ``{"recovery": 0.1}`` of the last mark."""

    security_id: str
    type: str
    ex_date: pd.Timestamp
    payload: dict = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "ex_date", pd.Timestamp(self.ex_date))


def latest_versions(actions: list[CorporateAction]) -> list[CorporateAction]:
    best: dict[tuple, CorporateAction] = {}
    for a in actions:
        key = (a.security_id, a.type, a.ex_date)
        if key not in best or a.version > best[key].version:
            best[key] = a
    return sorted(best.values(), key=lambda a: (a.ex_date, a.security_id, a.type))


def check_supported(actions: list[CorporateAction]) -> dict:
    """Which actions can be accounted for and which block the claims that depend on them."""
    unsupported = [a for a in actions if a.type not in SUPPORTED_ACTIONS]
    return {"supported": [a for a in actions if a.type in SUPPORTED_ACTIONS], "unsupported": unsupported,
            "blocked_claims": sorted({f"{a.security_id}: {a.type} on {a.ex_date.date()} is not supported, so results depending on holding it through that date are not claimed" for a in unsupported})}


def apply_action(ledger, action: CorporateAction, mark: float, successor_mark: float | None = None) -> None:
    """Apply one action to the platform's ledger. Quantities, cash and equity move exactly as the event says; the research price series is adjusted separately (``adjust_prices``)."""
    if action.type not in SUPPORTED_ACTIONS:
        raise UnsupportedCorporateAction(f"{action.type} is not supported ({', '.join(SUPPORTED_ACTIONS)} are)")
    iid, ts = action.security_id, action.ex_date
    qty = ledger.quantity(iid)
    if action.type == "split":
        ledger.split(ts, iid, float(action.payload["ratio"]), note=f"split {action.payload['ratio']:g}:1 (v{action.version})")
    elif action.type == "dividend":
        if qty:
            ccy = ledger._inst(iid).currency                                                              # noqa: SLF001
            ledger.post_cashflow(ts, "dividend", ccy, qty * float(action.payload["amount"]), iid, note=f"dividend {action.payload['amount']:g} per share (v{action.version})")
    elif action.type == "merger_cash":
        if qty:
            ccy = ledger._inst(iid).currency                                                              # noqa: SLF001
            ledger.settle_with_cash(ts, iid, {ccy: qty * float(action.payload["price"])}, mark, "corporate_action", f"cash merger at {action.payload['price']:g}")
    elif action.type == "delisting":
        if qty:
            ccy = ledger._inst(iid).currency                                                              # noqa: SLF001
            ledger.settle_with_cash(ts, iid, {ccy: qty * mark * float(action.payload.get("recovery", 0.0))}, mark, "corporate_action", f"delisting, recovery {action.payload.get('recovery', 0.0):.0%} of the last mark")
    elif action.type == "merger_stock":
        _exchange(ledger, ts, iid, action.payload["successor"], qty, mark, successor_mark, qty * float(action.payload["ratio"]), f"stock merger into {action.payload['successor']}")
    elif action.type == "spinoff":
        if qty and successor_mark:
            from ..ledger import Fill

            new_qty = qty * float(action.payload["ratio"])
            value_moved = new_qty * successor_mark
            ccy = ledger._inst(iid).currency                                                                  # noqa: SLF001
            ledger.revalue(ts, {iid: mark - value_moved / qty})                                              # the parent's price falls by the value that left with the spin-off shares
            ledger.apply_fill(ts, Fill(ts, action.payload["new_security"], new_qty, successor_mark, tags=("spinoff",), settlement=True), successor_mark)
            ledger.post_cashflow(ts, "corporate_action", ccy, value_moved, action.payload["new_security"], note="shares received in the spin-off (no cash paid)")        # the fill paid for them; they were free


def _exchange(ledger, ts, old: str, new: str, old_qty: float, old_mark: float, new_mark: float | None, new_qty: float, note: str) -> None:
    if not old_qty:
        return
    if not new_mark:
        raise ValueError(f"{note}: no successor mark on {ts.date()}; the exchange cannot be valued")
    from ..ledger import Fill

    ccy = ledger._inst(old).currency                                                                      # noqa: SLF001
    ledger.settle_with_cash(ts, old, {ccy: old_qty * old_mark}, old_mark, "corporate_action", note)
    ledger.apply_fill(ts, Fill(ts, new, new_qty, new_mark, tags=("merger",), settlement=True), new_mark)         # the cash from the old shares pays for the new ones: value is unchanged


def adjust_prices(close: pd.Series, actions: list[CorporateAction], total_return: bool = True) -> pd.Series:
    """The RESEARCH series, in the terms of the latest price: splits always, and (``total_return``) dividends reinvested at the ex-date close, so that the series' return over any stretch equals the return of holding the shares and collecting the cash. Executable prices are the unadjusted ones; cash distributions are booked as cash by the ledger, so using the adjusted
    series for execution and the ledger's dividends together would count them twice."""
    adj = close.astype(float).copy()
    factor = pd.Series(1.0, index=close.index)
    for a in sorted(latest_versions(actions), key=lambda x: x.ex_date, reverse=True):
        if a.ex_date not in close.index:
            continue
        i = close.index.get_loc(a.ex_date)
        if a.type == "split":
            factor.iloc[:i] = factor.iloc[:i] / float(a.payload["ratio"])
        elif a.type == "dividend" and total_return and i > 0:
            ex_close, d = float(close.iloc[i]), float(a.payload["amount"])
            factor.iloc[:i] = factor.iloc[:i] * ex_close / (ex_close + d)                                       # reinvested at the ex-date close: the day's total return is (P + D) / P_previous
    return adj * factor


# ------------------------------------------------------------------------------------------------------------------ P1-03: point-in-time liquidity and tradability
def pit_adv(volume: pd.DataFrame, price: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Average daily dollar volume known BEFORE each date: the mean of the previous ``window`` days, so the value on day t never includes day t or anything later. Needs the full window."""
    dollars = (volume * price).where(volume.notna() & price.notna())
    return dollars.rolling(window, min_periods=window).mean().shift(1)


def pit_screens(price: pd.DataFrame, volume: pd.DataFrame | None, membership_ok: pd.DataFrame | None = None, window: int = 20, min_adv: float | None = 1e6, min_age: int = 60,
                halted: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(eligible, reason)``: a date x asset boolean frame of what could be bought on each date, and the first reason an asset was not. Every input is known before the order: the listing age counts prior
    sessions, the liquidity is ``pit_adv``, membership and halts are as of the date. Missing liquidity is a reason, not a pass."""
    age = price.notna().cumsum().shift(1).fillna(0)
    adv = pit_adv(volume, price, window) if volume is not None else pd.DataFrame(np.nan, index=price.index, columns=price.columns)
    reasons = np.full(price.shape, "", dtype=object)
    ok = pd.DataFrame(True, index=price.index, columns=price.columns)

    def mark(mask: pd.DataFrame, why: str) -> None:
        nonlocal ok
        reasons[(mask & ok).to_numpy()] = why                                                                   # the FIRST reason an asset was not eligible
        ok = ok & ~mask

    if membership_ok is not None:
        mark(~membership_ok.reindex(index=price.index, columns=price.columns).fillna(False).astype(bool), "not a member or not listed")
    if halted is not None:
        mark(halted.reindex(index=price.index, columns=price.columns).fillna(False).astype(bool), "halted or suspended")
    mark(price.isna(), "no price")
    mark(age < min_age, f"listed fewer than {min_age} sessions")
    if min_adv is not None:                                                                                      # no liquidity screen was asked for: none is applied (and none is assumed passed)
        mark(adv.isna(), "no liquidity history (not assumed liquid)")
        mark(adv < min_adv, f"prior average dollar volume below {min_adv:,.0f}")
    return ok, pd.DataFrame(reasons, index=price.index, columns=price.columns)


# ------------------------------------------------------------------------------------------------------------------ P1-04: the decision-to-fill timeline
CONVENTIONS = ("next_close", "next_open", "same_close_auction")


@dataclass(frozen=True)
class Timeline:
    """When an order is decided, submitted and filled, and from which price. ``decision_at`` is the close of the decision day (information cutoff); ``submitted_at`` the next opportunity to send an order;
    ``filled_at`` the execution session; ``price_source`` names the price used."""

    decision_at: pd.Timestamp
    submitted_at: pd.Timestamp
    filled_at: pd.Timestamp
    price_source: str
    cutoff: str


def schedule(decision_dates, sessions, convention: str = "next_close", auction_model: str | None = None) -> list[Timeline]:
    """The timeline of each decision on a real list of sessions. The default is the first realistically supported later opportunity (``next_close``: submitted after the decision close, filled at the next
    session's close; ``next_open``: filled at the next open). A same-bar fill (``same_close_auction``) is refused unless an auction model is named: a decision made from a close cannot trade at that close
    without one. A decision with no later session has no timeline."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    if convention == "same_close_auction" and not auction_model:
        raise ValueError("a same-bar fill needs an explicit intrabar information or auction model (auction_model=...): a decision made from the close cannot trade at that close without one")
    s = pd.DatetimeIndex(sessions)
    out = []
    for d in pd.DatetimeIndex(decision_dates):
        i = s.searchsorted(d, side="right") if convention != "same_close_auction" else s.searchsorted(d)
        if i >= len(s):
            continue
        fill = s[i]
        out.append(Timeline(d, d if convention == "same_close_auction" else s[i], fill, {"next_close": "close", "next_open": "open", "same_close_auction": f"closing auction ({auction_model})"}[convention],
                            "orders are sent after the decision close" if convention != "same_close_auction" else f"decision inside the auction cutoff, modelled by {auction_model}"))
    return out


def lag_audit(dataset_lag: int, engine_lag: int, extra_lag: int = 0, convention: str = "next_close") -> dict:
    """How many sessions separate a decision from the return it earns, layer by layer, so that a lag is applied once: the target's entry lag is a property of the LABEL, the engine's lag (plus any extra delay)
    is the execution; the position a decision creates earns from the session after it was filled. ``double_lagged`` flags a configuration that applies a delay in both the target and the engine to the same fill."""
    execution = engine_lag + extra_lag
    return {"label_entry_lag": dataset_lag, "execution_lag": execution, "convention": convention, "first_return_earned_after_sessions": execution,
            "double_lagged": bool(dataset_lag > 0 and execution > dataset_lag and extra_lag == 0 and engine_lag != dataset_lag),
            "note": "the label starts dataset.lag sessions after the decision and the engine fills engine_lag + extra_lag sessions after it; the two must describe the same fill"}


# ------------------------------------------------------------------------------------------------------------------ P1-05: itemized costs that reconcile
COST_KINDS = {"commission": "explicit_fee", "exchange_fee": "explicit_fee", "spread": "price_concession", "slippage": "price_concession", "impact": "price_concession", "borrow": "financing",
              "financing": "financing"}


def validate_cost_spec(spec: Mapping[str, Mapping]) -> None:
    """Refuse a cost specification that would charge one thing twice. ``spec`` maps a component to ``{"bps": ..., "covers": [...]}``; a price concession (spread, slippage, impact) that already covers a named
    component must not also appear as an explicit fee of that name."""
    explicit = {k for k, v in spec.items() if COST_KINDS.get(k) == "explicit_fee" and v.get("bps", 0)}
    for name, v in spec.items():
        if COST_KINDS.get(name) == "price_concession" and v.get("bps", 0):
            both = explicit & set(v.get("covers", []))
            if both:
                raise ValueError(f"{name} already covers {sorted(both)}, which is also charged as an explicit fee: the same cost would be counted twice")


def itemize(ledger) -> pd.Series:
    """Costs by component from the ledger's journal, as positive amounts in the base currency: commission and exchange fees (``fee``), ``spread``, ``slippage``, ``impact``, ``borrow``, ``financing``."""
    by = ledger.pnl_by("category")
    return pd.Series({k: -float(by.get(k, 0.0)) for k in ("fee", "spread", "slippage", "impact", "borrow", "financing")})


def daily_costs(ledger) -> pd.DataFrame:
    """The same components by date: positive amounts in the base currency, from the journal's entries on that date. Their sum over all dates is ``itemize``."""
    j = ledger.journal_frame()
    cols = ["fee", "spread", "slippage", "impact", "borrow", "financing"]
    if j.empty:
        return pd.DataFrame(columns=cols, dtype=float)
    j = j[j["category"].isin(cols)]
    if j.empty:
        return pd.DataFrame(columns=cols, dtype=float)
    out = -j.groupby([pd.to_datetime(j["ts"]).dt.normalize(), "category"])["pnl"].sum().unstack(fill_value=0.0)
    return out.reindex(columns=cols, fill_value=0.0)


def reconcile_net(gross_pnl: float, items: pd.Series, net_pnl: float, tolerance: float = 1e-6) -> dict:
    """Gross P&L less the itemized charges must equal net P&L. Returns the gap; ``reconciled`` is False beyond ``tolerance``."""
    gap = float(gross_pnl - items.sum() - net_pnl)
    return {"gross": float(gross_pnl), "costs": {k: float(v) for k, v in items.items()}, "net": float(net_pnl), "gap": gap, "reconciled": bool(abs(gap) <= tolerance)}


# ------------------------------------------------------------------------------------------------------------------ P1-06: impact and its calibration
@dataclass(frozen=True)
class ImpactSpec:
    """Square-root impact: price move = ``coefficient`` x sigma x sqrt(participation), participation = quantity / ADV in the SAME unit (shares or dollars), sigma the daily volatility as a fraction. The
    calibration says where ``coefficient`` came from; ``assumed`` is never reported as a measured cost."""

    coefficient: float = 0.5
    max_participation: float = 1.0
    calibration: Mapping = field(default_factory=lambda: {"status": "assumed", "note": "an assumed coefficient, not calibrated to executions"})
    fallback: str = "block"                                              # with no ADV or sigma: "block" (no estimate) or "conservative" (use the cap)

    def label(self) -> str:
        c = self.calibration
        if c.get("status") == "measured" and all(c.get(k) is not None for k in ("sample", "date", "method", "error")):
            return "measured"
        return "assumed (uncalibrated)"


def _engine_model(spec: ImpactSpec):
    from ..engine.costs import ImpactModel                                         # the platform's own cost module: the square-root law is its arithmetic, not a second implementation

    return ImpactModel(kind="sqrt", y=spec.coefficient, max_participation=spec.max_participation)


def impact_bps(spec: ImpactSpec, quantity: float, adv: float, sigma: float) -> dict:
    """Impact in basis points of price for an order of ``quantity`` against ``adv`` (same unit) at daily volatility ``sigma``. The arithmetic is ``src.engine.costs.ImpactModel`` (price move = coefficient x sigma x
    sqrt(participation), participation capped); what this adds is the policy around it: zero quantity costs nothing, a missing ADV or volatility follows the fallback (the engine's own model would return zero
    impact there, which is never an honest default), and the calibration decides whether the cost may be called measured."""
    if quantity == 0:
        return {"bps": 0.0, "status": "zero_size", "label": spec.label()}
    model = _engine_model(spec)
    if not (np.isfinite(adv) and adv > 0 and np.isfinite(sigma) and sigma >= 0):
        if spec.fallback == "conservative" and np.isfinite(sigma):
            part = spec.max_participation
            return {"bps": float(model.impact_price(1.0, part, 1.0, sigma) * 1e4), "status": "fallback_conservative", "label": spec.label(), "participation": part}
        return {"bps": float("nan"), "status": "no_estimate", "label": spec.label(), "reason": "no ADV or volatility: impact is not assumed to be zero"}
    part = min(abs(quantity) / adv, spec.max_participation)
    return {"bps": float(model.impact_price(1.0, quantity, adv, sigma) * 1e4), "status": "capped" if abs(quantity) / adv > spec.max_participation else "ok", "label": spec.label(), "participation": float(part)}


# ------------------------------------------------------------------------------------------------------------------ P1-07: order lifecycle
LEGAL = {"pending": {"partial", "filled", "rejected", "expired", "cancelled"}, "partial": {"partial", "filled", "expired", "cancelled"}, "filled": set(), "rejected": set(), "expired": set(), "cancelled": set()}


class OrderError(ValueError):
    pass


@dataclass
class TrackedOrder:
    order_id: int
    security_id: str
    quantity: float
    state: str = "pending"
    filled: float = 0.0
    history: list = field(default_factory=list)
    fill_ids: set = field(default_factory=set)
    reason: str = ""

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled


class OrderBook:
    """Orders with legal state transitions and idempotent fills: a fill id is applied once, however often it is submitted."""

    def __init__(self):
        self.orders: dict[int, TrackedOrder] = {}
        self._next = 1

    def submit(self, security_id: str, quantity: float, ts) -> TrackedOrder:
        if quantity == 0 or not np.isfinite(quantity):
            raise OrderError("an order needs a non-zero finite quantity")
        o = TrackedOrder(self._next, security_id, float(quantity))
        o.history.append((pd.Timestamp(ts), "pending", ""))
        self.orders[o.order_id] = o
        self._next += 1
        return o

    def _move(self, o: TrackedOrder, state: str, ts, reason: str = "") -> None:
        if state not in LEGAL[o.state]:
            raise OrderError(f"order {o.order_id}: {o.state} -> {state} is not a legal transition")
        o.state, o.reason = state, reason or o.reason
        o.history.append((pd.Timestamp(ts), state, reason))

    def fill(self, order_id: int, fill_id: str, quantity: float, ts) -> TrackedOrder:
        o = self.orders[order_id]
        if fill_id in o.fill_ids:
            return o                                                                              # a retry of a fill already applied
        tol = self.tolerance(o.quantity)
        if np.sign(quantity) != np.sign(o.quantity) or abs(quantity) > abs(o.remaining) + tol:
            raise OrderError(f"order {order_id}: a fill of {quantity:g} does not fit the remaining {o.remaining:g}")
        o.fill_ids.add(fill_id)
        o.filled += quantity
        if abs(o.remaining) <= tol:
            o.filled = o.quantity                                                                 # rounding noise is not an open remainder: the order is complete, exactly
        self._move(o, "filled" if o.remaining == 0.0 else "partial", ts)
        return o

    @staticmethod
    def tolerance(quantity: float) -> float:
        """How far two share counts of this size may differ and still be the same number: a few units in the last place of a double. An absolute 1e-12 is right for whole shares up to about a thousand and
        wrong for six million shares in lots of a millionth (a crypto or fractional-share order), where rounding to the lot alone moves the count by about 1e-9."""
        return max(1e-12, 8.0 * np.finfo(float).eps * abs(quantity))

    def amend(self, order_id: int, quantity: float, ts, reason: str) -> TrackedOrder:
        """Change the size of an order that has not been filled at all (same side). Recorded as an event; a filled or part-filled order cannot be re-sized."""
        o = self.orders[order_id]
        if o.state != "pending" or o.filled != 0.0:
            raise OrderError(f"order {order_id}: only an unfilled pending order can be amended (it is {o.state}, filled {o.filled:g})")
        if quantity == 0 or not np.isfinite(quantity) or np.sign(quantity) != np.sign(o.quantity):
            raise OrderError(f"order {order_id}: an amendment must keep the side and be non-zero")
        o.history.append((pd.Timestamp(ts), "pending", f"amended {o.quantity:g} -> {quantity:g}: {reason}"))
        o.quantity = float(quantity)
        return o

    def reject(self, order_id: int, ts, reason: str) -> TrackedOrder:
        o = self.orders[order_id]
        self._move(o, "rejected", ts, reason)
        return o

    def cancel(self, order_id: int, ts, reason: str = "cancelled") -> TrackedOrder:
        o = self.orders[order_id]
        self._move(o, "cancelled", ts, reason)
        return o

    def expire(self, order_id: int, ts) -> TrackedOrder:
        o = self.orders[order_id]
        self._move(o, "expired", ts, "time in force ended")
        return o

    def open(self) -> list[TrackedOrder]:
        return [o for o in self.orders.values() if o.state in ("pending", "partial")]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"order_id": o.order_id, "security_id": o.security_id, "quantity": o.quantity, "filled": o.filled, "state": o.state, "reason": o.reason,
                              "events": len(o.history)} for o in self.orders.values()])


# ------------------------------------------------------------------------------------------------------------------ P1-08: borrow
class BorrowBook:
    """Dated shares available to borrow, fees and recalls. ``availability`` is a date x security frame of shares (NaN = no record), ``fee`` the annual rate (NaN = unknown). In strict mode a short that is not
    covered by a record is refused; outside it the scenario fee must be stated."""

    def __init__(self, availability: pd.DataFrame, fee: pd.DataFrame | None = None, recalls: Mapping[str, list] | None = None, strict: bool = True, scenario_fee: float | None = None):
        self.availability, self.fee, self.strict, self.scenario_fee = availability, fee, strict, scenario_fee
        self.recalls = {k: [(pd.Timestamp(d), float(q)) for d, q in v] for k, v in (recalls or {}).items()}

    @property
    def empty(self) -> bool:
        """True when the book holds no availability record at all (so under strict validation every new short is refused)."""
        return not bool(self.availability.notna().to_numpy().any())

    def available(self, security_id: str, date) -> float:
        a = self.availability.get(security_id)
        if a is None:
            return float("nan")
        v = a.asof(pd.Timestamp(date))
        return float(v) if v == v else float("nan")

    def rate(self, security_id: str, date) -> float:
        if self.fee is not None and security_id in self.fee:
            v = self.fee[security_id].asof(pd.Timestamp(date))
            if v == v:
                return float(v)
        if self.scenario_fee is not None and not self.strict:
            return float(self.scenario_fee)
        return float("nan")

    def allowed_short(self, security_id: str, date, current_short: float, wanted: float) -> tuple[float, str]:
        """The shares of an additional short ``wanted`` (positive) that may be opened, and why not the rest."""
        have = self.available(security_id, date)
        if not np.isfinite(have):
            if self.strict:
                return 0.0, "no borrow record: a new short is refused under strict validation"
            return wanted, f"no borrow record: scenario fee {self.scenario_fee} assumed"
        room = max(have - current_short, 0.0)
        return min(wanted, room), "" if wanted <= room else f"only {room:g} shares can be located"

    def recalled(self, security_id: str, date) -> float:
        """Shares that must be returned by ``date`` (cumulative recalls on or before it)."""
        return float(sum(q for d, q in self.recalls.get(security_id, []) if d <= pd.Timestamp(date)))


# ------------------------------------------------------------------------------------------------------------------ P1-09: capacity
def fill_with_limits(order_shares: float, adv_shares: float, max_participation: float | None, lot: int = 1) -> dict:
    """What can be filled in one session: the order capped at ``max_participation`` of ADV, rounded down to whole lots. Missing ADV blocks the trade (no liquidity record is not infinite capacity)."""
    if order_shares == 0:
        return {"filled": 0.0, "remaining": 0.0, "status": "no_order"}
    if max_participation is None:
        cap = abs(order_shares)
    elif not (np.isfinite(adv_shares) and adv_shares > 0):
        return {"filled": 0.0, "remaining": order_shares, "status": "blocked_missing_adv", "reason": "no point-in-time ADV: capacity is not assumed"}
    else:
        cap = max_participation * adv_shares
    take = min(abs(order_shares), cap)
    lots = take / lot
    take = np.floor(lots + max(1e-9, 4.0 * np.finfo(float).eps * lots)) * lot                    # a count that is a whole number of lots up to rounding (0.3 / 0.1 is 2.9999...) is that whole number, not one lot less
    filled = float(np.sign(order_shares) * take)
    return {"filled": filled, "remaining": float(order_shares - filled), "status": "full" if abs(order_shares - filled) < 1e-9 else ("partial" if take > 0 else "blocked_lot_or_cap")}


def capacity_curve(weights: pd.Series, price: pd.Series, adv_dollars: pd.Series, capitals, max_participation: float = 0.10, impact: ImpactSpec | None = None, sigma: pd.Series | None = None) -> pd.DataFrame:
    """For each predeclared capital level: the share of the target that can be traded in one session under the participation cap, and the impact cost of what is traded in basis points of the traded
    notional. Capital changes the dollar order; the ADV does not: more capital means a lower fill share and a higher cost. Assets without ADV contribute nothing and are counted as blocked."""
    impact = impact or ImpactSpec()
    rows = []
    for cap in capitals:
        notional = weights.abs() * cap
        room = max_participation * adv_dollars
        ok = adv_dollars.notna() & (adv_dollars > 0)
        traded = pd.Series(np.where(ok, np.minimum(notional, room.fillna(0.0)), 0.0), index=weights.index)
        fill_share = float(traded.sum() / notional.sum()) if notional.sum() > 0 else float("nan")
        costs = []
        for a in weights.index:
            if traded[a] > 0:
                costs.append(impact_bps(impact, traded[a], float(adv_dollars[a]), float(sigma[a]) if sigma is not None else 0.01)["bps"] * traded[a] / 1e4)
        cost_bps = float(np.nansum(costs) / traded.sum() * 1e4) if traded.sum() > 0 and costs else 0.0
        rows.append({"capital": float(cap), "fill_share": fill_share, "traded_notional": float(traded.sum()), "impact_bps_of_traded": cost_bps, "blocked_assets": int((~ok & (weights.abs() > 0)).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------------ P1-10: closed sessions, stale marks, halts and delistings
@dataclass(frozen=True)
class MarkPolicy:
    """Separate rules for valuing, trading and recognising loss. A missing quote is neither a zero return nor a free liquidation: the last good mark values the position while the market is closed or the
    quote stale (flagged), nothing can be traded at it, and a delisting or an over-long halt recognises loss against the recovery assumption on the settlement date."""

    max_stale_days: int = 5
    delisting_recovery: float = 0.0
    settlement_lag_days: int = 5


def classify_marks(price: pd.Series, closed: pd.Series | None = None, halted: pd.Series | None = None, delisted_on=None, policy: MarkPolicy = MarkPolicy()) -> pd.DataFrame:
    """Per date: the value to carry (``value_mark``), whether it is tradable, and the state: ``ok``, ``closed``, ``stale``, ``halted``, ``delisted``. A carried mark is not tradable."""
    idx = price.index
    closed = closed.reindex(idx).fillna(False) if closed is not None else pd.Series(False, index=idx)
    halted = halted.reindex(idx).fillna(False) if halted is not None else pd.Series(False, index=idx)
    last, age, rows = np.nan, 0, []
    for d in idx:
        p = price.loc[d]
        is_delisted = delisted_on is not None and d >= pd.Timestamp(delisted_on)
        if is_delisted:
            state = "delisted"
        elif halted.loc[d]:
            state = "halted"
        elif closed.loc[d]:
            state = "closed"
        elif not np.isfinite(p):
            state = "stale"
        else:
            state = "ok"
        if state == "ok":
            last, age = float(p), 0
        else:
            age += 1
            if state == "stale" and age > policy.max_stale_days:
                state = "stale_expired"
        rows.append({"date": d, "value_mark": last, "tradable": state == "ok", "state": state, "age": age})
    return pd.DataFrame(rows).set_index("date")


def delisting_loss(last_mark: float, quantity: float, policy: MarkPolicy = MarkPolicy()) -> dict:
    """Cash recovered and loss recognised when a position is delisted: ``recovery`` of the last mark, arriving ``settlement_lag_days`` later."""
    recovered = quantity * last_mark * policy.delisting_recovery
    return {"recovered_cash": float(recovered), "loss": float(quantity * last_mark - recovered), "settles_after_days": policy.settlement_lag_days}


# ------------------------------------------------------------------------------------------------------------------ P1-12: currencies
class FXBook:
    """Timestamped currency rates (units of quote per unit of base), as-of lookups only, conversion spreads and settlement lags. A missing rate raises: it is never 1, and a rate observed later is never used
    to translate an earlier fill."""

    def __init__(self, base: str = "USD", spread_bps: float = 0.0, settlement_days: Mapping[str, int] | None = None):
        self.base, self.spread_bps = base, spread_bps
        self.settlement_days = dict(settlement_days or {})
        self._series: dict[str, pd.Series] = {}

    def set_series(self, ccy: str, rates: pd.Series) -> None:
        """``rates``: base currency units per one unit of ``ccy``, indexed by timestamp."""
        self._series[ccy] = rates.sort_index().dropna()

    def rate_at(self, ccy: str, ts) -> float:
        if ccy == self.base:
            return 1.0
        s = self._series.get(ccy)
        if s is None:
            raise KeyError(f"no FX series for {ccy}: a rate is not guessed")
        t = pd.Timestamp(ts)
        past = s.loc[:t]
        if past.empty:
            raise KeyError(f"no {ccy} rate on or before {t}: a later rate is not used")
        return float(past.iloc[-1])

    def convert(self, amount: float, ccy: str, ts, to_base: bool = True) -> dict:
        """The value of ``amount`` of ``ccy`` in the base currency at the rate in force at ``ts``, with the conversion spread charged against the converter."""
        r = self.rate_at(ccy, ts)
        mid = amount * r if to_base else amount / r
        cost = abs(mid) * self.spread_bps / 1e4
        return {"value": float(mid - np.sign(mid) * cost), "mid": float(mid), "spread_cost": float(cost), "rate": r}

    def settle_date(self, ccy: str, trade_date, sessions) -> pd.Timestamp:
        n = self.settlement_days.get(ccy, 0)
        s = pd.DatetimeIndex(sessions)
        i = s.searchsorted(pd.Timestamp(trade_date))
        return s[min(i + n, len(s) - 1)]

    def exposures(self, balances: Mapping[str, float], ts) -> dict:
        return {c: {"native": float(v), "base": float(v * self.rate_at(c, ts))} for c, v in balances.items()}

    def unsettled(self, trades: list, sessions, on) -> dict:
        """Native cash that has been booked but has not settled on ``on``: ``trades`` are ``{"ccy", "amount", "trade_date"}`` (signed, native units); each settles ``settlement_days[ccy]`` sessions after its trade
        date. The ledger books cash on the trade date; this is the report of what was still in transit, per currency."""
        on = pd.Timestamp(on)
        out: dict[str, float] = {}
        for tr in trades:
            if pd.Timestamp(tr["trade_date"]) <= on < self.settle_date(tr["ccy"], tr["trade_date"], sessions):
                out[tr["ccy"]] = out.get(tr["ccy"], 0.0) + float(tr["amount"])
        return out


# ------------------------------------------------------------------------------------------------------------------ P1-13: futures and crypto derivatives
@dataclass(frozen=True)
class FuturesSpec:
    """A dated contract specification: the multiplier, margins and expiry in force. A new spec with a later ``effective`` supersedes it from that date."""

    contract: str
    multiplier: float
    initial_margin: float
    maintenance_margin: float
    expiry: pd.Timestamp
    effective: pd.Timestamp = pd.Timestamp("1900-01-01")

    def __post_init__(self):
        if self.multiplier <= 0 or self.maintenance_margin > self.initial_margin:
            raise ValueError("multiplier must be positive and maintenance margin at most the initial margin")
        object.__setattr__(self, "expiry", pd.Timestamp(self.expiry))
        object.__setattr__(self, "effective", pd.Timestamp(self.effective))


@dataclass(frozen=True)
class FeeSchedule:
    """A venue's fee tiers: ``tiers`` is ``((min trailing 30-day volume, maker bps, taker bps), ...)`` in ascending order of volume; the highest tier whose minimum is met applies. A negative maker rate is a
    rebate. ``source`` says whether the schedule is the venue's published one or an assumption of the run (the default): a schedule is never reported as measured unless it says so."""

    tiers: tuple
    venue: str = ""
    source: str = "assumed"

    def __post_init__(self):
        mins = [t[0] for t in self.tiers]
        if not self.tiers or mins[0] != 0 or mins != sorted(mins) or len(set(mins)) != len(mins):
            raise ValueError("tiers need a first tier at volume 0 and strictly ascending minimum volumes")

    def rate_bps(self, trailing_volume: float, maker: bool) -> float:
        if not np.isfinite(trailing_volume) or trailing_volume < 0:
            raise ValueError("the trailing volume is unknown: the fee tier is not guessed")
        tier = [t for t in self.tiers if trailing_volume >= t[0]][-1]
        return float(tier[1] if maker else tier[2])

    def fee(self, notional: float, trailing_volume: float, maker: bool) -> float:
        """The fee on ``notional`` (positive = paid, negative = rebate)."""
        return float(abs(notional) * self.rate_bps(trailing_volume, maker) / 1e4)


def spec_at(specs: list[FuturesSpec], contract: str, date) -> FuturesSpec:
    d = pd.Timestamp(date)
    valid = sorted([s for s in specs if s.contract == contract and s.effective <= d], key=lambda s: s.effective)
    if not valid:
        raise KeyError(f"no specification of {contract} in force on {d.date()}")
    return valid[-1]


def executable_contract(chain: list[FuturesSpec], date, roll_days: int = 5) -> str:
    """The contract that can actually be traded on ``date``: the nearest one that is not within ``roll_days`` of expiry. A continuous series is a research construct; it is never the thing that is traded."""
    d = pd.Timestamp(date)
    live = sorted({s.contract: s for s in chain if s.expiry > d + pd.Timedelta(days=roll_days)}.values(), key=lambda s: s.expiry)
    if not live:
        raise KeyError(f"no tradable contract on {d.date()}")
    return live[0].contract


def roll_events(chain: list[FuturesSpec], dates, roll_days: int = 5) -> list[dict]:
    """The dates on which the executable contract changes, with the contracts closed and opened. The roll is a trade at two prices; the gap between them is not P&L."""
    out, prev = [], None
    for d in pd.DatetimeIndex(dates):
        c = executable_contract(chain, d, roll_days)
        if prev is not None and c != prev:
            out.append({"date": d, "close": prev, "open": c})
        prev = c
    return out


def variation_margin(quantity: float, settle_prev: float, settle_now: float, multiplier: float) -> float:
    """Daily mark-to-market cash on a futures position: quantity x multiplier x (settle - previous settle). Long gains when the price rises."""
    return float(quantity * multiplier * (settle_now - settle_prev))


def margin_call(equity_in_account: float, quantity: float, price: float, spec: FuturesSpec) -> dict:
    """Whether the account is below maintenance margin and the variation needed to restore initial margin."""
    need_m, need_i = abs(quantity) * spec.maintenance_margin, abs(quantity) * spec.initial_margin
    return {"called": bool(equity_in_account < need_m), "maintenance": float(need_m), "initial": float(need_i), "top_up": float(max(need_i - equity_in_account, 0.0)) if equity_in_account < need_m else 0.0}


def funding_payment(quantity: float, mark: float, rate: float, multiplier: float = 1.0) -> float:
    """Perpetual funding for one interval: positive when the account RECEIVES. A positive rate means longs pay shorts, so a long position (quantity > 0) pays ``quantity x mark x rate``."""
    return float(-quantity * multiplier * mark * rate)


# ------------------------------------------------------------------------------------------------------------------ P1-14: coverage and capability
def capability_report(bundle=None, *, data_status: str | None = None, membership: Membership | None = None, actions: list[CorporateAction] | None = None, borrow: BorrowBook | None = None,
                      fx: FXBook | None = None, futures_chain: list | None = None, adapter_modes: tuple = ("vectorized_legacy",)) -> dict:
    """What the data and the adapter support, and the limits on a claim that follow. Everything is read from the actual objects: no field is assumed present. ``blockers`` lists what stops a realistic-execution
    claim; ``limitations`` what qualifies a weaker one."""
    has = lambda name: bundle is not None and getattr(bundle, name, None) is not None                          # noqa: E731
    fields = {"close": bundle is not None, "open": has("open"), "high": has("high"), "low": has("low"), "volume": has("volume"), "bid_ask": has("bid") and has("ask"), "marks": has("marks")}
    status = data_status or ("unknown" if bundle is None else getattr(bundle, "name", "bundle"))
    rep = {"data_status": status, "execution_fields": fields,
           "historical_universe": membership.survivorship() if membership is not None else {"coverage": "none", "limitation": "no dated membership: the universe is today's survivors"},
           "corporate_actions": {"provided": len(actions or []), "types": sorted({a.type for a in actions or []}), "unsupported": [a.type for a in (actions or []) if a.type not in SUPPORTED_ACTIONS]},
           "borrow": "none" if borrow is None or borrow.empty else ("records" if not borrow.strict else "strict records"), "fx": sorted(fx._series) if fx is not None else [],      # noqa: SLF001
           "contract_history": len(futures_chain or []), "modes": list(adapter_modes)}
    blockers, limits = [], []
    if status in ("synthetic", "simulated") or "synthetic" in str(status).lower() or "simulated" in str(status).lower():
        limits.append("the market data is synthetic or simulated: results check the machinery, not a market, and cannot validate a strategy or qualify one operationally")
    if rep["historical_universe"]["limitation"]:
        limits.append(rep["historical_universe"]["limitation"])
    if not fields["volume"]:
        blockers.append("no volume: participation limits, capacity and impact cannot be measured (they are not assumed)")
    if not fields["open"] and not fields["bid_ask"]:
        limits.append("only closing prices: fills use the next close with modelled spread; opening-auction and intraday fills are not represented")
    if rep["borrow"] == "none":
        blockers.append("no borrow records: short positions cannot be validated beyond a labelled scenario")
    if rep["corporate_actions"]["provided"] == 0:
        limits.append("no corporate-action events: splits, dividends and mergers are not accounted for beyond the data vendor's adjustment")
    if rep["corporate_actions"]["unsupported"]:
        blockers.append(f"unsupported corporate actions: {', '.join(sorted(set(rep['corporate_actions']['unsupported'])))}")
    rep["blockers"], rep["limitations"] = blockers, limits
    rep["realistic_execution_claim"] = "blocked" if blockers else ("limited" if limits else "supported")
    return rep
