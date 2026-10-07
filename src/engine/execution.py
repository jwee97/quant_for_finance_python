"""The execution simulator: from an order and the market state at one instant to fills.

Given the order, a price snapshot (a quote with a bid and an ask, or a bar/settlement/mark price with an assumed spread) and the cost models, :func:`simulate_fill` decides how much
trades and at what price:

* **market** orders buy at ``mid + half spread + impact + slippage`` and sell below the mid by the same amount;
* **limit** orders fill only if the market has reached the limit: a buy when the ask is at or below the limit (fill at the ask, never above the limit), or, on bar data with a
  ``low``, when the bar's low touched the limit (fill at the limit price: the conservative assumption of queue priority is that you are filled at your price, not better);
* **stop** orders trigger when the price trades through the stop (last or mid at or beyond it) and then behave as market orders; **stop-limit** orders then behave as limit orders;
* **participation**: an order takes at most its participation share of the event's volume, so a large order fills in pieces over several events;
* **FOK** orders fill completely or not at all; **IOC** orders take what is available and cancel the rest (decided by the engine from the result);
* **book depth**: when the quote carries depth levels (``reference_values = {"bids": [[price, size], ...], "asks": [...]}``, best first) a market, stop or marketable limit order WALKS THE BOOK: it
  takes each level in turn up to its size, pays the volume-weighted average price, and the part the displayed depth cannot absorb stays unfilled (a partial fill, or none at all for a
  fill-or-kill order). The spread cost is the touch against the mid and the impact is the average price against the touch; the square-root impact model is not applied on top.

The simulator is a pure function of its inputs, which is what makes replay deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import CostSchedule, LiquidityModel
from .orders import Order


@dataclass(frozen=True)
class Quote:
    """What the venue shows at the instant of the attempt."""

    mid: float
    bid: float = float("nan")
    ask: float = float("nan")
    last: float = float("nan")
    volume: float = float("nan")
    low: float = float("nan")
    high: float = float("nan")
    adv: float = float("nan")
    sigma: float = float("nan")
    depth: tuple | None = None       # ((bid price, size), ...), ((ask price, size), ...) best first, when the venue shows more than the touch


@dataclass(frozen=True)
class FillResult:
    quantity: float                  # signed quantity executed in this attempt (0 if none)
    price: float
    spread_price: float
    impact_price: float
    extra_price: float
    fee: float
    fee_currency: str
    mid: float
    triggered: bool = False
    note: str = ""


NO_FILL = FillResult(0.0, float("nan"), 0.0, 0.0, 0.0, 0.0, "", float("nan"))


def simulate_fill(order: Order, inst, quote: Quote, costs: CostSchedule, liquidity: LiquidityModel | None = None) -> FillResult:
    q_rem = order.remaining
    if q_rem == 0.0 or not np.isfinite(quote.mid):
        return NO_FILL
    side = 1.0 if q_rem > 0 else -1.0
    triggered = order.triggered
    # --- stops: decide whether the stop is hit by this observation
    if order.type in ("stop", "stop_limit") and not triggered:
        ref_hi = np.nanmax([quote.high, quote.last, quote.mid]) if np.isfinite(quote.high) else np.nanmax([quote.last, quote.mid])
        ref_lo = np.nanmin([quote.low, quote.last, quote.mid]) if np.isfinite(quote.low) else np.nanmin([quote.last, quote.mid])
        hit = (side > 0 and ref_hi >= order.stop_price) or (side < 0 and ref_lo <= order.stop_price)
        if not hit:
            return NO_FILL
        triggered = True
    # --- how much can trade
    cap = (liquidity or LiquidityModel()).fill_cap(order.participation, quote.volume)
    qty = side * min(abs(q_rem), cap)
    if order.tif == "FOK" and abs(qty) < abs(q_rem):
        return NO_FILL
    qty = inst.round_quantity(qty) if cap < abs(q_rem) else qty
    if qty == 0.0:
        return NO_FILL
    first = order.filled == 0.0
    if quote.depth is not None:
        res = _walk_book(order, inst, quote, costs, side, abs(qty), triggered, first)
        if res is not None:
            return res
    cb = costs.estimate(inst, qty, quote.mid, quote.bid, quote.ask, quote.adv, quote.sigma, first)
    exec_px = quote.mid + side * cb.price_distance
    # --- limit logic (limit orders, and stop-limit orders after the trigger)
    is_limit = order.type == "limit" or (order.type == "stop_limit" and triggered)
    if is_limit:
        lp = order.limit_price
        if side > 0:
            ask = quote.mid + cb.spread_price if not np.isfinite(quote.ask) else quote.ask
            if ask <= lp:
                exec_px = min(exec_px, lp)
            elif np.isfinite(quote.low) and quote.low <= lp:
                exec_px = lp
            else:
                return NO_FILL
        else:
            bid = quote.mid - cb.spread_price if not np.isfinite(quote.bid) else quote.bid
            if bid >= lp:
                exec_px = max(exec_px, lp)
            elif np.isfinite(quote.high) and quote.high >= lp:
                exec_px = lp
            else:
                return NO_FILL
    # the price distance actually paid is capped by the limit; keep the components consistent with the booked price
    spread_p, impact_p, extra_p = cb.spread_price, cb.impact_price, cb.extra_price
    paid = abs(exec_px - quote.mid)
    if paid < cb.price_distance:                                   # a limit order that fills better than a market order: scale the components down
        f = paid / cb.price_distance if cb.price_distance > 0 else 0.0
        spread_p, impact_p, extra_p = spread_p * f, impact_p * f, extra_p * f
    models = costs.models_for(inst)
    fee, fee_ccy = models["commission"].fee(inst, qty, exec_px, first)
    for extra in models.get("extra_fees", ()):
        fee += extra.fee(inst, qty, exec_px, first)[0]
    return FillResult(float(qty), float(exec_px), float(spread_p), float(impact_p), float(extra_p), float(fee), fee_ccy, float(quote.mid), triggered)


def _walk_book(order: Order, inst, quote: Quote, costs: CostSchedule, side: float, want: float, triggered: bool, first: bool) -> FillResult | None:
    """Fill against displayed depth: consume the opposite side level by level (up to the limit price for limit orders). Returns ``None`` when the quote has no usable depth on that side."""
    levels = quote.depth[1] if side > 0 else quote.depth[0]
    levels = [(float(p), float(q)) for p, q in levels if np.isfinite(p) and np.isfinite(q) and q > 0]
    if not levels:
        return None
    is_limit = order.type == "limit" or (order.type == "stop_limit" and triggered)
    if is_limit:
        lp = order.limit_price
        levels = [(p, q) for p, q in levels if (p <= lp if side > 0 else p >= lp)]
        if not levels:
            return NO_FILL
    remaining, notional, taken = want, 0.0, 0.0
    for price, size in levels:
        take = min(remaining, size)
        notional += take * price
        taken += take
        remaining -= take
        if remaining <= 1e-12:
            break
    if taken <= 0.0:
        return NO_FILL
    if order.tif == "FOK" and remaining > 1e-12:
        return NO_FILL
    qty = inst.round_quantity(side * taken)
    if qty == 0.0:
        return NO_FILL
    vwap = notional / taken
    touch = levels[0][0]
    models = costs.models_for(inst)
    extra = models["slippage"].extra(quote.mid)
    exec_px = vwap + side * extra
    if is_limit:                                                           # never worse than the limit
        exec_px = min(exec_px, order.limit_price) if side > 0 else max(exec_px, order.limit_price)
    spread_p = max(side * (touch - quote.mid), 0.0)
    impact_p = max(side * (vwap - touch), 0.0)
    extra_p = max(side * (exec_px - vwap), 0.0)
    fee, fee_ccy = models["commission"].fee(inst, qty, exec_px, first)
    for more in models.get("extra_fees", ()):
        fee += more.fee(inst, qty, exec_px, first)[0]
    return FillResult(float(qty), float(exec_px), float(spread_p), float(impact_p), float(extra_p), float(fee), fee_ccy, float(quote.mid), triggered, "book")
