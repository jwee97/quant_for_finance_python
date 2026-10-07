"""Transaction costs, financing and liquidity as first-class, configurable models.

A fill's all-in price is ``mid +/- (half spread + impact + extra slippage)`` and a fee is charged on top::

    cost = fixed fee + spread cost + impact cost (+ slippage)

* :class:`CommissionModel`: per order, per contract and in basis points of notional with a minimum; exchange fees are another commission model added to the schedule.
* :class:`SpreadModel`: the OBSERVED half spread when the market data carries a quote, otherwise a fallback in basis points of the mid (a daily bar has no spread, so the cost
  of crossing it is an explicit assumption), never below a fraction of a tick.
* :class:`ImpactModel`: the square-root law ``Y sigma sqrt(|Q| / ADV)`` (Almgren et al.; Toth et al.) with the daily volatility ``sigma`` and the average daily volume ADV estimated from
  point-in-time bars, or a linear-in-participation model; impact depends on instrument liquidity, order size, volatility and (through the spread) on the book.
* :class:`SlippageModel`: extra adverse price in basis points (latency, queueing).
* :class:`CostSchedule` selects the models by instrument id, then instrument type, then asset class, then default, so FX, futures, crypto and options each carry their own costs.
* :class:`FinancingModel`: deposit and borrow rates per currency, borrow fees per security or coin, accrued daily by the engine; perpetual funding and margin interest are posted by
  the lifecycle handlers.
* :class:`LiquidityModel`: a participation limit per event (an order may take at most ``max_participation`` of the volume), so large orders fill over several events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostBreakdown:
    spread_price: float = 0.0
    impact_price: float = 0.0
    extra_price: float = 0.0
    fee: float = 0.0
    fee_currency: str = ""

    @property
    def price_distance(self) -> float:
        return self.spread_price + self.impact_price + self.extra_price


@dataclass(frozen=True)
class CommissionModel:
    per_order: float = 0.0
    per_contract: float = 0.0
    bps: float = 0.0
    minimum: float = 0.0
    currency: str = ""

    def fee(self, inst, quantity: float, price: float, first_fill: bool = True) -> tuple[float, str]:
        """The fee for one execution. The per-order fee and the minimum apply to the first fill of an order only (an order filled in pieces pays them once)."""
        amount = (self.per_order if first_fill else 0.0) + self.per_contract * abs(quantity) + self.bps * 1e-4 * inst.notional(price, quantity)
        if first_fill:
            amount = max(amount, self.minimum)
        return (amount if amount > 0 else 0.0), self.currency or inst.currency


@dataclass(frozen=True)
class SpreadModel:
    fallback_bps: float = 1.0                   # FULL spread in bps of the mid when the data has no quote
    min_half_ticks: float = 0.5
    use_quotes: bool = True

    def half_spread(self, inst, mid: float, bid: float = float("nan"), ask: float = float("nan")) -> float:
        if self.use_quotes and np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
            hs = (ask - bid) / 2.0
        else:
            hs = self.fallback_bps * 1e-4 * abs(mid) / 2.0
        return max(hs, self.min_half_ticks * inst.tick_size)


@dataclass(frozen=True)
class SlippageModel:
    bps: float = 0.0

    def extra(self, mid: float) -> float:
        return self.bps * 1e-4 * abs(mid)


@dataclass(frozen=True)
class ImpactModel:
    kind: str = "sqrt"                          # sqrt, linear or none
    y: float = 0.5                              # sqrt: price move of ``y`` daily sigmas for an order equal to a full day's volume
    k: float = 0.1                              # linear: price move of ``k`` daily sigmas per unit of participation
    max_participation: float = 1.0              # participation used in the formula is capped here
    default_adv: float = float("nan")
    default_sigma: float = float("nan")         # daily volatility (a fraction) when no history is available

    def impact_price(self, mid: float, quantity: float, adv: float, sigma: float) -> float:
        if self.kind == "none" or quantity == 0:
            return 0.0
        adv = adv if np.isfinite(adv) and adv > 0 else self.default_adv
        sigma = sigma if np.isfinite(sigma) and sigma > 0 else self.default_sigma
        if not (np.isfinite(adv) and adv > 0 and np.isfinite(sigma)):
            return 0.0
        part = min(abs(quantity) / adv, self.max_participation)
        move = self.y * np.sqrt(part) if self.kind == "sqrt" else self.k * part
        return float(move * sigma * abs(mid))


@dataclass
class CostSchedule:
    """The cost models in force for every instrument, selected most-specific first (instrument id, instrument type, asset class, default)."""

    default: dict = field(default_factory=lambda: {"commission": CommissionModel(), "spread": SpreadModel(), "impact": ImpactModel(kind="none"), "slippage": SlippageModel(), "extra_fees": ()})
    overrides: dict = field(default_factory=dict)     # key (id | type | asset class) -> partial dict of models
    fx_conversion_bps: float = 0.0                    # charged on the notional converted when a physical settlement exchanges currencies

    def set(self, key: str, **models) -> "CostSchedule":
        self.overrides.setdefault(key, {}).update(models)
        return self

    def models_for(self, inst) -> dict:
        out = dict(self.default)
        for key in (inst.asset_class, inst.instrument_type, inst.instrument_id):
            out.update(self.overrides.get(key, {}))
        return out

    def estimate(self, inst, quantity: float, mid: float, bid: float = float("nan"), ask: float = float("nan"), adv: float = float("nan"), sigma: float = float("nan"),
                 first_fill: bool = True) -> CostBreakdown:
        m = self.models_for(inst)
        hs = m["spread"].half_spread(inst, mid, bid, ask)
        imp = m["impact"].impact_price(mid, quantity, adv, sigma)
        extra = m["slippage"].extra(mid)
        px = mid + np.sign(quantity) * (hs + imp + extra)
        fee, ccy = m["commission"].fee(inst, quantity, px, first_fill)
        for f in m.get("extra_fees", ()):
            more, _ = f.fee(inst, quantity, px, first_fill)
            fee += more
        return CostBreakdown(float(hs), float(imp), float(extra), float(fee), ccy)

    def round_trip_bps(self, inst, mid: float, quantity: float = 1.0, **kw) -> float:
        """Round-trip cost (enter and exit) in basis points of notional, a quick way to compare instruments."""
        c = self.estimate(inst, abs(quantity), mid, **kw)
        notional = inst.notional(mid, quantity)
        return float(2.0 * (c.price_distance * abs(quantity) * inst.contract_multiplier + c.fee) / notional * 1e4) if notional > 0 else float("nan")


@dataclass
class FinancingModel:
    deposit_rates: dict = field(default_factory=dict)           # currency -> annual rate earned on positive balances
    borrow_spread: float = 0.005                                 # added to the deposit rate on negative balances
    borrow_rates: dict = field(default_factory=dict)            # currency -> explicit annual rate on negative balances (overrides deposit + spread)
    short_borrow_rates: dict = field(default_factory=dict)      # instrument id -> annual borrow fee on short positions
    default_short_borrow: float = 0.0
    day_count: float = 365.0
    margin_interest_spread: float = 0.0                          # extra annual rate on the amount of margin in use

    def short_rates_for(self, iids) -> dict:
        return {i: self.short_borrow_rates.get(i, self.default_short_borrow) for i in iids if self.short_borrow_rates.get(i, self.default_short_borrow) > 0}


@dataclass(frozen=True)
class LiquidityModel:
    max_participation: float | None = None                       # share of an event's volume one order may take; None = unlimited
    adv_window: int = 20
    min_volume: float = 0.0

    def fill_cap(self, order_participation: float | None, volume: float) -> float:
        """The largest quantity (in contracts) that may be executed against an event with this ``volume`` (inf when no limit applies or the volume is unknown)."""
        p = order_participation if order_participation is not None else self.max_participation
        if p is None or not np.isfinite(volume):
            return float("inf")
        return max(p * volume, 0.0)


def realised_sigma(closes: pd.Series, window: int = 20) -> float:
    """Daily volatility of log returns over the last ``window`` observations (NaN if too few)."""
    c = closes.dropna().tail(window + 1)
    if len(c) < 5:
        return float("nan")
    return float(np.log(c).diff().dropna().std(ddof=1))
