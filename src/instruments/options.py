"""Options: contract specification, payoff, exercise and settlement rules, Greeks and surface linkage.

An :class:`Option` is one listed contract: underlying, call or put, strike, expiry, exercise style (``european`` or ``american``) and settlement (``cash`` for index and many futures
options, ``physical`` for options that deliver the underlying). The premium is paid in full (the ``premium`` cash style): a long option is an asset, a short option a liability, and a
short option needs margin (``margin_requirement``: the Reg-T-style rule ``max(a S - OTM, b S) + premium`` per share, with ``initial_margin = a`` and ``maintenance_margin = b``).

``underlying_kind`` decides the pricing model for Greeks and the delivery: ``spot`` and ``index`` use Black-Scholes-Merton; ``future`` uses Black-76 and exercising delivers a futures
position at the strike (and, because futures are variation-margined, an immediate cash settlement of the intrinsic value). ``surface_id`` links the contract to a volatility surface so
that marks and Greeks can be modelled when a quote is missing.

Exercise: a long option is automatically exercised at expiry when its intrinsic value exceeds ``auto_exercise_threshold`` (currency per unit; exchanges use one cent); the holder of an
American option may exercise earlier. The matching short position is assigned. Early assignment of a short American option is decided by the engine's assignment rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Instrument, register_kind

RIGHTS = ("call", "put")
STYLES = ("european", "american")
UNDERLYING_KINDS = ("spot", "future", "index")


@register_kind("option")
@dataclass(frozen=True, kw_only=True)
class Option(Instrument):
    right: str = "call"
    strike: float = 0.0
    exercise_style: str = "european"
    underlying_kind: str = "spot"
    surface_id: str | None = None
    auto_exercise_threshold: float = 0.01
    inverse: bool = False                     # crypto options quoted and settled in the coin

    CASH_STYLE = "premium"

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.right not in RIGHTS or self.exercise_style not in STYLES or self.underlying_kind not in UNDERLYING_KINDS:
            raise ValueError(f"right in {RIGHTS}, exercise_style in {STYLES}, underlying_kind in {UNDERLYING_KINDS}")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.expiry is None or self.underlying_id is None:
            raise ValueError("an option needs an underlying_id and an expiry")

    @property
    def is_call(self) -> bool:
        return self.right == "call"

    def notional(self, price: float = 0.0, quantity: float = 1.0) -> float:
        """Underlying notional controlled by the position (contracts x multiplier x strike), not the premium."""
        return abs(quantity) * self.contract_multiplier * self.strike

    def intrinsic(self, underlying_price: float) -> float:
        """Intrinsic value per unit of the underlying."""
        return max(underlying_price - self.strike, 0.0) if self.is_call else max(self.strike - underlying_price, 0.0)

    def is_in_the_money(self, underlying_price: float) -> bool:
        return self.intrinsic(underlying_price) > 0.0

    def moneyness(self, underlying_price: float) -> float:
        return float(np.log(underlying_price / self.strike))

    def time_to_expiry(self, ts) -> float:
        """Years (actual/365) from ``ts`` to expiry, floored at zero."""
        return max((self.expiry - pd.Timestamp(ts)).total_seconds() / (365.0 * 86400.0), 0.0)

    def payoff(self, underlying_price: float, quantity: float = 1.0) -> float:
        """Cash payoff at expiry in ``currency`` for a signed ``quantity`` of contracts."""
        return quantity * self.contract_multiplier * self.intrinsic(underlying_price)

    def will_auto_exercise(self, underlying_price: float) -> bool:
        return self.intrinsic(underlying_price) > self.auto_exercise_threshold

    def exercise_settlement(self, quantity: float, underlying_price: float) -> dict:
        """What exercise (long, ``quantity > 0``) or assignment (short, ``quantity < 0``) does, as ``cash`` (currency -> amount) and ``delivery`` ((underlying_id, signed quantity, price)).

        Cash-settled: the intrinsic value in cash. Physical on a spot underlying: a call buys the underlying at the strike (cash out, underlying in) and a put sells it. Physical on a future:
        a futures position at the strike is created, and the intrinsic value arrives as the first variation margin when the new position is marked to the underlying's price (no cash is paid at exercise itself)."""
        q = quantity * self.contract_multiplier
        if self.settlement_type == "cash" or self.underlying_kind == "index":
            return {"cash": {self.currency: quantity * self.contract_multiplier * self.intrinsic(underlying_price)}, "delivery": None}
        direction = 1.0 if self.is_call else -1.0
        if self.underlying_kind == "future":
            return {"cash": {}, "delivery": (self.underlying_id, direction * quantity, self.strike)}
        return {"cash": {self.currency: -direction * q * self.strike}, "delivery": (self.underlying_id, direction * q, self.strike)}

    def margin_requirement(self, quantity: float, underlying_price: float, premium: float) -> float:
        """Margin in ``currency`` for a position of ``quantity`` contracts at the given underlying price and option premium. Long options need none (the premium was paid)."""
        if quantity >= 0:
            return 0.0
        otm = max(self.strike - underlying_price, 0.0) if self.is_call else max(underlying_price - self.strike, 0.0)
        a = self.initial_margin or 0.20
        b = self.maintenance_margin or 0.10
        per_unit = max(a * underlying_price - otm, b * underlying_price) + premium
        return abs(quantity) * self.contract_multiplier * per_unit

    def greeks(self, underlying_price: float, vol: float, ts, rate: float = 0.0, dividend: float = 0.0) -> dict:
        """Per-unit Greeks (delta, gamma, vega, theta, rho, ...) from Black-Scholes-Merton (spot, index) or Black-76 (future), with expiry in years from ``ts``."""
        from ..derivatives.greeks import black76_greeks, bsm_greeks

        T = self.time_to_expiry(ts)
        if self.underlying_kind == "future":
            g = black76_greeks(underlying_price, self.strike, T, rate, vol, self.is_call)
        else:
            g = bsm_greeks(underlying_price, self.strike, T, rate, dividend, vol, self.is_call)
        return {k: float(np.asarray(v).ravel()[0]) for k, v in g.items()}

    def model_price(self, underlying_price: float, vol: float, ts, rate: float = 0.0, dividend: float = 0.0) -> float:
        from ..derivatives.pricing import black76_price, bsm_price

        T = self.time_to_expiry(ts)
        if self.underlying_kind == "future":
            return float(np.asarray(black76_price(underlying_price, self.strike, T, rate, vol, self.is_call)).ravel()[0])
        return float(np.asarray(bsm_price(underlying_price, self.strike, T, rate, dividend, vol, self.is_call)).ravel()[0])


def option_id(underlying_id: str, expiry, right: str, strike: float) -> str:
    return f"{underlying_id}-{pd.Timestamp(expiry):%Y%m%d}-{'C' if right == 'call' else 'P'}{strike:g}"


def make_option(underlying_id: str, expiry, right: str, strike: float, multiplier: float = 100.0, currency: str = "USD", underlying_kind: str = "spot", exercise_style: str = "european",
                settlement_type: str | None = None, calendar: str = "US", tick_size: float = 0.01, **kwargs) -> Option:
    """One option contract with sensible exchange defaults (physical delivery for spot and futures underlyings, cash for an index)."""
    st = settlement_type or ("cash" if underlying_kind == "index" else "physical")
    defaults = dict(instrument_id=option_id(underlying_id, expiry, right, strike), asset_class="option", instrument_type="option", currency=currency, tick_size=tick_size,
                    contract_multiplier=multiplier, calendar=calendar, underlying_id=underlying_id, expiry=pd.Timestamp(expiry), settlement_type=st, margin_type="premium",
                    initial_margin=0.20, maintenance_margin=0.10, right=right, strike=strike, exercise_style=exercise_style, underlying_kind=underlying_kind)
    defaults.update(kwargs)
    return Option(**defaults)


def make_option_chain(underlying_id: str, expiries, strikes, rights=("call", "put"), **kwargs) -> list[Option]:
    """Every (expiry, strike, right) combination as an option contract."""
    return [make_option(underlying_id, e, r, float(k), **kwargs) for e in expiries for k in strikes for r in rights]
