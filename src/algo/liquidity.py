"""Liquidity seeking: work an order where the market is deep, and step back when it is not.

A volume-following algorithm trades its schedule whatever the market looks like, so in a crisis it keeps hitting a thin, wide, jumpy book at the worst possible price. A liquidity-seeking algorithm reads
what it can see each interval and scales its pace by how good the liquidity is:

    quality = (volume printing / volume expected) / (spread / normal spread  *  depth shortfall)         pace = base pace * clip(quality ** sensitivity, floor, cap)

Heavy volume in a calm, tight, deep market is quality above one: trade more, up to ``cap`` times the base pace. A crisis (spreads four times wider, a book two and a half times thinner, even with panic volume)
is quality far below one: trade only ``floor`` of the base pace and wait. What it holds back is spread over the rest of the plan by the engine, so the order is still finished by the deadline, usually at
a higher pace after the storm. It avoids paying crisis prices for impact and spread; it pays in being exposed to the price for longer, which is a price risk it does not hedge. Whether that is a good bargain
depends on which way the price goes, which is why the tests state only what it does to cost and participation.

It wraps a base algorithm (default VWAP): ``LiquiditySeeking(ImplementationShortfall(1e-3))``.
"""

from __future__ import annotations

import numpy as np

from .algos import VWAP
from .market import Market, Order, Scenario
from .simulate import Algo, State


class LiquiditySeeking(Algo):
    name = "liquidity_seeking"
    category = "liquidity"
    description = "Liquidity seeking: scales the pace by how good the liquidity is (volume against its profile, the spread and the book depth), trading little in a thin, wide, jumpy market and more in a deep one, and finishes later in the plan."

    def __init__(self, base: Algo | None = None, sensitivity: float = 1.0, floor: float = 0.15, cap: float = 2.5):
        if sensitivity < 0 or not 0 < floor <= 1 <= cap:
            raise ValueError("sensitivity >= 0 and 0 < floor <= 1 <= cap")
        self.base, self.sensitivity, self.floor, self.cap = base or VWAP(), sensitivity, floor, cap

    def prepare(self, order: Order, market: Market, scenario: Scenario) -> None:
        super().prepare(order, market, scenario)
        self.base.prepare(order, market, scenario)

    def plan(self):
        return self.base.plan()

    def quality(self, state: State) -> np.ndarray:
        """Liquidity quality now: one in a normal interval, below one when volume is light or the market is wide and thin."""
        volume_ratio = state.volume / max(state.expected_volume, 1e-9)
        return volume_ratio / max(state.spread_mult * state.impact_mult, 1e-9)

    def want(self, state: State) -> np.ndarray:
        scale = np.clip(self.quality(state) ** self.sensitivity, self.floor, self.cap)
        return self.base.want(state) * scale
