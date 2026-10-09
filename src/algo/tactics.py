"""Adaptation tactics: how a running order reacts to what the market does, wrapped around any base algorithm.

A schedule is a plan made before the day. A *tactic* changes the pace as the day unfolds.

**Aggressive in the money (AIM)** and **passive in the money (PIM)** are price-based scaling tactics. They watch the price move since arrival in standard deviations ``z`` (positive when it favours you: a buyer's
price has fallen) and scale the planned rate by ``exp(+- strength * z)``, within ``[floor, cap]``:

* AIM speeds up when the price is in your favour and slows down when it is not. It is the tactic for markets that overshoot and come back (mean reversion): it buys the dip and waits out the spike.
* PIM does the reverse: it slows down when the price is in your favour, in case the move continues, and speeds up when it is against you, to limit how much a bad move can cost. It is the tactic for markets
  that trend, and the one that caps the damage of an adverse move.

Neither knows which kind of market it is in: each is a bet on the market's memory, and the sign of that memory decides which one pays (the tests show both directions). What the shares left over do is the
engine's job: whatever a tactic holds back is spread over the rest of the plan in its proportions, and the last interval finishes the order.

**Target cost** keeps the temporary impact you pay on track with what the plan expected. At every interval it takes the budget that is left (the plan's expected temporary impact less what has been paid) and
chooses, from the efficient family of schedules for the shares and intervals that are left (shapes from the quadratic model, cost from the market's own power law), the *least risky* one whose expected impact
fits the budget (minimise risk subject to a cost constraint on the remainder):
overspend, because volume dried up and your participation rose, and it slows down; underspend, because the market was deep, and it speeds up to cut the price risk. It tracks impact only: the price's own moves are
risk, not cost, and no tactic here pretends to control them.

All three are applied by wrapping a base algorithm: ``AIM(VWAP())``, ``PIM(ImplementationShortfall(1e-3), strength=0.5)``, ``TargetCost(ImplementationShortfall(1e-3))``.
"""

from __future__ import annotations

import numpy as np

from . import optimize as opt
from .impact import temporary_fraction
from .market import Market, Order, Scenario
from .simulate import Algo, State


class Tactic(Algo):
    """Base class: a tactic delegates planning to the algorithm it wraps and changes only the pace."""

    category = "tactic"

    def __init__(self, base: Algo):
        self.base = base

    @property
    def name(self) -> str:                                                                                      # type: ignore[override]
        return f"{self.tag}({self.base.name})"

    def prepare(self, order: Order, market: Market, scenario: Scenario) -> None:
        super().prepare(order, market, scenario)
        self.base.prepare(order, market, scenario)

    def plan(self):
        return self.base.plan()


class _PriceScaling(Tactic):
    sign = 1.0
    tag = "scaling"

    def __init__(self, base: Algo, strength: float = 0.35, floor: float = 0.25, cap: float = 4.0):
        super().__init__(base)
        if strength < 0 or not 0 < floor <= 1 <= cap:
            raise ValueError("strength >= 0 and 0 < floor <= 1 <= cap")
        self.strength, self.floor, self.cap = strength, floor, cap

    def multiplier(self, state: State) -> np.ndarray:
        """The factor on the planned rate: one when the price is where it began, above one on the side the tactic leans into."""
        return np.clip(np.exp(self.sign * self.strength * state.favourable()), self.floor, self.cap)

    def want(self, state: State) -> np.ndarray:
        return self.base.want(state) * self.multiplier(state)


class AIM(_PriceScaling):
    """Aggressive in the money: faster when the price is in your favour, slower when it is against you."""

    sign = 1.0
    tag = "aim"
    description = "Aggressive in the money: speeds up when the price has moved in your favour since arrival and slows down when it has not. Pays in markets that overshoot and revert."


class PIM(_PriceScaling):
    """Passive in the money: slower when the price is in your favour, faster when it is against you."""

    sign = -1.0
    tag = "pim"
    description = "Passive in the money: slows down when the price has moved in your favour (the move may continue) and speeds up when it has not, to limit the loss. Pays in trending markets."


class TargetCost(Tactic):
    tag = "target_cost"
    description = "Target cost: keeps the temporary impact paid on track with the plan's expected figure by re-choosing, at every interval, the least risky schedule for the shares left whose expected impact fits the budget left."

    def __init__(self, base: Algo, grid: int = 48):
        super().__init__(base)
        self.grid = grid

    def prepare(self, order: Order, market: Market, scenario: Scenario) -> None:
        super().prepare(order, market, scenario)
        plan = self.base.plan()
        if plan is None:
            raise ValueError("a target cost needs a base algorithm with a plan")
        window = order.window(market)
        V = market.expected_volume()[window]
        w = market.variance_weights()[window]
        ref, X = order.reference(market), order.shares
        n = len(V)
        rho = X / V.sum()
        k = ref * market.eta * market.sigma * rho ** (market.beta - 1.0) / V                                    # the quadratic model of the temporary impact, exact at the average participation
        self.target = float((ref * plan * temporary_fraction(plan, V, market.sigma, market.eta, market.beta)).sum())
        self.exponent = 1.0 + market.beta
        c_exact = ref * market.eta * market.sigma * V ** (-market.beta)                                         # the market's own power law: a slice q costs c_j q^(1 + beta)
        ras = np.r_[0.0, np.logspace(-7, 3, self.grid - 1)]
        lam = ras * 1e4 / (ref * X) * ref ** 2 * market.sigma ** 2
        cost = np.empty((n, len(ras)))
        first = np.empty((n, len(ras)))
        for i in range(n):
            m = n - i
            L = np.tril(np.ones((m, m)))
            LtW = L.T * w[i:]
            for j, lj in enumerate(lam):
                H = 2.0 * (np.diag(k[i:]) + lj * LtW @ L)
                g = -2.0 * lj * (LtW @ np.ones(m))
                q = opt.solve_qp(H, g, 1.0)
                cost[i, j] = float((c_exact[i:] * q ** self.exponent).sum())                                    # the shape is the quadratic model's, its cost is the power law's
                first[i, j] = q[0]
        self._cost = np.maximum.accumulate(cost, axis=1)
        self._first = first

    def want(self, state: State) -> np.ndarray:
        budget = np.maximum(self.target - state.temp_spent, 0.0)
        ratio = budget / np.maximum(state.remaining, 1.0) ** self.exponent                                      # impact the remainder can afford per share to the power 1 + beta
        c, f = self._cost[state.i], self._first[state.i]
        share = np.interp(ratio, c, f)                                                                          # the first slice of the cheapest-risk schedule that fits the budget
        return state.remaining * share
