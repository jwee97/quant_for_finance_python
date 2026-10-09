"""The named algorithms: the schedules of a trading desk's screen, each a small class the simulator can run.

    TWAP, VWAP, POV           the benchmark-tracking algorithms: equal slices, slices in proportion to the expected volume, a fixed share of the volume as it prints
    ArrivalPrice              front-loaded to stay near the price at the moment the order arrived; ``urgency`` sets how fast
    ImplementationShortfall   the Almgren-Chriss optimum for a risk aversion, optionally with a view on the price drift (alpha)
    MinCost, MinCostRisk, MinRiskCost, Balanced, PriceImprovement     the best-execution goals of :mod:`src.algo.optimize`
    ExponentialTrade, ExponentialResidual, TradeRate                 the one-parameter schedule families (the parameter is fitted unless given)

Every algorithm plans in the order's own window with expected volumes and the simulator finishes what falls behind in the plan's proportions. Which orders to send for each slice is the job of a placement
:class:`~src.algo.simulate.Style`.
"""

from __future__ import annotations

import numpy as np

from . import optimize as opt
from .market import Market, Order, Scenario
from .simulate import Algo, State


class _Planned(Algo):
    """An algorithm whose answer is a fixed schedule computed at the start."""

    def __init__(self):
        self._plan = None

    def prepare(self, order: Order, market: Market, scenario: Scenario) -> None:
        super().prepare(order, market, scenario)
        self._plan = np.asarray(self.build(order, market), float)
        if abs(self._plan.sum() - order.shares) > 1e-6 * order.shares or self._plan.min() < -1e-9:
            raise RuntimeError(f"{self.name} produced an invalid schedule")

    def build(self, order: Order, market: Market) -> np.ndarray:
        raise NotImplementedError

    def plan(self) -> np.ndarray:
        return self._plan


class TWAP(_Planned):
    name, category = "twap", "benchmark"
    description = "Time-weighted average price: the same number of shares in every interval, whatever the market does."

    def build(self, order, market):
        n = order.length(market)
        return np.full(n, order.shares / n)


class VWAP(_Planned):
    name, category = "vwap", "benchmark"
    description = "Volume-weighted average price: slices in proportion to the volume expected in each interval, so the order follows the market's own rhythm (the cheapest schedule when risk does not matter)."

    def build(self, order, market):
        v = market.expected_volume()[order.window(market)]
        return order.shares * v / v.sum()


class POV(Algo):
    """Participate: trade ``rate`` of the market's volume as it prints. The order finishes when its shares run out, which is earlier than the horizon for a high rate, or in the last interval for a low one."""

    name, category = "pov", "benchmark"
    description = "Percentage of volume: a fixed share of the volume that actually prints, so it speeds up in a busy market and slows down in a quiet one. Finishes early at a high rate."

    def __init__(self, rate: float = 0.10):
        if not 0 < rate < 1:
            raise ValueError("rate must be between 0 and 1")
        self.rate = rate

    def want(self, state: State) -> np.ndarray:
        return self.rate / (1.0 - self.rate) * state.volume


class ImplementationShortfall(_Planned):
    name, category = "is", "shortfall"
    description = "Implementation shortfall (Perold; Almgren-Chriss): the schedule minimising expected cost plus a risk-aversion times the variance, optionally leaning with an expected price drift (alpha)."

    def __init__(self, risk_aversion: float = 1e-3, alpha_bps: float = 0.0):
        super().__init__()
        self.risk_aversion, self.alpha_bps = risk_aversion, alpha_bps

    def build(self, order, market):
        return opt.optimal_schedule(order, market, self.risk_aversion, self.alpha_bps)


class ArrivalPrice(_Planned):
    name, category = "arrival_price", "shortfall"
    description = "Arrival price: front-loaded to stay close to the price when the order arrived. ``urgency`` between 0 (as patient as VWAP) and 1 (nearly everything at once) sets the risk aversion; no view on drift."

    def __init__(self, urgency: float = 0.5):
        super().__init__()
        if not 0 <= urgency <= 1:
            raise ValueError("urgency must be between 0 and 1")
        self.urgency = urgency

    def build(self, order, market):
        risk_aversion = 0.0 if self.urgency == 0 else 10.0 ** (-4.0 + 5.0 * self.urgency)           # 1e-4 (patient) to 10 (impatient)
        return opt.optimal_schedule(order, market, risk_aversion, 0.0)


class MinCost(_Planned):
    name, category = "min_cost", "goal"
    description = "Minimise cost: the cheapest schedule whatever the risk (the VWAP schedule unless there is a view on drift)."

    def __init__(self, alpha_bps: float = 0.0):
        super().__init__()
        self.alpha_bps = alpha_bps

    def build(self, order, market):
        return opt.min_cost(order, market, self.alpha_bps)


class MinCostRisk(_Planned):
    name, category = "min_cost_risk", "goal"
    description = "Minimise cost with a risk constraint: the cheapest schedule whose timing risk stays under ``max_risk_bps``."

    def __init__(self, max_risk_bps: float = 60.0, alpha_bps: float = 0.0):
        super().__init__()
        self.max_risk_bps, self.alpha_bps = max_risk_bps, alpha_bps

    def build(self, order, market):
        return opt.min_cost_given_risk(order, market, self.max_risk_bps, self.alpha_bps)["schedule"]


class MinRiskCost(_Planned):
    name, category = "min_risk_cost", "goal"
    description = "Minimise risk with a cost constraint: the least risky schedule whose expected cost stays under ``max_cost_bps`` (the cheapest schedule if the cap is out of reach)."

    def __init__(self, max_cost_bps: float = 20.0, alpha_bps: float = 0.0):
        super().__init__()
        self.max_cost_bps, self.alpha_bps = max_cost_bps, alpha_bps

    def build(self, order, market):
        return opt.min_risk_given_cost(order, market, self.max_cost_bps, self.alpha_bps)["schedule"]


class Balanced(_Planned):
    name, category = "balanced", "goal"
    description = "Balance cost and risk: the standard cost-risk trade-off for the investor's risk aversion."

    def __init__(self, risk_aversion: float = 1e-3, alpha_bps: float = 0.0):
        super().__init__()
        self.risk_aversion, self.alpha_bps = risk_aversion, alpha_bps

    def build(self, order, market):
        return opt.balanced(order, market, self.risk_aversion, self.alpha_bps)["schedule"]


class PriceImprovement(_Planned):
    name, category = "price_improvement", "goal"
    description = "Price improvement: the schedule with the best chance that the shortfall comes in below ``target_bps``."

    def __init__(self, target_bps: float = 30.0, alpha_bps: float = 0.0):
        super().__init__()
        self.target_bps, self.alpha_bps = target_bps, alpha_bps

    def build(self, order, market):
        return opt.price_improvement(order, market, self.target_bps, self.alpha_bps)["schedule"]


class ExponentialTrade(_Planned):
    name, category = "exp_trade", "technique"
    description = "Trade schedule exponential: the trade rate decays like exp(-kappa t); kappa is fitted to cost plus risk unless given."

    def __init__(self, kappa: float | None = None, risk_aversion: float = 1e-3):
        super().__init__()
        self.kappa, self.risk_aversion = kappa, risk_aversion

    def build(self, order, market):
        if self.kappa is None:
            fit = opt.fit_exponential_trade(order, market, self.risk_aversion)
            self.kappa = fit["kappa"]
            return fit["schedule"]
        return opt.exponential_trade(order, market, self.kappa)


class ExponentialResidual(_Planned):
    name, category = "exp_residual", "technique"
    description = "Residual schedule exponential: the shares still to trade decay like X exp(-kappa t); each interval trades a fixed fraction of what is left and the last sweeps the rest."

    def __init__(self, kappa: float | None = None, risk_aversion: float = 1e-3):
        super().__init__()
        self.kappa, self.risk_aversion = kappa, risk_aversion

    def build(self, order, market):
        if self.kappa is None:
            fit = opt.fit_exponential_residual(order, market, self.risk_aversion)
            self.kappa = fit["kappa"]
            return fit["schedule"]
        return opt.exponential_residual(order, market, self.kappa)


class TradeRate(_Planned):
    name, category = "trade_rate", "technique"
    description = "Trade rate parameter: one participation rate describes the whole strategy; the rate is fitted to cost plus risk unless given."

    def __init__(self, rate: float | None = None, risk_aversion: float = 1e-3):
        super().__init__()
        self.rate, self.risk_aversion = rate, risk_aversion

    def build(self, order, market):
        if self.rate is None:
            fit = opt.fit_trade_rate(order, market, self.risk_aversion)
            self.rate = fit["rate"]
            return fit["schedule"]
        return opt.trade_rate(order, market, self.rate)


ALGORITHMS = {cls.name: cls for cls in (TWAP, VWAP, POV, ImplementationShortfall, ArrivalPrice, MinCost, MinCostRisk, MinRiskCost, Balanced, PriceImprovement, ExponentialTrade, ExponentialResidual, TradeRate)}
