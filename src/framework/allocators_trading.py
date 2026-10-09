"""Trading-aware portfolio construction: a book that plans several months ahead because trading costs money and the forecast fades.

    multi_period    every month-end, plan the holdings for the next few months on a decaying forecast path with proportional and impact costs and the book's limits (model-predictive control),
                    trade the first step, and re-plan when the next forecast arrives (:mod:`src.equity.multiperiod`)

The forecast is the combined forecast of the models in the spec, as for the other forecast-driven books. Data through the month-end only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..equity.multiperiod import cost_matrix, decaying_path, mpc_step
from ..equity.portfolio import PRESETS
from ..features.sleeves import month_end_dates
from .allocation import Allocator, Context
from .registry import register_allocator


@register_allocator("multi_period", "Multi-period trading: plan the holdings of the next months on a fading forecast with trading costs and the book's limits, trade the first step, re-plan next month")
class MultiPeriodBook(Allocator):
    """A one-period optimiser trades to wherever today's forecast says, and pays for it again next month when the forecast has moved. This book looks ahead: the expected return of a position fades by
    ``persistence`` each month (``auto``: the correlation of the forecast with last month's), so it plans the next ``horizon`` months, charges ``cost_bps`` on every unit traded and an impact cost
    ``impact`` times the covariance on the square of the trade, and takes the first step of the plan. Costs make it trade less than a one-period book: it moves part of the way to where it wants to be and
    does nothing while the position is close enough (the no-trade region of a proportional cost). ``book`` names the limits (``long_only``: fully invested; ``130_30``, ``market_neutral`` ...: net
    exposure fixed and gross exposure at most the book's). The plan is a quadratic programme over ``plan_horizon`` months with the unconstrained value of the rest of the path as the terminal
    reward (Skaf and Boyd's approximate dynamic programming); a month with no solution leaves the book as it was."""

    def __init__(self, book: str = "long_only", risk_aversion: float = 5.0, horizon: int = 6, plan_horizon: int = 3, persistence="auto", cost_bps: float = 10.0, impact: float = 0.0,
                 max_weight: float = 0.10, lookback: int = 252, discount: float = 1.0):
        if book not in PRESETS:
            raise ValueError(f"book must be one of {', '.join(PRESETS)}")
        if risk_aversion <= 0 or horizon < 1 or plan_horizon < 1 or cost_bps < 0 or impact < 0 or not 0 < max_weight <= 1 or lookback < 60 or not 0 < discount <= 1:
            raise ValueError("risk_aversion > 0, horizon and plan_horizon >= 1, costs >= 0, 0 < max_weight <= 1, lookback >= 60, 0 < discount <= 1")
        if persistence != "auto" and not 0.0 <= float(persistence) <= 1.0:
            raise ValueError("persistence is 'auto' or a number between 0 and 1")
        self.book, self.risk_aversion, self.horizon, self.plan_horizon = book, float(risk_aversion), int(horizon), int(plan_horizon)
        self.persistence = persistence if persistence == "auto" else float(persistence)
        self.cost_bps, self.impact, self.max_weight, self.lookback, self.discount = float(cost_bps), float(impact), float(max_weight), int(lookback), float(discount)

    def required_assets(self) -> int:
        return 4

    def constraints(self, n: int) -> dict:
        base = PRESETS[self.book]
        gl, gs = base["gross_long"], base["gross_short"]
        cap = max(self.max_weight, 1.3 * (gl + gs) / n)                                          # a universe too small for the limit gets one that lets the book be fully invested
        out = {"lb": -cap if gs > 0 else 0.0, "ub": cap, "net": (gl - gs, gl - gs)}
        if gs > 0:
            out["max_gross"] = gl + gs
        return out

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.covariance import estimate_covariance, nearest_positive_definite

        if ctx.forecasts is None:
            raise ValueError("multi_period needs forecasts")
        fc, bundle = ctx.forecasts, ctx.bundle
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        held, last, previous, correlations = pd.Series(0.0, index=bundle.assets), None, None, []
        for origin in month_end_dates(bundle.index):
            position = bundle.index.get_loc(origin)
            if position < self.lookback or origin not in fc.mean.index:
                continue
            window = bundle.returns.iloc[position - self.lookback + 1:position + 1]
            mu_row = fc.mean.loc[origin]
            live = [a for a in bundle.assets if bool(bundle.investable.loc[origin, a]) and window[a].notna().mean() > 0.9 and np.isfinite(mu_row.get(a, np.nan))]
            if len(live) < self.required_assets():
                continue
            if previous is not None:                                                                  # how persistent the forecast is: its cross-sectional correlation with last month's
                both = [a for a in live if a in previous.index and np.isfinite(previous[a])]
                if len(both) >= 8 and mu_row[both].std() > 0 and previous[both].std() > 0:
                    correlations.append(float(np.corrcoef(mu_row[both], previous[both])[0, 1]))
            previous = mu_row.copy()
            if last is not None:                                                                      # what the book has grown into by now
                growth = (bundle.prices.iloc[position] / bundle.prices.iloc[last]).fillna(1.0)
                held = held * growth
                total = float(held.abs().sum())
                held = held * (float(out.iloc[last].abs().sum()) / total) if total > 0 else held
            phi = self.persistence if self.persistence != "auto" else float(np.clip(np.mean(correlations[-12:]), 0.0, 0.95)) if len(correlations) >= 3 else 0.5
            cov = nearest_positive_definite(estimate_covariance(window[live].fillna(0.0), "shrinkage", self.lookback, annualise=True)).to_numpy() * (fc.horizon / 252.0)
            path = decaying_path(mu_row[live].to_numpy(), phi, self.horizon)
            x_prev = held.reindex(live).fillna(0.0).to_numpy()
            Lam = cost_matrix(cov, self.impact, "risk")
            kappa = self.cost_bps * 1e-4
            limits = self.constraints(len(live))
            x, plan = mpc_step(path, cov, self.risk_aversion, x_prev, Lam, kappa if kappa > 0 else None, self.discount, plan_horizon=self.plan_horizon, **limits)
            if not plan.ok:                                                                           # no solution this month: the book stays as it was
                continue
            row = pd.Series(0.0, index=bundle.assets)
            row[live] = x
            out.loc[origin] = row
            held, last = row, position
        return out.ffill().fillna(0.0)
