"""Constrained long-short books as an allocator: 130/30, 120/20, market neutral, dollar neutral and long-only, with sector and beta neutrality, a turnover limit and trading costs.

    constrained_long_short    every month-end, mean-variance on the model forecast and a trailing shrinkage covariance, under the named book and the neutrality limits, solved as a quadratic
                              programme (:mod:`src.equity.portfolio`); when a month has no solution the book holds what it has

The forecast is the combined forecast of the models in the spec, as for the other forecast-driven books, so it takes any model or combination. Data through the month-end only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..equity.portfolio import PRESETS, BookSpec, optimise_book, preset
from ..features.sleeves import month_end_dates
from .allocation import Allocator, Context
from .registry import register_allocator


@register_allocator("constrained_long_short", "Constrained long-short book (130/30, 120/20, market or dollar neutral, long-only) by mean-variance on the forecast, with optional sector and beta neutrality, a turnover limit and trading costs")
class ConstrainedLongShort(Allocator):
    """The textbook way to use a good ranking: buy the stocks the forecast likes, sell the ones it dislikes and fund part of the longs with the shorts, within limits that keep the bet on the ranking and off
    everything else. ``book`` names the gross exposures (``130_30``: 130% long and 30% short, net 100%; ``market_neutral``: 100% each side; ``dollar_neutral``: 50% each side; ``long_only``). Each
    month-end it solves

        maximise  mu'w - (risk_aversion / 2) w'Sigma w - 12 * cost * |w - h|     subject to the book, each name at most ``max_weight`` of the fund on either side,

    with ``mu`` the annualised forecast, ``Sigma`` the annualised shrinkage covariance of the trailing ``lookback`` days and ``h`` what the book holds after the month's moves. ``sector_neutral`` makes
    the net weight of each group (the bundle's groups, else its asset classes) equal to its share of the names, within ``group_tolerance``; ``beta_neutral`` holds the beta to the equal-weight market at
    ``beta_target`` within ``beta_tolerance``; ``max_turnover`` (a fraction of the fund per month, 0 for none) limits the trade. A universe too small for the limits has them widened just enough to hold
    the book. A month in which the programme has no solution leaves the book as it was."""

    def __init__(self, book: str = "130_30", risk_aversion: float = 5.0, max_weight: float = 0.10, sector_neutral: bool = False, group_tolerance: float = 0.02, beta_neutral: bool = False,
                 beta_target: float = 0.0, beta_tolerance: float = 0.05, max_turnover: float = 0.0, cost_bps: float = 5.0, lookback: int = 252, confidence_weighted: bool = False):
        if book not in PRESETS:
            raise ValueError(f"book must be one of {', '.join(PRESETS)}")
        if risk_aversion <= 0 or not 0 < max_weight <= 1 or group_tolerance < 0 or beta_tolerance < 0 or max_turnover < 0 or cost_bps < 0 or lookback < 60:
            raise ValueError("risk_aversion > 0, 0 < max_weight <= 1, tolerances, max_turnover and cost_bps >= 0, lookback >= 60")
        self.book, self.risk_aversion, self.max_weight = book, float(risk_aversion), float(max_weight)
        self.sector_neutral, self.group_tolerance = bool(sector_neutral), float(group_tolerance)
        self.beta_neutral, self.beta_target, self.beta_tolerance = bool(beta_neutral), float(beta_target), float(beta_tolerance)
        self.max_turnover, self.cost_bps, self.lookback, self.confidence_weighted = float(max_turnover), float(cost_bps), int(lookback), bool(confidence_weighted)

    def required_assets(self) -> int:
        return 4

    def spec_for(self, names: list, groups: dict | None, beta: np.ndarray | None, held: np.ndarray) -> BookSpec:
        n = len(names)
        base = preset(self.book)
        cap = max(self.max_weight, 1.3 * (base.gross_long + base.gross_short) / n)                                                   # a universe too small for the limit gets a cap that lets every dollar of the book sit on its own name
        return preset(self.book, max_long=cap, max_short=cap, risk_aversion=self.risk_aversion,
                      groups=groups if self.sector_neutral else None, group_tolerance=self.group_tolerance if self.sector_neutral and groups else None,
                      beta=beta if self.beta_neutral else None, beta_target=self.beta_target, beta_tolerance=self.beta_tolerance if self.beta_neutral and beta is not None else None,
                      max_turnover=self.max_turnover if self.max_turnover > 0 and held.sum() != 0 else None, trade_cost=self.cost_bps * 1e-4 * 12.0)

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.covariance import estimate_covariance, nearest_positive_definite

        if ctx.forecasts is None:
            raise ValueError("constrained_long_short needs forecasts")
        fc, bundle = ctx.forecasts, ctx.bundle
        scale = 252.0 / fc.horizon
        groups_all = dict(bundle.group or bundle.asset_class or {})
        market = bundle.returns.where(bundle.investable).mean(axis=1)
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        held, last = pd.Series(0.0, index=bundle.assets), None
        for origin in month_end_dates(bundle.index):
            position = bundle.index.get_loc(origin)
            if position < self.lookback or origin not in fc.mean.index:
                continue
            window = bundle.returns.iloc[position - self.lookback + 1:position + 1]
            mu_row = fc.mean.loc[origin]
            live = [a for a in bundle.assets if bool(bundle.investable.loc[origin, a]) and window[a].notna().mean() > 0.9 and np.isfinite(mu_row.get(a, np.nan))]
            if len(live) < self.required_assets():
                continue
            if last is not None:                                                                           # what the previous book has grown into by now
                growth = (bundle.prices.iloc[position] / bundle.prices.iloc[last]).fillna(1.0)
                held = held * growth
                total = float(held.abs().sum())
                held = held * (float(out.iloc[last].abs().sum()) / total) if total > 0 else held
            mu = mu_row[live] * scale
            if self.confidence_weighted:
                mu = mu * fc.confidence.loc[origin, live].fillna(0.5)
            cov = nearest_positive_definite(estimate_covariance(window[live].fillna(0.0), "shrinkage", self.lookback, annualise=True)).to_numpy()
            m = market.iloc[position - self.lookback + 1:position + 1]
            var = float(m.var())
            beta = np.array([window[a].fillna(0.0).cov(m) / var if var > 0 else 1.0 for a in live])
            groups = {a: groups_all[a] for a in live if a in groups_all}
            held_live = held.reindex(live).fillna(0.0).to_numpy()
            spec = self.spec_for(live, groups, beta, held_live)
            result = optimise_book(mu.to_numpy(), cov, spec, held=held_live, names=live, max_iter=12000)
            if not result.ok:
                continue
            row = pd.Series(0.0, index=bundle.assets)
            row[live] = result.weights.to_numpy()
            out.loc[origin] = row
            held, last = row, position
        return out.ffill().fillna(0.0)
