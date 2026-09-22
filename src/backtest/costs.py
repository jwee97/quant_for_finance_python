"""Transaction costs (Ch. 22 §22.2.2, spec §22).

Baseline model, linear in traded notional::

    TO_t = sum_i |w_{i,t} - w_{i,t-1}|
    TC_t = c * TO_t
    R_t^net = R_t^gross - TC_t

Two refinements that matter for an ETF universe:

*Per-asset costs.* DBC does not trade like SPY. A single universe-wide
spread assumption flatters exactly the strategies that concentrate in the
illiquid names, which is the opposite of prudent.

*Cost sensitivity as a first-class output.* The question is never "what is
the net Sharpe at 10 bps", it is "at what cost level does this strategy stop
working?". ``breakeven_cost`` answers that directly, and a strategy whose
breakeven cost is below a plausible real-world spread is dead regardless of
how good its gross numbers look.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LinearCostModel:
    """``cost = (bps / 10000) * traded notional``.

    ``cost_bps`` is the *one-way* cost of trading 100% of capital, applied to
    the absolute weight change. A round trip therefore costs twice this.
    """

    cost_bps: float = 10.0
    per_asset_bps: dict[str, float] = field(default_factory=dict)

    def rates(self, columns) -> pd.Series:
        return pd.Series(
            {c: float(self.per_asset_bps.get(c, self.cost_bps)) for c in columns}, dtype=float
        ) / 1e4

    def trade_costs(self, trades: pd.DataFrame) -> pd.Series:
        """Cost per date given a frame of weight changes."""
        rates = self.rates(trades.columns)
        return (trades.abs() * rates).sum(axis=1)

    def apply(self, gross_returns: pd.Series, trades: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        costs = self.trade_costs(trades).reindex(gross_returns.index).fillna(0.0)
        return gross_returns - costs, costs

    def describe(self) -> dict:
        return {
            "model": "linear",
            "cost_bps": self.cost_bps,
            "n_overrides": len(self.per_asset_bps),
            "mean_effective_bps": float(np.mean(list(self.per_asset_bps.values())))
            if self.per_asset_bps else self.cost_bps,
        }


def turnover(weights: pd.DataFrame) -> pd.Series:
    """``sum_i |w_{i,t} - w_{i,t-1}|``, with the first row counted as the
    cost of building the initial book rather than as free."""
    trades = weights.diff()
    trades.iloc[0] = weights.iloc[0]
    return trades.abs().sum(axis=1)


def trades_from_weights(target: pd.DataFrame, held: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weight changes actually traded.

    When ``held`` is supplied (the drifted book), trades are measured against
    what is genuinely in the portfolio rather than against last period's
    target. This matters: a buy-and-hold book has *zero* turnover, but
    comparing consecutive targets would show turnover every time prices move.
    """
    if held is None:
        trades = target.diff()
        trades.iloc[0] = target.iloc[0]
        return trades
    aligned = held.reindex_like(target)
    trades = target - aligned
    trades.iloc[0] = target.iloc[0]
    return trades


def cost_sensitivity(gross_returns: pd.Series, trades: pd.DataFrame, levels=(0.0, 5.0, 10.0, 25.0, 50.0),
                     per_asset_bps: dict[str, float] | None = None,
                     periods_per_year: int = 252) -> pd.DataFrame:
    """Net performance across a grid of cost assumptions (spec §22).

    When ``per_asset_bps`` is given, each level scales the per-asset profile
    rather than replacing it, so the relative liquidity ranking is preserved.
    """
    base = float(np.mean(list(per_asset_bps.values()))) if per_asset_bps else None
    rows = []
    for level in levels:
        if per_asset_bps and base:
            scaled = {k: v * level / base for k, v in per_asset_bps.items()}
            model = LinearCostModel(level, scaled)
        else:
            model = LinearCostModel(level)
        net, costs = model.apply(gross_returns, trades)
        clean = net.dropna()
        vol = clean.std(ddof=1) * np.sqrt(periods_per_year)
        rows.append(
            {
                "cost_bps": level,
                "net_ann_return": float(clean.mean() * periods_per_year),
                "net_vol": float(vol),
                "net_sharpe": float(clean.mean() * periods_per_year / vol) if vol > 0 else np.nan,
                "annual_cost_drag": float(costs.mean() * periods_per_year),
                "cost_share_of_gross": float(
                    costs.mean() / gross_returns.mean()) if gross_returns.mean() > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def breakeven_cost(gross_returns: pd.Series, trades: pd.DataFrame,
                   periods_per_year: int = 252) -> float:
    """Cost in bps at which the net return is exactly zero.

    ``breakeven = mean gross return / mean turnover``, in basis points. The
    single most useful number for deciding whether a signal is implementable:
    if it is 3 bps, no execution desk can save the strategy.
    """
    aligned_turnover = trades.abs().sum(axis=1).reindex(gross_returns.index).fillna(0.0)
    mean_turnover = float(aligned_turnover.mean())
    mean_gross = float(gross_returns.mean())
    if mean_turnover <= 0:
        return float("inf") if mean_gross > 0 else float("nan")
    return mean_gross / mean_turnover * 1e4


def turnover_statistics(weights: pd.DataFrame, trades: pd.DataFrame | None = None,
                        periods_per_year: int = 252) -> dict:
    trades = trades if trades is not None else trades_from_weights(weights)
    daily = trades.abs().sum(axis=1)
    return {
        "mean_daily_turnover": float(daily.mean()),
        "annualised_turnover": float(daily.mean() * periods_per_year),
        "median_daily_turnover": float(daily.median()),
        "max_daily_turnover": float(daily.max()),
        "share_of_days_trading": float((daily > 1e-8).mean()),
        "mean_turnover_on_trading_days": float(daily[daily > 1e-8].mean()) if (daily > 1e-8).any() else 0.0,
    }
