"""Attribution and capacity for a pipeline result: where did the return come from, and how much money can the strategy run?

    brinson_vs_benchmark    allocation / selection / interaction by asset class against an equal-weight benchmark, Carino-linked so the effects
                            sum to the cumulative active return; leverage and cash are an explicit CASH sector so any book is admissible
    model_attribution       the combined book's return split into each forecast model's share of its own standalone return, the interaction
                            (stack, risk scaling, regime limits) and the costs; the rows add to the net return exactly
    cost_breakdown          linear (spread and commission) against square-root market impact when the specification gives an AUM
    capacity_by_aum         net Sharpe at a grid of AUM, and the AUM at which it halves
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.attribution import brinson_fachler, cumulative, monthly_inputs
from ..backtest.impact import ImpactSettings, capacity_curve, capacity_from_curve, impact_cost_frame, market_state, sharpe

DEFAULT_GRID = (1e7, 1e8, 1e9, 1e10, 1e11)


def brinson_vs_benchmark(result, bundle) -> pd.DataFrame:
    """Linked Brinson-Fachler effects, in return points, by asset class, for the evaluation window."""
    index = bundle.index
    held = result.weights.reindex(index)
    inv = bundle.investable.reindex(index).fillna(False).astype(float)
    bench_held = inv.div(inv.sum(axis=1).replace(0.0, np.nan), axis=0)
    a = monthly_inputs(held, bundle.returns, result.gross_returns)
    b = monthly_inputs(bench_held, bundle.returns)
    start = pd.Period(result.start, "M") + 1                                       # the first full month after the evaluation start
    months = a["weights"].index[a["weights"].index >= start]
    months = [m for m in months if a["weights"].loc[m].notna().any() and b["weights"].loc[m].notna().any()]
    if len(months) < 12:
        return pd.DataFrame()
    wp, wb, ret = a["weights"].loc[months].fillna(0.0).copy(), b["weights"].loc[months].fillna(0.0).copy(), a["asset_returns"].loc[months].fillna(0.0).copy()
    wp["CASH"], wb["CASH"], ret["CASH"] = 1.0 - wp.sum(axis=1), 1.0 - wb.sum(axis=1), 0.0
    sector_of = {c: bundle.asset_class.get(c, "unknown") for c in bundle.assets}
    sector_of["CASH"] = "cash / leverage"
    br = brinson_fachler(wp, wb, ret, sector_of)
    linked = (br.linked() * 100.0).round(10)
    engine_total = cumulative(a["portfolio_return"].loc[months])
    weights_total = cumulative(br.portfolio_return)
    recon = pd.DataFrame({"allocation": 0.0, "selection": 0.0, "interaction": 0.0, "total": 100.0 * (engine_total - weights_total)}, index=["reconciliation (daily drift and rebalancing vs start-of-month weights)"])
    table = pd.concat([linked, recon])
    table.loc["TOTAL (cumulative active return, points)"] = table.sum(axis=0)
    return table


def standalone_shares(panels: dict, rule: str, weights: pd.DataFrame | None) -> pd.DataFrame:
    """Each model's share of the combined forecast on each date, according to the combination rule."""
    names = list(panels)
    index = next(iter(panels.values())).mean.index
    if weights is not None:
        return weights.reindex(index)[names].ffill().fillna(1.0 / len(names))
    if rule == "equal":
        return pd.DataFrame(1.0 / len(names), index=index, columns=names)
    if rule == "confidence":
        raw = pd.DataFrame({n: panels[n].confidence.mean(axis=1) for n in names})
    elif rule == "precision":
        raw = pd.DataFrame({n: (1.0 / panels[n].std ** 2).replace([np.inf, -np.inf], np.nan).mean(axis=1) for n in names})
    else:
        raise ValueError(f"no share definition for combination rule '{rule}'")
    raw = raw.fillna(0.0)
    total = raw.sum(axis=1).replace(0.0, np.nan)
    return raw.div(total, axis=0).fillna(1.0 / len(names))


def model_attribution(result, gross_streams: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Rows: each model's share-weighted standalone gross return, the interaction, the costs; they sum to the net return (in return points)."""
    window = result.net_returns.loc[result.start:].dropna().index
    aligned = shares.shift(1).reindex(window)                                   # the share on the decision date earns the next day's return
    contrib = (aligned * gross_streams.reindex(window)).sum() * 100.0
    net, costs = float(result.net_returns.reindex(window).sum() * 100.0), float(result.costs.reindex(window).sum() * 100.0)
    rows = {f"model: {n}": float(contrib[n]) for n in contrib.index}
    rows["interaction (stack, risk scaling, limits)"] = net + costs - float(contrib.sum())
    rows["costs"] = -costs
    out = pd.Series(rows, name="return_points").to_frame()
    out.loc["NET RETURN (sum of daily net returns)"] = net
    return out


def cost_breakdown(result, trades_rates: pd.Series, sigma: pd.DataFrame, adv: pd.DataFrame, aum: float, settings: ImpactSettings) -> pd.DataFrame:
    window = result.net_returns.loc[result.start:].dropna().index
    cost, participation = impact_cost_frame(result.trades.reindex(window).fillna(0.0), aum, sigma, adv, trades_rates, settings)
    linear = (result.trades.reindex(window).fillna(0.0).abs() * trades_rates.reindex(result.trades.columns).to_numpy()[None, :]).sum(axis=1)
    total = cost.sum(axis=1)
    return pd.DataFrame({"annual_bps": {"spread and commission (linear)": float(1e4 * linear.mean() * 252), "market impact (square root)": float(1e4 * (total - linear).mean() * 252),
                                        "total": float(1e4 * total.mean() * 252)},
                         "mean_participation": {"total": float(participation.where(result.trades.reindex(window).abs() > 1e-12).stack().mean())}})


def capacity_by_aum(result, bundle, engine, grid=DEFAULT_GRID, coefficient: float = 1.0) -> tuple[pd.Series, object]:
    """Net Sharpe at each AUM in ``grid`` from the result's gross returns and trades, and the AUM at which it falls to half the linear-cost Sharpe."""
    volume = bundle.volume if bundle.volume is not None else bundle.prices * 0.0 + 1e9
    sigma, adv = market_state(bundle.prices, volume, bundle.returns)
    rates = engine.cost_model.rates(result.trades.columns)
    window = result.net_returns.loc[result.start:].index
    gross, trades = result.gross_returns.reindex(window), result.trades.reindex(window).fillna(0.0)
    curve = capacity_curve(gross, trades, grid, sigma, adv, rates, ImpactSettings(coefficient))
    linear = sharpe(gross - (trades.abs() * rates.reindex(trades.columns).to_numpy()[None, :]).sum(axis=1))
    return curve, capacity_from_curve(curve, linear)
