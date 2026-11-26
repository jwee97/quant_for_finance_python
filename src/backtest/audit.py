"""The audit trail of a weight-based backtest: from a signal to a trade to the money it made, one row per trade, reconciled to the result.

For every trade the engine made (a change of the book in one asset on one date) the trail records the chain that produced it:

``signal_date``, ``signal``  the date the decision was made and the signal's value then (what the target weights were built from);
``target``                  the weight the strategy asked for on that date;
``trade_date``              the date the engine struck the book (``signal_lag`` trading days later: the decision is never traded on the information of the day it used);
``pre_weight``, ``traded``, ``post_weight``  the weight that had drifted into the book, the change executed, and the book after it;
``cost``                    the trading cost of that trade as a fraction of equity (the cost model's rate for the asset times the weight traded);
``pnl``                     the asset's contribution to the gross return from the day after the trade until its next trade (``held_{t-1} x r_t`` summed over those days);
``days_held``               how many return days that covers.

Reconciliation (``reconcile``): the sum of every row's ``pnl`` equals the total gross return stream of the result, and the sum of ``cost`` equals its total costs, day by day, to rounding. A trail that does not
reconcile says so and by how much; nothing is forced to zero. Returns the gap, not a verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_trail(result, targets: pd.DataFrame, returns: pd.DataFrame, signal: pd.DataFrame | None = None, signal_lag: int | None = None, cost_rates: pd.Series | dict | None = None) -> pd.DataFrame:
    """One row per (trade date, asset) with a nonzero traded weight; see the module docstring. ``targets`` are the weights as the strategy stamped them (the same frame given to ``BacktestEngine.run``), ``returns``
    the asset returns, ``signal`` an optional frame of the signal values on the decision dates."""
    lag = int(signal_lag if signal_lag is not None else (result.config or {}).get("signal_lag", 1))
    held, traded = result.weights, result.trades
    cols = list(held.columns)
    idx = held.index
    ret = returns.reindex(idx).reindex(columns=cols).fillna(0.0)
    if cost_rates is None:
        bps = (result.config or {}).get("cost_bps", 0.0)
        per = (result.config or {}).get("per_asset_bps", {}) or {}
        rates = pd.Series({c: float(per.get(c, bps)) for c in cols}) / 1e4
    else:
        rates = pd.Series(cost_rates, dtype=float).reindex(cols).fillna(0.0)
    prior = held.shift(1)
    contrib = (prior.fillna(0.0) * ret)                                                         # the gross contribution of each asset on each day (the book held at the previous close)
    rows: list[dict] = []
    key: dict[tuple, int] = {}
    for i, d in enumerate(idx):
        for a in cols:
            q = float(traded.at[d, a]) if (d in traded.index and a in traded.columns and pd.notna(traded.at[d, a])) else 0.0
            if abs(q) <= 1e-12:
                continue
            dec = idx[i - lag] if i - lag >= 0 else None
            post = float(held.at[d, a]) if pd.notna(held.at[d, a]) else 0.0
            rows.append({"trade_date": d, "asset": a, "decision_date": dec,
                         "signal": None if (signal is None or dec is None or a not in signal.columns or dec not in signal.index or pd.isna(signal.at[dec, a])) else float(signal.at[dec, a]),
                         "target": None if (dec is None or a not in targets.columns or dec not in targets.index or pd.isna(targets.at[dec, a])) else float(targets.at[dec, a]),
                         "pre_weight": post - q, "traded": q, "post_weight": post, "cost": abs(q) * float(rates[a]), "pnl": 0.0, "days_held": 0})
            key[(d, a)] = len(rows) - 1
    last: dict[str, int | None] = {a: None for a in cols}
    for d in idx:
        for a in cols:
            c, w = float(contrib.at[d, a]), (0.0 if pd.isna(prior.at[d, a]) else float(prior.at[d, a]))
            if c != 0.0 or abs(w) > 0.0:
                if last[a] is None:                                                             # a position that no trade of the trail explains: reported as its own row, never dropped
                    rows.append({"trade_date": pd.NaT, "asset": a, "decision_date": None, "signal": None, "target": None, "pre_weight": 0.0, "traded": 0.0, "post_weight": w, "cost": 0.0, "pnl": 0.0, "days_held": 0})
                    last[a] = len(rows) - 1
                rows[last[a]]["pnl"] += c                                                       # the return of the book held at the previous close belongs to the latest trade struck before today
                rows[last[a]]["days_held"] += 1 if abs(w) > 0.0 else 0
            if (d, a) in key:
                last[a] = key[(d, a)]
    return pd.DataFrame(rows)


def reconcile(trail: pd.DataFrame, result, tolerance: float = 1e-9) -> dict:
    """The trail against the result: total gross P&L and total costs, with the gap and whether it is within ``tolerance``. A gap is reported, not hidden."""
    gross_total = float(result.gross_returns.fillna(0.0).sum())
    cost_total = float(result.costs.fillna(0.0).sum())
    pnl, cost = float(trail["pnl"].sum()) if len(trail) else 0.0, float(trail["cost"].sum()) if len(trail) else 0.0
    return {"gross_return_sum": gross_total, "trail_pnl": pnl, "pnl_gap": pnl - gross_total, "cost_sum": cost_total, "trail_cost": cost, "cost_gap": cost - cost_total, "tolerance": tolerance,
            "reconciled": abs(pnl - gross_total) <= tolerance and abs(cost - cost_total) <= tolerance, "trades": int(len(trail))}
