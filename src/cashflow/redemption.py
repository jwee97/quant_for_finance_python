"""Meeting a redemption: which assets to sell, and what selling them costs.

A fund that must raise cash from its holdings can sell everything in proportion (the mix stays the same, the illiquid positions are sold as hard as the liquid ones), sell what has drifted above
target (the mix is repaired), or sell the most liquid assets first (cheapest to sell, but the book that is left is less liquid and further from target). :func:`redemption_cost` prices one
choice with the liquidation cost model of the execution algorithms (:mod:`src.algo.liquidation`: spread, power-law temporary impact and permanent impact for a constant participation rate), and
:func:`compare_redemption_policies` lines the policies up.
"""

from __future__ import annotations

import pandas as pd

from ..algo.liquidation import liquidation_profile
from .policies import POLICIES, allocate_flow


def redemption_cost(weights: pd.Series, aum: float, redemption: float, adv: pd.Series, sigma: pd.Series, policy: str = "pro_rata", targets: pd.Series | None = None, cash: float = 0.0,
                    participation: float = 0.10, spread=0.0004, **impact) -> dict:
    """The cost of raising ``redemption`` (a fraction of the fund) under ``policy``.

    ``weights`` are the fund's weights by asset, ``adv`` the average daily dollar volume and ``sigma`` the daily volatility of each asset, ``targets`` the weights the fund aims at (default: the weights
    it has now) and ``cash`` the cash it holds as a fraction of the fund (used by the ``cash`` policy). Returns ``sales`` (a table by asset: dollars sold, days to sell at ``participation`` of the
    day's dollar volume, cost and risk in basis points), ``cost_dollars``, ``cost_bps`` of the amount raised, ``days`` until the last sale is done, ``drift`` (half the sum of absolute differences
    between the weights after the sales and the targets, as a fraction of the fund left) and ``risk_bps`` (the average timing risk of the sales, weighted by their size)."""
    if not 0 < redemption < 1:
        raise ValueError("redemption is a fraction of the fund between 0 and 1")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    weights = pd.Series(weights, dtype=float)
    target = weights if targets is None else pd.Series(targets, dtype=float).reindex(weights.index).fillna(0.0)
    values = weights * aum
    trades = allocate_flow(values, target, -redemption * aum, policy, cash * aum, pd.Series(adv, dtype=float).reindex(weights.index))
    sold = (-trades).clip(lower=0.0)
    sales = liquidation_profile(sold, adv, sigma, spread=spread, participation=participation, **impact)
    sales = sales[sales["value"] > 0]
    raised = float(sold.sum())
    dollars = float(sales["dollars"].sum()) if len(sales) else 0.0
    after = (values + trades).clip(lower=0.0)
    left = float(after.sum()) + max(cash * aum - max(redemption * aum - raised, 0.0), 0.0)
    drift = 0.5 * float(((after / left) - target.reindex(after.index).fillna(0.0)).abs().sum()) if left > 0 else float("nan")
    weight = sales["value"].abs() / sales["value"].abs().sum() if len(sales) and sales["value"].abs().sum() > 0 else pd.Series(dtype=float)
    return {"sales": sales, "raised": raised, "cost_dollars": dollars, "cost_bps": 1e4 * dollars / raised if raised > 0 else 0.0, "cost_bps_of_fund": 1e4 * dollars / aum,
            "days": float(sales["days"].max()) if len(sales) else 0.0, "drift": drift, "risk_bps": float((sales["risk_bps"] * weight).sum()) if len(sales) else 0.0, "policy": policy}


def compare_redemption_policies(weights: pd.Series, aum: float, redemption: float, adv: pd.Series, sigma: pd.Series, policies=("pro_rata", "correct_drift", "liquid"), **kwargs) -> pd.DataFrame:
    """One row per policy: the cost of the redemption in basis points of what was raised and of the fund, the days to finish, the drift left behind and the sales' timing risk."""
    rows = {}
    for policy in policies:
        r = redemption_cost(weights, aum, redemption, adv, sigma, policy=policy, **kwargs)
        rows[policy] = {"cost_bps": r["cost_bps"], "cost_bps_of_fund": r["cost_bps_of_fund"], "days": r["days"], "drift_after": r["drift"], "risk_bps": r["risk_bps"], "assets_sold": int(len(r["sales"])),
                        "raised": r["raised"]}
    return pd.DataFrame(rows).T


__all__ = ["redemption_cost", "compare_redemption_policies"]
