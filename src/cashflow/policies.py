"""What to buy with a deposit and what to sell to pay a withdrawal.

Money moves in and out of a portfolio for reasons that have nothing to do with the strategy: contributions arrive, investors redeem, dividends are paid, liabilities fall due. Each such cash flow
is a trading decision, and the choice of which assets to trade decides how far the portfolio drifts from its target, how much turnover it creates and how much of the flow is spent on costs.

``allocate_flow`` turns one flow into dollar trades under a named policy:

    pro_rata        trade every asset in proportion to its TARGET weight (the flow is added to, or taken from, the whole portfolio alike; drift is left as it is)
    correct_drift   use the flow to repair drift: a deposit buys the assets below their target value, a withdrawal sells those above it; only when the flow exceeds all the drift is the rest pro rata
    rebalance       after the flow, trade every asset back to its target weight of the new total (corrects all drift, trades the most)
    cash            a deposit waits in cash, a withdrawal is paid from cash (and pro rata for anything cash cannot cover)
    liquid          (withdrawals) sell the most liquid assets first: the cheapest to sell, the most drift

Everything is in dollars. ``values`` are the current dollar positions by asset (signed), ``target`` the target weights.
"""

from __future__ import annotations

import pandas as pd

POLICIES = ("pro_rata", "correct_drift", "rebalance", "cash", "liquid")


def _weights_of(target: pd.Series) -> pd.Series:
    """The target scaled to sum to one (the shares in which a pro-rata flow is split); all zeros stay zeros."""
    total = float(target.sum())
    return target / total if abs(total) > 1e-12 else target * 0.0


def allocate_flow(values: pd.Series, target: pd.Series, flow: float, policy: str = "correct_drift", cash: float = 0.0, liquidity: pd.Series | None = None) -> pd.Series:
    """Dollar trades (positive buys, negative sells) that put a deposit (``flow > 0``) to work or raise the cash for a withdrawal (``flow < 0``).

    For ``pro_rata``, ``correct_drift`` and ``liquid`` the trades add up to ``flow``: the whole flow is invested or raised. ``rebalance`` also invests (or raises) whatever cash is held, so that the
    assets end at their target weights of everything; ``cash`` trades nothing for a deposit and, for a withdrawal, only what ``cash`` cannot cover. ``liquidity`` (any score where more means easier to
    sell, such as average dollar volume) orders the sales of the ``liquid`` policy."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    values = pd.Series(values, dtype=float)
    target = pd.Series(target, dtype=float).reindex(values.index).fillna(0.0)
    flow = float(flow)
    zero = pd.Series(0.0, index=values.index)
    if flow == 0.0 and policy != "rebalance":
        return zero
    share = _weights_of(target)
    if policy == "pro_rata":
        return share * flow
    if policy == "rebalance":
        nav = float(values.sum()) + float(cash) + flow
        return target * nav - values
    if policy == "cash":
        if flow >= 0:
            return zero
        short = max(-flow - max(float(cash), 0.0), 0.0)
        return share * -short
    if policy == "liquid":
        if flow > 0:
            return share * flow
        need, trades = -flow, zero.copy()
        order = (liquidity.reindex(values.index).fillna(0.0) if liquidity is not None else values.abs()).sort_values(ascending=False).index
        for asset in order:
            if need <= 1e-12:
                break
            sell = min(max(float(values[asset]), 0.0), need)
            trades[asset] = -sell
            need -= sell
        if need > 1e-12:
            trades = trades - share * need                                                                      # nothing liquid left to sell: the rest comes pro rata
        return trades
    desired = target * (float(values.sum()) + flow)                                                              # correct_drift: where each position would sit if the invested total grew by the flow
    gap = desired - values
    if flow > 0:
        short = gap.clip(lower=0.0)
        room = float(short.sum())
        if room >= flow:
            return short * (flow / room) if room > 0 else zero
        return short + share * (flow - room)
    excess = (-gap).clip(lower=0.0)
    room, need = float(excess.sum()), -flow
    if room >= need:
        return -excess * (need / room) if room > 0 else zero
    return -excess - share * (need - room)
