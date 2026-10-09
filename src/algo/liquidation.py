"""Liquidation costs: how long a portfolio takes to sell and what that costs.

A position worth ``X`` dollars in a stock that trades ``ADV`` dollars a day cannot be sold at once: at a share ``p`` of the day's volume it takes ``X / (p ADV)`` days, during which the price can move
against you (the timing risk) and your own selling moves it (impact). :func:`liquidation_profile` prices that for every position with the same cost model as the execution algorithms
(:mod:`src.algo.impact`: half the spread, power-law temporary impact ``eta sigma (q/V)^beta`` on each day's slice, and the permanent impact ``gamma sigma X / (2 ADV)``) for a CONSTANT-RATE liquidation,
and :func:`liquidation_horizon` says how long it takes to turn a given share of the whole portfolio into cash. The same arithmetic caps positions in the ``liquidity_cap`` allocator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def liquidation_profile(values, adv, sigma, spread=0.0004, participation: float = 0.10, eta: float = 0.142, beta: float = 0.6, gamma: float = 0.30) -> pd.DataFrame:
    """The liquidation of each position at ``participation`` of the day's dollar volume, one slice per day.

    ``values`` are signed dollar positions by ticker, ``adv`` the average daily dollar volume, ``sigma`` the daily return volatility (0.012 = 1.2%), ``spread`` the quoted spread as a fraction of price
    (a scalar or by ticker). Returns one row per ticker with ``value``, ``days`` to liquidate, the cost in basis points of the position's value split into ``spread_bps``, ``temporary_bps`` and
    ``permanent_bps`` (``cost_bps`` their sum), the cost in ``dollars``, and ``risk_bps``: the standard deviation of the shortfall that price moves add over the days it takes, in basis points of the value."""
    if not 0 < participation <= 1:
        raise ValueError("0 < participation <= 1")
    values = pd.Series(values, dtype=float)
    adv, sigma = pd.Series(adv, dtype=float).reindex(values.index), pd.Series(sigma, dtype=float).reindex(values.index)
    spread = pd.Series(spread, index=values.index, dtype=float) if np.isscalar(spread) else pd.Series(spread, dtype=float).reindex(values.index)
    rows = {}
    for ticker, value in values.items():
        x, v, s, half = abs(float(value)), float(adv[ticker]), float(sigma[ticker]), 0.5 * float(spread[ticker])
        if x <= 0 or not np.isfinite(v) or v <= 0 or not np.isfinite(s):
            rows[ticker] = {"value": float(value), "days": 0.0 if x <= 0 else np.inf, "spread_bps": 0.0, "temporary_bps": 0.0, "permanent_bps": 0.0, "cost_bps": 0.0 if x <= 0 else np.inf,
                            "dollars": 0.0 if x <= 0 else np.inf, "risk_bps": 0.0 if x <= 0 else np.inf}
            continue
        per_day = participation * v
        full, rest = int(x // per_day), x - int(x // per_day) * per_day
        slices = np.array([per_day] * full + ([rest] if rest > 1e-9 * x else []))
        temporary = float((slices * eta * s * (slices / v) ** beta).sum() / x)                               # the volume left to others is the whole day's ADV: q / V with V = adv
        permanent = 0.5 * gamma * s * x / v
        remaining = x - np.cumsum(slices)                                                                  # still held at the end of each day
        risk = s * float(np.sqrt((remaining ** 2).sum()) / x) if len(slices) > 1 else 0.0
        rows[ticker] = {"value": float(value), "days": x / per_day, "spread_bps": 1e4 * half, "temporary_bps": 1e4 * temporary, "permanent_bps": 1e4 * permanent,
                        "cost_bps": 1e4 * (half + temporary + permanent), "dollars": x * (half + temporary + permanent), "risk_bps": 1e4 * risk}
    return pd.DataFrame(rows).T[["value", "days", "spread_bps", "temporary_bps", "permanent_bps", "cost_bps", "dollars", "risk_bps"]]


def liquidation_horizon(values, adv, participation: float = 0.10, shares=(0.5, 0.9, 0.99)) -> dict:
    """Days to turn each share of the portfolio's gross value into cash when every position is sold in parallel at ``participation`` of its own dollar volume.

    A position that cannot be sold at all (no volume data) never counts as liquidated, so a share the portfolio cannot reach comes back as ``inf``."""
    values = pd.Series(values, dtype=float)
    adv = pd.Series(adv, dtype=float).reindex(values.index)
    gross = float(values.abs().sum())
    if gross <= 0:
        return {s: 0.0 for s in shares}
    rate = (participation * adv).where(adv > 0, 0.0).fillna(0.0).to_numpy()
    size = values.abs().to_numpy()
    out = {}
    for share in shares:
        lo, hi = 0.0, float(np.max(np.where(rate > 0, size / np.maximum(rate, 1e-300), 0.0)))
        if np.minimum(size, rate * hi).sum() < share * gross * (1.0 - 1e-9):
            out[share] = float("inf")
            continue
        for _ in range(60):                                                                                  # the liquidated value is increasing in the days: bisect
            mid = 0.5 * (lo + hi)
            if np.minimum(size, rate * mid).sum() >= share * gross:
                hi = mid
            else:
                lo = mid
        out[share] = float(hi)
    return out
