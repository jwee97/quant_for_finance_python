"""High-frequency research simulators: auto market making and rebate / liquidity trading from order-flow pressure.

**Auto market making** (:func:`auto_market_making`) is the Avellaneda-Stoikov market maker of :mod:`src.microstructure.market_making` in the vocabulary of this package: quotes shaded by inventory, against symmetric
quotes with the same average spread, with a share of fills coming from informed traders. The decomposition of its profit (spread captured, price moves on inventory, losses to informed flow) is the point.

**Rebate and liquidity trading** (:func:`simulate_rebate_trading`) is a maker that lives on exchange rebates and spread, and whose problem is that fills are not random. The market's signed order flow ``xi``
(positive: buyers pushing) is persistent and moves the mid price a step later, ``mid[t + 1] - mid[t] = lam xi[t] + noise``, and it also decides which of our quotes gets hit: buyers lift our ask, sellers hit
our bid. So a passive ask fills most when the price is about to rise (we sold too cheap) and a passive bid when it is about to fall: *adverse selection*. A **naive** maker quotes both sides always and pays
that tax. A **pressure-aware** maker estimates the pressure from the flow it can see (``latency`` steps old), and withdraws the side that the pressure is about to hit: it keeps the rebate and spread of the
safe side and skips most of the losing fills, so it also tends to hold inventory in the direction the price is about to move. The profit of each is split into rebate, spread captured and ``price_pnl`` (what the
price did to the inventory: ``adverse_selection`` is its first step after each fill, ``inventory_pnl`` the rest, because flow persists and prices keep moving). The aware maker's edge fades as its view of the
flow gets older (flow is only predictable while it persists).

Everything is stylised: one share per fill, no queue priority, the fill probabilities and the impact of flow are parameters. The aware maker's edge is built into the world it is tested in (flow does move
prices); the lesson is its *size* against rebate and spread, and how latency erodes it, not that the world is like this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..microstructure.market_making import compare_strategies


def auto_market_making(n_paths: int = 1000, steps: int = 600, informed_fraction: float = 0.2, gamma: float = 0.1, sigma: float = 2.0, k: float = 1.5, A: float = 140.0,
                       max_inventory: int | None = 20, seed: int = 0) -> pd.DataFrame:
    """Avellaneda-Stoikov quotes against symmetric quotes of the same average spread, on the same random numbers: profit, its spread, and the inventory each carried."""
    return compare_strategies(n_paths=n_paths, steps=steps, informed_fraction=informed_fraction, gamma=gamma, sigma=sigma, k=k, A=A, max_inventory=max_inventory, seed=seed)


def simulate_rebate_trading(strategy: str = "aware", paths: int = 400, steps: int = 4000, persistence: float = 0.85, impact: float = 0.30, noise: float = 0.5, half_spread: float = 1.0, rebate: float = 0.2,
                            base_fill: float = 0.15, tilt: float = 0.30, threshold: float = 0.25, latency: int = 0, max_inventory: float = 25.0, seed: int = 0) -> dict:
    """One maker on ``paths`` independent sessions of ``steps`` steps, in ticks. ``strategy``: ``naive`` (quote both sides always) or ``aware`` (withdraw the side the estimated flow would hit: no ask when the
    flow estimate exceeds ``threshold``, no bid when it is below ``-threshold``). The flow estimate is the flow ``latency`` steps ago carried forward by its persistence. Both stop quoting a side at ``max_inventory``."""
    if strategy not in ("naive", "aware"):
        raise ValueError("strategy must be 'naive' or 'aware'")
    if not 0 <= persistence < 1 or latency < 0 or paths < 2 or steps < 10:
        raise ValueError("0 <= persistence < 1, latency >= 0, paths >= 2, steps >= 10")
    rng = np.random.default_rng(seed)
    xi = np.zeros((paths, steps + 1))
    scale = np.sqrt(1.0 - persistence ** 2)                                                                    # unit-variance flow whatever its persistence
    eps = rng.standard_normal((paths, steps)) * scale
    for t in range(steps):
        xi[:, t + 1] = persistence * xi[:, t] + eps[:, t]
    z = rng.standard_normal((paths, steps))
    u_bid, u_ask = rng.random((paths, steps)), rng.random((paths, steps))
    inv = np.zeros(paths)
    cash = np.zeros(paths)
    mid = np.zeros(paths)
    parts = {"rebate": np.zeros(paths), "spread": np.zeros(paths), "adverse": np.zeros(paths)}
    fills = np.zeros(paths)
    for t in range(steps):
        flow = xi[:, t]
        seen = xi[:, max(t - latency, 0)] * persistence ** min(latency, t)                                       # what the maker can see: old flow, carried forward by its persistence
        p_bid = np.clip(base_fill * (1.0 - tilt * flow), 0.0, 1.0)                                              # sellers (negative flow) hit the bid
        p_ask = np.clip(base_fill * (1.0 + tilt * flow), 0.0, 1.0)
        quote_bid, quote_ask = inv < max_inventory, inv > -max_inventory
        if strategy == "aware":
            quote_ask = quote_ask & ~(seen > threshold)                                                         # buy pressure is about to lift the ask: do not offer
            quote_bid = quote_bid & ~(seen < -threshold)
        buy = quote_bid & (u_bid[:, t] < p_bid)
        sell = quote_ask & (u_ask[:, t] < p_ask)
        move = impact * flow + noise * z[:, t]                                                                  # the mid's change over the step, which follows the flow
        cash += np.where(sell, mid + half_spread, 0.0) - np.where(buy, mid - half_spread, 0.0) + rebate * (buy * 1.0 + sell * 1.0)
        inv += buy * 1.0 - sell * 1.0
        parts["rebate"] += rebate * (buy * 1.0 + sell * 1.0)
        parts["spread"] += half_spread * (buy * 1.0 + sell * 1.0)
        parts["adverse"] += np.where(buy, move, 0.0) - np.where(sell, move, 0.0)                                # a unit bought loses if the mid then falls (move < 0): the sign convention makes losses negative
        fills += buy * 1.0 + sell * 1.0
        mid = mid + move
    wealth = cash + inv * mid
    inventory_pnl = wealth - parts["rebate"] - parts["spread"] - parts["adverse"]
    sd = float(wealth.std(ddof=1))
    return {"strategy": strategy, "latency": latency, "wealth": wealth, "mean_wealth": float(wealth.mean()), "std_wealth": sd, "sharpe": float(wealth.mean() / sd) if sd > 0 else float("nan"),
            "rebate": float(parts["rebate"].mean()), "spread": float(parts["spread"].mean()), "adverse_selection": float(parts["adverse"].mean()), "inventory_pnl": float(inventory_pnl.mean()),
            "price_pnl": float((wealth - parts["rebate"] - parts["spread"]).mean()),
            "fills": float(fills.mean()), "fills_per_step": float(fills.mean() / steps), "mean_abs_inventory": float(np.abs(inv).mean()), "se_wealth": sd / np.sqrt(paths)}


def pressure_table(latencies=(0, 1, 2, 4, 8, 16), **kwargs) -> pd.DataFrame:
    """The naive maker and the pressure-aware maker at several latencies on the same random numbers: profit and its parts per session."""
    rows = {"naive": simulate_rebate_trading("naive", **kwargs)}
    for L in latencies:
        rows[f"aware (latency {L})"] = simulate_rebate_trading("aware", latency=L, **kwargs)
    keys = ("mean_wealth", "se_wealth", "rebate", "spread", "price_pnl", "adverse_selection", "inventory_pnl", "fills", "mean_abs_inventory")
    return pd.DataFrame({name: {k: r[k] for k in keys} for name, r in rows.items()}).T
