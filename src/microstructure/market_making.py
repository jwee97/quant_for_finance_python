"""Market making: the Avellaneda-Stoikov quotes and a simulator that tests them against naive quoting, with the P&L split into its sources.

Avellaneda & Stoikov (2008): a market maker with inventory ``q`` and risk aversion ``gamma`` facing a mid price with volatility ``sigma`` and order arrivals at distance ``delta`` from the mid
with intensity ``A exp(-k delta)`` should quote around the RESERVATION price ``r = s - q gamma sigma^2 (T - t)`` (shaded against the inventory it already holds) with total spread
``delta_a + delta_b = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)``. A long maker lowers both quotes to sell, a short maker raises them to buy; the optimal spread widens with volatility and risk
aversion and with the time left.

``simulate_market_making`` runs ``n_paths`` independent days with an arithmetic Brownian mid, fills at the Poisson rates above, optional ``informed_fraction`` of fills that are followed by a price move
against the maker (adverse selection), and optional inventory limits. It reports terminal wealth, inventory statistics and the decomposition ``wealth = spread capture + inventory P&L + adverse selection``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_reservation_price(mid, inventory, gamma: float, sigma: float, time_left: float):
    return mid - inventory * gamma * sigma ** 2 * time_left


def as_optimal_spread(gamma: float, sigma: float, k: float, time_left: float) -> float:
    """Total optimal spread ``gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)``."""
    return gamma * sigma ** 2 * time_left + (2.0 / gamma) * np.log(1.0 + gamma / k)


def as_quotes(mid, inventory, gamma: float, sigma: float, k: float, time_left: float) -> tuple:
    """``(bid, ask)`` placed symmetrically around the reservation price at the optimal spread."""
    r = as_reservation_price(mid, inventory, gamma, sigma, time_left)
    half = 0.5 * as_optimal_spread(gamma, sigma, k, time_left)
    return r - half, r + half


def simulate_market_making(strategy: str = "as", n_paths: int = 2000, horizon: float = 1.0, steps: int = 1000, s0: float = 100.0, sigma: float = 2.0, gamma: float = 0.1, k: float = 1.5, A: float = 140.0,
                           symmetric_half_spread: float | None = None, max_inventory: int | None = None, informed_fraction: float = 0.0, impact: float = 0.5, seed: int = 0) -> dict:
    """Monte Carlo of a market maker. ``strategy``: ``as`` (Avellaneda-Stoikov quotes) or ``symmetric`` (quotes at the mid +- ``symmetric_half_spread``, which defaults to the AS average half spread).

    Fills in a step of length ``dt`` occur on the bid with probability ``A exp(-k delta_b) dt`` and on the ask with ``A exp(-k delta_a) dt`` (``delta`` the distance from the mid). With
    probability ``informed_fraction`` a fill is informed: the mid then moves by ``impact * (delta + typical spread)`` against the maker over the next step. ``max_inventory`` stops quoting on the side that
    would breach the limit. Returns terminal wealth and inventory arrays, the decomposition and summary statistics."""
    if strategy not in ("as", "symmetric"):
        raise ValueError("strategy must be 'as' or 'symmetric'")
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    base_half = 0.5 * as_optimal_spread(gamma, sigma, k, horizon / 2.0)
    sym_half = base_half if symmetric_half_spread is None else symmetric_half_spread
    s = np.full(n_paths, s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    spread_capture = np.zeros(n_paths)
    adverse = np.zeros(n_paths)
    n_trades = np.zeros(n_paths)
    inv_sq = np.zeros(n_paths)
    pending_shock = np.zeros(n_paths)
    for step in range(steps):
        t_left = horizon - step * dt
        if strategy == "as":
            bid, ask = as_quotes(s, q, gamma, sigma, k, t_left)
        else:
            bid, ask = s - sym_half, s + sym_half
        delta_b, delta_a = np.maximum(s - bid, 1e-6), np.maximum(ask - s, 1e-6)
        p_b = np.minimum(A * np.exp(-k * delta_b) * dt, 1.0)
        p_a = np.minimum(A * np.exp(-k * delta_a) * dt, 1.0)
        if max_inventory is not None:
            p_b = np.where(q >= max_inventory, 0.0, p_b)
            p_a = np.where(q <= -max_inventory, 0.0, p_a)
        buy_fill = rng.random(n_paths) < p_b                                       # a seller hits our bid: we buy
        sell_fill = rng.random(n_paths) < p_a
        cash += np.where(sell_fill, ask, 0.0) - np.where(buy_fill, bid, 0.0)
        q += sell_fill * -1.0 + buy_fill * 1.0
        spread_capture += np.where(sell_fill, ask - s, 0.0) + np.where(buy_fill, s - bid, 0.0)
        n_trades += buy_fill + sell_fill
        informed = rng.random(n_paths) < informed_fraction
        shock = impact * np.where(buy_fill & informed, -(delta_b + 0.5), 0.0) + impact * np.where(sell_fill & informed, (delta_a + 0.5), 0.0)   # signed against the maker
        s_new = s + sigma * np.sqrt(dt) * rng.standard_normal(n_paths) + pending_shock
        adverse += shock * (buy_fill * 1.0 - sell_fill * 1.0)                      # the loss on the unit just acquired (positive shock x sold unit, negative x bought unit)
        pending_shock = shock
        s = s_new
        inv_sq += q ** 2
    wealth = cash + q * s
    inventory_pnl = wealth - spread_capture - adverse                                # price moves on the inventory that are not informed-flow losses
    return {"wealth": wealth, "inventory": q, "spread_capture": spread_capture, "adverse_selection": adverse, "inventory_pnl": inventory_pnl, "n_trades": n_trades,
            "mean_wealth": float(wealth.mean()), "std_wealth": float(wealth.std(ddof=1)), "mean_abs_inventory": float(np.abs(q).mean()), "std_inventory": float(q.std(ddof=1)),
            "mean_trades": float(n_trades.mean()), "sharpe": float(wealth.mean() / wealth.std(ddof=1)) if wealth.std(ddof=1) > 0 else float("nan"), "mean_squared_inventory": float((inv_sq / steps).mean())}


def compare_strategies(**kwargs) -> pd.DataFrame:
    """Avellaneda-Stoikov against symmetric quoting with the same average spread, on common random numbers: the table that shows what inventory shading buys (lower inventory risk for similar
    profit)."""
    rows = {}
    for name in ("as", "symmetric"):
        r = simulate_market_making(strategy=name, **kwargs)
        rows[name] = {k: r[k] for k in ("mean_wealth", "std_wealth", "sharpe", "mean_trades", "mean_abs_inventory", "std_inventory", "mean_squared_inventory")}
    return pd.DataFrame(rows).T
