"""Black-box (profit-seeking) strategies on a simulated intraday market: pair trading and statistical arbitrage of an ETF against its basket.

These are *research simulators*: they take a market with a stated structure (a mean-reverting spread, a premium that decays with a given half-life), apply the strategy exactly as a desk would code it
(a z-score rule, thresholds, latency, costs) and report what it earns. They answer *how much edge a given half-life, cost and delay leave*, not *whether such an edge exists in a real market*: that is a
question for data you do not have here, and a strategy that wins in the simulation it was designed for has demonstrated nothing about the world.

**Pair trading** (:func:`simulate_pair_trading`). Log prices ``A`` and ``B`` with ``log A = alpha + beta log B + s`` where the log spread ``s`` is an Ornstein-Uhlenbeck process with a half-life in bars (or, as the
control, a random walk with the same bar-to-bar variability: no edge). The rule is the classic one: the z-score of ``s`` against its own trailing window, enter when it is beyond ``entry`` standard deviations (sell
the spread when high, buy when low), leave inside ``exit`` or at a ``stop``, and be flat at the close. The hedge ratio ``beta`` is taken as known. Orders reach the market ``latency`` bars after the decision and
each change of position pays ``cost_bps`` on both legs. Profit is on a unit of dollars in ``A``, in bps.

**ETF arbitrage** (:func:`simulate_etf_arbitrage`). An ETF trades at its net asset value (the basket's value) times ``exp(d)`` where the premium ``d`` is mean reverting with a half-life of a few bars. Sell the
ETF and buy the basket when the premium exceeds ``entry_bps``, the reverse when it is below ``-entry_bps``, and leave when it has closed to ``exit_bps``. With latency the premium has partly gone by the time
the order lands, so the profit per trade falls with the delay; :func:`latency_table` shows how fast. Profit is in bps of the position; ``cost_bps`` is the round-trip cost of trading the ETF and the basket.
"""

from __future__ import annotations

import numpy as np


def _delay(target: np.ndarray, latency: int) -> np.ndarray:
    """The position held when an order decided ``latency`` bars ago has just landed (flat before the first one)."""
    if latency <= 0:
        return target
    out = np.zeros_like(target)
    out[:, latency:] = target[:, :-latency]
    return out


def _round_trips(position: np.ndarray, pnl: np.ndarray) -> dict:
    """Round-trip statistics from the position and the per-bar profit of every day: how many, what share won, how long they last, what they earn."""
    trips, wins, holds, profits = 0, 0, [], []
    for day in range(position.shape[0]):
        p, g = position[day], pnl[day]
        start = None
        for t in range(len(p)):
            if start is None and p[t] != 0:
                start, acc = t, 0.0
            if start is not None:
                acc += g[t]
                if p[t] == 0 or p[t] != p[start]:
                    trips += 1
                    wins += acc > 0
                    holds.append(t - start)
                    profits.append(acc)
                    start = None if p[t] == 0 else t
                    acc = 0.0 if start is None else g[t]
        if start is not None:
            trips += 1
            wins += acc > 0
            holds.append(len(p) - start)
            profits.append(acc)
    return {"round_trips": trips, "win_rate": wins / trips if trips else float("nan"), "average_hold_bars": float(np.mean(holds)) if holds else 0.0,
            "average_trip_bps": float(np.mean(profits)) if profits else 0.0}


def _summary(daily_bps: np.ndarray, extra: dict) -> dict:
    sd = float(daily_bps.std(ddof=1)) if len(daily_bps) > 1 else float("nan")
    return {"days": int(len(daily_bps)), "mean_daily_bps": float(daily_bps.mean()), "std_daily_bps": sd, "sharpe": float(daily_bps.mean() / sd * np.sqrt(252.0)) if sd and sd > 0 else float("nan"),
            "se_daily_bps": sd / np.sqrt(len(daily_bps)) if len(daily_bps) > 1 else float("nan"), **extra}


def simulate_pair_trading(days: int = 60, bars: int = 390, half_life: float = 60.0, spread_sd: float = 0.0008, beta: float = 1.2, entry: float = 2.0, exit: float = 0.3, stop: float = 5.0,
                          window: int = 240, cost_bps: float = 1.0, latency: int = 1, mean_reverting: bool = True, seed: int = 0) -> dict:
    """Trade a pair for ``days`` independent days of ``bars`` bars. ``mean_reverting=False`` makes the spread a random walk with the same bar-to-bar variability: the control that has no edge."""
    if days < 2 or bars < 20 or window < 10 or half_life <= 0 or entry <= exit or stop <= entry or latency < 0 or cost_bps < 0:
        raise ValueError("days >= 2, bars >= 20, window >= 10, half_life > 0, stop > entry > exit, latency >= 0, cost_bps >= 0")
    rng = np.random.default_rng(seed)
    rho = 0.5 ** (1.0 / half_life)
    innovation = spread_sd * np.sqrt(1.0 - rho ** 2)
    total = window + bars
    s = np.empty((days, total))
    s[:, 0] = rng.normal(0.0, spread_sd, days)
    eps = rng.standard_normal((days, total)) * innovation
    for t in range(1, total):
        s[:, t] = (rho if mean_reverting else 1.0) * s[:, t - 1] + eps[:, t]
    csum, csq = np.cumsum(s, axis=1), np.cumsum(s ** 2, axis=1)
    target = np.zeros((days, bars))
    state = np.zeros(days)
    for t in range(bars):
        j = window + t                                                                                        # the trailing window is the `window` bars before bar j
        n = window
        top, bottom = csum[:, j - 1], csum[:, j - 1 - n] if j - 1 - n >= 0 else 0.0
        mean = (top - bottom) / n
        var = (csq[:, j - 1] - (csq[:, j - 1 - n] if j - 1 - n >= 0 else 0.0)) / n - mean ** 2
        z = (s[:, j] - mean) / np.sqrt(np.maximum(var, 1e-18))
        flat = state == 0
        enter_short, enter_long = flat & (z > entry), flat & (z < -entry)
        leave_short = (state == -1) & ((z < exit) | (z > stop))
        leave_long = (state == 1) & ((z > -exit) | (z < -stop))
        state = np.where(enter_short, -1.0, np.where(enter_long, 1.0, np.where(leave_short | leave_long, 0.0, state)))
        if t >= bars - 1 - latency:
            state = np.zeros(days)                                                                            # flat by the close, allowing for the delay of the last orders
        target[:, t] = state
    position = _delay(target, latency)
    ds = np.diff(s[:, window - 1:], axis=1)                                                                    # bar t's change of the spread, from the close of bar t - 1
    held = np.c_[np.zeros(days), position[:, :-1]]                                                             # the position carried into each bar
    gross = held * ds
    trade_cost = cost_bps * 1e-4 * (1.0 + abs(beta)) * np.abs(np.diff(np.c_[np.zeros(days), position], axis=1))
    pnl = gross - trade_cost
    daily = 1e4 * pnl.sum(axis=1)
    extra = {"cost_bps_total": float(1e4 * trade_cost.sum(axis=1).mean()), "gross_bps": float(1e4 * gross.sum(axis=1).mean()), **_round_trips(position, 1e4 * pnl)}
    return {"daily_bps": daily, "position": position, "spread": s[:, window:], **_summary(daily, extra)}


def simulate_etf_arbitrage(days: int = 60, bars: int = 390, half_life: float = 4.0, premium_sd_bps: float = 6.0, entry_bps: float = 4.0, exit_bps: float = 0.5, cost_bps: float = 2.0,
                           latency: int = 0, seed: int = 0) -> dict:
    """Trade the premium of an ETF to its net asset value. The premium is an OU process (stationary standard deviation ``premium_sd_bps``, half-life ``half_life`` bars): sell the ETF and buy the basket above
    ``entry_bps``, the reverse below ``-entry_bps``, leave inside ``exit_bps``. Orders land ``latency`` bars late; each unit of position changed pays ``cost_bps`` (the ETF and the basket, one way)."""
    if days < 2 or bars < 20 or half_life <= 0 or entry_bps <= exit_bps or latency < 0 or cost_bps < 0 or premium_sd_bps <= 0:
        raise ValueError("days >= 2, bars >= 20, half_life > 0, entry_bps > exit_bps, latency >= 0, cost_bps >= 0, premium_sd_bps > 0")
    rng = np.random.default_rng(seed)
    rho = 0.5 ** (1.0 / half_life)
    d = np.empty((days, bars))
    d[:, 0] = rng.normal(0.0, premium_sd_bps, days)
    eps = rng.standard_normal((days, bars)) * premium_sd_bps * np.sqrt(1.0 - rho ** 2)
    for t in range(1, bars):
        d[:, t] = rho * d[:, t - 1] + eps[:, t]
    target = np.zeros((days, bars))
    state = np.zeros(days)
    for t in range(bars):
        flat = state == 0
        state = np.where(flat & (d[:, t] > entry_bps), -1.0, np.where(flat & (d[:, t] < -entry_bps), 1.0,
                         np.where(((state == -1) & (d[:, t] < exit_bps)) | ((state == 1) & (d[:, t] > -exit_bps)), 0.0, state)))
        if t >= bars - 1 - latency:
            state = np.zeros(days)
        target[:, t] = state
    position = _delay(target, latency)
    dd = np.diff(d, axis=1, prepend=d[:, :1])
    held = np.c_[np.zeros(days), position[:, :-1]]
    gross = held * dd                                                                                         # short the premium (-1) gains when it falls
    trade_cost = cost_bps * np.abs(np.diff(np.c_[np.zeros(days), position], axis=1))
    pnl = gross - trade_cost
    daily = pnl.sum(axis=1)
    extra = {"cost_bps_total": float(trade_cost.sum(axis=1).mean()), "gross_bps": float(gross.sum(axis=1).mean()), **_round_trips(position, pnl)}
    return {"daily_bps": daily, "position": position, "premium": d, **_summary(daily, extra)}


def latency_table(latencies=(0, 1, 2, 4, 8), **kwargs):
    """The ETF arbitrage at several latencies on the same random numbers: profit per day, round trips and profit per trip. The premium decays by half every ``half_life`` bars, so the edge goes the same way."""
    import pandas as pd

    rows = {}
    for L in latencies:
        r = simulate_etf_arbitrage(latency=L, **kwargs)
        rows[L] = {k: r[k] for k in ("mean_daily_bps", "sharpe", "round_trips", "win_rate", "average_trip_bps", "gross_bps", "cost_bps_total")}
    return pd.DataFrame(rows).T.rename_axis("latency_bars")
