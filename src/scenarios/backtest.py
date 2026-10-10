"""Backtest a rule on generated scenarios, not only on the one history that happened.

``scenario_backtest(strategy, returns, generator, n_paths, horizon)`` generates ``n_paths`` paths of daily returns for the assets, hands each to ``strategy`` (a function from a DataFrame of daily returns to a DataFrame
of weights held *at the close of each day*, which earns the next day's return), charges a proportional cost on the turnover and returns the distribution of the outcomes: the annualised return, volatility, Sharpe ratio,
maximum drawdown and terminal wealth across paths. The question it answers is how a rule behaves over the range of futures a model of the past thinks possible; it cannot say anything the generator cannot
produce, so it is a robustness check next to the walk-forward backtest, never a substitute for it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .generators import generate


def evaluate_path(returns: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """Daily net return of weights decided at each close and earned the next day."""
    w = weights.reindex(returns.index).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    return gross - turnover * cost_bps / 1e4


def summarise(net: np.ndarray, periods: int = 252) -> dict:
    wealth = np.cumprod(1.0 + net)
    dd = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    sd = net.std(ddof=1) if len(net) > 1 else np.nan
    return {"return": float(net.mean() * periods), "volatility": float(sd * np.sqrt(periods)), "sharpe": float(net.mean() / sd * np.sqrt(periods)) if sd and sd > 0 else float("nan"),
            "max_drawdown": dd, "terminal_wealth": float(wealth[-1])}


def scenario_backtest(strategy: Callable[[pd.DataFrame], pd.DataFrame], returns: pd.DataFrame, generator: str = "bootstrap", n_paths: int = 100, horizon: int = 504, seed: int = 0,
                      cost_bps: float = 0.0, **params) -> pd.DataFrame:
    """One row per path. ``params`` go to the generator (for example ``kind='stationary'`` for the bootstrap)."""
    scen = generate(generator, returns, n_paths, horizon, seed, **params)
    idx = pd.bdate_range("2000-01-03", periods=horizon)
    rows = []
    for k in range(n_paths):
        r = pd.DataFrame(scen[k], index=idx, columns=returns.columns)
        rows.append(summarise(evaluate_path(r, strategy(r), cost_bps).to_numpy()))
    return pd.DataFrame(rows)
