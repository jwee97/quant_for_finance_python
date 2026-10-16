"""Simulated markets with a known truth, for testing forecasters: a method must find structure where it was planted and must not find it where it was not."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.base import History


def to_history(log_returns: np.ndarray, start: str = "2010-01-04", high_low: bool = False, seed: int = 0) -> History:
    """A history from daily log returns (the price starts at 100). With ``high_low`` every day gets a high and a low around the close, as wide as the day's move suggests."""
    close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    index = pd.bdate_range(start, periods=len(close))
    if not high_low:
        return History(close, index)
    rng = np.random.default_rng(seed)
    move = np.abs(np.concatenate([[0.0], log_returns]))
    spread = 0.5 * move + 0.003 * rng.random(len(close))
    return History(close, index, close * np.exp(spread), close * np.exp(-spread))


def iid(n: int, sigma: float = 0.01, mu: float = 0.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


def garch(n: int, seed: int = 0, omega: float = 2e-6, alpha: float = 0.08, gamma: float = 0.08, beta: float = 0.86, nu: float = 8.0) -> np.ndarray:
    """GJR-GARCH(1,1) returns with Student-t shocks: volatility that clusters and a fall that raises it more than a rise."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n + 300) * np.sqrt((nu - 2) / nu)
    var = omega / (1 - alpha - gamma / 2 - beta)
    out = np.empty(n + 300)
    for t in range(n + 300):
        out[t] = np.sqrt(var) * z[t]
        var = omega + (alpha + gamma * (out[t] < 0)) * out[t] ** 2 + beta * var
    return out[300:]


def regimes(n: int, seed: int = 0, calm=(0.0004, 0.006), storm=(-0.001, 0.02), stay=(0.99, 0.97)) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Markov-switching returns. Returns ``(returns, states)`` with state 0 calm and 1 stormy."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < stay[state[t - 1]] else 1 - state[t - 1]
    mu = np.where(state == 0, calm[0], storm[0])
    sd = np.where(state == 0, calm[1], storm[1])
    return rng.normal(mu, sd), state


def ar1(n: int, phi: float = 0.25, sigma: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


def jumps(n: int, seed: int = 0, sigma: float = 0.006, rate: float = 0.02, mean: float = -0.02, sd: float = 0.04) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = rng.poisson(rate, n)
    return rng.normal(0, sigma, n) + k * mean + np.sqrt(k) * sd * rng.standard_normal(n)


def trend(n: int, slope: float = 0.0015, sigma: float = 0.002, seed: int = 0) -> np.ndarray:
    """A steady drift with little noise: a price that trends."""
    return np.random.default_rng(seed).normal(slope, sigma, n)


def slow_drift(n: int, amplitude: float = 0.002, period: int = 300, sigma: float = 0.004, seed: int = 0) -> np.ndarray:
    """Returns whose average swings slowly between a rising and a falling market: the trend that a trend filter is built to follow."""
    rng = np.random.default_rng(seed)
    return amplitude * np.sin(2 * np.pi * np.arange(n) / period) + rng.normal(0, sigma, n)


def reverting(n: int, strength: float = 0.4, sigma: float = 0.01, seed: int = 0) -> np.ndarray:
    """Returns whose mean pulls back against the move of the last month: after a rise the next days tend to fall, by ``strength`` volatilities of a day times the squashed size of the rise."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    r[:21] = rng.normal(0, sigma, 21)
    for t in range(21, n):
        z = r[t - 21:t].sum() / (sigma * np.sqrt(21))
        r[t] = -strength * sigma * np.tanh(z) + rng.normal(0, sigma)
    return r
