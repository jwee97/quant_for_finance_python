"""Drawdown, ruin and growth-optimal sizing.

A strategy's Sharpe ratio says nothing about whether you survive it. These tools put probabilities on the paths:

* **Brownian-motion closed forms** (``prob_fall_below``, ``prob_ruin_brownian``, ``expected_max_drawdown``): for arithmetic Brownian motion with drift ``mu`` and
  volatility ``sigma``, the chance of EVER falling ``x`` below the starting level is ``exp(-2 mu x / sigma^2)``. This is the probability that a drawdown that
  begins at a peak reaches depth ``x``, and the ruin probability for a barrier ``x`` below capital. The maximum drawdown over a long horizon is NOT bounded by it
  (drawdowns recur from every new peak); for finite horizons use ``brownian_max_drawdown_probability`` (simulation) or the bootstrap functions.
* **Simulation from data** (``drawdown_distribution``, ``prob_drawdown_exceeds``): block-bootstrap the observed returns and measure the maximum drawdown, the
  time under water and the time to recover on each path, so the answer keeps the fat tails and autocorrelation of the data.
* **Kelly sizing** (``kelly_fraction``, ``kelly_growth``, ``drawdown_constrained_kelly``): the leverage that maximises long-run log growth, ``f* = mu / sigma^2``,
  and the fractional-Kelly rule that caps the probability of a given drawdown (Thorp; MacLean, Thorp & Ziemba 2010).
* **Discrete ruin** (``gambler_ruin_probability``): the classical random-walk ruin probability for a bet with win probability ``p`` and payoff ratio ``b``.
* **Conditional drawdown-at-risk** (``cdar``): the mean of the worst ``(1 - alpha)`` of drawdowns (Chekhlov, Uryasev & Zabarankin 2005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from .resampling import bootstrap_indices, optimal_block_length


# ------------------------------------------------------------------------------------------------------------------- drawdowns
def drawdown_series(returns: pd.Series, log: bool = False) -> pd.Series:
    """Fractional drawdown from the running peak of the compounded wealth curve (0 at a peak, negative below)."""
    r = returns.fillna(0.0)
    wealth = np.exp(r.cumsum()) if log else (1.0 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Each peak-to-recovery episode: start (peak), trough, recovery date (NaT if not recovered), depth, days to trough and days to recover."""
    dd = drawdown_series(returns)
    rows = []
    in_dd, start = False, None
    idx = dd.index
    for i, v in enumerate(dd.to_numpy()):
        if v < 0 and not in_dd:
            in_dd, start = True, i - 1 if i > 0 else 0
        elif v == 0 and in_dd:
            seg = dd.iloc[start: i + 1]
            trough = seg.idxmin()
            rows.append({"peak": idx[start], "trough": trough, "recovery": idx[i], "depth": float(seg.min()), "days_to_trough": int(idx.get_loc(trough) - start),
                         "days_to_recover": int(i - idx.get_loc(trough))})
            in_dd = False
    if in_dd:
        seg = dd.iloc[start:]
        trough = seg.idxmin()
        rows.append({"peak": idx[start], "trough": trough, "recovery": pd.NaT, "depth": float(seg.min()), "days_to_trough": int(idx.get_loc(trough) - start), "days_to_recover": np.nan})
    return pd.DataFrame(rows)


def max_drawdown_of_paths(paths: np.ndarray) -> np.ndarray:
    """Maximum drawdown of each row of a returns matrix (simple returns)."""
    wealth = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(wealth, axis=1)
    return (wealth / peak - 1.0).min(axis=1)


def drawdown_distribution(returns, horizon: int = 252, n_paths: int = 5000, kind: str = "stationary", block: float | None = None, seed: int = 0) -> dict:
    """Bootstrap ``n_paths`` paths of ``horizon`` returns and summarise the maximum drawdown of each (negative numbers)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    block = optimal_block_length(r, kind="stationary" if kind == "stationary" else "circular") if block is None else block
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, horizon))
    for k in range(n_paths):
        idx = bootstrap_indices(max(horizon, n), kind, block, rng)[:horizon] % n if horizon <= n else np.concatenate([bootstrap_indices(n, kind, block, rng) for _ in range(-(-horizon // n))])[:horizon]
        paths[k] = r[idx]
    mdd = max_drawdown_of_paths(paths)
    return {"mdd": mdd, "mean": float(mdd.mean()), "median": float(np.median(mdd)), "p95": float(np.quantile(mdd, 0.05)), "p99": float(np.quantile(mdd, 0.01)),
            "block": block, "horizon": horizon}


def prob_drawdown_exceeds(returns, depth: float, horizon: int = 252, n_paths: int = 5000, seed: int = 0, **kwargs) -> float:
    """Bootstrap probability that the maximum drawdown over ``horizon`` days is worse than ``-depth``."""
    d = drawdown_distribution(returns, horizon, n_paths, seed=seed, **kwargs)
    return float((d["mdd"] <= -abs(depth)).mean())


def cdar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Conditional drawdown-at-risk: the average of the worst ``1 - alpha`` share of the drawdown series (a positive number)."""
    dd = -drawdown_series(returns).to_numpy()
    k = max(1, int(np.ceil((1 - alpha) * len(dd))))
    return float(np.sort(dd)[-k:].mean())


# ------------------------------------------------------------------------------------------------- Brownian-motion closed forms
def prob_fall_below(mu: float, sigma: float, depth: float) -> float:
    """P(``X`` EVER falls ``depth`` below its starting level) for ``dX = mu dt + sigma dW``: ``exp(-2 mu depth / sigma^2)`` for ``mu > 0``, 1 otherwise.
    Equivalent to the probability that a drawdown beginning at a peak now reaches ``depth``. ``depth`` is in the units of ``X`` (log wealth: a log drawdown)."""
    if mu <= 0:
        return 1.0
    return float(np.exp(-2.0 * mu * depth / sigma ** 2))


def brownian_max_drawdown_probability(mu: float, sigma: float, depth: float, horizon: float, n_sim: int = 20000, steps: int = 1000, seed: int = 0) -> float:
    """P(maximum drawdown from the running peak of ``dX = mu dt + sigma dW`` reaches ``depth`` within ``horizon``), by Monte Carlo (biased slightly low by discretisation)."""
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    x = np.cumsum(mu * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_sim, steps)), axis=1)
    x = np.concatenate([np.zeros((n_sim, 1)), x], axis=1)
    return float(((np.maximum.accumulate(x, axis=1) - x).max(axis=1) >= depth).mean())


def prob_ruin_brownian(mu: float, sigma: float, capital_fraction_lost: float) -> float:
    """Probability of ever losing the fraction ``f`` of starting capital for geometric Brownian motion: the log-wealth must fall ``-ln(1 - f)``."""
    return prob_fall_below(mu - 0.5 * sigma ** 2, sigma, -np.log(1.0 - capital_fraction_lost))


def expected_max_drawdown(mu: float, sigma: float, horizon: float, n_sim: int = 20000, seed: int = 0, steps: int = 1000) -> float:
    """Expected maximum drawdown (absolute units) of arithmetic Brownian motion ``dX = mu dt + sigma dW`` over ``horizon``.

    ``mu = 0`` has the exact value ``sigma sqrt(horizon) sqrt(pi / 2)`` (the maximum drawdown has the law of the maximum of a Brownian motion). For ``mu != 0``
    there is no elementary closed form (Magdon-Ismail & Atiya 2004 give an asymptotic approximation in terms of the drift-to-volatility ratio), so this
    is a Monte Carlo on ``steps`` grid points; the discretisation bias is of order ``sigma sqrt(horizon / steps)`` and is downward.
    """
    if abs(mu) < 1e-14:
        return float(sigma * np.sqrt(horizon) * np.sqrt(np.pi / 2.0))
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    x = np.cumsum(mu * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_sim, steps)), axis=1)
    x = np.concatenate([np.zeros((n_sim, 1)), x], axis=1)
    return float((np.maximum.accumulate(x, axis=1) - x).max(axis=1).mean())


# --------------------------------------------------------------------------------------------------------------------- Kelly
def kelly_fraction(mu: float, sigma: float, risk_free: float = 0.0) -> float:
    """Growth-optimal leverage for a continuously rebalanced position in an asset with excess drift ``mu - r`` and volatility ``sigma``: ``f* = (mu - r) / sigma^2``."""
    return float((mu - risk_free) / sigma ** 2)


def kelly_growth(f: float, mu: float, sigma: float, risk_free: float = 0.0) -> float:
    """Expected log growth rate at leverage ``f``: ``r + f (mu - r) - f^2 sigma^2 / 2``. Positive up to ``2 f*``; beyond it you lose money despite a positive edge."""
    return float(risk_free + f * (mu - risk_free) - 0.5 * f ** 2 * sigma ** 2)


def kelly_from_returns(returns, shrink: float = 1.0) -> float:
    """Kelly leverage estimated from a return sample, ``mean / variance`` times ``shrink`` (use 0.25-0.5: estimation error makes full Kelly dangerous)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return float(shrink * r.mean() / r.var(ddof=1))


def multi_asset_kelly(mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0, max_gross: float | None = None) -> np.ndarray:
    """Growth-optimal weights ``Sigma^-1 (mu - r)`` for several assets, optionally scaled down to a gross-leverage cap."""
    w = np.linalg.solve(np.asarray(cov, dtype=float), np.asarray(mu, dtype=float) - risk_free)
    if max_gross is not None and np.abs(w).sum() > max_gross:
        w = w * max_gross / np.abs(w).sum()
    return w


def drawdown_constrained_kelly(mu: float, sigma: float, max_drawdown: float, tolerance: float = 0.05) -> dict:
    """Fraction ``c`` of full Kelly such that the probability of EVER losing ``max_drawdown`` of the starting wealth is at most ``tolerance``.

    For continuous rebalancing at ``c f*`` the chance of wealth ever touching the fraction ``x = 1 - max_drawdown`` of its starting value is ``x^(2/c - 1)``
    (Thorp; MacLean, Thorp & Ziemba 2010), so ``c = 2 / (1 + ln(tolerance) / ln(x))`` capped at 1. At full Kelly the probability is ``x`` itself: betting
    Kelly means a 20% loss happens 80% of the time. Measured from the starting level, so a drawdown from a later peak can be deeper.
    """
    fstar = kelly_fraction(mu, sigma)
    x = 1.0 - max_drawdown
    c = float(min(1.0, 2.0 / (1.0 + np.log(tolerance) / np.log(x))))
    prob = lambda c_: float(x ** (2.0 / c_ - 1.0)) if c_ > 0 else 0.0
    return {"kelly": fstar, "fraction": c, "leverage": float(c * fstar), "prob_drawdown_at_kelly": prob(1.0), "prob_drawdown_at_choice": prob(c),
            "growth": float(kelly_growth(c * fstar, mu, sigma))}


def gambler_ruin_probability(p: float, b: float = 1.0, bankroll_units: int = 20, bet_fraction_units: int = 1) -> float:
    """Probability of ever going broke betting a fixed number of units on an event that pays ``b`` per unit with probability ``p`` (win: ``+b``, lose: ``-1``).

    For even money ``b = 1`` this is the classical ``((1-p)/p)^N`` when ``p > 1/2``. For general ``b`` the ruin probability is ``z^N`` where ``z`` solves
    ``p z^b + (1 - p)/z = 1`` in (0, 1), with ``N = bankroll / bet`` bets of capital (exact because a losing bet moves capital down by exactly one unit).
    """
    n = bankroll_units / bet_fraction_units
    if p * b - (1 - p) <= 0:
        return 1.0
    f = lambda z: p * z ** b + (1.0 - p) / z - 1.0                  # E[z^X] = 1 for X = +b w.p. p, -1 w.p. 1-p; the downward step is exactly -1 (skip-free)
    z = optimize.brentq(f, 1e-9, 1.0 - 1e-9)
    return float(z ** n)
