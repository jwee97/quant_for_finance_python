"""Optimal execution: the Almgren-Chriss trade schedule and its comparison with TWAP and VWAP.

Almgren & Chriss (2000): liquidate ``X`` shares in ``N`` steps over ``T``. Permanent impact ``gamma`` moves the price for everyone; temporary impact ``eta`` is paid on each slice.
The trader minimises ``E[cost] + lambda Var[cost]``: selling fast has high temporary impact but little price risk, selling slowly the opposite. The solution holds ``x(t) = X sinh(kappa (T - t)) / sinh(kappa T)`` shares
at time ``t`` with ``kappa^2 = lambda sigma^2 / eta`` (continuous approximation); as ``lambda -> 0`` the schedule is linear (TWAP) and a larger ``lambda`` front-loads. ``efficient_frontier`` traces expected cost
against its standard deviation over ``lambda``. ``vwap_schedule`` follows a volume profile and ``implementation_shortfall`` evaluates any schedule on simulated price paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def almgren_chriss_schedule(X: float, T: float, n: int, sigma: float, eta: float, lam: float) -> pd.DataFrame:
    """Holdings ``x_k`` (shares still to sell at the start of step ``k``) for ``k = 0 .. n``, and the trades ``n_k = x_{k-1} - x_k``. ``sigma`` is the per-unit-time price volatility, ``eta`` the temporary
    impact coefficient (price per share traded per unit time, so a step of ``tau`` selling ``n_k`` shares pays ``eta n_k / tau`` per share)."""
    t = np.linspace(0.0, T, n + 1)
    if lam <= 0:
        x = X * (1.0 - t / T)
    else:
        kappa = np.sqrt(lam * sigma ** 2 / eta)
        x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    return pd.DataFrame({"time": t, "holdings": x, "trade": np.r_[0.0, -np.diff(x)]})


def expected_cost_and_variance(schedule: pd.DataFrame, sigma: float, eta: float, gamma: float = 0.0) -> dict:
    """Expected implementation shortfall ``0.5 gamma X^2 + eta sum n_k^2 / tau`` and variance ``sigma^2 sum tau x_k^2`` (x at the END of each step, i.e. the remaining holdings)."""
    t, x, trades = schedule["time"].to_numpy(), schedule["holdings"].to_numpy(), schedule["trade"].to_numpy()
    tau = np.diff(t)
    X = x[0]
    cost = 0.5 * gamma * X ** 2 + eta * float((trades[1:] ** 2 / tau).sum())
    var = sigma ** 2 * float((tau * x[1:] ** 2).sum())
    return {"expected_cost": cost, "std_cost": float(np.sqrt(var)), "variance": var}


def efficient_frontier(X: float, T: float, n: int, sigma: float, eta: float, gamma: float = 0.0, lambdas=None) -> pd.DataFrame:
    """Expected cost and its standard deviation across risk aversions: every point is the cheapest schedule for its risk, and TWAP (``lam = 0``) is the extreme of lowest risk-adjusted preference for speed."""
    lambdas = (np.logspace(-1.0, 1.3, 12) ** 2 * eta / sigma ** 2) if lambdas is None else np.asarray(lambdas)           # default: kappa T from 0.1 to 20, from nearly TWAP to nearly immediate
    rows = []
    for lam in np.r_[0.0, lambdas]:
        s = almgren_chriss_schedule(X, T, n, sigma, eta, lam)
        rows.append({"lambda": lam, **expected_cost_and_variance(s, sigma, eta, gamma)})
    return pd.DataFrame(rows).set_index("lambda")


def twap_schedule(X: float, n: int) -> np.ndarray:
    return np.full(n, X / n)


def vwap_schedule(X: float, volume_profile) -> np.ndarray:
    """Trade a share of ``X`` in each interval proportional to the expected volume profile."""
    v = np.asarray(volume_profile, dtype=float)
    return X * v / v.sum()


def implementation_shortfall(trades: np.ndarray, sigma: float, eta: float, gamma: float, tau: float, s0: float = 100.0, n_paths: int = 5000, seed: int = 0) -> dict:
    """Simulate the sale of ``trades`` (shares per step, positive) on arithmetic Brownian prices with permanent impact ``gamma`` per share sold and temporary impact ``eta / tau`` per share traded.
    Shortfall = ``X s0 - proceeds`` (positive = cost). Returns the mean and standard deviation across paths."""
    rng = np.random.default_rng(seed)
    n = len(trades)
    price = np.full(n_paths, s0)
    proceeds = np.zeros(n_paths)
    for k in range(n):
        exec_price = price - eta * trades[k] / tau
        proceeds += trades[k] * exec_price
        price = price + sigma * np.sqrt(tau) * rng.standard_normal(n_paths) - gamma * trades[k]
    shortfall = trades.sum() * s0 - proceeds
    return {"mean": float(shortfall.mean()), "std": float(shortfall.std(ddof=1)), "shortfall": shortfall}
