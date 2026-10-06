"""Statistical arbitrage: trade the spread between assets that should move together, betting it closes.

Every model here is dollar-neutral by construction (a long leg against a short leg, or an asset against a basket) and every
one has the same failure mode: a relationship that held in the past can break, and the spread then keeps widening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..signals.pca_strategy import pca_stat_arb_signal
from ._common import present

DEFAULT_PAIRS = (("EEM", "EFA"), ("GLD", "SLV"), ("IEF", "TLT"), ("LQD", "HYG"), ("SPY", "QQQ"), ("IWM", "SPY"))


@register_model("pca_residual", "stat-arb", "PCA residual trading: fade each asset's deviation from what three principal components explain")
class PCAResidual(ForecastModel):
    """After removing the common factors, what is left in an asset's return is noise that reverts."""

    name, family = "pca_residual", "stat-arb"

    def __init__(self, components: int = 3, window: int = 252, zscore_window: int = 21):
        self.components, self.window, self.zscore_window = components, window, zscore_window

    def score(self, data):
        return pca_stat_arb_signal(data.returns, self.components, self.window, self.zscore_window).where(data.investable)


def kalman_spread(log_left: np.ndarray, log_right: np.ndarray, delta: float = 1e-5, obs_var: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic hedge ratio by Kalman filter: ``log_left = alpha + beta * log_right`` with ``(alpha, beta)`` a random walk.

    Returns the one-step-ahead innovation (spread) and its standard deviation. The innovation at ``t`` is the prediction error
    BEFORE observing ``t`` is used to update the state, so it uses data through ``t`` only.
    """
    n = len(log_left)
    state = np.zeros(2)
    cov = np.eye(2)
    q = delta / (1.0 - delta) * np.eye(2)
    innovation = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    for t in range(n):
        if not (np.isfinite(log_left[t]) and np.isfinite(log_right[t])):
            continue
        h = np.array([1.0, log_right[t]])
        cov = cov + q
        pred = h @ state
        s = h @ cov @ h + obs_var
        innovation[t], sd[t] = log_left[t] - pred, np.sqrt(s)
        gain = cov @ h / s
        state = state + gain * innovation[t]
        cov = cov - np.outer(gain, h) @ cov
    return innovation, sd


@register_model("kalman_pairs", "stat-arb", "Pairs trading with a Kalman-filter hedge ratio: fade the standardised innovation of each pair")
class KalmanPairs(ForecastModel):
    """A pair's hedge ratio drifts; filtering it avoids the stale-regression problem of a fixed-window hedge."""

    name, family = "kalman_pairs", "stat-arb"

    def __init__(self, pairs: tuple = DEFAULT_PAIRS, delta: float = 1e-5, obs_var: float = 1e-3, clip: float = 3.0):
        self.pairs, self.delta, self.obs_var, self.clip = tuple(tuple(p) for p in pairs), delta, obs_var, clip

    def score(self, data):
        out = pd.DataFrame(0.0, index=data.index, columns=data.assets)
        log = np.log(data.prices)
        for left, right in self.pairs:
            if left not in data.assets or right not in data.assets:
                continue
            innovation, sd = kalman_spread(log[left].to_numpy(), log[right].to_numpy(), self.delta, self.obs_var)
            z = pd.Series(innovation / sd, index=data.index).clip(-self.clip, self.clip)
            out[left] = out[left] - z                  # spread too high: short the left leg, long the right
            out[right] = out[right] + z
        return out.where(data.investable)


@register_model("cointegration_pairs", "stat-arb", "Rolling-cointegration pairs: trade a pair only while an Engle-Granger test on the last year says it is cointegrated")
class CointegrationPairs(ForecastModel):
    """Trade only relationships that currently look stationary; pairs that fail the test are left alone."""

    name, family = "cointegration_pairs", "stat-arb"

    def __init__(self, pairs: tuple = DEFAULT_PAIRS, window: int = 252, refit_every: int = 21, p_value: float = 0.10, z_window: int = 63, clip: float = 3.0):
        self.pairs, self.window, self.refit_every, self.p_value, self.z_window, self.clip = tuple(tuple(p) for p in pairs), window, refit_every, p_value, z_window, clip

    def score(self, data):
        from statsmodels.tsa.stattools import coint

        out = pd.DataFrame(0.0, index=data.index, columns=data.assets)
        log = np.log(data.prices)
        n = len(log)
        for left, right in self.pairs:
            if left not in data.assets or right not in data.assets:
                continue
            y, x = log[left].to_numpy(), log[right].to_numpy()
            z = np.zeros(n)
            for start in range(self.window, n, self.refit_every):
                lo, hi = start - self.window, start
                ys, xs = y[lo:hi], x[lo:hi]
                if not (np.isfinite(ys).all() and np.isfinite(xs).all()):
                    continue
                try:
                    p = coint(ys, xs, trend="c", autolag="aic")[1]
                except Exception:
                    continue
                if p > self.p_value:
                    continue
                beta, alpha = np.polyfit(xs, ys, 1)
                spread = y[lo:min(start + self.refit_every, n)] - (alpha + beta * x[lo:min(start + self.refit_every, n)])
                recent = spread[: self.window]
                mean, sd = recent[-self.z_window:].mean(), recent[-self.z_window:].std(ddof=1)
                if sd > 0:
                    seg = (spread[self.window:] - mean) / sd
                    z[start:start + len(seg)] = np.clip(seg, -self.clip, self.clip)
            zs = pd.Series(z, index=data.index)
            out[left] = out[left] - zs
            out[right] = out[right] + zs
        return out.where(data.investable)


@register_model("sparse_basket", "stat-arb", "Sparse basket mean reversion: regress each asset on a LASSO-selected basket of the others and fade the cumulative residual")
class SparseBasket(ForecastModel):
    """The few other assets that actually explain an ETF define its fair value; the residual from that sparse basket reverts."""

    name, family = "sparse_basket", "stat-arb"

    def __init__(self, window: int = 252, refit_every: int = 21, alpha: float = 0.05, z_window: int = 21):
        self.window, self.refit_every, self.alpha, self.z_window = window, refit_every, alpha, z_window

    def score(self, data):
        from sklearn.linear_model import Lasso

        r = data.returns.fillna(0.0).to_numpy()
        n, k = r.shape
        resid = np.full((n, k), np.nan)
        for start in range(self.window, n, self.refit_every):
            lo, hi, end = start - self.window, start, min(start + self.refit_every, n)
            train = r[lo:hi]
            mu, sd = train.mean(axis=0), train.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            std = (train - mu) / sd
            future = (r[start:end] - mu) / sd
            for j in range(k):
                others = [c for c in range(k) if c != j]
                model = Lasso(alpha=self.alpha, max_iter=2000, fit_intercept=False).fit(std[:, others], std[:, j])
                resid[start:end, j] = future[:, j] - future[:, others] @ model.coef_
        residual = pd.DataFrame(resid, index=data.index, columns=data.assets)
        cumulative = residual.rolling(self.z_window, min_periods=self.z_window).sum()
        z = cumulative / (residual.rolling(252, min_periods=60).std(ddof=1) * np.sqrt(self.z_window)).replace(0.0, np.nan)
        return (-z.clip(-4, 4)).where(data.investable)


def simulate_nav_arbitrage(n_days: int = 2520, premium_sd: float = 0.0008, half_life_days: float = 1.5, cost_bps: float = 1.0, entry_sd: float = 1.5,
                           seed: int = 7) -> dict:
    """A SIMULATION (there is no NAV data in this repository): an ETF whose price premium to NAV is an Ornstein-Uhlenbeck process.

    A trader who buys a discount and sells a premium beyond ``entry_sd`` standard deviations, round-tripping at ``cost_bps``, earns the
    reversion minus costs. Returns the daily P&L, its annualised Sharpe and the break-even cost: the edge is real only while the
    premium's volatility is larger than the cost of crossing the spread twice.
    """
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1.0 / half_life_days)
    shock = premium_sd * np.sqrt(1 - phi ** 2)
    premium = np.zeros(n_days)
    for t in range(1, n_days):
        premium[t] = phi * premium[t - 1] + shock * rng.standard_normal()
    threshold = entry_sd * premium_sd
    position = np.zeros(n_days)
    position[1:] = np.where(premium[:-1] > threshold, -1.0, np.where(premium[:-1] < -threshold, 1.0, 0.0))
    gross = position[1:] * (premium[1:] - premium[:-1])                # position -1 = short the premium: profits when it falls
    trades = np.abs(np.diff(position))
    pnl = gross - trades * cost_bps / 1e4
    sharpe = float(np.sqrt(252) * pnl.mean() / pnl.std(ddof=1)) if pnl.std(ddof=1) > 0 else float("nan")
    per_trade = gross.sum() / max(trades.sum(), 1)
    return {"pnl": pnl, "sharpe": sharpe, "gross_per_trade_bps": float(1e4 * per_trade), "breakeven_cost_bps": float(1e4 * per_trade), "n_trades": int(trades.sum())}
