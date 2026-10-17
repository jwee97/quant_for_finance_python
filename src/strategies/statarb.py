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

DEFAULT_PAIRS = (("EEM", "EFA"), ("GLD", "SLV"), ("IEF", "TLT"), ("LQD", "HYG"), ("SPY", "QQQ"), ("IWM", "SPY"))


@register_model("pca_residual", "stat-arb", "PCA residual trading: fade each asset's deviation from what three principal components explain")
class PCAResidual(ForecastModel):
    """After removing the common factors, what is left in an asset's return is noise that reverts."""

    name, family = "pca_residual", "stat-arb"

    def __init__(self, components: int = 3, window: int = 252, zscore_window: int = 21):
        self.components, self.window, self.zscore_window = components, window, zscore_window

    def required_assets(self) -> int:
        return max(super().required_assets(), int(self.components) + 1)           # something must be left over once the components are removed

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
    min_assets = 2                                              # a pair

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
    min_assets = 2                                              # a pair

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


@register_model("ou_threshold_pairs", "stat-arb", "Pairs trading at the best levels for an Ornstein-Uhlenbeck spread: refit the spread's reversion by maximum likelihood and trade the levels that earn the most per day after costs (Bertram)")
class OUThresholdPairs(ForecastModel):
    """Most pairs rules enter at a round number of standard deviations. If the spread really is a mean-reverting (Ornstein-Uhlenbeck) process there is a best pair of levels, and it depends on how
    fast the spread reverts and what a round trip costs: wide enough that each trade clears its cost, narrow enough that trades come often. This rule estimates the process on a rolling window,
    solves for those levels (``probability.ou.optimal_thresholds``) and holds each pair until the spread reaches the opposite level (``exit_scale = 1``) or the mean (``exit_scale = 0``).
    A pair trades only while the window's spread passes a Dickey-Fuller test (``df_critical``, the Engle-Granger value) and reverts with a half-life between ``min_half_life`` and
    ``max_half_life`` days; a spread that runs past ``stop_sd`` standard deviations is closed, and the pair stays out until the spread is back inside its entry level."""

    name, family, position_mode = "ou_threshold_pairs", "stat-arb", "time_series"
    structured = True                                           # the hedged pair trade IS the strategy: the weights are built here, not ranked
    min_assets = 2                                              # a pair

    def __init__(self, pairs: tuple = DEFAULT_PAIRS, window: int = 252, refit_every: int = 21, cost_bps: float = 5.0, min_half_life: float = 2.0, max_half_life: float = 40.0,
                 exit_scale: float = 1.0, stop_sd: float = 4.0, df_critical: float = -3.4):
        if window < 60 or refit_every < 1 or cost_bps < 0 or not 0 < min_half_life < max_half_life or not -1.0 <= exit_scale <= 1.0 or stop_sd <= 1.0 or df_critical >= 0:
            raise ValueError("window >= 60, refit_every >= 1, cost_bps >= 0, 0 < min_half_life < max_half_life, -1 <= exit_scale <= 1, stop_sd > 1 and a negative df_critical")
        self.pairs, self.window, self.refit_every, self.cost_bps = tuple(tuple(p) for p in pairs), int(window), int(refit_every), float(cost_bps)
        self.min_half_life, self.max_half_life, self.exit_scale, self.stop_sd, self.df_critical = float(min_half_life), float(max_half_life), float(exit_scale), float(stop_sd), float(df_critical)

    def pair_positions(self, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The position in the spread ``y - beta x`` (``+1`` long it: long ``y``, short ``x``) on each date and the hedge ratio ``beta`` in force. Every window of ``refit_every`` dates uses
        the hedge ratio, the Ornstein-Uhlenbeck fit and the trading levels estimated on the ``window`` dates before it, so a date uses nothing from its own future."""
        from ..probability.ou import dickey_fuller_t, fit_ou, optimal_thresholds

        n = len(y)
        pos, hedge = np.zeros(n), np.full(n, np.nan)
        state, blocked = 0.0, False
        for start in range(self.window, n, self.refit_every):
            end = min(start + self.refit_every, n)
            ys, xs = y[start - self.window:start], x[start - self.window:start]
            if not (np.isfinite(ys).all() and np.isfinite(xs).all()):
                state = 0.0
                continue
            beta, alpha = np.polyfit(xs, ys, 1)
            hedge[start:end] = beta
            try:
                window_spread = ys - (alpha + beta * xs)
                if dickey_fuller_t(window_spread) > self.df_critical:
                    raise ValueError("the spread cannot be told from a random walk")        # a regression of one random walk on another leaves a residual that looks like reversion
                fit = fit_ou(window_spread, 1.0 / 252.0)
                theta, mu, sigma = fit.estimates["theta"], fit.estimates["mu"], fit.estimates["sigma"]
                half_life = np.log(2.0) / theta * 252.0
                if not self.min_half_life <= half_life <= self.max_half_life:
                    raise ValueError("half-life outside the range worth trading")
                levels = optimal_thresholds(theta, 0.0, sigma, 2.0 * (1.0 + abs(beta)) * self.cost_bps / 1e4)
            except ValueError:
                state = 0.0                                              # no usable reversion in this window: flat until the next refit
                continue
            d = levels["half_width"]
            sd_eq = sigma / np.sqrt(2.0 * theta)
            stop = max(self.stop_sd * sd_eq, d + sd_eq)
            leave = self.exit_scale * d
            dev = y[start:end] - (alpha + beta * x[start:end]) - mu      # the spread's deviation from its mean (the levels are symmetric about it)
            for i, v in enumerate(dev):
                if not np.isfinite(v):
                    pos[start + i] = state
                    continue
                if state == 0.0:
                    if blocked:
                        blocked = abs(v) >= d                            # after a stop, wait for the spread to come back inside before trading again
                    elif -stop <= v <= -d:
                        state = 1.0
                    elif d <= v <= stop:
                        state = -1.0
                elif state > 0:
                    if v < -stop:
                        state, blocked = 0.0, True
                    elif v >= leave:
                        state = -1.0 if (self.exit_scale >= 1.0 and d <= v <= stop) else 0.0
                else:
                    if v > stop:
                        state, blocked = 0.0, True
                    elif v <= -leave:
                        state = 1.0 if (self.exit_scale >= 1.0 and -stop <= v <= -d) else 0.0
                pos[start + i] = state
        return pos, hedge

    def weights(self, data):
        out = pd.DataFrame(0.0, index=data.index, columns=data.assets)
        log = np.log(data.prices)
        active = [(left, right) for left, right in self.pairs if left in data.assets and right in data.assets]
        for left, right in active:
            pos, hedge = self.pair_positions(log[left].to_numpy(), log[right].to_numpy())
            beta = np.where(np.isfinite(hedge), hedge, 1.0)
            share = pos / (1.0 + np.abs(beta))                           # a unit of spread is $1 of the left leg against $beta of the right leg
            out[left] = out[left] + share / len(active)
            out[right] = out[right] - share * beta / len(active)
        return out

    def score(self, data):
        return self.weights(data).where(data.investable)


@register_model("sparse_basket", "stat-arb", "Sparse basket mean reversion: regress each asset on a LASSO-selected basket of the others and fade the cumulative residual")
class SparseBasket(ForecastModel):
    """The few other assets that actually explain an ETF define its fair value; the residual from that sparse basket reverts."""

    name, family = "sparse_basket", "stat-arb"
    min_assets = 3                                              # each asset is regressed on a basket of the others

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
