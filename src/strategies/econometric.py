"""Strategies built on the econometrics and probability layers: adaptive trend by Kalman filter, volatility management with GARCH and EVT, a shrunk VAR lead-lag model.

``kalman_trend``       the filtered slope of a local linear trend in log price: a trend estimate whose memory is set by a signal-to-noise ratio, not a window
``garch_vol_managed``  Moreira-Muir volatility management using a walk-forward GARCH forecast instead of last month's realised variance
``evt_risk_managed``   exposure inversely proportional to a conditional-EVT expected-shortfall forecast: sizing on the tail, not the variance
``bvar_lead_lag``      each asset's next-month return forecast from last month's returns of all assets, with Minnesota shrinkage (cross-asset predictability)

Every score on day ``t`` uses data through ``t`` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..features.sleeves import month_end_dates
from ._common import daily_vol


def steady_state_gains(q_level: float, q_slope: float, tol: float = 1e-13, max_iter: int = 100000) -> tuple[float, float]:
    """Kalman gains of the local linear trend ``(level, slope)`` with unit observation noise and process noise ``diag(q_level, q_slope)``, from iterating the Riccati
    recursion to its fixed point. The filter then reduces to the alpha-beta recursion ``level += g1 v``, ``slope += g2 v``."""
    T = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_level, q_slope])
    P = np.eye(2) * 1e3
    Z = np.array([1.0, 0.0])
    K = np.zeros(2)
    for _ in range(max_iter):
        F = P[0, 0] + 1.0
        K_new = P @ Z / F
        Pf = P - np.outer(K_new, Z @ P)
        P_next = T @ Pf @ T.T + Q
        if np.max(np.abs(K_new - K)) < tol:
            K = K_new
            break
        K, P = K_new, P_next
    return float(K[0]), float(K[1])


def alpha_beta_slope(log_price: pd.DataFrame, g1: float, g2: float) -> pd.DataFrame:
    """Filtered slope of the local linear trend for every column, vectorised over assets. A missing price skips the update (the state is carried forward)."""
    y = log_price.to_numpy(dtype=float)
    n, k = y.shape
    level = np.full(k, np.nan)
    slope = np.zeros(k)
    out = np.full((n, k), np.nan)
    started = np.zeros(k, dtype=bool)
    for t in range(n):
        obs = np.isfinite(y[t])
        new = obs & ~started
        level[new] = y[t][new]
        started |= new
        pred = level + slope
        v = np.where(obs & started & ~new, y[t] - pred, 0.0)
        level = np.where(started, np.where(obs & ~new, pred + g1 * v, np.where(new, level, pred)), np.nan)
        slope = np.where(started, slope + g2 * v, 0.0)
        out[t] = np.where(started, slope, np.nan)
    return pd.DataFrame(out, index=log_price.index, columns=log_price.columns)


@register_model("kalman_trend", "time-series", "Adaptive trend by Kalman filter: the filtered slope of a local linear trend in log price, in units of the asset's own daily volatility")
class KalmanTrend(ForecastModel):
    """A trend is a persistent drift hidden in noisy prices; the Kalman filter estimates it recursively, and its signal-to-noise ratio sets the memory."""

    name, family, position_mode = "kalman_trend", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, level_noise: float = 0.02, slope_noise: float = 1e-5, vol_halflife: float = 60.0, clip: float = 3.0, scale: float = 0.15):
        if level_noise <= 0 or slope_noise <= 0 or vol_halflife <= 0 or clip <= 0 or scale <= 0:
            raise ValueError("level_noise, slope_noise, vol_halflife, clip and scale must be positive")
        self.level_noise, self.slope_noise, self.vol_halflife, self.clip, self.scale = level_noise, slope_noise, vol_halflife, clip, scale

    def score(self, data):
        vol = daily_vol(data.returns, self.vol_halflife)
        g1, g2 = steady_state_gains(self.level_noise, self.slope_noise)
        slope = alpha_beta_slope(np.log(data.prices), g1, g2)
        z = slope / vol.replace(0.0, np.nan) / self.scale          # slope in volatility units; `scale` is the typical size of that ratio
        return np.tanh(z / self.clip).where(data.investable & z.notna())


def _typical_scale(forecast: pd.DataFrame, min_history: int, max_scale: float) -> pd.DataFrame:
    """Exposure ``c / forecast`` with ``c`` the expanding mean of the forecast (so average exposure is about one), capped."""
    typical = forecast.expanding(min_periods=min_history).mean()
    return (typical / forecast.replace(0.0, np.nan)).clip(upper=max_scale)


@register_model("garch_vol_managed", "volatility", "Volatility-managed long using a walk-forward GARCH(1,1) volatility forecast in place of trailing realised variance")
class GarchVolManaged(ForecastModel):
    """If a better volatility forecast is available, scaling exposure by it should capture more of the volatility-timing benefit than a one-month lookback does."""

    name, family, position_mode = "garch_vol_managed", "volatility", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, model: str = "garch", dist: str = "normal", min_train: int = 750, refit_every: int = 504, max_scale: float = 3.0):
        if model not in ("garch", "gjr", "tgarch", "egarch") or dist not in ("normal", "t") or min_train < 250 or refit_every < 21 or max_scale <= 0:
            raise ValueError("model in {garch, gjr, tgarch, egarch}, dist in {normal, t}, min_train >= 250, refit_every >= 21, max_scale > 0")
        self.model, self.dist, self.min_train, self.refit_every, self.max_scale = model, dist, min_train, refit_every, max_scale

    def score(self, data):
        from ..econometrics.garch import walk_forward_volatility

        var = pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        for a in data.assets:
            r = data.returns[a].dropna()
            if len(r) <= self.min_train + 21:
                continue
            wf = walk_forward_volatility(r, self.model, self.dist, self.min_train, self.refit_every)
            var[a] = (wf["forecast"] ** 2).reindex(data.index)
        # walk_forward_volatility indexes each forecast by the day it APPLIES to (built from returns through the day before); shifting by -1 puts the forecast of
        # tomorrow's variance on today's date, where it is known at the close
        return _typical_scale(var.shift(-1), 252, self.max_scale).where(data.investable)


@register_model("evt_risk_managed", "volatility", "Tail-risk-managed long: exposure inversely proportional to a conditional-EVT expected-shortfall forecast")
class EVTRiskManaged(ForecastModel):
    """Losses, not variance, are what drawdown limits bind on: size positions by the fitted tail so exposure falls when the tail is heavy or the scale is high."""

    name, family, position_mode = "evt_risk_managed", "volatility", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, p: float = 0.99, min_obs: int = 750, refit_every: int = 63, halflife: float = 40.0, max_scale: float = 3.0):
        if not 0.9 <= p < 1 or min_obs < 500 or refit_every < 5 or halflife <= 0 or max_scale <= 0:
            raise ValueError("0.9 <= p < 1, min_obs >= 500, refit_every >= 5, halflife > 0, max_scale > 0")
        self.p, self.min_obs, self.refit_every, self.halflife, self.max_scale = p, min_obs, refit_every, halflife, max_scale

    def score(self, data):
        from ..probability.evt import dynamic_evt_var

        es = pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        for a in data.assets:
            r = data.returns[a].dropna()
            if len(r) <= self.min_obs + 21:
                continue
            f = dynamic_evt_var(r, self.p, self.halflife, 0.90, self.min_obs, self.refit_every)
            es[a] = f["es"].reindex(data.index)
        return _typical_scale(es.shift(-1), 252, self.max_scale).where(data.investable)       # the forecast indexed by the day it applies to, placed on the day it is known


@register_model("bvar_lead_lag", "cross-sectional", "Cross-asset lead-lag: a Minnesota-shrinkage VAR(1) on monthly returns forecasts each asset's next month from every asset's last month")
class BVARLeadLag(ForecastModel):
    """Some assets move first and others follow (credit before equities, bonds before the dollar); a heavily shrunk VAR can pick that up without overfitting."""

    name, family, position_mode = "bvar_lead_lag", "cross-sectional", "cross_sectional"

    def __init__(self, lags: int = 1, tightness: float = 0.1, min_months: int = 60, refit_every: int = 3):
        if lags < 1 or tightness <= 0 or min_months < 36 or refit_every < 1:
            raise ValueError("lags >= 1, tightness > 0, min_months >= 36, refit_every >= 1")
        self.lags, self.tightness, self.min_months, self.refit_every = lags, tightness, min_months, refit_every

    def score(self, data):
        from ..econometrics.factor import fit_bvar

        months = month_end_dates(data.index)
        monthly = data.prices.loc[months].pct_change()
        live = [a for a in data.assets if monthly[a].notna().sum() > self.min_months + self.lags + 12]
        out = pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        if len(live) < 3:
            return out
        panel = monthly[live]
        start = panel.dropna(how="any").index[0] if panel.dropna(how="any").shape[0] else None
        if start is None:
            return out
        model = None
        for i, date in enumerate(panel.index):
            hist = panel.loc[:date].dropna(how="any")
            if len(hist) < self.min_months:
                continue
            if model is None or i % self.refit_every == 0:
                model = fit_bvar(hist, self.lags, lam=self.tightness, random_walk_mean=0.0)
            else:
                model._Y = hist.to_numpy(float)
            out.loc[date, live] = model.forecast(1).iloc[0].to_numpy()
        return out.ffill().where(data.investable)
