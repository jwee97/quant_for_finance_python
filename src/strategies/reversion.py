"""Short-horizon mean reversion: prices that fall sharply on little news tend to bounce.

The Connors RSI(2) pullback, internal bar strength, Bollinger and stochastic reversion, the three-down-days rule, an Ornstein-Uhlenbeck half-life filter, reversion only
inside ranges (low ADX), and the weekly or monthly cross-sectional reversal of Jegadeesh (1990) and Lehmann (1990). The time-series rules are long-only by default
(``shorts=False``): short-term reversion is a pullback in an up-trend, and shorting strength in a bull market is the classic way to lose with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..models.probabilistic import rsi as rsi_01
from ._common import stateful
from .trend import ADXTrend


def _gate(score: pd.DataFrame, data) -> pd.DataFrame:
    return score.where(data.investable)


@register_model("rsi2", "time-series", "Connors RSI(2) pullback: buy when the 2-day RSI is below 10 in an up-trend (price above the 200-day average), sell when price closes above its 5-day average")
class RSI2(ForecastModel):
    """A deep two-day oversold reading inside a rising market is forced selling that is usually reversed within days (liquidity provision)."""

    name, family, position_mode = "rsi2", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, rsi_window: int = 2, entry: float = 10.0, trend_window: int = 200, exit_window: int = 5, shorts: bool = False):
        if rsi_window < 2 or not 0 < entry < 50 or trend_window < 20 or exit_window < 2:
            raise ValueError("rsi_window >= 2, 0 < entry < 50, trend_window >= 20, exit_window >= 2")
        self.rsi_window, self.entry, self.trend_window, self.exit_window, self.shorts = rsi_window, entry, trend_window, exit_window, shorts

    def score(self, data):
        close = data.prices
        rsi = 100.0 * rsi_01(close, self.rsi_window)
        trend = close.rolling(self.trend_window, min_periods=self.trend_window).mean()
        exit_ma = close.rolling(self.exit_window, min_periods=self.exit_window).mean()
        long_in, long_out = (rsi < self.entry) & (close > trend), close > exit_ma
        short_in, short_out = ((rsi > 100 - self.entry) & (close < trend), close < exit_ma) if self.shorts else (None, None)
        return _gate(stateful(long_in, long_out, short_in, short_out).where(rsi.notna() & trend.notna()), data)


@register_model("ibs_reversion", "time-series", "Internal bar strength reversion: buy when the close sits in the bottom fifth of the day's range, sell when it recovers to the top third")
class IBSReversion(ForecastModel):
    """A close near the day's low means sellers pushed to the end; the next session tends to open higher. Without highs and lows a 10-day range of closes stands in."""

    name, family, position_mode = "ibs_reversion", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, entry: float = 0.2, exit: float = 0.7, trend_window: int = 200, shorts: bool = False):
        if not 0 < entry < exit < 1 or trend_window < 20:
            raise ValueError("0 < entry < exit < 1 and trend_window >= 20")
        self.entry, self.exit, self.trend_window, self.shorts = entry, exit, trend_window, shorts

    def score(self, data):
        close = data.prices
        high = data.high if data.high is not None else close.rolling(10, min_periods=10).max()
        low = data.low if data.low is not None else close.rolling(10, min_periods=10).min()
        ibs = (close - low) / (high - low).replace(0.0, np.nan)
        trend = close.rolling(self.trend_window, min_periods=self.trend_window).mean()
        long_in, long_out = (ibs < self.entry) & (close > trend), ibs > self.exit
        short_in, short_out = ((ibs > 1 - self.entry) & (close < trend), ibs < 1 - self.exit) if self.shorts else (None, None)
        return _gate(stateful(long_in, long_out, short_in, short_out).where(ibs.notna() & trend.notna()), data)


@register_model("bollinger_reversion", "time-series", "Bollinger band reversion: buy a close below the lower band (20 days, 2 deviations), hold until price returns to the middle band")
class BollingerReversion(ForecastModel):
    """A close two standard deviations below its own mean is a stretch that usually partly closes; the middle band is the natural exit."""

    name, family, position_mode = "bollinger_reversion", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, window: int = 20, k: float = 2.0, shorts: bool = False):
        if window < 5 or k <= 0:
            raise ValueError("window >= 5 and k > 0")
        self.window, self.k, self.shorts = window, k, shorts

    def score(self, data):
        close = data.prices
        mid, sd = close.rolling(self.window, min_periods=self.window).mean(), close.rolling(self.window, min_periods=self.window).std()
        long_in, long_out = close < mid - self.k * sd, close > mid
        short_in, short_out = (close > mid + self.k * sd, close < mid) if self.shorts else (None, None)
        return _gate(stateful(long_in, long_out, short_in, short_out).where(mid.notna() & sd.notna()), data)


@register_model("stochastic_reversion", "time-series", "Stochastic oscillator reversion: buy when the smoothed %K falls below 20 in an up-trend, sell when it recovers above 50")
class StochasticReversion(ForecastModel):
    """Where the close sits in its 14-day range, smoothed over three days: a low reading in a rising market marks a pullback rather than a reversal of trend."""

    name, family, position_mode = "stochastic_reversion", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, window: int = 14, smooth: int = 3, entry: float = 20.0, exit: float = 50.0, trend_window: int = 200):
        if window < 3 or smooth < 1 or not 0 < entry < exit < 100 or trend_window < 20:
            raise ValueError("window >= 3, smooth >= 1, 0 < entry < exit < 100, trend_window >= 20")
        self.window, self.smooth, self.entry, self.exit, self.trend_window = window, smooth, entry, exit, trend_window

    def score(self, data):
        close = data.prices
        high = data.high if data.high is not None else close
        low = data.low if data.low is not None else close
        hh, ll = high.rolling(self.window, min_periods=self.window).max(), low.rolling(self.window, min_periods=self.window).min()
        k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
        d = k.rolling(self.smooth, min_periods=self.smooth).mean()
        trend = close.rolling(self.trend_window, min_periods=self.trend_window).mean()
        out = stateful((d < self.entry) & (close > trend), d > self.exit)
        return _gate(out.where(d.notna() & trend.notna()), data)


@register_model("consecutive_down", "time-series", "Down-streak pullback (Connors): buy after three straight lower closes in an up-trend, sell on the first higher close")
class ConsecutiveDown(ForecastModel):
    """Three or more down days in a row inside an up-trend is a pause that tends to be bought back quickly."""

    name, family, position_mode = "consecutive_down", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, days: int = 3, trend_window: int = 200):
        if days < 2 or trend_window < 20:
            raise ValueError("days >= 2 and trend_window >= 20")
        self.days, self.trend_window = days, trend_window

    def score(self, data):
        close = data.prices
        down = (close < close.shift(1)).astype(float)
        streak = down.rolling(self.days, min_periods=self.days).sum() == self.days
        trend = close.rolling(self.trend_window, min_periods=self.trend_window).mean()
        out = stateful(streak & (close > trend), close > close.shift(1))
        return _gate(out.where(trend.notna() & close.shift(self.days).notna()), data)


@register_model("short_term_reversal", "cross-sectional", "Short-term reversal (Jegadeesh, Lehmann): buy last week's laggards and sell its leaders, in volatility units")
class ShortTermReversal(ForecastModel):
    """Last week's biggest movers partly reverse as liquidity providers are paid for absorbing order-flow imbalance."""

    name, family = "short_term_reversal", "reversion"
    rebalance = "weekly"

    def __init__(self, lookback: int = 5, vol_window: int = 63):
        if lookback < 1 or vol_window < 10:
            raise ValueError("lookback >= 1 and vol_window >= 10")
        self.lookback, self.vol_window = lookback, vol_window

    def score(self, data):
        ret = data.prices / data.prices.shift(self.lookback) - 1.0
        vol = data.returns.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(self.lookback)
        return _gate(-ret / vol.replace(0.0, np.nan), data)


@register_model("ou_reversion", "time-series", "Ornstein-Uhlenbeck reversion: fade the deviation from the 120-day mean, but only where a unit-root test rejects a random walk and the half-life is short")
class OUReversion(ForecastModel):
    """Fitting an AR(1) to log prices says whether an asset pulls back to its mean at all, and how fast. On a short window a random walk looks mean-reverting by chance
    (the estimate of the AR coefficient is biased down), so a deviation only counts if the Dickey-Fuller t-statistic is below its critical value and the half-life is short."""

    name, family, position_mode = "ou_reversion", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, window: int = 120, max_half_life: float = 30.0, df_critical: float = -2.86):
        if window < 40 or max_half_life <= 0 or df_critical >= 0:
            raise ValueError("window >= 40, max_half_life > 0 and a negative df_critical")
        self.window, self.max_half_life, self.df_critical = window, max_half_life, df_critical

    def fit(self, data) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rolling half-life (days) and Dickey-Fuller t-statistic of log prices: dx_t = a + g x_{t-1} + e."""
        x = np.log(data.prices)
        lag, dx, w = x.shift(1), x.diff(), self.window
        var_lag = lag.rolling(w, min_periods=w).var()
        gamma = dx.rolling(w, min_periods=w).cov(lag) / var_lag
        resid_var = (dx.rolling(w, min_periods=w).var() - gamma ** 2 * var_lag).clip(lower=0.0)
        t_stat = gamma / np.sqrt(resid_var / ((w - 2) * var_lag))
        b = 1.0 + gamma
        with np.errstate(invalid="ignore", divide="ignore"):
            half_life = pd.DataFrame(np.where((b > 0) & (b < 1), np.log(2.0) / -np.log(b), np.inf), index=x.index, columns=x.columns)
        return half_life, t_stat

    def score(self, data):
        x = np.log(data.prices)
        z = (x - x.rolling(self.window, min_periods=self.window).mean()) / x.rolling(self.window, min_periods=self.window).std().replace(0.0, np.nan)
        half_life, t_stat = self.fit(data)
        active = (half_life <= self.max_half_life) & (t_stat < self.df_critical)
        return _gate((-z / 2.0).clip(-2, 2).where(active, 0.0).where(z.notna() & t_stat.notna()), data)


@register_model("range_reversion", "time-series", "Range-bound reversion: fade the 20-day z-score, but only while the ADX shows no trend (below 20)")
class RangeReversion(ForecastModel):
    """Mean reversion works inside ranges and fails in trends; the ADX decides which one the market is in."""

    name, family, position_mode = "range_reversion", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, window: int = 20, adx_window: int = 14, adx_max: float = 20.0):
        if window < 5 or adx_window < 2 or adx_max <= 0:
            raise ValueError("window >= 5, adx_window >= 2, adx_max > 0")
        self.window, self.adx_window, self.adx_max = window, adx_window, adx_max

    def score(self, data):
        close = data.prices
        z = (close - close.rolling(self.window, min_periods=self.window).mean()) / close.rolling(self.window, min_periods=self.window).std().replace(0.0, np.nan)
        _, _, adx = ADXTrend(self.adx_window, 0.0).indicators(data)
        return _gate((-np.tanh(z / 2.0)).where(adx < self.adx_max, 0.0).where(z.notna() & adx.notna()), data)


