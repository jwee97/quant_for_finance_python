"""Time-series strategies: each asset is judged on its own history, not against the others.

Position mode is ``time_series``: a positive score is a long position, a negative one a short, in proportion to the
calibrated forecast, with no cross-sectional demeaning. (Gen 1 momentum and mean reversion are cross-sectional.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.mean_reversion import price_zscore
from ..features.momentum import volatility_scaled_momentum
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import atr_pct, daily_vol, stateful, tanh_score


@register_model("momentum", "time-series/cross-sectional", "Generation 1 momentum: 126-day return, skipping the last day, divided by volatility")
class Momentum(ForecastModel):
    """Assets that have risen keep rising for a while (under-reaction, herding, slow-moving capital)."""

    name, family = "momentum", "momentum"

    def __init__(self, lookback: int = 126, skip: int = 1, vol_lookback: int = 63):
        self.lookback, self.skip, self.vol_lookback = lookback, skip, vol_lookback

    def score(self, data):
        return volatility_scaled_momentum(data.prices, data.returns, self.lookback, self.skip, self.vol_lookback).where(data.investable)


@register_model("mean_reversion", "cross-sectional", "Generation 1 mean reversion: fade the 21-day price z-score")
class MeanReversion(ForecastModel):
    """Prices that have stretched away from their recent average snap back (liquidity provision, overreaction)."""

    name, family = "mean_reversion", "reversion"

    def __init__(self, lookback: int = 21, basis: str = "log"):
        self.lookback, self.basis = lookback, basis

    def score(self, data):
        return (-price_zscore(data.prices, self.lookback, self.basis)).where(data.investable)


@register_model("dual_momentum", "time-series", "Antonacci dual momentum: hold the top-k assets by 12-month return, but only those that beat cash")
class DualMomentum(ForecastModel):
    """Absolute momentum (beat cash) keeps you out of bear markets; relative momentum (rank) picks the leaders."""

    name, family, position_mode = "dual_momentum", "momentum", "time_series"

    def __init__(self, lookback: int = 252, skip: int = 21, top_k: int = 4, cash: str = "SHY"):
        self.lookback, self.skip, self.top_k, self.cash = lookback, skip, top_k, cash

    def score(self, data):
        ret = data.prices.shift(self.skip) / data.prices.shift(self.lookback) - 1.0
        cash = ret[self.cash] if self.cash in ret.columns else pd.Series(0.0, index=ret.index)
        excess = ret.sub(cash, axis=0).where(data.investable)
        eligible = excess.drop(columns=[self.cash], errors="ignore")
        rank = eligible.rank(axis=1, ascending=False)
        chosen = ((rank <= self.top_k) & (eligible > 0)).astype(float).where(eligible.notna())
        return chosen.reindex(columns=data.prices.columns).fillna(0.0).where(excess.notna().any(axis=1), np.nan)


@register_model("vol_breakout", "time-series", "Volatility breakout: go long above the moving average plus k ATRs, short below it minus k ATRs, exit at the average")
class VolatilityBreakout(ForecastModel):
    """A move beyond what recent volatility explains marks a new regime of trend (Keltner-channel logic)."""

    name, family, position_mode = "vol_breakout", "trend", "time_series"
    book = "sleeves"

    def __init__(self, window: int = 20, k: float = 2.0, atr_window: int = 14):
        self.window, self.k, self.atr_window = window, k, atr_window

    def score(self, data):
        close = data.prices
        ma = close.rolling(self.window, min_periods=self.window).mean()
        band = self.k * atr_pct(data, self.atr_window) * close
        valid = ma.notna() & band.notna()
        out = stateful(close > ma + band, close < ma, close < ma - band, close > ma)
        return out.where(valid).where(data.investable)


@register_model("donchian", "time-series", "Donchian channel trend following (turtle rules): enter on a 55-day breakout, exit on the 20-day opposite channel")
class Donchian(ForecastModel):
    """Trends are long enough that buying new highs and selling new lows earns more than it costs."""

    name, family, position_mode = "donchian", "trend", "time_series"
    book = "sleeves"

    def __init__(self, entry: int = 55, exit: int = 20):
        self.entry, self.exit = entry, exit

    def score(self, data):
        close = data.prices
        high = data.high if data.high is not None else close
        low = data.low if data.low is not None else close
        up_entry = high.rolling(self.entry).max().shift(1)
        dn_entry = low.rolling(self.entry).min().shift(1)
        up_exit = high.rolling(self.exit).max().shift(1)
        dn_exit = low.rolling(self.exit).min().shift(1)
        valid = up_entry.notna() & dn_entry.notna()
        out = stateful(close > up_entry, close < dn_exit, close < dn_entry, close > up_exit)
        return out.where(valid).where(data.investable)


@register_model("ma_crossover", "time-series", "Moving-average crossover: the 50-day average against the 200-day average, scaled by volatility")
class MovingAverageCrossover(ForecastModel):
    """Prices above their long average are in an up-trend; a smooth version of the golden/death cross."""

    name, family, position_mode = "ma_crossover", "trend", "time_series"
    book = "sleeves"

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast, self.slow = fast, slow

    def score(self, data):
        close = data.prices
        gap = close.rolling(self.fast).mean() / close.rolling(self.slow).mean() - 1.0
        scale = daily_vol(data.returns) * np.sqrt(self.fast)
        return tanh_score(gap / scale.replace(0.0, np.nan)).where(data.investable)


@register_model("trend_atr", "time-series", "Trend following with an ATR filter: follow the 50/200 trend only when the gap exceeds k ATRs (avoid whipsaw)")
class TrendATR(ForecastModel):
    """A moving-average cross inside the noise band is noise; require the trend to be large relative to daily range."""

    name, family, position_mode = "trend_atr", "trend", "time_series"
    book = "sleeves"

    def __init__(self, fast: int = 50, slow: int = 200, k: float = 0.5, atr_window: int = 14):
        self.fast, self.slow, self.k, self.atr_window = fast, slow, k, atr_window

    def score(self, data):
        close = data.prices
        gap = close.rolling(self.fast).mean() - close.rolling(self.slow).mean()
        strength = gap / (atr_pct(data, self.atr_window) * close).replace(0.0, np.nan)
        return (np.sign(strength) * (strength.abs() > self.k)).where(strength.notna()).where(data.investable)
