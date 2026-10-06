"""Cross-sectional strategies: assets are ranked against each other and the book is long the top, short the bottom.

Several classic equity factors (value, quality) need company fundamentals, which this ETF universe does not have. The two
"proxy" models below are price-based stand-ins and are named as such; they are not the factors a stock-picking fund means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.regimes import market_proxy
from ..framework.registry import register_model
from ._common import only, present


@register_model("xs_momentum", "cross-sectional", "Classic 12-1 month cross-sectional momentum: rank by the return from 12 months ago to 1 month ago")
class CrossSectionalMomentum(ForecastModel):
    """Winners over the past year, excluding the last month (which tends to reverse), keep outperforming."""

    name, family = "xs_momentum", "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback, self.skip = lookback, skip

    def score(self, data):
        return (data.prices.shift(self.skip) / data.prices.shift(self.lookback) - 1.0).where(data.investable)


@register_model("relative_strength", "cross-sectional", "Relative strength: six-month change in each asset's price relative to the market proxy")
class RelativeStrength(ForecastModel):
    """An asset whose price ratio to the market is rising is gaining leadership."""

    name, family = "relative_strength", "momentum"

    def __init__(self, lookback: int = 126):
        self.lookback = lookback

    def score(self, data):
        market = (1.0 + market_proxy(data)).cumprod()
        ratio = np.log(data.prices.div(market, axis=0))
        return (ratio - ratio.shift(self.lookback)).where(data.investable)


@register_model("low_volatility", "cross-sectional", "Low volatility: favour the assets with the lowest trailing volatility")
class LowVolatility(ForecastModel):
    """Investors overpay for risky assets (leverage constraints, lottery preferences), so the calm ones earn more per unit of risk."""

    name, family = "low_volatility", "defensive"

    def __init__(self, window: int = 252):
        self.window = window

    def score(self, data):
        return (-data.returns.rolling(self.window, min_periods=self.window // 2).std(ddof=1)).where(data.investable)


@register_model("defensive_beta", "cross-sectional", "Defensive / betting against beta: favour assets with the lowest beta to the market proxy")
class DefensiveBeta(ForecastModel):
    """High-beta assets are bid up by constrained investors, so low-beta assets have higher risk-adjusted returns."""

    name, family = "defensive_beta", "defensive"

    def __init__(self, window: int = 252):
        self.window = window

    def score(self, data):
        m = market_proxy(data)
        cov = data.returns.rolling(self.window, min_periods=self.window // 2).cov(m)
        var = m.rolling(self.window, min_periods=self.window // 2).var()
        return (-(cov.div(var, axis=0))).where(data.investable)


@register_model("value_proxy", "cross-sectional", "A price-based stand-in for value: long-horizon reversal, favouring assets that have fallen most over five years")
class ValueProxy(ForecastModel):
    """Assets that are cheap relative to their own long history mean-revert (De Bondt and Thaler). Not fundamental value: no earnings, no book value."""

    name, family = "value_proxy", "value"

    def __init__(self, lookback: int = 1260):
        self.lookback = lookback

    def score(self, data):
        return (-(data.prices / data.prices.shift(self.lookback) - 1.0)).where(data.investable)


@register_model("quality_proxy", "cross-sectional", "A price-based stand-in for quality: smoothness of the one-year price path (R-squared of log price on time)")
class QualityProxy(ForecastModel):
    """Steady, smooth gainers are held by patient owners and drift further (the 'frog in the pan' effect). Not accounting quality: no profitability or leverage."""

    name, family = "quality_proxy", "quality"

    def __init__(self, window: int = 252):
        self.window = window

    def score(self, data):
        log = np.log(data.prices)
        t = pd.Series(np.arange(len(log)), index=log.index, dtype=float)
        corr = log.rolling(self.window, min_periods=self.window).corr(t)
        slope_sign = np.sign(log - log.shift(self.window))
        return (corr ** 2 * slope_sign).where(data.investable)


YIELD_MAP = {"SHY": "DGS2", "AGG": "DGS5", "IEF": "DGS10", "TLT": "DGS20"}


@register_model("carry", "cross-sectional", "Carry: favour the bond ETFs whose yield exceeds the cash rate by the most")
class Carry(ForecastModel):
    """Assets that pay more than cash earn that extra yield if prices do not move; carry is a risk premium, not a free lunch."""

    name, family = "carry", "carry"
    requires = ("DGS2", "DGS5", "DGS10", "DGS20", "DFF")

    def __init__(self):
        pass

    def score(self, data):
        self.require(data)
        out = pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        for asset in present(data, YIELD_MAP):
            out[asset] = data.macro[YIELD_MAP[asset]] - data.macro["DFF"]
        return out.where(data.investable)
