"""Risk-based and characteristic factors applied across assets: betting against beta, low idiosyncratic risk, lottery demand (MAX), illiquidity, and value with momentum.

Betting against beta (Frazzini and Pedersen 2014) is a structured, beta-neutral trade; the others are scores. Everything uses trailing data only. Several of these were found
in single stocks; on a handful of ETFs they are weaker and noisier, so treat results as a test of the idea on this universe rather than a replication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.regimes import market_proxy
from ..framework.registry import register_model
from ._common import month_end_flags, monthly_weights


def _rank_z(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank scaled to (-0.5, 0.5); missing stays missing."""
    return frame.rank(axis=1).sub(0.5).div(frame.notna().sum(axis=1), axis=0) - 0.5


def _beta(data, window: int, shrink: float) -> pd.DataFrame:
    market = market_proxy(data)
    cov = data.returns.rolling(window, min_periods=window // 2).cov(market)
    raw = cov.div(market.rolling(window, min_periods=window // 2).var(), axis=0)
    return shrink * raw + (1.0 - shrink) * 1.0                       # shrink toward a beta of one, as Frazzini and Pedersen do


@register_model("bab", "cross-sectional", "Betting against beta (Frazzini-Pedersen): long low-beta assets and short high-beta assets, each leg scaled to beta one, rebalanced monthly")
class BettingAgainstBeta(ForecastModel):
    """Leverage-constrained investors bid up high-beta assets, so low-beta assets earn more per unit of market risk; the trade is leveraged long and beta-neutral."""

    name, family, position_mode = "bab", "defensive", "time_series"
    structured = True

    def __init__(self, window: int = 252, shrink: float = 0.6, min_assets: int = 4):
        if window < 60 or not 0 <= shrink <= 1 or min_assets < 3:
            raise ValueError("window >= 60, 0 <= shrink <= 1, min_assets >= 3")
        self.window, self.shrink, self.min_assets = window, shrink, min_assets

    def weights(self, data):
        beta = _beta(data, self.window, self.shrink).where(data.investable)
        flags = month_end_flags(data.index)
        rows = {}
        for date in beta.index[flags.to_numpy()]:
            b = beta.loc[date].dropna()
            if len(b) < self.min_assets:
                continue
            z = b.rank()
            dev = z - z.mean()
            k = 2.0 / dev.abs().sum()
            low, high = k * (-dev).clip(lower=0.0), k * dev.clip(lower=0.0)        # w_L and w_H each sum to one
            long = low / (low * b).sum()                                              # scale each leg to a portfolio beta of one
            short = high / (high * b).sum()
            rows[date] = (long - short).reindex(beta.columns).fillna(0.0)
        if not rows:
            return pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        return monthly_weights(pd.DataFrame(rows).T, data.index).where(beta.notna().sum(axis=1) >= self.min_assets, axis=0)

    def score(self, data):
        return self.weights(data).where(data.investable)


@register_model("low_idio_vol", "cross-sectional", "Low idiosyncratic volatility (Ang-Hodrick-Xing-Zhang): favour assets whose returns are least explained by noise the market does not share")
class LowIdiosyncraticVolatility(ForecastModel):
    """Assets with high residual volatility have been persistently over-priced by investors who treat them as lottery tickets."""

    name, family = "low_idio_vol", "defensive"

    def __init__(self, window: int = 252):
        if window < 60:
            raise ValueError("window must be at least 60 days")
        self.window = window

    def score(self, data):
        market = market_proxy(data)
        w = self.window
        beta = data.returns.rolling(w, min_periods=w // 2).cov(market).div(market.rolling(w, min_periods=w // 2).var(), axis=0)
        resid = data.returns - beta.shift(1).mul(market, axis=0)
        return (-(resid.rolling(w, min_periods=w // 2).std() * np.sqrt(252.0))).where(data.investable)


@register_model("max_effect", "cross-sectional", "MAX effect (Bali-Cakici-Whitelaw): avoid assets with an extreme recent daily gain, favour those without a lottery-like spike")
class MaxEffect(ForecastModel):
    """Investors overpay for assets that recently had a huge one-day gain (a lottery preference), so those assets earn less afterwards."""

    name, family = "max_effect", "defensive"

    def __init__(self, window: int = 21):
        if window < 5:
            raise ValueError("window must be at least 5 days")
        self.window = window

    def score(self, data):
        return (-data.returns.rolling(self.window, min_periods=self.window).max()).where(data.investable)


@register_model("amihud_illiquidity", "cross-sectional", "Illiquidity premium (Amihud): favour assets whose price moves most per dollar traded, averaged over 63 days (needs volume)")
class AmihudIlliquidity(ForecastModel):
    """Assets that are costly to trade must pay a higher expected return to attract holders; the price impact per dollar traded is the measure."""

    name, family = "amihud_illiquidity", "liquidity"

    def __init__(self, window: int = 63):
        if window < 10:
            raise ValueError("window must be at least 10 days")
        self.window = window

    def score(self, data):
        if data.volume is None:
            raise KeyError("amihud_illiquidity needs traded volume, which this bundle does not have")
        dollars = (data.prices * data.volume).replace(0.0, np.nan)
        impact = data.returns.abs() / dollars * 1e6
        return np.log(impact.rolling(self.window, min_periods=self.window).mean().replace(0.0, np.nan)).where(data.investable)


@register_model("value_momentum", "cross-sectional", "Value and momentum together (Asness-Moskowitz-Pedersen): the average rank of five-year reversal and 12-1 month momentum")
class ValueMomentum(ForecastModel):
    """Value and momentum are negatively correlated, so combining them gives a steadier premium than either one; a five-year loss stands in for cheapness."""

    name, family = "value_momentum", "multi-factor"

    def __init__(self, value_lookback: int = 1260, momentum_lookback: int = 252, skip: int = 21, value_weight: float = 0.5):
        if value_lookback <= momentum_lookback or momentum_lookback <= skip + 20 or not 0 <= value_weight <= 1:
            raise ValueError("value_lookback > momentum_lookback > skip + 20 and 0 <= value_weight <= 1")
        self.value_lookback, self.momentum_lookback, self.skip, self.value_weight = value_lookback, momentum_lookback, skip, value_weight

    def score(self, data):
        p = data.prices
        value = -(p.shift(self.skip) / p.shift(self.value_lookback) - 1.0)
        momentum = p.shift(self.skip) / p.shift(self.momentum_lookback) - 1.0
        both = _rank_z(value).where(momentum.notna()) * self.value_weight + _rank_z(momentum).where(value.notna()) * (1.0 - self.value_weight)
        return both.where(data.investable)
