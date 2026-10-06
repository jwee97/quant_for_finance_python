"""Macro strategies: tilt the asset-class mix on the state of inflation, the curve, the dollar and risk appetite.

Each is a time-series rule that sets a score per asset class, using only macro series with their publication lags.
Class membership comes from the universe configuration: equity, rates, credit, commodity, real_estate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import expanding_z, in_classes, present, zero_except


def _by_class(data, mapping: dict[str, pd.Series]) -> pd.DataFrame:
    values = {}
    for cls, series in mapping.items():
        for asset in in_classes(data, cls):
            values[asset] = series
    return zero_except(data.index, data.assets, values)


@register_model("inflation_rotation", "macro", "Inflation rotation: overweight commodities and gold and underweight long bonds when breakeven inflation is rising")
class InflationRotation(ForecastModel):
    """Real assets hedge rising inflation; nominal long bonds are hurt by it."""

    name, family, position_mode = "inflation_rotation", "macro", "time_series"
    requires = ("T10YIE",)

    def __init__(self, window: int = 126):
        self.window = window

    def score(self, data):
        self.require(data)
        s = np.tanh(expanding_z(data.macro["T10YIE"].diff(self.window).dropna(), 504).reindex(data.index) / 2.0)
        scores = _by_class(data, {"commodity": s})
        for a in present(data, ("TLT", "IEF")):
            scores[a] = -s
        return scores.where(data.investable)


@register_model("yield_curve_regime", "macro", "Yield-curve regimes: defensive (bonds, gold, no equity) when the 10y-3m curve is inverted, risk-on otherwise")
class YieldCurveRegime(ForecastModel):
    """An inverted curve has preceded most recessions; the market's own forecast of weaker growth and lower rates."""

    name, family, position_mode = "yield_curve_regime", "macro", "time_series"
    requires = ("T10Y3M",)

    def score(self, data):
        self.require(data)
        inverted = np.tanh(-data.macro["T10Y3M"] / 0.5).clip(lower=0.0)
        normal = 1.0 - inverted
        scores = _by_class(data, {"equity": 0.5 * normal - inverted, "credit": 0.5 * normal - inverted, "rates": inverted - 0.2 * normal,
                                  "commodity": 0.0 * inverted, "real_estate": 0.25 * normal - 0.5 * inverted})
        for a in present(data, ("GLD",)):
            scores[a] = 0.5 * inverted
        return scores.where(data.investable)


@register_model("dollar_strength", "macro", "Dollar strength: short emerging-market equities, commodities and gold when the broad dollar is rising")
class DollarStrength(ForecastModel):
    """A stronger dollar tightens global financial conditions and weighs on dollar-priced commodities and emerging markets."""

    name, family, position_mode = "dollar_strength", "macro", "time_series"
    requires = ("DTWEXBGS",)

    def __init__(self, window: int = 126, exposed: tuple = ("EEM", "DBC", "GLD", "SLV")):
        self.window, self.exposed = window, tuple(exposed)

    def score(self, data):
        self.require(data)
        s = np.tanh(expanding_z(data.macro["DTWEXBGS"].diff(self.window).dropna(), 504).reindex(data.index) / 2.0)
        return zero_except(data.index, data.assets, {a: -s for a in self.exposed}).where(data.investable)


@register_model("commodity_supercycle", "macro", "Commodity supercycle: follow the five-year trend of the commodity assets")
class CommoditySupercycle(ForecastModel):
    """Commodity cycles last a decade because supply responds slowly; a five-year rise signals the upswing is under way."""

    name, family, position_mode = "commodity_supercycle", "macro", "time_series"

    def __init__(self, window: int = 1260):
        self.window = window

    def score(self, data):
        ret = data.prices / data.prices.shift(self.window) - 1.0
        s = ret.apply(lambda c: np.tanh(expanding_z(c.dropna(), 504, 4.0).reindex(c.index) / 2.0))
        return zero_except(data.index, data.assets, {a: s[a] for a in in_classes(data, "commodity")}).where(data.investable)


@register_model("risk_on_off", "macro", "Risk-on / risk-off: a composite of VIX, the credit spread, the curve and MOVE sets equity and credit exposure against bonds and gold")
class RiskOnOff(ForecastModel):
    """Stress indicators move together; when they rise, risk assets keep falling for a while and safe assets keep rising."""

    name, family, position_mode = "risk_on_off", "macro", "time_series"
    requires = ("VIX", "BAA10Y", "T10Y3M", "MOVE")

    def score(self, data):
        self.require(data)
        parts = [expanding_z(data.macro[c].dropna(), 504).reindex(data.index) for c in ("VIX", "BAA10Y", "MOVE")]
        parts.append(-expanding_z(data.macro["T10Y3M"].dropna(), 504).reindex(data.index))
        off = np.tanh(pd.concat(parts, axis=1).mean(axis=1) / 1.5)
        scores = _by_class(data, {"equity": -off, "credit": -off, "rates": off, "commodity": -0.3 * off, "real_estate": -off})
        for a in present(data, ("GLD",)):
            scores[a] = 0.5 * off
        return scores.where(data.investable)


POSITIONING_MAP = {"cftc_es": "SPY", "cftc_nq": "QQQ", "cftc_ust10": "IEF", "cftc_gold": "GLD", "cftc_silver": "SLV", "cftc_crude": "DBC"}


@register_model("cftc_positioning", "macro", "CFTC positioning: fade crowded speculative net positioning in the futures matching each ETF (sign configurable)")
class CftcPositioning(ForecastModel):
    """When leveraged funds or managed money are extremely long, the marginal buyer may already be in. The Generation 3 test found no
    significant predictive power for the next month, so treat this as a documented null result and a template for non-price data, not an edge."""

    name, family, position_mode = "cftc_positioning", "macro", "time_series"

    def __init__(self, sign: int = -1, min_periods: int = 756, mapping: dict | None = None):
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 (fade) or +1 (follow)")
        if min_periods < 52:
            raise ValueError("min_periods must be at least 52 trading days")
        self.sign, self.min_periods, self.mapping = sign, min_periods, dict(mapping or POSITIONING_MAP)

    def score(self, data):
        have = {k: a for k, a in self.mapping.items() if k in data.macro.columns and data.macro[k].notna().any() and a in data.assets}
        if not have:
            raise KeyError(f"cftc_positioning needs one of {sorted(self.mapping)} in the bundle's macro frame (run the Generation 3 download, stage 23)")
        values = {asset: self.sign * np.tanh(expanding_z(data.macro[key].dropna(), self.min_periods).reindex(data.index).ffill() / 2.0) for key, asset in have.items()}
        return zero_except(data.index, data.assets, values).where(data.investable)
