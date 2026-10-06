"""Volatility strategies built on implied-volatility indices (VIX, VIX3M, VXN, GVZ, OVX) against realised volatility.

This universe has no option or volatility-futures instruments, so these do not trade volatility directly. They trade the
underlying ETF, using the volatility risk premium (implied minus realised) as the forecast of its return. Dispersion trading
needs single-name and index options and is not built (see docs/strategies/dispersion.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import expanding_z, in_classes, present, zero_except


def _realised_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)


@register_model("vrp_timing", "volatility", "Volatility risk premium timing: be long equities when implied variance (VIX squared) exceeds realised variance")
class VRPTiming(ForecastModel):
    """Investors pay more for protection than realised volatility justifies; that premium also forecasts equity returns (Bollerslev, Tauchen and Zhou)."""

    name, family, position_mode = "vrp_timing", "volatility", "time_series"
    requires = ("VIX",)

    def __init__(self, realised_window: int = 21, proxy: str = "SPY"):
        self.realised_window, self.proxy = realised_window, proxy

    def score(self, data):
        self.require(data)
        proxy = self.proxy if self.proxy in data.assets else data.assets_in("equity")[0]
        implied_var = (data.macro["VIX"] / 100.0) ** 2
        realised_var = _realised_vol(data.returns[[proxy]], self.realised_window)[proxy] ** 2
        z = expanding_z((implied_var - realised_var).dropna(), 504).reindex(data.index)
        return zero_except(data.index, data.assets, {a: np.tanh(z / 2.0) for a in in_classes(data, "equity")}).where(data.investable)


@register_model("variance_carry", "volatility", "Variance carry: be long equities when the volatility term structure is in contango (VIX3M above VIX)")
class VarianceCarry(ForecastModel):
    """Contango means the market expects calm to continue and pays to hold protection; harvesting it is selling insurance."""

    name, family, position_mode = "variance_carry", "volatility", "time_series"
    requires = ("VIX", "VIX3M")

    def score(self, data):
        self.require(data)
        slope = np.log(data.macro["VIX3M"] / data.macro["VIX"])
        z = expanding_z(slope.dropna(), 504).reindex(data.index)
        return zero_except(data.index, data.assets, {a: np.tanh(z / 2.0) for a in in_classes(data, "equity")}).where(data.investable)


IMPLIED = {"SPY": "VIX", "QQQ": "VXN", "GLD": "GVZ", "DBC": "OVX"}


@register_model("implied_vs_realized", "volatility", "Implied versus realised volatility by asset: long where the implied index exceeds trailing realised volatility by the most")
class ImpliedVsRealised(ForecastModel):
    """Where fear (implied) runs ahead of experience (realised), the premium paid for protection is large and tends to be earned by the seller."""

    name, family = "implied_vs_realized", "volatility"
    requires = ("VIX", "VXN", "GVZ", "OVX")

    def __init__(self, realised_window: int = 21):
        self.realised_window = realised_window

    def score(self, data):
        self.require(data)
        realised = _realised_vol(data.returns, self.realised_window)
        out = pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        for asset in present(data, IMPLIED):
            implied = data.macro[IMPLIED[asset]] / 100.0
            out[asset] = (implied - realised[asset]) / implied
        return out.where(data.investable)
