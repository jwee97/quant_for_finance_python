"""Fixed-income strategies, expressed with Treasury ETFs and the Treasury yield curve.

SHY, IEF and TLT stand in for the 2-, 10- and 20-year points. Their durations are NOT assumed: they are estimated each day from
the previous year of ETF returns against changes in the matching constant-maturity yield, using only data through that day.
The spread trades are DV01-neutral on those estimates. The yield series are constant-maturity yields, not the ETFs' own holdings,
so this is an approximation, said here rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import expanding_z, present, zero_except

LEG = {"SHY": "DGS2", "IEF": "DGS10", "TLT": "DGS20"}


def rolling_duration(data, asset: str, series: str, window: int = 252) -> pd.Series:
    """``-cov(r, dy) / var(dy)`` over the trailing window: the price sensitivity to a one-unit (100 percentage point) change in yield."""
    dy = data.macro[series].diff() / 100.0
    r = data.returns[asset]
    cov = r.rolling(window, min_periods=window // 2).cov(dy)
    var = dy.rolling(window, min_periods=window // 2).var()
    return (-(cov / var.replace(0.0, np.nan))).clip(lower=0.25)


class _Curve(ForecastModel):
    family = "fixed income"
    structured = True
    position_mode = "time_series"
    requires = ("DGS2", "DGS5", "DGS10", "DGS20")

    def durations(self, data) -> pd.DataFrame:
        return pd.DataFrame({a: rolling_duration(data, a, LEG[a]) for a in present(data, LEG)})

    def signal(self, data) -> pd.Series:
        raise NotImplementedError

    def legs(self, data, signal: pd.Series, durations: pd.DataFrame) -> dict[str, pd.Series]:
        raise NotImplementedError

    def weights(self, data):
        self.require(data)
        dur = self.durations(data)
        legs = self.legs(data, self.signal(data), dur)
        return zero_except(data.index, data.assets, legs)

    def score(self, data):
        w = self.weights(data)
        return w.where(data.investable)


@register_model("curve_steepener", "fixed income", "Curve steepener: long the 2-year (SHY), short the 20-year (TLT), DV01-neutral, sized by the momentum of the slope")
class CurveSteepener(_Curve):
    """Curve moves persist: a slope that has been steepening keeps steepening while the Fed or term premia adjust."""

    name = "curve_steepener"

    def __init__(self, change_window: int = 63):
        self.change_window = change_window

    def signal(self, data):
        slope = data.macro["DGS20"] - data.macro["DGS2"]
        change = slope.diff(self.change_window)
        return np.tanh(expanding_z(change.dropna(), 504, 4.0).reindex(data.index) / 2.0)

    def legs(self, data, signal, durations):
        return {"SHY": signal / durations["SHY"], "TLT": -signal / durations["TLT"]}


@register_model("butterfly", "fixed income", "Butterfly: long the 10-year belly against the 2- and 20-year wings, DV01-neutral, when the belly looks cheap")
class Butterfly(_Curve):
    """The belly yield relative to the average of the wings mean-reverts: 2 x 10y - 2y - 20y is a stationary curvature measure."""

    name = "butterfly"

    def signal(self, data):
        fly = 2.0 * data.macro["DGS10"] - data.macro["DGS2"] - data.macro["DGS20"]
        return np.tanh(expanding_z(fly.dropna(), 504, 4.0).reindex(data.index) / 2.0)

    def legs(self, data, signal, durations):
        return {"IEF": signal / durations["IEF"], "SHY": -0.5 * signal / durations["SHY"], "TLT": -0.5 * signal / durations["TLT"]}


MATURITY = {"SHY": ("DGS2", None, 2, None), "AGG": ("DGS5", "DGS2", 5, 2), "IEF": ("DGS10", "DGS5", 10, 5), "TLT": ("DGS20", "DGS10", 20, 10)}


@register_model("carry_rolldown", "fixed income", "Carry plus roll-down: favour the bond ETFs with the highest yield over cash plus the roll down the curve")
class CarryRolldown(ForecastModel):
    """A bond earns its yield and, if the curve is unchanged, ages down it: price gains of duration times the local slope per year."""

    name, family = "carry_rolldown", "fixed income"
    requires = ("DGS2", "DGS5", "DGS10", "DGS20", "DFF")

    def score(self, data):
        self.require(data)
        out = pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        for asset in present(data, MATURITY):
            y, y_short, m, m_short = MATURITY[asset]
            carry = (data.macro[y] - data.macro["DFF"]) / 100.0
            if y_short is None:
                roll = 0.0
            else:
                duration = rolling_duration(data, asset, y)
                roll = duration * (data.macro[y] - data.macro[y_short]) / 100.0 / (m - m_short)
            out[asset] = carry + roll
        return out.where(data.investable)


@register_model("duration_timing", "fixed income", "Duration timing: extend duration when yields have been falling, shorten when they have been rising")
class DurationTiming(ForecastModel):
    """Yield changes trend (policy is gradual), so recent yield direction forecasts bond returns."""

    name, family, position_mode = "duration_timing", "fixed income", "time_series"
    requires = ("DGS10",)

    def __init__(self, window: int = 126, assets: tuple = ("IEF", "TLT", "AGG")):
        self.window, self.assets = window, tuple(assets)

    def score(self, data):
        self.require(data)
        change = data.macro["DGS10"].diff(self.window)
        s = np.tanh(-expanding_z(change.dropna(), 504, 4.0).reindex(data.index) / 2.0)
        return zero_except(data.index, data.assets, {a: s for a in self.assets}).where(data.investable)
