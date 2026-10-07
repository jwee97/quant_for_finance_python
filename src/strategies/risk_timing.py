"""Risk-managed exposure: scale or switch exposure on measures of risk rather than on a forecast of return.

Volatility-managed portfolios (Moreira and Muir 2017), buying equities after a volatility spike, and credit-versus-rates risk appetite. These are risk overlays as much as
strategies: their main effect is to lower exposure when risk is high, which improves the Sharpe ratio when risk is persistent and return per unit of risk is not higher in turbulence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import expanding_z, in_classes, stateful, zero_except

RISK_CLASSES = ("equity", "credit", "real_estate", "commodity")
SAFE_CLASSES = ("rates", "fixed_income")


@register_model("vol_managed_long", "time-series", "Volatility-managed long (Moreira-Muir): hold each asset in inverse proportion to its last month's realised variance, long only")
class VolatilityManagedLong(ForecastModel):
    """Volatility is persistent but the return earned per unit of risk is not higher when it is high, so shrinking exposure in turbulence raises the Sharpe ratio."""

    name, family, position_mode = "vol_managed_long", "volatility", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, window: int = 21, max_scale: float = 3.0, min_history: int = 252):
        if window < 5 or max_scale <= 0 or min_history < window:
            raise ValueError("window >= 5, max_scale > 0, min_history >= window")
        self.window, self.max_scale, self.min_history = window, max_scale, min_history

    def score(self, data):
        variance = (data.returns ** 2).rolling(self.window, min_periods=self.window).sum()
        typical = variance.expanding(min_periods=self.min_history).mean()             # the constant c is the history's own average variance, so average exposure is about one
        return (typical / variance.replace(0.0, np.nan)).clip(upper=self.max_scale).where(data.investable)


@register_model("vix_spike_reversion", "time-series", "Buy fear: hold equities after the VIX jumps more than two standard deviations above its 63-day mean, exit when it falls back to the mean")
class VIXSpikeReversion(ForecastModel):
    """Implied-volatility spikes overshoot because investors pay up for protection, and equities have tended to recover once the panic fades."""

    name, family, position_mode = "vix_spike_reversion", "volatility", "time_series"
    rebalance = "daily"
    book = "sleeves"
    requires = ("VIX",)

    def __init__(self, window: int = 63, entry_z: float = 2.0, classes: tuple = ("equity",)):
        if window < 20 or entry_z <= 0:
            raise ValueError("window >= 20 and entry_z > 0")
        self.window, self.entry_z, self.classes = window, entry_z, tuple(classes)

    def score(self, data):
        self.require(data)
        vix = data.macro["VIX"].reindex(data.index).ffill()
        mean, sd = vix.rolling(self.window, min_periods=self.window).mean(), vix.rolling(self.window, min_periods=self.window).std()
        z = (vix - mean) / sd.replace(0.0, np.nan)
        targets = in_classes(data, *self.classes) or list(data.assets)
        enter = pd.DataFrame({a: z > self.entry_z for a in targets}).reindex(columns=data.assets, fill_value=False)
        leave = pd.DataFrame({a: vix < mean for a in targets}).reindex(columns=data.assets, fill_value=False)
        out = stateful(enter, leave)
        return out.where(z.notna(), axis=0).where(data.investable)


@register_model("credit_spread_timing", "time-series", "Credit risk appetite: when credit has been beating rates over three months (relative to its own history) hold risk assets, otherwise lean to safety")
class CreditSpreadTiming(ForecastModel):
    """Credit markets price default risk faster than equities; credit underperforming government bonds warns that risk appetite is fading."""

    name, family, position_mode = "credit_spread_timing", "macro", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, window: int = 63, history: int = 504):
        if window < 10 or history < 2 * window:
            raise ValueError("window >= 10 and history >= 2 * window")
        self.window, self.history = window, history

    def score(self, data):
        credit, safe = in_classes(data, "credit"), in_classes(data, *SAFE_CLASSES)
        if not credit or not safe:
            raise KeyError("credit_spread_timing needs at least one asset of class 'credit' and one of class 'rates' or 'fixed_income'")
        logp = np.log(data.prices)
        ratio = logp[credit].mean(axis=1) - logp[safe].mean(axis=1)               # the credit-minus-rates relative performance index
        signal = np.tanh(expanding_z(ratio.diff(self.window).dropna(), self.history, 4.0).reindex(data.index) / 2.0)
        risk = in_classes(data, *RISK_CLASSES)
        values = {a: signal for a in risk}
        values.update({a: -0.5 * signal for a in safe if a not in risk})
        return zero_except(data.index, data.assets, values).where(signal.notna(), axis=0).where(data.investable)
