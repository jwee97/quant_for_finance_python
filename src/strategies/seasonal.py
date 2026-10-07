"""Calendar effects: turn of the month, Halloween (sell in May) and each asset's own calendar-month history.

The calendar is public in advance, so a rule that depends only on the date is not look-ahead. ``seasonal_rank`` uses only earlier years' returns in the same calendar month.
These are among the most-cited anomalies and among the most data-mined: treat a good backtest here with extra suspicion and count it as a trial.
"""

from __future__ import annotations

import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import in_classes, zero_except


def _targets(data, classes: tuple) -> list[str]:
    chosen = in_classes(data, *classes) if classes else []
    return chosen or list(data.assets)                 # a universe without class labels gets the rule on every asset


@register_model("turn_of_month", "time-series", "Turn-of-the-month effect: hold equities from the last trading day of a month through the third trading day of the next, otherwise stay out")
class TurnOfMonth(ForecastModel):
    """Pension and payroll inflows, and the settlement of month-end flows, concentrate buying around the month boundary; most of the equity premium has historically come in these few days."""

    name, family, position_mode = "turn_of_month", "seasonal", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, before: int = 1, after: int = 3, classes: tuple = ("equity",)):
        if before < 0 or after < 0 or before + after < 1:
            raise ValueError("before >= 0, after >= 0 and at least one day in the window")
        self.before, self.after, self.classes = before, after, tuple(classes)

    def flags(self, index: pd.DatetimeIndex) -> pd.Series:
        idx = pd.DatetimeIndex(index)
        month = pd.Series(idx.to_period("M"), index=idx)
        from_start = month.groupby(month).cumcount()                                   # 0 = first trading day of the month
        from_end = month.groupby(month).cumcount(ascending=False)                      # 0 = last trading day of the month
        return ((from_start < self.after) | (from_end < self.before)).astype(float)

    def score(self, data):
        flag = self.flags(data.index)
        score = zero_except(data.index, data.assets, {a: flag for a in _targets(data, self.classes)})
        return score.where(data.investable)


@register_model("sell_in_may", "time-series", "Halloween indicator (sell in May): hold equities from November through April and stay out from May through October")
class SellInMay(ForecastModel):
    """Equity returns have historically been concentrated in the winter half of the year (Bouman and Jacobsen 2002), with no agreed risk explanation: a famous, fragile rule."""

    name, family, position_mode = "sell_in_may", "seasonal", "time_series"
    book = "sleeves"

    def __init__(self, start_month: int = 11, end_month: int = 4, classes: tuple = ("equity",)):
        if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
            raise ValueError("months are 1 to 12")
        self.start_month, self.end_month, self.classes = start_month, end_month, tuple(classes)

    def score(self, data):
        m = pd.Series(pd.DatetimeIndex(data.index).month, index=data.index)
        inside = (m >= self.start_month) | (m <= self.end_month) if self.start_month > self.end_month else (m >= self.start_month) & (m <= self.end_month)
        score = zero_except(data.index, data.assets, {a: inside.astype(float) for a in _targets(data, self.classes)})
        return score.where(data.investable)


@register_model("seasonal_rank", "cross-sectional", "Same-month seasonality (Keloharju-Linnainmaa-Nyberg): rank assets by their average return in this calendar month over earlier years")
class SeasonalRank(ForecastModel):
    """Assets with a recurring pattern in a given month (heating-oil demand, tax-loss selling, earnings cycles) tend to repeat it; only earlier years are used."""

    name, family = "seasonal_rank", "seasonal"

    def __init__(self, min_years: int = 5):
        if min_years < 2:
            raise ValueError("min_years must be at least 2")
        self.min_years = min_years

    def score(self, data):
        monthly = (1.0 + data.returns.fillna(0.0)).groupby(data.returns.index.to_period("M")).prod() - 1.0
        monthly = monthly.where(data.prices.notna().groupby(data.prices.index.to_period("M")).any())
        calendar_month = pd.Series(monthly.index.month, index=monthly.index)
        history = monthly.groupby(calendar_month).transform(lambda s: s.shift(1).expanding(min_periods=self.min_years).mean())     # earlier years only
        daily = history.reindex(data.index.to_period("M")).set_axis(data.index)
        return daily.where(data.investable)
