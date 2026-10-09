"""Dated external cash flows: contributions and withdrawals as a series of dollars."""

from __future__ import annotations

import pandas as pd

EVERY = {"weekly": "W", "monthly": "M", "quarterly": "Q", "annual": "Y"}


def period_starts(index: pd.DatetimeIndex, every: str = "monthly") -> pd.DatetimeIndex:
    """The first trading day of each week, month, quarter or year in ``index``."""
    if every not in EVERY:
        raise ValueError(f"every must be one of {sorted(EVERY)}")
    index = pd.DatetimeIndex(index).sort_values()
    return pd.DatetimeIndex(pd.Series(index, index=index.to_period(EVERY[every])).groupby(level=0).min().to_numpy())


def flow_schedule(index: pd.DatetimeIndex, deposit: float = 0.0, withdraw: float = 0.0, deposit_every: str = "monthly", withdraw_every: str = "monthly", growth: float = 0.0) -> pd.Series:
    """External flows in dollars on the first trading day of each period: deposits positive, withdrawals negative (give both as positive amounts).

    ``growth`` is a yearly rate at which both amounts grow from the first date (a contribution that rises with pay, a withdrawal that keeps up with inflation). The first period's flow is dated on
    the first trading day of the index, so a series that starts mid-month still gets that month's flow."""
    if deposit < 0 or withdraw < 0:
        raise ValueError("deposit and withdraw are amounts (give them positive); withdrawals are made negative here")
    index = pd.DatetimeIndex(index).sort_values()
    out = pd.Series(0.0, index=index)
    for amount, every, sign in ((deposit, deposit_every, 1.0), (withdraw, withdraw_every, -1.0)):
        if amount == 0.0:
            continue
        for day in period_starts(index, every):
            years = (day - index[0]).days / 365.25
            out[day] += sign * amount * (1.0 + growth) ** years
    return out
