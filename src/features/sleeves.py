"""Asset-class sleeves: equal-weighted groups of ETFs (Generation 2).

A sleeve turns a noisy 15-asset panel into a handful of economically meaningful
return streams. Used by the regime models (equity / rates / real assets as the
hidden-Markov emissions) and by the macro predictability study (five asset
classes as forecast targets).

Equal weighting within a sleeve is deliberate: no estimated weights, so no
estimation error, and a membership list that is declared in config before any
result is seen.
"""

from __future__ import annotations

import pandas as pd


def sleeve_returns(returns: pd.DataFrame, sleeves: dict[str, list[str]],
                   investable: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily equal-weighted return of each sleeve.

    Averages over the members that are investable and have a return on that
    date, so a sleeve that gains a member mid-sample (HYG in 2007) simply
    averages over more assets from then on, rather than carrying a hole.
    """
    data = returns.where(investable) if investable is not None else returns
    out = {}
    for name, members in sleeves.items():
        present = [m for m in members if m in data.columns]
        if not present:
            raise KeyError(f"sleeve '{name}' has no members in the return panel: {members}")
        out[name] = data[present].mean(axis=1, skipna=True)
    return pd.DataFrame(out, index=returns.index)


def monthly_compound(daily: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Calendar-month compounded return, stamped at the month's last trading day."""
    grouped = (1.0 + daily.fillna(0.0)).groupby(daily.index.to_period("M")).prod() - 1.0
    last_day = pd.Series(daily.index, index=daily.index).groupby(daily.index.to_period("M")).max()
    grouped.index = pd.DatetimeIndex(last_day.loc[grouped.index].to_numpy())
    return grouped


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last observed trading day of every calendar month."""
    index = pd.DatetimeIndex(index)
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period("M")).max().to_numpy())
