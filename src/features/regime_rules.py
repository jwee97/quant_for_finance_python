"""Transparent, causal, named regimes (Generation 2, Priority 1).

Rule-based regimes are the baseline any statistical regime model has to justify
itself against. They are less clever and far easier to audit: every definition
fits in a sentence and every input is a price, a return or a published macro
figure available on the day.

All four are causal. The state on day t is a function of data up to and
including t, so it is known at the close of t and tradable from t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .macro import expanding_percentile


def bear_market_state(price: pd.Series, drawdown_trigger: float = 0.20,
                      recovery: float = 0.20) -> pd.Series:
    """1 during a bear market, 0 otherwise.

    The conventional rule: a bear market is *confirmed* when the price falls
    ``drawdown_trigger`` below its running peak, and ends when it has risen
    ``recovery`` above the trough reached since. Confirmation therefore comes
    late by construction (after a 20% fall), which is a property of the
    definition, not a defect to be tuned away.
    """
    values = price.to_numpy(dtype=float)
    state = np.zeros(len(values))
    first = np.flatnonzero(np.isfinite(values))
    if len(first) == 0:
        return pd.Series(state, index=price.index, name="bear")
    peak = trough = values[first[0]]
    bear = False
    for i, p in enumerate(values):
        if np.isfinite(p):
            if not bear:
                peak = max(peak, p)
                if p <= peak * (1.0 - drawdown_trigger):
                    bear, trough = True, p
            else:
                trough = min(trough, p)
                if p >= trough * (1.0 + recovery):
                    bear, peak = False, p
        state[i] = 1.0 if bear else 0.0
    return pd.Series(state, index=price.index, name="bear")


def volatility_regime(returns: pd.Series, halflife: float = 21.0, high_percentile: float = 0.80,
                      low_percentile: float = 0.20, min_history: int = 252,
                      annualisation: int = 252) -> pd.DataFrame:
    """High / low volatility from the percentile of EWMA volatility within its own history.

    The percentile is expanding: today's volatility is ranked against every
    value up to and including today, never against the full sample.
    """
    variance = (returns ** 2).ewm(halflife=halflife, adjust=False).mean()
    vol = np.sqrt(variance * annualisation)
    percentile = expanding_percentile(vol, min_periods=min_history)
    return pd.DataFrame({
        "vol": vol,
        "percentile": percentile,
        "high": (percentile >= high_percentile).astype(float).where(percentile.notna()),
        "low": (percentile <= low_percentile).astype(float).where(percentile.notna()),
    })


def inflation_shock(cpi_yoy: pd.Series, stock: pd.Series, bond: pd.Series, cpi_min: float = 0.04,
                    corr_min: float = 0.0, window: int = 126) -> pd.DataFrame:
    """Inflation at or above ``cpi_min`` AND bonds failing to hedge equities.

    ``cpi_yoy`` must be the series AS KNOWN on each date (published, lagged),
    which the macro panel provides. The stock-bond correlation is the trailing
    ``window``-day correlation of daily returns; a non-negative value means the
    usual diversification between the two has gone.
    """
    correlation = stock.rolling(window, min_periods=window // 2).corr(bond)
    known = cpi_yoy.reindex(stock.index)
    shock = ((known >= cpi_min) & (correlation >= corr_min)).astype(float)
    shock = shock.where(known.notna() & correlation.notna())
    return pd.DataFrame({"cpi_yoy": known, "stock_bond_corr": correlation, "shock": shock})


def liquidity_crisis(vol_percentile: pd.Series, credit: pd.Series, safe: pd.Series,
                     vol_percentile_min: float = 0.90, window: int = 21,
                     credit_max: float = -0.05) -> pd.DataFrame:
    """A volatility spike AND credit underperforming safe duration by ``credit_max`` over ``window`` days."""
    def compounded(r: pd.Series) -> pd.Series:
        return np.expm1(np.log1p(r.fillna(0.0)).rolling(window, min_periods=window).sum())

    relative = compounded(credit) - compounded(safe)
    crisis = ((vol_percentile >= vol_percentile_min) & (relative <= credit_max)).astype(float)
    crisis = crisis.where(vol_percentile.notna() & relative.notna())
    return pd.DataFrame({"relative_credit_return": relative, "crisis": crisis})
