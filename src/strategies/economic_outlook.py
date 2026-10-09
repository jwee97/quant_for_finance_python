"""Economic-outlook strategies: the yield curve and the credit cycle, each turned into a few states whose consequences are LEARNED from the past, not assumed.

    curve_quadrant        yield-curve strategy: the 63-day change in the level and the slope of the curve gives four states (bull/bear steepener/flattener); each asset is held in the direction its
                          own matured history says it has paid in the state the curve is in now
    credit_cycle_rotation credit strategy: the credit spread's level (high or low) and direction (widening or tightening) give four phases of the credit cycle; assets are held in the direction
                          each has paid in the current phase

The textbook table ("bear flattener: sell long bonds, buy cash") is a hypothesis, and the sign of the stock-bond relationship has changed with the inflation regime. The rule here is to measure: for every
(state, asset) pair it uses only the next-month returns that were complete by the decision date and trades the pair when their average is ``tstat`` standard errors from zero, with the standard
error corrected for the overlap of the monthly windows. States and assets with too little evidence stay flat. Without a real edge in the data most of the book is flat most of the time, which is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import in_classes

CURVE_STATES = ("bull flattener", "bull steepener", "bear flattener", "bear steepener")        # code = 2 * (the level rose) + (the slope rose)
CREDIT_STATES = ("recovery", "expansion", "contraction", "late cycle")                          # code = 2 * (spreads widening) + (spreads low)


def conditional_tstat(state: pd.Series, returns: pd.DataFrame, horizon: int, min_obs: int = 500) -> pd.DataFrame:
    """For every date and asset: the t-statistic of the asset's mean next-``horizon``-day return in the state the market is in on that date.

    The mean uses only (state, forward return) pairs that had MATURED by the date: the pair dated ``s`` is in from day ``s + horizon``, when its last return is known. Windows of ``horizon`` days
    overlap, so the effective number of independent observations is ``n / horizon`` and the standard error is ``sd * sqrt(horizon / n)``. States with fewer than ``min_obs`` matured observations
    (days, not independent windows) give NaN. ``state`` holds integer codes, NaN where the state is unknown."""
    if horizon < 1:
        raise ValueError("horizon >= 1")
    idx = returns.index
    codes = state.reindex(idx).to_numpy(dtype=float)
    log_r = np.log1p(returns.where(returns > -1.0))
    forward = log_r.rolling(horizon, min_periods=horizon).sum().shift(-horizon).to_numpy()                 # the log return over s+1 .. s+horizon, known at s+horizon
    ok = np.isfinite(forward)
    x = np.where(ok, forward, 0.0)
    T, K = x.shape
    out = np.full((T, K), np.nan)

    def matured(a: np.ndarray) -> np.ndarray:                                                                # the running total of pairs dated <= t - horizon
        c = np.cumsum(a, axis=0)
        return np.vstack([np.zeros((horizon, a.shape[1])), c[:T - horizon]]) if T > horizon else np.zeros_like(c)

    for code in np.unique(codes[np.isfinite(codes)]):
        member = (codes == code)[:, None]
        n, s1, s2 = matured((member & ok).astype(float)), matured(np.where(member, x, 0.0)), matured(np.where(member, x * x, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s1 / n
            var = np.maximum(s2 / n - mean ** 2, 1e-12)
            t = mean / np.sqrt(var * horizon / n)
        now = codes == code
        out[now] = np.where(n[now] >= min_obs, t[now], np.nan)
    return pd.DataFrame(out, index=idx, columns=returns.columns)


def _score(t: pd.DataFrame, tstat: float, scale: float) -> pd.DataFrame:
    """tanh of the t-statistic where it is at least ``tstat`` in size, else 0 where there is evidence to look at and NaN where there is none yet."""
    s = np.tanh(t / scale)
    return s.where(t.abs() >= tstat, 0.0).where(t.notna())


class _LearnedStates(ForecastModel):
    """Common arithmetic: ``states(data)`` gives the state codes, ``score`` learns what each state has paid."""

    family, position_mode = "macro", "time_series"
    book, rebalance = "sleeves", "weekly"
    state_names: tuple = ()

    def __init__(self, horizon: int = 21, min_obs: int = 500, tstat: float = 1.5, scale: float = 2.0, assets: tuple = ()):
        if horizon < 1 or min_obs < 60 or tstat < 0 or scale <= 0:
            raise ValueError("horizon >= 1, min_obs >= 60, tstat >= 0, scale > 0")
        self.horizon, self.min_obs, self.tstat, self.scale, self.assets = int(horizon), int(min_obs), tstat, scale, tuple(assets)

    def states(self, data) -> pd.Series:
        raise NotImplementedError

    def named_states(self, data) -> pd.Series:
        """The state on every date as text (``unknown`` while it is not defined)."""
        code = self.states(data)
        names = pd.Series(["unknown"] * len(code), index=code.index, dtype=object)
        for k, label in enumerate(self.state_names):
            names[code == k] = label
        return names

    def score(self, data):
        chosen = [a for a in self.assets if a in data.assets] or list(data.assets)
        t = conditional_tstat(self.states(data), data.returns[chosen], self.horizon, self.min_obs)
        out = _score(t, self.tstat, self.scale).reindex(columns=data.assets)
        return out.where(data.investable)


@register_model("curve_quadrant", "macro", "Yield-curve strategy: the change in the level and slope of the curve gives four states (bull/bear steepener/flattener); each asset is held in the direction it has paid in the current state")
class CurveQuadrant(_LearnedStates):
    """Rates fall or rise (bull or bear) and the curve steepens or flattens: the four combinations mean different things for growth, inflation and policy (a bull steepener is typically an easing cycle,
    a bear flattener a tightening one) and so for bonds, equities, credit and gold. Instead of fixing the table, the strategy measures the average next-month return of each asset in each state from
    earlier data only and trades the combinations that are significant. Needs the 2- and 10-year Treasury yields (``DGS2``, ``DGS10``) in the bundle's macro frame."""

    name = "curve_quadrant"
    requires = ("DGS2", "DGS10")
    state_names = CURVE_STATES

    def __init__(self, window: int = 63, horizon: int = 21, min_obs: int = 500, tstat: float = 1.5, scale: float = 2.0, assets: tuple = ()):
        super().__init__(horizon, min_obs, tstat, scale, assets)
        if window < 5:
            raise ValueError("window >= 5")
        self.window = int(window)

    def states(self, data) -> pd.Series:
        self.require(data)
        d2, d10 = data.macro["DGS2"].diff(self.window), data.macro["DGS10"].diff(self.window)
        level, slope = (d2 + d10) / 2.0, d10 - d2
        code = 2.0 * (level > 0).astype(float) + (slope > 0).astype(float)
        return code.where(level.notna() & slope.notna() & (level != 0) & (slope != 0)).reindex(data.index)


@register_model("credit_cycle_rotation", "macro", "Credit strategy: the credit spread's level (high or low) and direction (widening or tightening) give four phases of the credit cycle; each asset is held in the direction it has paid in the current phase")
class CreditCycleRotation(_LearnedStates):
    """Credit spreads lead the cycle: they tighten in recovery and expansion, begin to widen late in the cycle and blow out in contractions, and risky assets, Treasuries and gold earn very different
    returns in each phase. The spread is the BAA-Treasury spread (``BAA10Y``) when the bundle has it, otherwise the relative performance of the credit assets against the Treasury assets (spreads
    widen when credit lags). High or low is against the spread's own trailing three-year median; widening or tightening is its change over ``window`` days. As in ``curve_quadrant`` the
    consequences are measured from earlier data and only the significant ones are traded."""

    name = "credit_cycle_rotation"
    state_names = CREDIT_STATES

    def __init__(self, window: int = 63, horizon: int = 21, min_obs: int = 500, tstat: float = 1.5, scale: float = 2.0, assets: tuple = ()):
        super().__init__(horizon, min_obs, tstat, scale, assets)
        if window < 5:
            raise ValueError("window >= 5")
        self.window = int(window)

    def spread(self, data) -> pd.Series:
        """The credit spread, or the minus cumulative relative return of credit against Treasuries as a stand-in (higher means wider)."""
        if "BAA10Y" in data.macro.columns and data.macro["BAA10Y"].notna().any():
            return data.macro["BAA10Y"].reindex(data.index).ffill()
        credit, treasuries = in_classes(data, "credit"), in_classes(data, "rates")
        if not credit or not treasuries:
            raise KeyError("credit_cycle_rotation needs the BAA10Y macro series, or an asset of class credit (HYG, LQD) and an asset of class rates (IEF, TLT, SHY)")
        relative = data.returns[credit].where(data.investable[credit]).mean(axis=1) - data.returns[treasuries].where(data.investable[treasuries]).mean(axis=1)
        return -relative.fillna(0.0).cumsum()

    def states(self, data) -> pd.Series:
        s = self.spread(data)
        median = s.rolling(756, min_periods=252).median()
        change = s.diff(self.window)
        widening, low = (change > 0).astype(float), (s <= median).astype(float)
        return (2.0 * widening + low).where(change.notna() & median.notna() & (change != 0))
