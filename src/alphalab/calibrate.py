"""From a standardised signal to an expected residual return: the basic forecasting formula.

    alpha_(t,i)  =  IC_t  x  omega_(t,i)  x  z_(t,i)

``z`` is the signal standardised across assets (mean 0, variance 1) on date ``t``; ``omega`` the residual volatility of the asset over the forecast horizon; ``IC`` the information coefficient the signal is
believed to have. Everything on the right is known at ``t``, so the left is a forecast with units: a return over the horizon, which is what a portfolio optimiser needs and what a raw score is not.

``IC_t`` is estimated from the ICs that had MATURED by ``t`` (a window of the last ``window``) and shrunk toward zero by the signal-to-noise ratio of its own estimate: ``mean x mean^2 / (mean^2 + se^2)``
with ``se`` the standard error of the mean. A signal whose average IC is many standard errors from zero is taken at face value; one that is within a standard error or two of zero is taken at a fraction of it,
and a signal that has worked for three weeks is not believed as one that has worked for five years.

``reliability`` checks the result: a calibrated alpha has a slope of one in the cross-sectional regression of the realised residual return on it. A slope below one means the alpha is too big (overconfident);
above one, too timid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..signals.transform import cross_sectional_zscore, winsorize
from .ic import ic_series, known_at


def market_return(returns: pd.DataFrame) -> pd.Series:
    """The equal-weighted cross-sectional mean return: the default 'market' when none is supplied."""
    return returns.mean(axis=1)


def residual_returns(returns: pd.DataFrame, market: pd.Series | None = None, window: int = 252, min_periods: int = 60) -> pd.DataFrame:
    """Returns with the market removed: ``r_(t,i) - beta_(t-1,i) m_t`` with ``beta`` the trailing regression slope on the market, estimated on data up to ``t-1`` (no look-ahead)."""
    m = (market if market is not None else market_return(returns)).reindex(returns.index)
    cov = returns.rolling(window, min_periods=min_periods).cov(m)
    var = m.rolling(window, min_periods=min_periods).var()
    beta = cov.div(var, axis=0).shift(1)
    return returns - beta.mul(m, axis=0)


def residual_vol(returns: pd.DataFrame, horizon: int = 1, window: int = 60, market: pd.Series | None = None, beta_window: int = 252, min_periods: int | None = None) -> pd.DataFrame:
    """The residual volatility of each asset over ``horizon`` periods (trailing, as of each date): the rolling standard deviation of its residual return times ``sqrt(horizon)``."""
    resid = residual_returns(returns, market, beta_window)
    return resid.rolling(window, min_periods=min_periods or max(window // 2, 10)).std(ddof=1) * np.sqrt(max(int(horizon), 1))


def forward_residual(returns: pd.DataFrame, horizon: int = 1, market: pd.Series | None = None) -> pd.DataFrame:
    """The residual return over the ``horizon`` periods AFTER each date, the quantity an alpha forecasts (NaN where the window is incomplete)."""
    resid = residual_returns(returns, market)
    growth = (1.0 + resid.fillna(0.0)).cumprod()
    return growth.shift(-int(horizon)) / growth - 1.0


def standardise(signal: pd.DataFrame, investable: pd.DataFrame | None = None, winsor: float = 0.02) -> pd.DataFrame:
    """Winsorise across assets and z-score across assets on every date."""
    work = signal.where(investable.reindex_like(signal).fillna(False)) if investable is not None else signal
    return cross_sectional_zscore(winsorize(work, winsor))


@dataclass
class AlphaForecast:
    alpha: pd.DataFrame        # expected residual return over the horizon, dates x assets
    z: pd.DataFrame            # the standardised signal
    omega: pd.DataFrame        # residual volatility over the horizon
    ic_hat: pd.Series          # the IC used on each date (matured, shrunk)
    ic: pd.Series              # the realised IC of each date (usable only once matured)
    horizon: int


def estimate_ic(ic: pd.Series, horizon: int, window: int = 252, min_obs: int = 30, shrink: bool = True) -> pd.Series:
    """The IC believed at each date: the mean of the matured ICs in the last ``window``, shrunk by ``mean^2 / (mean^2 + se^2)`` (``shrink=False`` leaves it alone); NaN until ``min_obs`` have matured.
    The standard error ignores the overlap of forward windows, so with a long horizon it is too small and the shrinkage too weak: use a longer window or ``shrink`` toward zero by hand."""
    known = known_at(ic, horizon)
    r = known.rolling(window, min_periods=min_obs)
    n, mean, std = r.count(), r.mean(), r.std(ddof=1)
    if shrink:
        se2 = std ** 2 / n
        mean = mean * mean ** 2 / (mean ** 2 + se2)
    return mean.where(n >= min_obs)


def calibrate(signal: pd.DataFrame, returns: pd.DataFrame, horizon: int = 1, window: int = 252, min_obs: int = 30, shrink: bool = True, vol_window: int = 60,
              market: pd.Series | None = None, investable: pd.DataFrame | None = None, method: str = "spearman", min_assets: int = 5) -> AlphaForecast:
    """Walk-forward calibration of ``signal`` into expected residual returns. Nothing at date ``t`` uses a return after ``t``: the IC is the matured one and the volatility trailing."""
    z = standardise(signal, investable)
    fwd = forward_residual(returns, horizon, market)
    ic = ic_series(z, fwd.reindex(z.index), method, min_assets)
    ic_hat = estimate_ic(ic, horizon, window, min_obs, shrink).reindex(z.index)
    omega = residual_vol(returns, horizon, vol_window, market).reindex(z.index)
    alpha = z.mul(ic_hat, axis=0) * omega
    return AlphaForecast(alpha=alpha, z=z, omega=omega, ic_hat=ic_hat, ic=ic.reindex(z.index), horizon=int(horizon))


def reliability(alpha: pd.DataFrame, realised: pd.DataFrame, bins: int = 5, min_assets: int = 8) -> dict:
    """How well the alpha is calibrated against the realised residual return it forecast.

    ``slope``: the mean over dates of the cross-sectional regression slope of realised on alpha (1 = calibrated), with its standard error and t-statistic against 1. ``bins``: the pooled mean alpha and
    mean realised return in quantile bins of the alpha, which should lie on the diagonal."""
    a, r = alpha.align(realised, join="inner")
    ok = a.notna() & r.notna()
    slopes = []
    for t in a.index[ok.sum(axis=1) >= min_assets]:
        m = ok.loc[t].to_numpy()
        x, y = a.loc[t].to_numpy()[m], r.loc[t].to_numpy()[m]
        x, y = x - x.mean(), y - y.mean()
        d = float(x @ x)
        if d > 1e-18:
            slopes.append(float(x @ y) / d)
    s = np.asarray(slopes)
    if len(s) < 3:
        return {"slope": float("nan"), "se": float("nan"), "t_vs_one": float("nan"), "n": int(len(s)), "bins": pd.DataFrame(), "verdict": "not enough dates"}
    mean, se = float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s)))
    pooled = pd.DataFrame({"alpha": a.where(ok).stack(), "realised": r.where(ok).stack()}).dropna()
    table = pd.DataFrame()
    if len(pooled) >= bins * 10:
        pooled["bin"] = pd.qcut(pooled["alpha"], bins, labels=False, duplicates="drop")
        table = pooled.groupby("bin").agg(alpha=("alpha", "mean"), realised=("realised", "mean"), n=("alpha", "size"))
    t1 = (mean - 1.0) / se if se > 0 else float("nan")
    verdict = "calibrated" if abs(t1) < 2 and mean > 0 else ("overconfident: alphas are too large" if mean < 1 and mean > 0 else ("too timid: alphas are too small" if mean > 1 else "no relation: the signal does not forecast"))
    return {"slope": mean, "se": se, "t_vs_one": t1, "n": int(len(s)), "bins": table, "verdict": verdict}
