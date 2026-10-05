"""Alpha-combination engine (Generation 3, Priority 13).

Every alpha is a daily cross-sectional forecast. The engine standardises them
onto one scale, measures what each is worth (information coefficient, its
trailing information ratio, the standalone net Sharpe of the book it would
trade, its turnover and decay), decides how much to trust each, and combines
the FORECASTS, not the books. Combining forecasts matters: a momentum long and a
reversion short in the same asset net to a smaller trade before any cost is
charged, which stream-level blending (Stage 10) cannot do.

Everything here is causal. An IC computed for date ``t`` uses the return over
``t+1 .. t+h``, so it is only KNOWN at ``t+h``; trailing statistics use the IC
series shifted by ``h``. Trailing net Sharpe uses returns through the date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.regression import cross_sectional_ic
from .transform import clip_signal, cross_sectional_demean, cross_sectional_zscore, winsorize

ANN = 252.0


def standardise_alpha(signal: pd.DataFrame, investable: pd.DataFrame | None = None, winsorize_quantile: float = 0.02,
                      clip: float = 3.0) -> pd.DataFrame:
    """Winsorise, demean across assets, z-score across assets, clip: the Generation 1 stack without sizing."""
    work = signal.where(investable.reindex_like(signal).fillna(False)) if investable is not None else signal.copy()
    work = cross_sectional_zscore(cross_sectional_demean(winsorize(work, winsorize_quantile)))
    return clip_signal(work, clip)


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compounded return over the ``horizon`` days AFTER each date (NaN where the window is incomplete)."""
    growth = (1.0 + returns.fillna(0.0)).cumprod()
    return growth.shift(-horizon) / growth - 1.0


def matured_ic(signal: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> pd.Series:
    """Daily rank IC, shifted so the value on date ``t`` is the IC that is KNOWN on ``t`` (computed for ``t - horizon``)."""
    ic = cross_sectional_ic(signal, fwd).reindex(signal.index)
    return ic.shift(horizon)


def trailing_ir(series: pd.Series, window: int = 504, min_obs: int = 126) -> pd.Series:
    """Trailing mean over trailing standard deviation (not annualised)."""
    mean = series.rolling(window, min_periods=min_obs).mean()
    std = series.rolling(window, min_periods=min_obs).std(ddof=1)
    return mean / std.replace(0.0, np.nan)


def trailing_sharpe(returns: pd.Series, window: int = 504, min_obs: int = 252) -> pd.Series:
    mean = returns.rolling(window, min_periods=min_obs).mean()
    std = returns.rolling(window, min_periods=min_obs).std(ddof=1)
    return ANN ** 0.5 * mean / std.replace(0.0, np.nan)


def trust_weights(scores: pd.DataFrame, update_dates: pd.DatetimeIndex, shrink: float = 0.0) -> pd.DataFrame:
    """Weights proportional to the positive part of each alpha's score, sampled on ``update_dates``, forward-filled.

    ``shrink`` blends toward equal weights. If no alpha has a positive score on an update date, or the scores do not
    exist yet, the weights are equal: this is a combination rule, not a market-timing switch.
    """
    n = scores.shape[1]
    equal = np.full(n, 1.0 / n)
    rows = {}
    for date in update_dates:
        if date not in scores.index:
            continue
        s = scores.loc[date].to_numpy(dtype=float)
        positive = np.where(np.isfinite(s), np.clip(s, 0.0, None), 0.0)
        raw = positive / positive.sum() if positive.sum() > 0 else equal
        rows[date] = (1.0 - shrink) * raw + shrink * equal
    table = pd.DataFrame.from_dict(rows, orient="index", columns=scores.columns).reindex(scores.index).ffill()
    return table.fillna(1.0 / n)


def combine_alphas(standardised: dict[str, pd.DataFrame], weights: pd.DataFrame) -> pd.DataFrame:
    """Weighted sum of the standardised forecasts; weights are renormalised over the alphas present for each asset-date."""
    names = list(standardised)
    num = None
    den = None
    for name in names:
        z = standardised[name]
        w = weights[name].reindex(z.index).to_numpy()[:, None] * np.ones((1, z.shape[1]))
        present = z.notna().to_numpy()
        term = np.where(present, z.to_numpy() * w, 0.0)
        mass = np.where(present, w, 0.0)
        num = term if num is None else num + term
        den = mass if den is None else den + mass
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    first = standardised[names[0]]
    return pd.DataFrame(out, index=first.index, columns=first.columns)


def ic_decay(signal: pd.DataFrame, returns: pd.DataFrame, horizons=(1, 5, 10, 21, 63)) -> pd.Series:
    """Mean rank IC against the return over each horizon after the signal date."""
    return pd.Series({h: float(cross_sectional_ic(signal, forward_returns(returns, h)).mean()) for h in horizons}, name="mean_ic")


def annual_turnover(weights: pd.DataFrame) -> float:
    """Mean annual one-way turnover of a daily weight book."""
    return float(weights.diff().abs().sum(axis=1).mean() * ANN)
