"""Return construction and descriptive statistics (Ch. 8 §8.2).

All research returns in this project are **total returns** built from adjusted
prices, so that distributions are income rather than a price loss (Ch. 7
§7.5.1). Log returns are used wherever additivity matters (volatility scaling,
multi-period aggregation); simple returns are used wherever a portfolio
aggregation over assets matters, because only simple returns aggregate
linearly across a cross-section::

    R_p = sum_i w_i r_i            (simple returns, cross-sectional)
    r_(t,t+k) = sum log returns    (log returns, time-series)

Mixing the two is one of the quieter ways to produce a wrong backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.dates import TRADING_DAYS_PER_YEAR


def simple_returns(prices: pd.DataFrame | pd.Series, periods: int = 1) -> pd.DataFrame | pd.Series:
    """``P_t / P_{t-k} - 1``."""
    return prices.pct_change(periods)


def log_returns(prices: pd.DataFrame | pd.Series, periods: int = 1) -> pd.DataFrame | pd.Series:
    """``log(P_t) - log(P_{t-k})``."""
    return np.log(prices).diff(periods)


def to_simple(log_ret: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.expm1(log_ret)


def to_log(simple_ret: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log1p(simple_ret)


def forward_returns(returns: pd.DataFrame, horizon: int = 1, compound: bool = True) -> pd.DataFrame:
    """Return realised over ``(t, t+h]``, stamped at ``t``.

    This is the target variable for every IC and regression in the project.
    Stamping at ``t`` is what makes ``corr(signal_t, forward_return_t)`` an
    honest predictive statistic: the signal is known at ``t`` and the return
    is entirely in the future. The last ``h`` rows are necessarily NaN, which
    is correct -- their outcome has not happened yet.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if compound:
        growth = (1.0 + returns).rolling(horizon).apply(np.prod, raw=True) - 1.0
    else:
        growth = returns.rolling(horizon).sum()
    return growth.shift(-horizon)


def cumulative_returns(returns: pd.DataFrame | pd.Series, initial: float = 1.0) -> pd.DataFrame | pd.Series:
    """Growth of ``initial`` unit(s), treating missing returns as no position."""
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(curve: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Drawdown from the running maximum of an equity curve."""
    return curve / curve.cummax() - 1.0


def max_drawdown(curve: pd.Series) -> float:
    return float(drawdown(curve).min())


def drawdown_table(curve: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` deepest peak-to-trough episodes, with recovery dates."""
    dd = drawdown(curve)
    under = dd < -1e-12
    if not under.any():
        return pd.DataFrame(columns=["start", "trough", "end", "depth", "length_days", "recovery_days"])
    episode = (under != under.shift()).cumsum()[under]
    rows = []
    for _, block in dd[under].groupby(episode):
        trough = block.idxmin()
        start_pos = curve.index.get_loc(block.index[0])
        start = curve.index[max(start_pos - 1, 0)]
        after = dd.loc[block.index[-1]:]
        recovered = after[after >= -1e-12]
        end = recovered.index[0] if len(recovered) else pd.NaT
        rows.append(
            {
                "start": start,
                "trough": trough,
                "end": end,
                "depth": float(block.min()),
                "length_days": int((pd.Timestamp(end) - pd.Timestamp(start)).days) if end is not pd.NaT else np.nan,
                "recovery_days": int((pd.Timestamp(end) - pd.Timestamp(trough)).days) if end is not pd.NaT else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("depth").head(top).reset_index(drop=True)
    return out


def annualised_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float((1.0 + clean).prod() ** (periods_per_year / len(clean)) - 1.0)


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def describe_returns(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    """Univariate EDA table (Ch. 8 §8.2.1/§8.2.3).

    Includes the third and fourth moments and a Jarque-Bera test, because the
    normality assumption is exactly what the parametric VaR of Ch. 21 rests on:
    the table below is the evidence for preferring historical simulation.
    """
    rows = {}
    for column in returns.columns:
        series = returns[column].dropna()
        if len(series) < 30:
            continue
        curve = cumulative_returns(series)
        jb_stat, jb_p = stats.jarque_bera(series.to_numpy())
        ann_vol = annualised_volatility(series, periods_per_year)
        ann_ret = annualised_return(series, periods_per_year)
        downside = series[series < 0.0]
        rows[column] = {
            "n_obs": int(len(series)),
            "mean_ann": float(series.mean() * periods_per_year),
            "cagr": ann_ret,
            "vol_ann": ann_vol,
            "sharpe_naive": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "skew": float(stats.skew(series, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(series, bias=False)),
            "jarque_bera": float(jb_stat),
            "jb_pvalue": float(jb_p),
            "min": float(series.min()),
            "q01": float(series.quantile(0.01)),
            "median": float(series.median()),
            "q99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "downside_vol_ann": float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 2 else np.nan,
            "max_drawdown": max_drawdown(curve),
            "hit_rate": float((series > 0).mean()),
            "autocorr_lag1": float(series.autocorr(1)) if len(series) > 3 else np.nan,
        }
    return pd.DataFrame(rows).T


def autocorrelation(returns: pd.DataFrame, lags: int = 21) -> pd.DataFrame:
    """Autocorrelation by lag, per asset (Ch. 8 §8.2.3).

    Two questions in one table: is there time-series momentum/reversal in raw
    returns, and are returns independent enough for naive standard errors
    (they are not -- see ``models.regression`` and its HAC estimator).
    """
    out = {}
    for column in returns.columns:
        series = returns[column].dropna()
        out[column] = [series.autocorr(lag) for lag in range(1, lags + 1)]
    frame = pd.DataFrame(out, index=pd.RangeIndex(1, lags + 1, name="lag"))
    return frame


def ljung_box(returns: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box test for serial correlation in returns and squared returns."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = {}
    for column in returns.columns:
        series = returns[column].dropna()
        if len(series) < 50:
            continue
        raw = acorr_ljungbox(series, lags=[lags], return_df=True)
        squared = acorr_ljungbox(series ** 2, lags=[lags], return_df=True)
        rows[column] = {
            "lb_stat_returns": float(raw["lb_stat"].iloc[0]),
            "lb_pvalue_returns": float(raw["lb_pvalue"].iloc[0]),
            "lb_stat_squared": float(squared["lb_stat"].iloc[0]),
            "lb_pvalue_squared": float(squared["lb_pvalue"].iloc[0]),
        }
    return pd.DataFrame(rows).T


def rolling_correlation(returns: pd.DataFrame, reference: str, window: int = 126) -> pd.DataFrame:
    """Rolling correlation of every asset against ``reference``."""
    if reference not in returns.columns:
        raise KeyError(f"{reference} not in returns")
    base = returns[reference]
    return returns.rolling(window).corr(base).drop(columns=[reference], errors="ignore")


def average_pairwise_correlation(returns: pd.DataFrame, window: int = 126) -> pd.Series:
    """Mean off-diagonal correlation through time.

    The single most useful diversification diagnostic in a multi-asset book:
    when it spikes, the number of independent bets collapses exactly when it
    is needed most.
    """
    n = returns.shape[1]
    if n < 2:
        return pd.Series(dtype=float)
    rolling = returns.rolling(window).corr()
    values = []
    index = []
    for stamp, block in rolling.groupby(level=0):
        matrix = block.droplevel(0).to_numpy(dtype=float)
        if matrix.shape[0] != n or not np.isfinite(matrix).all():
            continue
        off = matrix[~np.eye(n, dtype=bool)]
        values.append(float(np.nanmean(off)))
        index.append(stamp)
    return pd.Series(values, index=pd.DatetimeIndex(index), name="avg_pairwise_corr")
