"""Performance metrics (Ch. 22 §22.2.5).

Conventions fixed once, here, so that every table in the project is
comparable:

* Annualisation uses 252 trading days.
* CAGR is geometric, from the compounded equity curve, not ``mean * 252``.
  The gap between the two is itself informative (volatility drag).
* Sharpe ratios are excess of the configured risk-free series when one is
  supplied, and explicitly labelled "naive" when it is not.
* Sortino uses downside deviation about zero, not about the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..features.returns import cumulative_returns, drawdown, max_drawdown
from ..utils.dates import TRADING_DAYS_PER_YEAR, year_fraction

ANN = TRADING_DAYS_PER_YEAR


def cagr(returns: pd.Series, periods_per_year: int = ANN) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    curve = cumulative_returns(clean)
    years = year_fraction(clean.index)
    if not np.isfinite(years) or years <= 0:
        years = len(clean) / periods_per_year
    return float(curve.iloc[-1] ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series, periods_per_year: int = ANN) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series | float = 0.0,
                 periods_per_year: int = ANN) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    excess = clean - (risk_free.reindex(clean.index).fillna(0.0) if isinstance(risk_free, pd.Series)
                      else risk_free / periods_per_year)
    vol = excess.std(ddof=1)
    return float(excess.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else float("nan")


def sortino_ratio(returns: pd.Series, target: float = 0.0, periods_per_year: int = ANN) -> float:
    clean = returns.dropna()
    downside = clean[clean < target]
    if len(downside) < 2:
        return float("nan")
    downside_dev = np.sqrt(np.mean((downside - target) ** 2))
    return float((clean.mean() - target) / downside_dev * np.sqrt(periods_per_year)) if downside_dev > 0 else np.nan


def calmar_ratio(returns: pd.Series, periods_per_year: int = ANN) -> float:
    drop = abs(max_drawdown(cumulative_returns(returns.dropna())))
    return float(cagr(returns, periods_per_year) / drop) if drop > 1e-12 else float("nan")


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    clean = returns.dropna() - threshold
    gains, losses = clean[clean > 0].sum(), -clean[clean < 0].sum()
    return float(gains / losses) if losses > 1e-12 else float("nan")


def value_at_risk(returns: pd.Series, alpha: float = 0.95) -> float:
    """Historical VaR as a positive loss number."""
    clean = returns.dropna()
    return float(-np.quantile(clean, 1.0 - alpha)) if len(clean) else float("nan")


def conditional_value_at_risk(returns: pd.Series, alpha: float = 0.95) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    cutoff = np.quantile(clean, 1.0 - alpha)
    tail = clean[clean <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


def tail_ratio(returns: pd.Series, quantile: float = 0.05) -> float:
    clean = returns.dropna()
    lower = abs(np.quantile(clean, quantile))
    upper = abs(np.quantile(clean, 1.0 - quantile))
    return float(upper / lower) if lower > 1e-12 else float("nan")


def performance_summary(returns: pd.Series, risk_free: pd.Series | float = 0.0,
                        turnover: pd.Series | None = None, costs: pd.Series | None = None,
                        benchmark: pd.Series | None = None, periods_per_year: int = ANN,
                        label: str = "") -> dict:
    """The standard metric block used in every comparison table."""
    clean = returns.dropna()
    if clean.empty:
        return {}
    curve = cumulative_returns(clean)
    dd = drawdown(curve)
    vol = annualised_volatility(clean, periods_per_year)
    # The third and fourth moments are undefined for a constant series, and
    # scipy warns about catastrophic cancellation rather than returning NaN.
    dispersed = float(clean.std(ddof=1)) > 1e-12

    out = {
        "start": clean.index[0].date().isoformat(),
        "end": clean.index[-1].date().isoformat(),
        "n_obs": int(len(clean)),
        "cagr": cagr(clean, periods_per_year),
        "ann_return_arith": float(clean.mean() * periods_per_year),
        "ann_vol": vol,
        "sharpe": sharpe_ratio(clean, risk_free, periods_per_year),
        "sortino": sortino_ratio(clean, 0.0, periods_per_year),
        "calmar": calmar_ratio(clean, periods_per_year),
        "omega": omega_ratio(clean),
        "max_drawdown": float(dd.min()),
        "max_drawdown_date": dd.idxmin().date().isoformat(),
        "time_underwater_share": float((dd < -1e-12).mean()),
        "hit_rate": float((clean > 0).mean()),
        "best_day": float(clean.max()),
        "worst_day": float(clean.min()),
        "skew": float(stats.skew(clean, bias=False)) if dispersed else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(clean, bias=False)) if dispersed else float("nan"),
        "var_95": value_at_risk(clean, 0.95),
        "cvar_95": conditional_value_at_risk(clean, 0.95),
        "tail_ratio": tail_ratio(clean),
        "total_return": float(curve.iloc[-1] - 1.0),
    }
    if label:
        out["label"] = label
    if turnover is not None:
        aligned = turnover.reindex(clean.index).fillna(0.0)
        out["ann_turnover"] = float(aligned.mean() * periods_per_year)
    if costs is not None:
        aligned = costs.reindex(clean.index).fillna(0.0)
        out["ann_cost_drag"] = float(aligned.mean() * periods_per_year)
    if benchmark is not None:
        bench = benchmark.reindex(clean.index).dropna()
        common = clean.index.intersection(bench.index)
        if len(common) > 30:
            active = clean.loc[common] - bench.loc[common]
            tracking = active.std(ddof=1) * np.sqrt(periods_per_year)
            out["tracking_error"] = float(tracking)
            out["information_ratio"] = float(active.mean() * periods_per_year / tracking) if tracking > 0 else np.nan
            beta_denominator = float(bench.loc[common].var(ddof=1))
            beta = float(clean.loc[common].cov(bench.loc[common]) / beta_denominator) if beta_denominator > 0 else np.nan
            out["beta_to_benchmark"] = beta
            out["alpha_ann"] = float((clean.loc[common].mean() - beta * bench.loc[common].mean()) * periods_per_year)
    return out


def comparison_table(strategies: dict[str, pd.Series], risk_free: pd.Series | float = 0.0,
                     benchmark: pd.Series | None = None, periods_per_year: int = ANN) -> pd.DataFrame:
    rows = {name: performance_summary(series, risk_free, benchmark=benchmark,
                                      periods_per_year=periods_per_year)
            for name, series in strategies.items()}
    return pd.DataFrame({k: v for k, v in rows.items() if v}).T


def rolling_sharpe(returns: pd.Series, window: int = 252, periods_per_year: int = ANN) -> pd.Series:
    mean = returns.rolling(window, min_periods=max(window // 2, 40)).mean()
    vol = returns.rolling(window, min_periods=max(window // 2, 40)).std(ddof=1)
    return (mean / vol.replace(0.0, np.nan)) * np.sqrt(periods_per_year)


def rolling_volatility(returns: pd.Series, window: int = 63, periods_per_year: int = ANN) -> pd.Series:
    return returns.rolling(window, min_periods=max(window // 2, 20)).std(ddof=1) * np.sqrt(periods_per_year)


def annual_returns(returns: pd.Series) -> pd.Series:
    """Calendar-year returns: the table a reader checks first."""
    grouped = returns.dropna().groupby(returns.dropna().index.year)
    return grouped.apply(lambda block: float((1.0 + block).prod() - 1.0))


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    clean = returns.dropna()
    monthly = clean.groupby([clean.index.year, clean.index.month]).apply(
        lambda block: float((1.0 + block).prod() - 1.0)
    )
    monthly.index.names = ["year", "month"]
    return monthly.unstack("month")


def deflated_sharpe_ratio(observed_sharpe: float, n_trials: int, n_obs: int,
                          skew: float = 0.0, kurtosis: float = 3.0,
                          periods_per_year: int = ANN) -> float:
    """Probability that a Sharpe ratio survives the number of trials run.

    Bailey and Lopez de Prado. The intuition is the one Ch. 22 §22.2.4 keeps
    returning to: if you tried 72 parameter combinations, the best one is
    expected to look good even when none of them is. This converts the
    observed Sharpe ratio into the probability that it exceeds what selection
    alone would have produced.
    """
    if n_trials < 1 or n_obs < 10 or not np.isfinite(observed_sharpe):
        return float("nan")
    euler = 0.5772156649015329
    # Expected maximum Sharpe ratio from n_trials independent noise strategies.
    quantile_a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    quantile_b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max = (1.0 - euler) * quantile_a + euler * quantile_b

    sr = observed_sharpe / np.sqrt(periods_per_year)      # per-period
    threshold = expected_max / np.sqrt(n_obs)             # per-period benchmark
    variance = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2) / (n_obs - 1)
    if variance <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - threshold) / np.sqrt(variance)))


def probabilistic_sharpe_ratio(observed_sharpe: float, benchmark_sharpe: float, n_obs: int,
                               skew: float = 0.0, kurtosis: float = 3.0,
                               periods_per_year: int = ANN) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``."""
    if n_obs < 10 or not np.isfinite(observed_sharpe):
        return float("nan")
    sr = observed_sharpe / np.sqrt(periods_per_year)
    benchmark = benchmark_sharpe / np.sqrt(periods_per_year)
    variance = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2) / (n_obs - 1)
    if variance <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - benchmark) / np.sqrt(variance)))
