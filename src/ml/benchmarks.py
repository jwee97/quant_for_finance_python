"""Matched benchmarks and causal exposures (spec v3, Release 5, P1-18).

A strategy that is long the market in a rising market has not shown skill. The comparators here are matched to what the strategy actually held, from information known *before* each day:

* ``beta_matched``  - the market return scaled by the strategy's beta, estimated on the window that ends the day before (so the hedge ratio is a trading decision that could have been made);
* ``net_exposure_matched`` - the market return scaled by the strategy's net exposure the day before (a dollar-neutral book is matched by cash);
* ``sector_matched`` - the strategy's net weight in each group applied to that group's equal-weight return (it asks whether the bets were within or across groups);
* ``risk_matched``  - the market scaled to the strategy's trailing volatility;
* ``descriptive_equal_weight`` - the plain daily-rebalanced equal-weight universe, kept as a *descriptive* comparison and labelled as unmatched.

``incremental`` then reports the active return and its uncertainty (HAC), the regression of the strategy on the benchmark (intercept and beta, HAC) and the paired Sharpe difference, so that the economic
effect and the statistical evidence are shown together. Factors are whatever the caller supplies; the platform has the equal-weight market and no other traded factor, and the report says so.

All exposures follow the engine's convention: the book held at the close of ``t`` earns the return of ``t + 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats.regression import ols
from . import inference as inf

VERSION = "benchmarks-v1"


def equal_weight(returns: pd.DataFrame, investable: pd.DataFrame | None = None) -> pd.Series:
    """The daily-rebalanced equal-weight return of the assets investable that day (descriptive; it is not matched to anything)."""
    r = returns.where(investable.reindex_like(returns).fillna(False)) if investable is not None else returns
    return r.mean(axis=1, skipna=True).fillna(0.0)


def book_exposures(held: pd.DataFrame) -> pd.DataFrame:
    """Net, gross, long and short exposure of the book in force at each close."""
    h = held.fillna(0.0)
    return pd.DataFrame({"net": h.sum(axis=1), "gross": h.abs().sum(axis=1), "long": h.clip(lower=0.0).sum(axis=1), "short": (-h.clip(upper=0.0)).sum(axis=1)})


def causal_beta(strategy: pd.Series, factor: pd.Series, window: int = 252, min_obs: int = 60) -> pd.Series:
    """The slope of ``strategy`` on ``factor`` estimated from the ``window`` observations that end the day BEFORE each date (``NaN`` until ``min_obs`` exist). Changing any return from day ``t`` on cannot change
    the beta used on day ``t`` or earlier."""
    both = pd.concat([strategy, factor], axis=1, keys=["y", "x"]).dropna()
    y, x = both["y"], both["x"]
    mx, my = x.rolling(window, min_periods=min_obs).mean(), y.rolling(window, min_periods=min_obs).mean()
    cov = (x * y).rolling(window, min_periods=min_obs).mean() - mx * my
    var = (x * x).rolling(window, min_periods=min_obs).mean() - mx ** 2
    beta = (cov / var.where(var > 1e-18)).shift(1)
    return beta.reindex(strategy.index)


def beta_matched(strategy: pd.Series, market: pd.Series, window: int = 252, min_obs: int = 60) -> tuple[pd.Series, pd.Series]:
    """``(benchmark, beta)``: ``benchmark_t = beta_t * market_t`` with ``beta_t`` known before ``t``. Days without an estimate have no benchmark (they are not filled)."""
    beta = causal_beta(strategy, market, window, min_obs)
    return (beta * market.reindex(strategy.index)).rename("beta_matched"), beta.rename("beta")


def net_exposure_matched(held: pd.DataFrame, market: pd.Series) -> pd.Series:
    """The market scaled by the net exposure of the book in force the day before. A dollar-neutral long-short book has net zero, so its matched benchmark is cash."""
    net = book_exposures(held)["net"].shift(1)
    return (net * market.reindex(net.index)).rename("net_exposure_matched")


def sector_matched(held: pd.DataFrame, returns: pd.DataFrame, group: dict[str, str]) -> pd.Series:
    """The strategy's net weight in each group (the day before) times the equal-weight return of that group. Assets with no group are one group of their own, named ``unclassified``."""
    cols = list(held.columns)
    labels = pd.Series({c: group.get(c) or "unclassified" for c in cols})
    net_by_group = held.fillna(0.0).T.groupby(labels).sum().T.shift(1)
    rets = returns.reindex(index=held.index, columns=cols)
    group_ret = rets.T.groupby(labels).mean().T
    return (net_by_group * group_ret.reindex(columns=net_by_group.columns)).sum(axis=1, min_count=1).rename("sector_matched")


def risk_matched(strategy: pd.Series, market: pd.Series, window: int = 63, min_obs: int = 30) -> pd.Series:
    """The market scaled so that its volatility equals the strategy's, both measured on the window ending the day before."""
    sv = strategy.rolling(window, min_periods=min_obs).std(ddof=1).shift(1)
    mv = market.rolling(window, min_periods=min_obs).std(ddof=1).shift(1)
    return ((sv / mv.where(mv > 0)) * market.reindex(strategy.index)).rename("risk_matched")


def turnover_cost(weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """The cost of trading a benchmark's weights at the strategy's one-way cost: ``cost_bps`` per unit of traded weight, on the same convention as the vectorized engine."""
    traded = weights.diff()
    traded.iloc[0] = weights.iloc[0]                                                                    # the first day builds the book from nothing
    return (traded.abs().sum(axis=1) * cost_bps / 1e4).rename("benchmark_cost")


def incremental(strategy: pd.Series, benchmark: pd.Series, *, kind: str, frequency: str = "daily", label: str = "", horizon: int = 1, lags: int | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """The strategy against one benchmark on their common days: the active return (``strategy - benchmark``) with a HAC standard error, the HAC regression ``strategy = alpha + beta benchmark + e`` (alpha and
    beta with their uncertainty), and the paired annualised Sharpe difference. ``kind`` names the matching so a report cannot present an unmatched comparison as a matched one."""
    both = pd.concat([strategy, benchmark], axis=1, join="inner", keys=["s", "b"]).replace([np.inf, -np.inf], np.nan).dropna()
    active = both["s"] - both["b"]
    out: dict = {"kind": kind, "matched": kind != "descriptive_equal_weight", "n_aligned": int(len(both)), "dropped": {"strategy": int(strategy.dropna().shape[0] - len(both)),
                                                                                                                            "benchmark": int(benchmark.dropna().shape[0] - len(both))}}
    out["active_mean"] = inf.hac_mean(active, frequency=frequency, label=f"{label} active vs {kind}", horizon=horizon, lags=lags).as_dict()
    out["sharpe_difference"] = inf.paired_difference(both["s"], both["b"], statistic="sharpe", frequency=frequency, label=f"{label} vs {kind}", n_boot=n_boot, seed=seed, horizon=horizon, lags=lags).as_dict()
    if len(both) >= inf.MIN_OBS and both["b"].std(ddof=1) > 0:
        res = ols(both["s"].to_numpy(), both["b"].to_numpy().reshape(-1, 1), cov="HAC", lags=inf.overlap_lags(len(both), horizon, lags))
        names = list(res.params.index)
        out["regression"] = {"estimator": "OLS with Newey-West (Bartlett) standard errors", "lags": inf.overlap_lags(len(both), horizon, lags),
                             "alpha": {"value": float(res.params.iloc[0]), "se": float(res.bse.iloc[0]), "p_value": float(res.pvalues.iloc[0])},
                             "beta": {"value": float(res.params.iloc[1]), "se": float(res.bse.iloc[1]), "p_value": float(res.pvalues.iloc[1])}, "terms": names,
                             "note": "alpha is per period; beta is the exposure of the strategy to the benchmark over the whole sample (the time-varying beta used to MATCH the benchmark is the causal one)"}
    else:
        out["regression"] = {"eligibility": "insufficient", "reason": f"{len(both)} common observations or a constant benchmark"}
    return out
