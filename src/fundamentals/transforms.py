"""Transformations of a factor panel (dates by securities), each one using only the cross-section of its own date or the past of its own security.

Cross-sectional: winsorise, z-score, rank, percentile, rank or de-mean within a sector, industry, country or market, fill a missing value by a stated rule, and **neutralise** (remove what the
factor shares with sector, industry, country, market, size, beta or volatility by a regression on each date). Time-series: change, percentage change and a security's own historical z-score.

A pipeline is a list of steps, each a small dictionary (``{"op": "winsorize", "pct": 1}``), so that a factor study is a piece of data that can be stored, shown, hashed and run again. The
pipeline always starts by removing what is not investable, so that a rank or a mean is taken over the companies that could have been bought. A missing value stays missing unless a ``fill`` step
says otherwise; nothing is ever turned into a zero silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..signals.transform import cross_sectional_zscore, winsorize as _winsorize
from .errors import FundamentalsError
from .research import GROUPS, ResearchContext

WINSOR_LEVELS = (0.5, 1.0, 2.5, 5.0)         # percent cut from each tail
CONTINUOUS = ("mcap", "beta", "volatility")
MIN_GROUP = 5                                 # a sector with fewer companies on a date cannot say what is high for it


# ------------------------------------------------------------------------------------------------------------------ cross-sectional
def winsorize(panel: pd.DataFrame, pct: float = 1.0) -> pd.DataFrame:
    """Clip each date to its own ``pct`` and ``100 - pct`` percentiles (pct in percent: 0.5, 1, 2.5, 5)."""
    if pct <= 0:
        return panel
    if pct >= 50:
        raise FundamentalsError("winsorising at 50% or more would leave nothing")
    return _winsorize(panel, pct / 100.0)


def zscore(panel: pd.DataFrame) -> pd.DataFrame:
    return cross_sectional_zscore(panel)


def rank(panel: pd.DataFrame, pct: bool = False) -> pd.DataFrame:
    """Rank within each date, 1 for the smallest (average rank for ties); ``pct`` gives the share of the cross-section at or below, in (0, 1]."""
    return panel.rank(axis=1, method="average", pct=pct).where(panel.notna())


def _groups(panel: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    return labels.reindex(panel.columns).fillna("unknown").astype(str).to_numpy()


def group_rank(panel: pd.DataFrame, labels: pd.Series, min_size: int = MIN_GROUP) -> pd.DataFrame:
    """Percentile rank (0-1) within each group on each date; a group with fewer than ``min_size`` companies on a date gives missing values."""
    codes = _groups(panel, labels)
    long = panel.T
    ranks = long.groupby(codes).rank(pct=True, method="average")
    size = long.notna().groupby(codes).transform("sum")
    return ranks.where(size >= min_size).T.where(panel.notna())


def group_demean(panel: pd.DataFrame, labels: pd.Series, min_size: int = MIN_GROUP) -> pd.DataFrame:
    """Subtract the group's mean on each date (a group smaller than ``min_size`` gives missing values)."""
    codes = _groups(panel, labels)
    long = panel.T
    mean = long.groupby(codes).transform("mean")
    size = long.notna().groupby(codes).transform("sum")
    return (long - mean).where(size >= min_size).T.where(panel.notna())


def group_zscore(panel: pd.DataFrame, labels: pd.Series, min_size: int = MIN_GROUP) -> pd.DataFrame:
    codes = _groups(panel, labels)
    long = panel.T
    grouped = long.groupby(codes)
    sd = grouped.transform("std").replace(0.0, np.nan)
    size = long.notna().groupby(codes).transform("sum")
    return ((long - grouped.transform("mean")) / sd).where(size >= min_size).T.where(panel.notna())


def fill(panel: pd.DataFrame, method: str, labels: pd.Series | None = None) -> pd.DataFrame:
    """The only ways a missing value gets a value, each one named in the study: ``group_median`` (the median of the company's sector, industry or country on that date) or ``neutral`` (zero, to
    be used after standardising, where zero means "no view")."""
    if method == "neutral":
        return panel.fillna(0.0)
    if method == "group_median":
        if labels is None:
            raise FundamentalsError("group_median needs a grouping")
        codes = _groups(panel, labels)
        median = panel.T.groupby(codes).transform("median").T
        return panel.fillna(median)
    raise FundamentalsError("fill method must be 'group_median' or 'neutral'")


# ------------------------------------------------------------------------------------------------------------------ time-series
def delta(panel: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    """The change over the last ``periods`` rebalance dates (a figure unchanged between filings has a change of zero until the next filing)."""
    return panel - panel.shift(periods)


def pct_change(panel: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    """The proportional change over the last ``periods`` dates; missing where the earlier value is not positive (a change from a loss is not a growth rate)."""
    base = panel.shift(periods)
    out = panel / base.where(base > 0) - 1.0
    return out.where(np.isfinite(out))


def hist_zscore(panel: pd.DataFrame, window: int = 36, min_periods: int = 12) -> pd.DataFrame:
    """Each security's value against its own past: the z-score of today's value among the previous ``window`` dates' values (today's included)."""
    mean = panel.rolling(window, min_periods=min_periods).mean()
    sd = panel.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return (panel - mean) / sd


# ------------------------------------------------------------------------------------------------------------------ neutralisation
def _controls(ctx: ResearchContext, continuous: tuple, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """The continuous controls on the dates of the panel being neutralised (looked up by date, not by position)."""
    panels = {}
    for name in continuous:
        if name == "mcap":
            cap = ctx.market_cap_usd()
            panels[name] = np.log(cap.where(cap > 0))
        elif name == "beta":
            panels[name] = ctx.beta()
        elif name == "volatility":
            panels[name] = ctx.volatility()
        else:
            raise FundamentalsError(f"unknown control {name!r}; choose from {CONTINUOUS}")
    return {k: v.reindex(index=index, columns=ctx.ids) for k, v in panels.items()}


def _dummies(ctx: ResearchContext, by: tuple) -> pd.DataFrame:
    """Category indicator columns for the groups in ``by`` (the first level of each dropped, the intercept stands for it)."""
    cols = []
    for kind in by:
        cols.append(pd.get_dummies(ctx.group(kind).reindex(ctx.ids), dtype=float).iloc[:, 1:])
    return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=ctx.ids)


def neutralize(panel: pd.DataFrame, ctx: ResearchContext, by: tuple = (), continuous: tuple = (), min_obs: int = 30) -> pd.DataFrame:
    """The part of the factor that is not explained, on each date, by the groups in ``by`` (sector, industry, country, region, market) and the continuous ``continuous`` controls (log market
    capitalisation, beta, volatility): the residual of a cross-sectional least-squares regression with an intercept. A date with fewer than ``min_obs`` complete companies gives missing values.

    With groups alone the residual is the factor less its group's mean, which is computed directly; with continuous controls the regression is solved on each date.
    """
    by, continuous = tuple(by), tuple(continuous)
    for kind in by:
        if kind not in GROUPS:
            raise FundamentalsError(f"cannot neutralise by {kind!r}; choose from {GROUPS} and, as controls, {CONTINUOUS}")
    for name in continuous:
        if name not in CONTINUOUS:
            raise FundamentalsError(f"unknown control {name!r}; choose from {CONTINUOUS}")
    if not by and not continuous:
        return panel
    if not continuous and len(by) == 1:
        return group_demean(panel, ctx.group(by[0]), 2)
    out = np.full(panel.shape, np.nan)
    aligned = panel.reindex(columns=ctx.ids)
    dummies = _dummies(ctx, by)                                           # they do not change over time: built once
    controls = _controls(ctx, continuous, panel.index)
    for t in range(len(panel)):
        design = pd.concat([dummies] + [controls[name].iloc[t].rename(name) for name in continuous], axis=1)
        y = aligned.iloc[t]
        mask = y.notna() & design.notna().all(axis=1)
        if mask.sum() < max(min_obs, design.shape[1] + 5):
            continue
        X = np.column_stack([np.ones(int(mask.sum())), design[mask].to_numpy(float)])
        for k in range(len(continuous)):                                      # standardise a continuous control so that the fit is well conditioned
            col = X[:, X.shape[1] - len(continuous) + k]
            X[:, X.shape[1] - len(continuous) + k] = (col - col.mean()) / (col.std() or 1.0)
        fit, *_ = np.linalg.lstsq(X, y[mask].to_numpy(float), rcond=None)
        row = np.full(len(ctx.ids), np.nan)
        row[np.flatnonzero(mask.to_numpy())] = y[mask].to_numpy(float) - X @ fit
        out[t] = pd.Series(row, index=ctx.ids).reindex(panel.columns).to_numpy()
    return pd.DataFrame(out, index=panel.index, columns=panel.columns)


# ------------------------------------------------------------------------------------------------------------------ pipelines
OPS = ("winsorize", "zscore", "rank", "percentile", "sector_rank", "industry_rank", "country_rank", "group_demean", "group_zscore", "delta", "pct_change", "hist_zscore", "neutralize",
       "fill", "sign", "clip")


def apply(panel: pd.DataFrame, steps: list[dict] | None, ctx: ResearchContext, investable: bool = True) -> pd.DataFrame:
    """Run a pipeline on a raw factor panel. Steps are dictionaries with an ``op`` (one of :data:`OPS`) and its parameters; the investable mask is applied first."""
    out = panel.where(ctx.investable().reindex_like(panel).fillna(False)) if investable else panel
    for step in steps or []:
        op = step.get("op")
        if op == "winsorize":
            out = winsorize(out, float(step.get("pct", 1.0)))
        elif op == "zscore":
            out = zscore(out)
        elif op == "rank":
            out = rank(out)
        elif op == "percentile":
            out = rank(out, pct=True)
        elif op in ("sector_rank", "industry_rank", "country_rank"):
            out = group_rank(out, ctx.group(op.split("_")[0]), int(step.get("min_size", MIN_GROUP)))
        elif op == "group_demean":
            out = group_demean(out, ctx.group(step["by"]), int(step.get("min_size", MIN_GROUP)))
        elif op == "group_zscore":
            out = group_zscore(out, ctx.group(step["by"]), int(step.get("min_size", MIN_GROUP)))
        elif op == "delta":
            out = delta(out, int(step.get("periods", 12)))
        elif op == "pct_change":
            out = pct_change(out, int(step.get("periods", 12)))
        elif op == "hist_zscore":
            out = hist_zscore(out, int(step.get("window", 36)), int(step.get("min_periods", 12)))
        elif op == "neutralize":
            out = neutralize(out, ctx, tuple(step.get("by", ())), tuple(step.get("continuous", ())))
        elif op == "fill":
            out = fill(out, step.get("method", "neutral"), ctx.group(step["by"]) if step.get("by") else None)
        elif op == "sign":
            out = out * (1.0 if int(step.get("direction", 1)) >= 0 else -1.0)
        elif op == "clip":
            limit = float(step.get("limit", 3.0))
            out = out.clip(-limit, limit)
        else:
            raise FundamentalsError(f"unknown transformation {op!r}; choose from {', '.join(OPS)}")
    return out.replace([np.inf, -np.inf], np.nan)


def describe(steps: list[dict] | None) -> str:
    """A sentence for the study record."""
    if not steps:
        return "raw values"
    parts = []
    for s in steps:
        op = s.get("op", "?")
        extra = {k: v for k, v in s.items() if k != "op"}
        parts.append(op + (f"({', '.join(f'{k}={v}' for k, v in extra.items())})" if extra else ""))
    return " -> ".join(parts)
