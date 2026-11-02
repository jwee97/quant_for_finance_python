"""Targets: what a model is asked to predict, defined so that the label never overlaps the information in the features.

The timing convention is the backtest engine's. A decision made at the close of ``t`` is traded ``lag`` days later (the engine's ``signal_lag``, 1 by default) and earns returns from the close of ``t + lag``
onwards. A target with horizon ``h`` therefore looks at the window from the close of ``t + lag`` to the close of ``t + lag + h``: it starts AFTER the features end and measures what the position decided
at ``t`` would actually earn. The row stamped ``t`` has a label only once ``t + lag + h`` has happened; the last ``lag + h`` rows of a dataset have features but no label, which is what the live
prediction uses.

Names (``h`` is the horizon in days; the default horizons are 1, 5 and 20):

``fwd_ret_h``      the compounded return over the window
``fwd_vol_h``      the annualised standard deviation of the daily returns in the window
``fwd_dd_h``       the worst fall below the starting price inside the window (zero or negative)
``excess_ret_h``   ``fwd_ret_h`` minus the benchmark's (the equal-weight universe, or a ticker)
``sector_ret_h``   ``fwd_ret_h`` minus the mean of the asset's group (sector) on that date; an asset with no group, or with no peer that has a value, has no target
``resid_ret_h``    ``fwd_ret_h`` minus ``beta`` times the benchmark's, with ``beta`` the regression slope of the asset's daily return on the benchmark's over the trailing ``BETA_WINDOW`` days *ending at the decision date*
                   (at least ``BETA_MIN`` of them), so the loading is known when the decision is made and never uses the label window
``rank_h``         the cross-sectional percentile rank (0..1) of ``fwd_ret_h`` among the assets on that date
``topbottom_h``    1 for the top quantile of ``fwd_ret_h`` on a date, 0 for the bottom quantile, missing in between (a classification target)
``direction_h``    1 if ``fwd_ret_h`` is positive, else 0
``riskadj_h``      ``fwd_ret_h`` over the asset's trailing 63-day volatility scaled to the horizon (the trailing volatility is known at ``t``)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ANN = 252.0
BETA_WINDOW = 252            # trailing daily observations that estimate a loading, ending at the decision date
BETA_MIN = 120               # fewest of them for a loading to exist
KINDS = {
    "fwd_ret": "forward return", "fwd_vol": "forward realised volatility", "fwd_dd": "forward maximum drawdown", "excess_ret": "forward return minus the benchmark's",
    "sector_ret": "forward return minus the mean of the asset's sector/group peers (needs the bundle's group map)",
    "resid_ret": "forward return minus beta times the benchmark's (beta from the trailing daily returns known at the decision)",
    "rank": "cross-sectional rank of the forward return", "topbottom": "top versus bottom quantile of the forward return (classification)", "direction": "sign of the forward return (classification)",
    "riskadj": "forward return per unit of trailing volatility",
}
CLASSIFICATION = ("topbottom", "direction")
_NAME = re.compile(r"^(?P<kind>[a-z_]+?)_(?P<h>\d+)$")


def parse_target(name: str) -> tuple[str, int]:
    """``("fwd_ret", 5)`` from ``"fwd_ret_5"``; raises ``ValueError`` for a name that is not a target."""
    m = _NAME.match(name)
    if not m or m.group("kind") not in KINDS:
        raise ValueError(f"unknown target '{name}'; the kinds are {sorted(KINDS)} followed by the horizon in days, e.g. fwd_ret_5")
    h = int(m.group("h"))
    if not 1 <= h <= 252:
        raise ValueError("the target horizon must be between 1 and 252 days")
    return m.group("kind"), h


def is_classification(name: str) -> bool:
    return parse_target(name)[0] in CLASSIFICATION


def label_end(h: int, lag: int) -> int:
    """Rows between the feature date and the last day the label uses: ``lag + h``. Training rows closer than this to a test date have labels that overlap it."""
    return int(lag) + int(h)


def forward_return(prices: pd.DataFrame, h: int, lag: int) -> pd.DataFrame:
    """``P[t + lag + h] / P[t + lag] - 1`` for each asset (missing if either price is missing)."""
    return prices.shift(-(lag + h)) / prices.shift(-lag) - 1.0


def forward_vol(returns: pd.DataFrame, h: int, lag: int) -> pd.DataFrame:
    """Annualised standard deviation of the daily returns of days ``t + lag + 1 .. t + lag + h`` (needs at least two of them)."""
    if h < 5:
        raise ValueError("a forward volatility target needs a horizon of at least 5 days (a standard deviation of fewer returns is noise)")
    return (returns.rolling(h, min_periods=h).std(ddof=1) * np.sqrt(ANN)).shift(-(lag + h))


def forward_drawdown(prices: pd.DataFrame, h: int, lag: int) -> pd.DataFrame:
    """The lowest price in ``t + lag .. t + lag + h`` over the price at ``t + lag``, minus 1."""
    low = prices.rolling(h + 1, min_periods=h + 1).min().shift(-(lag + h))
    return low / prices.shift(-lag) - 1.0


def benchmark_return(prices: pd.DataFrame, h: int, lag: int, benchmark: str = "equal_weight") -> pd.Series:
    """The benchmark's forward return: the cross-sectional mean of the assets' forward returns, or one asset's."""
    fwd = forward_return(prices, h, lag)
    if benchmark == "equal_weight":
        return fwd.mean(axis=1, skipna=True)
    if benchmark not in prices.columns:
        raise ValueError(f"the benchmark '{benchmark}' is not in the universe; use equal_weight or one of {list(prices.columns)[:8]}")
    return fwd[benchmark]


def benchmark_daily(returns: pd.DataFrame, benchmark: str = "equal_weight") -> pd.Series:
    """The benchmark's daily return: the cross-sectional mean of the assets' daily returns, or one asset's."""
    if benchmark == "equal_weight":
        return returns.mean(axis=1, skipna=True)
    if benchmark not in returns.columns:
        raise ValueError(f"the benchmark '{benchmark}' is not in the universe; use equal_weight or one of {list(returns.columns)[:8]}")
    return returns[benchmark]


def rolling_beta(returns: pd.DataFrame, bench: pd.Series, window: int = BETA_WINDOW, min_periods: int = BETA_MIN) -> pd.DataFrame:
    """The slope of each asset's daily return on the benchmark's over the last ``window`` observations *including the row's own date*: row ``t`` holds what was knowable at the close of ``t``. Missing where fewer
    than ``min_periods`` pairs exist or the benchmark did not vary."""
    cov = returns.rolling(window, min_periods=min_periods).cov(bench)
    var = bench.rolling(window, min_periods=min_periods).var()
    return cov.div(var.where(var > 0.0), axis=0)


def group_relative(fwd: pd.DataFrame, groups: dict | None) -> pd.DataFrame:
    """``fwd`` minus the mean of ``fwd`` over the asset's group on each date. An asset with no group, and any asset whose group has no other member with a value on that date, is missing: relative to nothing is not
    relative to zero."""
    if not groups:
        raise ValueError("a sector-relative target needs a group map (the bundle has none); use excess_ret or resid_ret, or load a bundle that carries sectors")
    out = pd.DataFrame(np.nan, index=fwd.index, columns=fwd.columns)
    labels = {a: groups.get(a) for a in fwd.columns}
    for g in sorted({v for v in labels.values() if v}, key=str):
        cols = [a for a in fwd.columns if labels[a] == g]
        if len(cols) < 2:
            continue
        sub = fwd[cols]
        enough = sub.notna().sum(axis=1) >= 2
        out[cols] = sub.sub(sub.mean(axis=1), axis=0).where(enough, axis=0)
    return out


def describe(name: str, groups: dict | None = None, benchmark: str = "equal_weight", lag: int = 1) -> dict:
    """What a target means, for the dataset's record (spec v3, P2-01): its unit, its factors and how their loadings are estimated, and where its classification comes from and whether that classification is
    point in time. A static group map is not point in time, and the record says so."""
    kind, h = parse_target(name)
    out = {"name": name, "kind": kind, "horizon": h, "lag": lag, "unit": "decimal return over the window" if kind in ("fwd_ret", "excess_ret", "sector_ret", "resid_ret") else KINDS[kind],
           "window": f"close of decision date + {lag} to close of decision date + {lag + h} sessions"}
    if kind in ("excess_ret", "resid_ret"):
        out["benchmark"] = benchmark
    if kind == "resid_ret":
        out.update(factors=["benchmark return (single factor)"], loading=f"slope of the asset's daily return on the benchmark's over the trailing {BETA_WINDOW} sessions ending at the decision date, at least {BETA_MIN} observations",
                   other_factors="size, value and momentum factors are not used: no point-in-time factor data is available in this platform")
    if kind == "sector_ret":
        out["classification"] = {"source": "the bundle's group map", "members": len(groups or {}), "groups": sorted({str(v) for v in (groups or {}).values() if v}),
                                 "point_in_time": False, "note": "a static map applied to every date: reclassifications and listings are not dated, so a sector-relative target inherits that look-ahead; treat it as a research convenience"}
    return out


def make_target(name: str, prices: pd.DataFrame, returns: pd.DataFrame, lag: int = 1, benchmark: str = "equal_weight", top_quantile: float = 0.3, groups: dict | None = None) -> pd.DataFrame:
    """The target ``name`` as a wide frame (date x asset). ``groups`` (asset to sector) is needed only by ``sector_ret``."""
    kind, h = parse_target(name)
    fwd = forward_return(prices, h, lag)
    if kind == "fwd_ret":
        return fwd
    if kind == "sector_ret":
        return group_relative(fwd, groups)
    if kind == "resid_ret":
        beta = rolling_beta(returns, benchmark_daily(returns, benchmark))
        return fwd - beta.mul(benchmark_return(prices, h, lag, benchmark), axis=0)
    if kind == "fwd_vol":
        return forward_vol(returns, h, lag)
    if kind == "fwd_dd":
        return forward_drawdown(prices, h, lag)
    if kind == "excess_ret":
        return fwd.sub(benchmark_return(prices, h, lag, benchmark), axis=0)
    if kind == "rank":
        return fwd.rank(axis=1, pct=True)
    if kind == "direction":
        return (fwd > 0).astype(float).where(fwd.notna())
    if kind == "topbottom":
        pct = fwd.rank(axis=1, pct=True)
        usable = fwd.notna().to_numpy() & (fwd.notna().sum(axis=1).to_numpy() >= 4)[:, None]                  # a ranking of fewer than four assets says little
        out = pd.DataFrame(np.nan, index=fwd.index, columns=fwd.columns)
        out = out.mask(pct >= 1.0 - top_quantile, 1.0).mask(pct <= top_quantile, 0.0)
        return out.where(pd.DataFrame(usable, index=fwd.index, columns=fwd.columns))
    vol = returns.rolling(63, min_periods=40).std(ddof=1)                                    # riskadj: the trailing volatility is known at t
    return fwd / (vol.replace(0.0, np.nan) * np.sqrt(h))


def make_targets(names: list[str], prices: pd.DataFrame, returns: pd.DataFrame, lag: int = 1, benchmark: str = "equal_weight", groups: dict | None = None) -> dict[str, pd.DataFrame]:
    """Several targets at once, keyed by name."""
    return {n: make_target(n, prices, returns, lag, benchmark, groups=groups) for n in names}
