"""The data-quality engine: flag the observations that should not be trusted blindly, without deleting or "fixing" anything.

Flags (``fundamental_observations.quality_flag``), most serious first. Research can exclude any set of them; the default excludes the serious ones.

``identity_error``       assets differ from liabilities plus equity by more than a few per cent in that period (a mapping error, a missing line, or a company that consolidates in an unusual way)
``negative_value``       a quantity that cannot be negative (revenue, assets, debt) is negative
``suspicious_jump``      a level more than eight times above or below the median of the periods around it (the signature of a unit error: thousands against units)
``provider_disagreement``  two providers report the same figure for the same period and differ by more than the tolerance
``restated``             a later vintage replaced the figure (set when the restatement is stored)
``derived``              computed by the application from other figures of the same period (gross profit = revenue - cost of revenue)
``estimated``            the provider's field is only an approximation of the canonical metric
``ok``                   nothing to say

Whether a period's *filing date* is a real date or an assumed lag is a property of the period (``financial_periods.pit_quality``), not of its figures, and the point-in-time engine reports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import canonical
from .store import FundamentalsStore

SEVERE = ("identity_error", "negative_value", "suspicious_jump", "provider_disagreement")
INFORMATIVE = ("restated", "derived", "estimated")
DEFAULT_EXCLUDE = SEVERE
IDENTITY_TOLERANCE = 0.03
JUMP_FACTOR = 8.0
JUMP_WINDOW_YEARS, JUMP_WINDOW_QUARTERS, JUMP_WINDOW_HALVES = 3, 6, 3      # periods either side that count as the neighbours
MIN_POINTS = 5
DISAGREEMENT = 0.05
JUMP_METRICS = ("revenue", "total_assets", "shareholders_equity", "shares_outstanding", "operating_cash_flow", "net_income")
# A level is a suspicious jump when it is more than this many times above or below the median of the periods around it. Earnings and operating cash flow cross zero and fall to a tenth of a good year
# without any error; a share count steps by the split ratio (NVIDIA's ten for one, Alphabet's twenty for one) when a company splits. A thousandfold unit error is far beyond all three.
JUMP_FACTORS = {"net_income": 100.0, "operating_cash_flow": 100.0, "shares_outstanding": 30.0}
SHARE_SCALE = 20.0          # diluted weighted shares and shares outstanding differing by more than this factor are counted in different units (thousands against units)


def _level_outliers(values: np.ndarray, window: int, factor: float = JUMP_FACTOR) -> np.ndarray:
    """Which points of a series lie more than ``factor`` times above or below the median of the (up to ``2 * window``) points around them, excluding themselves. At least three neighbours with a
    non-zero median are needed to judge a point."""
    n = len(values)
    out = np.zeros(n, dtype=bool)
    mag = np.abs(values)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        around = np.concatenate([mag[lo:i], mag[i + 1:hi]])
        around = around[np.isfinite(around) & (around > 0)]
        if len(around) < 3 or not np.isfinite(mag[i]) or mag[i] <= 0:
            continue
        ratio = mag[i] / float(np.median(around))
        out[i] = ratio > factor or ratio < 1.0 / factor
    return out


def check_security(store: FundamentalsStore, security_id: int) -> dict:
    """Flag the observations of one company. Returns counts per flag set in this pass."""
    obs = store.observations([security_id])
    periods = store.periods([security_id])
    if obs.empty or periods.empty:
        return {}
    data = obs.merge(periods[["period_id", "fiscal_period", "period_end", "vintage"]], on="period_id")
    updates: dict[tuple[int, str], str] = {}

    def flag(period_id: int, code: str, name: str) -> None:
        current = updates.get((period_id, code))
        if current is None or SEVERE.index(name) < SEVERE.index(current):
            updates[(period_id, code)] = name

    # a quantity that cannot be negative
    nonneg = {c for c, m in canonical.METRICS.items() if m.nonnegative}
    for row in data[data["metric_code"].isin(nonneg) & (data["value"] < 0)].itertuples():
        flag(row.period_id, row.metric_code, "negative_value")

    # assets = liabilities + equity (+ minority interest)
    wide = data.pivot_table(index="period_id", columns="metric_code", values="value", aggfunc="last")
    if {"total_assets", "total_liabilities", "shareholders_equity"} <= set(wide.columns):
        minority = wide["minority_interest"].fillna(0.0) if "minority_interest" in wide.columns else 0.0
        gap = (wide["total_assets"] - wide["total_liabilities"] - wide["shareholders_equity"] - minority).abs() / wide["total_assets"].abs().replace(0, np.nan)
        for period_id in gap.index[gap > IDENTITY_TOLERANCE]:
            for code in ("total_assets", "total_liabilities", "shareholders_equity"):
                if code in wide.columns and np.isfinite(wide.at[period_id, code]):
                    flag(int(period_id), code, "identity_error")

    # a level that sits an order of magnitude away from its neighbours (a unit error: thousands against units). Compared with the median of the periods around it, so a block of bad quarters
    # is found as a block and a company that simply grows fast is not: the same series is read separately for years, quarters and half-years, which have different scales
    first_vintage = data[data["vintage"] == 0]
    for kinds, window in ((("FY",), JUMP_WINDOW_YEARS), (("Q1", "Q2", "Q3", "Q4"), JUMP_WINDOW_QUARTERS), (("H1", "H2"), JUMP_WINDOW_HALVES)):
        group = first_vintage[first_vintage["fiscal_period"].isin(kinds)]
        for code in JUMP_METRICS:
            series = group[group["metric_code"] == code].sort_values("period_end")
            if len(series) < MIN_POINTS:
                continue
            for k in np.flatnonzero(_level_outliers(series["value"].to_numpy(float), window, JUMP_FACTORS.get(code, JUMP_FACTOR))):
                flag(int(series["period_id"].iloc[k]), code, "suspicious_jump")

    # a share count in thousands against one in units: the diluted weighted average against the count in issue (a filer's own tagging error, found in real filings)
    shares = first_vintage[first_vintage["metric_code"].isin(("shares_outstanding", "weighted_avg_shares_diluted"))].pivot_table(index="period_id", columns="metric_code", values="value", aggfunc="last")
    if {"shares_outstanding", "weighted_avg_shares_diluted"} <= set(shares.columns):
        ratio = (shares["weighted_avg_shares_diluted"] / shares["shares_outstanding"].replace(0, np.nan)).abs()
        for period_id in ratio.index[(ratio > SHARE_SCALE) | (ratio < 1.0 / SHARE_SCALE)]:
            flag(int(period_id), "weighted_avg_shares_diluted", "suspicious_jump")

    # two providers, one period, one figure: the figures as first filed, and not the ones that are only approximations or computed by the application
    approximate = {(provider, code) for provider, codes in canonical.APPROXIMATE.items() for code in codes}
    approximate_sources = {(provider, source) for provider, sources in canonical.APPROXIMATE_SOURCES.items() for source in sources}
    pairs = list(zip(data["provider"], data["metric_code"]))
    sources = list(zip(data["provider"], data["source_metric"].astype(str)))
    comparable = data[(data["vintage"] == 0) & ~data["source_metric"].astype(str).str.startswith("derived:") & ~pd.Series([p in approximate for p in pairs], index=data.index)
                      & ~pd.Series([s in approximate_sources for s in sources], index=data.index)]
    multi = comparable[comparable.groupby(["fiscal_period", "period_end", "metric_code"])["provider"].transform("nunique") > 1]
    if not multi.empty:
        spread = multi.groupby(["fiscal_period", "period_end", "metric_code"])["value"].agg(lambda s: (s.max() - s.min()) / max(abs(s).max(), 1e-12))
        for key in spread.index[spread > DISAGREEMENT]:
            for row in multi[(multi["fiscal_period"] == key[0]) & (multi["period_end"] == key[1]) & (multi["metric_code"] == key[2])].itertuples():
                flag(row.period_id, row.metric_code, "provider_disagreement")

    # the flag each observation deserves: a serious finding, else the informative one it was stored with (so re-running the check can clear a flag that no longer applies)
    def natural(row) -> str:
        if row.is_restated:
            return "restated"
        if str(row.source_metric).startswith("derived:"):
            return "derived"
        return "estimated" if (row.provider, row.metric_code) in approximate or (row.provider, str(row.source_metric)) in approximate_sources else "ok"

    final = [updates.get((int(r.period_id), r.metric_code), natural(r)) for r in data.itertuples()]
    changed = data.assign(final=final)
    changed = changed[changed["final"] != changed["quality_flag"]]
    rows = [(r.final, int(r.period_id), r.metric_code) for r in changed.itertuples()]
    if rows:
        store.set_quality(rows)
    counts: dict[str, int] = {}
    for name in updates.values():
        counts[name] = counts.get(name, 0) + 1
    return counts


def excluded_mask(flags: pd.Series, exclude: tuple = DEFAULT_EXCLUDE) -> pd.Series:
    return flags.isin(exclude)
