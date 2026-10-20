"""The data-quality engine: flag the observations that should not be trusted blindly, without deleting or "fixing" anything.

Flags (``fundamental_observations.quality_flag``), most serious first. Research can exclude any set of them; the default excludes the serious ones.

``identity_error``       assets differ from liabilities plus equity by more than a few per cent in that period (a mapping error, a missing line, or a company that consolidates in an unusual way)
``negative_value``       a quantity that cannot be negative (revenue, assets, debt) is negative
``suspicious_jump``      a level that jumps by a large factor and falls back, or by an enormous factor at once (the signature of a unit error: thousands against units)
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
EXTREME_JUMP = 30.0
DISAGREEMENT = 0.05
JUMP_METRICS = ("revenue", "total_assets", "shareholders_equity", "shares_outstanding", "operating_cash_flow", "net_income")


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

    # a level that jumps and comes back (a unit error), or jumps by an absurd factor
    annual = data[(data["fiscal_period"] == "FY") & (data["vintage"] == 0)]
    for code in JUMP_METRICS:
        series = annual[annual["metric_code"] == code].sort_values("period_end")
        if len(series) < 3:
            continue
        v = series["value"].to_numpy(float)
        ids = series["period_id"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(v[1:] / v[:-1])
        spikes: set[int] = set()
        for k in range(1, len(v)):
            up = ratio[k - 1]
            if not np.isfinite(up) or v[k - 1] == 0:
                continue
            returns = k + 1 < len(v) and np.isfinite(ratio[k]) and ratio[k] < 1.0 / JUMP_FACTOR
            if up > JUMP_FACTOR and returns:                                       # up and back down: the middle point is the error
                spikes.add(k)
                flag(int(ids[k]), code, "suspicious_jump")
            elif (k - 1) not in spikes and (up > EXTREME_JUMP or up < 1.0 / EXTREME_JUMP):      # one absurd step (the recovery after a spike is not a second error)
                flag(int(ids[k]), code, "suspicious_jump")

    # two providers, one period, one figure
    multi = data[data.groupby(["fiscal_period", "period_end", "metric_code"])["provider"].transform("nunique") > 1]
    if not multi.empty:
        spread = multi.groupby(["fiscal_period", "period_end", "metric_code"])["value"].agg(lambda s: (s.max() - s.min()) / max(abs(s).max(), 1e-12))
        for key in spread.index[spread > DISAGREEMENT]:
            for row in multi[(multi["fiscal_period"] == key[0]) & (multi["period_end"] == key[1]) & (multi["metric_code"] == key[2])].itertuples():
                flag(row.period_id, row.metric_code, "provider_disagreement")

    # the flag each observation deserves: a serious finding, else the informative one it was stored with (so re-running the check can clear a flag that no longer applies)
    approximate = {(provider, code) for provider, codes in canonical.APPROXIMATE.items() for code in codes}

    def natural(row) -> str:
        if row.is_restated:
            return "restated"
        if str(row.source_metric).startswith("derived:"):
            return "derived"
        return "estimated" if (row.provider, row.metric_code) in approximate else "ok"

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
