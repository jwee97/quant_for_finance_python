"""Is a price file fit to research on? The checks to run before trusting any backtest on your own data.

Errors stop a run (the pipeline cannot or should not proceed); warnings are things to look at. Nothing is changed or filled: the
platform's rule since Generation 1 is that an anomaly is flagged and adjudicated from evidence, never silently deleted.
"""

from __future__ import annotations

import pandas as pd


def check_prices(prices: pd.DataFrame, min_history: int = 500, min_assets: int = 5, extreme_return: float = 0.40, stale_run: int = 5,
                 max_gap_days: int = 7) -> dict:
    errors, warnings, per_asset = [], [], {}
    if not isinstance(prices.index, pd.DatetimeIndex):
        errors.append("the index is not a DatetimeIndex")
        return {"errors": errors, "warnings": warnings, "assets": {}, "summary": {}}
    if prices.index.has_duplicates:
        errors.append(f"{int(prices.index.duplicated().sum())} duplicate dates")
    if not prices.index.is_monotonic_increasing:
        errors.append("dates are not sorted ascending")
    if errors:                                         # a broken index makes every later check meaningless
        return {"errors": errors, "warnings": warnings, "assets": {}, "summary": {"assets": int(prices.shape[1]), "dates": int(prices.shape[0]),
                                                                                  "first": str(prices.index.min().date()), "last": str(prices.index.max().date()), "common_history_days": 0}}
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    bad = int((numeric.isna() & prices.notna()).sum().sum())
    if bad:
        errors.append(f"{bad} values are not numeric")
    if (numeric <= 0).any().any():
        errors.append("prices at or below zero: " + ", ".join(numeric.columns[(numeric <= 0).any()]))
    if prices.shape[1] < min_assets:
        errors.append(f"{prices.shape[1]} assets; the portfolio builders need at least {min_assets} (engine.min_assets)")
    gaps = pd.Series(prices.index).diff().dt.days
    long_gaps = gaps[gaps > max_gap_days]
    if len(long_gaps):
        warnings.append(f"{len(long_gaps)} calendar gap(s) longer than {max_gap_days} days, the longest {int(long_gaps.max())} days (around {prices.index[long_gaps.idxmax()].date()})")
    returns = numeric.pct_change()
    for column in numeric.columns:
        s = numeric[column].dropna()
        row = {"first": str(s.index[0].date()) if len(s) else None, "observations": int(len(s)), "missing_after_start": int(numeric[column].isna().iloc[numeric[column].index.get_loc(s.index[0]):].sum()) if len(s) else 0}
        if len(s) < min_history:
            warnings.append(f"{column}: only {len(s)} observations (< {min_history})")
        if row["missing_after_start"] > 0:
            warnings.append(f"{column}: {row['missing_after_start']} missing price(s) after its first observation (left missing, never filled)")
        r = returns[column].dropna()
        big = r[r.abs() > extreme_return]
        if len(big):
            row["extreme_returns"] = [f"{d.date()}: {v:+.0%}" for d, v in big.head(3).items()]
            warnings.append(f"{column}: {len(big)} daily return(s) beyond +-{extreme_return:.0%}, e.g. {row['extreme_returns'][0]} (split, error or real?)")
        same = (s.diff() == 0)
        runs = same.groupby((~same).cumsum()).sum()
        if len(runs) and runs.max() >= stale_run:
            warnings.append(f"{column}: a run of {int(runs.max())} identical consecutive prices (stale data?)")
            row["longest_stale_run"] = int(runs.max())
        per_asset[column] = row
    summary = {"assets": int(prices.shape[1]), "dates": int(prices.shape[0]), "first": str(prices.index[0].date()), "last": str(prices.index[-1].date()),
               "common_history_days": int(prices.dropna(how="any").shape[0])}
    if summary["common_history_days"] < min_history:
        warnings.append(f"only {summary['common_history_days']} days on which every asset has a price")
    return {"errors": errors, "warnings": warnings, "assets": per_asset, "summary": summary}


def format_report(report: dict) -> str:
    s = report["summary"]
    lines = []
    if s:
        lines.append(f"{s['assets']} assets, {s['dates']} dates, {s['first']} to {s['last']}; {s['common_history_days']} days with every asset present")
    for e in report["errors"]:
        lines.append(f"ERROR   {e}")
    for w in report["warnings"]:
        lines.append(f"WARNING {w}")
    if not report["errors"] and not report["warnings"]:
        lines.append("no problems found")
    lines.append(f"{len(report['errors'])} error(s), {len(report['warnings'])} warning(s)")
    return "\n".join(lines)
