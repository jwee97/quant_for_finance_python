"""A ``PipelineResult`` as plain JSON for the charts: series, monthly grids, tables, flags and a few headline statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANN = 252.0


def clean(value):
    """Recursively turn numpy / pandas scalars into JSON types; NaN and infinity become null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    if isinstance(value, pd.Interval):
        return f"{value.left:.2f} to {value.right:.2f}"
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def _series(values: pd.Series, digits: int = 5) -> list:
    return [None if not math.isfinite(v) else round(float(v), digits) for v in values.to_numpy(dtype=float)]


def records(table, limit: int = 60) -> list[dict]:
    """A table (frame or series) as a list of row dictionaries, index first."""
    frame = table if isinstance(table, pd.DataFrame) else table.to_frame()
    frame = frame.head(limit).reset_index(drop=isinstance(frame.index, pd.RangeIndex))
    frame.columns = [str(c) for c in frame.columns]
    return clean(frame.to_dict(orient="records"))


def _monthly(net: pd.Series) -> list[dict]:
    grouped = (1.0 + net).groupby([net.index.year, net.index.month]).prod() - 1.0
    return [{"year": int(y), "month": int(m), "value": round(float(v), 5)} for (y, m), v in grouped.items()]


def _annual(net: pd.Series) -> list[dict]:
    out = []
    for year, r in net.groupby(net.index.year):
        out.append({"year": int(year), "value": round(float((1.0 + r).prod() - 1.0), 5), "days": int(len(r)), "partial": bool(len(r) < 240)})
    return out


def _headline(net: pd.Series, gross: pd.Series) -> dict:
    """Headline numbers over the traded window (the same arithmetic the page uses for its tiles)."""
    sd = net.std(ddof=1)
    equity = (1.0 + net).cumprod()
    gsd = gross.std(ddof=1)
    return {"sharpe": ANN ** 0.5 * net.mean() / sd if sd > 0 else float("nan"), "gross_sharpe": ANN ** 0.5 * gross.mean() / gsd if gsd > 0 else float("nan"),
            "cagr": float(equity.iloc[-1] ** (ANN / len(net)) - 1.0), "ann_vol": float(sd * ANN ** 0.5), "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "n_days": int(len(net)), "start": str(net.index[0].date())}


def _stats(net: pd.Series) -> dict:
    daily = net.dropna()
    monthly = (1.0 + daily).groupby([daily.index.year, daily.index.month]).prod() - 1.0
    return {"hit_rate_daily": float((daily > 0).mean()), "hit_rate_monthly": float((monthly > 0).mean()), "best_day": float(daily.max()), "worst_day": float(daily.min()),
            "best_month": float(monthly.max()), "worst_month": float(monthly.min()), "var_95_daily": float(-daily.quantile(0.05)), "skew": float(daily.skew()),
            "kurtosis": float(daily.kurt()), "years": float(len(daily) / ANN)}


def result_payload(result, bundle, flags: list[dict], universe: dict, spec: dict, yaml_text: str, seconds: float) -> dict:
    net = result.net_returns.loc[result.start:].dropna()
    active = net[net.abs() > 0]
    if len(active):                                   # a book that waits for enough assets earns exactly zero until it starts: do not chart or score the wait
        net = net.loc[active.index[0]:]
    start = net.index[0]
    gross = result.gross_returns.reindex(net.index).fillna(0.0)
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    rolling = net.rolling(int(ANN)).apply(lambda x: ANN ** 0.5 * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan, raw=True)
    benches = {}
    for name, r in result.benchmarks.items():
        b = r.reindex(net.index)
        if b.notna().sum() > 20:
            benches[name] = _series((1.0 + b.fillna(0.0)).cumprod(), 4)
    weights = result.weights.loc[start:]
    held = weights.shift(1).reindex(net.index)
    gross_exposure = weights.abs().sum(axis=1).resample("W").last().dropna()
    net_exposure = weights.sum(axis=1).reindex(gross_exposure.index)
    average = pd.DataFrame({"average_weight": weights.mean(), "average_abs_weight": weights.abs().mean(), "latest_weight": weights.iloc[-1]})
    tables = {}
    for key, table in result.tables.items():
        if isinstance(table, (pd.DataFrame, pd.Series)) and len(table):
            tables[key] = records(table)
    return clean({
        "name": result.spec.name, "universe": universe, "spec": spec, "yaml": yaml_text, "seconds": round(seconds, 1),
        "metrics": {**{k: v for k, v in result.metrics.items() if not isinstance(v, str) or k in ("start", "end", "capacity_note")}, **_headline(net, gross)},
        "validation": {"deflated_sharpe_probability": result.validation.get("deflated_sharpe_probability"), "n_trials": result.validation.get("n_trials"),
                       "causality": result.validation.get("causality")},
        "flags": flags, "stats": _stats(net),
        "dates": [str(d.date()) for d in net.index], "equity": _series(equity, 4), "benchmarks": benches, "drawdown": _series(drawdown, 4), "rolling_sharpe": _series(rolling, 3),
        "exposure": {"dates": [str(d.date()) for d in gross_exposure.index], "gross": _series(gross_exposure, 3), "net": _series(net_exposure, 3)},
        "monthly": _monthly(net), "annual": _annual(net),
        "weights": records(average.sort_values("average_abs_weight", ascending=False), 40),
        "contribution": records(result.tables["attribution_by_asset"], 40) if "attribution_by_asset" in result.tables else [],
        "held_days": int((held.abs().sum(axis=1) > 0).sum()),
        "tables": tables,
    })
