"""Neighbourhood robustness, delay decay and capacity decay of a saved experiment (spec v3, Release 5, P1-20 and P1-21).

Every function re-runs the experiment's SAVED predictions (``experiments.rebacktest``): the model is not refitted, so each point is a portfolio, signal, delay or execution variation of the same
forecasts. Each variation is a child experiment with a new id and an operation of the study (a counted look, so it appears in the lineage and in the selection tests' family), and the metric read is the
one declared on the segment declared: by default the validation segment, so a development-mode parent never has its protected segments read. Nothing here picks the best point:

* ``neighborhood`` evaluates the whole predeclared grid around the baseline and reports the distribution (minimum, quartiles, maximum, the share of points that are positive, the baseline's rank), not
  the best cell. The grid and its hash are fixed before the first run.
* ``delay_decay`` applies extra sessions between decision and fill on the real timeline (``backtest.extra_lag``): the vectorized engine delays the weights, never the realised returns, so no return is
  moved into an earlier decision, and costs follow the changed turnover. The decay is the ratio to the undelayed result and is undefined (not 0, not 1) when that result is not positive.
* ``capacity_decay`` re-runs the realistic execution mode at a predeclared list of capital levels; it needs volume, and says so when it has none.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from . import experiments as xp

VERSION = "sensitivity-v1"
MAX_POINTS = 64
SECTIONS = ("signal", "backtest", "execution")
METRICS = {"sharpe": "sharpe", "ann_return": "ann_return", "max_drawdown": "max_drawdown", "total_return": "total_return"}


def _known(section: str) -> set:
    from . import execution as xe
    from .config import DEFAULTS

    return set(xe.SETTING_KEYS) if section == "execution" else set(DEFAULTS.get(section, {}))


def _overrides(point: dict) -> dict:
    out: dict = {}
    for path, value in point.items():
        section, _, key = path.partition(".")
        if section not in SECTIONS or not key:
            raise ValueError(f"'{path}' is not a re-backtest setting: use signal.*, backtest.* or execution.* (a model or dataset setting would need a refit, which this does not do)")
        if key not in _known(section):
            raise ValueError(f"'{path}' is not a setting of the {section} section (its settings are {sorted(_known(section))}): a grid over a setting that does nothing would report a flat, flattering neighbourhood")
        out.setdefault(section, {})[key] = value
    return out


def grid_points(grid: dict[str, list]) -> list[dict]:
    """The Cartesian product of a predeclared grid, in a fixed order. Refused when empty or larger than ``MAX_POINTS`` (a bigger grid is a search, and is run as one)."""
    if not grid or any(not isinstance(v, (list, tuple)) or not v for v in grid.values()):
        raise ValueError("the grid is a mapping from a setting to a non-empty list of values")
    keys = sorted(grid)
    pts = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]
    if len(pts) > MAX_POINTS:
        raise ValueError(f"{len(pts)} points: more than {MAX_POINTS}. A grid this large is a search; declare a smaller neighbourhood or run it as a search")
    return pts


def grid_hash(grid: dict) -> str:
    return hashlib.sha256(json.dumps({k: list(v) for k, v in sorted(grid.items())}, default=str, sort_keys=True).encode()).hexdigest()[:16]


def _metric(exp, segment: str, metric: str):
    stats = exp.report["segments"]["stats"].get(segment)
    if not stats or stats.get(METRICS[metric]) is None:
        return None
    v = float(stats[METRICS[metric]])
    return v if np.isfinite(v) else None


def _quantiles(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"min": float(a.min()), "p25": float(np.quantile(a, 0.25)), "median": float(np.median(a)), "p75": float(np.quantile(a, 0.75)), "max": float(a.max()), "std": float(a.std(ddof=1)) if len(a) > 1 else float("nan")}


def neighborhood(registry, exp_id: str, grid: dict[str, list], *, segment: str = "validation", metric: str = "sharpe", save: bool = False, **market) -> dict:
    """Run every point of ``grid`` (settings of the signal, backtest and execution sections) on the saved predictions of ``exp_id`` and describe the distribution of ``metric`` on ``segment``. The baseline
    is the experiment itself. ``market`` passes ``bundle``/``config``/``builder``/``market`` to the re-run when an execution setting needs volume."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    pts = grid_points(grid)
    for pt in pts:
        _overrides(pt)                                                                                  # a setting that is not a re-backtest setting refuses the whole grid before anything runs
    base = registry.report(exp_id)["segments"]["stats"].get(segment)
    base_v = float(base[METRICS[metric]]) if base and base.get(METRICS[metric]) is not None else None
    rows = []
    for i, pt in enumerate(pts):
        try:
            child = xp.rebacktest(registry, exp_id, {**_overrides(pt), "name": f"neighbour-{i:02d}"}, save=save, **market)
        except (xp.ExperimentError, ValueError) as exc:
            rows.append({"point": pt, "value": None, "experiment_id": None, "state": "failed", "reason": str(exc)})
            continue
        v = _metric(child, segment, metric)
        rows.append({"point": pt, "value": v, "experiment_id": child.id if save else None, "state": "ok" if v is not None else "undefined", "reason": "" if v is not None else f"no {metric} on {segment}"})
    vals = [r["value"] for r in rows if r["value"] is not None]
    out = {"version": VERSION, "kind": "neighborhood", "segment": segment, "metric": metric, "grid": {k: list(v) for k, v in grid.items()}, "grid_hash": grid_hash(grid), "declared_points": len(pts),
           "evaluated": len(vals), "failed": sum(1 for r in rows if r["state"] == "failed"), "baseline": base_v, "points": rows,
           "notes": ["the whole predeclared grid is reported; the best cell is not selected", f"every point is a re-run of the saved predictions on {segment}: the model was not refitted and no protected segment was read by this analysis"]}
    if len(vals) < 3:
        out.update(eligibility="insufficient", reason=f"{len(vals)} points evaluated: a distribution needs at least 3", summary=None)
        return out
    out["eligibility"] = "applied"
    if len({round(v, 12) for v in vals}) == 1:
        out["notes"].append("every point gave the same value: the settings varied may not affect this result (a flat neighbourhood is not evidence of robustness)")
    out["summary"] = {**_quantiles(vals), "distinct_values": len({round(v, 12) for v in vals}), "share_positive": float(np.mean(np.asarray(vals) > 0)),
                      "share_at_least_baseline": None if base_v is None else float(np.mean(np.asarray(vals) >= base_v)),
                      "baseline_rank_from_top": None if base_v is None else int(1 + sum(v > base_v for v in vals)), "baseline_minus_median": None if base_v is None else float(base_v - np.median(vals))}
    return out


def delay_decay(registry, exp_id: str, delays=(0, 1, 2, 5), *, segment: str = "validation", metric: str = "sharpe", save: bool = False, **market) -> dict:
    """The metric after ``d`` extra sessions between decision and fill, for each ``d``, and its ratio to ``d = 0``. The extra delay is added to the experiment's own lag on its own timeline."""
    rows = []
    cur_extra = int(registry.report(exp_id)["config"]["backtest"].get("extra_lag", 0))
    for d in delays:
        child = xp.rebacktest(registry, exp_id, {"backtest": {"extra_lag": cur_extra + int(d)}, "name": f"delay-{int(d)}"}, save=save, **market)
        stats = child.report["segments"]["stats"].get(segment) or {}
        rows.append({"extra_sessions": int(d), "value": _metric(child, segment, metric), "ann_turnover": child.report["metrics"]["oos"].get("ann_turnover"), "ann_cost_drag": child.report["metrics"]["oos"].get("ann_cost_drag"),
                     "n_days": stats.get("n_days"), "experiment_id": child.id if save else None})
    base = next((r["value"] for r in rows if r["extra_sessions"] == 0), None)
    for r in rows:
        if base is None or not base > 0 or r["value"] is None:
            r["ratio_to_undelayed"] = None
        else:
            r["ratio_to_undelayed"] = float(r["value"] / base)
    why = "" if base is not None and base > 0 else "the undelayed result is missing or not positive, so a ratio to it is undefined (not zero, not one)"
    return {"version": VERSION, "kind": "delay_decay", "segment": segment, "metric": metric, "rows": rows, "eligibility": "applied" if not why else "insufficient", "reason": why,
            "notes": ["the delay is applied to the decision on the real timeline; realised returns are never shifted into an earlier decision", "costs are recomputed from the turnover of the delayed book"]}


def capacity_decay(registry, exp_id: str, capitals, *, segment: str = "validation", metric: str = "sharpe", save: bool = False, execution: dict | None = None, **market) -> dict:
    """The realistic-mode result at each capital level. Needs the realistic execution mode's market data (volume): without it every level is reported as not established."""
    rows = []
    for c in capitals:
        try:
            child = xp.rebacktest(registry, exp_id, {"execution": {**(execution or {}), "capital": float(c)}, "name": f"capacity-{c:g}"}, save=save, **market)
        except (xp.ExperimentError, ValueError) as exc:
            rows.append({"capital": float(c), "value": None, "state": "not_established", "reason": str(exc)})
            continue
        blk = child.report["execution"]
        rows.append({"capital": float(c), "value": _metric(child, segment, metric), "state": "ok", "net_return": blk["gross_net"]["net_return"], "costs_pct_of_capital": blk["gross_net"]["costs_pct_of_capital"],
                     "impact": blk["costs"]["impact"], "capacity_status": blk["capacity"]["status"], "experiment_id": child.id if save else None})
    ok = [r for r in rows if r["state"] == "ok" and r["value"] is not None]
    return {"version": VERSION, "kind": "capacity_decay", "segment": segment, "metric": metric, "rows": rows, "eligibility": "applied" if len(ok) >= 2 else "insufficient",
            "reason": "" if len(ok) >= 2 else "fewer than two capital levels could be evaluated (the realistic mode needs volume)",
            "notes": ["the realistic mode is a cash-equity, daily-bar evaluation; the impact coefficient is an assumption unless calibrated"]}
