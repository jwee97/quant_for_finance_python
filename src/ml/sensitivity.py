"""Neighbourhood robustness, delay decay and capacity decay of a saved experiment (spec v3, Release 5, P1-20 and P1-21).

Every function re-runs the experiment's SAVED predictions (``experiments.rebacktest``): the model is not refitted, so each point is a portfolio, signal, delay or execution variation of the same
forecasts. Each variation is a child experiment with a new id and an operation of the study (a counted look, so it appears in the lineage and in the selection tests' family), and the metric read is the
one declared on the segment declared: by default the validation segment, so a development-mode parent never has its protected segments read. Nothing here picks the best point:

* ``neighborhood`` evaluates the whole predeclared grid around the baseline and reports the distribution (minimum, quartiles, maximum, the share of points that are positive, the baseline's rank), not
  the best cell. The grid and its hash are fixed before the first run. A grid over ``signal.*``, ``backtest.*`` or ``execution.*`` re-runs the saved predictions; a grid over ``model.params.*`` (the model's
  own hyper-parameters) REFITS the model at every point, in development mode, on the same windows (``model_neighborhood``): each point is a new experiment and a counted look at the study, and no protected
  segment is read. The two kinds are not mixed in one grid.
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
MAX_REFIT_POINTS = 12                                       # each point of a model grid is a refit of the whole walk-forward
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
            raise ValueError(f"'{path}' is not a re-backtest setting: use signal.*, backtest.* or execution.* (the model's own hyper-parameters are model.params.NAME, in a grid of their own that refits the model; a dataset or split setting would change the panel, which a neighbourhood does not do)")
        if key not in _known(section):
            raise ValueError(f"'{path}' is not a setting of the {section} section (its settings are {sorted(_known(section))}): a grid over a setting that does nothing would report a flat, flattering neighbourhood")
        out.setdefault(section, {})[key] = value
    return out


MODEL_PREFIX = "model.params."


def is_model_grid(grid: dict) -> bool:
    """True for a grid over the model's own hyper-parameters; a grid that mixes them with re-backtest settings is refused."""
    kinds = {str(k).startswith(MODEL_PREFIX) for k in grid}
    if kinds == {True, False}:
        raise ValueError("a grid varies either the model's own hyper-parameters (model.params.*, each point refits the model) or the signal, backtest and execution settings (each point re-runs the saved predictions), not both")
    return kinds == {True}


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
    if is_model_grid(grid):
        return model_neighborhood(registry, exp_id, grid, segment=segment, metric=metric, **market)
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
    return _describe(rows, base_v, {"version": VERSION, "kind": "neighborhood", "segment": segment, "metric": metric, "grid": {k: list(v) for k, v in grid.items()}, "grid_hash": grid_hash(grid), "declared_points": len(pts),
                                    "notes": ["the whole predeclared grid is reported; the best cell is not selected", f"every point is a re-run of the saved predictions on {segment}: the model was not refitted and no protected segment was read by this analysis"]})


def _describe(rows: list[dict], base_v, out: dict) -> dict:
    """The distribution of the metric over the evaluated points (or why there is none), added to ``out``."""
    vals = [r["value"] for r in rows if r["value"] is not None]
    out.update(evaluated=len(vals), failed=sum(1 for r in rows if r["state"] == "failed"), baseline=base_v, points=rows)
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


def _model_points(registry, exp_id: str, grid: dict) -> tuple[dict, list[dict]]:
    """The experiment's saved configuration and the grid's points as model parameter sets. Every point is built (and so validated by the model's own constructor) before any is run: an unknown
    hyper-parameter, a value the model refuses or a task the model does not do refuses the whole grid."""
    from .models import make_model, task_for
    from .targets import is_classification

    cfg = registry.report(exp_id)["config"]
    kind, params = cfg["model"]["kind"], dict(cfg["model"].get("params") or {})
    pts = grid_points(grid)
    if len(pts) > MAX_REFIT_POINTS:
        raise ValueError(f"{len(pts)} points, each a refit of the model: more than {MAX_REFIT_POINTS}. Declare a smaller neighbourhood, or run the grid as a search or a tuning study")
    task = task_for(is_classification(cfg["dataset"]["target"]))
    for pt in pts:
        point = {k[len(MODEL_PREFIX):]: v for k, v in pt.items()}
        if not all(k for k in point):
            raise ValueError("a model setting is written model.params.NAME")
        make_model(kind, {**params, **point}, task, int(cfg.get("seed", 0)))
    return cfg, pts


def model_neighborhood(registry, exp_id: str, grid: dict[str, list], *, segment: str = "validation", metric: str = "sharpe", **market) -> dict:
    """Refit the experiment's model at every point of a predeclared grid of its own hyper-parameters (``model.params.NAME``) and describe the distribution of ``metric`` on ``segment``. The baseline is the
    experiment itself. Each point is a new experiment, in development mode on the same windows and the same data (the run stops after the validation segment, so a protected segment is never predicted),
    and a counted look at the experiment's study; the whole grid is reported and no point is chosen. ``market`` passes ``bundle``/``config``/``builder`` for the data, as for the other analyses."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    if segment != "validation":
        raise ValueError("a neighbourhood of model hyper-parameters refits the model in development mode, which predicts the validation segment only: the test and forward segments are not read")
    from . import research_controls as rc

    cfg, pts = _model_points(registry, exp_id, grid)
    base = registry.report(exp_id)["segments"]["stats"].get(segment)
    base_v = float(base[METRICS[metric]]) if base and base.get(METRICS[metric]) is not None else None
    sections = {k: cfg[k] for k in ("dataset", "split", "model", "signal", "backtest", "execution") if k in cfg}
    study = rc.study_of_experiment(registry, exp_id)
    rows = []
    for i, pt in enumerate(pts):
        params = {**(cfg["model"].get("params") or {}), **{k[len(MODEL_PREFIX):]: v for k, v in pt.items()}}
        try:
            exp = xp.run_experiment({**sections, "model": {"kind": cfg["model"]["kind"], "params": params}, "seed": int(cfg.get("seed", 0)), "name": f"model-neighbour-{i:02d}"}, explain=False, final_model=False,
                                    robustness=False, registry=registry, study=study, stop_after="validation", **market)
        except (xp.ExperimentError, ValueError) as exc:
            rows.append({"point": pt, "value": None, "experiment_id": None, "state": "failed", "reason": str(exc)})
            continue
        v = _metric(exp, segment, metric)
        rows.append({"point": pt, "value": v, "experiment_id": exp.id, "state": "ok" if v is not None else "undefined", "reason": "" if v is not None else f"no {metric} on {segment}"})
    return _describe(rows, base_v, {"version": VERSION, "kind": "neighborhood", "refit": True, "segment": segment, "metric": metric, "grid": {k: list(v) for k, v in grid.items()}, "grid_hash": grid_hash(grid),
                                    "declared_points": len(pts),
                                    "notes": ["the whole predeclared grid is reported; the best cell is not selected",
                                              f"every point REFITS the model in development mode on the same windows and is a new experiment and a counted look at the study; only the {segment} segment is read"]})

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
