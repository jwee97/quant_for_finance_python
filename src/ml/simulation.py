"""Scheduled walk-forward trading simulation (phase 7).

One call runs the whole loop a live system would run, and records it so it can be reproduced and audited:

1. *universe*: the dataset's point-in-time ``investable`` mask decides who can be held on each date (a membership table for the universe lab is separate: ``src/universelab``);
2. *features and labels*: built causally, targets that look ``lag + horizon`` days ahead are purged from every training window (plus the embargo);
3. *train*: the model is refitted on a schedule (``split.refit_every``, expanding or rolling), each fit seeing only rows whose labels had settled by its cutoff; each fit is a recorded **model version**;
4. *validate*: at each refit date the decision policy is chosen from a grid, scoring every candidate policy on the days **before** that date only (``select_schedule``): thresholds and sizing are chosen on past data;
5. *predict* the next unseen period with the model that was current, *decide*, *execute* with realistic lag and costs, and record every trade, cost, cash balance and holding;
6. *advance* to the next refit and repeat.

The model versions, the policy chosen at each refit with the scores it was chosen on, and the configuration hashes are saved with the experiment (``report['simulation']``), so a run is reproducible and its
choices are open to inspection. The comparison at the end is honest about what selecting the policy bought: the schedule's result against each fixed policy over the same days.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import experiments as ex
from .backtest import run_backtest, trim_to_signal
from .config import experiment_config, merge
from .signals import build_weights
from .training import walk_forward_train

VERSION = "simulation-v1"
MAX_GRID = 12


def default_grid(base: dict | None = None) -> list[dict]:
    """A small grid of decision policies: how selective to be (the entry share) and whether to use a volatility stop. The base policy's other settings are kept."""
    out = []
    for q in (0.1, 0.2, 0.3):
        for stop in (None, 2.5):
            pol = {"entry_quantile": q, "exit_quantile": min(1.0, round(2 * q, 4)), "stop_loss_vol": stop}
            out.append({"id": f"entry {q:.0%}" + (f", stop {stop:g} vol" if stop else ", no stop"), "policy": pol})
    return out


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0)) if len(x) > 5 and x.std(ddof=1) > 0 else float("nan")


def model_versions(train, cfg: dict) -> list[dict]:
    """One record per refit: when it was fitted (its information cutoff), the last day its labels depended on, a hash of exactly the rows it trained on, and the dates it predicted."""
    prod = train.producers()
    params = hashlib.sha256(json.dumps(cfg["model"], sort_keys=True, default=str).encode()).hexdigest()[:10]
    feats = hashlib.sha256(",".join(train.features).encode()).hexdigest()[:10]
    out = []
    for fold, g in prod.groupby("fold", sort=True):
        d = g.index
        out.append({"version": f"v{int(fold):03d}", "fold": int(fold), "fit_cutoff": str(pd.Timestamp(g["fit_cutoff"].iloc[0]).date()), "label_end_max": str(pd.Timestamp(g["label_end_max"].iloc[0]).date()),
                    "train_rows_hash": str(g["train_rows_hash"].iloc[0]), "predicts_from": str(d[0].date()), "predicts_to": str(d[-1].date()), "n_days": int(len(d)), "model": cfg["model"]["kind"],
                    "params_hash": params, "features_hash": feats, "seed": int(cfg["seed"]), "window": cfg["split"]["window"], "embargo": int(cfg["split"]["embargo"])})
    return out


def candidate_nets(cfg: dict, grid: list[dict], ds, train, progress=None) -> dict[str, pd.Series]:
    """The net return series each candidate policy would have produced over the whole prediction history, with the same scores, costs and lag. A candidate's return on a day depends only on data before it."""
    progress = progress or (lambda m: None)
    scores = train.scores_wide(ds)
    vol = ex.trailing_vol(ds.returns)
    conf = ex.confidence_from_scores(scores, None)
    lag = ds.lag
    first = scores.dropna(how="all").index[0]
    out = {}
    for i, g in enumerate(grid):
        progress(f"scoring candidate policy {i + 1} of {len(grid)}")
        sig = {**cfg["signal"], **{f"decision_{k}": v for k, v in g["policy"].items()}}
        sig.pop("decision_schedule", None)
        w = build_weights(scores, sig, vol, conf, None, None, ds.investable, None, ds.returns)
        res = run_backtest(w, ds.returns, ds.investable, cfg["backtest"], lag, g["id"])
        out[g["id"]] = trim_to_signal(res, first, int(sig.get("lag_total", lag)))["net"]
    return out


def select_schedule(nets: dict[str, pd.Series], starts: list[pd.Timestamp], grid: list[dict], default_id: str, *, lookback_days: int = 252, min_days: int = 63, embargo: int = 5) -> list[dict]:
    """At each refit date (``starts``), the candidate with the best Sharpe ratio over the ``lookback_days`` that END ``embargo`` days before it, using only those days. With fewer than ``min_days`` of
    history for a candidate the default is kept. Returns one record per refit: the choice and every candidate's score, so the selection can be inspected."""
    frame = pd.DataFrame(nets)
    choices = []
    for s in starts:
        end = frame.index.searchsorted(pd.Timestamp(s)) - int(embargo)                 # exclusive: days at or after (start - embargo) are not used
        window = frame.iloc[max(0, end - int(lookback_days)):max(0, end)]
        scores = {k: _sharpe(window[k]) for k in frame.columns if window[k].notna().sum() >= min_days}
        scores = {k: v for k, v in scores.items() if np.isfinite(v)}
        if scores:
            best = max(scores, key=lambda k: (scores[k], k == default_id))             # ties go to the default
            choices.append({"from": str(pd.Timestamp(s).date()), "chosen": best, "score": scores[best], "scores": scores, "days_used": int(window.notna().sum().max()), "basis": "selected on prior data"})
        else:
            choices.append({"from": str(pd.Timestamp(s).date()), "chosen": default_id, "score": None, "scores": {}, "days_used": int(window.notna().sum().max()) if len(window) else 0,
                            "basis": f"default: fewer than {min_days} days of history"})
    return choices


def simulate(overrides: dict | None = None, grid: list[dict] | None = None, *, bundle=None, config=None, builder=None, registry=None, dataset=None, info=None, save: bool = True,
             lookback_days: int = 252, min_days: int = 63, progress=None, market: dict | None = None):
    """Run the scheduled simulation and return ``(experiment, simulation_block)``. ``overrides`` is an experiment configuration (the position rule is forced to ``decision_policy`` on an event-driven book)."""
    t0 = time.perf_counter()
    progress = progress or (lambda m: None)
    over = merge(overrides or {}, {"signal": {"method": "decision_policy"}, "backtest": {"rebalance": "on_change", "target_vol": None}})
    cfg = experiment_config(over)
    grid = grid or default_grid()
    if not 2 <= len(grid) <= MAX_GRID:
        raise ValueError(f"the policy grid has between 2 and {MAX_GRID} candidates (a larger search is a weaker claim)")
    ids = [g["id"] for g in grid]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate policy ids must be unique")
    ds, info = ex.prepare(cfg, bundle, config, builder, dataset, info, progress)
    progress("training walk-forward")
    train = walk_forward_train(ds, cfg["model"], cfg["split"], cfg["seed"], progress=progress)
    versions = model_versions(train, cfg)
    nets = candidate_nets(cfg, grid, ds, train, progress)
    starts = [pd.Timestamp(v["predicts_from"]) for v in versions]
    base_pol = {k[len("decision_"):]: v for k, v in cfg["signal"].items() if k.startswith("decision_") and k != "decision_schedule"}
    default_id = ids[0]
    choices = select_schedule(nets, starts, grid, default_id, lookback_days=lookback_days, min_days=min_days, embargo=int(cfg["split"]["embargo"]) + int(ds.horizon))
    by_id = {g["id"]: g["policy"] for g in grid}
    schedule, prev = [], None
    for c in choices:
        if c["chosen"] != prev:                                                          # a change of policy is a schedule entry; keeping the same one is not
            schedule.append({"from": c["from"], "policy": by_id[c["chosen"]]})
            prev = c["chosen"]
    final = merge(over, {"signal": {"decision_schedule": schedule}})
    block = {"version": VERSION, "model_versions": versions, "oos_violations": train.oos_violations(), "policy_grid": grid, "choices": choices, "schedule": schedule, "selection": {"lookback_days": lookback_days, "min_days": min_days,
             "embargo_days": int(cfg["split"]["embargo"]) + int(ds.horizon), "default": default_id, "n_candidates": len(grid)}, "base_policy": base_pol,
             "reproducibility": {"config_hash": ex.config_hash(experiment_config(final)), "dataset_key": ex.dataset_key(cfg), "seed": int(cfg["seed"]), "refits": len(versions)}}
    exp = ex.run_experiment(final, bundle, config, builder, registry, progress, save, dataset=ds, info=info, trials_extra=len(grid) - 1, market=market, train=train, report_extra={"simulation": block},
                            explain=False, final_model=False)
    oos = exp.net
    block["comparison"] = {"scheduled": {"sharpe": _sharpe(oos), "days": int(oos.notna().sum())},
                           "fixed": [{"id": k, "sharpe": _sharpe(v.reindex(oos.index))} for k, v in nets.items()],
                           "reading": "The scheduled run chose its policy on days before each refit. Each fixed row is one policy held for the whole run; the best of them is hindsight and is not what a person could have chosen in advance."}
    block["seconds"] = round(time.perf_counter() - t0, 1)
    exp.report["simulation"] = ex.jsonable(block)
    if registry is not None and save:                                                    # the comparison needs the finished run: write the report again with it
        (registry.experiment_dir(exp.id) / "report.json").write_text(json.dumps(ex.jsonable(exp.report), allow_nan=False))
    return exp, block
