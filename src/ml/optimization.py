"""Hyperparameter and setting search that counts every trial and keeps the reserved segments out of reach.

``ops.hpo.Study`` does the proposing (random, TPE, grid, halving) and the bookkeeping; this module supplies what the machine-learning pipeline adds:

* ``apply_params``      dotted paths (``model.params.alpha``, ``signal.n``, ``dataset.horizon``) written into an experiment configuration
* ``validation_score``  the objective: validation Sharpe less penalties for turnover, drawdown, instability across the validation sub-periods and the gap between the train and validation Sharpe
* ``tune``              a search over a space of such paths; every trial is a full walk-forward experiment that stops at the end of the validation segment (``stop_after``), so the test and forward
  segments are never predicted, let alone looked at
* the selection report  the deflated Sharpe ratio of the winner, the probability of backtest overfitting and Romano-Wolf p-values over ALL trials' validation returns (``Study.selection_report``)

The default objective weights are in ``config/ml_research.yaml`` (``search.objective_weights``). Annual turnover enters in units of 20 times a year.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from ..ops.hpo import Study
from .config import build_key, config_hash, dataset_key, experiment_config, merge
from .datasets import Dataset
from .experiments import Experiment, ExperimentError, prepare, run_experiment
from .registry import Registry

TURNOVER_UNIT = 20.0                                                         # annual one-way turnover that counts as 1.0 in the score


def apply_params(cfg: dict, params: dict) -> dict:
    """A copy of ``cfg`` with each dotted-path parameter written in (``"model.params.alpha": 3.0`` sets ``cfg["model"]["params"]["alpha"]``). The target follows the horizon when only the horizon is searched."""
    out = copy.deepcopy(cfg)
    if "model.kind" in params and params["model.kind"] != out["model"]["kind"]:
        out["model"]["params"] = {}                                              # the previous kind's hyperparameters mean nothing to the new one
    for path, value in params.items():
        node = out
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    if "dataset.horizon" in params and "dataset.target" not in params:
        kind = out["dataset"]["target"].rsplit("_", 1)[0]
        out["dataset"]["target"] = f"{kind}_{int(params['dataset.horizon'])}"
    return out


def validation_score(report: dict, weights: dict | None = None) -> tuple[float, dict]:
    """The search objective and its parts: ``validation_sharpe - w_turnover * turnover - w_drawdown * |max drawdown| - w_instability * instability - w_gap * max(0, train_sharpe - validation_sharpe)``.
    NaN if the validation segment has no Sharpe ratio."""
    w = {"turnover": 0.5, "max_drawdown": 0.3, "instability": 0.2, "train_validation_gap": 0.5, **(weights or {})}
    st = report["segments"]["stats"]
    val = st.get("validation")
    if not val or val.get("sharpe") is None:
        return float("nan"), {}
    sharpe = float(val["sharpe"])
    turnover = float(report["metrics"]["oos"].get("ann_turnover") or 0.0) / TURNOVER_UNIT
    dd = abs(float(val.get("max_drawdown") or 0.0))
    inst = report["overfit"].get("instability")
    inst = 0.0 if inst is None else float(inst)
    tr = st.get("train", {}).get("sharpe")
    gap = max(0.0, float(tr) - sharpe) if tr is not None else 0.0
    score = sharpe - w["turnover"] * turnover - w["max_drawdown"] * dd - w["instability"] * inst - w["train_validation_gap"] * gap
    return float(score), {"validation_sharpe": sharpe, "turnover": turnover, "max_drawdown": dd, "instability": inst, "train_validation_gap": gap}


class TrialRunner:
    """Evaluates candidate parameter sets as search-mode experiments on a shared dataset (rebuilt only when a parameter that changes the data is searched)."""

    def __init__(self, base_overrides: dict, bundle=None, config=None, builder=None, registry: Registry | None = None, study: str = "search", objective_weights: dict | None = None,
                 info: dict | None = None):
        self.base = experiment_config(base_overrides)
        self.bundle, self.config, self.builder, self.registry, self.study, self.weights = bundle, config, builder, registry, study, objective_weights
        self.info = info
        self._datasets: dict[str, Dataset] = {}
        self.records: list[dict] = []

    def _dataset_for(self, cfg: dict) -> Dataset:
        key = build_key(cfg)
        if key not in self._datasets:
            ds, info = prepare(cfg, self.bundle, self.config, self.builder, None, self.info)
            self.info = info
            self._datasets[key] = ds
        return self._datasets[key]

    def evaluate(self, params: dict, stage: str = "random") -> tuple[float, "object", dict]:
        """Run one candidate; returns ``(score, validation net returns, details)``. A configuration that cannot run scores NaN (and is still counted)."""
        t0 = time.perf_counter()
        cfg = apply_params(self.base, params)
        number = len(self.records)
        try:
            cfg = experiment_config(cfg, name=f"{self.study}-{number}")
            exp = run_experiment(cfg, dataset=self._dataset_for(cfg), info=self.info, registry=None, save=False, explain=False, final_model=False, robustness=False, stop_after="validation")
            score, parts = validation_score(exp.report, self.weights)
            net = exp.net[exp.labels == "validation"]
            detail = {"score": score, **parts, "ic": exp.report["ic"]["all"]["mean"], "config_hash": config_hash(cfg)}
        except ExperimentError as exc:                                              # a configuration that cannot run is a result of the search, not a reason to stop it
            score, net, detail = float("nan"), None, {"error": str(exc)}
        except Exception as exc:                                                    # so is an unexpected failure, recorded with its type so that it is not mistaken for a bad configuration
            score, net, detail = float("nan"), None, {"error": f"{type(exc).__name__}: {exc}"}
        seconds = time.perf_counter() - t0
        self.records.append({"number": number, "params": dict(params), "stage": stage, "seconds": round(seconds, 2), **detail})
        if self.registry is not None:
            self.registry.log_trial(self.study, dataset_key(self.base), number, params, score, seconds, stage)
        return score, net, detail


def tune(base_overrides: dict, space: dict, n_trials: int = 20, method: str = "random", seed: int = 0, bundle=None, config=None, builder=None, registry: Registry | None = None,
         study_name: str = "tune", objective_weights: dict | None = None, progress=None) -> dict:
    """Search ``space`` (dotted path -> ``ops.hpo`` distribution) with ``method`` (random, tpe, grid, halving not supported here) for ``n_trials`` candidates. Returns the trials, the best parameters,
    and the selection report that deflates the best one by the number of trials."""
    runner = TrialRunner(base_overrides, bundle, config, builder, registry, study_name, objective_weights)
    st = Study(space, "maximize", seed)

    def objective(params):
        if progress:
            progress(f"{study_name}: trial {len(runner.records) + 1}")
        score, net, _ = runner.evaluate(params, method)
        if not np.isfinite(score):
            raise ValueError("candidate could not be evaluated")
        return score, net

    st.optimize(objective, n_trials=n_trials, method=method)
    return finish(st, runner)


def finish(st: Study, runner: TrialRunner) -> dict:
    """Summarise a study: the trial table, the best, and the selection report when at least two trials produced returns."""
    ok = [t for t in st.trials if t.error is None and np.isfinite(t.value)]
    out = {"n_trials": st.n_trials, "n_failed": st.n_trials - len(ok), "trials": runner.records, "best": None, "selection": None}
    if ok:
        b = st.best
        out["best"] = {"number": b.number, "params": b.params, "score": b.value, **{k: v for k, v in runner.records[b.number].items() if k in ("validation_sharpe", "turnover", "max_drawdown", "instability", "train_validation_gap", "ic")}}
    if len([t for t in ok if t.returns is not None]) >= 3:
        try:
            out["selection"] = st.selection_report(n_boot=200)
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            out["selection"] = {"error": str(exc)}
    return out


def finalize(base_overrides: dict, params: dict, bundle=None, config=None, builder=None, registry: Registry | None = None, trials_extra: int = 0, parent_id: str | None = None,
             progress=None, robustness: bool | None = None) -> Experiment:
    """Re-run a chosen parameter set on ALL segments (the first time the test and forward segments are predicted for it), save it, and count the search's trials against it."""
    cfg = apply_params(experiment_config(base_overrides), params)
    return run_experiment(merge(cfg, {}), bundle=bundle, config=config, builder=builder, registry=registry, progress=progress, trials_extra=trials_extra, parent_id=parent_id, robustness=robustness)
