"""Hyperparameter and setting search that counts every trial and keeps the reserved segments out of reach.

``ops.hpo.Study`` does the proposing (random, TPE with an evenly spread start, a Gaussian process, grid, halving) and the bookkeeping; this module supplies what the machine-learning pipeline adds:

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

from ..ops.hpo import Param, Study
from . import research_controls as rc
from .config import build_key, config_hash, dataset_key, experiment_config, merge
from .datasets import Dataset
from .experiments import Experiment, ExperimentError, prepare, run_experiment
from .models import model_catalog
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
        self.series: dict[int, "object"] = {}                                       # trial number -> its validation net returns, for the constrained selection that follows the search

    def _dataset_for(self, cfg: dict) -> Dataset:
        key = build_key(cfg)
        if key not in self._datasets:
            ds, info = prepare(cfg, self.bundle, self.config, self.builder, None, self.info)
            self.info = info
            self._datasets[key] = ds
        return self._datasets[key]

    def kind_of(self, params: dict) -> str:
        """What kind of try a candidate is, for the study's lineage: a model choice, a signal or portfolio variation, or a hyperparameter trial."""
        keys = list(params)
        if "model.kind" in keys:
            return "model_choice"
        if keys and all(k.startswith("backtest.") for k in keys):
            return "portfolio_variation"
        if keys and all(k.startswith(("signal.", "backtest.")) for k in keys):
            return "signal_variation"
        return "hpo_trial"

    def evaluate(self, params: dict, stage: str = "random") -> tuple[float, "object", dict]:
        """Run one candidate; returns ``(score, validation net returns, details)``. A configuration that cannot run scores NaN (and is still counted). With a registry the trial is registered in the study BEFORE it
        runs and finished as done, failed or cancelled: a failed trial is a try, and a cancelled search leaves the trial it was on in the record."""
        t0 = time.perf_counter()
        cfg = apply_params(self.base, params)
        number = len(self.records)
        op = None
        if self.registry is not None:
            op = rc.begin(self.registry, self.kind_of(params), rc.study_of(self.base), adopt=True, config_hash=config_hash(cfg), dataset_key=dataset_key(self.base), seed=int(cfg.get("seed", 0)),
                          detail={"search": self.study, "stage": stage, "number": number, "params": params})
        try:
            with (op if op is not None else _NoOp()):
                try:
                    cfg = experiment_config(cfg, name=f"{self.study}-{number}")
                    exp = run_experiment(cfg, dataset=self._dataset_for(cfg), info=self.info, registry=None, save=False, explain=False, final_model=False, robustness=False, stop_after="validation")
                    score, parts = validation_score(exp.report, self.weights)
                    net = exp.net[exp.labels == "validation"]
                    detail = {"score": score, **parts, "ic": exp.report["ic"]["all"]["mean"], "config_hash": config_hash(cfg), "evidence": self._evidence(exp)}
                    self.series[number] = net
                    if op is not None:
                        op.finish_detail = {"score": score if score == score else None}
                        self._keep_stream(op, net)
                except rc.ResearchCancelled:
                    raise
                except ExperimentError as exc:                                              # a configuration that cannot run is a result of the search, not a reason to stop it
                    score, net, detail = float("nan"), None, {"error": str(exc)}
                except Exception as exc:                                                    # so is an unexpected failure, recorded with its type so that it is not mistaken for a bad configuration
                    score, net, detail = float("nan"), None, {"error": f"{type(exc).__name__}: {exc}"}
                if op is not None and "error" in detail:
                    op.fail(detail["error"])
        finally:
            seconds = time.perf_counter() - t0
        self.records.append({"number": number, "params": dict(params), "stage": stage, "seconds": round(seconds, 2), "operation": None if op is None else op.op_key, **detail})
        if self.registry is not None:
            self.registry.log_trial(self.study, dataset_key(self.base), number, params, score, seconds, stage)
        return score, net, detail


    @staticmethod
    def _evidence(exp) -> dict:
        """What the constrained selection needs from this trial besides its returns (validation days only): turnover, coverage, exposure, liquidity and the complexity inputs."""
        from . import model_selection as ms
        from .comparison import _n_features

        return ms.evidence_from_run(exp.report, exp.net, exp.labels, exp.turnover, exp.weights, exp.scores, n_features=_n_features(exp), train_seconds=exp.train.seconds, fold_list=exp.train.folds)

    def _keep_stream(self, op, net) -> None:
        """The trial's validation net returns, written atomically beside the registry: the selection tests (DSR, PBO, Reality Check, SPA, FDR) need the streams of all the study's tries, and a trial that is not
        kept cannot be tested later."""
        import os

        d = self.registry.root / "trials" / op.study_id
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{op.op_key}.parquet.tmp"
        net.rename("net").to_frame().to_parquet(tmp)
        os.replace(tmp, d / f"{op.op_key}.parquet")


class _NoOp:
    def __enter__(self):
        return None

    def __exit__(self, *exc) -> bool:
        return False


def current_settings(base_overrides: dict, space: dict) -> dict:
    """The value each searched setting has in this configuration (what the form shows now): the configuration's own, else the model's default. A tuning that starts from these has the person's current
    settings as its first trial, so the answer can be compared with them."""
    cfg = experiment_config(base_overrides)
    defaults = next((m["params"] for m in model_catalog() if m["kind"] == cfg["model"]["kind"]), {})
    out = {}
    for path in space:
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        value = node.get(parts[-1]) if isinstance(node, dict) else None
        if value is None and path.startswith("model.params."):
            value = defaults.get(parts[-1])
        if value is None:
            raise ValueError(f"the current value of {path} is not known, so it cannot be tuned from the current settings")
        out[path] = value
    return out


def include(space: dict, start: dict) -> dict:
    """The space with every distribution stretched to contain the starting value (a choice added, a range widened): the starting settings are a trial the search learns from, so they must lie in it."""
    out = {}
    for path, p in space.items():
        v = start[path]
        if p.kind == "categorical":
            out[path] = p if v in p.choices else Param("categorical", choices=tuple(p.choices) + (v,))
            continue
        if p.kind == "log" and not v > 0:
            raise ValueError(f"{path} must be above zero to be tuned on a logarithmic scale")
        lo, hi = min(p.low, v), max(p.high, v)
        out[path] = p if (lo, hi) == (p.low, p.high) else Param(p.kind, int(lo) if p.kind == "int" else float(lo), int(hi) if p.kind == "int" else float(hi))
    return out


def tune(base_overrides: dict, space: dict, n_trials: int = 20, method: str = "random", seed: int = 0, bundle=None, config=None, builder=None, registry: Registry | None = None,
         study_name: str = "tune", objective_weights: dict | None = None, progress=None, start: dict | None = None, stop=None, selection: dict | bool | None = None) -> dict:
    """Search ``space`` (dotted path -> ``ops.hpo`` distribution) with ``method`` (random, tpe, bayes or optuna; grid and halving are not supported here) for ``n_trials`` candidates. Returns the trials, the best
    parameters, and the selection report that deflates the best one by the number of trials.

    ``start`` (a value for every path of the space, inside it: see ``include``) is run first as trial 0 and counts as one of the ``n_trials``; no configuration is then run twice, a TPE or Gaussian
    process search begins proposing after a third of the trials, and what comes before is an evenly spread (scrambled Sobol) sample rather than random draws: with 8 to 30 trials in a space of several
    settings, gaps and clumps in the start are most of what the search has to learn from. ``stop()`` returning ``True`` ends the search after the trial it is on.

    ``selection`` (off by default) adds ``model_selection`` to the result: the constrained multi-objective selection over every trial's pooled validation returns (see ``model_selection``). ``best`` stays the
    trial with the highest composite score; ``model_selection.selected_trial`` is the one the constrained selection chooses."""
    runner = TrialRunner(base_overrides, bundle, config, builder, registry, study_name, objective_weights)
    st = Study(space, "maximize", seed)

    def objective(params):
        if progress:
            progress(f"{study_name}: trial {len(runner.records) + 1} of {n_trials}")
        score, net, _ = runner.evaluate(params, method)
        if not np.isfinite(score):
            raise ValueError("candidate could not be evaluated")
        return score, net

    asked = {"stop": False}

    def callback(_, __):
        asked["stop"] = bool(stop())
        return asked["stop"]

    if start is None:
        st.optimize(objective, n_trials=n_trials, method=method, callback=callback if stop is not None else None)
    else:
        st._run(objective, dict(start))
        extra = {"n_startup": max(4, n_trials // 3), "quasi_start": True} if method in ("tpe", "bayes", "optuna") else {}
        st.optimize(objective, n_trials=max(0, n_trials - 1), method=method, callback=callback if stop is not None else None, unique=True, **extra)
    out = finish(st, runner)
    out["start_number"] = None if start is None else 0
    out["stopped"] = "user" if asked["stop"] else None
    if selection:
        from . import model_selection as ms

        pol = ms.policy_from({} if selection is True else selection)
        chosen = ms.select_trials(runner.records, runner.series, pol, runner.base["model"]["kind"])
        out["model_selection"] = ms.store_result(registry, chosen, pol, study_id=rc.study_of(runner.base), actor="tune")
    return out


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
             progress=None, robustness: bool | None = None, name: str | None = None, evaluation=None) -> Experiment:
    """Re-run a chosen parameter set on ALL segments: the first time the test and forward segments are predicted for it. That is a look at protected outcomes, so with a registry it needs ``evaluation``, the
    context of a frozen evaluation plan (``research_controls.begin_evaluation``; ``frozen_eval.compat_finalize`` makes one for the old ``finalize`` request). Without a registry nothing is recorded and the run is
    labelled unrecorded. ``trials_extra`` counts trials that are NOT in the registry; a search that logged its trials there passes 0 (each would otherwise be counted twice)."""
    if registry is not None and evaluation is None:
        raise rc.HoldoutAccessError("finalising reads the test and forward outcomes, so it needs a frozen evaluation plan: create and freeze one (POST /api/ml/evaluation-plan) and evaluate it (POST /api/ml/evaluate). "
                                    "The old finalize shortcut is routed through frozen_eval.compat_finalize, which does exactly that")
    cfg = apply_params(experiment_config(base_overrides), params)
    if name:
        cfg["name"] = str(name)[:80]
    return run_experiment(merge(cfg, {}), bundle=bundle, config=config, builder=builder, registry=registry, progress=progress, trials_extra=trials_extra, parent_id=parent_id, robustness=robustness,
                          evaluation=evaluation, kind="finalize")
