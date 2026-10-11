"""Automated strategy search: random, then Bayesian, then genetic, with every trial counted and the test and forward segments kept out of reach until the end.

A candidate is a complete experiment configuration: the model and its hyperparameters, the horizon and target, the signal method and its settings, the rebalance frequency. Each candidate is evaluated as a
SEARCH-MODE experiment (``run_experiment(..., stop_after="validation")``): it is trained walk-forward only as far as the end of the validation segment, so the test and forward segments are never predicted. Its
score is ``optimization.validation_score``. The three stages share one record of trials:

1. random      candidates drawn from the whole space, a few per model kind: a map of what works at all
2. bayesian    per model kind, a Tree-structured Parzen Estimator proposes candidates near the good ones found so far (``ops.hpo.Study``)
3. genetic     the best candidates across kinds become a population; offspring combine two parents' settings (uniform crossover) and mutate one or two; the fittest survive

At the end the ``top_k`` candidates are run once, on every segment, as full experiments (explanations, final model, registry) and the deflated Sharpe ratio of each counts ALL the trials of the search. The report
says how many configurations were tried, because a leaderboard without that number is advertising.
"""

from __future__ import annotations

import time

import numpy as np

from ..ops.hpo import Categorical, Int, Study, Trial
from .config import dataset_key
from .models import model_catalog, search_space
from .optimization import TrialRunner, apply_params, finalize
from .registry import Registry
from .targets import is_classification

DEFAULT_KINDS = ("ridge", "lasso", "elasticnet", "random_forest", "hist_gb", "online_ridge", "kalman")
SHARED_SPACE = {"dataset.horizon": Categorical(1, 5, 10, 20), "signal.method": Categorical("top_n_long_short", "proportional", "rank_normalised", "long_only_top_quantile"),
                "signal.n": Int(1, 4), "signal.holding_period": Categorical(1, 5, 20), "backtest.rebalance": Categorical("daily", "weekly", "monthly")}


def kind_space(kind: str, shared: dict | None = None) -> dict:
    """The space of one model kind: its own hyperparameters (as ``model.params.*``) plus the shared signal settings."""
    own = {f"model.params.{k}": v for k, v in search_space(kind).items()}
    return {**own, **(shared if shared is not None else SHARED_SPACE)}


def _with_kind(params: dict, kind: str) -> dict:
    return {"model.kind": kind, **params}


def _available_kinds(kinds, target: str) -> list[str]:
    cat = {m["kind"]: m for m in model_catalog()}
    task = "classification" if is_classification(target) else "regression"
    return [k for k in kinds if k in cat and cat[k]["available"] and task in cat[k]["tasks"]]


class SearchRunner(TrialRunner):
    """A trial runner whose candidates may change the model kind."""

    def evaluate(self, params: dict, stage: str = "random"):
        return super().evaluate(self._flatten(params), stage)

    def _flatten(self, params: dict) -> dict:
        kind = params.get("model.kind", self.base["model"]["kind"])
        keep = {k: v for k, v in params.items() if not k.startswith("model.params.") or k.split(".", 2)[2] in search_space(kind)}
        return keep


def mutate(params: dict, spaces: dict, rng: np.random.Generator, n_mut: int = 1) -> dict:
    """Resample ``n_mut`` randomly chosen settings from their distributions."""
    child = dict(params)
    keys = [k for k in child if k in spaces[child["model.kind"]]]
    for k in rng.choice(keys, size=min(n_mut, len(keys)), replace=False):
        child[k] = spaces[child["model.kind"]][k].sample(rng)
    return child


def crossover(a: dict, b: dict, spaces: dict, rng: np.random.Generator) -> dict:
    """Uniform crossover: each shared signal setting comes from either parent; the model kind and its hyperparameters come together from one parent (hyperparameters mean nothing across kinds)."""
    lead = a if rng.random() < 0.5 else b
    child = {"model.kind": lead["model.kind"], **{k: v for k, v in lead.items() if k.startswith("model.params.")}}
    for k in SHARED_SPACE:
        pick = a if rng.random() < 0.5 else b
        if k in pick:
            child[k] = pick[k]
    for k in spaces[child["model.kind"]]:
        child.setdefault(k, spaces[child["model.kind"]][k].sample(rng))
    return child


def search(base_overrides: dict, bundle=None, config=None, builder=None, registry: Registry | None = None, kinds: tuple = DEFAULT_KINDS, n_random: int = 12, n_bayes: int = 12,
           generations: int = 3, population: int = 8, top_k: int = 3, seed: int = 0, study_name: str = "search", objective_weights: dict | None = None, progress=None,
           finalize_top: bool = True) -> dict:
    """Run the three-stage search and (optionally) finalise the best ``top_k``. Returns the stages, the trials, the selection report over all trials and the finalists' reports."""
    t0 = time.perf_counter()
    runner = SearchRunner(base_overrides, bundle, config, builder, registry, study_name, objective_weights)
    target = runner.base["dataset"]["target"]
    use = _available_kinds(kinds, target)
    if not use:
        raise ValueError("none of the requested model kinds can run on this target")
    spaces = {k: kind_space(k) for k in use}
    rng = np.random.default_rng(seed)
    pool: list[dict] = []                                                    # {params, score}
    studies = {k: Study(spaces[k], "maximize", seed) for k in use}
    shared_returns = Study({}, "maximize", seed)                            # holds every trial's validation returns for the final selection report

    def run(kind: str, params: dict, stage: str) -> float:
        cand = _with_kind(params, kind)
        if progress:
            progress(f"{stage}: trial {len(runner.records) + 1}")
        score, net, detail = runner.evaluate(cand, stage)
        shared_returns.trials.append(Trial(len(shared_returns.trials), cand, score if np.isfinite(score) else float("nan"), net if np.isfinite(score) else None, 0.0, None, detail.get("error")))
        pool.append({"params": cand, "score": score})
        return score

    per_kind = max(2, n_random // len(use))
    for k in use:                                                            # 1. random
        for _ in range(per_kind):
            run(k, {p: dist.sample(rng) for p, dist in spaces[k].items()}, "random")
    for k in use:                                                            # 2. bayesian: seed each kind's study with its random trials, then let it propose
        study = studies[k]
        for rec in [p for p in pool if p["params"]["model.kind"] == k]:
                study.trials.append(Trial(len(study.trials), {x: y for x, y in rec["params"].items() if x != "model.kind"}, rec["score"], None, 0.0))
    ranked_kinds = sorted(use, key=lambda k: -max([p["score"] for p in pool if p["params"]["model.kind"] == k and np.isfinite(p["score"])] or [-9.0]))
    tpe_budget = max(0, n_bayes)
    for i in range(tpe_budget):
        k = ranked_kinds[i % max(1, min(len(ranked_kinds), 3))]              # the three most promising kinds share the budget
        proposal = studies[k]._tpe(np.random.default_rng(seed + 1000 + i), n_startup=min(6, per_kind))
        score = run(k, proposal, "bayes")
        studies[k].trials.append(Trial(len(studies[k].trials), proposal, score, None, 0.0))
    ok = sorted([p for p in pool if np.isfinite(p["score"])], key=lambda p: -p["score"])   # 3. genetic
    pop = ok[: max(2, population)]
    for g in range(max(0, generations)):
        if len(pop) < 2:
            break
        children = []
        for _ in range(max(2, population // 2)):
            i, j = rng.choice(len(pop), size=2, replace=False)
            child = mutate(crossover(pop[i]["params"], pop[j]["params"], spaces, rng), spaces, rng, 1 + int(rng.random() < 0.3))
            if any(child == q["params"] for q in pool):
                continue
            children.append({"params": child, "score": run(child["model.kind"], {k: v for k, v in child.items() if k != "model.kind"}, f"genetic-{g + 1}")})
        pop = sorted([p for p in pop + children if np.isfinite(p["score"])], key=lambda p: -p["score"])[: max(2, population)]
    n_trials = len(runner.records)
    result = {"n_trials": n_trials, "stages": {s: sum(1 for r in runner.records if r["stage"].startswith(s)) for s in ("random", "bayes", "genetic")}, "kinds": use, "trials": runner.records,
              "best": None, "selection": None, "finalists": [], "seconds": 0.0, "note": "Scores are validation-only; the test and forward segments were not predicted during the search."}
    good = sorted([r for r in runner.records if r.get("score") is not None and np.isfinite(r.get("score", np.nan))], key=lambda r: -r["score"])
    if good:
        result["best"] = good[0]
    if len([t for t in shared_returns.trials if t.returns is not None]) >= 3:
        try:
            result["selection"] = shared_returns.selection_report(n_boot=200)
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            result["selection"] = {"error": str(exc)}
    if finalize_top and good:
        for rec in good[: max(1, top_k)]:
            if progress:
                progress(f"finalising trial {rec['number']}")
            params = rec["params"]
            cfg = apply_params(runner.base, params)
            exp = finalize(cfg, {}, bundle, config, builder, registry, trials_extra=n_trials, parent_id=None, robustness=False)
            result["finalists"].append({"trial": rec["number"], "experiment_id": exp.id, "name": exp.cfg["name"], "params": params, "validation_score": rec["score"], "segments": exp.report["segments"]["stats"],
                                        "overfit": exp.report["overfit"], "trials": exp.report["trials"]["dataset_trials"]})
    result["seconds"] = round(time.perf_counter() - t0, 1)
    result["dataset_key"] = dataset_key(runner.base)
    return result
