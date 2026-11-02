"""Controlled comparisons against a frozen baseline (spec v3, sections 9.3 and 10.2).

An enhancement is judged by one change at a time, on the same market sample, folds, costs and constraints as the baseline, on development data only, with the rule that decides the outcome written down
before the numbers exist. This module is that discipline as code, shared by the Release 6 families (target, feature, horizon) and the Release 7 families that follow:

* ``new_card`` builds the experiment card of section 10.2 from a baseline and its candidates (each a *change* laid over the baseline's overrides) and refuses a change that alters anything the family does not
  allow (``diff_paths``) or alters nothing at all;
* ``freeze`` stores it, append-only, with the time it was frozen; the card cannot change afterwards and a second look needs a second card;
* ``run`` runs the baseline and every candidate through ``run_experiment`` as development runs (``stop_after="validation"``: the test and forward outcomes are not predicted), checks that the data and the fold
  boundaries were the same, compares each candidate with the baseline on the dates both have, and stores the result next to the card: the metrics with their uncertainty, the verdict, the negative findings
  and the gate decision (always "not evaluated": a development comparison is not a frozen final evaluation);
* ``get`` returns the card with its results, which is the frozen experiment card the dashboard shows.

The primary metric is paired (the same dates in both arms). For ``ic`` it is the mean difference of the daily rank IC against the common raw forward return of the baseline's horizon, with a Newey-West
interval whose lag is at least ``horizon - 1``; for ``net_sharpe`` it is the Sharpe difference of the daily net returns in a joint stationary bootstrap. A candidate is called *supported on development* only if
the Holm-adjusted p-value over all the card's candidates is below the level, the interval excludes zero, and the difference is at least the declared minimum effect; otherwise the verdict is
``inconclusive`` or ``worse``, and both stay in the record. Nothing here promotes anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import inference as inf
from . import parsimony as pa
from .backtest import label_days
from .config import build_key, config_hash, dataset_key, experiment_config, merge
from .signals import DERIVED_KEYS
from .targets import forward_return

VERSION = "comparison-v1"
IGNORED = ("name", "notes", "study", "registry")
SEGMENT = "validation"

# the experiment families of spec section 10.2 and what each may change; ``folds`` says whether the baseline's fold boundaries must be reproduced exactly
FAMILIES = {
    "target": {"label": "Target engineering", "baseline": "raw future returns", "candidate": "relative or residual targets", "allowed": ("dataset.target",), "primary": "ic",
               "secondary": ["net_sharpe", "turnover", "costs", "coverage"], "folds": "identical"},
    "feature": {"label": "Feature engineering", "baseline": "existing features", "candidate": "simple motivated feature sets, then stability-selected features", "allowed": ("model", "dataset.feature_groups"), "primary": "ic",
                "secondary": ["net_sharpe", "coverage", "selection_stability", "n_features"], "folds": "identical"},
    "horizon": {"label": "Horizon", "baseline": "the baseline horizon", "candidate": "another horizon with its matching holding period", "allowed": ("dataset.target", "dataset.horizon", "signal.holding_period"),
                "primary": "net_sharpe", "secondary": ["ic_own_horizon", "turnover", "costs", "coverage"], "folds": "comparable"},
    "portfolio": {"label": "Portfolio construction", "baseline": "existing corrected sizing", "candidate": "cost-aware allocation", "allowed": ("signal",), "primary": "net_sharpe",
                  "secondary": ["turnover", "costs", "max_drawdown"], "folds": "identical"},
    "ensemble": {"label": "Ensembles", "baseline": "development-selected single model", "candidate": "diversified ensemble", "allowed": ("model",), "primary": "net_sharpe",
                 "secondary": ["ic", "turnover", "n_features"], "folds": "identical"},
    "rebalancing": {"label": "Rebalancing", "baseline": "fixed schedule", "candidate": "no-trade region", "allowed": ("signal", "backtest", "execution"), "primary": "net_sharpe",
                    "secondary": ["turnover", "costs", "max_drawdown"], "folds": "identical"},
    "risk_control": {"label": "Risk control", "baseline": "static exposure", "candidate": "causal regime exposure", "allowed": ("signal", "backtest", "dataset.regime_detector"), "primary": "net_sharpe",
                     "secondary": ["max_drawdown", "turnover"], "folds": "identical"},
    "execution": {"label": "Execution", "baseline": "simplified fills", "candidate": "capacity-aware fills", "allowed": ("execution", "backtest"), "primary": "net_sharpe",
                  "secondary": ["costs", "turnover"], "folds": "identical"},
    # a combination is built by ``combination_card`` from components that were each compared on their own; it may change what its components changed and nothing else
    "combination": {"label": "Combination of components already compared", "baseline": "the baseline the components were each compared against", "candidate": "several components at once",
                    "allowed": (), "primary": "net_sharpe", "secondary": ["ic", "turnover", "costs", "max_drawdown"], "folds": "comparable"},
}
PRIMARY_METRICS = ("ic", "net_sharpe")


class ComparisonError(ValueError):
    """A comparison that cannot be set up or run as declared; the message says why."""


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, default=str, separators=(",", ":"), allow_nan=False)


def digest(x, n: int = 16) -> str:
    return hashlib.sha256(_canon(x).encode()).hexdigest()[:n]


def _leaves(d, prefix=""):
    if isinstance(d, dict) and d:
        for k, v in d.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, d


def diff_paths(a: dict, b: dict) -> list[str]:
    """The dotted paths at which two full configurations differ, leaf by leaf; the name, notes and study (labels, not settings) are ignored."""
    la = {k: v for k, v in _leaves({k: v for k, v in a.items() if k not in IGNORED})}
    lb = {k: v for k, v in _leaves({k: v for k, v in b.items() if k not in IGNORED})}
    return sorted(k for k in set(la) | set(lb) if _canon(la.get(k, "<absent>")) != _canon(lb.get(k, "<absent>")))


def _ladder_rank(change: dict) -> int:
    groups = (change.get("dataset") or {}).get("feature_groups")
    if groups is not None and "model" not in change:
        return 0 if len(groups) == 1 else 1
    return 2


def _post_model(changed: list[str]) -> bool:
    """True if every changed setting acts after the predictions exist (the position rule, the costs and schedule, the execution), so the arm can reuse the baseline's saved predictions."""
    return bool(changed) and all(p.split(".")[0] in ("signal", "backtest", "execution") for p in changed)


def _allowed(path: str, allowed: tuple) -> bool:
    return any(path == a or path.startswith(a + ".") for a in allowed)


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjustment (monotone, controls the family-wise error rate); ``None`` stays ``None``."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    out: list = [None] * len(pvalues)
    running = 0.0
    m = len(order)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(pvalues[i])))
        out[i] = running
    return out


# --------------------------------------------------------------------------------------------------------------------- the card
def new_card(hypothesis: str, family: str, baseline: dict, candidates: list[dict], *, study_id: str, one_change_description: str, primary_metric: str | None = None,
             secondary_metrics: list[str] | None = None, parsimony: dict | None = None, uncertainty_method: str | None = None, minimum_coverage: float = 0.8, min_effect: float = 0.0, level: float = 0.05,
             robustness_grid: dict | None = None, final_evaluation_policy: str | None = None, compute_budget: dict | None = None, n_boot: int = 1000, seed: int = 0, base_defaults: dict | None = None,
             lineage: dict | None = None, allowed: tuple | None = None, baseline_selection: dict | None = None) -> dict:
    """The declared part of the card of spec section 10.2. Candidates are experiment overrides; each is merged into its full configuration and must differ from the baseline only where the family allows."""
    if family not in FAMILIES:
        raise ComparisonError(f"unknown experiment family '{family}'; the families are {sorted(FAMILIES)}")
    fam = FAMILIES[family]
    if not candidates:
        raise ComparisonError("a comparison needs at least one candidate")
    if not str(hypothesis).strip() or not str(one_change_description).strip():
        raise ComparisonError("state the hypothesis and the one change before the numbers exist: both are required")
    primary = primary_metric or fam["primary"]
    if primary not in PRIMARY_METRICS:
        raise ComparisonError(f"the primary metric must be one of {list(PRIMARY_METRICS)}")
    if not 0.0 < level < 1.0 or not 0.0 <= minimum_coverage <= 1.0 or min_effect < 0:
        raise ComparisonError("level must be in (0, 1), minimum_coverage in [0, 1] and min_effect non-negative")
    if parsimony is not None:
        pa.validate_rule(dict(parsimony))
    allowed_paths = tuple(allowed) if allowed is not None else fam["allowed"]
    if family == "combination" and not lineage:
        raise ComparisonError("a combination card is built from components that were compared on their own: use combination_card or ablation_card")
    bcfg = experiment_config({**baseline, "study": study_id}, defaults=base_defaults)
    cands = []
    for i, change in enumerate(candidates):
        over = merge(baseline, change)                                                                    # a candidate is the baseline with ONE declared change laid over it
        ccfg = experiment_config({**over, "study": study_id}, defaults=base_defaults)
        derived = [p for p in diff_paths(bcfg, ccfg) if p.startswith("signal.") and p.split(".", 1)[1] in DERIVED_KEYS]                       # filled in by the configuration from the other sections, not declared
        paths = [p for p in diff_paths(bcfg, ccfg) if p not in derived]
        stray = [p for p in paths if not _allowed(p, allowed_paths)]
        if not paths:
            raise ComparisonError(f"candidate {i + 1} is the baseline: it changes nothing")
        if stray:
            raise ComparisonError(f"candidate {i + 1} differs from the baseline in {stray}, which the {family} family does not allow ({list(allowed_paths)}): a comparison changes one thing, on matched assumptions")
        name = ccfg["name"] if ccfg["name"] != bcfg["name"] and ccfg["name"] not in [c["name"] for c in cands] else f"{bcfg['name']} #{i + 1}"[:80]            # an arm has a name of its own, whatever it changes
        cands.append({"name": name, "change": change, "overrides": over, "config_hash": config_hash(ccfg), "changed": paths, "derived": derived})
    ladder = None
    if family == "feature":                                                                                  # economically motivated, low-complexity sets are compared before the selected one
        rank = lambda c: _ladder_rank(c["change"])                                                           # noqa: E731
        cands.sort(key=rank)                                                                                  # a stable sort: the order given is kept within a rung
        ladder = {"rungs": ["one feature group", "several feature groups", "a different model or selection"], "order": [c["name"] for c in cands], "ranks": [rank(c) for c in cands],
                  "simple_first": True, "has_low_complexity_candidate": any(rank(c) == 0 for c in cands),
                  "note": "single-group feature sets, each with an economic story, are compared before the stability-selected set; a card with none declares no low-complexity comparison"}
    content = {
        "version": VERSION, "hypothesis": str(hypothesis).strip(), "study_id": study_id, "family": family, "family_label": fam["label"], "baseline_version": config_hash(bcfg),
        "baseline": {"name": bcfg["name"], "overrides": baseline, "config_hash": config_hash(bcfg)}, "candidates": cands, "candidate_version": [c["config_hash"] for c in cands],
        "one_change_description": str(one_change_description).strip(), "dataset_key": dataset_key(bcfg),
        "execution_cost_risk_capital_assumptions": {"backtest": bcfg["backtest"], "execution": bcfg.get("execution"), "signal": bcfg["signal"], "split": bcfg["split"]},
        "primary_metric": primary, "secondary_metrics": list(secondary_metrics or fam["secondary"]), "selection_rule": {"parsimony": parsimony, "supported_if": {"holm_level": level, "min_effect": min_effect,
                                                                                                                                                              "interval_excludes_zero": True}},
        "uncertainty_method": uncertainty_method or ("Newey-West mean of the paired daily rank IC difference, lag at least horizon - 1" if primary == "ic" else "joint stationary bootstrap of the paired Sharpe difference"),
        "multiple_testing_scope": {"method": "holm", "family_size": len(cands), "level": level, "note": "over the candidates of this card; looks at the same data in other cards are counted by the study, not here"},
        "compute_search_budget": compute_budget or {"runs": 1 + len(cands), "n_boot": n_boot}, "minimum_coverage": minimum_coverage, "robustness_grid": robustness_grid or {},
        "final_evaluation_policy": final_evaluation_policy or "none: this card compares on development data (the validation segment). A final evaluation needs a frozen evaluation plan and its own reservation.",
        "settings": {"n_boot": int(n_boot), "seed": int(seed), "segment": SEGMENT, "level": level, "min_effect": min_effect}, "ladder": ladder,
    }
    if lineage:                                                                                           # a combination or an ablation: its components, and the tests it shares a family with
        content["lineage"] = lineage
        content["multiple_testing_scope"] = {**content["multiple_testing_scope"], "family_size": len(cands) + len(lineage.get("inherited", [])),
                                             "note": "over the candidates of this card AND the tests of its components: a combination does not reuse the components' findings for free"}
    if baseline_selection:                                                                                # the baseline was itself the best of several runs: recorded, and a reason to read a gain over it as conservative
        content["baseline_selection"] = baseline_selection
    content["card_id"] = "cmp-" + digest(content, 12)
    return content


def freeze(registry, card: dict, actor: str = "api") -> dict:
    """Store the card, append-only, and return it with the time it was frozen. Freezing the same card again returns the stored one."""
    from . import diagnostics as dx

    found = dx.latest(registry, card["card_id"], "comparison_card")
    if found is not None:
        return {**found["content"], "frozen_at": found["created"], "already_frozen": True}
    dx.store(registry, "comparison_card", card["card_id"], card, study_id=card["study_id"], actor=actor)
    found = dx.latest(registry, card["card_id"], "comparison_card")
    return {**found["content"], "frozen_at": found["created"], "already_frozen": False}


def get(registry, card_id: str) -> dict:
    """The frozen card with its results: every field of spec section 10.2 (``results``, ``negative_findings``, ``gate_decision`` and ``evidence_artifacts`` are empty until the card has been run)."""
    from . import diagnostics as dx

    card = dx.latest(registry, card_id, "comparison_card")
    if card is None:
        raise ComparisonError(f"unknown comparison card '{card_id}'")
    res = dx.latest(registry, card_id, "comparison_result")
    out = {**card["content"], "frozen_at": card["created"], "results": None, "negative_findings": [], "gate_decision": {"gate": "B", "status": "not_evaluated", "reason": "the card has not been run"},
           "evidence_artifacts": {}, "observed": None, "state": "frozen"}
    if res is not None:
        r = res["content"]
        out.update(results=r.get("results"), negative_findings=r.get("negative_findings", []), gate_decision=r.get("gate_decision"), evidence_artifacts=r.get("evidence_artifacts", {}),
                   observed=r.get("observed"), state=r.get("state", "run"), ran_at=res["created"])
    return out


FEATURE_GROUPS = ("price", "momentum", "mean_reversion", "volatility", "volume", "relative")
WEIGHTINGS = ("equal", "dynamic", "regime", "stacked", "stacked_regime")
PRESETS = {"target": ("excess_ret", "sector_ret", "resid_ret"), "horizon": (1, 5, 10, 20), "feature": ("stable_select", *(f"group:{g}" for g in FEATURE_GROUPS)), "portfolio": ("cost_aware", "calibrated_sizing"),
           "rebalancing": ("no_trade:<margin>", "trade_filter:<k>"), "risk_control": ("regime_exposure:vol_quantile", "regime_exposure:hmm", "regime_exposure:markov_switching"),
           "ensemble": tuple(f"weighting:{w}" for w in WEIGHTINGS), "execution": ("realistic",)}
PARAMETRISED = {"rebalancing": ("no_trade", "trade_filter"), "risk_control": ("regime_exposure",), "ensemble": ("weighting",)}              # presets written kind:value


def _parametrised(family: str, choice, options: dict) -> dict:
    """One preset written ``kind:value`` as a change laid over the baseline."""
    kind, _, value = str(choice).partition(":")
    if kind not in PARAMETRISED.get(family, ()):
        raise ComparisonError(f"{choice!r} is not a {family} preset; the presets are {list(PRESETS[family])}")
    try:
        if kind == "no_trade":
            return {"signal": {"no_trade_margin": float(value)}}
        if kind == "trade_filter":
            return {"signal": {"trade_filter_k": float(value)}}
        if kind == "regime_exposure":
            engine, _, states = value.partition(":")
            return {"signal": {"regime_exposure": engine, **({"regime_states": int(states)} if states else {})}}
    except ValueError:
        raise ComparisonError(f"{choice!r}: the value after the colon must be a number") from None
    if value not in WEIGHTINGS:                                                                                  # the ensemble weightings
        raise ComparisonError(f"{choice!r}: the weighting must be one of {list(WEIGHTINGS)}")
    members = str(options.get("members") or "").strip()
    if not members:
        raise ComparisonError("an ensemble preset needs the members of the ensemble (for example ridge,lasso,elasticnet)")
    return {"model": {"kind": "model_ensemble", "params": {"members": members, "weighting": value}}}


def preset_changes(family: str, baseline: dict, choices: list, options: dict | None = None) -> list[dict]:
    """The standard candidates of a family as changes to ``baseline``: another kind of target at the baseline's horizon; other horizons, each with its matching holding period; the baseline's linear model behind a
    stability selection; the cost-aware optimiser in place of the baseline's position rule. A choice the baseline already is (its own horizon, say) is left out."""
    if family not in PRESETS:
        raise ComparisonError(f"{family} has no presets; give the changes yourself")
    cfg = experiment_config(baseline)
    out = []
    for choice in choices:
        if family in PARAMETRISED:
            out.append(_parametrised(family, choice, options or {}))
            continue
        if choice not in PRESETS[family]:
            raise ComparisonError(f"{choice!r} is not a {family} preset; the presets are {list(PRESETS[family])}")
        if family == "execution":
            out.append({"execution": {"mode": "realistic_cash_equity", **{k: v for k, v in (options or {}).items() if k in ("capital", "max_participation")}}})
        elif family == "portfolio" and choice == "calibrated_sizing":
            out.append({"signal": {"method": "calibrated_sizing", "holding_period": 1}})
        elif family == "target":
            h = int(cfg["dataset"]["horizon"])
            out.append({"dataset": {"target": f"{choice}_{h}"}})
        elif family == "horizon":
            if int(choice) != int(cfg["dataset"]["horizon"]):
                from .targets import parse_target

                kind = parse_target(cfg["dataset"]["target"])[0]
                out.append({"dataset": {"target": f"{kind}_{int(choice)}", "horizon": int(choice)}, "signal": {"holding_period": int(choice)}})
        elif family == "feature" and str(choice).startswith("group:"):
            out.append({"dataset": {"feature_groups": [str(choice).split(":", 1)[1]]}})                       # one economically motivated group on its own
        elif family == "feature":
            base_kind = cfg["model"]["kind"] if cfg["model"]["kind"] in ("linear", "ridge", "lasso", "elasticnet", "logistic") else "ridge"
            params = {"base": base_kind}
            if base_kind == cfg["model"]["kind"] and "alpha" in (cfg["model"].get("params") or {}):
                params["alpha"] = cfg["model"]["params"]["alpha"]
            out.append({"model": {"kind": "stable_select", "params": params}})
        else:
            out.append({"signal": {"method": "cost_aware_mean_variance", "holding_period": 1, "opt_every": int(cfg["dataset"]["horizon"])}})
    if not out:
        raise ComparisonError("the presets left no candidate: each one is the baseline itself")
    return out


def list_cards(registry, study_id: str | None = None, limit: int = 50) -> list[dict]:
    """The frozen cards, newest first, each with whether it has been run and what its candidates' verdicts were."""
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = c.execute("SELECT subject_id, MAX(seq) AS seq FROM analysis_records WHERE kind='comparison_card' " + ("AND study_id=? " if study_id else "") + "GROUP BY subject_id ORDER BY seq DESC LIMIT ?",
                         ((study_id, int(limit)) if study_id else (int(limit),))).fetchall()
    out = []
    for r in rows:
        card = get(registry, r["subject_id"])
        verdicts = [a.get("verdict") for a in ((card["results"] or {}).get("candidates") or [])]
        out.append({"card_id": card["card_id"], "family": card["family"], "study_id": card["study_id"], "hypothesis": card["hypothesis"], "frozen_at": card["frozen_at"], "state": card["state"],
                    "primary_metric": card["primary_metric"], "candidates": [c["name"] for c in card["candidates"]], "verdicts": verdicts})
    return out


# --------------------------------------------------------------------------------------------------------------------- one experiment, seen as development series
def _series_from(ds, plan, scores: pd.DataFrame, net: pd.Series, turnover: pd.Series, costs: pd.Series, labels: pd.Series, label_horizon: int) -> dict:
    fwd = forward_return(ds.prices, label_horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)
    ic = ev.rank_ic_wide(scores, fwd).dropna()
    ic_seg = label_days(ic.index, plan.labels())
    own = ev.rank_ic_wide(scores, forward_return(ds.prices, ds.horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)).dropna()
    own_seg = label_days(own.index, plan.labels())
    dev_dates = pd.DatetimeIndex([d for d, s in zip(plan.dates, label_days(pd.DatetimeIndex(plan.dates), plan.labels())) if s == SEGMENT])
    scored = (scores.reindex(dev_dates).notna().sum(axis=1) >= 2) if len(dev_dates) else pd.Series(dtype=bool)
    return {"ic": ic[ic_seg == SEGMENT], "ic_own": own[own_seg == SEGMENT], "net": net[labels == SEGMENT], "turnover": turnover[labels == SEGMENT], "costs": costs[labels == SEGMENT],
            "coverage": float(scored.mean()) if len(scored) else float("nan"), "dev_dates": int(len(dev_dates))}


def _series(exp, label_horizon: int) -> dict:
    return _series_from(exp.dataset, exp.train.plan, exp.scores, exp.net, exp.turnover, exp.costs, exp.labels, label_horizon)


def _series_saved(registry, exp_id: str, label_horizon: int | None = None):
    """The development series of an experiment that was saved earlier (its predictions, returns, turnover and costs), with its configuration and dataset; ``label_horizon`` defaults to its own."""
    from . import experiments as xp

    cfg, ds, train, _ = xp.load_run(registry, exp_id)
    ret = pd.read_parquet(registry.root / "experiments" / exp_id / "returns.parquet")
    return _series_from(ds, train.plan, train.scores_wide(ds), ret["net"], ret["turnover"], ret["costs"], ret["segment"], int(ds.horizon) if label_horizon is None else int(label_horizon)), cfg, ds


def compare_saved(registry, baseline_id: str, candidate_id: str, *, n_boot: int = 1000, seed: int = 0, scope: str = "one pre-declared comparison of two saved experiments") -> dict:
    """The paired comparison of two experiments that already exist (a meta-labelled child with its parent, a combination with a member) on the validation segment, on the dates both have. Both are read
    against the baseline's raw forward return of its own horizon. They must share the assets, the horizon and the lag, or the comparison is refused with the difference named."""
    base_series, bcfg, bds = _series_saved(registry, baseline_id)
    label_h = int(bds.horizon)
    cand_series, ccfg, cds = _series_saved(registry, candidate_id, label_h)
    problems = []
    if list(cds.assets) != list(bds.assets):
        problems.append("the two experiments do not trade the same assets")
    if (int(cds.horizon), int(cds.lag)) != (int(bds.horizon), int(bds.lag)):
        problems.append(f"the horizon and lag differ ({bds.horizon}/{bds.lag} against {cds.horizon}/{cds.lag})")
    if problems:
        return {"eligibility": "not_comparable", "reason": "; ".join(problems), "baseline": baseline_id, "candidate": candidate_id}
    hold = max(int(bcfg["signal"].get("holding_period", 1)), int(ccfg["signal"].get("holding_period", 1)))
    return {"eligibility": "applied", "baseline": baseline_id, "candidate": candidate_id, "segment": SEGMENT, "label_horizon": label_h,
            "metrics": _clean(compare_series(base_series, cand_series, label_horizon=label_h, hold=hold, n_boot=int(n_boot), seed=int(seed), scope=scope)),
            "note": "paired on the validation segment, on the dates both have; the candidate may differ from the baseline in more than one respect, so this is a description of the pair, not a controlled one-change test"}


def _max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _n_features(exp) -> float:
    chosen = [len(d["stable_selection"]["chosen"]) for d in (exp.train.diagnostics or []) if "stable_selection" in d]
    return float(np.mean(chosen)) if chosen else float(len(exp.dataset.features))


def manifest(exp) -> dict:
    """What an arm was actually run on: the data, the fold boundaries and the feature, target and output schemas, hashed."""
    ds, tr = exp.dataset, exp.train
    folds = [[f.get("fold"), f.get("train_start"), f.get("train_end"), f.get("test_start"), f.get("test_end"), bool(f.get("skipped"))] for f in tr.folds]
    return {"data_fingerprint": ds.meta.get("data_fingerprint"), "input_manifest_hash": digest(ds.meta.get("input_manifest") or {}), "assets": list(ds.assets), "first_date": str(ds.dates[0].date()),
            "last_date": str(ds.dates[-1].date()), "fold_manifest": folds, "fold_manifest_hash": digest(folds), "feature_schema_hash": digest(sorted(ds.features)),
            "target_schema_hash": digest({"target": tr.target, "horizon": ds.horizon, "lag": ds.lag}), "output_schema_hash": digest({"model": tr.kind, "task": tr.task}),
            "n_train_rows": [int(f.get("n_train", 0)) for f in tr.folds]}


def selection_stability(exp) -> dict | None:
    from .featsel import selection_stability as stab

    diags = [d.get("stable_selection") for d in (exp.train.diagnostics or [])]
    return stab(diags) if any(diags) else None


# --------------------------------------------------------------------------------------------------------------------- the paired comparison
def compare_series(base: dict, cand: dict, *, label_horizon: int, hold: int, n_boot: int, seed: int, scope: str) -> dict:
    """The paired metrics of one candidate against the baseline, from their development series. Each difference is on the dates both have, with the count dropped from each side."""
    bi, ci = base["ic"], cand["ic"]
    ic_diff = (ci - bi).dropna()
    out: dict = {"ic": {"baseline": float(bi.mean()) if len(bi) else None, "candidate": float(ci.mean()) if len(ci) else None,
                        "difference": inf.hac_mean(ic_diff, horizon=label_horizon, label="candidate minus baseline daily rank IC", scope=scope).as_dict(),
                        "aligned": int(len(ic_diff)), "dropped": {"only_baseline": int(len(bi.index.difference(ci.index))), "only_candidate": int(len(ci.index.difference(bi.index)))},
                        "label": f"rank IC against the raw {label_horizon}-session forward return, the same label for both arms"}}
    bo, co = base["ic_own"], cand["ic_own"]
    out["ic_own_horizon"] = {"baseline": float(bo.mean()) if len(bo) else None, "candidate": float(co.mean()) if len(co) else None,
                             "note": "each arm's IC against its own horizon's return: ICs of different horizons are not commensurable and are not compared"}
    res = inf.paired_difference(cand["net"], base["net"], statistic="sharpe", n_boot=n_boot, seed=seed, horizon=max(1, hold), label="candidate minus baseline net Sharpe", scope=scope)
    sh = lambda s: None if len(s) < inf.MIN_OBS or not s.std(ddof=1) > 0 else float(s.mean() / s.std(ddof=1) * math.sqrt(252.0))              # noqa: E731
    out["net_sharpe"] = {"baseline": sh(base["net"]), "candidate": sh(cand["net"]), "difference": res.as_dict(), "aligned": res.sample.get("aligned"), "dropped": res.sample.get("dropped")}
    for key, label in (("turnover", "mean daily turnover"), ("costs", "mean daily cost")):
        b, c = base[key], cand[key]
        out[key] = {"baseline": float(b.mean()) if len(b) else None, "candidate": float(c.mean()) if len(c) else None, "label": label}
    out["max_drawdown"] = {"baseline": _max_drawdown(base["net"]), "candidate": _max_drawdown(cand["net"])}
    out["coverage"] = {"baseline": base["coverage"], "candidate": cand["coverage"], "label": "share of development dates with at least two scored assets"}
    out["risk_matched"] = _risk_matched(base["net"], cand["net"])
    return out


def _risk_matched(base: pd.Series, cand: pd.Series) -> dict:
    """The candidate's risk against the baseline's on the dates both have, and the candidate's drawdown after scaling it to the baseline's volatility: a risk control that only lowers exposure lowers drawdown
    too, and the matched figure says how much of a drawdown gain is more than that."""
    both = pd.concat([base, cand], axis=1, join="inner").dropna()
    if len(both) < 60 or not (both.iloc[:, 0].std(ddof=1) > 0 and both.iloc[:, 1].std(ddof=1) > 0):
        return {"eligibility": "insufficient", "reason": "fewer than 60 aligned days or a return stream with no variation"}
    b, c = both.iloc[:, 0], both.iloc[:, 1]
    scale = float(b.std(ddof=1) / c.std(ddof=1))
    return {"eligibility": "applied", "aligned_days": int(len(both)), "volatility_ratio": float(1.0 / scale), "max_drawdown_baseline": _max_drawdown(b), "max_drawdown_candidate": _max_drawdown(c),
            "max_drawdown_candidate_at_baseline_volatility": _max_drawdown(c * scale),
            "note": "the last figure is the candidate's drawdown if its returns were scaled to the baseline's volatility: the drawdown gain that is not just a smaller position"}


def _verdict(primary: dict, adj_p: float | None, level: float, min_effect: float) -> tuple[str, str]:
    d = primary["difference"]
    if d["eligibility"] != "applied":
        return "inconclusive", f"the paired difference could not be estimated ({d['eligibility']}: {d.get('reason') or 'insufficient'})"
    v, (lo, hi), p = d["value"], d["interval"], adj_p
    if p is not None and p < level and lo > 0 and v >= min_effect:
        return "supported_on_development", f"the paired difference is {v:+.4g} (interval {lo:+.4g} to {hi:+.4g}), Holm-adjusted p = {p:.3g}: distinguishable from zero on the development segment"
    if p is not None and p < level and hi < 0:
        return "worse", f"the paired difference is {v:+.4g} (interval {lo:+.4g} to {hi:+.4g}), Holm-adjusted p = {p:.3g}: the candidate is worse than the baseline"
    why = "the interval includes zero" if lo <= 0 <= hi else f"the effect {v:+.4g} is below the declared minimum {min_effect:g}" if lo > 0 and v < min_effect else "the adjusted p-value is not below the level"
    return "inconclusive", f"the paired difference is {v:+.4g} (interval {lo:+.4g} to {hi:+.4g}): {why}"


def decide(card: dict, per_candidate: list[dict]) -> list[dict]:
    """Attach the verdict to each compared candidate: Holm over the card's candidates on the primary metric, then the interval and minimum-effect conditions."""
    primary = card["primary_metric"]
    s = card["settings"]
    ps = []
    for c in per_candidate:
        d = (c.get("metrics") or {}).get(primary, {}).get("difference") if c.get("status") == "compared" else None
        ps.append(d["p_value"] if d and d["eligibility"] == "applied" else None)
    inherited = [x.get("p_value") for x in ((card.get("lineage") or {}).get("inherited") or [])]              # the components' own tests belong to the family of a combination
    adj = holm(ps + inherited)[: len(ps)]
    for c, p_adj in zip(per_candidate, adj):
        if c.get("status") != "compared":
            c["verdict"], c["reason"] = "not_compared", c.get("reason", "the candidate was not compared")
            continue
        c["adjusted_p_value"] = p_adj
        c["verdict"], c["reason"] = _verdict(c["metrics"][primary], p_adj, float(s["level"]), float(s["min_effect"]))
    return per_candidate


# --------------------------------------------------------------------------------------------------------------------- running a card
def _run_arm(xp, overrides: dict, study_id: str, *, bundle, config, builder, registry, progress, dataset=None):
    return xp.run_experiment({**overrides, "study": study_id}, bundle=bundle, config=config, builder=builder, registry=registry, progress=progress, stop_after=SEGMENT, explain=False, final_model=False,
                             robustness=False, entry_point="comparison", dataset=dataset)


def run(registry, card_id: str, *, bundle=None, config=None, builder=None, progress=None) -> dict:
    """Run a frozen card once. The baseline and each candidate are development runs of the same study; a second call returns the stored result and runs nothing (another look needs another card)."""
    from . import diagnostics as dx
    from . import experiments as xp

    stored = dx.latest(registry, card_id, "comparison_result")
    if stored is not None:
        return {**get(registry, card_id), "reused": True}
    card = get(registry, card_id)
    progress = progress or (lambda msg: None)
    fam = FAMILIES[card["family"]]
    study = card["study_id"]
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        progress("baseline")
        base = _run_arm(xp, card["baseline"]["overrides"], study, bundle=bundle, config=config, builder=builder, registry=registry, progress=progress)
    except (xp.ExperimentError, ValueError) as exc:
        dx.store(registry, "comparison_result", card_id, {"state": "failed", "results": None, "negative_findings": [{"what": "the baseline could not be run", "detail": str(exc)}],
                                                        "gate_decision": {"gate": "B", "status": "not_evaluated", "reason": "the baseline failed"}, "evidence_artifacts": {}, "observed": None},
                 study_id=study, actor="comparison")
        raise ComparisonError(f"the baseline could not be run: {exc}") from None
    label_h = int(base.dataset.horizon)
    s = card["settings"]
    base_series = _series(base, label_h)
    base_man = manifest(base)
    arms = []
    experiments = {"baseline": base.id}
    for i, cand in enumerate(card["candidates"]):
        progress(f"candidate {i + 1} of {len(card['candidates'])}")
        row = {"name": cand["name"], "config_hash": cand["config_hash"], "changed": cand["changed"]}
        try:
            if _post_model(cand["changed"]):                                                                 # nothing before the position rule changed: the arm re-runs the baseline's own predictions, so the two
                change = {k: v for k, v in cand["change"].items() if k in ("signal", "backtest", "execution")}   # alternatives are evaluated on the same forecasts, the same risk model and the same folds
                exp = xp.rebacktest(registry, base.id, {**change, "name": cand["name"]}, progress=progress, bundle=bundle, config=config, builder=builder)
                row["arm"] = "re-run of the baseline's saved predictions"
            else:
                same_panel = build_key(experiment_config({**cand["overrides"], "study": study})) == build_key(base.cfg)
                exp = _run_arm(xp, {**cand["overrides"], "name": cand["name"]}, study, bundle=bundle, config=config, builder=builder, registry=registry, progress=progress, dataset=base.dataset if same_panel else None)
                row["arm"] = "retrained"
        except (xp.ExperimentError, ValueError) as exc:
            row.update(status="failed", reason=str(exc))
            arms.append(row)
            continue
        experiments[cand["name"]] = exp.id
        man = manifest(exp)
        problems = []
        if (man["data_fingerprint"], man["assets"]) != (base_man["data_fingerprint"], base_man["assets"]):
            problems.append("the data or the assets differ from the baseline's")
        bounds = lambda m: [f[1:5] for f in m["fold_manifest"]]                                                                      # noqa: E731
        if fam["folds"] == "identical" and bounds(man) != bounds(base_man):
            problems.append("the fold boundaries differ from the baseline's")
        row["manifest"] = {k: man[k] for k in ("fold_manifest_hash", "feature_schema_hash", "target_schema_hash", "output_schema_hash", "n_train_rows")}
        if problems:
            row.update(status="not_comparable", reason="; ".join(problems) + ": a comparison is on matched data and folds")
            arms.append(row)
            continue
        hold = int(exp.cfg["signal"].get("holding_period", 1))
        metrics = compare_series(base_series, _series(exp, label_h), label_horizon=label_h, hold=max(hold, int(base.cfg["signal"].get("holding_period", 1))), n_boot=int(s["n_boot"]), seed=int(s["seed"]),
                                 scope=f"{len(card['candidates'])} candidate(s) on this card, Holm-adjusted")
        row.update(status="compared", metrics=metrics, n_features=_n_features(exp), selection_stability=selection_stability(exp), kind=exp.train.kind)
        arms.append(row)
    decide(card, arms)
    parsimony = _apply_parsimony(card, base, arms)
    findings = _negative_findings(card, arms, base_series)
    results = {"baseline": {"name": base.cfg["name"], "experiment_id": base.id, "n_features": _n_features(base), "kind": base.train.kind,
                            "metrics": {"ic": base_series["ic"].mean() if len(base_series["ic"]) else None, "coverage": base_series["coverage"]}}, "candidates": arms, "parsimony": parsimony,
               "segment": SEGMENT, "label_horizon": label_h, "ran_at": started}
    if (card.get("lineage") or {}).get("kind") == "ablation":
        results["ablation"] = _ablation_reading(card, arms)
    record = {"state": "run", "results": _clean(results), "negative_findings": findings, "gate_decision": {"gate": "B", "status": "not_evaluated",
              "reason": "a development-segment comparison is not a frozen final evaluation: gate B reads the evaluation plan, and nothing here promotes a strategy"},
              "evidence_artifacts": {"experiments": experiments, "card_id": card_id}, "observed": {"baseline": _clean(base_man)}}
    dx.store(registry, "comparison_result", card_id, record, study_id=study, actor="comparison")
    return {**get(registry, card_id), "reused": False}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _apply_parsimony(card: dict, base, arms: list[dict]) -> dict | None:
    rule = (card.get("selection_rule") or {}).get("parsimony")
    if not rule:
        return None
    primary = card["primary_metric"]

    def point(m):
        return m[primary]["candidate"]

    def se(m):
        d = m[primary]["difference"]
        return None if d["eligibility"] != "applied" else (d["extras"].get("se") if primary == "ic" else d["extras"].get("bootstrap_se"))

    ok = [a for a in arms if a.get("status") == "compared"]
    if not ok:
        return {"eligibility": "insufficient", "reason": "no candidate was compared"}
    cands = [{"id": "baseline", "score": ok[0]["metrics"][primary]["baseline"], "se": None, "complexity": pa.complexity_of(base.train.kind, _n_features(base))}]
    for a in ok:
        paired = se(a["metrics"])
        cands.append({"id": a["name"], "score": point(a["metrics"]), "se": None, "complexity": pa.complexity_of(a["kind"], a["n_features"]), "se_diff": {"baseline": paired} if paired is not None else {}})
    out = pa.choose(cands, rule)
    out["note_on_scope"] = "each candidate is compared with the baseline by the paired standard error of their difference; two candidates have no paired error here, so a gap between them is not called indistinguishable"
    return _clean(out)


def _negative_findings(card: dict, arms: list[dict], base_series: dict) -> list[dict]:
    """Everything that did not go the candidate's way, kept in the record: verdicts that are not 'supported', failures, runs that could not be compared, thin coverage and cost or turnover increases."""
    out = []
    mincov = float(card["minimum_coverage"])
    for a in arms:
        if a["status"] != "compared":
            out.append({"candidate": a["name"], "what": a["status"], "detail": a.get("reason")})
            continue
        if a["verdict"] != "supported_on_development":
            out.append({"candidate": a["name"], "what": a["verdict"], "detail": a["reason"]})
        cov = a["metrics"]["coverage"]["candidate"]
        if cov is not None and not math.isnan(cov) and cov < mincov:
            out.append({"candidate": a["name"], "what": "coverage_below_minimum", "detail": f"{cov:.2f} of development dates are scored; the card declares at least {mincov:.2f}"})
        t = a["metrics"]["turnover"]
        if t["baseline"] is not None and t["candidate"] is not None and t["candidate"] > 1.05 * t["baseline"]:
            out.append({"candidate": a["name"], "what": "higher_turnover", "detail": f"mean daily turnover {t['candidate']:.4g} against {t['baseline']:.4g}"})
        rm = a["metrics"].get("risk_matched") or {}
        if rm.get("eligibility") == "applied" and rm["max_drawdown_candidate"] > rm["max_drawdown_baseline"] and rm["max_drawdown_candidate_at_baseline_volatility"] <= rm["max_drawdown_baseline"]:
            out.append({"candidate": a["name"], "what": "drawdown_gain_is_lower_exposure",
                        "detail": f"the drawdown is {rm['max_drawdown_candidate']:.3f} against {rm['max_drawdown_baseline']:.3f}, but at the baseline's volatility it would be {rm['max_drawdown_candidate_at_baseline_volatility']:.3f}: "
                                  "the gain is a smaller position, not better timing"})
        d = a["metrics"]["net_sharpe"]["difference"]
        if card["primary_metric"] != "net_sharpe" and d["eligibility"] == "applied" and d["value"] is not None and d["value"] < 0:
            out.append({"candidate": a["name"], "what": "net_sharpe_lower", "detail": f"the net Sharpe difference is {d['value']:+.3g}: a better rank IC did not become a better net result"})
    return out


# --------------------------------------------------------------------------------------------------------------------- combinations, ablations and the development-selected baseline
_MISSING = object()


def _get_path(cfg: dict, path: str):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _nest(path: str, value) -> dict:
    out: dict = {}
    cur = out
    parts = path.split(".")
    for part in parts[:-1]:
        cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
    return out


def _components(registry, parents: list[dict]) -> tuple[dict, list[dict]]:
    """Check the components of a combination: each is a candidate of a card that was run, compared on its own, against one and the same baseline. Returns that baseline's overrides and the components."""
    if not isinstance(parents, list) or len(parents) < 2 or any(not isinstance(p, dict) or not p.get("card_id") or not p.get("candidate") for p in parents):
        raise ComparisonError("a combination needs at least two components, each {card_id, candidate}")
    seen, comps, baseline, base_hash = set(), [], None, None
    for p in parents:
        key = (p["card_id"], p["candidate"])
        if key in seen:
            raise ComparisonError(f"component {p['candidate']!r} of {p['card_id']} is listed twice")
        seen.add(key)
        card = get(registry, p["card_id"])
        if card["state"] != "run" or not card.get("results"):
            raise ComparisonError(f"card {p['card_id']} has not been run: each enhancement is compared on its own before any combination is tried")
        cand = next((c for c in card["candidates"] if c["name"] == p["candidate"]), None)
        arm = next((a for a in card["results"]["candidates"] if a["name"] == p["candidate"]), None)
        if cand is None or arm is None:
            raise ComparisonError(f"card {p['card_id']} has no candidate named {p['candidate']!r}")
        if arm.get("status") != "compared":
            raise ComparisonError(f"component {p['candidate']!r} of {p['card_id']} was {arm.get('status')}: only components that were compared can be combined")
        if base_hash is None:
            base_hash, baseline = card["baseline_version"], card["baseline"]["overrides"]
        elif card["baseline_version"] != base_hash:
            raise ComparisonError("the components were compared against different baselines: a combination is evaluated against one baseline")
        d = ((arm.get("metrics") or {}).get(card["primary_metric"]) or {}).get("difference") or {}
        change = {k: v for k, v in cand["change"].items() if k not in IGNORED}
        comps.append({"card_id": p["card_id"], "candidate": p["candidate"], "family": card["family"], "change": change, "verdict": arm.get("verdict"), "adjusted_p_value": arm.get("adjusted_p_value"),
                      "metric": card["primary_metric"], "p_value": d.get("p_value"), "study": card["study_id"]})
    return baseline, comps


def _merge_components(comps: list[dict]) -> dict:
    """The components' changes laid together. Two components that set the same setting to different values cannot be combined as they are, and the refusal names them."""
    seen, merged = {}, {}
    for c in comps:
        for path, val in _leaves(c["change"]):
            if path in seen and _canon(seen[path][1]) != _canon(val):
                raise ComparisonError(f"components {seen[path][0]!r} and {c['candidate']!r} both set {path}, to different values: they cannot be combined as they are")
            seen[path] = (c["candidate"], val)
        merged = merge(merged, c["change"])
    return merged


def _lineage(kind: str, comps: list[dict]) -> dict:
    return {"kind": kind, "components": [{k: c[k] for k in ("card_id", "candidate", "family", "verdict", "adjusted_p_value")} for c in comps],
            "inherited": [{k: c[k] for k in ("card_id", "candidate", "metric", "p_value")} for c in comps],
            "rule": "enhancements are compared one at a time first; a combination is another logged candidate whose multiple-testing family includes its components' own tests, not a free reuse of the best findings"}


def combination_card(registry, parents: list[dict], *, hypothesis: str, one_change_description: str, study_id: str | None = None, **kw) -> dict:
    """The card that tests several components together against the baseline they were each compared against. Every component must already have been run on its own (one change at a time comes first);
    the candidate is the components' changes laid together; its Holm family includes the components' tests. ``kw`` are the optional settings of ``new_card``."""
    baseline, comps = _components(registry, parents)
    change = _merge_components(comps)
    allowed = tuple(sorted({a for c in comps for a in FAMILIES[c["family"]]["allowed"]}))
    return new_card(hypothesis, "combination", baseline, [{**change, "name": f"combination of {len(comps)}"}], study_id=study_id or comps[0]["study"], one_change_description=one_change_description,
                    lineage=_lineage("combination", comps), allowed=allowed, **kw)


def ablation_card(registry, parents: list[dict], *, hypothesis: str | None = None, one_change_description: str | None = None, study_id: str | None = None, **kw) -> dict:
    """The leave-one-out card of a combination: the baseline is the full combination and each candidate removes one component (its settings return to the original baseline's), so a verdict of 'worse'
    says the removed component helps. Its Holm family includes the components' tests like the combination's."""
    from .config import experiment_config

    baseline, comps = _components(registry, parents)
    change = _merge_components(comps)
    full = merge(baseline, change)
    base_cfg = experiment_config(baseline)
    cands = []
    for c in comps:
        rev: dict = {}
        for path, _ in _leaves(c["change"]):
            parts = path.split(".")
            if "params" in parts:                                                                        # a model's parameters are replaced as a whole, so the whole mapping goes back
                parts = parts[: parts.index("params") + 1]
            sub = ".".join(parts)
            val = _get_path(base_cfg, sub)
            rev = merge(rev, _nest(sub, None if val is _MISSING else val))                                # a setting the baseline never had is switched off (null), not left on
        rev["name"] = f"without {c['family']}:{c['candidate']}@{c['card_id'][4:10]}"[:80]                      # two cards can each have a candidate of the same generated name: the card id tells them apart
        cands.append(rev)
    allowed = tuple(sorted({a for c in comps for a in FAMILIES[c["family"]]["allowed"]}))
    return new_card(hypothesis or "each component of the combination contributes: without it the combination is worse", "combination", full, cands, study_id=study_id or comps[0]["study"],
                    one_change_description=one_change_description or "remove one component of the combination at a time", lineage=_lineage("ablation", comps), allowed=allowed, **kw)


def _ablation_reading(card: dict, arms: list[dict]) -> list[dict]:
    """What removing each component did, in words: worse means the component helps, supported means it hurts, anything else means no distinguishable contribution."""
    words = {"worse": "the component helps: without it the combination is worse", "supported_on_development": "the component hurts: without it the combination is better",
             "inconclusive": "no distinguishable contribution on the development segment", "not_compared": "not evaluated"}
    comps = (card.get("lineage") or {}).get("components") or []
    return [{"component": comp["candidate"], "card_id": comp["card_id"], "family": comp["family"], "verdict_of_removing_it": arm.get("verdict"), "reading": words.get(arm.get("verdict"), "not evaluated"),
             "adjusted_p_value": arm.get("adjusted_p_value")} for comp, arm in zip(comps, arms)]


def select_baseline(registry, base: dict, members: list[tuple[str, dict]], *, study_id: str, bundle=None, config=None, builder=None, progress=None, metric: str = "ic") -> dict:
    """The single model that development would have chosen: each member run alone on the baseline's data, folds, position rule and costs, and the best by the card's primary metric on the validation
    segment. Every run is a counted development look of the study. The winner is the best of ``N``: an optimistic estimate that an ensemble must beat, which makes a gain over it conservative."""
    from . import experiments as xp
    from .evaluation import sharpe_of

    if not 2 <= len(members) <= 8:
        raise ComparisonError("choose between two and eight members")
    if metric not in PRIMARY_METRICS:
        raise ComparisonError(f"the metric must be one of {list(PRIMARY_METRICS)}")
    rows = []
    for i, (kind, params) in enumerate(members):
        if progress:
            progress(f"member {i + 1} of {len(members)}: {kind}")
        exp = _run_arm(xp, {**merge(base, {"model": {"kind": kind, "params": params}}), "name": f"member-{kind}"[:80]}, study_id, bundle=bundle, config=config, builder=builder, registry=registry, progress=progress)
        ser = _series(exp, int(exp.dataset.horizon))
        score = float(ser["ic"].mean()) if metric == "ic" and len(ser["ic"]) else (float(sharpe_of(ser["net"].dropna())) if metric == "net_sharpe" else float("nan"))
        rows.append({"kind": kind, "params": params, "score": score if np.isfinite(score) else None, "experiment_id": exp.id, "development_days": int(len(ser["net"]))})
    ranked = [r for r in rows if r["score"] is not None]
    if not ranked:
        raise ComparisonError("no member produced a score on the development segment")
    best = max(ranked, key=lambda r: r["score"])
    return {"metric": metric, "segment": SEGMENT, "members": rows, "selected": {k: best[k] for k in ("kind", "params", "score", "experiment_id")}, "n_candidates": len(members),
            "note": f"the best of {len(members)} on the {SEGMENT} segment: an optimistic estimate of that model, so an ensemble that beats it has beaten a baseline that was selected in its favour"}
