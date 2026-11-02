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
             robustness_grid: dict | None = None, final_evaluation_policy: str | None = None, compute_budget: dict | None = None, n_boot: int = 1000, seed: int = 0, base_defaults: dict | None = None) -> dict:
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
    bcfg = experiment_config({**baseline, "study": study_id}, defaults=base_defaults)
    cands = []
    for i, change in enumerate(candidates):
        over = merge(baseline, change)                                                                    # a candidate is the baseline with ONE declared change laid over it
        ccfg = experiment_config({**over, "study": study_id}, defaults=base_defaults)
        paths = diff_paths(bcfg, ccfg)
        stray = [p for p in paths if not _allowed(p, fam["allowed"])]
        if not paths:
            raise ComparisonError(f"candidate {i + 1} is the baseline: it changes nothing")
        if stray:
            raise ComparisonError(f"candidate {i + 1} differs from the baseline in {stray}, which the {family} family does not allow ({list(fam['allowed'])}): a comparison changes one thing, on matched assumptions")
        cands.append({"name": ccfg["name"], "change": change, "overrides": over, "config_hash": config_hash(ccfg), "changed": paths})
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
PRESETS = {"target": ("excess_ret", "sector_ret", "resid_ret"), "horizon": (1, 5, 10, 20), "feature": ("stable_select", *(f"group:{g}" for g in FEATURE_GROUPS)), "portfolio": ("cost_aware",)}


def preset_changes(family: str, baseline: dict, choices: list) -> list[dict]:
    """The standard candidates of a family as changes to ``baseline``: another kind of target at the baseline's horizon; other horizons, each with its matching holding period; the baseline's linear model behind a
    stability selection; the cost-aware optimiser in place of the baseline's position rule. A choice the baseline already is (its own horizon, say) is left out."""
    if family not in PRESETS:
        raise ComparisonError(f"{family} has no presets; give the changes yourself")
    cfg = experiment_config(baseline)
    out = []
    for choice in choices:
        if choice not in PRESETS[family]:
            raise ComparisonError(f"{choice!r} is not a {family} preset; the presets are {list(PRESETS[family])}")
        if family == "target":
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
def _series(exp, label_horizon: int) -> dict:
    ds, plan = exp.dataset, exp.train.plan
    fwd = forward_return(ds.prices, label_horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)
    ic = ev.rank_ic_wide(exp.scores, fwd).dropna()
    ic_seg = label_days(ic.index, plan.labels())
    net_seg = exp.labels
    own = ev.rank_ic_wide(exp.scores, forward_return(ds.prices, ds.horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)).dropna()
    own_seg = label_days(own.index, plan.labels())
    dev_dates = pd.DatetimeIndex([d for d, s in zip(plan.dates, label_days(pd.DatetimeIndex(plan.dates), plan.labels())) if s == SEGMENT])
    scored = (exp.scores.reindex(dev_dates).notna().sum(axis=1) >= 2) if len(dev_dates) else pd.Series(dtype=bool)
    return {"ic": ic[ic_seg == SEGMENT], "ic_own": own[own_seg == SEGMENT], "net": exp.net[net_seg == SEGMENT], "turnover": exp.turnover[net_seg == SEGMENT], "costs": exp.costs[net_seg == SEGMENT],
            "coverage": float(scored.mean()) if len(scored) else float("nan"), "dev_dates": int(len(dev_dates))}


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
    return out


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
    adj = holm(ps)
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
            same_panel = build_key(experiment_config({**cand["overrides"], "study": study})) == build_key(base.cfg)
            exp = _run_arm(xp, cand["overrides"], study, bundle=bundle, config=config, builder=builder, registry=registry, progress=progress, dataset=base.dataset if same_panel else None)
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
        d = a["metrics"]["net_sharpe"]["difference"]
        if card["primary_metric"] != "net_sharpe" and d["eligibility"] == "applied" and d["value"] is not None and d["value"] < 0:
            out.append({"candidate": a["name"], "what": "net_sharpe_lower", "detail": f"the net Sharpe difference is {d['value']:+.3g}: a better rank IC did not become a better net result"})
    return out
