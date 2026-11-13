"""A predeclared simplicity tie-break for model selection (spec v3, Release 6, P2-17).

Picking the best-scoring of several candidates rewards luck as much as skill, and the more complex candidates have more ways to be lucky. The rule here is the one declared *before* the scores are seen:

* ``one_se`` - take the best candidate, then the simplest one whose score is not distinguishable from it: a candidate is kept if ``best - score <= k * se_of_the_gap``. ``se_of_the_gap`` is the standard error of
  the paired difference when the caller has one, and otherwise ``sqrt(se_best^2 + se_candidate^2)``, which treats the two scores as independent and so can only overstate the uncertainty of the gap (it
  favours the simpler model more than the paired figure would). With no standard error at all the tie-break cannot be applied and the best score wins, with that said.
* ``penalty`` - maximise ``score - lambda * complexity`` with one scalar complexity measure.

Complexity is declared too: ``n_features``, ``family_rank`` (linear 1, state-space 2, trees 4, networks 6, ensembles 7, agents 8) or ``lexicographic`` (rank, then features). The rule belongs in the evaluation
plan's ``decision_policy.parsimony``; ``choose`` reports whether the plan that was frozen before the results contains exactly the rule that was used, and says plainly when it was only supplied with the data.
"""

from __future__ import annotations

import math

import numpy as np

VERSION = "parsimony-v1"
RULES = ("one_se", "penalty")
COMPLEXITY = ("n_features", "family_rank", "lexicographic")
FAMILY_RANK = {"linear": 1, "ridge": 1, "lasso": 1, "elasticnet": 1, "logistic": 1, "ar": 1, "stable_select": 2, "kalman": 2, "online_ridge": 2, "online_logistic": 2, "regime_ridge": 2, "rff_ridge": 3, "hmm_regime": 3,
               "markov_switching": 3, "random_forest": 4, "hist_gb": 4, "xgboost": 4, "lightgbm": 4, "catboost": 4, "lgbm_ranker": 4, "xgb_ranker": 4, "lgbm_quantile": 4, "ngboost": 4, "autoencoder": 5, "vae": 5,
               "lstm": 6, "gru": 6, "tcn": 6, "patchtst": 6, "nbeats": 6, "nhits": 6, "itransformer": 6, "dlinear": 5, "tft": 6, "gcn": 6, "gat": 6, "model_ensemble": 7, "ppo": 8, "sac": 8, "recurrent_ppo": 8}


class ParsimonyError(ValueError):
    """A rule that cannot be applied as declared."""


def family_rank(kind: str) -> int:
    return FAMILY_RANK.get(kind, 5)


def complexity_of(kind: str, n_features: float) -> dict:
    """The complexity of a candidate: its family's rank and the number of features it uses (the mean number its folds chose, for a model that selects)."""
    return {"family_rank": family_rank(kind), "n_features": float(n_features)}


def validate_rule(rule: dict) -> dict:
    name = rule.get("rule")
    if name not in RULES:
        raise ParsimonyError(f"the rule must be one of {list(RULES)}")
    if rule.get("complexity") not in COMPLEXITY:
        raise ParsimonyError(f"complexity must be declared as one of {list(COMPLEXITY)}")
    if name == "one_se" and not float(rule.get("k", -1.0)) >= 0.0:
        raise ParsimonyError("one_se needs k >= 0 (the number of standard errors of the gap that still counts as 'not distinguishable')")
    if name == "penalty":
        if not float(rule.get("lambda", -1.0)) >= 0.0:
            raise ParsimonyError("penalty needs lambda >= 0 (score units per unit of complexity)")
        if rule["complexity"] == "lexicographic":
            raise ParsimonyError("a penalty needs one scalar complexity measure: n_features or family_rank")
    return rule


def _key(c: dict, how: str) -> tuple:
    cx = c["complexity"]
    if how == "n_features":
        return (cx["n_features"],)
    if how == "family_rank":
        return (cx["family_rank"],)
    return (cx["family_rank"], cx["n_features"])


def _scalar(c: dict, how: str) -> float:
    return float(c["complexity"][how])


def choose(candidates: list[dict], rule: dict, plan: dict | None = None) -> dict:
    """Apply a declared rule. ``candidates``: ``{"id", "score" (higher is better), "se" (of the score, or None), "complexity": {"family_rank", "n_features"}, "se_diff": {other_id: se of the paired gap}}``."""
    rule = validate_rule(dict(rule))
    usable = [c for c in candidates if c.get("score") is not None and math.isfinite(float(c["score"]))]
    skipped = [{"id": c["id"], "reason": "no finite score"} for c in candidates if c not in usable]
    declared = None
    if plan is not None:
        pol = ((plan.get("content") or {}).get("decision_policy") or {}).get("parsimony")
        declared = bool(plan.get("state") in ("frozen", "consumed") and pol == rule)
    base = {"version": VERSION, "rule": rule, "predeclared": declared, "skipped": skipped,
            "predeclared_note": ("the rule is in the frozen evaluation plan" if declared else "the rule was supplied with the results: it is not evidence that it was chosen before them") if plan is not None
            else "no evaluation plan was given, so nothing shows the rule was fixed before the scores were seen"}
    if not usable:
        return {**base, "eligibility": "insufficient", "winner": None, "best_by_score": None, "table": [], "reason": "no candidate has a finite score"}
    best = max(usable, key=lambda c: float(c["score"]))                                              # the first of equal scores in the order given
    how = rule["complexity"]
    table = []
    if rule["rule"] == "one_se":
        eligible = []
        for c in usable:
            gap = float(best["score"]) - float(c["score"])
            row = {"id": c["id"], "score": float(c["score"]), "se": c.get("se"), "complexity": c["complexity"], "gap_to_best": gap}
            if c is best:
                row.update(se_of_gap=None, eligible=True, reason="the best score")
            else:
                sd = (best.get("se_diff") or {}).get(c["id"], (c.get("se_diff") or {}).get(best["id"]))
                how_sd = "paired" if sd is not None else None
                if sd is None and best.get("se") is not None and c.get("se") is not None:
                    sd, how_sd = math.hypot(float(best["se"]), float(c["se"])), "independent (sqrt of the sum of squared standard errors)"
                if sd is None or not math.isfinite(float(sd)):
                    row.update(se_of_gap=None, eligible=False, reason="no standard error for the gap: it cannot be called indistinguishable from the best")
                else:
                    ok = gap <= float(rule["k"]) * float(sd)
                    row.update(se_of_gap=float(sd), gap_in_se=gap / float(sd) if sd > 0 else None, eligible=bool(ok), how=how_sd,
                               reason=("within " + f"{rule['k']:g} standard error(s) of the best: not distinguishable" if ok else f"{gap / float(sd):.2f} standard errors below the best" if sd > 0 else "below the best with no spread"))
            if row["eligible"]:
                eligible.append(c)
            table.append(row)
        winner = min(eligible, key=lambda c: (_key(c, how), -float(c["score"]), str(c["id"])))
        applied = len(eligible) > 1
        note = None if applied else "no other candidate was indistinguishable from the best (or none had a standard error): the best score wins"
    else:
        lam = float(rule["lambda"])
        for c in usable:
            adj = float(c["score"]) - lam * _scalar(c, how)
            table.append({"id": c["id"], "score": float(c["score"]), "se": c.get("se"), "complexity": c["complexity"], "adjusted": adj, "eligible": True, "reason": f"score - {lam:g} x {how}"})
        top = max(table, key=lambda r: (r["adjusted"], -_scalar({"complexity": r["complexity"]}, how)))
        winner = next(c for c in usable if c["id"] == top["id"])
        applied, note = True, None
    return {**base, "eligibility": "applied", "winner": winner["id"], "best_by_score": best["id"], "changed_choice": winner["id"] != best["id"], "tie_break_applied": bool(applied), "table": table, "note": note,
            "uncertainty": "the standard errors are those of the development scores; a simpler winner is the model whose extra benefit is within them, not a model shown to be as good"}


def simulate_selection(true_scores: list[float], ses: list[float], rule: dict, draws: int = 2000, seed: int = 0) -> dict:
    """How often the rule picks each candidate when the true scores are given and the observed ones carry independent noise of the stated standard errors. Candidates are ordered simplest first. The frequencies
    are Monte Carlo estimates and carry their own error, which the caller should report."""
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(true_scores))
    best_wins = np.zeros(len(true_scores))
    for _ in range(draws):
        obs = np.asarray(true_scores) + rng.normal(0.0, ses)
        cand = [{"id": str(i), "score": float(obs[i]), "se": float(ses[i]), "complexity": {"family_rank": 1, "n_features": float(i + 1)}} for i in range(len(true_scores))]
        out = choose(cand, rule)
        wins[int(out["winner"])] += 1
        best_wins[int(out["best_by_score"])] += 1
    return {"picked": (wins / draws).tolist(), "best_score_picked": (best_wins / draws).tolist(), "draws": draws, "monte_carlo_se_at_half": float(np.sqrt(0.25 / draws))}
