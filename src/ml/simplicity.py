"""Complexity-aware selection: the simplest eligible model that is not meaningfully worse than the highest-ranked one (spec v3, Release 6 and 8 follow-up, P2-17).

Picking the best-scoring candidate rewards luck as much as skill, and the more complex a candidate, the more ways it has to be lucky. ``choose_simplest`` takes the candidates that survived the eligibility
rules, already ranked, and prefers the simplest one whose performance is **not meaningfully worse than the highest-ranked candidate**. There is no fixed penalty in Sharpe units: a penalty of ``0.1`` per
feature is a number someone made up, and it trades performance for simplicity at a rate nothing in the data supports. What the data does support is how far two return streams could be apart by chance, so
"not meaningfully worse" is decided from the **paired** difference of the two candidates on the dates both have, with a **dependence-aware** uncertainty estimate (a joint stationary bootstrap that resamples
the same blocks in both series, with the block length chosen from the series' own dependence), and never from two unrelated standard errors.

Two rules, and the difference between them is the point of this module:

* ``one_se`` (the default) is a **selection heuristic**. A candidate is kept when the point gap to the best is at most ``k`` standard errors of the paired gap. Failing to find a difference is not finding
  that there is none: a short or noisy sample has a wide standard error and admits more candidates, so the heuristic prefers simplicity *more* when the evidence is weak. Its output says "not
  distinguishable by this rule" and never "equivalent".
* ``non_inferiority`` is a **formal test** and needs a margin ``delta`` that the policy declares before the results (in the primary metric's own units, for example 0.25 of annualised Sharpe). Each candidate is
  tested against H0: ``best - candidate >= delta`` with the bootstrap distribution of the paired gap; the p-values are Holm-adjusted over the comparisons, and a candidate is *non-inferior within the margin* when
  H0 is rejected at the level. The two-sided counterpart (the candidate is also not better by more than the margin) is reported as ``equivalent_within_margin``, from the same draws. The margin is an
  assumption about what matters economically, not an estimate, and the claim holds on the validation data only.

Complexity is a vector, not a number: the family's rank, the number of features, the packages the model needs, the cost of training it, how long it takes to predict and how stable its performance was across
the validation sub-periods. Dimensions are compared in a declared order (``priority``); costs and instabilities are compared on a logarithmic grid so that a 10% difference in wall-clock time is a tie, not a
winner. A dimension that was not measured is *not simple*: unmeasured sorts last, never as zero. The best candidate's selection as the maximum of N is itself optimistic on the same data, so the paired gaps
are biased upward, which makes the tolerance conservative (a simpler model has to be close to an overstated best).
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..stats.inference import bootstrap_indices, optimal_block_length
from . import inference as inf
from . import parsimony as pa

VERSION = "simplicity-v1"
RULES = ("one_se", "non_inferiority")
DIMENSIONS = ("model_complexity", "n_features", "dependencies", "training_cost", "inference_latency", "stability")
COUNTED = ("model_complexity", "n_features", "dependencies")                  # compared exactly; the others on a logarithmic grid of ratio 1 + cost_tolerance
PAIRED = {"net_sharpe": "sharpe", "net_ann_return": "ann_return"}             # primary metrics whose paired difference has a bootstrap here (the daily IC has its own HAC route)
PERIODS = inf.periods_per_year("daily")


class SimplicityError(ValueError):
    """A rule that cannot be applied as declared."""


@dataclass(frozen=True)
class SimplicityPolicy:
    """The declared rule. Every default is a convention that can be changed; none is tuned to a data set."""

    enabled: bool = True
    rule: str = "one_se"                   # one_se (a heuristic) | non_inferiority (a formal test: needs ``margin``)
    k: float = 1.0                         # one_se: the gap to the best, in standard errors of the paired gap, that still counts as 'not distinguishable'
    margin: float | None = None            # non_inferiority: the largest loss of the primary metric that is accepted, in its own units, declared before the results
    level: float = 0.05                    # non_inferiority: the one-sided level (Holm-adjusted over the comparisons with the best)
    priority: tuple = DIMENSIONS           # the order in which the complexity dimensions decide
    cost_tolerance: float = 0.25           # relative difference under which two training costs, latencies or instabilities are a tie
    n_boot: int = 1000                     # stationary-bootstrap replications of the paired gap
    seed: int = 0
    require_paired: bool = False           # True: a candidate whose gap has only unpaired errors cannot be called 'not meaningfully worse'

    def __post_init__(self):
        if self.rule not in RULES:
            raise SimplicityError(f"the rule must be one of {list(RULES)}")
        if not float(self.k) >= 0.0:
            raise SimplicityError("k >= 0 (the number of standard errors of the gap that still counts as 'not distinguishable')")
        if self.rule == "non_inferiority" and (self.margin is None or not float(self.margin) > 0.0):
            raise SimplicityError("non_inferiority is a formal test and needs a margin > 0, declared before the results (in the units of the primary metric)")
        if not 0.0 < float(self.level) < 1.0 or int(self.n_boot) < inf.MIN_BOOT or not float(self.cost_tolerance) >= 0.0:
            raise SimplicityError(f"level in (0, 1), n_boot >= {inf.MIN_BOOT} and cost_tolerance >= 0")
        bad = [d for d in self.priority if d not in DIMENSIONS]
        if bad or not self.priority or len(set(self.priority)) != len(self.priority):
            raise SimplicityError(f"priority is a non-empty list of distinct dimensions from {list(DIMENSIONS)}")

    def hash(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:16]


def policy_from(raw: dict | None) -> SimplicityPolicy:
    """A policy from a plain mapping; unknown keys are refused."""
    raw = dict(raw or {})
    allowed = set(SimplicityPolicy.__dataclass_fields__)
    extra = sorted(set(raw) - allowed)
    if extra:
        raise SimplicityError(f"a simplicity policy has {sorted(allowed)}, not {extra}")
    if "priority" in raw:
        raw["priority"] = tuple(raw["priority"])
    return SimplicityPolicy(**raw)


# ------------------------------------------------------------------------------------------------------------------------------ the complexity vector
@functools.lru_cache(maxsize=1)
def _needs() -> dict:
    from .models import MODEL_INFO

    return {m.kind: tuple(sorted(set(([m.optional_package] if m.optional_package else []) + list(m.needs)))) for m in MODEL_INFO}


def dependencies_of(kind: str, params: dict | None = None) -> list[str]:
    """The optional packages a model needs (an ensemble needs those of its members)."""
    needs = set(_needs().get(kind, ()))
    if kind == "model_ensemble":
        for token in str((params or {}).get("members") or "").split(","):
            needs |= set(_needs().get(token.strip().split("[", 1)[0], ()))
    return sorted(needs)


def _finite(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


def complexity_vector(kind: str, params: dict | None = None, *, n_features=None, n_params=None, train_seconds=None, inference_ms=None, instability=None) -> dict:
    """What a candidate costs to keep: ``None`` means *not measured*, which is never read as free."""
    deps = dependencies_of(kind, params)
    return {"model_complexity": pa.family_rank(kind), "n_features": _finite(n_features), "dependencies": len(deps), "dependency_names": deps, "training_cost": _finite(train_seconds),
            "inference_latency": _finite(inference_ms), "stability": _finite(instability), "n_params": _finite(n_params)}


def _bucket(v, tol: float) -> float:
    if v is None:
        return math.inf                                                      # not measured: not simple
    if v <= 0:
        return -math.inf
    return float(math.floor(math.log(v) / math.log1p(tol))) if tol > 0 else float(v)


def order_key(cx: dict, policy: SimplicityPolicy) -> tuple:
    """Where a complexity vector sorts: the declared dimensions in order, smaller first, unmeasured last. Costs sit on a logarithmic grid, so the order is total (no intransitive near-ties)."""
    key = []
    for d in policy.priority:
        v = cx.get(d)
        key.append(math.inf if v is None else float(v)) if d in COUNTED else key.append(_bucket(v, float(policy.cost_tolerance)))
    return tuple(key)


# ------------------------------------------------------------------------------------------------------------------------------ the paired gap
def _stat(statistic: str):
    if statistic == "sharpe":
        return lambda u: inf._sharpe_per_period(u) * math.sqrt(PERIODS)        # noqa: SLF001
    return lambda u: float(u.mean()) * PERIODS


def paired_gap(best: pd.Series, cand: pd.Series, *, statistic: str = "sharpe", n_boot: int = 1000, seed: int = 0, horizon: int = 1) -> dict:
    """``statistic(best) - statistic(candidate)`` on the dates both have, and its distribution in a joint stationary bootstrap: the same resampled blocks are applied to both series, so their dependence on each
    other and in time is kept, and the block length comes from the dependence of the difference itself (Politis and White), never below the forecast horizon. ``draws`` is the bootstrap distribution."""
    if statistic not in ("sharpe", "ann_return"):
        raise SimplicityError("the paired bootstrap is for sharpe and ann_return")
    a, b = pd.Series(best).astype(float), pd.Series(cand).astype(float)
    both = pd.concat([a, b], axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    dropped = {"only_best": int(len(a.dropna()) - len(both)), "only_candidate": int(len(b.dropna()) - len(both))}
    if len(both) < inf.MIN_OBS:
        return {"how": "unavailable", "reason": f"{len(both)} common dates: fewer than {inf.MIN_OBS}", "aligned": int(len(both)), "dropped": dropped}
    x, y = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    f = _stat(statistic)
    est = f(x) - f(y)
    block = max(float(optimal_block_length(x - y, kind="stationary")), float(horizon))
    rng = np.random.default_rng(seed)
    draws = np.array([f(x[idx]) - f(y[idx]) for idx in (bootstrap_indices(len(both), "stationary", block, rng) for _ in range(int(n_boot)))])
    draws = draws[np.isfinite(draws)]
    if len(draws) < inf.MIN_BOOT or not math.isfinite(est):
        return {"how": "unavailable", "reason": f"{len(draws)} usable bootstrap replications: fewer than {inf.MIN_BOOT}", "aligned": int(len(both)), "dropped": dropped}
    return {"how": "paired_bootstrap", "estimate": float(est), "se": float(draws.std(ddof=1)), "draws": draws, "aligned": int(len(both)), "dropped": dropped, "mean_block": block,
            "correlation": float(np.corrcoef(x, y)[0, 1])}


def _gap(best: dict, cand: dict, metric: str, policy: SimplicityPolicy, horizon: int) -> dict:
    """The gap between the best and one candidate, as paired as the inputs allow, with how it was obtained."""
    stat = PAIRED.get(metric)
    rb, rc = best.get("returns"), cand.get("returns")
    if stat is not None and rb is not None and rc is not None:
        return paired_gap(rb, rc, statistic=stat, n_boot=policy.n_boot, seed=policy.seed, horizon=horizon)
    ib, ic = best.get("ic"), cand.get("ic")
    if metric == "ic" and ib is not None and ic is not None:                                              # the daily rank IC: a mean, with a Newey-West error that allows for overlapping labels
        d = (pd.Series(ib).astype(float) - pd.Series(ic).astype(float)).dropna()
        res = inf.hac_mean(d, horizon=horizon, label="best minus candidate daily rank IC")
        if res.eligibility != "applied":
            return {"how": "unavailable", "reason": f"the paired IC gap could not be estimated ({res.eligibility}: {res.reason})", "aligned": int(len(d))}
        return {"how": "paired_hac", "estimate": float(res.value), "se": float(res.extras["se"]), "aligned": int(len(d)), "dropped": {}}
    seb, sec = best.get("score_se"), cand.get("score_se")
    if seb is not None and sec is not None:                                                               # no paired series: two errors taken as independent, which overstates the uncertainty of the gap
        return {"how": "unpaired", "estimate": float(best["score"]) - float(cand["score"]), "se": math.hypot(float(seb), float(sec)), "aligned": None, "dropped": {}}
    return {"how": "unavailable", "reason": "no paired return series, no daily IC series and no standard errors: the gap has no uncertainty estimate", "aligned": None}


def _holm(ps: list) -> list:
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    out, run, m = [None] * len(ps), 0.0, len(idx)
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * float(ps[i])))
        out[i] = run
    return out


def _one_sided_ps(g: dict, margin: float) -> tuple:
    """``(p for H0: gap >= margin, p for H0: gap <= -margin, p for H0: gap <= 0)`` from the bootstrap draws (add-one) or, for a HAC gap, the normal approximation."""
    if g["how"] == "paired_bootstrap":
        d = g["draws"]
        n = len(d) + 1.0
        return (1.0 + float((d >= margin).sum())) / n, (1.0 + float((d <= -margin).sum())) / n, (1.0 + float((d <= 0.0).sum())) / n
    se, est = float(g["se"]), float(g["estimate"])
    if not se > 0:
        return None, None, None
    return float(sps.norm.cdf((est - margin) / se)), float(sps.norm.sf((est + margin) / se)), float(sps.norm.sf(est / se))


def choose_simplest(candidates: list[dict], policy: SimplicityPolicy | None = None, *, metric: str = "net_sharpe", horizon: int = 1) -> dict:
    """Apply a declared rule to eligible candidates. Each candidate is ``{"id", "score" (higher is better), "complexity": complexity_vector(...), "returns": net return series (optional, paired),
    "ic": daily rank IC series (optional), "score_se": optional}``; the first candidate with the highest score is the highest-ranked one. The winner is the simplest candidate that qualifies, the
    highest-ranked one always qualifying; the result says what kind of statement qualifying is."""
    policy = policy or SimplicityPolicy()
    usable = [c for c in candidates if c.get("score") is not None and math.isfinite(float(c["score"]))]
    base = {"version": VERSION, "policy": asdict(policy), "policy_hash": policy.hash(), "metric": metric, "rule": policy.rule}
    if not usable:
        return {**base, "eligibility": "insufficient", "highest_ranked": None, "selected": None, "reason": "no candidate has a finite score", "table": []}
    best = max(usable, key=lambda c: float(c["score"]))
    gaps = {c["id"]: _gap(best, c, metric, policy, horizon) for c in usable if c is not best}
    rows, ps_hi, ps_lo = [], {}, {}
    for c in usable:
        row = {"id": c["id"], "score": float(c["score"]), "complexity": c["complexity"], "order_key": [None if math.isinf(k) else k for k in order_key(c["complexity"], policy)]}
        if c is best:
            rows.append({**row, "role": "highest_ranked", "gap": 0.0, "se_of_gap": None, "how": None, "qualifies": True, "statistical_status": "highest_ranked", "reason": "the highest-ranked candidate"})
            continue
        g = gaps[c["id"]]
        usable_gap = g["how"] in ("paired_bootstrap", "paired_hac") or (g["how"] == "unpaired" and not policy.require_paired)
        row.update(role="candidate", how=g["how"], aligned=g.get("aligned"), dropped=g.get("dropped"), gap=g.get("estimate"), se_of_gap=g.get("se"), mean_block=g.get("mean_block"))
        if not usable_gap:
            row.update(qualifies=False, statistical_status="no_uncertainty_estimate", reason=g.get("reason") or "an unpaired error is not accepted by this policy (require_paired)")
        elif policy.rule == "one_se":
            ok = float(g["estimate"]) <= float(policy.k) * float(g["se"])
            row.update(qualifies=bool(ok), gap_in_se=float(g["estimate"]) / float(g["se"]) if g["se"] > 0 else None, statistical_status="not_distinguishable_by_heuristic" if ok else "worse_than_the_heuristic_allows",
                       reason=(f"the gap {g['estimate']:+.4g} is within {policy.k:g} standard error(s) ({g['se']:.4g}) of the paired gap" if ok else
                               f"the gap {g['estimate']:+.4g} is {g['estimate'] / g['se']:.2f} standard errors of the paired gap" if g["se"] > 0 else "below the best with no spread"))
        else:
            p_ni, p_lo, p_worse = _one_sided_ps(g, float(policy.margin))
            ps_hi[c["id"]], ps_lo[c["id"]] = p_ni, p_lo
            row.update(p_non_inferior=p_ni, p_not_better_by_margin=p_lo, p_worse=p_worse)
        rows.append(row)
    if policy.rule == "non_inferiority":
        ids = list(ps_hi)
        adj_hi = dict(zip(ids, _holm([ps_hi[i] for i in ids])))
        adj_lo = dict(zip(ids, _holm([ps_lo[i] for i in ids])))
        for row in rows:
            if row["id"] not in ps_hi:
                continue
            ph, pl = adj_hi[row["id"]], adj_lo[row["id"]]
            ok = ph is not None and ph < float(policy.level)
            eq = bool(ok and pl is not None and pl < float(policy.level))
            worse = row.get("p_worse") is not None and row["p_worse"] < float(policy.level)
            row.update(adjusted_p_non_inferior=ph, adjusted_p_not_better_by_margin=pl, qualifies=bool(ok), equivalent_within_margin=eq,
                       statistical_status="non_inferior_within_margin" if ok else "demonstrably_worse" if worse else "not_shown_non_inferior",
                       reason=(f"H0 'the best is better by {policy.margin:g} or more' is rejected (Holm-adjusted one-sided p = {ph:.3g} < {policy.level:g})" if ok else
                               f"H0 'the best is better by {policy.margin:g} or more' is not rejected (Holm-adjusted one-sided p = {ph:.3g}): absence of a rejection is not evidence of inferiority either"
                               if ph is not None else "no p-value"))
    keep = [c for c, r in zip(usable, rows) if r["qualifies"]]
    winner = min(keep, key=lambda c: (order_key(c["complexity"], policy), -float(c["score"]), str(c["id"])))
    formal = policy.rule == "non_inferiority"
    claim = ({"kind": "formal_non_inferiority", "statement": f"the selected model is non-inferior to the highest-ranked within a margin of {policy.margin:g} of {metric} at the one-sided level {policy.level:g}, "
              f"Holm-adjusted over {len(ps_hi)} comparison(s) (H0: the best is better by the margin or more); equivalence within the margin is reported separately and holds only where the table says so",
              "caveats": ["the margin is a declared assumption about what matters, not an estimate", "the highest-ranked candidate is the maximum of N on these same data, so every gap to it is overstated",
                          "the claim is about the validation data only: nothing here says the models will perform alike out of sample"]} if formal else
             {"kind": "selection_heuristic", "statement": f"the selected model is the simplest whose {metric} was not distinguishable from the highest-ranked one by the rule 'gap <= {policy.k:g} standard error(s) of the paired gap'. "
              "This is a heuristic for choosing among candidates; it is not a test of equivalence, and it does not show the models perform equally",
              "caveats": ["failing to show a difference is not showing there is none: a wider standard error admits more candidates", "the highest-ranked candidate is the maximum of N on these same data, so every gap to it is overstated",
                          "the standard error is a bootstrap one on the validation data only"]})
    return {**base, "eligibility": "applied", "highest_ranked": best["id"], "selected": winner["id"], "changed_choice": winner["id"] != best["id"], "qualifying": [c["id"] for c in keep],
            "table": rows, "claim": claim, "equivalence_claimed": bool(formal and any(r.get("equivalent_within_margin") for r in rows if r["id"] == winner["id"])),
            "note": ("only the highest-ranked candidate qualified, so the choice is unchanged" if len(keep) == 1 else
                     f"{len(keep)} candidates qualify; the simplest by {list(policy.priority)} is chosen, with unmeasured dimensions counted as not simple")}
