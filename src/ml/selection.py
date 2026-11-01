"""Which selection-bias procedures may be applied to a study, with what assumptions, and what they say (spec v3, P0-27 and P0-28).

The platform already has the procedures (``stats.inference``, ``validation.multiple_testing``). What it lacked was a layer that decides whether one *applies* to the evidence at hand, says what it assumes and what
it cannot see, and refuses to let a procedure that was not applied look like one that was passed. Each procedure returns an ``Evidence`` with a status:

* ``applied``         it ran; ``statistic`` holds its numbers and ``statistic['passes']`` whether they meet the criterion that was declared before they were seen
* ``not_applicable``  its assumptions do not hold for this evidence (one candidate has nothing to select among; no benchmark defined)
* ``unsupported``     the platform cannot compute what the procedure needs (a return stream that was not kept)
* ``insufficient``    the sample is too short or the candidates too few for the procedure to say anything
* ``failed``          it was applicable and raised: the error is kept

Only ``applied`` evidence with ``passes`` can support a gate; ``gates`` treats the others as a reason the gate is blocked, never as a pass. Every result is persisted (``selection_evidence``, append-only) with its
assumptions, the candidate and sample scope, how serial and cross-candidate dependence was handled, and the limitations that remain. The criteria (the error level, the minimum deflated Sharpe probability, the
maximum probability of backtest overfitting) are parameters stated with the result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

STATUSES = ("applied", "not_applicable", "unsupported", "insufficient", "failed")
CRITERIA = {"alpha": 0.05, "dsr_min": 0.95, "pbo_max": 0.5, "min_obs": 60}
PROCEDURES = ("dsr", "pbo", "reality_check", "spa", "fdr")


@dataclass
class Evidence:
    method: str
    status: str
    reason: str = ""
    assumptions: list = field(default_factory=list)
    scope: dict = field(default_factory=dict)
    dependence: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)
    statistic: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"method": self.method, "status": self.status, "reason": self.reason, "assumptions": self.assumptions, "scope": self.scope, "dependence": self.dependence, "limitations": self.limitations,
                "statistic": self.statistic}


def _scope(M: pd.DataFrame, n_tries: int) -> dict:
    return {"candidates_with_returns": int(M.shape[1]), "tries_recorded": int(n_tries), "observations": int(M.shape[0]), "first": str(M.index[0].date()) if len(M) and hasattr(M.index[0], "date") else None,
            "last": str(M.index[-1].date()) if len(M) and hasattr(M.index[-1], "date") else None, "sample": "the validation segment on which the candidates were compared (never the test or forward segments)"}


def _guard(M: pd.DataFrame, n_tries: int, method: str, criteria: dict, min_candidates: int = 2) -> Evidence | None:
    """The applicability conditions every procedure shares: enough candidates, enough observations, finite returns."""
    scope = _scope(M, n_tries)
    if M.shape[1] < min_candidates:
        return Evidence(method, "not_applicable", f"{method} corrects the choice among candidates and needs the return streams of at least {min_candidates}; this study has {M.shape[1]}", scope=scope)
    if M.shape[0] < int(criteria["min_obs"]):
        return Evidence(method, "insufficient", f"{M.shape[0]} common observations is fewer than the {int(criteria['min_obs'])} the procedure needs to say anything", scope=scope)
    if not np.isfinite(M.to_numpy()).all():
        return Evidence(method, "unsupported", "the candidates' return streams have gaps on the common dates; the procedure needs complete streams", scope=scope)
    flat = constant_candidates(M)
    if flat:
        return Evidence(method, "unsupported", f"the return stream of {flat} is constant: its Sharpe ratio is undefined, and a statistic built on it would be a number without meaning. ``run_procedures`` excludes such "
                                               "candidates and records the exclusion; called directly, the procedure refuses them", scope=scope)
    return None


def constant_candidates(M: pd.DataFrame) -> list[str]:
    """Candidates whose return stream does not vary beyond rounding (a strategy that never traded, or held a flat book): their Sharpe ratio, t-statistic and rank are undefined."""
    from .inference import is_constant

    return [str(c) for c in M.columns if is_constant(M[c].to_numpy(dtype=float))]


def _sharpes(M: pd.DataFrame) -> pd.Series:
    sd = M.std(ddof=1)
    return (M.mean() / sd.where(sd > 0)).fillna(0.0)


def deflated(M: pd.DataFrame, n_tries: int, criteria: dict) -> Evidence:
    from ..ops.hpo import effective_trials
    from ..stats.inference import deflated_sharpe_ratio

    bad = _guard(M, n_tries, "dsr", criteria)
    if bad:
        return bad
    sharpes = _sharpes(M)
    best = sharpes.idxmax()
    n_eff, rho = effective_trials(M)
    eff = deflated_sharpe_ratio(M[best].to_numpy(), n_trials=n_eff)
    classic = deflated_sharpe_ratio(M[best].to_numpy(), trial_sharpes=sharpes.to_numpy(), n_trials=max(n_tries, M.shape[1]))
    return Evidence("dsr", "applied", assumptions=["per-period returns; the Sharpe standard error uses skewness and kurtosis (Mertens) and treats the returns as serially independent",
                                                   "the maximum Sharpe of N unskilled tries has the expected value of Bailey and Lopez de Prado (2014)"],
                    scope=_scope(M, n_tries), dependence={"serial": "not adjusted (no autocorrelation correction in this form of the standard error)", "between_tries": f"effective number of independent tries {n_eff:.1f} from "
                                                          f"their average return correlation {rho:.2f}"},
                    limitations=["tries without a return stream (failed or cancelled ones) count in the raw look number but not in the effective number or the variance",
                                 "the probability is for the best of the recorded candidates on the validation sample; it is not a statement about the holdout"] + (["the raw count of tries exceeds the candidates with returns"] if n_tries > M.shape[1] else []),
                    statistic={"best_candidate": str(best), "annual_sharpe": float(sharpes[best] * np.sqrt(252)), "probability": eff["probability"], "probability_classic": classic["probability"],
                               "benchmark_annual_sharpe": float(eff["benchmark"] * np.sqrt(252)), "effective_trials": n_eff, "mean_correlation": rho, "raw_tries": int(n_tries),
                               "criterion": f"probability >= {criteria['dsr_min']}", "passes": bool(eff["probability"] >= criteria["dsr_min"])})


def overfitting_probability(M: pd.DataFrame, n_tries: int, criteria: dict) -> Evidence:
    from ..validation.multiple_testing import pbo_cscv

    bad = _guard(M, n_tries, "pbo", criteria, min_candidates=3)
    if bad:
        return bad
    blocks = min(16, (M.shape[0] // 8) // 2 * 2)
    if blocks < 4:
        return Evidence("pbo", "insufficient", f"{M.shape[0]} observations allow {blocks} blocks of 8; CSCV needs at least 4", scope=_scope(M, n_tries))
    res = pbo_cscv(M.to_numpy(), blocks=blocks)
    return Evidence("pbo", "applied", assumptions=["every candidate was evaluated on the same dates", "performance is the Sharpe ratio of a block", "blocks are exchangeable (the order of blocks carries no information)"],
                    scope=_scope(M, n_tries), dependence={"serial": f"contiguous blocks of about {M.shape[0] // blocks} days keep the dependence inside a block", "between_tries": "kept: the same blocks are used for every candidate"},
                    limitations=["PBO is a property of the selection procedure on this sample, not of the chosen candidate", "it needs at least 3 candidates and says nothing about the holdout"],
                    statistic={"pbo": float(res["pbo"]), "blocks": blocks, "mean_oos_of_best": float(res["mean_oos_of_best"]), "criterion": f"PBO <= {criteria['pbo_max']}", "passes": bool(res["pbo"] <= criteria["pbo_max"])})


def _block(T: int) -> float:
    return float(max(5, round(T ** (1.0 / 3.0))))


def reality_check(M: pd.DataFrame, n_tries: int, criteria: dict, benchmark: pd.Series | None, seed: int, n_boot: int) -> Evidence:
    from ..validation.multiple_testing import white_reality_check

    bad = _guard(M, n_tries, "reality_check", criteria)
    if bad:
        return bad
    excess = M if benchmark is None else M.sub(benchmark.reindex(M.index).fillna(0.0), axis=0)
    block = _block(len(M))
    res = white_reality_check(excess.to_numpy(), n_samples=n_boot, mean_block=block, seed=seed)
    return Evidence("reality_check", "applied", assumptions=["null: no candidate has a positive expected return" + (" over the benchmark" if benchmark is not None else " (the benchmark is cash, zero)"),
                                                            "the stationary bootstrap reproduces the dependence of the returns"], scope=_scope(M, n_tries),
                    dependence={"serial": f"stationary bootstrap, mean block {block:g} days", "between_tries": "kept: the same resampled days are used for every candidate"},
                    limitations=["tests the family of recorded candidates; candidates not recorded are not in the null", "the sample is the validation segment used to select"],
                    statistic={"p_value": float(res["p_value"]), "best_candidate": str(excess.columns[res["best_index"]]), "n_bootstrap": int(n_boot), "criterion": f"p <= {criteria['alpha']}",
                               "passes": bool(res["p_value"] <= criteria["alpha"])})


def superior_predictive_ability(M: pd.DataFrame, n_tries: int, criteria: dict, benchmark: pd.Series | None, seed: int, n_boot: int) -> Evidence:
    from ..validation.multiple_testing import hansen_spa

    bad = _guard(M, n_tries, "spa", criteria)
    if bad:
        return bad
    excess = M if benchmark is None else M.sub(benchmark.reindex(M.index).fillna(0.0), axis=0)
    block = _block(len(M))
    res = hansen_spa(excess.to_numpy(), n_samples=n_boot, mean_block=block, seed=seed)
    return Evidence("spa", "applied", assumptions=["null: no candidate is better than the benchmark" + (" (cash, zero)" if benchmark is None else ""), "studentised statistic; poor candidates are recentred (the consistent p-value)"],
                    scope=_scope(M, n_tries), dependence={"serial": f"stationary bootstrap, mean block {block:g} days", "between_tries": "kept"},
                    limitations=["a candidate whose bootstrap never varies has no t-statistic and is ignored", "the sample is the validation segment used to select"],
                    statistic={"p_consistent": float(res["p_consistent"]), "p_upper": float(res["p_upper"]), "p_lower": float(res["p_lower"]), "best_t": float(res["best_t"]), "n_bootstrap": int(n_boot),
                               "criterion": f"consistent p <= {criteria['alpha']}", "passes": bool(res["p_consistent"] <= criteria["alpha"])})


def false_discovery(M: pd.DataFrame, n_tries: int, criteria: dict) -> Evidence:
    from ..stats.inference import p_adjust, sharpe_standard_error

    bad = _guard(M, n_tries, "fdr", criteria)
    if bad:
        return bad
    sr = _sharpes(M)
    se = np.array([sharpe_standard_error(M[c].to_numpy(), "lo") for c in M.columns])
    p = sps.norm.sf((sr.to_numpy() / se))
    by, bh = p_adjust(p, "by"), p_adjust(p, "bh")
    i = int(np.argmin(p))
    return Evidence("fdr", "applied", assumptions=["one-sided test that each candidate's Sharpe ratio is above zero", "the Sharpe standard error corrects for autocorrelation (Lo, 2002)"], scope=_scope(M, n_tries),
                    dependence={"serial": "Lo (2002) autocorrelation correction", "between_tests": "the pass criterion uses Benjamini-Yekutieli, valid under any dependence; Benjamini-Hochberg is reported but assumes positive dependence"},
                    limitations=["controls the false discovery rate among the recorded candidates only", "the sample is the validation segment used to select"],
                    statistic={"best_candidate": str(M.columns[i]), "p_best": float(p[i]), "q_by_best": float(by[i]), "q_bh_best": float(bh[i]), "n_discoveries_by": int((by <= criteria["alpha"]).sum()),
                               "n_discoveries_bh": int((bh <= criteria["alpha"]).sum()), "criterion": f"BY q of the best <= {criteria['alpha']}", "passes": bool(by[i] <= criteria["alpha"])})


def criteria_from_policy(policy) -> dict:
    """The criteria a policy declares, as the keyword arguments of ``run_procedures`` (``alpha``, ``dsr_min``, ``pbo_max``). A threshold the policy leaves unset keeps the platform's default, which the gate does
    not read when a policy is adopted: the policy's own items judge the numbers."""
    out = dict(CRITERIA)
    for crit, key in (("alpha", "max_selection_p"), ("dsr_min", "min_deflated_probability"), ("pbo_max", "max_pbo")):
        if policy is not None and policy.required(key):
            out[crit] = float(policy.criteria[key])
    return out


def run_procedures(M: pd.DataFrame, n_tries: int, *, benchmark: pd.Series | None = None, criteria: dict | None = None, procedures: tuple = PROCEDURES, seed: int = 0, n_boot: int = 500) -> list[Evidence]:
    """Every requested procedure on the candidate matrix ``M`` (dates x candidates, net returns on the SAME validation dates). A procedure that raises is recorded as ``failed`` with the error."""
    crit = {**CRITERIA, **(criteria or {})}
    flat = constant_candidates(M) if np.isfinite(M.to_numpy()).all() else []
    if flat:                                                                                          # excluded from every statistic; the raw try count still includes them, so deflation is not helped
        M = M.drop(columns=flat)
    table = {"dsr": lambda: deflated(M, n_tries, crit), "pbo": lambda: overfitting_probability(M, n_tries, crit), "reality_check": lambda: reality_check(M, n_tries, crit, benchmark, seed, n_boot),
             "spa": lambda: superior_predictive_ability(M, n_tries, crit, benchmark, seed, n_boot), "fdr": lambda: false_discovery(M, n_tries, crit)}
    out = []
    for name in procedures:
        if name not in table:
            out.append(Evidence(name, "unsupported", f"the platform has no implementation of '{name}'; supported: {', '.join(table)}"))
            continue
        try:
            ev = table[name]()
        except Exception as exc:                                                                   # noqa: BLE001 - a procedure that cannot be computed is evidence of that, not a crash
            ev = Evidence(name, "failed", f"{type(exc).__name__}: {exc}", scope=_scope(M, n_tries))
        ev.statistic = {**ev.statistic, "criteria": crit} if ev.status == "applied" else ev.statistic
        if flat:
            ev.scope = {**ev.scope, "excluded_constant_candidates": flat}
            ev.limitations = [*ev.limitations, f"{len(flat)} candidate(s) with a constant return stream ({', '.join(flat)}) were excluded: their Sharpe ratio is undefined. They still count among the {n_tries} recorded tries"]
        out.append(ev)
    return out


# ------------------------------------------------------------------------------------------------------------------ a feature screen (P0-28)
def adjust_screen(ic: pd.DataFrame, *, alpha: float = 0.05, method: str = "holm", n_tested: int | None = None, min_features: int = 2) -> Evidence:
    """Selection adjustment for a feature screen. ``ic`` has one row per feature TESTED (the rejected ones too) with ``t`` (an overlap-corrected t-statistic of its rank IC; ``explainability.feature_ic``)
    and ``n``. Two-sided p-values (normal approximation) are adjusted for the number of features tested (Holm: familywise error, valid under any dependence; or ``by``: false discovery rate, valid under any
    dependence). The features that survive at ``alpha`` are the only ones the screen may claim. If fewer than ``min_features`` were tested, or the t-statistics are missing, the adjustment is ``unsupported``
    and says so rather than passing every feature."""
    from ..stats.inference import p_adjust

    if method not in ("holm", "by", "bonferroni"):
        return Evidence("feature_screen", "unsupported", f"the adjustment '{method}' is not supported for feature screens (holm, by, bonferroni are; Benjamini-Hochberg assumes dependence that correlated features do not have)")
    tested = int(n_tested if n_tested is not None else len(ic))
    t = pd.to_numeric(ic.get("t"), errors="coerce") if "t" in ic else pd.Series(dtype=float)
    ok = t.dropna()
    if tested < min_features or ok.empty:
        return Evidence("feature_screen", "unsupported", f"{tested} feature(s) tested with {len(ok)} usable t-statistic(s): there is no multiplicity to adjust for, or no statistic to adjust", scope={"features_tested": tested})
    p = 2.0 * sps.norm.sf(np.abs(ok.to_numpy()))
    padded = np.concatenate([p, np.ones(max(0, tested - len(ok)))])                                   # features without a statistic count as tested and never survive
    adj = p_adjust(padded, method)[: len(ok)]
    keep = sorted(str(f) for f, a in zip(ok.index, adj) if a <= alpha)
    return Evidence("feature_screen", "applied", assumptions=["the rank-IC t-statistic counts one independent observation per horizon (overlap-corrected)", "two-sided normal approximation for the p-value"],
                    scope={"features_tested": tested, "with_statistic": int(len(ok)), "sample": "the warm-up segment only"},
                    dependence={"between_features": f"{method} is valid under any dependence between the tests (features are correlated)", "serial": "overlap handled through the horizon in the t-statistic"},
                    limitations=["the screen controls error among the features tested here; features tried in other screens of the same study are separate looks (they are in the study's lineage)"],
                    statistic={"method": method, "alpha": alpha, "n_selected": len(keep), "selected": keep, "adjusted_p": {str(f): float(a) for f, a in zip(ok.index, adj)}, "criterion": f"adjusted p <= {alpha}",
                               "passes": bool(keep)})


def rule_select(scores: dict, rule: dict) -> list[str]:
    """The reproducible form of the screen: features whose recorded adjusted p-value is at most ``rule['alpha']``."""
    alpha = float(rule.get("alpha", 0.05))
    return [f for f, a in (rule.get("adjusted_p") or scores).items() if a <= alpha]


# ------------------------------------------------------------------------------------------------------------------ persistence
def persist(registry, study_id: str, evidence: list[Evidence], subject: str | None = None) -> list[int]:
    """Append the evidence to ``selection_evidence``; returns the row ids. Rows are never rewritten: a later run is another row."""
    ids = []
    with registry._lock, registry._conn() as c:                                                    # noqa: SLF001
        for ev in evidence:
            cur = c.execute("INSERT INTO selection_evidence (study_id, subject, method, status, assumptions, scope, dependence, limitations, reason, statistic, created) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (study_id, subject, ev.method, ev.status, json.dumps(ev.assumptions), json.dumps(ev.scope, default=str), json.dumps(ev.dependence, default=str), json.dumps(ev.limitations),
                             ev.reason, json.dumps(ev.statistic, default=str), time.strftime("%Y-%m-%d %H:%M:%S")))
            ids.append(int(cur.lastrowid))
    return ids


def trial_matrix(registry, study_id: str) -> tuple[pd.DataFrame, int, list[str]]:
    """``(returns, raw tries, notes)``: the validation net returns of the study's trials that kept a stream (``<registry>/trials/<study>/<operation>.parquet``), on their common dates, and the number of tries
    the lineage counts. Tries without a stream are named in the notes."""
    from . import research_controls as rc

    ops = [o for o in rc._latest(registry, rc.study_family(registry, study_id)) if o["counts"]]       # noqa: SLF001
    cols, missing = {}, []
    for o in ops:
        path = registry.root / "trials" / o["study_id"] / f"{o['op_key']}.parquet"
        if o["state"] == "done" and path.exists():
            cols[o["op_key"]] = pd.read_parquet(path)["net"]
        elif o["state"] == "done" and o["kind"] in ("hpo_trial", "model_choice", "signal_variation", "portfolio_variation"):
            missing.append(o["op_key"])
    notes = []
    if missing:
        notes.append(f"{len(missing)} completed try(ies) kept no return stream and are not in the matrix")
    if not cols:
        return pd.DataFrame(), len(ops), notes
    M = pd.DataFrame(cols).dropna()
    return M, len(ops), notes


def evaluate_study(registry, study_id: str, *, benchmark: pd.Series | None = None, criteria: dict | None = None, procedures: tuple = PROCEDURES, seed: int = 0, n_boot: int = 500, subject: str | None = None,
                   persist_rows: bool = True) -> dict:
    """Run the procedures on a study's recorded trials and persist the evidence. When the study has no return streams at all every procedure is recorded as ``unsupported`` with that reason."""
    M, n_tries, notes = trial_matrix(registry, study_id)
    if M.empty:
        ev = [Evidence(p, "unsupported", "this study has no recorded return streams for its tries (a study whose trials ran before streams were kept, or one with no completed trials)") for p in procedures]
    else:
        ev = run_procedures(M, n_tries, benchmark=benchmark, criteria=criteria, procedures=procedures, seed=seed, n_boot=n_boot)
    ids = persist(registry, study_id, ev, subject) if persist_rows else []
    return {"study_id": study_id, "n_tries": n_tries, "n_candidates": int(M.shape[1]), "notes": notes, "evidence": [dict(e.as_dict(), id=i) for e, i in zip(ev, ids or [None] * len(ev))],
            "statement": statement(ev)}


def statement(evidence: list[Evidence]) -> str:
    """One honest sentence: how many procedures applied and passed, and which were not applied and why. A procedure that was not applied is never counted as passed."""
    applied = [e for e in evidence if e.status == "applied"]
    passed = [e.method for e in applied if e.statistic.get("passes")]
    other = [f"{e.method} ({e.status})" for e in evidence if e.status != "applied"]
    text = f"{len(applied)} of {len(evidence)} procedures were applied and {len(passed)} met their criterion" + (f" ({', '.join(passed)})" if passed else "")
    return text + (f"; not applied: {', '.join(other)}" if other else "") + "."


def register_screen(registry, study: str, fid: str, fic: pd.DataFrame, alpha: float = 0.05, method: str = "holm") -> dict:
    """Record a univariate feature screen as adaptive feature discovery: one operation in the study (however many features it tested), the adjustment for the number tested, and every feature's adjusted
    p-value, rejected ones included, so the choice can be reproduced (``research_controls.reproduce_feature_choice``). Declared ``unsupported``, not passed, when there is nothing to adjust."""
    from . import research_controls as rc

    try:
        ev = adjust_screen(fic, alpha=alpha, method=method, n_tested=len(fic)) if len(fic) else Evidence("feature_screen", "unsupported", "no feature had an information coefficient on the warm-up segment")
        persist(registry, study, [ev], subject=fid)
        chosen = ev.statistic.get("selected", []) if ev.status == "applied" else []
        out = rc.register_feature_search(registry, study, [{"features": list(fic.index), "score": None, "accepted": bool(chosen), "reason": f"univariate rank-IC screen of {len(fic)} features on the warm-up segment"}],
                                         "rank IC t-statistic on the warm-up segment", selected=chosen, rule={"alpha": alpha, "method": method, "adjusted_p": ev.statistic.get("adjusted_p")}, adjustment=ev.as_dict())
    except Exception as exc:                                                                              # noqa: BLE001 - a diagnostic must not lose the dataset
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": ev.status, "reason": ev.reason, "study": study, "operation": out["variants"][0], "n_tested": len(fic), "n_selected": len(chosen), "selected": chosen[:40], "method": method,
            "criterion": ev.statistic.get("criterion"), "limitations": ev.limitations, "dependence": ev.dependence}
