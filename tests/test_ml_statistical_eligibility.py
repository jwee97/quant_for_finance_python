"""Release 3, P0-27 and the selection-adjustment part of P0-28: the eligibility layer for DSR, PBO/CSCV, Reality Check, SPA and FDR.

The numerical references are written here from the definitions (not by calling the module under test): the deflated Sharpe ratio of Bailey and Lopez de Prado (2014), CSCV by enumerating the block
combinations, Benjamini-Hochberg and Benjamini-Yekutieli by their step-up rules, Holm by its step-down rule. The status semantics are the contract: a procedure that did not apply is never a pass."""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.ml import registry as rg, selection as sel

warnings.filterwarnings("ignore")
EULER = 0.5772156649015329


def matrix(k=6, t=300, seed=0, edge=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.01, size=(t, k))
    if edge is not None:
        x[:, 0] += edge
    return pd.DataFrame(x, index=pd.bdate_range("2021-01-04", periods=t), columns=[f"c{i}" for i in range(k)])


# --------------------------------------------------------------------------------------------------------------------- independent references
def reference_dsr_probability(r, n_trials, var):
    r = np.asarray(r)
    n, mu, sd = len(r), r.mean(), r.std(ddof=1)
    sr = mu / sd
    g3, g4 = sps.skew(r), sps.kurtosis(r, fisher=False)
    bench = np.sqrt(var) * ((1 - EULER) * sps.norm.ppf(1 - 1 / n_trials) + EULER * sps.norm.ppf(1 - 1 / (n_trials * np.e)))
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (n - 1))
    return float(sps.norm.cdf((sr - bench) / se)), float(bench)


def reference_pbo(x, blocks):
    n, k = x.shape
    edges = np.linspace(0, n, blocks + 1).astype(int)
    parts = [x[edges[i]:edges[i + 1]] for i in range(blocks)]

    def sharpe(rows):
        return rows.mean(axis=0) / rows.std(axis=0, ddof=1)

    lam = []
    for ins in itertools.combinations(range(blocks), blocks // 2):
        out = [i for i in range(blocks) if i not in ins]
        s_in, s_out = sharpe(np.vstack([parts[i] for i in ins])), sharpe(np.vstack([parts[i] for i in out]))
        best = int(np.argmax(s_in))
        rank = 1 + int((s_out < s_out[best]).sum())                                      # 1 = the lowest out-of-sample Sharpe
        w = rank / (k + 1)
        lam.append(np.log(w / (1 - w)))
    return float(np.mean(np.asarray(lam) <= 0))


def reference_step_up(p, m_scale):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m * m_scale / rank)
        adj[i] = running
    return adj


def reference_holm(p):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for j, i in enumerate(order):
        running = max(running, (m - j) * p[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------------------------------------------------- applied procedures
def test_the_deflated_sharpe_ratio_matches_the_definition():
    M = matrix(k=8, t=400, seed=1, edge=0.0007)
    ev = sel.deflated(M, 12, sel.CRITERIA)
    assert ev.status == "applied"
    best = ev.statistic["best_candidate"]
    sr = M.mean() / M.std(ddof=1)
    assert best == sr.idxmax()
    from src.ops.hpo import effective_trials

    n_eff, _ = effective_trials(M)
    prob, bench = reference_dsr_probability(M[best].to_numpy(), n_eff, 1.0 / len(M))
    assert ev.statistic["probability"] == pytest.approx(prob, abs=1e-12) and ev.statistic["benchmark_annual_sharpe"] == pytest.approx(bench * np.sqrt(252), abs=1e-9)
    classic, _ = reference_dsr_probability(M[best].to_numpy(), 12, float(np.var(sr.to_numpy(), ddof=1)))
    assert ev.statistic["probability_classic"] == pytest.approx(classic, abs=1e-12) and ev.statistic["raw_tries"] == 12
    assert ev.dependence["serial"].startswith("not adjusted") and any("failed or cancelled" in s for s in ev.limitations)
    assert ev.statistic["passes"] == (ev.statistic["probability"] >= 0.95)


def test_deflation_penalises_a_search_in_proportion_to_its_size():
    strong = matrix(k=4, t=500, seed=2, edge=0.002)
    a = sel.deflated(strong, 4, sel.CRITERIA).statistic["probability_classic"]
    b = sel.deflated(strong, 400, sel.CRITERIA).statistic["probability_classic"]
    assert b < a, "more tries must lower the deflated probability of the same winner"


def test_pbo_matches_a_brute_force_cscv():
    M = matrix(k=5, t=96, seed=3)
    ev = sel.overfitting_probability(M, 5, sel.CRITERIA)
    assert ev.status == "applied" and ev.statistic["blocks"] == 12
    assert ev.statistic["pbo"] == pytest.approx(reference_pbo(M.to_numpy(), 12), abs=1e-12)
    assert ev.statistic["passes"] == (ev.statistic["pbo"] <= 0.5)
    pure_noise = [sel.overfitting_probability(matrix(k=6, t=160, seed=s), 6, sel.CRITERIA).statistic["pbo"] for s in range(10, 16)]
    assert 0.2 < np.mean(pure_noise) < 0.8, "with nothing to find, the in-sample winner is a coin flip out of sample"


def test_the_reality_check_and_spa_separate_a_real_edge_from_selected_noise():
    edge = matrix(k=6, t=500, seed=4, edge=0.0025)
    rc = sel.reality_check(edge, 6, sel.CRITERIA, None, seed=1, n_boot=400)
    sp = sel.superior_predictive_ability(edge, 6, sel.CRITERIA, None, seed=1, n_boot=400)
    assert rc.status == sp.status == "applied" and rc.statistic["p_value"] < 0.02 and sp.statistic["p_consistent"] < 0.02 and rc.statistic["passes"] and sp.statistic["passes"]
    assert rc.statistic["best_candidate"] == "c0" and "stationary bootstrap" in rc.dependence["serial"]
    noise_p = [sel.reality_check(matrix(k=8, t=250, seed=s), 8, sel.CRITERIA, None, seed=s, n_boot=300).statistic["p_value"] for s in range(20, 30)]
    assert np.mean(np.asarray(noise_p) <= 0.05) <= 0.3, "the best of eight noise streams is not significant after the correction"
    with_bench = sel.reality_check(edge, 6, sel.CRITERIA, pd.Series(0.0025, index=edge.index), seed=1, n_boot=300)      # a benchmark that earns the edge removes it
    assert with_bench.statistic["p_value"] > 0.05 and "over the benchmark" in with_bench.assumptions[0]


def test_false_discovery_control_matches_the_step_up_rules():
    M = matrix(k=7, t=400, seed=5, edge=0.0012)
    ev = sel.false_discovery(M, 7, sel.CRITERIA)
    from src.stats.inference import sharpe_standard_error

    sr = (M.mean() / M.std(ddof=1)).to_numpy()
    p = sps.norm.sf(sr / np.array([sharpe_standard_error(M[c].to_numpy(), "lo") for c in M.columns]))
    by, bh = reference_step_up(p, float(np.sum(1 / np.arange(1, 8)))), reference_step_up(p, 1.0)
    i = int(np.argmin(p))
    assert ev.status == "applied" and ev.statistic["q_by_best"] == pytest.approx(by[i], abs=1e-12) and ev.statistic["q_bh_best"] == pytest.approx(bh[i], abs=1e-12)
    assert ev.statistic["passes"] == (by[i] <= 0.05) and "valid under any dependence" in ev.dependence["between_tests"]
    assert (by >= bh - 1e-12).all(), "Benjamini-Yekutieli is never less strict than Benjamini-Hochberg"


# --------------------------------------------------------------------------------------------------------------------- status semantics
def test_a_procedure_that_does_not_apply_says_why_and_is_never_a_pass():
    one = matrix(k=1, t=300)
    for ev in (sel.deflated(one, 1, sel.CRITERIA), sel.reality_check(one, 1, sel.CRITERIA, None, 0, 50), sel.false_discovery(one, 1, sel.CRITERIA)):
        assert ev.status == "not_applicable" and "at least 2" in ev.reason and "passes" not in ev.statistic
    two = matrix(k=2, t=300)
    assert sel.overfitting_probability(two, 2, sel.CRITERIA).status == "not_applicable" and "at least 3" in sel.overfitting_probability(two, 2, sel.CRITERIA).reason
    short = matrix(k=5, t=30)
    assert {e.status for e in sel.run_procedures(short, 5, n_boot=50)} == {"insufficient"}
    gaps = matrix(k=5, t=200)
    gaps.iloc[10, 2] = np.nan
    assert {e.status for e in sel.run_procedures(gaps, 5, n_boot=50)} == {"unsupported"}
    bars = sel.overfitting_probability(matrix(k=5, t=30), 5, {**sel.CRITERIA, "min_obs": 20})
    assert bars.status == "insufficient" and "CSCV needs at least 4" in bars.reason
    unknown = sel.run_procedures(matrix(), 6, procedures=("dsr", "magic"), n_boot=50)
    assert unknown[0].status == "applied" and unknown[1].status == "unsupported" and "no implementation of 'magic'" in unknown[1].reason


def test_a_procedure_that_raises_is_recorded_as_failed_and_the_others_still_run(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("singular covariance")

    monkeypatch.setattr(sel, "deflated", boom)
    out = sel.run_procedures(matrix(), 6, procedures=("dsr", "fdr"), n_boot=50)
    assert [e.status for e in out] == ["failed", "applied"] and "RuntimeError: singular covariance" in out[0].reason


def test_the_statement_never_counts_an_unapplied_procedure_as_met():
    ev = [sel.Evidence("dsr", "applied", statistic={"passes": True}), sel.Evidence("pbo", "not_applicable", "needs 3 candidates"), sel.Evidence("spa", "insufficient", "short"),
          sel.Evidence("fdr", "applied", statistic={"passes": False})]
    text = sel.statement(ev)
    assert "2 of 4 procedures were applied and 1 met their criterion (dsr)" in text and "pbo (not_applicable)" in text and "spa (insufficient)" in text and "fdr" not in text.split("not applied:")[0].split("(dsr)")[1]


# --------------------------------------------------------------------------------------------------------------------- feature screens
def test_a_feature_screen_adjusts_for_every_feature_tested_including_the_rejected():
    ic = pd.DataFrame({"t": [4.2, 3.9, 0.4, -0.2, 2.1, 1.1], "n": 500}, index=list("abcdef"))
    p = 2 * sps.norm.sf(np.abs(ic["t"].to_numpy()))
    holm = reference_holm(p)
    ev = sel.adjust_screen(ic, alpha=0.05, method="holm")
    assert ev.status == "applied" and ev.statistic["selected"] == [f for f, a in zip(ic.index, holm) if a <= 0.05]
    assert all(ev.statistic["adjusted_p"][f] == pytest.approx(a, abs=1e-12) for f, a in zip(ic.index, holm))
    assert "valid under any dependence" in ev.dependence["between_features"] and ev.statistic["n_selected"] == len(ev.statistic["selected"])
    by = sel.adjust_screen(ic, alpha=0.05, method="by")
    assert len(by.statistic["selected"]) <= len(sel.adjust_screen(ic, alpha=0.05, method="bonferroni").statistic["selected"]) + len(ic)
    # features tested but without a statistic still count in the multiplicity
    ten = sel.adjust_screen(ic, alpha=0.05, method="holm", n_tested=60)
    assert set(ten.statistic["selected"]) <= set(ev.statistic["selected"]) and ten.scope["features_tested"] == 60


def test_a_feature_screen_with_nothing_to_adjust_is_declared_unavailable_not_passed():
    one = sel.adjust_screen(pd.DataFrame({"t": [5.0], "n": 400}, index=["only"]))
    assert one.status == "unsupported" and "no multiplicity" in one.reason and "passes" not in one.statistic
    assert sel.adjust_screen(pd.DataFrame({"t": [np.nan, np.nan]}, index=["a", "b"])).status == "unsupported"
    assert sel.adjust_screen(pd.DataFrame({"t": [3.0, 2.0]}, index=["a", "b"]), method="bh").status == "unsupported", "BH assumes a dependence correlated features do not have"
    assert sel.rule_select({}, {"alpha": 0.05, "adjusted_p": {"a": 0.01, "b": 0.2}}) == ["a"]


# --------------------------------------------------------------------------------------------------------------------- persistence
def test_evidence_is_persisted_append_only_with_its_assumptions_and_scope(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    M = matrix(k=6, t=300, seed=8, edge=0.002)
    ev = sel.run_procedures(M, 9, n_boot=100)
    ids = sel.persist(reg, "study-x", ev, subject="exp-1")
    with reg._conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM selection_evidence ORDER BY id")]
        assert [r["method"] for r in rows] == list(sel.PROCEDURES) and len(ids) == 5
        import json

        dsr = next(r for r in rows if r["method"] == "dsr")
        assert json.loads(dsr["scope"])["tries_recorded"] == 9 and json.loads(dsr["assumptions"]) and json.loads(dsr["limitations"]) and json.loads(dsr["dependence"])["serial"]
        with pytest.raises(Exception, match="append-only"):
            c.execute("UPDATE selection_evidence SET status='applied'")


def test_a_study_without_kept_return_streams_is_unsupported_and_says_so(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    out = sel.evaluate_study(reg, "empty-study", n_boot=50)
    assert {e["status"] for e in out["evidence"]} == {"unsupported"} and "no recorded return streams" in out["evidence"][0]["reason"]
    assert "0 of 5 procedures were applied" in out["statement"] and out["n_candidates"] == 0


# ===================================================================================================================== Release 5, P1-17: the conventions audited, and degenerate data
def skewed_stream(n=500, seed=11):
    """Right-skewed, fat-tailed per-period returns with a high Sharpe ratio, so that the skewness and kurtosis terms of the standard error matter."""
    rng = np.random.default_rng(seed)
    return 0.0022 + 0.01 * (0.7 * rng.standard_t(4, n) + 1.6 * (rng.exponential(1.0, n) - 1.0))


def test_dsr_uses_raw_kurtosis_n_minus_one_and_counts_trials_the_documented_way():
    r = skewed_stream()
    rest = {f"n{i}": np.random.default_rng(20 + i).normal(0.0, 0.01, len(r)) for i in range(5)}
    M = pd.DataFrame({"a": r, **rest}, index=pd.bdate_range("2019-01-01", periods=len(r)))
    ev = sel.deflated(M, 40, sel.CRITERIA)
    sr = (M.mean() / M.std(ddof=1)).to_numpy()
    classic, bench = reference_dsr_probability(r, 40, float(np.var(sr, ddof=1)))
    assert ev.statistic["best_candidate"] == "a" and ev.statistic["raw_tries"] == 40
    assert ev.statistic["probability_classic"] == pytest.approx(classic, abs=1e-12), "raw kurtosis (3 for a normal), the (n - 1) in the denominator and N = the larger of the recorded tries and the candidates"
    g3, g4_raw = sps.skew(r), sps.kurtosis(r, fisher=False)
    s = r.mean() / r.std(ddof=1)
    n = len(r)

    def prob(kurt, denom):
        return float(sps.norm.cdf((s - bench) / np.sqrt((1 - g3 * s + (kurt - 1) / 4 * s ** 2) / denom)))

    assert prob(g4_raw, n - 1) == pytest.approx(classic, abs=1e-12)
    assert abs(prob(g4_raw - 3, n - 1) - classic) > 1e-4, "using excess kurtosis where the formula wants raw kurtosis would change the answer, so the fixture can tell the two apart"
    assert abs(prob(g4_raw, n) - classic) > 1e-7 and g3 > 0.5 and g4_raw > 3.5, "the fixture is skewed and fat-tailed (a normal has skewness 0 and raw kurtosis 3)"
    fewer = sel.deflated(M, 6, sel.CRITERIA).statistic["probability_classic"]
    assert fewer > classic, "more recorded tries deflate more"
    assert "serial" in ev.dependence and ev.dependence["serial"].startswith("not adjusted"), "the audit says outright that no autocorrelation correction is applied"


def test_pbo_ignores_the_order_of_blocks_so_it_is_not_a_chronological_validation():
    M = matrix(k=5, t=96, seed=5)                                                                      # twelve blocks of eight days
    blocks = np.array_split(np.arange(96), 12)
    order = [3, 0, 7, 11, 1, 9, 4, 2, 10, 5, 8, 6]
    shuffled = M.iloc[np.concatenate([blocks[i] for i in order])].copy()
    shuffled.index = M.index                                                                           # the dates stay monotone; only which block sits where has changed
    a, b = sel.overfitting_probability(M, 5, sel.CRITERIA), sel.overfitting_probability(shuffled, 5, sel.CRITERIA)
    assert a.statistic["pbo"] == pytest.approx(b.statistic["pbo"], abs=1e-12) == pytest.approx(reference_pbo(M.to_numpy(), 12), abs=1e-12)
    assert "blocks are exchangeable" in " ".join(a.assumptions)

    def chronological(df):                                                                             # train on the first half, test on the second: this one DOES depend on the order
        s_in, s_out = df.iloc[:48].mean() / df.iloc[:48].std(), df.iloc[48:].mean() / df.iloc[48:].std()
        return float(s_out[s_in.idxmax()])

    assert abs(chronological(M) - chronological(shuffled)) > 1e-3, "a chronological split reacts to the order of the blocks; CSCV does not, so PBO is not a substitute for walk-forward validation"


@pytest.mark.parametrize("k,t,status,blocks", [(2, 200, "not_applicable", None), (5, 24, "insufficient", None), (5, 40, "applied", 4), (5, 60, "applied", 6), (5, 96, "applied", 12), (5, 400, "applied", 16)])
def test_pbo_needs_three_candidates_and_an_even_number_of_blocks_of_enough_days(k, t, status, blocks):
    M = matrix(k=k, t=t, seed=7)
    ev = sel.overfitting_probability(M, k, {**sel.CRITERIA, "min_obs": 20})
    assert ev.status == status
    if blocks:
        assert ev.statistic["blocks"] == blocks and blocks % 2 == 0
        edges = np.linspace(0, t, blocks + 1).astype(int)
        assert ev.statistic["pbo"] == pytest.approx(reference_pbo(M.to_numpy(), blocks), abs=1e-12) and np.diff(edges).min() >= 5


def test_the_reality_check_is_not_studentised_and_spa_is():
    rng = np.random.default_rng(3)
    n = 500
    M = pd.DataFrame({"A": rng.normal(0.0006, 0.002, n), "B": rng.normal(0.0012, 0.03, n), "n1": rng.normal(0.0, 0.01, n), "n2": rng.normal(0.0, 0.01, n)}, index=pd.bdate_range("2019-01-01", periods=n))
    rc = sel.reality_check(M, 4, sel.CRITERIA, None, seed=1, n_boot=1000)
    sp = sel.superior_predictive_ability(M, 4, sel.CRITERIA, None, seed=1, n_boot=1000)
    assert rc.statistic["best_candidate"] == "B", "White's statistic is the largest mean: the noisy candidate wins it"
    assert rc.statistic["p_value"] > 0.02 and not rc.statistic["passes"] or rc.statistic["p_value"] > 0.04, "and the noisy candidate's large mean is not significant"
    t_a = np.sqrt(n) * M["A"].mean() / M["A"].std(ddof=1)
    assert sp.statistic["best_t"] == pytest.approx(t_a, rel=0.1), "Hansen's statistic is the largest t-ratio: the quiet candidate with a modest mean wins it"
    assert sp.statistic["p_consistent"] < 0.01 < rc.statistic["p_value"] and sp.statistic["passes"]
    assert "studentised" in " ".join(sp.assumptions) and "stationary bootstrap" in rc.dependence["serial"]


def test_spa_recentres_hopeless_candidates_so_they_cannot_dilute_the_test():
    rng = np.random.default_rng(5)
    n = 500
    base = pd.DataFrame({**{f"n{i}": rng.normal(0.0, 0.01, n) for i in range(3)}, "edge": rng.normal(0.0004, 0.01, n)}, index=pd.bdate_range("2019-01-01", periods=n))
    augmented = base.copy()
    for i in range(30):
        augmented[f"bad{i}"] = rng.normal(-0.003, 0.01, n)
    a = sel.superior_predictive_ability(base, 4, sel.CRITERIA, None, seed=2, n_boot=1000).statistic
    b = sel.superior_predictive_ability(augmented, 34, sel.CRITERIA, None, seed=2, n_boot=1000).statistic
    assert b["p_consistent"] == pytest.approx(a["p_consistent"], abs=0.03), "thirty hopeless candidates leave the recentred p-value alone"
    assert b["p_upper"] > a["p_upper"] + 0.2, "and they inflate the un-recentred one, which is the weakness the recentring removes"
    for seed in range(6):
        X = pd.DataFrame(np.random.default_rng(seed).normal(0.0003, 0.01, (300, 6)), index=pd.bdate_range("2019-01-01", periods=300))
        s = sel.superior_predictive_ability(X, 6, sel.CRITERIA, None, seed=seed, n_boot=500).statistic
        assert s["p_lower"] <= s["p_consistent"] + 1e-12 and s["p_consistent"] <= s["p_upper"] + 1e-12, "Hansen's three p-values are ordered"


def test_fdr_reports_both_corrections_names_the_dependence_assumption_and_passes_on_the_one_valid_under_any_dependence():
    M = matrix(k=9, t=400, seed=8, edge=0.0006)
    ev = sel.false_discovery(M, 9, sel.CRITERIA)
    from src.stats.inference import sharpe_standard_error

    sr = (M.mean() / M.std(ddof=1)).to_numpy()
    p = sps.norm.sf(sr / np.array([sharpe_standard_error(M[c].to_numpy(), "lo") for c in M.columns]))
    m = len(p)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    bh, by = reference_step_up(p, 1.0), reference_step_up(p, harmonic)
    i = int(np.argmin(p))
    assert ev.statistic["q_bh_best"] == pytest.approx(bh[i], abs=1e-12) and ev.statistic["q_by_best"] == pytest.approx(by[i], abs=1e-12)
    assert ev.statistic["q_by_best"] <= ev.statistic["q_bh_best"] * harmonic + 1e-12 and ev.statistic["q_by_best"] >= ev.statistic["q_bh_best"]
    assert ev.statistic["criterion"].startswith("BY") and "valid under any dependence" in ev.dependence["between_tests"] and "assumes positive dependence" in ev.dependence["between_tests"]
    assert ev.statistic["passes"] == (by[i] <= 0.05), "the verdict is the Benjamini-Yekutieli one"


def test_an_adopted_policys_thresholds_become_the_criteria_the_evidence_is_recorded_under():
    from src.ml import policy as pol

    p = pol.ValidationPolicy("p", "v1", {"max_selection_p": 0.01, "min_deflated_probability": 0.9, "max_pbo": 0.3})
    assert sel.criteria_from_policy(p) == {**sel.CRITERIA, "alpha": 0.01, "dsr_min": 0.9, "pbo_max": 0.3}
    assert sel.criteria_from_policy(None) == sel.CRITERIA and sel.criteria_from_policy(pol.ValidationPolicy("q", "v1", {"min_segment_days": 100})) == sel.CRITERIA


# --------------------------------------------------------------------------------------------------------------------- degenerate data
def with_column(M, name, values):
    out = M.copy()
    out[name] = values
    return out


@pytest.mark.parametrize("label,build", [
    ("a candidate that never traded", lambda M: with_column(M, "c1", 0.0)),
    ("a constant positive candidate", lambda M: with_column(M, "c1", 0.001)),
    ("an exact duplicate", lambda M: with_column(M, "c1", M["c0"])),
    ("a giant outlier", lambda M: M.mask(pd.DataFrame(np.arange(M.size).reshape(M.shape) == 7, index=M.index, columns=M.columns), 5.0)),
    ("every candidate identical", lambda M: pd.DataFrame({c: M["c0"] for c in "abc"})),
])
def test_degenerate_candidates_never_produce_an_applied_result_with_a_missing_number_or_a_pass_from_nothing(label, build):
    M = build(matrix(k=5, t=200, seed=0))
    for ev in sel.run_procedures(M, M.shape[1], n_boot=100):
        assert ev.status in ("applied", "not_applicable", "insufficient", "unsupported", "failed"), label
        if ev.status == "applied":
            bad = [k for k, v in ev.statistic.items() if isinstance(v, float) and not np.isfinite(v)]
            assert not bad, f"{label}: {ev.method} is 'applied' with a non-finite {bad}"
        else:
            assert "passes" not in ev.statistic, f"{label}: {ev.method} did not apply and carries a verdict"
    flat = [e for e in sel.run_procedures(M, M.shape[1], n_boot=100) if "excluded_constant_candidates" in e.scope]
    if label in ("a candidate that never traded", "a constant positive candidate"):
        assert flat and flat[0].scope["excluded_constant_candidates"] == ["c1"] and "still count among the 5 recorded tries" in flat[0].limitations[-1]


def test_a_study_of_nothing_but_constant_streams_is_not_applicable_and_a_direct_call_refuses_a_constant_candidate():
    M = matrix(k=4, t=200, seed=1) * 0.0
    out = sel.run_procedures(M, 4, n_boot=100)
    assert {e.status for e in out} == {"not_applicable"} and all("needs the return streams of at least" in e.reason for e in out)
    one_flat = with_column(matrix(k=4, t=200, seed=1), "c1", 0.002)
    for fn in (sel.deflated, sel.false_discovery):
        ev = fn(one_flat, 4, sel.CRITERIA)
        assert ev.status == "unsupported" and "c1" in ev.reason and "Sharpe ratio is undefined" in ev.reason
