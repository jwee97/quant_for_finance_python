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
