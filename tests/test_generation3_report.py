"""The Generation 3 report must agree with the registry and tables it summarises.

Same idea as the Generation 2 consistency tests: the report is a snapshot of numbers a pipeline regenerates, so its
headline claims are parsed or recomputed and compared with the generated files. They skip cleanly when the
pipeline has not been run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "reports" / "generation3_report.md"
REGISTRY = ROOT / "experiments" / "registry.jsonl"
MINUS = "−"

pytestmark = pytest.mark.skipif(not (REPORT.exists() and REGISTRY.exists() and (TABLES / "stage25_capacity.csv").exists()),
                                reason="report, registry or Generation 3 tables not generated")


def _text() -> str:
    return REPORT.read_text(encoding="utf-8")


def _entries() -> list[dict]:
    rows = [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]
    return [d for d in rows if d["stage"] >= "stage21"]


def _signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}".replace("-", MINUS)


def test_decision_ledger_matches_the_registry():
    """'9 decision-bearing hypotheses, two retained' and which two."""
    decided = [d for d in _entries() if d["decision"] in ("retain", "reject")]
    retained = sorted(d["stage"] for d in decided if d["decision"] == "retain")
    assert len(decided) == 9, len(decided)
    assert retained == ["stage21_bayesian", "stage25_execution"]
    text = _text()
    assert "Of **9** decision-bearing hypotheses" in text and "**two** were retained" in text


def test_bayesian_numbers():
    entry = next(d for d in _entries() if d["stage"] == "stage21_bayesian" and "posterior-predictive book beats" in d["hypothesis"])["results"]
    text = _text()
    assert f"{entry['sharpe_bayes_predictive']:.3f}" in text and f"{entry['sharpe_mvo_sample']:.3f}" in text
    assert f"(+{entry['diff_bayes_predictive_vs_mvo_sample']:.3f}, p = {entry['p_bayes_predictive_vs_mvo_sample']:.2f})" in text
    assert f"({_signed(entry['diff_bayes_predictive_vs_M2_risk_parity'])}, p = {entry['p_bayes_predictive_vs_M2_risk_parity']:.3f})" in text
    stab = pd.read_csv(TABLES / "stage21_stability_summary.csv", index_col=0)
    assert f"{stab.loc['bayes_predictive', 'mean_pairwise_l1']:.2f}" in text and f"{stab.loc['mvo_sample', 'mean_pairwise_l1']:.2f}" in text
    assert f"{stab.loc['bayes_predictive', 'mean_effective_n']:.1f}" in text
    # the claim that Bayes-Stein is not more stable must stay true
    assert stab.loc["bayes_stein", "mean_pairwise_l1"] > 0.9 * stab.loc["mvo_sample", "mean_pairwise_l1"]


def test_online_learning_numbers():
    tests = pd.read_csv(TABLES / "stage22_forecast_tests.csv").set_index("model")
    text = _text()
    assert len(tests) == 3 and not bool(tests["passes"].any())
    assert bool(tests.loc["nlms", "bh_significant"]) and tests.loc["nlms", "mean_crps_difference"] > 0     # significant, in the wrong direction
    assert tests.loc["rls", "p_value"] > 0.10
    assert f"p = {tests.loc['rls', 'p_value']:.2f}" in text
    assert f"{tests.loc['nlms', 'mean_crps_model']:.4f}" in text and f"{tests.loc['kalman', 'mean_crps_model']:.4f}" in text
    agg = pd.read_csv(TABLES / "stage22_aggregation_performance.csv", index_col=0)
    assert f"{agg.loc['Hedge', 'sharpe']:.3f}" in text and f"{agg.loc['inverse-vol blend', 'sharpe']:.3f}" in text
    regret = pd.read_csv(TABLES / "stage22_regret.csv")
    assert regret.shape[0] >= 1


def test_nothing_in_the_non_price_stage_is_significant():
    declared = pd.read_csv(TABLES / "stage23_clark_west_declared.csv")
    assert len(declared) == 5 and not bool(declared["bh_significant"].any())
    text = _text()
    assert f"smallest p = {declared['p_value'].min():.3f}" in text
    assert f"{100 * declared['oos_r2'].mean():+.1f}%".replace("-", MINUS) in text
    positioning = pd.read_csv(TABLES / "stage23_positioning_tests.csv")
    assert not bool(positioning["bh_significant"].any()) and len(positioning) == 6
    assert f"smallest p = {positioning['p_beta'].min():.2f}" in text
    matched = pd.read_csv(TABLES / "stage23_clark_west_matched_posthoc.csv")
    assert matched["oos_r2"].mean() > 0 and not bool(matched["bh_significant"].any()), "the matched-sample sign flip is reported in the text"


def test_combination_engine_numbers():
    tests = pd.read_csv(TABLES / "stage24_tests.csv")
    perf = pd.read_csv(TABLES / "stage24_performance.csv", index_col=0)
    text = _text()
    assert not bool(tests["passes"].all())
    for _, r in tests.iterrows():
        assert f"{_signed(r['difference'])} (p = {r['p_value']:.2f})" in text, (r["a"], r["b"])
    for name in ("momentum", "cost_aware", "equal", "hedge"):
        assert f"{perf.loc[name, 'sharpe']:.3f}".replace("-", MINUS) in text.replace("-", MINUS)
    assert perf.loc["momentum", "sharpe"] > perf.loc["cost_aware", "sharpe"], "'below the Generation 1 momentum book'"
    corr = pd.read_csv(TABLES / "stage24_signal_correlation.csv", index_col=0)
    off = corr.where(~pd.DataFrame(__import__("numpy").eye(len(corr), dtype=bool), index=corr.index, columns=corr.columns)).abs().max().max()
    assert off < 0.4 and "largest 0.36" in text


def test_execution_identity_and_capacity_numbers():
    by_aum = pd.read_csv(TABLES / "stage25_sharpe_by_aum.csv", index_col=0)
    change = pd.read_csv(TABLES / "stage25_allocator_change.csv", index_col=0)
    capacity = pd.read_csv(TABLES / "stage25_capacity.csv", index_col=0)
    text = _text()
    zero = by_aum.columns[0]
    assert float(zero) == 0.0
    # the zero-AUM column IS the Generation 1 linear-cost net Sharpe
    assert (by_aum[zero].reindex(capacity.index) - capacity["linear_sharpe"]).abs().max() < 1e-6
    assert float(change["change"].abs().max()) <= 0.05 and change["change"].abs().idxmax() == "M9_mean_cvar"
    assert f"largest change {_signed(change['change'].min())}" in text
    for book in ("M0_equal_weight", "M3_momentum", "M5_momentum_plus_mr"):
        for column in by_aum.columns[:5]:
            assert f"{by_aum.loc[book, column]:.3f}".replace("-", MINUS) in text.replace("-", MINUS), (book, column)
    assert f"${float(capacity.loc['M3_momentum', 'capacity_usd']) / 1e6:.0f}m" in text
    assert capacity.loc[["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity", "M9_mean_cvar", "M11_hrp", "M12_herc"], "capacity_usd"].eq("above grid").all()
    sensitivity = pd.read_csv(TABLES / "stage25_sharpe_Y_sensitivity_1e9.csv", index_col=0)
    shift = sensitivity.loc["M9_mean_cvar", "2.000000"] - capacity.loc["M9_mean_cvar", "linear_sharpe"]
    assert shift < -0.05, "the report says the bound fails at the doubled coefficient"
    assert _signed(shift) in text


def test_band_numbers():
    tests = pd.read_csv(TABLES / "stage25_band_tests.csv").set_index("book")
    text = _text()
    assert not bool(tests["passes"].all()) and not bool(tests["bh_significant"].any())
    for book in ("M3_momentum", "M4_mean_reversion", "M5_momentum_plus_mr"):
        assert f"{_signed(tests.loc[book, 'difference'])} " in text
    sweep = pd.read_csv(TABLES / "stage25_band_sweep.csv")
    cut = {b: 1 - float(sweep[(sweep.book == b) & (sweep.band == 0.01)]["ann_turnover"].iloc[0]) / float(sweep[(sweep.book == b) & (sweep.band == 0.0)]["ann_turnover"].iloc[0])
           for b in sweep.book.unique()}
    assert 0.003 < min(cut.values()) and max(cut.values()) < 0.025, cut


def test_report_names_its_identity_and_every_figure_has_a_question():
    text = _text()
    from src.utils.config import load_config
    config = load_config()
    for scope in ("core", "all", "gen3"):
        assert config.fingerprint(scope) in text, scope
    assert re.search(r"Data version:\*\* `[0-9a-f]{6,}`", text)
    figures = ROOT / "reports" / "figures"
    if not figures.exists():
        pytest.skip("figures not generated")
    for number in range(42, 52):
        matches = list(figures.glob(f"fig{number}_*.png"))
        assert matches, f"figure {number} missing"
        assert matches[0].with_suffix(".txt").exists()
    assert "Figures 42-51" in text


def test_every_stage_has_a_section_and_the_omissions_are_listed():
    text = _text()
    for heading in ("(Stage 21)", "(Stage 22)", "(Stage 23)", "(Stage 24)", "(Stage 25)"):
        assert heading in text
    for omitted in ("Generation 4", "explainable machine learning", "ETF flows", "Almgren-Chriss optimal trajectory", "Priority 20"):
        assert omitted in text
