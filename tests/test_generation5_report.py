"""The Generation 5 report must agree with the registry and tables it summarises.

Same idea as the Generation 2-4 consistency tests: the report is a snapshot of numbers a pipeline regenerates, so its headline claims are parsed or
recomputed and compared with the generated files. They skip cleanly when the pipeline has not been run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "reports" / "generation5_report.md"
REGISTRY = ROOT / "experiments" / "registry.jsonl"

pytestmark = pytest.mark.skipif(not (REPORT.exists() and REGISTRY.exists() and (TABLES / "stage40_factor_regressions.csv").exists()),
                                reason="report, registry or Generation 5 tables not generated")


def _text() -> str:
    return REPORT.read_text(encoding="utf-8")


def _entries() -> list[dict]:
    return [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]


def _g5() -> list[dict]:
    return [d for d in _entries() if "stage30" <= d["stage"] < "stage41"]


def _t(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, **kw)


def test_decision_ledger_matches_the_registry():
    decided = [d for d in _g5() if d["decision"] in ("retain", "reject")]
    retained = sorted(d["stage"] for d in decided if d["decision"] == "retain")
    assert len(decided) == 14, len(decided)
    assert retained == ["stage30_library", "stage31_adaptive", "stage31_adaptive", "stage37_causal"], retained
    text = _text()
    assert "Fourteen decision-bearing hypotheses" in text and "four were retained" in text
    descriptive = {d["stage"] for d in _g5() if d["decision"] == "record"}
    assert {"stage34_foundation", "stage35_explain", "stage39_power", "stage40_factors"} <= descriptive


def test_stage30_numbers():
    ov = _t("stage30_overfitting.csv").iloc[0]
    tests = _t("stage30_search_tests.csv")
    text = _text()
    assert int(ov["specifications"]) == 42 and ov["best"] == "dual_momentum"
    assert f"{ov['best_net_sharpe']:.2f}" in text and f"{ov['equal_weight_sharpe']:.2f}" in text
    rc_cash = float(tests[(tests.benchmark == "cash") & (tests.test == "reality_check")]["p_value"].iloc[0])
    rc_ew = float(tests[(tests.benchmark == "equal_weight") & (tests.test == "reality_check")]["p_value"].iloc[0])
    assert rc_cash < 0.01 < 0.5 < rc_ew
    assert f"p = {rc_ew:.2f}" in text and f"p = {rc_cash:.4f}" in text
    assert f"{ov['pbo']:.3f}" in text
    comb = _t("stage30_combination.csv", index_col=0)
    for name in ("combination:equal", "combination:confidence", "combination:ic_weighted", "combination:cost_aware"):
        assert f"{comb.loc[name, 'sharpe']:.2f}" in text, name
    tests2 = _t("stage30_combination_tests.csv")
    row = tests2[(tests2.a == "combination:confidence") & (tests2.b == "combination:equal")].iloc[0]
    assert row["difference"] < 0 and bool(row["bh_significant"]) and f"{row['difference']:+.2f}".replace("-", "-") in text.replace("−", "-")


def test_stage31_numbers():
    perf = _t("stage31_allocation_performance.csv", index_col=0)
    tests = _t("stage31_allocation_tests.csv")
    placebo = next(d for d in _g5() if d["stage"] == "stage31_adaptive" and "regime path" in d["hypothesis"])["results"]
    text = _text()
    assert f"{perf.loc['regime switch (soft)', 'sharpe']:.2f}" in text and f"{perf.loc['regime switch (hard)', 'sharpe']:.2f}" in text
    assert not bool(tests["passes"].any())
    assert placebo["actual_sharpe"] > placebo["placebo_max"] and f"{placebo['one_sided_p']:.3f}" in text
    sig = _t("stage31_signal_tests.csv").iloc[0]
    assert 0.05 < sig["p_value"] < 0.10 and f"{sig['p_value']:.4f}" in text
    risk = _t("stage31_risk_tests.csv")
    assert (risk["p_value"] > 0.9).all()
    post = _t("stage31_signal_sensitivity_posthoc.csv")
    assert len(post) == 9


def test_crypto_numbers():
    perf = _t("stage32_performance.csv", index_col=0)
    tests = _t("stage32_tests.csv")
    text = _text()
    assert not bool(tests["passes"].any())
    assert f"{perf.loc['always-on carry', 'sharpe']:.2f}" in text and f"{perf.loc['funding_carry', 'sharpe']:.2f}" in text
    basis = tests[tests.a == "basis_reversion"].iloc[0]
    assert basis["difference"] < -3 and f"{basis['difference']:.2f}".replace("-", "-") in text.replace("−", "-")


def test_frontier_numbers():
    t = _t("stage33_forecast_tests.csv", index_col=0)
    b = _t("stage33_bayesian.csv", index_col=0)
    text = _text()
    assert not bool(t["passes"].any()) and (t["n_origins"] == 187).all()
    assert (t["p_vs_ridge"] < 0.01).all() and (t["diff_vs_ridge"] < 0).all(), "'all four beat the ridge'"
    assert f"{t.loc['nbeats', 'mean_crps_difference']:.6f}".replace("-", "-") in text.replace("−", "-")
    assert (b["mean_sigma_ratio"] < 1.01).all() and (b["mean_crps_difference"] >= 0).all()
    params = _t("stage33_parameter_counts.csv", index_col=0)
    for k in ("nbeats", "nhits", "timemixer", "gat"):
        assert f"{int(params.loc[k, 'parameters']):,}" in text, k
    vs = _t("stage33_vs_stage27.csv")
    assert (vs[vs.against == "patchtst"].set_index("model").loc[["nbeats", "nhits", "timemixer"], "p_value"] < 0.02).all()


def test_foundation_numbers():
    t = _t("stage34_tests.csv", index_col=0)
    s = _t("stage34_summary.csv").iloc[0]
    text = _text()
    assert t.loc["historical_mean", "mean_crps_difference"] > 0 and t.loc["historical_mean", "p_value"] < 0.001
    assert f"{s['mean_rank_ic']:.3f}" in text and f"{100 * s['share_positive_forecasts']:.0f}% of forecasts positive" in text
    assert (TABLES / "stage34_model.json").exists()


def test_explain_and_calibration_numbers():
    chk = _t("stage35_identity_checks.csv")
    corr = _t("stage35_importance_rank_correlation.csv", index_col=0)
    summary = _t("stage35_calibration_summary.csv", index_col=0)
    tests = _t("stage35_calibration_tests.csv")
    text = _text()
    assert (chk.loc[chk.kind == "shapley_efficiency", "max_gap"] < 1e-6).all() and (chk.loc[chk.kind == "ig_completeness", "max_relative_gap"] < 1e-2).all()
    assert int(chk["n"].iloc[0]) == 202
    off = corr.to_numpy()[~__import__("numpy").eye(len(corr), dtype=bool)]
    assert f"{off.min():.2f}".replace("-", "-") in text.replace("−", "-") and f"{off.max():.2f}" in text
    for arm in ("raw", "platt", "isotonic", "stage19"):
        assert f"{summary.loc[arm, 'log_loss']:.4f}" in text, arm
    declared = tests[tests.status == "declared"].iloc[0]
    assert declared["mean_log_loss_difference"] > 0 and f"p = {declared['p_value']:.2f}" in text
    assert summary["log_loss"].drop("stage19").idxmin() in ("raw", "isotonic_clipped_posthoc") and summary.loc["raw", "log_loss"] < summary.loc["platt", "log_loss"]


def test_diffusion_numbers():
    s = _t("stage36_var_summary.csv")
    tests = _t("stage36_var_tests.csv")
    real = _t("stage36_scenario_realism.csv", index_col=0)
    text = _text()
    eq = s[(s.book == "equal") & (s.level == 0.05)].set_index("method")
    for m in ("diffusion", "gaussian", "bootstrap"):
        assert int(eq.loc[m, "exceedances"]) in range(5, 20)
        assert f"{eq.loc[m, 'mean_pinball']:.5f}" in text, m
    declared = tests[tests.status == "declared"].set_index("b")
    assert not bool(declared["passes"].astype(bool).any())
    assert f"p = {declared.loc['gaussian', 'p_value']:.2f}" in text and f"p = {declared.loc['bootstrap', 'p_value']:.2f}" in text
    assert f"{real.loc['diffusion', 'excess_kurtosis_mean']:.2f}" in text and f"{real.loc['diffusion', 'corr_distance']:.3f}" in text


def test_causal_numbers():
    v = _t("stage37_validation.csv").set_index(["world", "estimator"])
    eff = _t("stage37_fomc_effects.csv", index_col=0)
    text = _text()
    assert abs(v.loc[("endogenous_iv", "2SLS"), "bias"]) < 0.06 and v.loc[("endogenous_iv", "2SLS"), "coverage_95"] > 0.9
    assert v.loc[("did_diverging", "DiD"), "bias"] > 0.5 and abs(v.loc[("did_parallel", "DiD"), "bias"]) < 0.05
    assert v.loc[("confounded_plr", "DML, GBM nuisance"), "bias"] > 0.3, "the declared DML validation failed and the report says so"
    assert abs(v.loc[("plr_oracle_posthoc", "DML final step, true nuisance"), "bias"]) < 0.03
    assert f"{v.loc[('confounded_plr', 'DML, GBM nuisance'), 'bias']:.2f}" in text
    for asset in ("SPY", "TLT"):
        assert f"{eff.loc[asset, 'dml_gbm_theta']:+.3f}".replace("-", "-") in text.replace("−", "-"), asset
    assert bool(eff["bh_significant"].any()) and int(eff["n"].iloc[0]) == 168
    did = _t("stage37_fomc_did.csv").iloc[0]
    assert f"{did['theta']:.3f}".replace("-", "-") in text.replace("−", "-")


def test_reinforcement_learning_numbers():
    perf = _t("stage38_performance.csv", index_col=0)
    tests = _t("stage38_tests.csv").iloc[0]
    gap = _t("stage38_overfitting_gap.csv")
    null = _t("stage38_random_policy_null.csv")
    text = _text()
    assert tests["difference"] < 0 and tests["p_value"] < 0.01
    assert f"{perf.loc['RL policy (3-seed average)', 'sharpe']:.3f}" in text and f"{perf.loc['equal weight (M0)', 'sharpe']:.3f}" in text
    assert gap["train_gain"].mean() > 0 > gap["oos_gain"].mean()
    assert f"{gap['train_gain'].mean():.4f}" in text and f"{gap['oos_gain'].mean():.4f}".replace("-", "-") in text.replace("−", "-")
    assert abs(null["null_monthly_sharpe"].mean() - 0.73) < 0.05


def test_power_numbers():
    mde = _t("stage39_sharpe_mde.csv")
    fm = _t("stage39_forecast_mde.csv")
    harm = _t("stage39_forecast_harm_posthoc.csv")
    sp = _t("stage39_sharpe_power.csv")
    text = _text()
    at = lambda te, y: float(mde[(mde.tracking_error == te) & (mde.years == y)]["min_detectable_sharpe_difference"].iloc[0])
    assert f"{at(0.06, 15.7):.2f}" in text and f"{at(0.03, 15.7):.2f}" in text and f"{at(0.06, 8.0):.2f}" in text
    assert mde[(mde.tracking_error == 0.06) & (mde.years == 4.0)]["min_detectable_sharpe_difference"].isna().all()
    for o in (100, 187, 400):
        assert f"{100 * float(fm[fm.origins == o]['min_detectable_r2'].iloc[0]):.2f}".rstrip("0").rstrip(".") in text
    zero = sp[sp.sharpe_difference == 0]["power"]
    assert 0.07 < zero.min() and zero.max() < 0.16 and f"{zero.mean():.3f}" in text
    row = harm[(harm.noise_scale == 0.1) & (harm.origins == 187)].iloc[0]
    assert f"{100 * row['rate_significantly_worse']:.0f}%" in text


def test_factor_numbers():
    t = _t("stage40_factor_regressions.csv", index_col=0)
    text = _text()
    assert t.loc["M3_momentum", "t_Mom"] > 5 and (t["alpha_p"] > 0.05).all()
    for book in t.index:
        assert f"{t.loc[book, 'r_squared']:.2f}" in text, book
    assert f"{t.loc['M3_momentum', 't_Mom']:.1f}" in text and f"{t.loc['M0_equal_weight', 'beta_Mkt-RF']:.2f}" in text


def test_report_names_its_identity_and_every_figure_has_a_question():
    from src.utils.config import load_config

    config = load_config()
    text = _text()
    for scope in ("core", "all", "gen3", "gen4", "gen5"):
        assert config.fingerprint(scope) in text, scope
    figures = ROOT / "reports" / "figures"
    if not figures.exists():
        pytest.skip("figures not generated")
    for number in range(59, 80):
        matches = list(figures.glob(f"fig{number}_*.png"))
        assert matches, f"figure {number} missing"
        assert matches[0].with_suffix(".txt").exists()
    assert "Figures 59-79" in text
    assert re.search(r"`[0-9a-f]{12}`", text)


def test_every_stage_has_a_section_and_the_omissions_are_listed():
    text = _text()
    for heading in ("(Stage 30)", "(Stage 31)", "(Stage 32)", "(Stage 33)", "(Stage 34", "(Stage 35)", "(Stage 36)", "(Stage 37)", "(Stage 38)", "(Stage 39)", "(Stage 40"):
        assert heading in text, heading
    for omitted in ("No language model was called", "TimesFM", "Dispersion trading", "cross-exchange", "Binance", "Docker", "causal forests"):
        assert omitted.lower() in text.lower(), omitted
    for tag in ("<<GEN5_FP>>", "<<RUNTIME>>", "<<LEDGER>>", "<<TESTS>>"):
        assert tag not in text, f"unfilled placeholder {tag}"
