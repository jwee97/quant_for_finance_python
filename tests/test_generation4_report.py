"""The Generation 4 report must agree with the registry and tables it summarises.

Same idea as the Generation 2 and 3 consistency tests: the report is a snapshot of numbers a pipeline regenerates, so its headline
claims are parsed or recomputed and compared with the generated files. They skip cleanly when the pipeline has not been run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "reports" / "generation4_report.md"
REGISTRY = ROOT / "experiments" / "registry.jsonl"
MINUS = "−"

pytestmark = pytest.mark.skipif(not (REPORT.exists() and REGISTRY.exists() and (TABLES / "stage26_search_tests.csv").exists()),
                                reason="report, registry or Generation 4 tables not generated")


def _text() -> str:
    return REPORT.read_text(encoding="utf-8")


def _entries() -> list[dict]:
    rows = [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]
    return [d for d in rows if d["stage"] >= "stage26"]


def _signed(x: float, digits: int = 5) -> str:
    return f"{x:+.{digits}f}".replace("-", MINUS)


def test_decision_ledger_matches_the_registry():
    decided = [d for d in _entries() if d["decision"] in ("retain", "reject")]
    retained = sorted(d["stage"] for d in decided if d["decision"] == "retain")
    assert len(decided) == 3 and retained == ["stage27_deep"]
    text = _text()
    assert "**Three decision-bearing hypotheses were declared; one was retained" in text


def test_search_numbers():
    tests = pd.read_csv(TABLES / "stage26_search_tests.csv")
    cash = tests[tests["benchmark"] == "cash"].set_index("test")
    over = pd.read_csv(TABLES / "stage26_overfitting.csv").iloc[0]
    grid = pd.read_csv(TABLES / "stage26_grid_results.csv")
    text = _text()
    assert len(grid) == 1584 and (grid["family"] == "momentum").sum() == 1440
    assert not (cash.loc["reality_check", "p_value"] <= 0.10 and cash.loc["spa_consistent", "p_value"] <= 0.10)
    assert f"| White's Reality Check p | {cash.loc['reality_check', 'p_value']:.3f} |" in text
    assert (f"| Hansen's SPA p (lower / consistent / upper) | {cash.loc['spa_lower', 'p_value']:.3f} / {cash.loc['spa_consistent', 'p_value']:.3f} / "
            f"{cash.loc['spa_upper', 'p_value']:.3f} |") in text
    assert f"{over['pbo']:.2f} |" in text and f"{over['mean_is_best_sharpe']:.2f} / {over['mean_oos_of_best_sharpe']:.2f}" in text
    assert f"| Deflated Sharpe probability of the best rule (1,584 trials) | {over['deflated_sharpe_probability_best']:.2f} |" in text
    assert f"{over['expected_best_sharpe_of_noise']:.2f} |" in text and over["best_net_sharpe"] < over["expected_best_sharpe_of_noise"]
    assert f"slope {_signed(over['degradation_slope'], 2)})" in text
    assert f"{100 * (grid['sharpe'] > 0).mean():.0f}%" in text and f"median {grid['sharpe'].median():.2f}" in text.replace("the median is", "median")


def test_the_grid_fades_with_the_sample():
    share = pd.read_csv(TABLES / "stage26_positive_share_by_sample.csv").set_index("sample")["share_positive_net_sharpe"]
    assert share["development"] > share["validation"] > share["final_holdout"]
    text = _text()
    assert f"{100 * share['development']:.0f}% in the development sample, {100 * share['validation']:.0f}% in validation and {100 * share['final_holdout']:.0f}% in the final holdout" in text


def test_parallel_equals_serial():
    timing = pd.read_csv(TABLES / "stage26_timing.csv").iloc[0]
    assert timing["max_abs_serial_vs_parallel"] < 1e-12 and timing["tasks_checked"] >= 24
    serial = timing["serial_s"] / timing["tasks_checked"] * timing["grid_rules"]
    text = _text()
    assert f"{timing['grid_wall_s']:.0f} s for the grid on four cores, against {serial:.0f} s extrapolated" in text


def test_deep_learning_numbers_and_the_controls_the_report_relies_on():
    table = pd.read_csv(TABLES / "stage27_forecast_tests.csv").set_index("model")
    post = pd.read_csv(TABLES / "stage27_posthoc_controls.csv").set_index(["a", "b"])
    text = _text()
    assert bool(table.loc["tsmixer", "passes"]) and not bool(table.loc["patchtst", "passes"])
    for model, label in (("patchtst", "Patch transformer"), ("tsmixer", "MLP-mixer"), ("linear_window", "Linear on the window (control)")):
        row = next(l for l in text.splitlines() if l.startswith(f"| {label}") or l.startswith(f"| **{label}"))
        assert f"{table.loc[model, 'mean_crps_model']:.5f}" in row, (model, row)
    assert f"{table.loc['tsmixer', 'p_value']:.3f}" in text
    # the qualification: the ridge is worse than doing nothing, and the mixer is not better than doing nothing
    assert post.loc[("frozen_ridge", "zero"), "mean_crps_difference_a_minus_b"] > 0
    assert post.loc[("frozen_ridge", "historical_mean"), "p_value"] < 0.05
    assert post.loc[("tsmixer", "zero"), "p_value"] > 0.10 and post.loc[("tsmixer", "historical_mean"), "p_value"] > 0.10
    for (a, b), row in post.iterrows():
        if (a, b) in {("frozen_ridge", "zero"), ("frozen_ridge", "historical_mean"), ("tsmixer", "zero"), ("tsmixer", "historical_mean"), ("patchtst", "historical_mean")}:
            assert f"{_signed(row['mean_crps_difference_a_minus_b'])}" in text, (a, b)
    counts = pd.read_csv(TABLES / "stage27_parameter_counts.csv", index_col=0)["parameters"]
    assert f"{counts['patchtst']:,}" in text and f"{counts['tsmixer']:,}" in text


def test_text_numbers():
    declared = pd.read_csv(TABLES / "stage28_clark_west_declared.csv")
    matched = pd.read_csv(TABLES / "stage28_clark_west_matched_posthoc.csv")
    features = pd.read_csv(TABLES / "stage28_features_by_statement.csv")
    agreement = pd.read_csv(TABLES / "stage28_action_agreement.csv", index_col=0)
    text = _text()
    assert not bool(declared["bh_significant"].any()) and len(declared) == 5
    assert f"smallest p = {declared['p_value'].min():.2f}" in text
    assert f"{100 * declared['oos_r2'].mean():.1f}%".replace("-", MINUS) in text or f"{100 * declared['oos_r2'].mean():.1f}".replace("-", MINUS) in text
    assert f"over {int(declared['n_obs'].min())} months" in text
    assert len(features) == 171 and "171 FOMC statements" in text.replace("(171 statements)", "171 FOMC statements") or "171 statements" in text
    rate = sum(agreement.loc[i, str(int(i))] if str(int(i)) in agreement.columns else 0 for i in agreement.index) / agreement.to_numpy().sum()
    assert f"{100 * rate:.1f}%" in text


def test_database_checks_and_identity():
    checks = pd.read_csv(TABLES / "stage29_acceptance_checks.csv")
    assert len(checks) == 6 and bool(checks["passed"].all())
    counts = pd.read_csv(TABLES / "stage29_table_counts.csv").set_index("table")["rows"]
    assert counts["grid_results"] == 1584
    text = _text()
    from src.utils.config import load_config

    config = load_config()
    for scope in ("core", "all", "gen3", "gen4"):
        assert config.fingerprint(scope) in text, scope
    assert re.search(r"Data version:\*\* `[0-9a-f]{6,}`", text)


def test_figures_exist_and_every_stage_has_a_section_and_the_omissions_are_listed():
    figures = ROOT / "reports" / "figures"
    for number in range(52, 59):
        matches = list(figures.glob(f"fig{number}_*.png"))
        assert matches and matches[0].with_suffix(".txt").exists(), number
    text = _text()
    assert "Figures 52-58" in text
    for heading in ("(Stage 26)", "(Stage 27)", "(Stage 28)", "(Stage 29)"):
        assert heading in text
    for omitted in ("Chronos", "Hydra", "explainable machine learning", "ray executor backend", "actual cloud deployment", "any run of a language model"):
        assert omitted in text
