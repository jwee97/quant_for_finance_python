"""The Generation 2 report must agree with the registry and tables it summarises.

Same idea as the Generation 1 consistency tests: the report is a snapshot of numbers a pipeline
regenerates, so its headline claims are parsed or recomputed and compared with the generated files.
They skip cleanly when the pipeline has not been run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "reports" / "generation2_report.md"
REGISTRY = ROOT / "experiments" / "registry.jsonl"
ERRATA = ROOT / "reports" / "errata"
MINUS = "−"

pytestmark = pytest.mark.skipif(not (REPORT.exists() and REGISTRY.exists()), reason="report or registry not generated")


def _text() -> str:
    return REPORT.read_text(encoding="utf-8")


def _registry() -> dict:
    return {d["experiment_id"]: d for d in (json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip())}


def _gen2():
    return {k: v for k, v in _registry().items() if "stage15" <= v["stage"] < "stage21"}


def test_decision_ledger_matches_the_registry():
    """'23 decision-bearing hypotheses, three retained' and which three."""
    entries = _gen2()
    decided = {k: v for k, v in entries.items() if v["decision"] in ("retain", "reject")}
    retained = sorted(k for k, v in decided.items() if v["decision"] == "retain")
    assert len(decided) == 23, len(decided)
    assert [entries[k]["stage"] for k in retained] == ["stage16_regimes", "stage17_dynamic_covariance", "stage19_probabilistic"]
    text = _text()
    assert "Of 23 decision-bearing hypotheses" in text and "**three** were retained" in text


def test_retained_results_carry_the_numbers_the_report_quotes():
    entries = _gen2()
    dcc = next(v for v in entries.values() if v["decision"] == "retain" and v["stage"].startswith("stage17"))["results"]
    kelly = next(v for v in entries.values() if v["decision"] == "retain" and v["stage"].startswith("stage19"))["results"]
    text = _text()
    assert f"{100 * (1 - dcc['volatility_ratio']):.1f}% less volatile" in text
    assert f"p = {dcc['p_value']:.3f}" in text
    assert f"+{kelly['sharpe_difference']:.2f} Sharpe" in text
    # the confounding controls the report relies on must still say what it says
    controls = pd.read_csv(TABLES / "stage19_posthoc_sizing_controls.csv").set_index(["strategy", "benchmark"])
    assert controls.loc[("fractional_kelly", "passive_equal_weight"), "difference"] < 0
    assert controls.loc[("fractional_kelly", "gaussian_sign_only"), "p_value"] > 0.10


def test_nothing_in_the_macro_stage_is_significant():
    cw = pd.read_csv(TABLES / "stage15_clark_west.csv")
    both = cw[cw["comparison"] == "both_vs_price"]
    assert len(both) == 5 and not bool(both["bh_significant"].any())
    assert not bool(cw["bh_significant"].any())


def test_drift_fix_section_matches_the_errata_files():
    errata = ERRATA / "drift_fix_final_comparison.csv"
    if not errata.exists():
        pytest.skip("errata not present")
    frame = pd.read_csv(errata, index_col=0)
    change = frame["sharpe_change"]
    text = _text()
    assert f"[{MINUS}{abs(change.min()):.3f}, +{change.max():.3f}]" in text
    before = frame["sharpe_before_fix"].sort_values(ascending=False).index.tolist()
    after = frame["sharpe_after_fix"].sort_values(ascending=False).index.tolist()
    assert before == after, "the ranking was reported as unchanged"
    for row_label, key in (("M1 inverse volatility", "M1_inverse_vol"), ("M3 momentum", "M3_momentum")):
        line = next(l for l in text.splitlines() if l.startswith(f"| {row_label} |"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert abs(float(cells[2].replace(MINUS, "-")) - frame.loc[key, "sharpe_after_fix"]) <= 0.005 + 1e-9
    breakeven = pd.read_csv(ERRATA / "drift_fix_breakeven_costs.csv", index_col=0)
    assert f"{breakeven.loc['M3_momentum', 'after_fix']:.1f}" in text


def test_the_generation_one_tables_agree_with_the_errata_after_column():
    errata = ERRATA / "drift_fix_final_comparison.csv"
    if not errata.exists() or not (TABLES / "stage12_final_comparison.csv").exists():
        pytest.skip("not generated")
    now = pd.read_csv(TABLES / "stage12_final_comparison.csv", index_col=0)["sharpe"]
    recorded = pd.read_csv(errata, index_col=0)["sharpe_after_fix"]
    assert (now - recorded).abs().max() < 5e-6, "the pipeline no longer reproduces the post-fix Generation 1 numbers"


def test_attribution_identities_hold_and_the_books_share_one_window():
    ident = pd.read_csv(TABLES / "stage20_identity_checks.csv", index_col=0)
    assert len(ident) == 28 and float(ident["max_abs_error"].max()) < 1e-9
    summary = pd.read_csv(TABLES / "stage20_active_return_summary.csv", index_col=0)
    assert summary["months"].nunique() == 1 and int(summary["months"].iloc[0]) == 233
    assert summary["benchmark_cumulative_return"].nunique() == 1
    linked = summary[["allocation", "selection", "interaction"]].sum(axis=1)
    assert (linked - summary["cumulative_active_return"]).abs().max() < 5e-8   # the CSV keeps 8 decimals
    assert f"{ident['max_abs_error'].max():.1e}" in _text()


def test_report_names_its_identity_and_every_generation_two_figure_has_a_question():
    text = _text()
    assert re.search(r"Data version:\*\* `[0-9a-f]{6,}`", text) and re.search(r"\(core: `[0-9a-f]{6,}`", text)
    figures = ROOT / "reports" / "figures"
    if not figures.exists():
        pytest.skip("figures not generated")
    for number in range(28, 42):
        matches = list(figures.glob(f"fig{number}_*.png"))
        assert matches, f"figure {number} missing"
        assert matches[0].with_suffix(".txt").exists()
    assert "Figures 28-41" in text


def test_every_stage_has_a_section_and_the_omissions_are_listed():
    text = _text()
    for heading in ("(Stage 15)", "(Stage 16)", "(Stage 17)", "(Stage 18)", "(Stage 19)", "(Stage 20)"):
        assert heading in text
    for omitted in ("Generation 3", "Generation 4", "Wishart", "factor-model performance attribution", "Priority 20"):
        assert omitted in text
