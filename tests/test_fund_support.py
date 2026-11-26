"""What the importer supports against what has been demonstrated: a format is never reported as accurate on real filings without a stored run on real, person-verified, held-out ones."""

from __future__ import annotations

import json

import pytest

from src.fundamentals.importer import support


def _row(m, key):
    return next(r for r in m["formats"] if r["id"] == key)


def test_the_stored_runs_show_synthetic_evidence_only_and_no_format_is_demonstrated_on_real_filings():
    m = support.matrix()
    assert m["formats"] and m["real_runs"] == []
    assert all(r["status"] != "demonstrated_on_real_filings" for r in m["formats"])
    pdf = _row(m, "pdf_text")
    assert pdf["status"] == "demonstrated_on_synthetic_only" and pdf["synthetic"]["cases"] >= 1 and pdf["real_verified_heldout_filings"] == 0
    assert "not measured on any real filing" in pdf["claim"] and "unverified" in pdf["claim"]
    assert _row(m, "xbrl")["status"] == "supported_unbenchmarked" and "never benchmarked" in _row(m, "xbrl")["claim"]
    assert "No format is reported as demonstrated on real filings" in m["note"]


def test_a_missing_dependency_blocks_a_format_and_says_what_is_needed(monkeypatch):
    monkeypatch.setattr(support.importlib.util, "find_spec", lambda name: None if name == "openpyxl" else object())
    r = _row(support.matrix(), "xlsx")
    assert r["status"] == "blocked" and r["blocked_by"] == ["openpyxl"] and "needs openpyxl" in r["claim"]


@pytest.mark.parametrize("verified,count,expected", [(True, 5, "demonstrated_on_real_filings"), (True, 4, "demonstrated_on_synthetic_only"), (False, 8, "demonstrated_on_synthetic_only")])
def test_five_verified_heldout_real_filings_are_needed_to_call_a_format_demonstrated(tmp_path, monkeypatch, verified, count, expected):
    cases = [{"format": "pdf", "split": "heldout", **({"verified_by": "a person"} if verified else {})} for _ in range(count)]
    (tmp_path / "real_run.json").write_text(json.dumps({"kind": "real", "cases": cases}))
    (tmp_path / "heldout_post_fix.json").write_text(json.dumps({"summary": {"by_format": {"pdf": {"cases": 20, "recall": 0.8, "critical_accuracy": 0.8, "wrong": 1, "confident_errors": 0, "statement_coverage": 1.0}}}}))
    monkeypatch.setattr(support, "RESULTS", tmp_path)
    m = support.matrix()
    assert _row(m, "pdf_text")["status"] == expected and m["real_runs"] == ["real_run.json"]
    if expected == "demonstrated_on_real_filings":
        assert _row(m, "pdf_text")["real_verified_heldout_filings"] == 5


def test_a_results_file_that_is_not_marked_real_does_not_count(tmp_path, monkeypatch):
    (tmp_path / "real_structures_after.json").write_text(json.dumps([{"case": "x"}]))
    (tmp_path / "real_other.json").write_text(json.dumps({"kind": "synthetic", "cases": [{"format": "pdf", "split": "heldout", "verified_by": "a person"}] * 9}))
    monkeypatch.setattr(support, "RESULTS", tmp_path)
    assert support.matrix()["real_runs"] == []
