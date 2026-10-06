"""docs/feature_audit.md must not cite files or symbols that do not exist, and every audited feature must appear in both tables."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "docs" / "feature_audit.md").read_text(encoding="utf-8")
REF = re.compile(r"`((?:src|tests|experiments|examples|docs|config)/[\w./-]+?\.(?:py|yaml|md))(?:::(\w+))?`")


def test_every_cited_path_and_symbol_exists():
    refs = REF.findall(TEXT)
    assert len(refs) > 60
    for path, symbol in refs:
        assert (ROOT / path).exists(), path
        if symbol:
            assert re.search(rf"(class|def) {symbol}\b|^{symbol}\b", (ROOT / path).read_text(encoding="utf-8"), re.M), (path, symbol)


def test_every_feature_has_a_before_row_and_an_after_row_with_a_legal_status():
    before = TEXT.split("## Phase 1")[1].split("## Phase 5")[0]
    after = TEXT.split("## Phase 5")[1].split("## Exploratory")[0]
    rows_before = [r for r in before.splitlines() if r.startswith("| ") and not r.startswith("| Feature") and not r.startswith("|---")]
    rows_after = [r for r in after.splitlines() if r.startswith("| ") and not r.startswith("| Feature") and not r.startswith("|---")]
    assert len(rows_before) >= 27 and len(rows_after) >= 27
    for row in rows_after:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert any(s in cells[2] for s in ("✅", "🟡", "❌")), row
        assert cells[3] and cells[4] and cells[5], row
    for row in rows_before:
        assert any(s in row.split("|")[2] for s in ("✅", "🟡", "❌")), row


def test_no_feature_with_missing_external_requirements_is_marked_complete():
    after = TEXT.split("## Phase 5")[1].split("## Exploratory")[0]
    for name in ("Alternative data", "LLM research assistant", "Distributed experiment execution", "TimesFM", "Macro feature layer"):
        row = next(r for r in after.splitlines() if r.startswith(f"| {name}"))
        assert "✅" not in row.split("|")[3], name
