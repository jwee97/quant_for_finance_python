"""The roadmap audit is true (every location imports), complete against the roadmap's own table, and the page is the catalogue written out."""

from __future__ import annotations

from pathlib import Path

from src.alphalab import catalog

# The roadmap's prioritised table, row by row: each must be covered by at least one catalogue item of that priority.
ROADMAP = {
    "Alpha forecasting and IC calibration": ("P0", "Alpha calibration"),
    "Multi-factor risk engine and attribution": ("P0", "Risk decomposition"),
    "Dynamic signal decay and trading rules": ("P0", "Alpha-decay paths"),
    "Robust covariance estimation and denoising": ("P0", "Ledoit-Wolf"),
    "Cost-aware constrained optimization": ("P0", "Cost-aware constrained"),
    "Black-Litterman and Bayesian views": ("P1", "Black-Litterman"),
    "Advanced downside and drawdown optimization": ("P1", "CDaR"),
    "Factor risk budgeting and hedging": ("P1", "Factor risk budgeting"),
    "HERC and Nested Clustered Optimization": ("P1", "Nested clustered"),
    "Selection, sizing and timing attribution": ("P1", "Selection, sizing and timing"),
    "Fundamental valuation and event-driven alpha": ("P1", "Event studies"),
    "Strategy capacity, crowding and leverage": ("P1", "Capacity"),
    "Graph-constrained portfolio construction": ("P2", "Graph-based"),
    "Robust and worst-case optimization": ("P2", "Robust optimisation"),
    "Copula-based synthetic market scenarios": ("P2", "Copula scenarios"),
    "Smart beta and benchmark efficiency research": ("P2", "Smart-beta"),
}


def test_every_location_in_the_catalogue_imports():
    assert catalog.check() == []


def test_every_roadmap_row_is_covered_and_nothing_is_listed_twice():
    keys = [(i.area, i.item) for i in catalog.ITEMS]
    assert len(keys) == len(set(keys)) and {i.status for i in catalog.ITEMS} <= catalog.STATUSES
    for row, (priority, word) in ROADMAP.items():
        assert any(i.priority == priority and word.lower() in i.item.lower() for i in catalog.ITEMS), (row, word)


def test_what_is_not_built_says_why_and_what_is_partial_says_what_is_missing():
    for i in catalog.ITEMS:
        if i.status in ("not built", "partial"):
            assert len(i.note) > 40, i.item
        if i.status == "not built":
            assert i.where == ()
        else:
            assert i.where, i.item


def test_the_page_is_the_catalogue_written_out():
    page = Path(catalog.__file__).resolve().parents[2] / "docs" / "roadmap_audit.md"
    assert page.read_text(encoding="utf-8") == catalog.markdown(), "run: python -m src.alphalab.catalog"
    text = catalog.markdown()
    assert "## Summary" in text and f"{len(catalog.ITEMS)} items" in text and "not built" in text
