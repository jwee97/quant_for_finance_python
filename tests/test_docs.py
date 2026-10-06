"""The documentation is part of the product: every guide has its sections, every path and figure it names exists, and every experiment it cites says what the guide claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GUIDES = sorted((ROOT / "docs" / "techniques").glob("*.md"))
SECTIONS = ["In one sentence", "The idea", "Why it matters", "How this repo uses it", "What we found", "Pitfalls", "Try it"]


def _front(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm), body


def _registry() -> dict[str, dict]:
    path = ROOT / "experiments" / "registry.jsonl"
    return {d["experiment_id"]: d for d in (json.loads(l) for l in path.read_text().splitlines() if l.strip())} if path.exists() else {}


def test_there_are_enough_guides_and_slugs_match_filenames():
    assert len(GUIDES) >= 40
    for g in GUIDES:
        meta, _ = _front(g)
        assert meta["slug"] == g.stem and meta["title"] and 1 <= meta["difficulty"] <= 3


@pytest.mark.parametrize("guide", GUIDES, ids=[g.stem for g in GUIDES])
def test_guide_structure_and_references(guide):
    meta, body = _front(guide)
    for heading in SECTIONS:
        assert f"## {heading}" in body, (guide.name, heading)
    for rel in meta["files"] + meta["tests"]:
        assert (ROOT / rel).exists(), (guide.name, rel)
    figures = ROOT / "reports" / "figures"
    if figures.exists():
        for n in meta["figures"]:
            assert list(figures.glob(f"fig{int(n):02d}_*.png")), (guide.name, n)
    known = {p.stem for p in (ROOT / "docs" / "techniques").glob("*.md")}
    for pre in meta["prerequisites"]:
        assert pre in known, (guide.name, pre)
    assert len(body.split()) > 250, guide.name


def test_every_model_in_a_guide_is_registered_and_every_registered_model_has_a_guide():
    from src.framework import MODELS, load_library

    load_library()
    registered = {e.name for e in MODELS.entries()}
    cited = {m for g in GUIDES for m in _front(g)[0]["models"]}
    assert cited <= registered, cited - registered
    assert registered <= cited, registered - cited


def test_cited_experiments_exist_and_belong_to_the_stages_of_the_guide():
    registry = _registry()
    if not registry:
        pytest.skip("registry not generated")
    for g in GUIDES:
        meta, body = _front(g)
        for exp in set(re.findall(r"EXP-\d{3}", body)):
            assert exp in registry, (g.name, exp)
            stage = int(registry[exp]["stage"][5:7])
            assert stage in meta["stages"], (g.name, exp, stage, meta["stages"])
