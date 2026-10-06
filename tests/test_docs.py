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


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def test_the_documented_strategy_example_runs_through_the_pipeline(tmp_path, monkeypatch):
    """The first code block of how_to_add_a_strategy.md is written to a module, registered, and run end to end."""
    import importlib
    import sys

    code = _python_blocks(ROOT / "docs" / "how_to_add_a_strategy.md")[0]
    (tmp_path / "my_strategies.py").write_text(code)
    monkeypatch.syspath_prepend(str(tmp_path))
    from src.framework import MODELS, Pipeline, load_default_bundle, load_library
    from src.utils.config import load_config

    config = load_config()
    load_library()
    sys.modules.pop("my_strategies", None)
    importlib.import_module("my_strategies")
    try:
        assert "high_proximity" in {e.name for e in MODELS.entries()}
        result = Pipeline({"name": "high_proximity", "models": [{"name": "high_proximity"}]}, config, load_default_bundle(config)).run()
    finally:
        MODELS._entries.pop("high_proximity", None)            # leave the global registry as the other tests expect it
        sys.modules.pop("my_strategies", None)
    assert all(c["ok"] for c in result.validation["causality"].values())
    assert result.metrics["n_days"] > 2000 and "deflated_sharpe_probability" in result.validation


def test_glossary_terms_are_unique_and_link_to_real_guides():
    from src.assistant.explain import load_glossary, load_guides
    from src.utils.config import load_config

    config = load_config()
    glossary, guides = load_glossary(config), load_guides(config)
    text = (ROOT / "docs" / "glossary.md").read_text(encoding="utf-8")
    assert len(glossary) == len(re.findall(r"^### ", text, re.M)) >= 100
    for term, entry in glossary.items():
        assert entry["definition"], term
        for slug in entry["see"]:
            assert slug in guides, (term, slug)


def test_explain_answers_terms_models_experiments_and_stages():
    from src.assistant.explain import explain
    from src.utils.config import load_config

    config = load_config()
    assert "best of N random trials" in explain("deflated sharpe", config)
    assert "dual momentum" in explain("dual_momentum", config).lower()
    assert "decision" in explain("stage 31", config).lower() or "Stage 31" in explain("stage 31", config)
    assert "Hypothesis" in explain("EXP-001", config) and "No experiment" in explain("EXP-999", config)
    assert "No entry" in explain("zzzzqqq", config)


def test_generated_docs_are_up_to_date():
    """`quant docs build` is deterministic; a stale committed copy means someone changed results or code without regenerating."""
    from src.framework.docs import chapter_map, findings, strategy_cards
    from src.utils.config import load_config

    config = load_config()
    if not (ROOT / "reports" / "tables" / "stage30_specs.csv").exists():
        pytest.skip("pipeline not run")
    for name, text in strategy_cards(config).items():
        assert (ROOT / "docs" / "strategies" / name).read_text(encoding="utf-8") == text, name
    assert (ROOT / "docs" / "chapter_map.md").read_text(encoding="utf-8") == chapter_map(config)
    assert (ROOT / "docs" / "generated" / "findings.md").read_text(encoding="utf-8") == findings(config)


def test_chapter_map_and_internal_links_resolve():
    for md in list((ROOT / "docs").rglob("*.md")):
        for target in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", md.read_text(encoding="utf-8")):
            if target.startswith("http"):
                continue
            assert (md.parent / target).resolve().exists(), (md.relative_to(ROOT), target)
