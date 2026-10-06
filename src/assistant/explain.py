"""``quant explain``: plain-language answers about a term, a technique, a strategy or a recorded experiment.

No language model is involved. The answers come from files in this repository: the glossary and the technique guides under ``docs/``, the plugin
registries (every model, detector and allocator carries a description), and the decision registry (every experiment's hypothesis, rule and result).
That makes the explanations checkable and identical on every machine, which is the point of a teaching tool for a field where confident wrong
explanations are common. ``explain("EXP-081")`` explains a result; ``explain("deflated sharpe")`` explains a concept; ``explain("stage 31")`` summarises a stage.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DECISION_MEANING = {
    "retain": "the declared rule was met, so the hypothesis was kept (this is not proof: it was one test among those declared)",
    "reject": "the declared rule was not met, so the hypothesis was not supported (this is not proof of no effect: see the power study)",
    "investigate": "the evidence was mixed and the stage says so; no decision rests on it",
    "record": "a descriptive result: it records what was found and carries no hypothesis test",
}


@dataclass
class Guide:
    slug: str
    title: str
    meta: dict
    body: str

    def section(self, name: str) -> str:
        m = re.search(rf"^## {re.escape(name)}\n+(.*?)(?=^## |\Z)", self.body, re.S | re.M)
        return m.group(1).strip() if m else ""


def _docs(config) -> Path:
    return config.root / "docs"


def load_guides(config) -> dict[str, Guide]:
    guides = {}
    for path in sorted((_docs(config) / "techniques").glob("*.md")):
        _, fm, body = path.read_text(encoding="utf-8").split("---", 2)
        meta = yaml.safe_load(fm)
        guides[meta["slug"]] = Guide(meta["slug"], meta["title"], meta, body)
    return guides


def load_glossary(config) -> dict[str, dict]:
    path = _docs(config) / "glossary.md"
    entries = {}
    if not path.exists():
        return entries
    for block in re.split(r"^### ", path.read_text(encoding="utf-8"), flags=re.M)[1:]:
        term, _, rest = block.partition("\n")
        also = re.search(r"^\*Also: (.*)\*$", rest, re.M)
        see = re.search(r"^See: (.*)$", rest, re.M)
        text = "\n".join(l for l in rest.splitlines() if l.strip() and not l.startswith("*Also:") and not l.startswith("See:")).strip()
        entries[term.strip()] = {"definition": text, "also": [a.strip() for a in also.group(1).split(",")] if also else [],
                                 "see": re.findall(r"\[([^\]]+)\]\(techniques/", see.group(1)) if see else []}
    return entries


def _registry(config) -> dict[str, dict]:
    path = config.root / "experiments" / "registry.jsonl"
    if not path.exists():
        return {}
    return {d["experiment_id"]: d for d in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("-", " ").replace("_", " ")).strip()


def _guide_summary(g: Guide, full: bool) -> str:
    if full:
        return f"{g.title}\n{'=' * len(g.title)}\n\n{g.body.split(chr(10), 2)[2].strip()}"
    lines = [g.title, "=" * len(g.title), "", g.section("In one sentence"), "", g.section("The idea"), "", "What we found here:", g.section("What we found"), "",
             f"Read more: docs/techniques/{g.slug}.md   (difficulty {g.meta['difficulty']}/3; chapter {g.meta['chapter']}; run it: see 'Try it' in the guide)"]
    return "\n".join(lines)


def explain_experiment(exp: dict, guides: dict[str, Guide]) -> str:
    meaning = DECISION_MEANING.get(exp["decision"], exp["decision"])
    results = {k: v for k, v in (exp.get("results") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    shown = "\n".join(f"  {k}: {v:.4g}" for k, v in list(results.items())[:10])
    stage_no = int(exp["stage"][5:7]) if exp["stage"][5:7].isdigit() else None
    related = [g for g in guides.values() if stage_no in (g.meta.get("stages") or [])]
    out = [f"{exp['experiment_id']}  [{exp['stage']}]  decision: {exp['decision'].upper()}", "", "Hypothesis:", f"  {exp['hypothesis']}", "",
           f"What the decision means: {meaning}.", "", "Key results:", shown or "  (none recorded)", "", "The stage's own notes:", f"  {exp.get('notes', '')}"]
    if related:
        out += ["", "Background to read: " + ", ".join(f"{g.slug}" for g in related[:4])]
    return "\n".join(out)


def explain(query: str, config, full: bool = False) -> str:
    from ..framework import ALLOCATORS, DETECTORS, MODELS, load_library

    q = query.strip()
    guides, glossary, registry = load_guides(config), load_glossary(config), _registry(config)
    if not q:
        return "Ask about a term (`quant explain sharpe ratio`), a technique (`quant explain risk parity`), a strategy (`quant explain dual_momentum`) or an experiment (`quant explain EXP-081`)."

    m = re.fullmatch(r"(?i)exp[- ]?(\d{1,3})", q)
    if m:
        key = f"EXP-{int(m.group(1)):03d}"
        return explain_experiment(registry[key], guides) if key in registry else f"No experiment {key} in the registry."

    m = re.fullmatch(r"(?i)stage\s*(\d{1,2})", q)
    if m:
        n = int(m.group(1))
        exps = [e for e in registry.values() if e["stage"][5:7] == f"{n:02d}"]
        related = [g for g in guides.values() if n in (g.meta.get("stages") or [])]
        if not exps and not related:
            return f"Nothing recorded for stage {n}."
        lines = [f"Stage {n}"] + [f"  {e['experiment_id']} [{e['decision']}] {e['hypothesis'][:140]}" for e in exps]
        lines += ["", "Guides: " + (", ".join(g.slug for g in related) or "none")]
        return "\n".join(lines)

    load_library()
    for registry_, kind in ((MODELS, "model"), (DETECTORS, "regime detector"), (ALLOCATORS, "allocator")):
        for entry in registry_.entries():
            if _norm(entry.name) == _norm(q):
                guide = next((g for g in guides.values() if entry.name in (g.meta.get("models") or [])), None)
                text = f"{entry.name}: a {kind} in the family '{entry.family}'.\n\n{entry.description}"
                if guide:
                    text += f"\n\nBackground: {guide.title} (docs/techniques/{guide.slug}.md)\n\n{guide.section('In one sentence')}"
                text += f"\n\nTry it: quant backtest --model {entry.name} --tearsheet" if kind == "model" else ""
                return text

    nq = _norm(q)
    for term, entry in glossary.items():
        if nq == _norm(term) or nq in {_norm(a) for a in entry["also"]}:
            text = f"{term}\n{'-' * len(term)}\n{entry['definition']}"
            if entry["also"]:
                text += f"\n(also called: {', '.join(entry['also'])})"
            if entry["see"]:
                text += "\n\nGuides: " + ", ".join(entry["see"])
            return text
    for slug, g in guides.items():
        if nq in (_norm(slug), _norm(g.title)):
            return _guide_summary(g, full)

    names = {**{_norm(t): ("term", t) for t in glossary}, **{_norm(a): ("term", t) for t, e in glossary.items() for a in e["also"]},
             **{_norm(g.title): ("guide", s) for s, g in guides.items()}, **{_norm(s): ("guide", s) for s in guides}}
    tokens = set(nq.split())
    subset = sorted((k for k in names if tokens and tokens <= set(k.split())), key=lambda k: (len(k.split()), k))
    if subset and len(subset[0].split()) <= len(tokens) + 1:
        kind, key = names[subset[0]]
        return explain(key, config, full)
    contains = [v for k, v in names.items() if nq and (nq in k or k in nq)]
    close = [names[k] for k in difflib.get_close_matches(nq, list(names), n=5, cutoff=0.6)]
    hits = list(dict.fromkeys(contains + close))
    if len(hits) == 1:
        kind, key = hits[0]
        return explain(key, config, full)
    if hits:
        return f"No exact entry for '{query}'. Closest: " + "; ".join(f"{k} ({v})" for k, v in hits[:6])
    return f"No entry for '{query}'. Try `quant list models`, the glossary (docs/glossary.md) or the guides in docs/techniques/."
