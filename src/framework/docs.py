"""Generated documentation: strategy cards, the chapter map and a findings digest.

Everything here is derived from files that already exist (the plugin registry, the technique guides' front matter, the strategy-library tables and
the decision registry), so it cannot drift from the code: ``quant docs build`` regenerates it and a test fails if the committed copy is stale.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd

from ..assistant.explain import load_guides

NOT_BUILT = """## What this library does not contain

- **Dispersion trading** is conceptual only: it needs single-stock and index option data this repository does not have.
- **Cross-exchange crypto spreads** are not built: only one venue (Deribit) was reachable from the build environment.
- **ETF flows, analyst and earnings revisions and CDS spreads** are not available as data here; the alternative data used are credit spreads, implied-volatility structure, CFTC positioning and jobless claims.
- **Value and quality** are price-based proxies, not accounting factors.
- **ETF NAV arbitrage** is a simulation, not a trade: no creation/redemption data exist here.
"""

FAMILY_CAVEATS = {
    "crypto": "One exchange (Deribit), four assets, about five years of data; the inverse perpetual is treated as linear.",
    "cross-sectional": "Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.",
    "stat-arb": "Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.",
    "fixed income": "Bond ETFs only; durations are estimated from rolling data, not taken from bond analytics.",
    "volatility": "Uses VIX-type indices as the implied-volatility input; no option-level data.",
    "macro": "Macro series enter with their publication lags; revisions are not modelled.",
    "machine learning": "Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.",
    "fundamental": "Reads company statements from a file you supply (data/user/fundamentals.csv); without it the model refuses to run, except the price-only momentum factors. Needs a wide cross-section of stocks, not 15 ETFs, and statements that are point-in-time.",
}


def _params(cls) -> str:
    init = cls.__dict__.get("__init__")
    if init is None:
        return "none"
    out = []
    for name, p in inspect.signature(init).parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.append(f"`{name}` = {p.default!r}" if p.default is not inspect.Parameter.empty else f"`{name}`")
    return ", ".join(out) or "none"


def library_models(registry) -> list:
    """The models shipped in ``src/strategies``; user-registered or test-registered models are not part of the documented library."""
    return [e for e in registry.entries() if getattr(e.factory, "__module__", "").startswith("src.strategies")]


def _book_flag(cls) -> str:
    return "--allocator sleeves " if getattr(cls, "book", None) == "sleeves" else ""


def _required_series(cls) -> str:
    """The macro series a strategy reads, 'none (prices only)' if it reads none; a strategy that runs on any few of them says how many."""
    series = ", ".join(getattr(cls, "requires", ()) or ())
    at_least = getattr(cls, "requires_at_least", None)
    return "none (prices only)" if not series else (f"{series} (any {at_least} of them)" if at_least else series)


def strategy_cards(config) -> dict[str, str]:
    from ..framework import MODELS, load_library

    load_library()
    guides = load_guides(config)
    tables = config.reports_dir("tables")
    lib = pd.read_csv(tables / "stage30_specs.csv", index_col=0) if (tables / "stage30_specs.csv").exists() else pd.DataFrame()
    crypto = pd.read_csv(tables / "stage32_performance.csv", index_col=0) if (tables / "stage32_performance.csv").exists() else pd.DataFrame()
    survey = pd.read_csv(tables / "library_survey.csv", index_col=0) if (tables / "library_survey.csv").exists() else pd.DataFrame()
    second = config.root / "docs" / "strategy_survey_2.md"
    second_survey = second.read_text(encoding="utf-8") if second.exists() else ""                      # strategies added after the first survey are listed in the second
    third = config.root / "docs" / "strategy_survey_3.md"
    third_survey = third.read_text(encoding="utf-8") if third.exists() else ""                         # and those added after that in the third
    cards = {}
    for entry in sorted(library_models(MODELS), key=lambda e: e.name):
        cls = entry.factory
        guide = next((g for g in guides.values() if entry.name in (g.meta.get("models") or [])), None)
        row = lib.loc[entry.name] if entry.name in lib.index else (crypto.loc[entry.name] if entry.name in crypto.index else None)
        doc = inspect.getdoc(cls) or ""
        lines = [f"# {entry.name}", "", f"*Family: {entry.family}*", "", "## What it bets on", "", entry.description, "", doc.split("\n\n")[0] if doc else "", "",
                 "## Inputs", "", f"- Macro or alternative series required: {_required_series(cls)}",
                 f"- Parameters: {_params(cls)}", "", "## Run it", "", "```bash", f"quant backtest --model {entry.name} {_book_flag(cls)}--tearsheet", "```", ""]
        if getattr(cls, "rebalance", None):
            lines[-1:-1] = [f"This rule declares a {cls.rebalance} rebalance; pass `execution: {{rebalance: monthly}}` in a spec to override it.", ""]
        if entry.name in survey.index and survey.loc[entry.name, "error"] != survey.loc[entry.name, "error"]:
            r = survey.loc[entry.name]
            survey_page = ("../strategy_survey_3.md" if f"strategies/{entry.name}.md" in third_survey else "../strategy_survey_2.md" if f"strategies/{entry.name}.md" in second_survey
                           else "../strategy_survey.md")
            lines += ["## In the strategy survey", "",
                      f"Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe {float(r['net_sharpe']):+.2f}, CAGR {float(r['cagr']):.1%}, volatility {float(r['volatility']):.1%}, "
                      f"max drawdown {float(r['max_drawdown']):.1%}, turnover {float(r['turnover']):.1f} times a year, deflated Sharpe probability {float(r['deflated_sharpe_probability']):.2f} "
                      f"counting every strategy in the survey as a trial. One run, not a test: see [the survey]({survey_page}) for how to read it.", ""]
        if row is not None:
            own = f" ({float(row['own_window_sharpe']):.2f} over its own, longer live window)" if "own_window_sharpe" in row.index and pd.notna(row["own_window_sharpe"]) else ""
            lines += ["## What happened in this repository", "",
                      f"Net of costs on the window the search used: Sharpe {float(row['sharpe']):.2f}{own}, CAGR {float(row['cagr']):.1%}, volatility {float(row['ann_vol']):.1%}, "
                      f"max drawdown {float(row['max_drawdown']):.1%}, turnover {float(row['ann_turnover']):.1f} times a year. "
                      "This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.", ""]
        if entry.family in FAMILY_CAVEATS:
            lines += ["## Caveats", "", FAMILY_CAVEATS[entry.family], ""]
        if guide:
            lines += ["## Learn more", "", f"[{guide.title}](../techniques/{guide.slug}.md)", ""]
        cards[f"{entry.name}.md"] = "\n".join(lines)
    index = ["# Strategy cards", "", "One card per model registered in the framework. Cards are generated by `quant docs build`; do not edit them by hand.", "",
             "| model | family | net Sharpe | what it bets on |", "|---|---|---|---|"]
    for entry in sorted(library_models(MODELS), key=lambda e: (e.family, e.name)):
        row = lib.loc[entry.name] if entry.name in lib.index else (crypto.loc[entry.name] if entry.name in crypto.index else None)
        sharpe = "" if row is None else format(float(row["sharpe"]), ".2f")
        if row is None and entry.name in survey.index and survey.loc[entry.name, "error"] != survey.loc[entry.name, "error"]:
            sharpe = format(float(survey.loc[entry.name, "net_sharpe"]), ".2f")
        index.append(f"| [{entry.name}]({entry.name}.md) | {entry.family} | {sharpe} | {entry.description.split(':')[0][:80]} |")
    index += ["", NOT_BUILT]
    cards["index.md"] = "\n".join(index)
    return cards


def chapter_map(config) -> str:
    guides = load_guides(config)
    figures = config.reports_dir("figures")

    def chapter_key(g):
        c = str(g.meta["chapter"]).replace("Ch. ", "")
        main = c.split(" ")[0].split(".")[0]
        return int(main) if main.isdigit() else 99

    lines = ["# Chapter map", "", "Where each topic of *Quantitative Finance with Case Studies in Python* is implemented in this repository, with the guide that explains it, the stage that runs it, the figures that show it and the tests that pin it.", "",
             "| Book chapter | Guide | Stage(s) | Code | Figures | Tests |", "|---|---|---|---|---|---|"]
    for g in sorted(guides.values(), key=lambda g: (chapter_key(g), str(g.meta["chapter"]), g.slug)):
        m = g.meta
        figs = ", ".join(f"{int(n)}" for n in m["figures"]) or "n/a"
        lines.append(f"| {m['chapter']} | [{g.title}](techniques/{g.slug}.md) | {', '.join(str(s) for s in m['stages'])} | {', '.join(f'`{f}`' for f in m['files'][:2])} | {figs} | {', '.join(f'`{t}`' for t in m['tests'])} |")
    return "\n".join(lines) + "\n"


def technique_index(config) -> str:
    guides = load_guides(config)
    lines = ["# Technique guides", "", "Each guide answers: what is it in one sentence, the idea, why it matters, how this repository uses it, what was found, what goes wrong, and how to run it. Difficulty 1 needs no prior knowledge; 3 assumes the guides it lists as prerequisites.", ""]
    for level, label in ((1, "Difficulty 1: start here"), (2, "Difficulty 2"), (3, "Difficulty 3")):
        lines += [f"## {label}", ""]
        for g in sorted((g for g in guides.values() if g.meta["difficulty"] == level), key=lambda g: g.title.lower()):
            lines.append(f"- [{g.title}]({g.slug}.md): {g.section('In one sentence')}")
        lines.append("")
    return "\n".join(lines)


def findings(config) -> str:
    path = config.root / "experiments" / "registry.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    counts = pd.Series([r["decision"] for r in rows]).value_counts()
    lines = ["# Findings digest", "", "Generated from `experiments/registry.jsonl`. A *retain* means the declared rule was met; a *reject* means it was not. Neither is proof.", "",
             "| decision | count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in counts.items()] + ["", "## Every decision-bearing hypothesis", "", "| id | stage | decision | hypothesis |", "|---|---|---|---|"]
    for r in rows:
        if r["decision"] in ("retain", "reject"):
            lines.append(f"| {r['experiment_id']} | {r['stage']} | {r['decision']} | {r['hypothesis'][:160].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def build_docs(config) -> list[Path]:
    root = config.root / "docs"
    written = []
    (root / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "generated").mkdir(parents=True, exist_ok=True)
    for name, text in strategy_cards(config).items():
        p = root / "strategies" / name
        p.write_text(text, encoding="utf-8")
        written.append(p)
    for rel, text in (("chapter_map.md", chapter_map(config)), ("generated/findings.md", findings(config)), ("techniques/index.md", technique_index(config))):
        p = root / rel
        p.write_text(text, encoding="utf-8")
        written.append(p)
    return written
