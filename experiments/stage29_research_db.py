"""Stage 29 - The research database (Generation 4, Priority 14).

Loads the registry, every result table, every figure caption and the Stage 26 grid into SQLite, verifies the
acceptance checks declared in ``config/research_db.yaml`` (nothing lost or altered in the move; the roadmap's example
query agrees with a direct pandas filter; the connection cannot write), saves the example queries and the template
assistant's answers, and records the counts. It makes no research claim. This stage runs last because it ingests the
output of every other stage.

Figure 58.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.assistant.sqlguard import ResearchAssistant, check_select, open_readonly
from src.research_db.builder import build_database, verify_database
from src.utils.plotting import PALETTE, new_axes, save_figure
from experiments.context import build_context

STAGE = "stage29_research_db"
QUESTIONS = ["Which hypotheses were retained?", "How many were rejected?", "What did stage 25 find?", "Describe EXP-068",
             "the best 5 books by sharpe", "momentum rules with turnover below 8"]
GENERATIONS = {"Generation 1": range(1, 15), "Generation 2": range(15, 21), "Generation 3": range(21, 26), "Generation 4": range(26, 30)}


def figure_database(counts: dict, decisions: pd.DataFrame, per_stage: pd.Series, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.barh(range(len(counts)), list(counts.values()), color=PALETTE[0])
    ax.set_yticks(range(len(counts)), list(counts))
    ax.set_xscale("log")
    for i, v in enumerate(counts.values()):
        ax.text(v, i, f" {v:,}", va="center", fontsize=9)
    ax.set_xlabel("Rows (log scale)")
    ax.set_title("What the database holds")
    ax = axes[1]
    bottom = np.zeros(len(decisions))
    for name, colour in (("retain", "#009E73"), ("reject", "#D55E00"), ("record", "#999999")):
        if name in decisions:
            ax.bar(range(len(decisions)), decisions[name].to_numpy(), bottom=bottom, color=colour, label=name)
            bottom += decisions[name].to_numpy()
    ax.set_xticks(range(len(decisions)), decisions.index, rotation=15, ha="right")
    ax.set_ylabel("Registry entries")
    ax.set_title("Every decision ever declared, by generation")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.bar(range(len(per_stage)), per_stage.to_numpy(), color=PALETTE[2])
    ax.set_xticks(range(len(per_stage)), [s.replace("stage", "") for s in per_stage.index], rotation=90, fontsize=7)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Result tables")
    ax.set_title("Result tables per stage")
    fig.suptitle("Figure 58. The research database: what is in it and what was decided", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "What does the research database hold, how many hypotheses did each generation declare and with what outcome, and how many "
                           "result tables does each stage contribute?", 58)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 29: research database").parse_args(argv)
    context, logger = build_context(STAGE, generation=4)
    cfg = context.config
    node = cfg.get("research_db", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 29 | research database (Generation 4, Priority 14)")
    logger.info("=" * 72)
    root = cfg.root
    db_path = root / str(node.get("path", "data/processed/research.db"))
    counts = build_database(root, db_path)
    logger.info("built %s: %s", db_path.relative_to(root), counts)
    checks = verify_database(root, db_path)
    for name, ok in checks.items():
        logger.info("acceptance: %-62s %s", name, "PASS" if ok else "FAIL")
    assert all(checks.values()), f"acceptance checks failed: {[k for k, v in checks.items() if not v]}"
    context.save_table(pd.DataFrame({"check": list(checks), "passed": list(checks.values())}), "stage29_acceptance_checks.csv", index=False)
    context.save_table(pd.DataFrame({"table": list(counts), "rows": list(counts.values())}), "stage29_table_counts.csv", index=False)

    connection = open_readonly(db_path)
    examples = node.get("example_queries", {}) or {}
    rows = []
    for name, sql in examples.items():
        frame = pd.read_sql_query(check_select(sql, limit=1000), connection)
        context.save_table(frame.head(200), f"stage29_query_{name}.csv", index=False)
        rows.append({"query": name, "sql": sql, "rows_returned": len(frame)})
        logger.info("query %-9s -> %d rows\n%s", name, len(frame), frame.head(5).to_string(index=False)[:900])
    context.save_table(pd.DataFrame(rows), "stage29_example_queries.csv", index=False)
    assistant = ResearchAssistant(db_path)
    answers = []
    for q in QUESTIONS:
        a = assistant.ask(q)
        answers.append({"question": q, "sql": a.sql, "rows": len(a.rows), "backend": a.backend})
    context.save_table(pd.DataFrame(answers), "stage29_assistant_answers.csv", index=False)
    logger.info("template assistant answered %d of %d questions", len(answers), len(QUESTIONS))

    experiments = pd.read_sql_query("SELECT experiment_id, stage, decision FROM experiments", connection)
    number = experiments["stage"].str.extract(r"stage(\d+)")[0].astype(int)
    by_generation = pd.DataFrame({g: experiments[number.isin(list(r))]["decision"].value_counts() for g, r in GENERATIONS.items()}).fillna(0).T
    logger.info("decisions by generation:\n%s", by_generation.astype(int).to_string())
    per_stage = pd.read_sql_query("SELECT stage, COUNT(*) AS n FROM result_files WHERE stage IS NOT NULL GROUP BY stage ORDER BY stage", connection).set_index("stage")["n"]
    connection.close()
    figure_database(counts, by_generation, per_stage, context.figure("fig58_research_database.png"))

    context.registry.log(
        "Descriptive: the research record loads into a queryable database with nothing lost or altered, and the roadmap's example query returns what a direct filter returns.",
        stage=STAGE, parameters={"engine": "sqlite", "schema_version": 1, "acceptance_checks": list(checks)},
        results={**{f"rows_{k}": float(v) for k, v in counts.items()}, "acceptance_checks_passed": float(sum(checks.values())),
                 "acceptance_checks_total": float(len(checks))},
        decision="record", notes="Engineering acceptance, not a hypothesis. The database is derived from the files and rebuilt on every run; the counts are as of this "
                                 "stage's own start, so this stage's own registry entry is not in it.",
    )
    logger.info("STAGE 29 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
