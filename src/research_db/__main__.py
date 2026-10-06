"""``python -m src.research_db "SELECT ..." [--db research|runs]``: run one guarded read-only query.

``research`` (default) is the database built from the research record (experiments, metrics, figures, books); ``runs`` is the experiment manager's own
database of your runs (tables ``runs``, ``run_metrics``, ``run_tables``)."""

from __future__ import annotations

import sys

import pandas as pd

from ..assistant.sqlguard import check_select, open_readonly
from ..utils.config import project_root


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    which = argv[argv.index("--db") + 1] if "--db" in argv else "research"
    if which == "runs":
        from ..utils.config import load_config

        node = load_config().get("framework.experiment_db", {}) or {}
        path = project_root() / str(node.get("path", "data/experiments.db"))
        sql = check_select(argv[0], allowed=("runs", "run_metrics", "run_tables"))
    elif which == "research":
        path = project_root() / "data" / "processed" / "research.db"
        sql = check_select(argv[0])
    else:
        raise SystemExit("--db must be 'research' or 'runs'")
    if not path.exists():
        raise SystemExit(f"{path} does not exist yet" + (" (run `quant backtest ...` first)" if which == "runs" else " (run stage 29)"))
    connection = open_readonly(path)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60):
        print(pd.read_sql_query(sql, connection).to_string(index=False))
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
