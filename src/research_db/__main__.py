"""``python -m src.research_db "SELECT ..."``: run one guarded read-only query against the research database."""

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
    sql = check_select(argv[0])
    connection = open_readonly(project_root() / "data" / "processed" / "research.db")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60):
        print(pd.read_sql_query(sql, connection).to_string(index=False))
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
