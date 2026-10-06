"""``python -m src.assistant "which hypotheses were retained?"``: ask the research database a recognised question."""

from __future__ import annotations

import sys

import pandas as pd

from ..utils.config import project_root
from .sqlguard import Refused, ResearchAssistant


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    try:
        answer = ResearchAssistant(project_root() / "data" / "processed" / "research.db").ask(" ".join(argv))
    except Refused as refusal:
        print(f"I will not guess. {refusal}")
        return 1
    print(f"-- {answer.backend}\n-- {answer.sql}  {answer.params if answer.params else ''}")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 70):
        print(answer.rows.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
