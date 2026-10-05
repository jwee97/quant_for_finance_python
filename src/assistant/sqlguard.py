"""A read-only SQL guard and a question-to-query assistant over the research database (Generation 4, Priority 16).

A language model asked to write SQL can write ``DROP TABLE``. The guard makes that harmless twice over: the text
must be one SELECT/WITH statement over whitelisted tables with no comments, and the connection is opened
read-only, so a statement that slips through still cannot write. The template backend answers a fixed list of
questions with parameterised queries and REFUSES the rest rather than guessing.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_TABLES = ("experiments", "metrics", "result_files", "figures", "books", "grid_results")
FORBIDDEN = {"insert", "update", "delete", "drop", "alter", "create", "attach", "detach", "pragma", "replace", "vacuum", "reindex", "analyze",
             "begin", "commit", "rollback", "savepoint", "release", "load_extension", "truncate", "grant", "revoke"}
ROW_LIMIT = 500


class UnsafeSQL(ValueError):
    """The statement is not a single read-only query over the allowed tables."""


class Refused(ValueError):
    """The assistant does not recognise the question and will not guess."""


def check_select(sql: str, allowed: tuple[str, ...] = ALLOWED_TABLES, limit: int = ROW_LIMIT) -> str:
    text = sql.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    if not text:
        raise UnsafeSQL("empty statement")
    if "--" in text or "/*" in text or "*/" in text:
        raise UnsafeSQL("comments are not allowed")
    if ";" in text:
        raise UnsafeSQL("one statement only")
    stripped = re.sub(r"'(?:[^']|'')*'", "''", text)                       # string literals cannot hide keywords from the checks below
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", stripped.lower())
    if not tokens or tokens[0] not in ("select", "with"):
        raise UnsafeSQL("only SELECT or WITH statements are allowed")
    bad = FORBIDDEN.intersection(tokens)
    if bad:
        raise UnsafeSQL(f"forbidden keyword(s): {sorted(bad)}")
    ctes = set(re.findall(r"\b([a-z_][a-z_0-9]*)\s+as\s*\(", stripped.lower()))
    for match in re.finditer(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z_0-9\.]*)", stripped, flags=re.IGNORECASE):
        table = match.group(1).lower()
        if table not in allowed and table not in ctes:
            raise UnsafeSQL(f"table '{table}' is not allowed")
    for match in re.finditer(r"\bfrom\s+((?:[A-Za-z_][A-Za-z_0-9\.]*(?:\s+(?:as\s+)?[A-Za-z_][A-Za-z_0-9]*)?\s*,\s*)+[A-Za-z_][A-Za-z_0-9\.]*)", stripped, flags=re.IGNORECASE):
        for part in match.group(1).split(","):                              # comma joins: every table in the list is checked
            table = part.split()[0].lower()
            if table not in allowed and table not in ctes:
                raise UnsafeSQL(f"table '{table}' is not allowed")
    if not re.search(r"\blimit\s+\d+\s*$", stripped, flags=re.IGNORECASE):
        text = f"{text} LIMIT {int(limit)}"
    return text


def open_readonly(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


@dataclass
class Answer:
    question: str
    sql: str
    params: tuple
    rows: pd.DataFrame
    backend: str


class TemplateBackend:
    """Recognised questions only. Each pattern yields ``(sql, params)``."""

    name = "template"

    def identity(self) -> str:
        return "template:v1"

    def to_query(self, question: str) -> tuple[str, tuple]:
        q = question.lower().strip().rstrip("?")
        number = re.search(r"(\d+(?:\.\d+)?)", q)
        stage = re.search(r"stage\s*(\d+)", q)
        exp = re.search(r"exp-?(\d+)", q)
        if exp:
            return ("SELECT e.experiment_id, e.stage, e.decision, e.hypothesis, m.key, m.value FROM experiments e LEFT JOIN metrics m "
                    "ON m.experiment_id = e.experiment_id WHERE e.experiment_id = ? ORDER BY m.key", (f"EXP-{int(exp.group(1)):03d}",))
        if "momentum" in q and "turnover" in q and number:
            return ("SELECT * FROM grid_results WHERE family = 'momentum' AND ann_turnover < ? ORDER BY sharpe DESC", (float(number.group(1)),))
        if "retained" in q:
            return ("SELECT experiment_id, stage, hypothesis FROM experiments WHERE decision = 'retain' ORDER BY experiment_id", ())
        if "rejected" in q:
            return ("SELECT stage, COUNT(*) AS rejected FROM experiments WHERE decision = 'reject' GROUP BY stage ORDER BY stage", ())
        if stage:
            return ("SELECT experiment_id, decision, hypothesis FROM experiments WHERE stage LIKE ? ORDER BY experiment_id", (f"stage{int(stage.group(1)):02d}%",))
        if "best" in q and ("book" in q or "sharpe" in q):
            n = int(float(number.group(1))) if number else 10
            return (f"SELECT source, name, sharpe FROM books ORDER BY sharpe DESC LIMIT {max(1, min(n, 100))}", ())
        raise Refused("I only answer: which hypotheses were retained / rejected; what stage N found; describe EXP-nnn; "
                      "the best N books by Sharpe; momentum rules with turnover below X.")


class SQLBackend:
    """Wraps a language model that writes SQL. Its output is untrusted: it goes through ``check_select`` like any other text."""

    name = "anthropic-sql"

    def __init__(self, model: str, client: Any = None):
        self.model, self.client = model, client

    def identity(self) -> str:
        return f"anthropic-sql:{self.model}"

    def to_query(self, question: str) -> tuple[str, tuple]:
        schema = "experiments, metrics, result_files, figures, books, grid_results (see config/research_db.yaml)"
        client = self.client
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        reply = client.messages.create(model=self.model, max_tokens=400, temperature=0.0,
                                       system=f"Write ONE read-only SQLite SELECT statement over: {schema}. Reply with SQL only.",
                                       messages=[{"role": "user", "content": question}])
        return "".join(getattr(b, "text", "") for b in reply.content).strip().strip("`"), ()


class ResearchAssistant:
    def __init__(self, db_path: str | Path, backend: Any | None = None):
        self.db_path, self.backend = Path(db_path), backend or TemplateBackend()

    def ask(self, question: str) -> Answer:
        sql, params = self.backend.to_query(question)
        safe = check_select(sql)
        connection = open_readonly(self.db_path)
        try:
            rows = pd.read_sql_query(safe, connection, params=params)
        finally:
            connection.close()
        return Answer(question, safe, tuple(params), rows, self.backend.identity())
