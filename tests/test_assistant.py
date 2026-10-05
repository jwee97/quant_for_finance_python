"""Document parsing, extraction backends and cache, and the read-only SQL assistant (Generation 4, Priority 16)."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.assistant.documents import ensure_fomc_corpus, is_statement, load_corpus, parse_statement, statement_links
from src.assistant.extraction import (AnthropicBackend, ExtractionCache, ExtractionError, LexiconBackend, SCHEMA, extract_corpus,
                                      schema_hash, validate)
from src.assistant.sqlguard import (ALLOWED_TABLES, Refused, ResearchAssistant, SQLBackend, TemplateBackend, UnsafeSQL, check_select,
                                    open_readonly)

HTML = """<html><head><title>Federal Reserve Board - FOMC statement</title></head><body><div id="article">
<h3>January 31, 2006 FOMC statement For immediate release Share</h3>
<p>The Federal Open Market Committee decided today to raise its target for the federal funds rate by 25 basis points to 4-1/2 percent.
Although recent economic data have been uneven, the expansion in economic activity appears solid. Core inflation has stayed relatively low in
recent months and longer-term inflation expectations remain contained. Possible increases in energy prices could add to inflation pressures.
The Committee judges that some further policy firming may be needed to keep the risks to the attainment of both sustainable economic growth and
price stability roughly in balance. Voting for the monetary policy action were: Ben Bernanke; Timothy Geithner.</p></div></body></html>"""
LEX = {"hawkish": ["inflation pressures", "firming", "solid", "tighten"], "dovish": ["accommodation", "weak", "downside risks"]}


def _backend():
    return LexiconBackend.from_config(LEX)


def test_statement_parsing_drops_furniture_and_the_voter_list():
    title, text = parse_statement(HTML)
    assert is_statement(title, text)
    assert text.startswith("The Federal Open Market Committee decided") and "Voting for" not in text and "Share" not in text
    assert not is_statement("Some other release", "Nothing here")


def test_links_are_found_on_both_url_styles_without_the_network():
    pages = {"https://www.federalreserve.gov/monetarypolicy/fomchistorical2006.htm": 'x href="/newsevents/press/monetary/20060131a.htm" y',
             "https://www.federalreserve.gov/monetarypolicy/fomchistorical2011.htm": 'href="/newsevents/pressreleases/monetary20110126a.htm"',
             "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm": 'href="/newsevents/pressreleases/monetary20240131a.htm"'}
    links = statement_links(2006, 2026, lambda u: pages.get(u, ""))
    assert list(links) == ["20060131", "20110126", "20240131"]


def test_corpus_is_cached_and_availability_is_the_day_after(tmp_path):
    calls = []

    def fake(url):
        calls.append(url)
        return HTML if "monetary2006" in url or "20060131" in url else 'href="/newsevents/press/monetary/20060131a.htm"'
    corpus = ensure_fomc_corpus(tmp_path, 2006, 2006, get=fake)
    assert len(corpus) == 1 and corpus.loc[0, "available_at"] == corpus.loc[0, "date"] + pd.Timedelta(days=1)
    n = len(calls)
    again = ensure_fomc_corpus(tmp_path, 2006, 2006, get=fake)
    assert len(calls) == n and again.equals(corpus)
    assert json.loads((tmp_path / "manifest.json").read_text())["20060131"]["sha256"]


def test_lexicon_features_and_causal_change():
    backend = _backend()
    _, text = parse_statement(HTML)
    first = backend.extract(text, [])
    assert first["action"] == 1 and first["change"] == 0.0 and first["tone"] > 0
    second = backend.extract("The Committee decided to maintain accommodation; economic activity is weak and downside risks remain.", [text])
    assert second["action"] == 0 and second["tone"] < 0 and 0.5 < second["change"] <= 1.0
    lowered = backend.extract("The Committee decided today to lower its target for the federal funds rate.", [text])
    assert lowered["action"] == -1
    same = backend.extract(text, [text])
    assert same["change"] < 1e-9


def test_validation_clips_coerces_and_rejects():
    assert validate({"tone": 3, "change": 7.0, "action": 4}) == {"tone": 3.0, "change": 1.0, "action": 1}
    with pytest.raises(ExtractionError):
        validate({"tone": 1, "change": 0.1})
    with pytest.raises(ExtractionError):
        validate({"tone": "x", "change": 0.1, "action": 0})
    with pytest.raises(ExtractionError):
        validate({"tone": float("nan"), "change": 0.1, "action": 0})


def test_cache_returns_identical_results_and_invalidates_on_any_change(tmp_path):
    corpus = pd.DataFrame({"date": pd.to_datetime(["2006-01-31", "2006-03-28"]), "available_at": pd.to_datetime(["2006-02-01", "2006-03-29"]),
                           "text": [parse_statement(HTML)[1], "The Committee decided to keep the target. Activity is weak."]})
    cache = ExtractionCache(tmp_path / "c.jsonl")
    calls = []

    class Counting(LexiconBackend):
        def extract(self, text, history):
            calls.append(1)
            return super().extract(text, history)

    backend = Counting.from_config(LEX)
    first = extract_corpus(corpus, backend, cache)
    assert len(calls) == 2
    again = extract_corpus(corpus, backend, ExtractionCache(tmp_path / "c.jsonl"))       # reloaded from disk
    assert len(calls) == 2 and first.equals(again)
    other = Counting.from_config({**LEX, "hawkish": LEX["hawkish"] + ["extra"]})
    extract_corpus(corpus, other, cache)
    assert len(calls) == 4                                                             # a different lexicon never reads the old answers
    changed = corpus.copy()
    changed.loc[1, "text"] += " More."
    extract_corpus(changed, backend, cache)
    assert len(calls) == 5 and schema_hash() == schema_hash(SCHEMA)


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply)])


def test_anthropic_backend_parses_validates_and_never_leaks_the_future_instruction():
    client = FakeClient('Here you go: {"tone": 4.5, "action": 1}')
    backend = AnthropicBackend("some-model", client)
    _, text = parse_statement(HTML)
    record = backend.extract(text, [])
    assert record["tone"] == 4.5 and record["action"] == 1 and record["change"] == 0.0
    call = client.calls[0]
    assert call["temperature"] == 0.0 and call["model"] == "some-model" and "afterwards" in call["system"]
    assert backend.identity().startswith("anthropic:some-model:") and backend.identity() != AnthropicBackend("other", client).identity()
    for bad in ("no json at all", '{"tone": "high", "action": 0}', '{"tone": 1}'):
        with pytest.raises(ExtractionError):
            AnthropicBackend("m", FakeClient(bad)).extract(text, [])


@pytest.mark.parametrize("sql", ["drop table books", "select * from books; drop table books", "select * from books -- hi", "select * from sqlite_master",
                                 "with a as (select 1) delete from books", "select * from books union select * from secret", "select * from books b, secret s",
                                 "update books set sharpe = 9", "attach database 'x' as y", "pragma table_info(books)", "", "select load_extension('x')",
                                 "select * from pragma_table_info('books')"])
def test_guard_rejects_unsafe_statements(sql):
    with pytest.raises(UnsafeSQL):
        check_select(sql)


def test_guard_accepts_reads_and_adds_a_limit():
    assert check_select("SELECT * FROM books;").endswith("LIMIT 500")
    assert check_select("select * from books limit 5") == "select * from books limit 5"
    assert "LIMIT" in check_select("with t as (select * from books) select * from t")
    assert check_select("select * from experiments e join metrics m on m.experiment_id = e.experiment_id")
    assert check_select("select 'drop table x' as note from books")                      # a keyword inside a string literal is only text


def _database(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        create table experiments(experiment_id text, stage text, hypothesis text, decision text);
        create table metrics(experiment_id text, key text, value real);
        create table result_files(path text); create table figures(number text);
        create table books(source text, stage text, name text, sharpe real);
        create table grid_results(id integer, family text, sharpe real, ann_turnover real);
        insert into experiments values ('EXP-001','stage01_data','h1','retain'),('EXP-002','stage21_bayesian','h2','reject'),('EXP-003','stage21_bayesian','h3','retain');
        insert into metrics values ('EXP-001','x',1.5);
        insert into books values ('a.csv','stage12','M1',0.85),('a.csv','stage12','M3',0.31);
        insert into grid_results values (1,'momentum',0.3,5.0),(2,'momentum',0.4,9.0),(3,'momentum',0.1,2.0),(4,'mean_reversion',0.9,1.0);
    """)
    connection.commit()
    connection.close()


def test_template_assistant_answers_known_questions_and_refuses_the_rest(tmp_path):
    _database(tmp_path / "r.db")
    assistant = ResearchAssistant(tmp_path / "r.db")
    assert list(assistant.ask("Which hypotheses were retained?").rows["experiment_id"]) == ["EXP-001", "EXP-003"]
    assert assistant.ask("how many rejected").rows.to_dict("records") == [{"stage": "stage21_bayesian", "rejected": 1}]
    rows = assistant.ask("momentum rules with turnover below 8").rows
    assert list(rows["id"]) == [1, 3] and rows["sharpe"].is_monotonic_decreasing
    assert list(assistant.ask("what did stage 21 find?").rows["experiment_id"]) == ["EXP-002", "EXP-003"]
    assert assistant.ask("describe EXP-001").rows.loc[0, "value"] == 1.5
    assert list(assistant.ask("the best 1 books by sharpe").rows["name"]) == ["M1"]
    with pytest.raises(Refused):
        assistant.ask("should I buy SPY tomorrow?")


def test_sql_model_output_is_untrusted_and_the_connection_cannot_write(tmp_path):
    _database(tmp_path / "r.db")
    hostile = ResearchAssistant(tmp_path / "r.db", SQLBackend("m", FakeClient("DROP TABLE books")))
    with pytest.raises(UnsafeSQL):
        hostile.ask("anything")
    good = ResearchAssistant(tmp_path / "r.db", SQLBackend("m", FakeClient("```select name from books where sharpe > 0.5```")))
    assert list(good.ask("which book has a high sharpe").rows["name"]) == ["M1"]
    connection = open_readonly(tmp_path / "r.db")
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("insert into books values ('x','y','z',1)")
    connection.close()
    assert set(ALLOWED_TABLES) >= {"experiments", "metrics", "books", "grid_results"}
