"""The Streamlit interface runs over the same services as the dashboard: it opens, shows the sources and what they permit, lists coverage from the database, and refuses what a source forbids."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.fundamentals import manual  # noqa: E402
from src.fundamentals.store import FundamentalsStore  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "fundamentals" / "streamlit_app.py")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "f.db"
    store = FundamentalsStore(db)
    manual.save(store, {"company": {"ticker": "STR1", "name": "Stream One", "exchange": "HK", "country": "HK", "currency": "HKD"}, "units": {"money": "units", "shares": "units"},
                        "periods": [{"fiscal_period": "FY", "period_end": "2021-12-31", "filing_date": "2022-03-01", "values": {"revenue": 100.0, "total_assets": 500.0}}]}, "correct")
    for name in ("EDINET_API_KEY", "OPENDART_API_KEY", "SEC_EDGAR_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FUNDAMENTALS_DB", str(db))
    monkeypatch.setenv("FUNDAMENTALS_PIT_DIR", str(tmp_path / "pit"))
    monkeypatch.setenv("STREAMLIT_APPTEST", "1")
    return AppTest.from_file(APP, default_timeout=60)


def test_the_app_opens_with_the_four_tabs_the_sources_and_the_coverage_from_the_database(app):
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Retrieve", "Upload and review", "Coverage", "Point in time"]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Years held"] == "1" and "Conflicts" in metrics
    sidebar = " ".join(str(x.value) for x in app.sidebar.markdown)
    assert "✗ EDINET" in sidebar and "✓ CVM" in sidebar


def test_a_source_that_asks_for_a_key_says_so_and_a_links_only_source_lists_nothing(app):
    app.run()
    app.selectbox(key="conn").set_value("edinet").run()
    assert any("EDINET_API_KEY" in w.value for w in app.warning)
    app.text_input(key="ident").set_value("7203")
    app.button(key="list").click().run()
    assert any("asks for a key" in e.value for e in app.error), "no request is made without the key"
    app.selectbox(key="conn").set_value("hkex").run()
    app.button(key="list").click().run()
    assert any("no interface for programs" in i.value for i in app.info)
