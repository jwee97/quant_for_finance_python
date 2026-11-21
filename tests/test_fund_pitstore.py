"""The point-in-time store: the SQLite database as Parquet queried with DuckDB. What is pinned: the availability date is the one the research engine computes; a restatement is a later vintage that becomes visible only
on its own date while the original stays; a figure with no known filing date is available after a conservative lag and flagged; provenance travels with the figure; and a manifest records the export."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.fundamentals import manual
from src.fundamentals.pit import PitSettings, availability
from src.fundamentals.store import FundamentalsStore

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
from src.fundamentals import pitstore  # noqa: E402

COMPANY = {"ticker": "PIT1", "name": "Pit One", "exchange": "US", "country": "US", "currency": "USD"}


def entry(revenue, filing, year=2022, assets=5000.0, **extra):
    return {"company": COMPANY, "units": {"money": "units", "shares": "units"}, "periods": [{"fiscal_period": "FY", "period_end": f"{year}-12-31", "filing_date": filing, "values": {"revenue": revenue, "total_assets": assets}}], **extra}


@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    manual.save(s, entry(1000.0, "2023-02-10"), "correct")
    sid = int(s.securities().iloc[0]["security_id"])
    manual.save(s, {"security_id": sid, "units": {"money": "units", "shares": "units"}, "periods": [{"fiscal_period": "FY", "period_end": "2022-12-31", "filing_date": "2023-02-10", "values": {"revenue": 950.0, "total_assets": 5000.0}}]},
                "restate", restated_on="2023-09-15")
    manual.save(s, {"security_id": sid, "units": {"money": "units", "shares": "units"}, "periods": [{"fiscal_period": "FY", "period_end": "2023-12-31", "values": {"revenue": 1100.0, "total_assets": 5200.0}}]}, "correct")   # no date
    return s


def test_the_table_has_a_row_per_vintage_and_the_availability_the_research_engine_computes(store):
    df = pitstore.frame(store)
    rev = df[df["metric_code"] == "revenue"].sort_values(["period_end", "vintage"])
    assert list(rev["vintage"]) == [0, 1, 0] and list(rev["value"]) == [1000.0, 950.0, 1100.0] and list(rev["is_restated"]) == [False, True, False]
    periods = store.periods()
    expected, estimated = availability(periods, PitSettings())
    got = rev.set_index(["period_end", "vintage"])["available_at"]
    by_id = dict(zip(periods["period_id"], zip(expected, estimated)))
    for _, r in periods.iterrows():
        assert got[(r["period_end"], r["vintage"])] == by_id[r["period_id"]][0], "one rule, in one place"
    assert got[(pd.Timestamp("2022-12-31"), 0)] == pd.Timestamp("2023-02-11") and got[(pd.Timestamp("2022-12-31"), 1)] == pd.Timestamp("2023-09-16"), "a restatement is public from the day it was published, plus the lag"
    assert bool(rev.iloc[2]["availability_estimated"]) and rev.iloc[2]["available_at"] == pd.Timestamp("2024-04-30"), "no filing date: 120 days after the year end plus the lag, flagged"


def test_as_of_gives_what_was_public_on_each_day_and_keeps_the_original_beside_the_restatement(store, tmp_path):
    manifest = pitstore.export(store, tmp_path / "pit")
    assert manifest["rows"] == 6 and manifest["securities"] == 1 and "US/2022" in manifest["partitions"] and manifest["settings"]["info_lag_days"] == 1 and (tmp_path / "pit" / "observations").exists()
    pit = pitstore.PitStore(tmp_path / "pit")
    rev = lambda day, **kw: pit.as_of(day, metrics=["revenue"], **kw).set_index("period_end")["value"].to_dict()  # noqa: E731
    assert rev("2023-02-10") == {}, "before the filing, nothing"
    assert rev("2023-03-01") == {pd.Timestamp("2022-12-31"): 1000.0}
    assert rev("2023-09-14") == {pd.Timestamp("2022-12-31"): 1000.0}, "the restatement is not known the day before it is published"
    assert rev("2023-09-16") == {pd.Timestamp("2022-12-31"): 950.0}, "after it, the restated figure"
    assert rev("2024-05-01") == {pd.Timestamp("2022-12-31"): 950.0, pd.Timestamp("2023-12-31"): 1100.0}
    assert rev("2024-05-01", include_estimated=False) == {pd.Timestamp("2022-12-31"): 950.0}, "a figure whose date is assumed can be left out"
    v = pit.vintages(int(store.securities().iloc[0]["security_id"]), "revenue")
    assert list(v["vintage"]) == [0, 1, 0] and list(v["value"]) == [1000.0, 950.0, 1100.0], "nothing was overwritten"
    assert pit.as_of("2024-05-01", countries=["JP"]).empty and len(pit.as_of("2024-05-01", countries=["US"])) == 4
    s = pit.summary()
    assert s.iloc[0]["country"] == "US" and s.iloc[0]["companies"] == 1 and s.iloc[0]["estimated"] == 2


def test_provenance_of_an_import_travels_with_the_figure_and_the_manifest_makes_a_result_reproducible(tmp_path):
    from src.fundamentals.importer.pipeline import ImportService
    from tests.test_fund_import import ixbrl
    s = FundamentalsStore(tmp_path / "f.db")
    manual.save(s, entry(900e6, "2022-02-01", year=2021, assets=4000e6), "correct")
    sid = int(s.securities().iloc[0]["security_id"])
    svc = ImportService(s)
    batch = svc.create_batch("b", sid)
    it = svc.add_file(batch, "0000000001-24-000005.htm", ixbrl())[0]
    svc.process(batch, hint={"filing_date": "2024-02-08"})
    svc.approve(it["item_id"], security={"security_id": sid})
    manifest = pitstore.export(s, tmp_path / "pit")
    again = pitstore.export(s, tmp_path / "pit")
    assert again["rows"] == manifest["rows"], "exporting again replaces the files, it does not add to them"
    got = pitstore.PitStore(tmp_path / "pit").as_of("2024-03-01", metrics=["revenue"])
    row = got[got["period_end"] == pd.Timestamp("2023-12-31")].iloc[0]
    assert row["value"] == 1200e6 and row["accession"] == "0000000001-24-000005" and row["taxonomy_tag"] == "us-gaap:Revenues" and row["extraction_method"] == "xbrl" and row["reported_value"] == 1200e6
    m = json.loads((tmp_path / "pit" / "manifest.json").read_text())
    assert m["schema"] == pitstore.SCHEMA and m["source_db"].endswith("f.db") and m["exported_at"].endswith("Z")


def test_an_empty_store_exports_and_a_missing_export_is_named(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    assert pitstore.export(s, tmp_path / "e")["rows"] == 0 and pitstore.PitStore(tmp_path / "e").as_of("2024-01-01").empty
    from src.fundamentals.errors import FundamentalsError
    with pytest.raises(FundamentalsError, match="no point-in-time export"):
        pitstore.PitStore(tmp_path / "nothing")
