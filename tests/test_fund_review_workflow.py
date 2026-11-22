"""The review workflow: the scanner discovers, a person confirms, only confirmed observations are stored and analysed.

End to end (PDF -> raw observations -> grouping -> corrections -> confirmation -> final approval -> database -> analytics) and the negative cases the workflow exists to prevent. The PDFs are synthetic
look-alikes (tests/fixtures/real_like.py); nothing here says anything about the original reports.
"""

from __future__ import annotations

import pandas as pd
import pytest

pymupdf = pytest.importorskip("pymupdf")

from src.fundamentals.errors import FundamentalsError  # noqa: E402
from src.fundamentals.importer import engine, observations  # noqa: E402
from src.fundamentals.importer.pipeline import ImportService  # noqa: E402
from src.fundamentals.pit import PeriodTables  # noqa: E402
from src.fundamentals.providers.base import CompanyProfile  # noqa: E402
from src.fundamentals.store import FundamentalsStore  # noqa: E402
from tests.fixtures import real_like as rl  # noqa: E402

HINT = {"apple": {"country": "US"}, "samsung": {"country": "KR"}}


@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    s.upsert_security(CompanyProfile("manual", "us:RNI", "RNI", "Round Numbers Inc", exchange="US", market="US", country="US", currency="USD"))
    return s


def imported(store, name="apple", filed="2020-10-30"):
    svc = ImportService(store)
    sid = int(store.securities().iloc[0]["security_id"])
    batch = svc.create_batch("t", security_id=sid)
    data, _ = rl.BUILDERS[name]()
    item = svc.add_file(batch, f"{name}.pdf", data)[0]["item_id"]
    svc.process(batch, hint={**HINT[name], "filing_date": filed})
    return svc, item, sid


def find(svc, item, label=None, metric=None, **kw):
    rows = svc.observations(item, limit=1000, **kw)["rows"]
    return [o for o in rows if (label is None or o["label"] == label) and (metric is None or o["metric_guess"] == metric)]


# ------------------------------------------------------------------------------------------------------------------------ layer 1: nothing meaningful is discarded
def test_unmapped_figures_are_kept_grouped_and_start_unconfirmed(store):
    svc, item, sid = imported(store)
    iphone = find(svc, item, "iPhone")
    assert iphone and iphone[0]["metric_guess"] is None and iphone[0]["value"] == 137781.0 and iphone[0]["status"] == "UNCONFIRMED" and iphone[0]["currency"] == "USD" and iphone[0]["unit"] == "millions"
    assert iphone[0]["page"] and iphone[0]["table_ref"] and iphone[0]["column_header"] and iphone[0]["bbox"], "every observation keeps its evidence"
    page = svc.observations(item, unmapped=True)
    assert page["total"] >= 2 and all(o["metric_guess"] is None for o in page["rows"])
    assert {"income_statement", "balance_sheet", "cash_flow"} <= set(page["by_group"]) and page["by_status"]["UNCONFIRMED"] == sum(page["by_group"].get(g, 0) for g in observations.GROUPS)
    assert svc.observations(item, q="iphone")["total"] >= 1 and svc.observations(item, group="balance_sheet")["total"] >= 4
    assert find(svc, item, "Total net sales", column_role="PRIOR_PERIOD") or find(svc, item, "Total net sales")


def test_change_columns_are_archived_with_their_role_never_as_a_reported_value(store):
    data, truth = rl.BUILDERS["toyota"]()
    draft, _ = engine.extract(data, "t.pdf", "pdf", {"country": "JP", "standard": "us-gaap"})
    roles = {o["column_role"] for o in draft["raw_observations"]}
    assert {"CURRENT_PERIOD", "PRIOR_PERIOD"} <= roles and roles & {"ABSOLUTE_CHANGE", "PERCENT_CHANGE"}
    assets = [o for o in draft["raw_observations"] if o["label"].lower().startswith("total assets") and o["column_role"] == "ABSOLUTE_CHANGE"]
    assert assets and all(not o["selected"] for o in assets)


def test_a_second_reading_never_drops_what_the_first_read():
    first = [{"label": "Revenue", "period_end": "2020-12-31", "value": 100.0, "currency": "USD"}, {"label": "Assets", "period_end": "2020-12-31", "value": 500.0, "currency": "USD"}]
    second = [{"label": "Revenue", "period_end": "2020-12-31", "value": 100.0, "currency": "USD"}, {"label": "Deferred revenue", "period_end": "2020-12-31", "value": 7.0, "currency": "USD"}]
    merged = observations.merge(first, second)
    assert [o["label"] for o in merged] == ["Revenue", "Assets", "Deferred revenue"] and "second reading" in merged[2]["method"]


# ------------------------------------------------------------------------------------------------------------------------ layer 2: confirmation belongs to the person
def test_high_confidence_does_not_confirm_and_nothing_unconfirmed_is_stored(store):
    svc, item, sid = imported(store)
    assert all(o["status"] == "UNCONFIRMED" for o in svc.observations(item, limit=1000)["rows"]), "not one observation is confirmed by the scanner, however confident"
    svc.review(item, {"confirm_periods": True})
    with pytest.raises(FundamentalsError, match="no observation has been confirmed"):
        svc.approve(item, security={"security_id": sid})
    assert store.observations([sid]).empty


def test_missing_confidence_does_not_imply_verification(store):
    from src.fundamentals.importer.pipeline import _unverified
    p = {"values": {"revenue": 1.0, "net_income": 2.0}, "sources": {"revenue": {"confidence": None}, "net_income": {"confidence": 0.99}}}
    assert set(_unverified(p)) == {"revenue", "net_income"}
    p["sources"]["revenue"]["edited"] = True
    assert "revenue" in _unverified(p), "an edit is not a confirmation"
    p["sources"]["revenue"]["confirmed"] = True
    assert set(_unverified(p)) == {"net_income"}


def test_an_edited_value_is_not_stored_until_it_is_confirmed_and_editing_withdraws_a_confirmation(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    rev = find(svc, item, metric="revenue", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "correct", "obs_id": rev["obs_id"], "value": 274000}])
    o = svc.observation(rev["obs_id"])
    assert o["status"] == "UNCONFIRMED" and o["user_value"] == 274000 and o["value"] == 274515 and o["history"][-1]["action"] == "correct"
    others = [x["obs_id"] for x in find(svc, item, metric="total_assets", column_role="CURRENT_PERIOD")]
    svc.decide(item, [{"action": "confirm", "ids": others}])
    svc.approve(item, security={"security_id": sid})
    obs = store.observations([sid])
    assert set(obs["metric_code"]) == {"total_assets"}, "the edited figure was never confirmed: it is not in research"


def test_a_corrected_and_confirmed_value_is_stored_as_the_person_typed_it_with_its_history(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    rev = find(svc, item, metric="revenue", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "correct", "obs_id": rev["obs_id"], "value": 274515.5, "confirm": True, "note": "page 3"}])
    o = svc.observation(rev["obs_id"])
    assert o["status"] == "CORRECTED_AND_CONFIRMED" and o["confirmed_at"] and o["history"][-1]["before"]["value"] == 274515 and o["history"][-1]["after"]["value"] == 274515.5
    svc.approve(item, security={"security_id": sid})
    obs = store.observations([sid])
    assert float(obs[obs["metric_code"] == "revenue"]["value"].iloc[0]) == pytest.approx(274515.5e6)
    with store.db.transaction() as c:
        row = c.execute("SELECT edited, verified, reported_value FROM observation_provenance WHERE metric_code = 'revenue'").fetchone()
    assert row[0] == 1 and row[1] == 1 and row[2] == 274515.5


def test_an_inferred_period_cannot_be_confirmed_until_a_person_accepts_or_corrects_it(store):
    svc, item, sid = imported(store, "samsung", "2021-03-10")                                            # bare years in the headings: the day and month are assumed
    guessed = [o for o in svc.observations(item, limit=1000)["rows"] if o["period_guessed"]]
    assert guessed
    with pytest.raises(FundamentalsError, match="inferred"):
        svc.decide(item, [{"action": "confirm", "obs_id": guessed[0]["obs_id"]}])
    svc.decide(item, [{"action": "confirm", "obs_id": guessed[0]["obs_id"], "accept_period": True}])
    assert svc.observation(guessed[0]["obs_id"])["status"] == "CONFIRMED"
    other = guessed[1]
    svc.decide(item, [{"action": "correct", "obs_id": other["obs_id"], "period_end": "2020-12-31", "confirm": True}])
    assert svc.observation(other["obs_id"])["status"] == "CORRECTED_AND_CONFIRMED"


def test_excluded_and_unresolved_observations_are_never_published(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    rev = find(svc, item, metric="revenue", column_role="CURRENT_PERIOD")[0]
    ni = find(svc, item, metric="net_income", column_role="CURRENT_PERIOD")[0]
    ta = find(svc, item, metric="total_assets", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "exclude", "obs_id": rev["obs_id"], "note": "footnote figure"}, {"action": "unresolve", "obs_id": ni["obs_id"]}, {"action": "confirm", "obs_id": ta["obs_id"]}])
    plan = svc.plan(item)
    assert [l["metric"] for l in plan["will_commit"]] == ["total_assets"] and plan["counts"]["excluded"] == 1 and plan["counts"]["unresolved"] == 1 and plan["counts"]["confirmed"] == 1
    svc.approve(item, security={"security_id": sid})
    assert set(store.observations([sid])["metric_code"]) == {"total_assets"}


# ------------------------------------------------------------------------------------------------------------------------ custom figures survive the whole workflow
def test_an_unmapped_observation_survives_review_and_is_stored_as_a_custom_metric_with_point_in_time_availability(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    iphone = find(svc, item, "iPhone", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "correct", "obs_id": iphone["obs_id"], "custom_label": "iPhone net sales", "confirm": True},
                      {"action": "add", "label": "Vehicle deliveries", "value": 1234, "period_end": "2020-09-26", "currency": "USD", "unit": "units", "confirm": True}])
    plan = svc.plan(item)
    assert {c["label"] for c in plan["will_commit_custom"]} == {"iPhone net sales", "Vehicle deliveries"}
    ta = find(svc, item, metric="total_assets", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "confirm", "obs_id": ta["obs_id"]}])
    out = svc.approve(item, security={"security_id": sid})
    assert out["custom_saved"] == 2
    got = store.custom_observations([sid])
    assert {r["label"] for r in got} == {"iPhone net sales", "Vehicle deliveries"} and all(r["metric_code"].startswith("custom:") and r["filing_date"] == "2020-10-30" for r in got)
    iph = next(r for r in got if r["label"] == "iPhone net sales")
    assert iph["original_label"] == "iPhone" and iph["value"] == 137781.0 and iph["unit"] == "millions" and iph["currency"] == "USD" and iph["page"]
    assert store.custom_observations([sid], as_of="2020-10-29") == [], "not known before the report was public"
    assert len(store.custom_observations([sid], as_of="2020-10-30")) == 2
    assert not any(c.startswith("custom:") for c in store.observations([sid])["metric_code"]), "custom figures never pollute the canonical table"


def test_unmapped_but_unconfirmed_figures_stay_in_the_archive_only(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    svc.decide(item, [{"action": "confirm_suggested"}])
    svc.approve(item, security={"security_id": sid})
    assert store.custom_observations([sid]) == []
    assert svc.observations(item, unmapped=True, status="UNCONFIRMED")["total"] >= 2


def test_currency_changes_do_not_relabel_or_merge_figures(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    rev = find(svc, item, metric="revenue", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "correct", "obs_id": rev["obs_id"], "currency": "EUR", "confirm": True}])
    ta = find(svc, item, metric="total_assets", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "confirm", "obs_id": ta["obs_id"]}])
    svc.approve(item, security={"security_id": sid})
    assert set(store.observations([sid])["metric_code"]) == {"total_assets"}, "a EUR figure is not merged into a USD report"
    custom = store.custom_observations([sid])
    assert [(r["metric_code"], r["currency"], r["value"]) for r in custom] == [("revenue", "EUR", 274515.0)], "stored under its own currency, with the number unchanged"


def test_units_are_per_observation_and_a_custom_multiplier_is_accepted(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    ta = find(svc, item, metric="total_assets", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "correct", "obs_id": ta["obs_id"], "unit": "1000", "confirm": True}])
    assert svc.observation(ta["obs_id"])["effective"]["unit"] == "1000"
    with pytest.raises(FundamentalsError, match="unit is"):
        svc.decide(item, [{"action": "correct", "obs_id": ta["obs_id"], "unit": "lots"}])
    with pytest.raises(FundamentalsError, match="three-letter"):
        svc.decide(item, [{"action": "correct", "obs_id": ta["obs_id"], "currency": "dollars"}])
    with pytest.raises(FundamentalsError, match="not a number"):
        svc.decide(item, [{"action": "correct", "obs_id": ta["obs_id"], "value": "abc"}])


def test_two_different_confirmed_figures_for_one_line_are_a_conflict_a_person_must_resolve(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    svc.decide(item, [{"action": "add", "label": "Total net sales (restated)", "value": 999, "period_end": "2020-09-26", "currency": "USD", "unit": "millions", "metric": "revenue", "confirm": True}])
    scanner = [o["obs_id"] for o in find(svc, item, metric="revenue", column_role="CURRENT_PERIOD") if o["origin"] == "scanner"]
    svc.decide(item, [{"action": "confirm", "ids": scanner}])
    with pytest.raises(FundamentalsError, match="another confirmed figure"):
        svc.approve(item, security={"security_id": sid})
    assert store.observations([sid]).empty
    svc.decide(item, [{"action": "exclude", "ids": scanner, "note": "the restated figure is the right one"}])
    svc.approve(item, security={"security_id": sid})
    assert float(store.observations([sid]).query("metric_code == 'revenue'")["value"].iloc[0]) == 999e6


# ------------------------------------------------------------------------------------------------------------------------ duplicates, restatements, reprocessing
def test_the_same_file_is_not_imported_twice_and_an_approved_item_cannot_be_approved_again(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    svc.decide(item, [{"action": "confirm_suggested"}])
    svc.approve(item, security={"security_id": sid})
    before = store.observations([sid]).shape
    with pytest.raises(FundamentalsError, match="already saved"):
        svc.approve(item, security={"security_id": sid})
    with pytest.raises(FundamentalsError, match="cannot be changed"):
        svc.decide(item, [{"action": "reset", "obs_id": 1}])
    batch = svc.create_batch("again", sid)
    data = svc._blob(svc.item(item)["content_hash"])
    again = svc.add_file(batch, "copy.pdf", data)[0]
    assert again["status"] == "duplicate" and store.observations([sid]).shape == before


def test_reprocessing_keeps_the_decisions_a_person_made(store):
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    ta = find(svc, item, metric="total_assets", column_role="CURRENT_PERIOD")[0]
    svc.decide(item, [{"action": "correct", "obs_id": ta["obs_id"], "value": 323000, "confirm": True}])
    svc.reprocess(item, {"country": "US"})
    o = svc.observation(ta["obs_id"])
    assert o["status"] == "CORRECTED_AND_CONFIRMED" and o["user_value"] == 323000 and not o["orphaned"]


def test_a_later_custom_value_is_a_new_vintage_and_the_earlier_one_is_still_what_was_known_before(store):
    sid = int(store.securities().iloc[0]["security_id"])
    svc = ImportService(store)
    for filed, value, label in (("2021-02-01", 100.0, "a.pdf"), ("2021-08-01", 110.0, "b.pdf")):
        data, _ = rl.BUILDERS["apple"]()
        batch = svc.create_batch(label, sid)
        item = svc.add_file(batch, label, data + (b"\n%" + label.encode()))[0]["item_id"]                 # a different file (different bytes), the same company and period
        svc.process(batch, hint={"country": "US", "filing_date": filed})
        svc.review(item, {"confirm_periods": True})
        svc.decide(item, [{"action": "add", "label": "Cloud revenue", "value": value, "period_end": "2020-09-26", "currency": "USD", "unit": "millions", "confirm": True},
                          {"action": "confirm", "ids": [find(svc, item, metric="total_assets", column_role="CURRENT_PERIOD")[0]["obs_id"]]}])
        try:
            svc.approve(item, security={"security_id": sid}, decision="restate")
        except FundamentalsError:
            svc.approve(item, security={"security_id": sid}, decision="correct")
    rows = store.custom_observations([sid])
    assert [(r["value"], r["vintage"]) for r in rows] == [(110.0, 1)]
    assert [r["value"] for r in store.custom_observations([sid], as_of="2021-03-01")] == [100.0], "the 110 was not public yet"
    with store.db.transaction() as c:
        assert c.execute("SELECT COUNT(*) FROM custom_observations").fetchone()[0] == 2, "the earlier vintage is kept"


# ------------------------------------------------------------------------------------------------------------------------ end to end into analytics
def test_end_to_end_pdf_to_confirmed_store_to_analytics_and_unconfirmed_figures_stay_out_of_factor_panels(store):
    svc, item, sid = imported(store)
    it = svc.item(item, full=True)
    assert it["draft"]["periods"] and svc.observations(item)["total"] > 20 and it["lifecycle"] in ("REVIEW_REQUIRED", "VALIDATED")
    svc.review(item, {"confirm_periods": True, "currency": "USD"})
    wanted = {"revenue", "net_income", "total_assets"}
    ids = [o["obs_id"] for o in svc.observations(item, limit=1000)["rows"] if o["selected"] and o["metric_guess"] in wanted]
    svc.decide(item, [{"action": "confirm", "ids": ids}])
    plan = svc.plan(item)
    assert plan["approvable"] and {l["metric"] for l in plan["will_commit"]} == wanted
    out = svc.approve(item, security={"security_id": sid})
    assert out["item"]["status"] == "saved" and out["saved"]
    obs = store.observations([sid])
    assert set(obs["metric_code"]) == wanted and (obs["quality_flag"] == "ok").all()
    tables = PeriodTables(store, [sid])
    assert tables.fy["revenue"].notna().all() and tables.fy["net_income"].notna().all()
    dates = pd.bdate_range("2021-03-01", "2021-06-30")
    panel = tables.panel("revenue", dates, "fy")
    assert panel.notna().any().any(), "a confirmed figure is visible to research after it was public"
    assert tables.panel("revenue", pd.bdate_range("2020-09-28", "2020-10-20"), "fy").isna().all().all(), "...and not before the filing date"
    unconfirmed = [c for c in ("operating_income", "gross_profit", "cost_of_revenue") if c in tables.fy]
    assert all(tables.fy[c].isna().all() for c in unconfirmed), "scanner-suggested but unconfirmed figures never reach a factor"
    with store.db.transaction() as c:
        statuses = dict(c.execute("SELECT status, COUNT(*) FROM raw_observations WHERE item_id = ? GROUP BY status", (item,)).fetchall())
    assert statuses.get("CONFIRMED", 0) == len(ids) and statuses["UNCONFIRMED"] > 20, "everything else stays in the archive"


def test_a_fallback_reading_never_removes_an_observation_the_first_reading_found(monkeypatch):
    data, _ = rl.BUILDERS["apple"]()
    first, _ = engine.extract(data, "a.pdf", "pdf", {"country": "US"})
    labels = {(o["label"], o["period_end"], o["value"]) for o in first["raw_observations"]}
    from src.fundamentals.importer.layout import pdfplumber_layout as _p
    if not _p.available():
        pytest.skip("pdfplumber is not installed")
    calls = iter([(0,), (1,), (1,), (1,)])
    monkeypatch.setattr(engine, "assess", lambda d: {"reliable": False, "pages": [2, 3, 4], "unreliable": [{"detail": "forced for the test"}]})
    monkeypatch.setattr(engine, "score", lambda d: next(calls))                                       # the second reading is judged better, so it is taken
    again, _ = engine.extract(data, "a.pdf", "pdf", {"country": "US"})
    kept = {(o["label"], o["period_end"], o["value"]) for o in again["raw_observations"]}
    assert labels <= kept, f"lost: {sorted(labels - kept)[:5]}"
    assert any(f["step"] == "second-table-reading" and f.get("accepted", True) for f in again["fallbacks"])


# ------------------------------------------------------------------------------------------------------------------------ sources: confirmed figures are not overwritten by a provider's update
def _two_sources(store):
    from src.fundamentals import manual
    from src.fundamentals.providers.base import CompanyProfile
    sid = int(store.securities().iloc[0]["security_id"])
    entry = {"security_id": sid, "units": {"money": "units", "shares": "units"}, "currency": "USD",
             "periods": [{"fiscal_period": "FY", "period_end": "2022-12-31", "fiscal_year": 2022, "filing_date": "2023-02-01", "currency": "USD", "values": {"revenue": 1000.0, "total_assets": 5000.0}}]}
    manual.save(store, entry, "correct")
    with store.db.transaction() as c:                                                                  # a provider's figure for the same period, a little different
        now = "2023-03-01T00:00:00Z"
        c.execute("INSERT INTO financial_periods (security_id, fiscal_year, fiscal_period, period_end, filing_date, reporting_currency, provider, vintage, pit_quality, created_at, updated_at) "
                  "VALUES (?, 2022, 'FY', '2022-12-31', '2023-02-20', 'USD', 'sec_edgar', 0, 'filed', ?, ?)", (sid, now, now))
        pid = c.execute("SELECT id FROM financial_periods WHERE provider = 'sec_edgar'").fetchone()[0]
        c.execute("INSERT INTO fundamental_observations (financial_period_id, metric_code, value, currency, provider, quality_flag, created_at) VALUES (?, 'revenue', 1100.0, 'USD', 'sec_edgar', 'ok', ?)", (pid, now))
    return sid


def test_a_providers_update_does_not_overwrite_a_confirmed_figure_and_the_difference_is_shown(store):
    from src.fundamentals import sources
    sid = _two_sources(store)
    rows = sources.compare(store, sid)
    rev = next(r for r in rows if r["metric"] == "revenue")
    assert rev["conflict"] and rev["providers"] == ["manual", "sec_edgar"] and rev["used"] == "manual" and rev["used_because"] == "provider priority"
    assert {x["provider"]: x["value"] for x in rev["sources"]} == {"manual": 1000.0, "sec_edgar": 1100.0} and abs(rev["max_relative_difference"] - 100 / 1100) < 1e-6
    assert [r["metric"] for r in sources.compare(store, sid, only_conflicts=True)] == ["revenue"]
    assert float(PeriodTables(store, [sid]).fy["revenue"].iloc[0]) == 1000.0, "the confirmed figure is what research sees"


def test_a_persons_choice_of_source_is_used_and_cannot_make_a_figure_visible_earlier_than_its_source(store):
    from src.fundamentals import sources
    sid = _two_sources(store)
    store.set_source_preference(sid, "revenue", "2022-12-31", "sec_edgar", "the filing is right")
    rev = next(r for r in sources.compare(store, sid) if r["metric"] == "revenue")
    assert rev["used"] == "sec_edgar" and rev["used_because"] == "your choice" and len(rev["sources"]) == 2, "nothing was deleted"
    t = PeriodTables(store, [sid])
    assert float(t.fy["revenue"].iloc[0]) == 1100.0
    assert str(t.fy["available_date"].iloc[0])[:10] >= "2023-02-20", "known only from the day the chosen source made it public"
    store.set_source_preference(sid, "revenue", "2022-12-31", None)
    assert float(PeriodTables(store, [sid]).fy["revenue"].iloc[0]) == 1000.0


def test_a_derived_figure_lists_the_inputs_it_read_with_their_source_and_is_not_calculated_when_one_is_missing(store):
    from src.fundamentals import lineage
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    ids = [o["obs_id"] for o in svc.observations(item, limit=1000)["rows"] if o["selected"] and o["metric_guess"] in ("net_income", "revenue", "total_assets")]
    svc.decide(item, [{"action": "confirm", "ids": ids}])
    svc.approve(item, security={"security_id": sid})
    tables = PeriodTables(store, [sid])
    npm = lineage.explain(store, tables, sid, "net_margin", "2020-09-26")
    assert npm["applicable"] and {i["input"] for i in npm["inputs"]} == {"net_income", "revenue"} and npm["value"] == pytest.approx(57411 / 274515)
    assert all(i["source"]["verified"] == 1 for i in npm["inputs"]) and npm["formula"] and npm["calculation_version"] and npm["currency"] == ["USD"]
    roe = lineage.explain(store, tables, sid, "roe", "2020-09-26")
    assert not roe["applicable"] and any(i["missing_current"] for i in roe["inputs"]) and roe["value"] is None and "missing inputs" in " ".join(roe["warnings"])
    assert lineage.explain(store, tables, sid, "net_margin", "1999-12-31")["applicable"] is False
    with pytest.raises(KeyError):
        lineage.explain(store, tables, sid, "nonexistent", "2020-09-26")


def test_alpha_inputs_show_what_a_factor_stands_on_and_that_unconfirmed_readings_are_not_in_it(store):
    from src.fundamentals import lineage
    svc, item, sid = imported(store)
    svc.review(item, {"confirm_periods": True})
    ids = [o["obs_id"] for o in svc.observations(item, limit=1000)["rows"] if o["selected"] and o["metric_guess"] == "revenue"]
    svc.decide(item, [{"action": "confirm", "ids": ids}])
    svc.approve(item, security={"security_id": sid})
    got = {r["field"]: r for r in lineage.alpha_inputs(store, ["revenue", "net_income", "no_such_field"])}
    assert got["revenue"]["securities"] == 1 and got["revenue"]["from_confirmed_imports"] >= 1 and got["revenue"]["status"] == "ok" and got["revenue"]["flagged_out"] == 0
    assert got["net_income"]["status"] == "no research-ready data" and got["net_income"]["unconfirmed_readings_waiting"] >= 1, "read by the scanner, never confirmed: waiting, not in the factor"
    assert got["no_such_field"]["known"] is False


def test_figures_in_another_currency_never_fill_a_gap_in_the_leading_providers_row(store):
    from src.fundamentals import manual
    sid = int(store.securities().iloc[0]["security_id"])
    manual.save(store, {"security_id": sid, "units": {"money": "units", "shares": "units"}, "currency": "USD", "periods": [
        {"fiscal_period": "FY", "period_end": "2022-12-31", "fiscal_year": 2022, "filing_date": "2023-02-01", "currency": "USD", "values": {"revenue": 1000.0}}]}, "correct")
    with store.db.transaction() as c:
        now = "2023-03-01T00:00:00Z"
        c.execute("INSERT INTO financial_periods (security_id, fiscal_year, fiscal_period, period_end, filing_date, reporting_currency, provider, vintage, pit_quality, created_at, updated_at) "
                  "VALUES (?, 2022, 'FY', '2022-12-31', '2023-01-15', 'KRW', 'sec_edgar', 0, 'filed', ?, ?)", (sid, now, now))
        pid = c.execute("SELECT id FROM financial_periods WHERE provider = 'sec_edgar'").fetchone()[0]
        c.execute("INSERT INTO fundamental_observations (financial_period_id, metric_code, value, currency, provider, quality_flag, created_at) VALUES (?, 'total_assets', 9e12, 'KRW', 'sec_edgar', 'ok', ?)", (pid, now))
    t = PeriodTables(store, [sid])
    assert float(t.fy["revenue"].iloc[0]) == 1000.0 and t.fy["total_assets"].isna().all(), "a KRW balance sheet is not mixed into a USD income statement"
