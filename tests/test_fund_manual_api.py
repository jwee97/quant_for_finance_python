"""Reports entered by hand, through the Fundamentals tab's endpoints (``/api/fund/manual/*``): the routes and the version, real data only, a table pasted from Excel read into a grid, a preview that
stores nothing, a save that creates the company and stores the periods with their dates, a correction and a restatement, what was entered read back and removed, every refusal as a message, answers
that are plain JSON, and a saved company that the rest of the tab (the company list, its page, the valuation, the research contexts) sees at once."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.webapp import api
from src.webapp.api import ApiError
from tests.test_fund_api import get, make_app, post
from tests.test_fund_manual import COMPANY, TABLE, entry, period

ROOT = Path(__file__).resolve().parents[1]
ROUTES_GET = ("/api/fund/manual/template", "/api/fund/manual/entries")
ROUTES_POST = ("/api/fund/manual/parse", "/api/fund/manual/preview", "/api/fund/manual/save", "/api/fund/manual/delete")


@pytest.fixture()
def app(tmp_path):
    return make_app(tmp_path)


def plain(value):
    json.dumps(value, allow_nan=False)                                                          # no NaN, no numpy: the server refuses to send anything else
    return value


def saved(app, **extra):
    return post(app, "/api/fund/manual/save", **{**entry(), **extra})


def test_the_routes_are_in_the_app_and_the_version_says_the_mode_is_there(app):
    for path in ROUTES_GET:
        assert path in app.extra_get
    for path in ROUTES_POST:
        assert path in app.extra_post
    assert api.API_VERSION >= 14 and app.catalog()["api_version"] >= 14
    assert not any("key" in p.lower() for p in ROUTES_POST)


def test_the_new_files_run_in_the_lanes_that_read_their_dependencies():
    if not (ROOT / ".github/workflows/ci.yml").exists():                                         # the Docker image is built without `.github`
        pytest.skip("the workflow files are not in this tree")
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())

    def text(job):
        return " ".join(str(step.get("run", "")) for step in ci["jobs"][job]["steps"])

    assert "tests/test_fund_manual.py" in text("compat"), "the library test runs with the other numerical tests on each Python version"
    assert "tests/test_fund_manual_api.py" in text("replit-python"), "the endpoint test runs in the dashboard lane"
    assert "tests/test_webapp_fundamentals_enter_browser.py" in text("browser")


def test_it_is_for_real_data_only_and_says_so(app):
    for path in ("/api/fund/manual/parse", "/api/fund/manual/preview", "/api/fund/manual/save", "/api/fund/manual/delete"):
        with pytest.raises(ApiError, match="real data"):
            post(app, path, db="demo", **entry())
    with pytest.raises(ApiError, match="real data"):
        get(app, "/api/fund/manual/entries", db="demo", id="1")
    with pytest.raises(ApiError, match="db must be one of"):
        post(app, "/api/fund/manual/preview", db="other", **entry())
    assert not app.root.joinpath("data", "user", "fundamentals_demo.db").exists()               # a refusal does not even make the demo database


def test_the_template_gives_the_lines_the_units_and_the_rules(app):
    t = plain(get(app, "/api/fund/manual/template"))
    assert [s["key"] for s in t["statements"]] == ["income", "balance", "cashflow", "shares"] and "millions" in t["units"] and t["max_periods"] == 60 and len(t["rules"]) >= 4


def test_a_pasted_table_is_read_into_periods_and_lines_and_a_bad_one_is_a_message(app):
    r = plain(post(app, "/api/fund/manual/parse", text=TABLE))
    assert [c["period_end"] for c in r["columns"]] == ["2021-12-31", "2022-12-31", "2023-12-31"] and r["values"]["revenue"] == [1000.0, 1200.0, 1500.0] and r["price"] == [20.0, 25.0, 30.0]
    for bad, why in [({}, "paste the table first"), ({"text": ""}, "paste the table first"), ({"text": "Item\tNotes\nRevenue\t5"}, "no header row of periods"), ({"text": 5}, "paste the table first")]:
        with pytest.raises(ApiError, match=why):
            post(app, "/api/fund/manual/parse", **bad)


def test_a_preview_checks_and_works_out_the_ratios_and_stores_nothing(app):
    r = plain(post(app, "/api/fund/manual/preview", **entry()))
    assert r["ok"] and r["ratios"]["computed"] > 60 and r["company"]["profile"]["ticker"] == "TEST1"
    assert app.fundamentals.bench("real").store.counts()["companies"] == 0
    bad = plain(post(app, "/api/fund/manual/preview", **entry(**{"periods": [period(2023, values={"total_assets": -1})]})))
    assert bad["ok"] is False and "ratios" not in bad and any(i["level"] == "error" for i in bad["issues"])
    with pytest.raises(ApiError, match="mode must be"):
        post(app, "/api/fund/manual/preview", mode="overwrite", **entry())


def test_a_save_creates_the_company_and_the_rest_of_the_tab_sees_it_at_once(app):
    wb = app.fundamentals.bench("real")
    wb._contexts["stale"] = object()                                                           # a research context built before a figure was entered or corrected is out of date
    r = plain(saved(app))
    assert r["created_company"] and r["added"] == 3 and r["ticker"] == "TEST1" and wb._contexts == {}
    sid = r["security_id"]
    listing = plain(get(app, "/api/fund/companies"))
    assert [(c["ticker"], c["source"], c["annual"]) for c in listing["companies"]] == [("TEST1", "manual", 3)]
    company = plain(get(app, "/api/fund/company", id=str(sid)))
    assert company["profile"]["source"] == "manual" and [a["fiscal_year"] for a in company["annual"]] == [2021, 2022, 2023] and company["annual"][2]["revenue"] == 1500e6
    assert company["annual"][2]["quality"] == "filed" and company["annual"][2]["provider"] == "manual" and company["annual"][2]["available"][:10] == "2024-02-11"
    assert plain(get(app, "/api/fund/status"))["databases"]["real"]["companies"] == 1
    plain(get(app, "/api/fund/valuation", id=str(sid)))


def test_a_second_save_corrects_and_a_ticker_that_exists_is_refused_with_the_way_out(app):
    sid = saved(app)["security_id"]
    with pytest.raises(ApiError, match="already in your database.*choose it from the list"):
        saved(app)
    again = plain(post(app, "/api/fund/manual/save", security_id=sid, units={"money": "millions", "shares": "millions"}, periods=[period(2021, values={"revenue": 1010})]))
    assert (again["corrected"], again["added"], again["created_company"]) == (1, 0, False)
    assert get(app, "/api/fund/manual/entries", id=str(sid))["periods"][0]["values"]["revenue"] == 1010e6


def test_a_restatement_is_a_new_version_and_needs_its_date(app):
    sid = saved(app)["security_id"]
    body = dict(security_id=sid, units={"money": "millions"}, periods=[period(2022, values={"revenue": 1250})], mode="restate")
    with pytest.raises(ApiError, match="needs the date it was published"):
        post(app, "/api/fund/manual/save", **body)
    r = plain(post(app, "/api/fund/manual/save", restated_on="2024-02-08", **body))
    assert r["restated"] == 1 and app.fundamentals.bench("real").store.counts()["restated_vintages"] == 1


def test_entries_are_read_back_and_a_period_can_be_removed(app):
    sid = saved(app)["security_id"]
    back = plain(get(app, "/api/fund/manual/entries", id=str(sid)))
    assert back["ticker"] == "TEST1" and back["suggested_unit"] == "millions" and len(back["periods"]) == 3
    gone = plain(post(app, "/api/fund/manual/delete", security_id=sid, fiscal_period="FY", period_end="2021-12-31"))
    assert gone["deleted"] == 1 and len(get(app, "/api/fund/manual/entries", id=str(sid))["periods"]) == 2
    for body, why in [({"security_id": sid, "fiscal_period": "FY", "period_end": "2021-12-31"}, "no period FY to 2021-12-31"), ({"security_id": sid}, "period type"),
                      ({"fiscal_period": "FY", "period_end": "2022-12-31"}, "say which company"), ({"security_id": 99, "fiscal_period": "FY", "period_end": "2022-12-31"}, "no company with id 99")]:
        with pytest.raises(ApiError, match=why):
            post(app, "/api/fund/manual/delete", **body)
    with pytest.raises(ApiError, match="say which company"):
        get(app, "/api/fund/manual/entries")
    with pytest.raises(ApiError, match="no company with id 99"):
        get(app, "/api/fund/manual/entries", id="99")


def test_every_refusal_is_a_message_and_nothing_is_half_saved(app):
    for body, why in [({"company": {"ticker": "bad ticker"}}, "ticker"), ({"company": None, "security_id": None}, "dictionary"), ({"periods": []}, "at least one period"),
                      ({"units": {"money": "lakhs"}}, "must be one of"), ({"periods": [{**period(2023), "period_end": "2030-12-31"}]}, "in the future"), ({"periods": [period(2023, values={"revenue": "abc"})]}, "not a number"),
                      ({"periods": [period(2023, filing_date="2020-01-01")]}, "before the period end")]:
        with pytest.raises(ApiError, match=why):
            saved(app, **body)
    store = app.fundamentals.bench("real").store
    assert store.counts()["companies"] == 0 and store.counts()["observations"] == 0
    assert COMPANY["ticker"] == "TEST1"
