"""Finzdata as a source of the Futures tab: the key, the finder, every contract of a root, the download, and the stored file that keeps what a thirty-day window would lose.

Nothing here touches the network: Finzdata is the fake transport of ``tests/fake_finzdata.py`` behind the app's real client, ledger and store, on a date the test chooses. What cannot be tested offline is
that the real service answers in the shape the fake uses (the service's OpenAPI document leaves the rows undefined); ``experiments/live_feed_smoke.py`` (the ``finzdata`` check) is how that is found out.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assets.free_data.http import Http
from src.webapp.api import ApiError
from src.webapp.finzdata import BUDGET_VARIABLE, FinzdataClient
from tests import free_data_helpers as fh
from tests.fake_finzdata import KEY, FakeFinzdata
from tests.test_webapp_assets_ml import _app
from tests.test_webapp_data import _safe, run
from tests.test_webapp_labs import _rich

NOW = pd.Timestamp("2026-10-29")


def attach(app, tmp_path, fake=None, key=KEY, today=NOW):
    fake = fake or FakeFinzdata(today=str(today.date()))
    app.data.finz.client = FinzdataClient(key=key, root=tmp_path, base_url="https://api.finzdata.com", transport=fake, sleep=lambda s: None, pause=0.0)
    app.data.finz.today = lambda: today
    return fake


@pytest.fixture
def fzapp(tmp_path, monkeypatch):
    monkeypatch.delenv(BUDGET_VARIABLE, raising=False)
    monkeypatch.delenv("FINZDATA_API_KEY", raising=False)
    app = _app(tmp_path, _rich, root=tmp_path)
    app.data.http = Http(fh.FakeFeeds({}), sleep=lambda s: None, pause=0.0, retries=0)
    return app, attach(app, tmp_path), tmp_path


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


# ------------------------------------------------------------------------------------------------------------------ the key and the status
def test_without_a_key_the_status_says_where_to_put_it_the_finder_still_works_and_a_download_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("FINZDATA_API_KEY", raising=False)
    app = _app(tmp_path, _rich, root=tmp_path)
    fake = attach(app, tmp_path, key="")
    st = get(app, "/api/data/status")
    z = st["finzdata"]
    assert z["configured"] is False and z["setup"] == "key" and "FINZDATA_API_KEY" in z["problem"] and z["key_variable"] == "FINZDATA_API_KEY" and z["stored"] == []
    assert st["kinds"]["futures"]["finzdata_status"] == z
    assert next(s for s in st["kinds"]["futures"]["sources"] if s["id"] == "finzdata")["needs_key"]
    assert [s["id"] for s in st["kinds"]["futures"]["sources"]] == ["yahoo", "finzdata", "cboe_cfe", "deribit", "itick"]
    assert get(app, "/api/data/search", kind="futures", source="finzdata", q="crude")["results"][0]["root"] == "CL"          # the list ships with the app
    with pytest.raises(ApiError, match="not set up"):
        app.extra_post["/api/data/fetch"]({"kind": "futures", "source": "finzdata", "items": ["ES"]})
    with pytest.raises(ApiError, match="not set up"):
        app.extra_post["/api/data/lookup"]({"kind": "futures", "source": "finzdata"})
    with pytest.raises(ApiError, match="not set up"):
        get(app, "/api/data/contracts", root="ES")
    assert fake.urls == []                                                                  # nothing was sent anywhere


def test_with_a_key_the_status_has_the_days_calls_and_the_market_list_and_never_the_key(fzapp):
    app, fake, _ = fzapp
    st = get(app, "/api/data/status")
    z = st["finzdata"]
    assert KEY not in json.dumps(st) and z["configured"] and z["calls_today"] == 0 and z["calls_left"] == z["budget"] == 900 and z["daily_allowance"] == 1000 and z["window_days"] == 30
    assert z["markets"] == {"bundled": 139, "fetched_at": None, "from_service": 0} and "forbid publishing" in z["terms"]
    fut = st["kinds"]["futures"]
    assert fut["finzdata"]["markets"] == 139 and fut["finzdata"]["groups"][0] == "Equity index" and fut["finzdata"]["max_calls_per_root"] == 60
    assert fake.urls == []                                                                  # the status costs no call


# ------------------------------------------------------------------------------------------------------------------ the finder and the contracts
def test_the_finder_searches_the_bundled_list_with_no_call_and_refuses_a_group_it_does_not_have(fzapp):
    app, fake, _ = fzapp
    found = get(app, "/api/data/search", kind="futures", source="finzdata", q="nasdaq")
    assert [r["root"] for r in found["results"]] == ["NQ", "MNQ"] and found["source"] == "finzdata" and found["results"][0]["exchange"] == "CME" and found["results"][0]["terms_known"]
    energy = get(app, "/api/data/search", kind="futures", source="finzdata", group="Energy", limit=5)
    assert energy["shown"] == 5 and energy["total"] == 12 and {r["group"] for r in energy["results"]} == {"Energy"} and "Equity index" in energy["groups"]
    assert get(app, "/api/data/search", kind="futures", source="finzdata")["total"] == 43                         # the featured markets when nothing is typed
    with pytest.raises(ApiError, match="group must be one of"):
        get(app, "/api/data/search", kind="futures", source="finzdata", group="Weather")
    assert fake.urls == []


def test_asking_for_the_services_own_list_is_one_call_and_the_finder_then_shows_its_counts(fzapp):
    app, fake, tmp_path = fzapp
    out = _safe(app.extra_post["/api/data/lookup"]({"kind": "futures", "source": "finzdata", "q": "euro"}))
    assert out["markets"] == 4 and out["new"] == [] and out["missing_count"] == 135 and len(out["missing"]) == 20 and out["with_contract_counts"] == 4 and out["source"] == "finzdata"
    assert [p for p, _ in fake.requests] == ["/v1/futures/roots"]
    assert (tmp_path / "data" / "user" / "finzdata_roots.json").exists()
    es = get(app, "/api/data/search", kind="futures", source="finzdata", q="es")["results"][0]
    assert es["root"] == "ES" and es["contracts"] == 24 and es["last_date"] == "2026-10-29" and len(fake.requests) == 1          # the search reads the kept list: no second call
    assert get(app, "/api/data/status")["finzdata"]["markets"]["from_service"] == 4


def test_every_contract_of_a_root_is_listed_with_its_expiry_and_status_in_one_call_and_then_from_the_file(fzapp):
    app, fake, _ = fzapp
    first = get(app, "/api/data/contracts", root="es")
    assert (first["root"], first["listed"], first["called"], first["source"]) == ("ES", 24, True, "finzdata")
    assert first["live"] + first["expired"] == 24 and first["live"] == 9 and first["stored"] == 0 and first["basis"] == {"service": 24}
    rows = first["contracts"]
    assert rows[0]["contract"] == "H23" and rows[-1]["contract"] == "Z28" and [r["expiry"] for r in rows] == sorted(r["expiry"] for r in rows)
    z26 = next(r for r in rows if r["contract"] == "Z26")
    assert (z26["code"], z26["expiry"], z26["status"]) == ("ESZ26", "2026-12-18", "live") and next(r for r in rows if r["contract"] == "U26")["status"] == "expired"
    again = get(app, "/api/data/contracts", root="ES")
    assert again["called"] is False and again["contracts"] == rows and [p for p, _ in fake.requests] == ["/v1/futures/contracts"]
    assert get(app, "/api/data/contracts", root="ES", refresh="1")["called"] is True and len(fake.requests) == 2
    for bad in ("", "not a root", "ES;DROP"):
        with pytest.raises(ApiError, match="not a Finzdata futures symbol"):
            get(app, "/api/data/contracts", root=bad)


# ------------------------------------------------------------------------------------------------------------------ a download
def test_a_download_saves_every_contract_with_open_interest_and_the_futures_tab_reads_it(fzapp):
    app, fake, tmp_path = fzapp
    out = _safe(run(app, kind="futures", source="finzdata", items=["es"]))
    r = out["results"][0]
    assert out["ok"] == 1 and out["failed"] == 0 and r["ok"] and r["name"] == "ES" and "8 contracts" in r["message"] and "(Finzdata, 2 calls)" in r["message"] and r["source"] == "Finzdata"
    record = next(x for x in out["downloads"] if x["name"] == "ES")
    assert record["item"] == "curve" and record["source_id"] == "finzdata" and record["has_open_interest"] and record["refresh"] == {"source": "finzdata", "items": ["ES"]} and record["multiplier"] == 50.0
    assert (tmp_path / "data" / "user" / "downloads" / "futures" / "ES.csv").exists()
    curve = get(app, "/api/futures/curve", root="ES")
    assert curve["curve"]["rows"][0]["contract"] == "Z26" and curve["curve"]["rows"][0]["open_interest"] is not None and curve["curve"]["date"] == "2026-10-29"
    assert curve["spec"]["multiplier"] == 50.0 and curve["parts"]["open_interest"] == "downloaded" if "parts" in curve else True
    assert KEY not in json.dumps(out) and KEY not in json.dumps(curve) and set(fake.keys) == {KEY}


def test_a_root_under_another_symbol_takes_the_platforms_terms_and_one_it_has_none_for_gets_placeholders_and_says_so(fzapp):
    app, _, _ = fzapp
    out = _safe(run(app, kind="futures", source="finzdata", items=["E6", "FDAX"]))
    euro, dax = (next(x for x in out["downloads"] if x["name"] == n) for n in ("E6", "FDAX"))
    assert euro["multiplier"] == 125000.0 and euro["tick_size"] == 0.00005 and euro["root_name"] == "Euro FX"          # E6 is the platform's 6E: its terms apply
    assert "not in the platform's table" not in json.dumps(euro["notes"])
    assert (dax["multiplier"], dax["tick_size"], dax["asset_class"], dax["exchange"], dax["root_name"]) == (1.0, 0.01, "equity_index", "Eurex", "DAX")
    assert any("contract size and tick of FDAX are not in the platform's table" in n for n in dax["notes"])
    curve = get(app, "/api/futures/curve", root="FDAX")
    assert curve["spec"]["multiplier"] == 1.0 and curve["spec"]["initial_margin"] > 0 and any("ILLUSTRATIVE" in n for n in curve["notes"])


def test_the_stored_file_grows_with_each_download_because_a_window_of_thirty_days_slides(fzapp):
    app, fake, tmp_path = fzapp
    run(app, kind="futures", source="finzdata", items=["ES"])
    first = app.data.finz.coverage("ES", NOW, 30)
    assert (first["stored_from"], first["stored_to"], first["gap_count"], first["left"], first["urgent"]) == ("2026-09-29", "2026-10-29", 0, 30, False)
    later = pd.Timestamp("2026-11-12")                                                      # two weeks on: the window now starts on 13 October
    fake.today = later
    app.data.finz.today = lambda: later
    out = run(app, kind="futures", source="finzdata", items=["ES"])
    cov = app.data.finz.coverage("ES", later, 30)
    assert (cov["stored_from"], cov["stored_to"], cov["gap_count"]) == ("2026-09-29", "2026-11-12", 0) and cov["days"] > first["days"] == 23
    daily = [q for p, q in fake.requests if p == "/v1/futures/daily"]
    assert "start" not in daily[0] and daily[1]["start"] == "2026-10-22"                                      # the second asked from a week before the newest stored bar
    record = next(x for x in out["downloads"] if x["name"] == "ES")
    assert record["first_date"] == "2026-09-29" and record["as_of"] == "2026-11-12" and any("Merged with the earlier download" in n for n in record["notes"])
    reg = app.data.finz.registry()["ES"]
    assert reg["updates"] == 2 and reg["window_days"] == 30 and reg["calls"] == 4
    stored = get(app, "/api/data/status")["finzdata"]["stored"]
    assert [s["root"] for s in stored] == ["ES"] and stored[0]["name"] == "E-mini S&P 500" and stored[0]["contracts"] >= 8


def test_days_that_slid_out_of_the_window_between_two_downloads_are_reported_as_a_gap(fzapp):
    app, fake, _ = fzapp
    run(app, kind="futures", source="finzdata", items=["ES"])
    later = pd.Timestamp("2026-12-21")                                                      # seven weeks on: 29 October to 21 November was never seen
    fake.today = later
    app.data.finz.today = lambda: later
    out = run(app, kind="futures", source="finzdata", items=["ES"])
    r = out["results"][0]
    record = next(x for x in out["downloads"] if x["name"] == "ES")
    assert "gap" in r["message"] and record["gap_count"] == 1 and record["gaps"][0]["from"] == "2026-10-29" and record["gaps"][0]["to"] == "2026-11-23"
    assert any("cannot be fetched again" in n and "download at least every 20 days" in n for n in record["notes"])
    cov = app.data.finz.coverage("ES", later, 30)
    assert cov["gap_count"] == 1 and cov["stored_from"] == "2026-09-29" and cov["stored_to"] == "2026-12-21"


def test_how_long_the_stored_history_has_before_a_window_passes_it_is_in_the_status(fzapp):
    app, _, _ = fzapp
    run(app, kind="futures", source="finzdata", items=["ES"])
    app.data.finz.today = lambda: pd.Timestamp("2026-11-24")                                # 26 days after the newest bar
    s = get(app, "/api/data/status")["finzdata"]["stored"][0]
    assert (s["days"], s["age"], s["left"], s["urgent"], s["window_days"]) == (23, 26, 4, True, 30)


def test_a_refusal_in_the_middle_of_a_run_keeps_what_was_stored_and_stops_asking(fzapp):
    app, fake, _ = fzapp
    fake.quota = 3                                                                          # ES takes two calls, CL's list the third, CL's bars are over the day's limit
    out = _safe(run(app, kind="futures", source="finzdata", items=["ES", "CL", "E6"]))
    by = {r["item"]: r for r in out["results"]}
    assert out["ok"] == 1 and out["failed"] == 2 and by["ES"]["ok"] and not by["CL"]["ok"] and not by["E6"]["ok"]
    assert "limit of calls for the day" in by["CL"]["message"] and by["E6"]["message"] == by["CL"]["message"]       # the rest were not even tried
    assert len(fake.requests) == 4 and [x["name"] for x in out["downloads"]] == ["ES"]
    assert "fatal" not in by["CL"]


def test_a_plan_without_a_market_fails_that_market_alone(fzapp):
    app, fake, _ = fzapp
    fake.forbidden = {"CL"}
    out = _safe(run(app, kind="futures", source="finzdata", items=["CL", "ES"]))
    by = {r["item"]: r for r in out["results"]}
    assert not by["CL"]["ok"] and "does not include this data" in by["CL"]["message"] and by["ES"]["ok"]


def test_the_downloads_calls_are_checked_against_the_days_allowance_before_it_starts(fzapp, monkeypatch):
    app, fake, tmp_path = fzapp
    monkeypatch.setenv(BUDGET_VARIABLE, "3")
    app.data.finz.client = FinzdataClient(key=KEY, root=tmp_path, base_url="https://api.finzdata.com", transport=fake, sleep=lambda s: None, pause=0.0)
    with pytest.raises(ApiError, match=r"at least 4 Finzdata calls.*3 are left today"):
        app.extra_post["/api/data/fetch"]({"kind": "futures", "source": "finzdata", "items": ["ES", "CL"]})
    assert fake.urls == []
    for bad, text in (({"items": []}, "items must be a list"), ({"items": ["x y"]}, "not a Finzdata futures symbol"), ({"items": ["ES"] * 41 + ["CL"]}, "1 to 40")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/data/fetch"]({"kind": "futures", "source": "finzdata", **bad})


def test_a_yahoo_download_of_the_same_root_keeps_the_open_interest_finzdata_stored(fzapp):
    from src.assets.free_data.futures import merge_history

    app, _, _ = fzapp
    run(app, kind="futures", source="finzdata", items=["ES"])
    held = app.data.finz.stored_frame("ES")
    yahoo = held[["date", "contract", "expiry", "price", "volume"]].assign(price=1.0)                  # Yahoo's rows for the same days: another price, no open interest
    merged = merge_history(held, yahoo)
    assert len(merged) == len(held) and (merged["price"] == 1.0).all() and merged["open_interest"].notna().all()


def test_a_downloaded_curve_with_more_than_twelve_contracts_is_listed_in_full_when_asked(fzapp):
    app, _, _ = fzapp
    run(app, kind="futures", source="finzdata", items=["CL"])                                 # crude oil has a contract every month, more than twelve of them trading
    default = get(app, "/api/futures/curve", root="CL")
    every = get(app, "/api/futures/curve", root="CL", contracts="all")
    listed = default["curve"]["listed"]
    assert len(default["curve"]["rows"]) == 12 and listed > 12 and len(every["curve"]["rows"]) == listed and every["curve"]["rows"][-1]["open_interest"] is not None
    assert [r["contract"] for r in every["curve"]["rows"][:12]] == [r["contract"] for r in default["curve"]["rows"]]
