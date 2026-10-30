"""Finzdata's futures: contract codes, the readers of its answers, and the curve file built from them.

Nothing here touches the network: the client is driven through ``tests/fake_finzdata.py``. The fake's rows are the likeliest shapes, not a copy of the service's (its OpenAPI document leaves the rows
undefined), so what these tests pin is that the readers survive the other spellings too, and say which columns they were sent when a field they need is missing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assets.free_data import finzdata as fz
from src.assets.free_data.http import FetchError
from src.webapp.finzdata import FinzdataClient, FinzdataError
from tests.fake_finzdata import KEY, TODAY, FakeFinzdata

NOW = pd.Timestamp(TODAY)


def client(fake, tmp_path=None, **kwargs):
    return FinzdataClient(key=KEY, root=tmp_path, base_url="https://api.finzdata.com", transport=fake, sleep=lambda s: None, pause=0.0, **kwargs)


def curve(fake=None, **kwargs):
    fake = fake or FakeFinzdata()
    start = kwargs.pop("start", NOW - pd.Timedelta(days=45))               # an update asks for more than the plan shows, as it does when it tops up an older file
    return fz.fetch_curve(client(fake), "ES", start=start, today=NOW, **kwargs), fake


# ------------------------------------------------------------------------------------------------------------------ contract codes
@pytest.mark.parametrize("code, root, expected", [("ESZ26", "ES", (2026, 12)), ("CLF27", "CL", (2027, 1)), ("CLZ2026", "CL", (2026, 12)), ("esh25", "ES", (2025, 3)), ("ESZ26.CME", "ES", (2026, 12)),
                                                  ("E6Z26", "E6", (2026, 12)), ("6EZ26", "E6", (2026, 12)), ("M2KU27", "M2K", (2027, 9)), ("NGF7", "NG", (2027, 1)), ("Z26", None, (2026, 12))])
def test_a_contract_code_is_decoded_to_its_delivery_month_and_year(code, root, expected):
    assert fz.decode_contract(code, root, reference="2026-10-29") == expected


def test_a_one_digit_year_is_the_nearest_year_that_ends_in_it_and_the_reference_moves_it():
    assert fz.decode_contract("CLZ6", "CL", "2026-10-29") == (2026, 12)
    assert fz.decode_contract("CLZ6", "CL", "2016-05-01") == (2016, 12)               # the last bar of an expired contract is the reference, so an old Z6 is not 2026
    assert fz.decode_contract("CLZ9", "CL", "2026-10-29") == (2029, 12)


@pytest.mark.parametrize("code", ["", "ES", "ESA26", "ESZ", "ES26", None])
def test_a_code_that_is_not_a_contract_is_refused(code):
    with pytest.raises(ValueError):
        fz.decode_contract(code, "ES")


def test_the_platforms_label_is_the_month_letter_and_two_digits():
    assert fz.label(2026, 12) == "Z26" and fz.label(2009, 3) == "H09"


def test_the_expiry_follows_the_exchange_rule_for_a_known_root_and_the_15th_for_any_other():
    assert fz.estimate_expiry("ES", 2026, 12) == (pd.Timestamp("2026-12-18"), "rule")                  # the third Friday
    assert fz.estimate_expiry("E6", 2026, 12) == (pd.Timestamp("2026-12-14"), "rule")                  # the platform's rule for the euro future, which it calls 6E
    assert fz.estimate_expiry("FDAX", 2026, 12) == (pd.Timestamp("2026-12-15"), "15th")                # Tuesday: a root the platform has no rule for
    assert fz.estimate_expiry("FDAX", 2026, 8) == (pd.Timestamp("2026-08-14"), "15th")                 # the 15th of August 2026 is a Saturday: the Friday before


# ------------------------------------------------------------------------------------------------------------------ the answers
def test_a_root_may_be_written_with_a_slash_and_in_any_case():
    assert [fz.clean_root(x) for x in ("/ES", "es", " ES ")] == ["ES"] * 3


def test_the_list_of_markets_is_read_whichever_names_the_columns_have():
    plain = fz.normalise_roots({"data": FakeFinzdata()._roots_rows()})
    alt = fz.normalise_roots({"data": FakeFinzdata(alt_names=True)._roots_rows()})
    assert [r["root"] for r in plain] == [r["root"] for r in alt] == ["ES", "CL", "E6", "FDAX"]
    assert plain[0] == {"root": "ES", "name": "E-mini S&P 500", "exchange": "CME", "group": "Equity index", "first_date": "2008-01-02", "last_date": TODAY, "contracts": 24}
    assert alt[0]["name"] == "E-mini S&P 500" and alt[0]["exchange"] == "CME" and alt[0]["group"] == "Equity index" and alt[0]["first_date"] == "2008-01-02" and alt[0]["contracts"] is None


def test_an_answer_with_no_symbol_names_the_columns_it_did_have():
    with pytest.raises(fz.ShapeError, match=r"no symbol.*the columns it sent are: label, venue"):
        fz.normalise_roots({"data": [{"label": "E-mini", "venue": "CME"}]})
    with pytest.raises(fz.ShapeError, match="no list of rows"):
        fz.normalise_roots({"detail": "x", "count": 3})
    with pytest.raises(fz.ShapeError, match="not objects"):
        fz.normalise_roots({"data": ["ES", "NQ"]})


def test_the_envelope_may_be_a_bare_list_or_a_document_with_one_list_in_it():
    rows = [{"root": "ES", "name": "E-mini"}]
    assert fz.rows_of(rows, "x") == rows and fz.rows_of({"data": rows}, "x") == rows and fz.rows_of({"markets": rows, "n": 1}, "x") == rows


def test_what_the_service_says_about_the_history_it_left_out_is_kept_as_plain_sentences():
    answer = {"data": [], "history_note": "Your plan   includes the last 30 days.", "upgrade": {"message": "Full history: $49", "url": "https://finzdata.com/shop"}}
    assert fz.notes_of(answer) == ["Your plan includes the last 30 days.", "Full history: $49"]
    assert fz.notes_of({"data": []}) == [] and fz.notes_of([]) == []


def test_the_contract_list_has_a_label_an_expiry_and_the_first_and_last_bar_of_every_contract():
    fake = FakeFinzdata()
    frame = fz.normalise_contracts({"data": fake._contract_rows("ES")}, "ES", NOW)
    assert len(frame) == 24 and frame["contract"].is_unique and frame["expiry"].is_monotonic_increasing
    z26 = frame[frame["contract"] == "Z26"].iloc[0]
    assert (z26["code"], z26["expiry"], z26["expiry_basis"], z26["first_date"], z26["last_date"]) == ("ESZ26", pd.Timestamp("2026-12-18"), "service", pd.Timestamp("2026-12-18") - pd.Timedelta(days=700), NOW)
    assert frame.attrs["undecodable"] == []


def test_without_an_expiry_a_contract_that_stopped_trading_expires_on_its_last_bar_and_a_live_one_by_rule():
    frame = fz.normalise_contracts({"data": FakeFinzdata(no_expiry=True)._contract_rows("ES")}, "ES", NOW)
    expired, live = frame[frame["contract"] == "U26"].iloc[0], frame[frame["contract"] == "Z26"].iloc[0]
    assert (expired["expiry_basis"], expired["expiry"]) == ("last bar", pd.Timestamp("2026-09-18"))
    assert (live["expiry_basis"], live["expiry"]) == ("rule", pd.Timestamp("2026-12-18"))
    other = fz.normalise_contracts({"data": FakeFinzdata(no_expiry=True)._contract_rows("FDAX")}, "FDAX", NOW)
    assert set(other["expiry_basis"]) == {"last bar", "15th"}


def test_a_code_that_cannot_be_read_is_left_out_and_listed():
    frame = fz.normalise_contracts({"data": [{"contract": "ESZ26"}, {"contract": "ES-SPREAD"}, {"contract": "ESH27"}]}, "ES", NOW)
    assert list(frame["contract"]) == ["Z26", "H27"] and frame.attrs["undecodable"] == ["ES-SPREAD"]
    with pytest.raises(fz.ShapeError, match="no contract code.*the columns it sent are: foo"):
        fz.normalise_contracts({"data": [{"foo": 1}]}, "ES", NOW)


def test_the_bars_are_read_with_the_settlement_when_there_is_one_and_the_close_when_not():
    plain = fz.normalise_bars({"data": [{"date": "2026-10-28", "contract": "ESZ26", "open": 1, "high": 3, "low": 1, "close": 2.5, "volume": 10, "open_interest": 7}]}, "ES")
    assert plain.iloc[0].to_dict() == {"date": pd.Timestamp("2026-10-28"), "code": "ESZ26", "price": 2.5, "open": 1.0, "high": 3.0, "low": 1.0, "volume": 10.0, "open_interest": 7.0}
    assert plain.attrs["price_column"] == "close"
    alt = fz.normalise_bars({"data": [{"trade_date": "2026-10-28T00:00:00Z", "symbol": "ESZ26", "settle": 2.4, "close": 2.5, "oi": 9}]}, "ES")
    assert alt.iloc[0]["price"] == 2.4 and alt.iloc[0]["open_interest"] == 9 and alt.attrs["price_column"] == "settle" and alt.iloc[0]["date"] == pd.Timestamp("2026-10-28")


def test_a_bar_without_a_price_or_with_a_price_that_is_not_positive_is_dropped_and_missing_columns_are_named():
    got = fz.normalise_bars({"data": [{"date": "2026-10-27", "close": 0}, {"date": "2026-10-28", "close": None}, {"date": "bad", "close": 5}, {"date": "2026-10-29", "close": "7.5"}]}, "ES")
    assert list(got["price"]) == [7.5]
    with pytest.raises(fz.ShapeError, match=r"no price \(settle, close or price\).*columns it sent are: date, volume"):
        fz.normalise_bars({"data": [{"date": "2026-10-28", "volume": 5}]}, "ES")
    with pytest.raises(fz.ShapeError, match="no date"):
        fz.normalise_bars({"data": [{"close": 5}]}, "ES")
    assert fz.normalise_bars({"data": []}, "ES").empty


# ------------------------------------------------------------------------------------------------------------------ the curve
def expected_contracts(fake, root="ES"):
    start = NOW - pd.Timedelta(days=30)
    return {r["contract"] for r in fake._bar_rows(root, None, start)}


def test_every_contract_of_a_root_comes_in_one_call_with_open_interest_and_labels_the_tab_reads():
    d, fake = curve()
    f = d.frame
    assert list(f.columns) == ["date", "contract", "expiry", "price", "open_interest", "volume"]
    assert [p for p, _ in fake.requests] == ["/v1/futures/contracts", "/v1/futures/daily"]                  # two calls for the whole root
    assert fake.requests[1][1] == {"root": "ES", "adjust": "none", "start": "2026-09-14", "format": "json", "limit": "1000000"}
    assert set(f["contract"]) == {c[2] + c[-2:] for c in expected_contracts(fake)} and len(set(f["contract"])) == 8
    assert f["open_interest"].notna().all() and (f["price"] > 0).all() and (f["expiry"] >= f["date"]).all()
    assert f["date"].min() == pd.Timestamp("2026-09-29") and f["date"].max() == NOW
    assert d.kind == "futures" and d.item == "curve" and d.name == "ES"
    m = d.meta
    assert (m["source_id"], m["source"], m["root"], m["calls"], m["mode"], m["as_of"], m["has_open_interest"]) == ("finzdata", "Finzdata", "ES", 2, "every contract in one call", TODAY, True)
    assert m["contracts"] == len(set(f["contract"])) and m["contracts_listed"] == 24 and m["furthest"] == "U28" and m["furthest_expiry"] == "2028-09-15"          # Z28 is listed and has not started trading
    assert any("last 30 days" in n for n in m["notes"]) and any("$49" in n for n in m["notes"]) and any("close" in n for n in m["notes"])


def test_a_service_that_insists_on_a_contract_is_asked_contract_by_contract_and_the_table_is_the_same():
    whole, _ = curve()
    fake = FakeFinzdata(per_contract_only=True)
    d, _ = curve(fake)
    asked = [c for c in fake.contracts("ES") if c["last"] >= NOW - pd.Timedelta(days=50) and c["first"] <= NOW]          # the contracts that can have a bar since 45 days ago (five days of slack)
    assert d.meta["mode"] == "one call per contract" and d.meta["calls"] == 2 + len(asked) == 11                     # the list, the refused root-wide call, one call per contract
    assert [q["contract"] for p, q in fake.requests if p == "/v1/futures/daily" and "contract" in q] == [c["code"] for c in asked]
    assert d.frame.equals(whole.frame)
    assert any("one by one" in n for n in d.meta["notes"])


def test_rows_that_do_not_say_which_contract_they_are_of_are_asked_for_contract_by_contract_too():
    fake = FakeFinzdata(anonymous_rows=True)
    d, _ = curve(fake)
    whole, _ = curve()
    assert d.meta["mode"] == "one call per contract" and d.frame.equals(whole.frame)


def test_other_spellings_of_every_column_give_the_same_curve():
    base, _ = curve()
    d, fake = curve(FakeFinzdata(alt_names=True))
    assert d.frame[["date", "contract", "price", "open_interest", "volume"]].equals(base.frame[["date", "contract", "price", "open_interest", "volume"]])
    assert d.frame["expiry"].equals(base.frame["expiry"])


def test_without_an_expiry_in_the_list_the_curve_still_has_one_for_every_row_and_none_before_its_date():
    d, _ = curve(FakeFinzdata(no_expiry=True))
    assert (d.frame["expiry"] >= d.frame["date"]).all() and d.frame["expiry"].notna().all()
    assert any("Expiry dates" in n and "rule" in n for n in d.meta["notes"])


def test_a_root_the_platform_has_no_rule_for_is_still_a_curve_and_says_how_its_expiries_were_found():
    d = fz.fetch_curve(client(FakeFinzdata(no_expiry=True)), "FDAX", start=NOW - pd.Timedelta(days=30), today=NOW,
                       terms={"root_name": "DAX", "exchange": "Eurex", "asset_class": "equity_index", "multiplier": 1.0, "tick_size": 0.01})
    assert d.meta["root_name"] == "DAX" and d.name == "FDAX" and set(d.frame["contract"]) == {"H27", "M27", "U27", "Z26", "Z27", "H28", "M28", "U28"}
    assert (d.frame["expiry"] >= d.frame["date"]).all() and d.frame.groupby("contract")["expiry"].first()["Z26"] == pd.Timestamp("2026-12-15")
    assert any("15th" in n and "Expiry dates" in n for n in d.meta["notes"])


def test_the_calls_are_capped_and_the_contracts_left_out_are_named():
    fake = FakeFinzdata(per_contract_only=True)
    d, _ = curve(fake, max_calls=5, start=NOW - pd.Timedelta(days=30))
    assert d.meta["calls"] == 5 and d.frame["contract"].nunique() == 3                                  # the list, the refused call and three contracts
    assert any("Only 3 of the" in n and "to stay within 5 calls" in n for n in d.meta["notes"])


def test_a_period_with_no_bar_stops_the_contract_by_contract_search_after_five_empty_answers():
    fake = FakeFinzdata(per_contract_only=True, window=1)
    with pytest.raises(FetchError, match="no daily bars for ES"):
        fz.fetch_curve(client(fake), "ES", start=NOW - pd.Timedelta(days=500), today=NOW, max_calls=200)   # the fake keeps one day of history; contracts that ended before it answer with nothing
    assert len([p for p, _ in fake.requests if p == "/v1/futures/daily"]) == 1 + 5                      # the refused root-wide call and five empty answers


@pytest.mark.parametrize("fake_kwargs, status, kind", [({"forbidden": ("ES",)}, 403, "plan"), ({"quota": 1}, 429, "quota")])
def test_a_refusal_of_the_plan_or_the_allowance_is_raised_and_not_tried_again_contract_by_contract(fake_kwargs, status, kind):
    fake = FakeFinzdata(**fake_kwargs)
    with pytest.raises(FinzdataError) as caught:
        fz.fetch_curve(client(fake), "ES", start=NOW - pd.Timedelta(days=30), today=NOW)
    assert (caught.value.status, caught.value.kind) == (status, kind)
    assert len([p for p, _ in fake.requests if p == "/v1/futures/daily"]) <= 1


def test_a_wrong_key_is_a_401_that_says_so_and_never_repeats_the_key():
    fake = FakeFinzdata()
    bad = FinzdataClient(key="fz_wrongwrongwrongwrongwrong", base_url="https://api.finzdata.com", transport=fake, sleep=lambda s: None, pause=0.0)
    with pytest.raises(FinzdataError, match="does not accept the key") as caught:
        fz.fetch_curve(bad, "ES", today=NOW)
    assert caught.value.kind == "key" and caught.value.fatal and "wrongwrong" not in str(caught.value)


def test_bars_of_contracts_the_list_does_not_have_are_refused_with_both_sides_named():
    contracts = fz.normalise_contracts({"data": [{"contract": "ESZ26"}]}, "ES", NOW)
    bars = fz.normalise_bars({"data": [{"date": "2026-10-28", "contract": "ESH30", "close": 5}]}, "ES")
    with pytest.raises(fz.ShapeError, match=r"ESH30.*is in the contract list.*ESZ26"):
        fz.curve_frame(bars, contracts, "ES")
    anonymous = fz.normalise_bars({"data": [{"date": "2026-10-28", "close": 5}]}, "ES")
    with pytest.raises(fz.ShapeError, match="do not say which contract"):
        fz.curve_frame(anonymous, contracts, "ES")
    assert len(fz.curve_frame(anonymous, contracts, "ES", code="ESZ26")) == 1                          # a call made for one contract knows which


def test_an_estimated_expiry_before_the_last_bar_of_a_finished_contract_is_moved_to_that_bar():
    contracts = fz.normalise_contracts({"data": [{"contract": "ESH26", "expiry": "2026-03-10"}, {"contract": "ESZ26", "expiry": "2026-12-18"}]}, "ES", NOW)
    bars = fz.normalise_bars({"data": [{"date": d, "contract": c, "close": 5} for d, c in (("2026-03-12", "ESH26"), ("2026-10-28", "ESZ26"))]}, "ES")
    got = fz.curve_frame(bars, contracts, "ES").set_index("contract")
    assert got.loc["H26", "expiry"] == pd.Timestamp("2026-03-12") and got.loc["Z26", "expiry"] == pd.Timestamp("2026-12-18")


def test_a_live_contract_that_traded_past_its_estimated_expiry_stays_in_the_curve_and_expires_on_its_last_bar():
    contracts = fz.normalise_contracts({"data": [{"contract": "CLX26"}, {"contract": "CLZ26"}]}, "CL", NOW)
    assert contracts.set_index("contract").loc["X26", "expiry"] < pd.Timestamp("2026-10-27")                       # the rule is days earlier than the bars
    bars = fz.normalise_bars({"data": [{"date": d, "contract": c, "close": 70} for d in ("2026-10-27", "2026-10-28") for c in ("CLX26", "CLZ26")]}, "CL")
    got = fz.curve_frame(bars, contracts, "CL")
    assert (got["expiry"] >= got["date"]).all() and len(got) == 4                                                   # the tab drops a row dated after its expiry, and that was the front contract
    assert (got[got["contract"] == "X26"]["expiry"] == pd.Timestamp("2026-10-28")).all()
    given = fz.normalise_contracts({"data": [{"contract": "CLX26", "expiry": "2026-10-20"}, {"contract": "CLZ26", "expiry": "2026-11-20"}]}, "CL", NOW)
    assert (fz.curve_frame(bars, given, "CL").query("contract == 'X26'")["expiry"] == pd.Timestamp("2026-10-20")).all()   # the service's own date for a live contract is not second-guessed


def test_the_notes_name_the_contracts_whose_expiry_was_moved_to_their_last_bar():
    fake = FakeFinzdata(no_expiry=True)
    d = fz.fetch_curve(client(fake), "CL", start=NOW - pd.Timedelta(days=12), today=NOW)
    assert (d.frame["expiry"] >= d.frame["date"]).all() and "X26" in set(d.frame[d.frame["date"] == NOW]["contract"])
    assert any("X26" in n and "last bar" in n for n in d.meta["notes"]) and set(d.meta["expiry_basis"]) == {"last bar", "rule"}
    clean, _ = curve()
    assert not any("their expiry is the day of their last bar" in n for n in clean.meta["notes"])


def test_a_curve_downloaded_here_loads_in_the_futures_tab_and_has_its_open_interest(tmp_path):
    from src.assets.futures_analytics import load_curve_file

    d, _ = curve()
    target = tmp_path / "ES.csv"
    d.frame.to_csv(target, index=False)
    data = load_curve_file(target, "ES", source="downloaded")
    assert data.long["open_interest"].notna().all() and data.long["contract"].nunique() == d.frame["contract"].nunique()
    assert not any("no open interest" in n for n in data.notes)


# ------------------------------------------------------------------------------------------------------------------ a window of thirty days
def test_the_days_no_download_caught_are_the_stretches_between_stored_dates_and_up_to_today():
    dates = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05", "2026-09-10", "2026-09-11", "2026-10-02"])
    assert fz.gaps(dates) == [{"from": "2026-08-05", "to": "2026-09-10", "days": 36}, {"from": "2026-09-11", "to": "2026-10-02", "days": 21}]
    assert fz.gaps(dates, today="2026-10-10")[-1] == {"from": "2026-10-02", "to": "2026-10-10", "days": 8, "open": True}
    assert fz.gaps(pd.bdate_range("2026-09-01", "2026-09-30"), today="2026-10-01") == []                # weekends and a long weekend are not gaps
    assert fz.gaps([]) == []


def test_how_many_days_are_left_before_the_window_slides_past_the_newest_stored_bar():
    assert fz.staleness("2026-10-20", "2026-10-29") == {"age": 9, "left": 21, "urgent": False}
    assert fz.staleness("2026-10-01", "2026-10-29") == {"age": 28, "left": 2, "urgent": True}
    assert fz.staleness("2026-12-01", "2026-10-29")["age"] == 0
