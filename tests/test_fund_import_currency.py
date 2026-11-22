"""Reporting currency at import: a report in another currency than the one held is blocked until it is converted (rates from the table, else Yahoo) or declared a real change."""

from __future__ import annotations

import pandas as pd
import pytest

from src.fundamentals import manual
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.importer.pipeline import ImportService
from src.fundamentals.providers.base import CompanyProfile
from src.fundamentals.store import FundamentalsStore


def _fx_fetch(calls, rate_from=0.0100, rate_to=0.0110):
    """A stand-in for Yahoo: JPYUSD=X rising steadily from ``rate_from`` to ``rate_to`` over the days asked for."""
    def fetch(symbols, start):
        calls.append((list(symbols), start))
        idx = pd.bdate_range(start, "2024-12-31")
        out = {}
        for s in symbols:
            out[s] = pd.DataFrame({"close": [rate_from + (rate_to - rate_from) * i / (len(idx) - 1) for i in range(len(idx))]}, index=idx) if s == "JPYUSD=X" else "no data found"
        return out
    return fetch


@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    s.upsert_security(CompanyProfile("manual", "jp:7203", "7203", "Toyota Test", exchange="T", market="JP", country="JP", currency="USD"))
    sid = int(s.securities().iloc[0]["security_id"])
    manual.save(s, {"security_id": sid, "units": {"money": "units", "shares": "units"}, "currency": "USD",
                    "periods": [{"fiscal_period": "FY", "period_end": "2022-12-31", "fiscal_year": 2022, "filing_date": "2023-03-01", "currency": "USD", "values": {"revenue": 900.0, "net_income": 90.0, "total_assets": 5000.0}}]})
    return s


def _draft(currency="JPY"):
    return {"method": "csv", "units": {"money": "units", "shares": "units"}, "currency": currency, "units_stated": True, "notes": [], "company": {},
            "periods": [{"fiscal_period": "FY", "period_end": "2023-12-31", "fiscal_year": 2023, "filing_date": "2024-03-01", "currency": currency,
                         "values": {"revenue": 1_000_000.0, "net_income": 100_000.0, "total_assets": 5_000_000.0, "shares_outstanding": 1000.0},
                         "sources": {"revenue": {"tag": None, "reported": 1_000_000.0, "unit": "units", "method": "csv"}}}]}


def _item(svc, store, draft):
    sid = int(store.securities().iloc[0]["security_id"])
    batch = svc.create_batch("t", security_id=sid)
    return svc.add_structured(batch, "r.csv", None, b"x" + repr(draft).encode(), draft, kind="file")["item_id"], sid


def test_a_report_in_another_currency_is_blocked_with_the_way_out(store):
    svc = ImportService(store)
    item, _ = _item(svc, store, _draft())
    issues = svc.item(item, full=True)["issues"]
    assert any(i["level"] == "error" and i["kind"] == "currency" and "Convert to USD" in i["message"] for i in issues)
    svc.decide(item, [{"action": "confirm_suggested"}])
    with pytest.raises(FundamentalsError, match="not approved: the report is in JPY"):
        svc.approve(item)


def test_converting_fetches_the_rates_once_uses_average_for_flows_and_closing_for_balances_and_keeps_the_original(store):
    calls = []
    svc = ImportService(store, fx_fetch=_fx_fetch(calls))
    item, sid = _item(svc, store, _draft())
    out = svc.review(item, {"convert_to": "USD"})
    d = out["reviewed"]
    p = d["periods"][0]
    assert d["currency"] == "USD" and p["currency"] == "USD" and calls and calls[0][0] == ["JPYUSD=X"] and len(calls) == 1
    assert not [i for i in out["issues"] if i["level"] == "error"] and any(i["kind"] == "currency_converted" for i in out["issues"])
    closing, average = p["values"]["total_assets"] / 5_000_000.0, p["values"]["revenue"] / 1_000_000.0
    assert p["values"]["shares_outstanding"] == 1000.0, "a count of shares is not an amount of money"
    assert 0.0100 < average < closing < 0.0110 + 1e-9, "a rising yen: the year's average is below the year-end rate"
    assert p["sources"]["revenue"]["original_currency"] == "JPY" and "converted to USD" in p["sources"]["revenue"]["unit"] and p["sources"]["revenue"]["reported"] == 1_000_000.0
    svc.decide(item, [{"action": "confirm_suggested"}])
    done = svc.approve(item)
    assert done["item"]["status"] == "saved"
    cur = set(store.periods([sid])["reporting_currency"].dropna())
    assert cur == {"USD"}
    with store.db.transaction() as c:
        row = c.execute("SELECT reported_value, reported_unit FROM observation_provenance WHERE metric_code = 'revenue' AND import_item_id = ?", (item,)).fetchone()
    assert row[0] == 1_000_000.0 and "JPY" in row[1] and "average rate" in row[1]
    fx = store.fx()
    assert len(fx[(fx["quote"] == "JPY")]) > 100 and set(fx["source"] if "source" in fx else ["yahoo"]) is not None


def test_stored_rates_are_used_and_yahoo_is_not_asked(store):
    idx = pd.bdate_range("2022-06-01", "2024-06-30")
    store.store_fx(pd.DataFrame({"date": idx, "base": "USD", "quote": "JPY", "rate": 100.0}), "test")
    calls = []
    svc = ImportService(store, fx_fetch=_fx_fetch(calls))
    item, _ = _item(svc, store, _draft())
    p = svc.review(item, {"convert_to": "USD"})["reviewed"]["periods"][0]
    assert calls == [] and p["values"]["revenue"] == pytest.approx(10_000.0) and p["values"]["total_assets"] == pytest.approx(50_000.0)


def test_no_rate_means_no_conversion_and_a_reason_not_a_guess(store):
    svc = ImportService(store, fx_fetch=lambda symbols, start: {s: "no data found" for s in symbols})
    item, _ = _item(svc, store, _draft())
    out = svc.review(item, {"convert_to": "USD"})
    assert any(i["level"] == "error" and "no JPY/USD exchange rate" in i["message"] for i in out["issues"])
    assert out["reviewed"]["periods"][0]["values"]["revenue"] == 1_000_000.0 and out["reviewed"]["currency"] == "JPY"
    svc.decide(item, [{"action": "confirm_suggested"}])
    with pytest.raises(FundamentalsError, match="not approved"):
        svc.approve(item)


def test_choosing_as_reported_again_restores_the_first_reading(store):
    svc = ImportService(store, fx_fetch=_fx_fetch([]))
    item, _ = _item(svc, store, _draft())
    svc.review(item, {"convert_to": "USD"})
    back = svc.review(item, {"convert_to": ""})["reviewed"]
    p = back["periods"][0]
    assert back["currency"] == "JPY" and p["values"]["revenue"] == 1_000_000.0 and "conversion" not in back and "original_currency" not in p["sources"]["revenue"]


def test_a_real_change_of_reporting_currency_can_be_declared_and_is_stored_as_reported(store):
    svc = ImportService(store)
    d = _draft()
    d["periods"][0]["values"] = {"revenue": 1000.0, "net_income": 100.0, "total_assets": 5000.0}         # the same size as the held figures, so the unit check has nothing to say
    item, sid = _item(svc, store, d)
    out = svc.review(item, {"currency_change_ok": True})
    assert not [i for i in out["issues"] if i["level"] == "error"] and any(i["level"] == "warning" and "really changed" in i["message"] for i in out["issues"])
    svc.decide(item, [{"action": "confirm_suggested"}])
    svc.approve(item)
    assert set(store.periods([sid])["reporting_currency"].dropna()) == {"USD", "JPY"}


def test_a_filing_with_two_currencies_inside_it_is_warned_about(store):
    svc = ImportService(store)
    d = _draft("USD")
    d["periods"].append({**_draft("JPY")["periods"][0], "fiscal_period": "FY", "period_end": "2021-12-31", "fiscal_year": 2021})
    item, _ = _item(svc, store, d)
    assert any("different currencies" in i["message"] for i in svc.item(item, full=True)["issues"])


def test_the_exchange_rate_basis_and_currencies_of_a_conversion_are_stored_with_the_figure(store):
    svc = ImportService(store, fx_fetch=_fx_fetch([]))
    item, sid = _item(svc, store, _draft())
    svc.review(item, {"convert_to": "USD"})
    svc.decide(item, [{"action": "confirm_suggested"}])
    svc.approve(item)
    with store.db.transaction() as c:
        flow = c.execute("SELECT fx_rate, fx_basis, fx_from, fx_to, fx_source, reported_value FROM observation_provenance WHERE metric_code = 'revenue'").fetchone()
    assert 0.0100 < flow[0] < 0.0111 and flow[1] == "average" and (flow[2], flow[3]) == ("JPY", "USD") and "Yahoo" in flow[4] and flow[5] == 1_000_000.0, "a flow at the average rate of its period, the original kept"
