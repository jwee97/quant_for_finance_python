"""Ticker changes keep the security; listings of one issuer are one issuer."""

from __future__ import annotations

import pytest

from src.fundamentals import security_master as sm
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.providers.base import CompanyProfile
from src.fundamentals.store import FundamentalsStore


@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    for t, ex in (("FB", "NASDAQ"), ("AAA", "NYSE"), ("AAA", "LSE")):
        s.upsert_security(CompanyProfile("manual", f"{ex}:{t}", t, f"{t} Inc", exchange=ex, market="US", country="US", currency="USD"))
    return s


def ids(store):
    f = store.securities()
    return {(r["ticker"], r["exchange"]): int(r["security_id"]) for _, r in f.iterrows()}


def test_a_ticker_change_keeps_the_security_and_both_names_resolve_as_of_their_dates(store):
    sid = ids(store)[("FB", "NASDAQ")]
    sm.rename_ticker(store, sid, "META", "2022-06-09")
    assert sm.resolve(store, "FB", as_of="2021-01-04") == {"security_id": sid, "status": "ok", "candidates": [sid]}
    assert sm.resolve(store, "META", as_of="2023-01-04")["security_id"] == sid
    assert sm.resolve(store, "META", as_of="2021-01-04")["status"] == "not found" and sm.resolve(store, "FB", as_of="2023-01-04")["status"] == "not found"
    h = sm.history(store, sid)
    assert [(x["ticker"], x["valid_to"]) for x in h] == [("FB", "2022-06-08"), ("META", None)]
    assert store.get_security(sid)["ticker"] == "META"


def test_a_rename_never_merges_two_securities_and_an_ambiguous_ticker_says_so(store):
    m = ids(store)
    with pytest.raises(FundamentalsError, match="never merges"):
        sm.rename_ticker(store, m[("FB", "NASDAQ")], "AAA", "2022-01-01", new_exchange="NYSE")
    amb = sm.resolve(store, "AAA")
    assert amb["status"] == "ambiguous" and sorted(amb["candidates"]) == sorted([m[("AAA", "NYSE")], m[("AAA", "LSE")]])
    assert sm.resolve(store, "AAA", exchange="LSE")["security_id"] == m[("AAA", "LSE")]


def test_one_company_listed_twice_counts_once_in_a_universe(store):
    m = ids(store)
    a, b, c = m[("AAA", "NYSE")], m[("AAA", "LSE")], m[("FB", "NASDAQ")]
    sm.link_listings(store, [a, b], "issuer:aaa", primary=b)
    assert sm.one_per_issuer(store, [a, b, c]) == sorted([b, c])
    with pytest.raises(FundamentalsError):
        sm.link_listings(store, [a], "x")
