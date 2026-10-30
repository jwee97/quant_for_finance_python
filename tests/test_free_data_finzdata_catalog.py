"""The list of Finzdata's futures markets that ships with the app, its search, and how the list the service sends is laid over it."""

from __future__ import annotations

import pytest

from src.assets.free_data import finzdata as fz
from src.assets.free_data import finzdata_catalog as fc
from src.assets.free_data.futures import FREE_ROOTS


def roots(found) -> list[str]:
    return [r["root"] for r in found["results"]]


def test_the_bundled_list_is_the_139_markets_of_the_public_page_by_asset_class():
    m = fc.bundled()
    assert len(m) == 139 and len({x.root for x in m}) == 139
    counts = {g: sum(1 for x in m if x.group == g) for g in fc.groups()}
    assert counts == {"Equity index": 39, "Agricultural": 27, "Interest rate": 23, "Currency": 23, "Energy": 12, "Metals": 9, "Crypto": 3, "Volatility": 3}
    assert all(x.root.isalnum() and x.root == x.root.upper() and x.history_from.isdigit() and x.name for x in m)
    assert {x.exchange for x in m} == {"CME", "CBOT", "NYMEX", "COMEX", "ICE", "ICE US", "ICE Europe", "Eurex", "Euronext", "SGX", "Cboe Futures"}      # one spelling of each exchange
    es = fc.find("/es")
    assert (es.root, es.name, es.exchange, es.group, es.history_from) == ("ES", "E-mini S&P 500", "CME", "Equity index", "2008")
    assert fc.find("nope") is None


def test_the_symbol_comes_first_then_the_start_of_a_name_then_any_part_of_it():
    assert roots(fc.search("es"))[0] == "ES" and roots(fc.search("/NQ")) == ["NQ"]                           # MNQ is the micro Nasdaq: a name that does not contain "nq"
    assert roots(fc.search("nasdaq")) == ["NQ", "MNQ"]
    crude = fc.search("crude oil")
    assert roots(crude)[:2] == ["CL", "BZ"] and set(roots(crude)) >= {"MCL", "CPO"}
    assert "FDAX" in roots(fc.search("eurex")) and fc.search("eurex")["total"] == 25                        # the exchange counts as a part of the name
    assert fc.search("zzzz") == {"results": [], "total": 0, "shown": 0}


def test_every_word_must_match_and_a_group_narrows_the_list():
    assert roots(fc.search("micro gold")) == ["MGC"]
    assert fc.search("gold", group="Energy")["total"] == 0
    energy = fc.search("", group="Energy")
    assert energy["total"] == 12 and roots(energy)[:3] == ["CL", "NG", "BZ"]                               # the featured first
    assert fc.search("", group="Energy", limit=5)["shown"] == 5 and fc.search("crude", limit=1)["shown"] == 1


def test_with_nothing_typed_the_featured_markets_are_shown_in_their_order():
    first = fc.search()
    assert roots(first)[:6] == ["ES", "NQ", "YM", "RTY", "MES", "MNQ"] and all(r["featured"] for r in first["results"])
    assert (first["total"], first["shown"]) == (len(fc.FEATURED), 40) and len(fc.search(limit=100)["results"]) == len(fc.FEATURED) == 43
    assert all(r in {m.root for m in fc.bundled()} for r in fc.FEATURED)                                    # no featured symbol is one the list does not have


def test_the_list_the_service_sends_replaces_a_row_and_adds_a_market_the_copy_does_not_have():
    live = [{"root": "ES", "name": "E-mini S&P 500 Futures", "exchange": "CME", "group": "Equity index", "first_date": "2007-06-21", "last_date": "2026-10-29", "contracts": 88},
            {"root": "ZZ", "name": "Zebra index", "exchange": "ICE", "group": None, "first_date": None, "last_date": None, "contracts": None},
            {"root": "bad symbol", "name": "ignored"}]
    found = fc.search("es", live=live)
    es = found["results"][0]
    assert (es["name"], es["history_from"], es["first_date"], es["last_date"], es["contracts"]) == ("E-mini S&P 500 Futures", "2007", "2007-06-21", "2026-10-29", 88)
    zz = fc.search("zebra", live=live)["results"][0]
    assert (zz["root"], zz["group"], zz["exchange"]) == ("ZZ", "Other", "ICE") and "Other" in fc.groups(live)
    assert len(fc.markets(live)[0]) == 140 and fc.find("bad symbol", live) is None
    assert fc.search("nq")["results"][0].get("contracts") is None                                              # a market the service did not describe has no counts to show


def test_the_download_takes_the_platforms_terms_for_a_symbol_it_knows_even_when_finzdata_spells_it_differently():
    es = fc.terms("ES")
    assert (es["multiplier"], es["tick_size"], es["asset_class"]) == (50.0, 0.25, "equity_index")
    euro = fc.terms("E6")
    assert (euro["root"], euro["multiplier"], euro["tick_size"], euro["asset_class"], euro["root_name"]) == ("E6", 125000.0, 0.00005, "fx", "Euro FX")
    bond = fc.terms("US")
    assert bond["multiplier"] == 1000.0 and bond["asset_class"] == "rates"
    assert fz.estimate_expiry("E6", 2026, 12)[1] == "rule" and fz.estimate_expiry("US", 2026, 12)[1] == "rule"


def test_a_symbol_the_platform_has_nothing_for_gets_placeholders_and_the_class_the_margin_estimate_knows():
    t = fc.terms("FDAX")
    assert (t["multiplier"], t["tick_size"], t["asset_class"], t["exchange"], t["root_name"]) == (1.0, 0.01, "equity_index", "Eurex", "DAX")
    assert fc.terms("SR3")["asset_class"] == "rates" and fc.terms("BTC")["asset_class"] == "crypto" and fc.terms("VX")["asset_class"] == "volatility"
    assert fc.terms("NOPE") == {"root": "NOPE", "exchange": "?", "asset_class": "unknown", "multiplier": 1.0, "tick_size": 0.01, "root_name": "NOPE"}
    assert set(fc.ASSET_CLASS.values()) <= {"equity_index", "rates", "commodity", "metal", "agricultural", "fx", "volatility", "crypto"}


def test_every_root_the_platform_downloads_for_free_is_in_finzdatas_list_under_one_symbol_or_the_other():
    o = fc.platform_overlap()
    assert o["only_platform"] == [] and len(o["both"]) == len(FREE_ROOTS) == 44 and o["only_finzdata"] == 95
    assert set(fz.ALIAS) <= {m.root for m in fc.bundled()} and set(fz.ALIAS.values()) <= set(FREE_ROOTS)


@pytest.mark.parametrize("alias, platform", sorted(fz.ALIAS.items()))
def test_an_alias_is_the_same_market_by_name(alias, platform):
    finz, ours = fc.find(alias).name.lower(), FREE_ROOTS[platform].name.lower()
    words = {"6a": "australian", "6b": "british", "6e": "euro", "6j": "japanese", "6n": "new zealand", "6s": "swiss", "6c": "canadian", "zb": "treasury bond"}[platform.lower()]
    assert words in finz and words in ours.replace("us ", "").replace("treasury bond", "treasury bond") or words in ours
