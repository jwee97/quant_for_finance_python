"""iTick's product lists, bundled: what can be searched without a call, how a typed product is read, and that the lists are what the downloads expect."""

from __future__ import annotations

import pytest

from src.assets.free_data import itick_catalog as ic


def test_the_bundled_lists_are_the_size_of_itick_published_ones_and_have_no_duplicate_codes():
    products = ic.futures_products()
    assert len(products) > 1500 and {p.region for p in products} == {"US", "HK", "CN"}
    assert len({(p.region, p.code) for p in products}) == len(products), "a code is unique within its region"
    assert all(p.code == p.code.upper() and ic.ITEM.match(p.item) and p.name and p.exchange and p.group for p in products)
    by_region = {r: sum(p.region == r for p in products) for r in ic.REGIONS}
    assert by_region["US"] > 1300 and by_region["HK"] > 150 and by_region["CN"] > 40
    assert all(not any("一" <= ch <= "鿿" for ch in p.name) for p in products), "names are in English"
    assert len(ic.coins()) > 150 and ic.coins()["BTC"] == "Bitcoin" and all(c.isalnum() and c == c.upper() for c in ic.coins())
    assert ic.groups()[0] == "Energy" and set(ic.groups()) >= {"Indexes", "Currencies", "Agriculture", "Metals", "Rates"}


def test_every_featured_product_is_listed_and_the_free_curve_roots_are_among_them():
    from src.assets.free_data.futures import FREE_ROOTS

    for region, codes in ic.FEATURED.items():
        for code in codes:
            assert ic.find(region, code) is not None, (region, code)
    assert set(FREE_ROOTS) <= set(ic.FEATURED["US"]), "whatever Yahoo can draw a curve for is one of the products people look for"
    assert ic.find("US", "NQ").name == "E-mini Nasdaq-100 Futures" and ic.find("US", "NQ").exchange == "CME" and ic.find("US", "NOPE") is None


def test_the_search_ranks_the_code_then_the_start_of_a_word_then_the_rest_and_filters():
    assert ic.search("")["results"][0]["item"] == "US:ES" and all(r["featured"] for r in ic.search("", limit=200)["results"])
    nasdaq = ic.search("nasdaq")
    assert nasdaq["results"][0]["item"] == "US:NQ" and nasdaq["total"] == len([p for p in ic.futures_products() if "nasdaq" in p.name.lower() or "nasdaq" in p.code.lower()])
    assert ic.search("hsi")["results"][0]["item"] == "HK:HSI", "a code first"
    assert [r["item"] for r in ic.search("hang seng", region="HK")["results"]][:2] == ["HK:HSI", "HK:MHI"]
    assert {r["region"] for r in ic.search("gold", region="CN")["results"]} == {"CN"} and ic.search("gold", region="CN")["results"][0]["item"] == "CN:AU"
    both = ic.search("micro gold")
    assert both["results"] and all("micro" in r["name"].lower() and "gold" in r["name"].lower() for r in both["results"]), "every word must match"
    assert ic.search("zzzzqqq")["results"] == [] and ic.search("zzzzqqq")["total"] == 0
    assert len(ic.search("e", limit=5)["results"]) == 5 and ic.search("e", limit=5)["shown"] == 5 and ic.search("e", limit=5)["total"] > 5
    assert {r["group"] for r in ic.search("crude", group="Energy", limit=50)["results"]} == {"Energy"}
    assert ic.search("", group="Agriculture")["total"] > 50 and ic.search("", region="HK")["total"] == sum(p.region == "HK" for p in ic.futures_products()), "choosing a market or a group without a word lists all of it"
    assert ic.search("", region="CN", limit=3)["results"][0]["item"] == "CN:IF", "the products people look for come first"


def test_a_typed_product_is_read_with_or_without_a_region():
    assert ic.parse_item("us:nq") == ("US", "NQ") and ic.parse_item(" HK : hsi ") == ("HK", "HSI") and ic.parse_item("nq") == ("US", "NQ") and ic.parse_item("CN:IF") == ("CN", "IF")
    for bad in ("", "US:", ":NQ", "XX:NQ", "US:N Q", "US:NQ/../x", "US:" + "N" * 13, "http://x", None):
        with pytest.raises(ValueError, match="not an iTick futures product"):
            ic.parse_item(bad)


def test_the_coin_search_finds_symbols_and_names():
    assert ic.search_coins("sol")["results"][0]["coin"] == "SOL" and ic.search_coins("bitcoin")["results"][0]["coin"] in ("BTC", "BCH", "WBTC", "BSV")
    first = [r["coin"] for r in ic.search_coins("")["results"]]
    assert first[:3] == ["BTC", "ETH", "SOL"] and ic.search_coins("zzzzqqq")["total"] == 0 and ic.search_coins("a", limit=3)["shown"] == 3
