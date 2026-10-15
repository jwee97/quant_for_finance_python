"""Searching every future Yahoo Finance lists from the Futures card and downloading the price history of one: the search over the bundled list, the lookup that asks Yahoo about the words typed, the
download of ``ES=F`` or a single contract as a series for the History view. Nothing here touches the network (Yahoo is a table of answers in the shape of the live service)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assets.free_data import yahoo_futures as yf
from src.webapp.api import ApiError
from tests import free_data_helpers as fh
from tests.test_webapp_data import YAHOO, _safe, feeds, run, yahoo_route
from tests.test_webapp_data import app as data_app  # noqa: F401  (the module's app fixture)

LOOKUP = yf.LOOKUP


def test_the_status_offers_yahoos_list_and_the_search_reads_it_with_no_call(data_app):  # noqa: F811
    app = data_app
    f = feeds(app, {})
    st = _safe(app.extra_get["/api/data/status"]({}))
    y = st["kinds"]["futures"]["yahoo"]
    assert y["products"] >= 300 and "Energy" in y["groups"] and y["years"] == [1, 2, 3, 5, 10, 20] and y["defaults"] == {"years": 5}
    found = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "source": "yahoo", "q": "lumber"}))
    assert found["source"] == "yahoo" and [r["item"] for r in found["results"]] == ["LBR=F"] and found["results"][0]["region"] == "YF" and "Energy" in found["groups"]
    metals = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "source": "yahoo", "group": "Metals", "limit": 5}))
    assert len(metals["results"]) == 5 and {r["group"] for r in metals["results"]} == {"Metals"} and metals["total"] > 15
    start = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "source": "yahoo"}))
    assert [r["item"] for r in start["results"]][:3] == ["CL=F", "NG=F", "HO=F"]
    other = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "q": "nasdaq"}))
    assert other["results"][0]["item"] == "US:NQ" and "source" not in other, "without a source the search is still iTick's list"
    with pytest.raises(ApiError, match="group must be one of"):
        app.extra_get["/api/data/search"]({"kind": "futures", "source": "yahoo", "group": "Everything"})
    assert f.calls == [], "searching the list makes no call"


def test_a_lookup_asks_yahoo_keeps_the_answer_and_the_next_search_finds_it(data_app, tmp_path):  # noqa: F811
    app = data_app
    docs = [fh.yahoo_future_doc("OATZ=F", "Oat Futures,Dec-2026", "CBT"), fh.yahoo_future_doc("OATZ26.CBT", "Oat Futures,Dec-2026", "CBT")]
    f = feeds(app, {LOOKUP: fh.yahoo_lookup(docs)})
    out = _safe(app.extra_post["/api/data/lookup"]({"kind": "futures", "source": "yahoo", "q": "zzoats"}))
    assert out["found"] == 2 and out["new"] == 2 and out["total"] == 2 and out["shown"] == 2 and out["pages"] == 1 and {r["item"] for r in out["results"]} == {"OATZ=F", "OATZ26.CBT"}
    assert "query=zzoats" in f.calls[0] and len(f.calls) == 1
    again = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "source": "yahoo", "q": "oat"}))
    assert again["results"][0]["item"] == "OATZ=F" and again["results"][0]["name"] == "Oat Futures", "kept: the search finds it without a call"
    assert (app.root / "data" / "user" / "yahoo_futures.json").exists() and len(f.calls) == 1
    for body, text in (({"kind": "crypto", "q": "sol"}, "only the futures of Yahoo"), ({"kind": "futures", "source": "itick", "q": "es"}, "only the futures of Yahoo"),
                       ({"kind": "futures", "source": "yahoo", "q": ""}, "a word or two"), ({"kind": "futures", "source": "yahoo", "q": "x" * 50}, "a word or two")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/data/lookup"](body)
    feeds(app, {LOOKUP: (429, "Too Many Requests")})
    with pytest.raises(ApiError, match="429|Too Many|refus|rate"):
        app.extra_post["/api/data/lookup"]({"kind": "futures", "source": "yahoo", "q": "wheat"})


def test_the_price_history_of_a_future_downloads_without_a_key_and_the_history_view_reads_it(data_app):  # noqa: F811
    app = data_app
    f = feeds(app, {YAHOO + "ES=F": yahoo_route(5000.0, days=300), YAHOO + "LBR=F": yahoo_route(500.0, days=300), YAHOO + "ZZ=F": (404, fh.YAHOO_MISSING)})
    result = run(app, kind="futures", source="yahoo_series", items=["ES=F", "lbr=f", "ZZ=F"], years=3)
    by = {r["item"]: r for r in result["results"]}
    assert result["ok"] == 2 and result["failed"] == 1 and by["ES=F"]["ok"] and by["LBR=F"]["ok"] and not by["ZZ=F"]["ok"] and "404" in by["ZZ=F"]["message"]
    assert by["ES=F"]["name"] == "YF_ES" and by["ES=F"]["what"] == "series" and "300 daily bars" in by["ES=F"]["message"] and "Yahoo Finance" in by["ES=F"]["message"]
    assert all("range=5y" in u for u in f.calls) and len(f.calls) == 3, "three years is asked for as five, one call each"
    names = {d["name"]: d for d in result["downloads"]}
    assert {"YF_ES", "YF_LBR"} <= set(names) and names["YF_ES"]["item"] == "series" and names["YF_ES"]["region"] == "YF" and names["YF_ES"]["refresh"] == {"source": "yahoo_series", "items": ["ES=F"], "years": 3}
    catalog = _safe(app.extra_get["/api/assets/catalog"]({}))
    row = next(x for x in catalog["futures_series"] if x["id"] == "YF_ES")
    assert row["code"] == "ES=F" and row["region"] == "YF" and row["name"] == "E-Mini S&P 500" and row["provider"] == "Yahoo Finance" and row["exchange"] == "CME"
    view = _safe(app.extra_get["/api/futures/series"]({"name": "YF_ES"}))
    assert view["code"] == "ES=F" and view["region"] == "YF" and view["label"] == "E-Mini S&P 500" and view["sources"] == {"prices": "downloaded"} and view["stats"]["bars"] == 300
    assert any("rolled by Yahoo" in n for n in view["notes"]) and view["moves"], "the notes say what the series is, and the biggest moves are listed"
    again = run(app, kind="futures", source="yahoo_series", items=["ES=F"], years=3)
    assert again["ok"] == 1 and len([d for d in again["downloads"] if d["name"] == "YF_ES"]) == 1, "downloading again replaces the file"
    for name in ("YF_ES", "YF_LBR"):
        app.data.store.delete("futures", name)


def test_only_symbols_yahoo_uses_for_futures_are_accepted_and_a_contract_has_its_own_name(data_app):  # noqa: F811
    app = data_app
    feeds(app, {YAHOO + "CLZ26.NYM": yahoo_route(90.0, days=120)})
    for bad in ("AAPL", "ES", "../etc/passwd", "ES=X", "^GSPC", ""):
        with pytest.raises(ApiError, match="not a futures symbol|Choose|items"):
            run(app, kind="futures", source="yahoo_series", items=[bad], years=1)
    result = run(app, kind="futures", source="yahoo_series", items=["CLZ26.NYM"], years=1)
    assert result["results"][0]["name"] == "YF_CLZ26_NYM" and "one contract" in json.dumps(_safe(app.extra_get["/api/futures/series"]({"name": "YF_CLZ26_NYM"}))["notes"])
    with pytest.raises(ApiError, match="source must be one of"):
        run(app, kind="futures", source="google", items=["ES=F"])
    app.data.store.delete("futures", "YF_CLZ26_NYM")
    assert pd.Series([1]).sum() == 1
