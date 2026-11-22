"""Massive as the keyed price source of the Backtest tab, through the whole app (catalog, check, run, test) against the stand-in service."""
import json

import pytest

from src.utils.config import load_config
from src.marketdata.client import MassiveClient
from src.marketdata.massive import MassiveProvider
from src.marketdata.service import MarketDataService
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder, massive_source
from tests.app_roots import isolated_root
from tests.fake_massive import KEY, FakeMassive
from tests.test_webapp import _default, _fetch, _run


def _app(tmp_path, key=KEY, **fake_kw):
    config = load_config()
    fake = FakeMassive(page=100000, today=str(__import__("pandas").Timestamp.now().date()), **fake_kw)
    service = MarketDataService(tmp_path, MassiveProvider(MassiveClient(key=key, root=tmp_path, session=fake, sleep=lambda s: None), root=tmp_path))
    builder = UniverseBuilder(config, TickerStore(tmp_path / "yahoo", fetch=_fetch), default_loader=_default, sources={"massive": massive_source(tmp_path, service, tmp_path / "massive")})
    return App(config, builder=builder, root=isolated_root(tmp_path / "root", config)), fake, tmp_path


@pytest.fixture()
def massive(tmp_path):
    return _app(tmp_path)


def test_the_catalog_lists_massive_without_ever_carrying_the_key(massive):
    app, fake, _ = massive
    sources = {s["name"]: s for s in app.catalog()["sources"]}
    assert set(sources) == {"yahoo", "massive"} and sources["massive"]["available"] and sources["massive"]["setup"] == "" and "MASSIVE_API_KEY" in sources["massive"]["note"]
    assert KEY not in json.dumps(app.catalog()) and fake.calls == []


def test_an_unset_key_shows_as_not_available_and_says_how_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    app, fake, _ = _app(tmp_path, key="")
    entry = next(s for s in app.catalog()["sources"] if s["name"] == "massive")
    assert entry["available"] is False and "MASSIVE_API_KEY" in entry["problem"] and entry["setup"] == "key"
    rows = app.check_tickers({"tickers": ["AAA", "NVDA"], "source": "massive"})["tickers"]
    assert rows["AAA"]["ok"] and not rows["NVDA"]["ok"] and "not set up" in rows["NVDA"]["error"]
    with pytest.raises(ApiError, match="Massive is selected but"):
        app.submit({"tickers": ["AAA", "NVDA"], "models": [{"name": "momentum"}], "source": "massive"})
    assert _run(app, tickers=["AAA", "BBB", "CCC"], source="massive")["status"] == "done"             # nothing to download, so the source does not matter
    assert fake.calls == []


def test_the_default_app_offers_both_sources_and_keeps_massive_in_its_own_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    app = App(load_config(), store=TickerStore(tmp_path / "prices", fetch=_fetch), root=tmp_path)
    assert {s["name"] for s in app.catalog()["sources"]} == {"yahoo", "massive"} and app.builder.sources["massive"].store.directory == tmp_path / "data" / "user" / "prices_massive"


def test_an_unknown_source_is_refused_everywhere(massive):
    app = massive[0]
    for odd in (["massive"], 5, {"a": 1}, "bloomberg", "itick"):
        with pytest.raises(ApiError, match="unknown data source"):
            app.check_tickers({"tickers": ["NEWONE"], "source": odd})


def test_checking_a_ticker_spends_no_request_until_you_run_and_unsupported_symbols_are_explained(massive):
    app, fake, _ = massive
    rows = app.check_tickers({"tickers": ["NVDA", "0700.HK", "ES=F", "EURUSD=X", "AAA"], "source": "massive", "start": "2016-01-01"})["tickers"]
    assert fake.calls == []
    assert rows["NVDA"]["ok"] and rows["NVDA"]["pending"] and rows["AAA"]["source"] == "platform"
    assert not rows["0700.HK"]["ok"] and "cannot place" in rows["0700.HK"]["error"]
    assert not rows["ES=F"]["ok"] and "continuous future" in rows["ES=F"]["error"] and "Futures tab" in rows["ES=F"]["error"]
    assert not rows["EURUSD=X"]["ok"] and "forex" in rows["EURUSD=X"]["error"]


def test_downloads_happen_in_the_run_are_cached_apart_from_yahoo_and_are_labelled(massive):
    app, fake, tmp = massive
    first = _run(app, tickers=["NVDA", "MSFT"], source="massive", start="2016-01-01", models=[{"name": "ma_crossover"}])
    assert first["status"] == "done", first
    u = first["result"]["universe"]
    assert u["source"] == "Massive" and u["provider"] == "massive" and u["tickers"] == ["NVDA", "MSFT"]
    assert sorted(p.name for p in (tmp / "massive").glob("*.csv")) == ["MSFT.csv", "NVDA.csv"] and not list((tmp / "yahoo").glob("*.csv"))
    assert json.loads((tmp / "massive" / "NVDA.json").read_text())["provider"] == "massive"
    n = len(fake.calls)
    again = _run(app, tickers=["NVDA", "MSFT"], source="massive", start="2016-01-01", models=[{"name": "ma_crossover"}])
    assert again["status"] == "done" and len(fake.calls) == n                                    # fresh on disk: nothing asked
    assert all(c["headers"]["Authorization"] == f"Bearer {KEY}" for c in fake.calls)
    st = app.data.market.store if hasattr(app, "data") else None
    from src.marketdata.storage import MarketStore
    series = MarketStore(tmp).list_series("massive", "stocks")
    assert {s["symbol"] for s in series} == {"NVDA", "MSFT"} and all(s["adjusted"] == 1 and s["origin"] == "download" for s in series)


def test_a_symbol_massive_does_not_know_fails_alone_with_the_reason(massive):
    app, fake, _ = massive
    view = _run(app, tickers=["NVDA", "NOPE1"], source="massive", start="2016-01-01", models=[{"name": "ma_crossover"}])
    assert view["status"] in ("done", "error")


def test_the_connection_test_reports_without_the_key(massive):
    app, fake, _ = massive
    out = app.test_source({"source": "massive"})
    assert out["ok"] and out["bars"] > 0 and "Massive answered" in out["message"] and KEY not in json.dumps(out) and len(fake.calls) == 1


def test_a_refused_key_stops_the_run_with_massives_words_and_the_key_is_nowhere(tmp_path):
    app, fake, _ = _app(tmp_path, key="WRONGKEY123456")
    out = app.test_source({"source": "massive"})
    assert not out["ok"] and "refused the key" in out["message"] and "WRONGKEY123456" not in json.dumps(out)
    view = _run(app, tickers=["NVDA", "MSFT"], source="massive", models=[{"name": "ma_crossover"}])
    assert view["status"] == "error" and "refused the key" in json.dumps(view) and "WRONGKEY123456" not in json.dumps(view)


def test_a_key_in_a_request_is_ignored(massive):
    app, fake, _ = massive
    foreign = "FOREIGNKEY987654"
    ok = _run(app, tickers=["NVDA", "MSFT"], source="massive", api_key=foreign, models=[{"name": "ma_crossover"}])
    assert ok["status"] == "done" and all(c["headers"]["Authorization"] == f"Bearer {KEY}" for c in fake.calls) and foreign not in json.dumps([ok, app.test_source({"source": "massive", "api_key": foreign})])
