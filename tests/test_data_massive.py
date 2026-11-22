"""The Free data endpoints with Massive as the source: futures curves, crypto spot history, option underlyings' prices, current and rebuilt chains, search, status, probe, sync and the legacy import."""
import json

import pandas as pd
import pytest

from src.marketdata.client import MassiveClient
from src.marketdata.massive import MassiveProvider
from src.webapp.api import ApiError
from tests.fake_massive import KEY, FakeMassive
from tests.test_webapp_assets_ml import _app, finished
from tests.test_webapp_data import _safe
from tests.test_webapp_labs import _rich

TODAY = str(pd.Timestamp.now().date())


@pytest.fixture
def mapp(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    a = _app(tmp_path, _rich, root=tmp_path)
    fake = FakeMassive(page=100000, today=TODAY)
    a.data.market.provider.client = MassiveClient(key=KEY, root=tmp_path, session=fake, sleep=lambda s: None)
    return a, fake, tmp_path


def run(app, **body):
    return finished(app, app.extra_post["/api/data/fetch"](body))


def test_a_futures_curve_is_built_from_contract_bars_and_says_what_it_lacks(mapp):
    app, fake, _ = mapp
    out = run(app, kind="futures", source="massive", items=["CL"], years=1)
    assert out["ok"] == 1 and out["results"][0]["ok"], out
    rec = next(d for d in out["downloads"] if d["name"] == "CL")
    assert rec["source"] == "Massive" and rec["contracts"] >= 3 and any("Open interest is not available" in n for n in rec["notes"])
    curve = _safe(app.extra_get["/api/futures/curve"]({"root": "CL"}))
    assert curve["provenance"]["kind"] in ("historical_observed", "observed_delayed", "mixed", "synthetic") and curve["root"] == "CL"
    assert all(c["headers"]["Authorization"] == f"Bearer {KEY}" for c in fake.calls)


def test_a_repeat_futures_download_asks_only_for_the_newest_sessions(mapp):
    app, fake, _ = mapp
    run(app, kind="futures", source="massive", items=["CL"], years=1)
    n = len([c for c in fake.calls if c["path"].startswith("/futures/v1/aggs")])
    run(app, kind="futures", source="massive", items=["CL"], years=1)
    again = len([c for c in fake.calls if c["path"].startswith("/futures/v1/aggs")]) - n
    assert again == 0                                               # fresh and complete: the stored bars are enough


def test_crypto_history_is_spot_and_labelled_so(mapp):
    app, fake, _ = mapp
    out = run(app, kind="crypto", source="massive", items=["SOL"], what=["history"], days=400)
    assert out["ok"] == 1, out
    rec = next(d for d in out["downloads"] if d["name"] == "SOL")
    assert "SPOT" in " ".join(rec["notes"]) and rec["source"] == "Massive (spot)"
    assert any(c["path"].startswith("/v2/aggs/ticker/X:SOLUSD/") for c in fake.calls)
    with pytest.raises(ApiError, match="spot history"):
        app.extra_post["/api/data/fetch"]({"kind": "crypto", "source": "massive", "items": ["SOL"], "what": ["funding"]})


def test_option_underlying_prices_and_current_chain_from_massive(mapp):
    app, fake, _ = mapp
    out = run(app, kind="options", items=["SPY"], what=["prices"], prices_source="massive", years=1)
    assert out["ok"] == 1 and any(d["name"] == "SPY_prices" for d in out["downloads"])
    chain = run(app, kind="options", items=["SPY"], what=["chain"], chain_source="massive", max_dte=90)
    rec = next(d for d in chain["downloads"] if d["name"] == "SPY")
    assert rec["source"] == "Massive" and any("provider and calculated values are never mixed" in n.lower() for n in rec["notes"]), rec["notes"]


def test_a_past_chain_is_rebuilt_from_closes_and_labelled_as_such(mapp):
    app, fake, _ = mapp
    out = run(app, kind="options", items=["SPY"], what=["chain"], chain_source="massive_past", chain_date="2024-06-12", max_dte=90)
    rec = next(d for d in out["downloads"] if d["name"].startswith("SPY_2024"))
    assert rec["source"].startswith("Massive (rebuilt") and rec["as_of"] == "2024-06-12" and any("daily CLOSE" in n or "daily close" in n.lower() for n in rec["notes"])
    with pytest.raises(ApiError, match="past date"):
        app.extra_post["/api/data/fetch"]({"kind": "options", "items": ["SPY"], "chain_source": "massive_past", "chain_date": "2999-01-01"})
    with pytest.raises(ApiError, match="YYYY-MM-DD"):
        app.extra_post["/api/data/fetch"]({"kind": "options", "items": ["SPY"], "chain_source": "massive_past", "chain_date": "soon"})


def test_without_a_key_massive_downloads_are_refused_before_any_request(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    a = _app(tmp_path, _rich, root=tmp_path)
    fake = FakeMassive()
    a.data.market.provider.client = MassiveClient(key="", root=tmp_path, session=fake)
    for body in ({"kind": "futures", "source": "massive", "items": ["CL"]}, {"kind": "crypto", "source": "massive", "items": ["SOL"], "what": ["history"]},
                 {"kind": "options", "items": ["SPY"], "chain_source": "massive"}):
        with pytest.raises(ApiError, match="MASSIVE_API_KEY"):
            a.extra_post["/api/data/fetch"](body)
    assert fake.calls == []
    st = _safe(a.extra_get["/api/data/status"]({}))
    assert st["massive"]["configured"] is False and "MASSIVE_API_KEY" in st["massive"]["problem"]


def test_a_plan_that_excludes_a_dataset_is_reported_in_massives_words_and_remembered(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    a = _app(tmp_path, _rich, root=tmp_path)
    fake = FakeMassive(page=100000, today=TODAY, deny=["/futures/v1/aggs"])
    a.data.market.provider.client = MassiveClient(key=KEY, root=tmp_path, session=fake, sleep=lambda s: None)
    with pytest.raises(Exception, match="does not include|Nothing could be downloaded"):
        run(a, kind="futures", source="massive", items=["CL"], years=1)
    caps = {f"{c['asset_class']}.{c['dataset']}": c for c in _safe(a.extra_get["/api/market/status"]({}))["capabilities"]}
    assert caps["futures.bars"]["status"] == "not_entitled" and caps["futures.contracts"]["status"] == "available"
    n = len(fake.calls)
    with pytest.raises(Exception, match="refused this dataset|Nothing could be downloaded"):
        run(a, kind="futures", source="massive", items=["CL"], years=1)
    assert not [c for c in fake.calls[n:] if c["path"].startswith("/futures/v1/aggs")]               # a known refusal is not asked again


def test_search_uses_massives_products_and_pairs(mapp):
    app, fake, _ = mapp
    f = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "source": "massive", "q": "crude"}))
    assert f["source"] == "massive" and f["results"][0]["item"] == "CL" and f["results"][0]["name"] == "Crude Oil"
    c = _safe(app.extra_get["/api/data/search"]({"kind": "crypto", "q": "bit"}))
    assert c["source"] == "massive" and isinstance(c["results"], list)


def test_market_status_probe_sync_and_migration_endpoints(mapp):
    app, fake, tmp = mapp
    st = _safe(app.extra_get["/api/market/status"]({}))
    assert st["configured"] and st["integrity"]["ok"] and all(c["status"] == "documented" for c in st["capabilities"]) and KEY not in json.dumps(st)
    probe = finished(app, app.extra_post["/api/market/probe"]({"asset_classes": ["stocks", "futures"]}))
    states = {f"{c['asset_class']}.{c['dataset']}": c["status"] for c in probe["capabilities"]}
    assert states["stocks.bars"] == "available" and states["futures.bars"] == "available" and states["options.contracts"] == "documented"      # only what was asked
    sync = finished(app, app.extra_post["/api/market/sync"]({"symbols": ["AAPL"], "asset_class": "stocks", "start": "2024-05-01"}))
    assert sync["results"][0]["ok"] and sync["results"][0]["added"] > 10 and sync["status"]["coverage"][0]["series"] == 1
    (tmp / "data/user/prices_itick").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5), "close": [1, 2, 3, 4, 5.0]}).to_csv(tmp / "data/user/prices_itick/OLD.csv", index=False)
    mig = _safe(app.extra_post["/api/market/migrate"]({}))
    assert mig["counts"]["imported"] == 1 and mig["integrity"]["ok"]
    cov = _safe(app.extra_get["/api/market/status"]({}))["coverage"]
    assert {c["provider"] for c in cov} == {"massive", "itick"}
