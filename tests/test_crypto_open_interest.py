"""Open interest by exchange and recent liquidations: the fetchers read each venue's documented format, a venue that fails is reported, and the derivatives view uses what was downloaded (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.assets import crypto_research as cr
from src.assets.free_data import crypto as fc
from src.assets.free_data.http import FetchError, Http
from src.assets.free_data.store import DownloadStore
from tests import free_data_helpers as fh
from tests.test_crypto_research_api import app  # noqa: F401
from tests.test_webapp_assets_ml import _safe

DAYS = [d for d in pd.date_range("2026-08-01", periods=40)]


def ms(d) -> str:
    return str(int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000))


def routes(skip=()):
    r = {}
    if "OKX" not in skip:
        r["https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume"] = {"code": "0", "data": [[ms(d), str(1e9 + 1e7 * i), "1"] for i, d in reversed(list(enumerate(DAYS)))]}
    if "BYBIT" not in skip:
        r["https://api.bybit.com/v5/market/open-interest"] = {"retCode": 0, "result": {"list": [{"openInterest": str(50_000 + 100 * i), "timestamp": ms(d)} for i, d in reversed(list(enumerate(DAYS)))]}}
    if "BINANCE" not in skip:
        r["https://fapi.binance.com/futures/data/openInterestHist"] = [{"sumOpenInterest": "1", "sumOpenInterestValue": str(2e9 - 1e7 * i), "timestamp": int(ms(d))} for i, d in enumerate(DAYS)]
    return r


def http(r):
    return Http(fh.FakeFeeds(r), sleep=lambda s: None, pause=0.0, retries=0)


def test_open_interest_is_read_per_venue_in_its_own_unit_and_a_failing_venue_is_reported():
    d = fc.fetch_open_interest(http(routes()), "BTC", 30)
    f = d.frame
    assert list(f.columns) == ["time", "OKX", "BYBIT", "BINANCE"] and len(f) == 30 and f["OKX"].iloc[-1] == pytest.approx(1e9 + 1e7 * 39) and f["BYBIT"].iloc[-1] == 50_000 + 100 * 39
    assert d.meta["units"] == {"OKX": "dollars", "BYBIT": "coins", "BINANCE": "dollars"} and d.name == "openinterest_BTC"
    partial = fc.fetch_open_interest(http(routes(skip=("BINANCE", "BYBIT"))), "BTC", 30)
    assert list(partial.frame.columns) == ["time", "OKX"] and any("Bybit" in n or "Binance" in n for n in partial.meta["notes"])
    with pytest.raises(FetchError, match="no venue gave open interest"):
        fc.fetch_open_interest(http({}), "BTC", 30)


def test_the_open_interest_view_gives_changes_and_percentiles_and_keeps_units_apart():
    d = fc.fetch_open_interest(http(routes()), "BTC", 40)
    out = cr.open_interest_view(d.frame, d.meta)
    okx = next(v for v in out["venues"] if v["venue"] == "OKX")
    assert okx["change_1d"] == pytest.approx((1e9 + 1e7 * 39) / (1e9 + 1e7 * 38) - 1) and okx["change_30d"] is not None and okx["percentile"] > 0.9 and okx["unit"] == "dollars"
    binance = next(v for v in out["venues"] if v["venue"] == "BINANCE")
    assert binance["change_7d"] < 0 and "different units" in out["units_note"]
    assert cr.open_interest_view(pd.DataFrame())["state"] == "unavailable"


def test_liquidations_are_a_short_window_with_the_side_that_was_liquidated():
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    r = {"https://www.okx.com/api/v5/public/liquidation-orders": {"code": "0", "data": [{"details": [{"ts": str(now - 3_600_000), "side": "sell", "posSide": "long", "sz": "100", "bkPx": "50000"}, {"ts": str(now), "side": "buy", "posSide": "short", "sz": "40", "bkPx": "50100"}]}]},
         "https://www.okx.com/api/v5/public/instruments": fh.okx_instrument("0.01")}
    d = fc.fetch_liquidations(http(r), "BTC")
    out = cr.liquidation_view(d.payload)
    assert out["events"] == 2 and out["longs_liquidated"] == 1 and out["shorts_liquidated"] == 1 and out["window_hours"] == pytest.approx(1.0)
    assert out["longs_notional"] == pytest.approx(100 * 0.01 * 50_000) and out["shorts_notional"] == pytest.approx(40 * 0.01 * 50_100) and "not a history" in out["note"]
    with pytest.raises(FetchError, match="no filled liquidations"):
        fc.fetch_liquidations(http({"https://www.okx.com/api/v5/public/liquidation-orders": {"code": "0", "data": []}}), "BTC")
    assert cr.liquidation_view(None)["state"] == "unavailable"


def test_the_derivatives_route_uses_downloaded_open_interest_and_still_lists_dated_futures_as_unavailable(app):  # noqa: F811
    d = fc.fetch_open_interest(http(routes()), "BTC", 40)
    DownloadStore(app.root).save(d)
    out = _safe(app.extra_get["/api/crypto/derivatives"]({"data_mode": "observed", "symbols": "BTC"}))
    assert out["open_interest"]["state"] == "ok" and out["sources"]["open_interest"] == "downloaded" and out["liquidations"]["state"] == "unavailable" and out["dated_futures"]["state"] == "unavailable"


def test_deribit_dated_futures_are_read_and_the_perpetual_is_left_out():
    rows = [{"instrument_name": "BTC-PERPETUAL", "mark_price": 50_010.0, "underlying_price": 50_000.0}, {"instrument_name": "BTC-27MAR26", "mark_price": 51_000.0, "underlying_price": 50_000.0, "open_interest": 5, "volume_usd": 9, "creation_timestamp": 1},
            {"instrument_name": "BTC-26JUN26", "mark_price": 52_000.0, "underlying_price": 50_000.0, "open_interest": 7, "volume_usd": 3, "creation_timestamp": 1}]
    d = fc.fetch_deribit_futures(http({"https://www.deribit.com/api/v2/public/get_book_summary_by_currency": {"result": rows}}), "BTC")
    assert [c["expiry"] for c in d.payload["contracts"]] == ["2026-03-27", "2026-06-26"] and d.name == "futures_BTC"
    with pytest.raises(FetchError):
        fc.fetch_deribit_futures(http({"https://www.deribit.com/api/v2/public/get_book_summary_by_currency": {"result": []}}), "BTC")
