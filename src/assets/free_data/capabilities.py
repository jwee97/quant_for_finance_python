"""What each exchange adapter can actually do, established by read-only live requests.

``PROBES`` lists, per venue and feature, one public GET (never authenticated, never a trading endpoint), the schema the adapter relies on, and the adapter that uses it. ``run`` performs them and returns a matrix:

``verified``       the request answered and the response has the fields the adapter reads (tested on the date stored with it)
``blocked``        the exchange or the network refused this location (HTTP 403, 451): NOT verified, and nothing is inferred about the endpoint
``failed``         the request answered with an error or a response without the expected fields
``unsupported``    the exchange documents no such public endpoint (a fixed statement, not a probe)
``not_implemented`` the exchange offers it and the endpoint answered, but the platform has no adapter for it yet

A mocked test proves the parser, never that the exchange still answers: ``verified`` is only ever set by a live answer.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .http import FetchError, Http

OKX = "https://www.okx.com/api/v5"
DERIBIT = "https://www.deribit.com/api/v2/public"


def _has(obj, *path):
    for p in path:
        if isinstance(obj, list):
            obj = obj[0] if obj else None
        if obj is None or not isinstance(obj, dict) or p not in obj:
            return False
        obj = obj[p]
    return True


def _okx_rows(fields):
    def check(d):
        rows = d.get("data") if isinstance(d, dict) else None
        if not isinstance(d, dict) or str(d.get("code")) != "0" or not rows:
            return False, "empty or error answer"
        first = rows[0]
        missing = [f for f in fields if (f not in first if isinstance(first, dict) else not isinstance(first, list))]
        return (not missing), ("missing fields " + ", ".join(missing) if missing else "ok")
    return check


def _result_rows(fields, key=None):
    def check(d):
        res = d.get("result") if isinstance(d, dict) else None
        if key and isinstance(res, dict):
            res = res.get(key)
        if not res:
            return False, "empty answer"
        first = res[0] if isinstance(res, list) else res
        missing = [f for f in fields if f not in first]
        return (not missing), ("missing fields " + ", ".join(missing) if missing else "ok")
    return check


PROBES = [
    # (venue, feature, url, params, validator, adapter, note)
    ("OKX", "funding_history", f"{OKX}/public/funding-rate-history", {"instId": "BTC-USDT-SWAP", "limit": 3}, _okx_rows(["fundingTime", "fundingRate"]), "free_data.crypto.okx_funding_daily", "settlements every 8 hours, newest first, paged by `after`"),
    ("OKX", "open_interest", f"{OKX}/rubik/stat/contracts/open-interest-volume", {"ccy": "BTC", "period": "1D"}, _okx_rows([]), "free_data.crypto.okx_open_interest", "rows are [time ms, open interest USD, volume USD], aggregated over the currency's contracts, newest first"),
    ("OKX", "open_interest_instrument", f"{OKX}/public/open-interest", {"instType": "SWAP", "instId": "BTC-USDT-SWAP"}, _okx_rows(["oi", "oiCcy", "oiUsd", "ts"]), None, "one instrument now; the adapter uses the currency-level history instead"),
    ("OKX", "liquidations", f"{OKX}/public/liquidation-orders", {"instType": "SWAP", "uly": "BTC-USDT", "state": "filled"}, _okx_rows(["details"]), "free_data.crypto.fetch_liquidations", "actual filled liquidation orders, a short recent window only (sz in contracts, bkPx the bankruptcy price)"),
    ("OKX", "contract_specs", f"{OKX}/public/instruments", {"instType": "SWAP", "instId": "BTC-USDT-SWAP"}, _okx_rows(["ctVal", "ctMult", "ctType", "lever", "settleCcy"]), "free_data.crypto._contract_value", "contract value, multiplier, linear/inverse type and settlement currency"),
    ("OKX", "margin_tiers", f"{OKX}/public/position-tiers", {"instType": "SWAP", "tdMode": "cross", "instFamily": "BTC-USDT"}, _okx_rows(["minSz", "maxSz", "mmr", "imr", "maxLever"]), "assets.margin_schedules.fetch_okx", "official risk tiers by position size"),
    ("OKX", "order_book", f"{OKX}/market/books", {"instId": "BTC-USDT-SWAP", "sz": 5}, _okx_rows(["bids", "asks", "ts"]), "free_data.crypto.okx_book", "levels [price, size, 0, orders]; swap sizes are contracts"),
    ("DERIBIT", "funding_history", f"{DERIBIT}/get_funding_rate_history", {"instrument_name": "BTC-PERPETUAL", "start_timestamp": int(time.time() * 1000) - 3 * 86_400_000, "end_timestamp": int(time.time() * 1000)}, _result_rows(["timestamp", "interest_8h", "index_price"]), "data.crypto (platform history)", "hourly records; interest_8h is the 8-hour funding rate"),
    ("DERIBIT", "dated_futures", f"{DERIBIT}/get_book_summary_by_currency", {"currency": "BTC", "kind": "future"}, _result_rows(["instrument_name", "mark_price", "estimated_delivery_price", "open_interest", "volume_usd", "creation_timestamp"]), "free_data.crypto.fetch_deribit_futures", "open interest is in USD notional (inverse contracts); the index is estimated_delivery_price"),
    ("DERIBIT", "contract_specs", f"{DERIBIT}/get_instruments", {"currency": "BTC", "kind": "future"}, _result_rows(["contract_size", "settlement_currency", "expiration_timestamp", "max_leverage", "taker_commission", "maker_commission", "instrument_type"]), None, "contract size, settlement currency, expiry, maximum leverage, commissions; the maintenance-margin schedule is NOT in the API"),
    ("DERIBIT", "liquidations", f"{DERIBIT}/get_last_trades_by_currency", {"currency": "BTC", "kind": "future", "count": 100}, _result_rows(["timestamp", "price", "amount", "direction"], "trades"), "free_data.crypto.fetch_deribit_liquidations", "public trades; those with a `liquidation` flag are liquidations (partial window; the flag is absent when none occurred)"),
    ("DERIBIT", "order_book", f"{DERIBIT}/get_order_book", {"instrument_name": "BTC-PERPETUAL", "depth": 5}, _result_rows(["bids", "asks", "timestamp", "change_id"]), "free_data.crypto.deribit_book", "carries change_id, usable for sequence checks on incremental feeds"),
    ("COINBASE", "order_book", "https://api.exchange.coinbase.com/products/BTC-USD/book", {"level": 2}, lambda d: (isinstance(d, dict) and "bids" in d and "asks" in d and "sequence" in d, "ok" if isinstance(d, dict) and "sequence" in d else "no sequence"), "free_data.crypto.coinbase_book", "spot only; carries a sequence number"),
    ("KRAKEN", "order_book", "https://api.kraken.com/0/public/Depth", {"pair": "XBTUSD", "count": 5}, lambda d: (isinstance(d, dict) and not d.get("error") and bool(d.get("result")), "ok"), "free_data.crypto.kraken_book", "spot only"),
    ("KRAKEN", "futures_tickers", "https://futures.kraken.com/derivatives/api/v3/tickers", {}, lambda d: (isinstance(d, dict) and bool(d.get("tickers")) and "openInterest" in d["tickers"][0], "ok"), None, "perpetuals and dated futures with open interest and funding; reachable, adapter not implemented"),
    ("BITSTAMP", "order_book", "https://www.bitstamp.net/api/v2/order_book/btcusd/", {}, lambda d: (isinstance(d, dict) and "bids" in d and "timestamp" in d, "ok"), "free_data.crypto.bitstamp_book", "spot only"),
    ("BYBIT", "open_interest", "https://api.bybit.com/v5/market/open-interest", {"category": "linear", "symbol": "BTCUSDT", "intervalTime": "1d", "limit": 5}, lambda d: (isinstance(d, dict) and bool(((d.get("result") or {}).get("list"))), "ok"), "free_data.crypto.bybit_open_interest", "daily open interest in coins"),
    ("BYBIT", "margin_tiers", "https://api.bybit.com/v5/market/risk-limit", {"category": "linear", "symbol": "BTCUSDT"}, lambda d: (isinstance(d, dict) and bool(((d.get("result") or {}).get("list"))), "ok"), None, "public risk-limit tiers"),
    ("BINANCE", "open_interest", "https://fapi.binance.com/futures/data/openInterestHist", {"symbol": "BTCUSDT", "period": "1d", "limit": 5}, lambda d: (isinstance(d, list) and bool(d) and "sumOpenInterestValue" in d[0], "ok"), "free_data.crypto.binance_open_interest", "about 30 days of daily history"),
]
UNSUPPORTED = [("COINBASE", "derivatives", "the Coinbase Exchange public API in use is spot only"), ("BITSTAMP", "derivatives", "Bitstamp lists no derivatives"), ("KRAKEN", "spot_derivatives_data", "the spot API carries no derivatives data (Kraken Futures is a separate API)"),
               ("BINANCE", "margin_tiers", "Binance's leverage brackets need a signed (authenticated) request; no public endpoint exists, so none is used")]


def probe(http: Http, venue, feature, url, params, validator, adapter, note, now=None) -> dict:
    when = (now or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")
    out = {"venue": venue, "feature": feature, "adapter": adapter, "note": note, "tested_at": when, "url": url, "sample": None}
    try:
        d = http.get_json(url, params)
    except FetchError as exc:
        out.update(status="blocked" if (exc.blocked or exc.status in (403, 451)) else "failed", http_status=exc.status, detail=str(exc)[:200])
        return out
    except Exception as exc:                                                                # noqa: BLE001 - a network failure is a result, not a crash
        out.update(status="failed", http_status=None, detail=f"{type(exc).__name__}: {exc}"[:200])
        return out
    ok, detail = validator(d)
    out.update(status=("verified" if adapter else "not_implemented") if ok else "failed", http_status=200, detail=detail, sample=d)
    return out


def run(http: Http | None = None, only: set | None = None) -> dict:
    http = http or Http()
    rows = [probe(http, *p) for p in PROBES if not only or p[0] in only]
    for venue, feature, why in UNSUPPORTED:
        if not only or venue in only:
            rows.append({"venue": venue, "feature": feature, "status": "unsupported", "detail": why, "adapter": None, "tested_at": None, "url": None, "sample": None, "note": why})
    return {"generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "rows": rows, "legend": {"verified": "answered live with the fields the adapter reads", "blocked": "refused from the network this was run on (403/451): unverified", "failed": "answered with an error or an unexpected schema",
            "unsupported": "no public endpoint", "not_implemented": "reachable, no adapter yet"}}
