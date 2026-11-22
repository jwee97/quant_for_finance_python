# Market data: Massive, the store and the legacy iTick import

```
Dashboard -> API (/api/data/*, /api/market/*) -> MarketDataService -> MassiveProvider -> MassiveClient -> api.massive.com
                                                  |
                                                  +-> MarketStore (SQLite, data/user/marketdata/marketdata.db) -> analytics
```

The code is `src/marketdata/`: `client.py` (one HTTP client), `models.py` (typed requests, answers, errors, capabilities), `massive.py` (the adapter), `storage.py` (the store), `service.py` (incremental
synchronisation, gap filling, past option chains, futures chains), `prices.py` (Massive as the Backtest tab's price source) and `legacy.py` (the iTick import). Asset-specific calculations (Greeks, curves,
carry, roll) stay in `src/assets/` and `src/derivatives/`; the adapter knows endpoints and shapes only. `src/assets/free_data/massive.py` turns what the service holds into the downloads the Futures, Options
and Crypto cards save.

## The key and the client

* `MASSIVE_API_KEY` from the environment or a git-ignored `.env`; it is sent only in `Authorization: Bearer`, never in an address (a `next_url` that carries `apiKey` is cleaned first), never to another host,
  never to the page; every message is redacted.
* One `requests.Session` with a connection pool; pagination by `next_url`; exponential backoff with `Retry-After` for HTTP 429, 5xx and dropped connections (`MASSIVE_RETRIES`); an optional limiter
  (`MASSIVE_CALLS_PER_MINUTE`, remembered on disk); cancellation between pages; usage counters per endpoint family (`data/user/massive/usage.json`).
* HTTP 401 is reported as a refused key, 403 as "your plan does not include this data" with Massive's own message, 404 as nothing at that address, 429 as the rate limit.

## Endpoints (Massive's documentation, read 2026-11-22)

| Asset class | Dataset | Endpoint | Plan, as the documentation states |
|---|---|---|---|
| stocks | bars | `/v2/aggs/ticker/{t}/range/{m}/{span}/{from}/{to}` | all Stocks plans; history and recency by plan |
| stocks | tickers, overview | `/v3/reference/tickers`, `/v3/reference/tickers/{t}` | all Stocks plans |
| stocks | splits, dividends | `/stocks/v1/splits`, `/stocks/v1/dividends` (the `/v3/reference` ones are deprecated) | all Stocks plans |
| stocks | trades, quotes | `/v3/trades/{t}`, `/v3/quotes/{t}` | Developer and Advanced; Advanced only |
| options | contracts | `/v3/reference/options/contracts` (`as_of`, `expired`) | all Options plans |
| options | chain snapshot | `/v3/snapshot/options/{underlying}` | Starter, Developer, Advanced |
| options | bars, trades, quotes | `/v2/aggs/...`, `/v3/trades|quotes/O:...` | all plans; Developer and Advanced; Advanced |
| futures | products, contracts | `/futures/v1/products`, `/futures/v1/contracts` | all Futures plans |
| futures | bars | `/futures/v1/aggs/{ticker}` (`resolution=1session`, `window_start`) | all Futures plans |
| futures | trades, snapshot | `/futures/v1/trades/{ticker}`, `/futures/v1/snapshot` | Developer and Advanced; Starter and up |
| crypto | bars, tickers | `/v2/aggs/ticker/X:BTCUSD/...`, `/v3/reference/tickers?market=crypto` | Currencies plans (as the page states) |

Not verified in the documentation, so not enabled: futures quotes; crypto trades and quotes; historical Greeks and implied volatility endpoints; flat files and WebSocket. No futures endpoint read carries open
interest. The stocks, options and crypto bars share one shape; futures use their own (nanosecond `window_start`, `session_end_date`, `settlement_price`).

## Entitlements

`MassiveProvider.get_provider_capabilities()` lists every documented dataset as `documented` until something asked. `probe()` (the card's **Check what my plan includes**, `quant massive-test --probe`) asks
once per dataset with a one-row request and records `available`, `not_entitled` (HTTP 403, with Massive's words), `no_data` or `error` in `data/user/massive/entitlements.json`. A dataset recorded as refused is
not asked again (the error says so and names the plan note) until the probe is run again.

## The store

SQLite, keyed by provider as well as instrument, bar size and adjustment, so Massive's and the legacy iTick bars of one symbol sit side by side and nothing overwrites anything across providers. Tables:
`bars`, `series` (first/last bar, rows, version, status, last synchronisation, endpoint, origin, covered range), `versions` (every change: rows added and revised, reason), `actions` (splits and dividends),
`quality` (the last data-quality findings), `jobs` (download status), `migrations`.

* **Incremental.** A synchronisation asks only for what is missing: newer bars (from a few days before the last one, to catch revisions), earlier history (once: the covered range is recorded, so a plan whose
  history is shorter than asked is not asked again), and gaps found inside the range (runs of three or more missing sessions; single holidays are not gaps). Intraday series are fetched in windows, each
  stored before the next, so a cancelled or crashed download resumes from what is stored.
* **Splits.** A split executed since the last synchronisation re-fetches an adjusted series (the whole history is re-adjusted); the revision is counted in the version log.
* **Quality.** Non-positive prices, high below low, close outside the range, negative volume, one-day moves above 40%, missing closes and gaps.
* **Expired futures contracts** are archived: marked `expired` and not asked again.

## What is rebuilt, and how it is labelled

* **Futures curve** (`futures_curve_history`): the contracts alive between two dates, each with its stored session bars. The price is the exchange settlement price of the session where Massive sends one, else
  the close (`price_source` says which). Open interest is empty. Roll, carry, calendar spreads, continuous series and the contract-level backtests of the Futures tab then work on real contracts.
* **Past option chain** (`reconstruct_option_chain`): the contract list as of the date plus each contract's bar of that day. The price is the daily close (a last trade); there is no bid or ask (`NaN`); contracts
  that did not trade are left out and counted; implied volatility and Greeks are calculated (Black-Scholes at a stated rate and yield) and labelled `calculated`. When saved for the Options tab, the close is
  used as both bid and ask and the file's record says `price_basis: daily_close`.
* **Current chain** (`get_option_chain`): the provider's snapshot; its implied volatility and Greeks are labelled `provider`, a missing one is `missing` (never invented), and the download's notes state how far
  Massive's volatility is from the one calculated here from the same mid.
* **Crypto**: spot only (`X:BTCUSD`). It is never presented as a futures price; Deribit stays the source of dated crypto futures and derivatives data.

## Legacy iTick data

`marketdata/legacy.py` reads `data/user/prices_itick/` and the downloads whose record says `source_id: itick` (without touching them) and stores them as provider `itick`, origin `legacy_import`, with the
original note and `adjusted = false` (iTick never documented adjustment). It runs once at start-up and is harmless to repeat (`quant market-migrate`, `POST /api/market/migrate`; a file is skipped when its path
and checksum were imported). The Futures History view still reads the old `US_NQ`-style files.

## Commands and endpoints

`quant massive-test [--probe]`, `quant market-sync SYMBOL... [--asset-class] [--start] [--force]`, `quant market-migrate`; `GET /api/market/status`, `POST /api/market/probe`, `POST /api/market/sync`,
`POST /api/market/migrate`; the cards use `/api/data/fetch` with `"source": "massive"` (futures, crypto), `"prices_source": "massive"` or `"chain_source": "massive" | "massive_past"` (options).

## What is and is not verified

The tests (`tests/test_marketdata_*.py`, `tests/test_data_massive.py`, `tests/test_webapp_massive.py`) drive everything through `tests/fake_massive.py`, a stand-in that follows the documented shapes. No real
Massive account has been used by the author of this migration: the first request with your key is the real test, and the probe shows what your plan answers.
