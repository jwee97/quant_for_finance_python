# Data health and provider capabilities

## Provider capability registry (`src/marketdata/capability_registry.py`, `GET /api/health/capabilities`)

Every (provider, dataset) row has exactly one status. The status is derived from evidence in the checkout, never written by hand:

| Status | Evidence required |
|---|---|
| `verified_live` | The generated exchange matrix (`scripts/verify_exchanges.py`, read-only public requests) or this machine's Massive entitlement probe recorded a live answer; the date is stored. Older than 30 days is flagged stale. |
| `verified_recorded` | A real response recorded earlier is committed under `tests/fixtures` and the adapter parses it. Shows the parser works on a real shape, not that the provider still answers. |
| `mock_tested` | A test file mentions the endpoint but the tests use responses they wrote themselves. Credit for a Massive endpoint is by path, and a path shared by two datasets is credited to both (the row says so). |
| `subscription_blocked` | The provider refused the dataset for this key (Massive entitlement probe). |
| `unsupported` | No public endpoint, or the platform has no adapter. |
| `not_tested` | Nothing has exercised it, or the live request was refused from the network it ran on (HTTP 403/451): nothing is inferred. |

State at the time of writing (see the page for the current numbers): **no Massive dataset is verified live** (this machine holds no `MASSIVE_API_KEY`, so the entitlement probe was never run); Massive rows are `mock_tested` or `not_tested`. The OKX and Deribit rows are `verified_live` from the run of 2026-11-25. Bybit and Binance open interest were refused from that network and are `not_tested`. EDINET, OpenDART, CVM and Finzdata are `mock_tested`. A Linux run says nothing about Windows.

## Data Health page (`Data health` in the sidebar)

* **Research readiness**: each series is `ready`, `ready with caveats` or `blocked`, with the reasons (default requirement: 250 observations, 95% of expected periods, no gap beyond 5, no error finding). The readiness of the active research session's instruments is shown too; an instrument the platform does not hold is blocked, not skipped.
* **Dataset coverage**: first and last observation, observations against expected on the series' own calendar (every day for crypto, business days otherwise, or the panel's own dates for the platform snapshot, where a price carried over a closed market is not counted as an observation), missing runs.
* **Data integrity**: duplicates, unsorted dates, non-positive prices, invalid OHLC, stale runs, large moves, long gaps, stale series, observations after a contract's expiry, crossed or locked or stale quotes, unmapped or ambiguous provider symbols, figures available before their period ended, split records against price jumps. Nothing is deleted or repaired.
* **Controls**: *Validate now* (a job; keeps a snapshot with the data's fingerprint so a later run shows whether the data changed), *Refresh from provider* (market-store series, needs the provider's key; the platform snapshot is versioned with the repository and says so), *Retry* after a failed run, *Missing periods* and *Source* per series, *Export findings (CSV)*, and *Record a correction* (the raw value is kept in an append-only log; stored data are never rewritten).

## What is not covered

The downloads of the Options, Futures and Crypto labs are listed on the Data management page but are not run through these checks; fundamentals have their own data-health view; real-time provider-error categories are the existing `MarketDataError.kind`. Intraday market-store series are not checked for gaps.

## Futures (`src/assets/futures_validation.py`)

Contract specifications are validated (positive multiplier and tick, currency, expiry after listing, no duplicate expiries, maintenance margin not above initial, prices on the tick grid). `executability` reports how far a continuous series' levels are from traded prices (ratio adjustment keeps levels positive, difference adjustment can reach zero or below). `roll_table` shows each roll's calendar spread, the roll yield it implies and the execution cost: the spread is not a profit or loss of the roll. One defect was found and fixed: the returns of a difference-adjusted continuous series were a percentage of the shifted level; they are now the return of the held contract (the existing test that pinned the old figure was changed on purpose).
