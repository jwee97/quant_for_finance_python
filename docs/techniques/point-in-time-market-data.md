---
title: "Point-in-time market data: events, availability timestamps, contracts and loaders"
slug: point-in-time-market-data
difficulty: 2
chapter: Platform
prerequisites: [data-integrity]
stages: []
files: [src/marketdata/events.py, src/marketdata/schema.py, src/marketdata/contracts.py, src/marketdata/store.py, src/marketdata/loaders.py, src/marketdata/adapters.py]
figures: []
tests: [tests/test_marketdata.py]
models: []
---

# Point-in-time market data: events, availability timestamps, contracts and loaders

## In one sentence

All market data enter the platform as one table of normalised events, each carrying both the time something happened and the time a trader could first have known it, and the engine delivers an event only at its availability time.

## The idea

A price stamped 16:00 that the vendor publishes at 17:30 does not exist for a 16:30 decision. A macro number that is later revised exists twice: the first release and the correction. Look-ahead enters a backtest when a loader ignores this. The cure is structural rather than careful: every observation is a row with `timestamp` (observation time) and `available_at` (first knowledge), the point-in-time store answers every query "as of" a decision time and sees only `available_at <= ts`, and a missing `available_at` is an error unless the caller states a lag rule explicitly (the number of rows filled that way is recorded).

The event types cover quotes, trades, bars, settlements, funding rates, open interest, yield curves, fixings, reference values and corporate actions with the same columns. A **data contract** states what a feed promises: time zone, calendar and session, publication lags by event type, how stale a quote may become, what to do about gaps (error, drop or forward-fill with a limit), revision policy (`latest_known` or `first_release`) and a `UniverseHistory` so dead instruments stay in a historical universe. Applying a contract converts to UTC (daylight-saving ambiguity is an error unless shifted), raises availability to the publication lag and reports what it changed.

## Why it matters

A backtest that cannot leak is worth more than one that is carefully checked for leaks. Staleness is the other silent failure of mixed-asset data: a future that stopped trading is still "the last price". Here age is an output of every price query and the engine refuses marks older than `max_mark_age`.

## How this repo uses it

Loaders read CSV, Parquet, Arrow, JSON Lines and SQL (with a column mapping for vendor names), and live-style sources stamp arrival time: `RestPollingAdapter` over `http_fetcher`, `WebSocketAdapter`, `ReplayStream`. Adapters turn price panels, futures tables, FX markets, funding, option chains and order books into events. `PointInTimeStore` keeps an audit of what it returned (`assert_no_lookahead`) and can be extended with new events for paper trading (new rows must be available strictly later than the stored ones).

## What we found

The tests check that a missing availability is refused, that data published after a decision are invisible to it, that first-release and latest-known revisions differ exactly as expected, that every file format round-trips including curve payloads, that REST polling over a local HTTP server and a WebSocket over a local server stamp arrival times and deliver only new events, that daylight-saving gaps are errors, and that stale quotes and gaps are flagged.

## Pitfalls

- A `lag` rule is an assumption; the right fix is the vendor's real publication time.
- Forward-filling a daily bar into a stale price hides the problem the staleness check exists to expose.
- Vendor timestamps are kept for audit and never used for ordering.
- A live feed's `available_at` is arrival time, which can be later than a vendor's claim.

## Try it

```python
import pandas as pd
from src.marketdata import PointInTimeStore, events_from_prices

px = pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2024-01-02", periods=3))
store = PointInTimeStore(events_from_prices(px, "bar", "16:00", lag="2h"))
print(store.latest("AAA", "bar", "2024-01-03 17:00").close, store.latest("AAA", "bar", "2024-01-03 19:00").close)
```
