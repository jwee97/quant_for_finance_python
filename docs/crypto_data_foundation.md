# Crypto data foundation: verified feeds, margin schedules, recorded history and replay

This page describes what the Crypto workspace rests on after the exchange verification, and what it does **not** claim.

## 1. Exchange adapters verified live

`python scripts/verify_exchanges.py` sends read-only public requests (no keys, no signed or trading endpoint) and writes `docs/exchange_capabilities.md` and `docs/generated/exchange_capabilities.json`; sanitized, trimmed copies of the real responses are kept in `tests/fixtures/exchanges/`. Only a live answer sets `verified`; a refused request is `blocked`, and nothing is inferred about it.

Run on **2026-11-25** from the build environment:

| Venue | Verified live (adapter + schema) | Refused / not verified | No public endpoint |
|---|---|---|---|
| OKX | funding history, currency-level open interest (USD, daily), filled liquidation orders (short recent window), contract specifications, risk tiers (`position-tiers`), order book | none | none |
| Deribit | funding history, dated futures (book summary), liquidation-flagged trades (partial window), order book | contract specifications reachable, adapter not implemented | maintenance-margin formula is documentation only, not in the API |
| Coinbase | spot order book (carries a sequence number) | | derivatives |
| Kraken | spot order book | futures tickers reachable, adapter not implemented | spot API has no derivatives data |
| Bitstamp | spot order book | | derivatives |
| Bybit | not verified: **HTTP 403** from this network (open interest, risk limits); the adapter exists and is tested only against mocks | | |
| Binance | not verified: **HTTP 451** (not served from this location); the adapter exists and is tested only against mocks | | leverage brackets need a signed request |

What the live run found and corrected: Deribit's futures summary has no `underlying_price` (the index is `estimated_delivery_price`), and its open interest is USD notional for inverse contracts. OKX's open-interest history is aggregated over all contracts of a currency, in dollars. OKX liquidation records carry `sz` in contracts and `bkPx`, with the side that was forced (a forced `sell` closes a long). Deribit flags liquidations on public trades; in the sampled window none occurred, which is reported as an answer, not an error.

**Mocked tests prove a parser, not that an exchange still answers.** The fixture tests read real captured responses; only the live script proves availability.

## 2. Margin schedules

`assets/margin_schedules.py` stores every retrieved schedule as a new, dated version under `data/user/margin_schedules/<VENUE>/<INSTRUMENT>_<MODE>/`. `ScheduleStore.at(venue, instrument, mode, when)` returns the version in force on a date and `None` before the first version: today's tiers are never back-dated.

* **Verified:** OKX, from its official public endpoint `GET /api/v5/public/position-tiers` (documentation: https://www.okx.com/docs-v5/en/#public-data-rest-api-get-position-tiers), with contract value, type and settlement currency from `public/instruments`. The endpoint states no effective date, so the retrieval time is the effective time. The tier size unit is taken to be contracts per OKX's documentation; this was not independently confirmed. Not in the endpoint: the maintenance deduction, partial-liquidation steps, liquidation fees, the insurance fund.
* **Not verifiable:** Deribit (only `max_leverage` and commissions are in the API; the margin formula is in prose), Bybit (endpoint refused from this network), Binance (signed endpoint only). These use a clearly labelled **hypothetical** model.
* Every liquidation result reports `margin_basis` per position, `tiers_verified`, and `exchange_accurate`, which is **always false**: even with verified tiers the exchange's liquidation engine is not modelled.

## 3. Recorded history

`assets/crypto_store.py` (compressed Parquet, zstd) under `data/user/crypto_history/`:

| Dataset | Path | Columns |
|---|---|---|
| Order books | `books/<VENUE>/<INSTRUMENT>/<YYYY-MM-DD>/part-<recv ms>.parquet` | venue, instrument, kind (snapshot/update), quality (ok/empty/crossed/gap), exchange_ts_ms, recv_ts_ms, sequence, prev_sequence, source, bids_px, bids_qty, asks_px, asks_qty |
| Funding | `funding/<VENUE>/<INSTRUMENT>.parquet` | funding_time_ms, rate, interval_hours, recv_ts_ms, source |
| Open interest | `open_interest/<VENUE>/<KEY>.parquet` | ts_ms, oi, unit, recv_ts_ms, source |
| Dated futures | `futures/<VENUE>/<COIN>.parquet` | obs_ts_ms, instrument, expiry, mark_price, index_price, open_interest (+unit), settlement_currency, contract_size, recv_ts_ms, source |

Depth, sampling interval, retention days and a size cap are configurable (`StoreConfig`, `crypto-record` flags). `python -m src.cli crypto-record --coin BTC --passes N --interval S` samples REST books of six venues, OKX funding settlements, OKX open interest and Deribit's dated futures; the page's Historical replay view has a "record one pass" button.

* **No history before recording began.** `coverage()` reports the first and last row, rows by quality, time gaps and sequence gaps.
* **Reconciliation:** `BookReconciler` applies incremental updates for OKX and Deribit (`prev_sequence` must equal the last sequence) and Coinbase (`sequence` must be last + 1). A gap or a crossed book marks the book **invalid** and later updates are ignored until the next snapshot; missing updates are never reconstructed. This logic is tested on constructed sequences; **no long-running websocket collector is shipped**, so live incremental ingestion is not implemented or verified.

## 4. Replay engine

`assets/crypto_replay.py`, labelled `historical_replay` (never mixed with the hypothetical execution simulator, whose replenishment is assumed).

* **Look-ahead:** a decision sees the latest valid book received at or before its time and no older than the staleness limit; funding for entry uses settlements strictly before the decision; the second leg of a spread meets the first book received after its delay.
* **Cross-exchange:** executable depth on both sides, taker fees, timestamp skew and staleness limits, fixed per-venue inventory (no transfers), partial fills, and unhedged second legs counted and excluded.
* **Funding carry:** spot bought at the asks, perpetual sold at the bids, each recorded settlement paid on the perpetual notional at the recorded mark (missing marks are counted), financing, margin check with a margin-call exit, open positions at the end reported rather than closed.
* **Dated-futures basis:** settled only on a recorded observation within a tolerance after expiry; otherwise reported as open.
* **Execution history:** a market order's cost against every recorded book (a distribution, not a path).
* Each result reports gross P&L, fees, funding, financing, slippage, net P&L, maximum drawdown, turnover, exposure, coverage and skipped decisions by reason, and a transaction-level audit trail.

Limits: REST snapshots only (fills between snapshots are not observed); no borrowing, withdrawal or custody costs; no transfers between venues; Deribit inverse-contract convexity is not modelled.

## 5. Navigation

Five workflows, each with internal views (`workflows` in `crypto.js`, generic support in `renderLab`): **Market Overview** (Market, Coin deep dive), **Derivatives Intelligence** (one view: funding intelligence, funding by venue, open interest, liquidations, dated futures), **Liquidity & Arbitrage** (Carry and basis, Order-book depth, Across exchanges, Execution simulator, Historical replay), **Strategy Laboratory** (Strategy validation, Signals, Legacy weekday backtest), **Portfolio Risk** (Liquidation risk with the verified-schedule refresh). The old ids `funding` and `basis` redirect to Derivatives Intelligence and Carry and basis (`aliases`); no analytic was removed.
