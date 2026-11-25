# Exchange capability matrix

Generated 2026-11-25T14:26:32.231217Z by `scripts/verify_exchanges.py` from read-only public requests. **Only a live answer sets `verified`.** A mocked test proves a parser, not that an exchange still answers.

| Venue | Feature | Status | Tested (UTC) | Detail | Adapter |
|---|---|---|---|---|---|
| OKX | funding_history | **verified** | 2026-11-25T14:26:25.876019Z | ok (settlements every 8 hours, newest first, paged by `after`) | free_data.crypto.okx_funding_daily |
| OKX | open_interest | **verified** | 2026-11-25T14:26:26.406428Z | ok (rows are [time ms, open interest USD, volume USD], aggregated over the currency's contracts, newest first) | free_data.crypto.okx_open_interest |
| OKX | open_interest_instrument | **not_implemented** | 2026-11-25T14:26:26.652904Z | ok (one instrument now; the adapter uses the currency-level history instead) |  |
| OKX | liquidations | **verified** | 2026-11-25T14:26:26.882779Z | ok (actual filled liquidation orders, a short recent window only (sz in contracts, bkPx the bankruptcy price)) | free_data.crypto.fetch_liquidations |
| OKX | contract_specs | **verified** | 2026-11-25T14:26:27.176906Z | ok (contract value, multiplier, linear/inverse type and settlement currency) | free_data.crypto._contract_value |
| OKX | margin_tiers | **verified** | 2026-11-25T14:26:27.426922Z | ok (official risk tiers by position size) | assets.margin_schedules.fetch_okx |
| OKX | order_book | **verified** | 2026-11-25T14:26:27.683733Z | ok (levels [price, size, 0, orders]; swap sizes are contracts) | free_data.crypto.okx_book |
| DERIBIT | funding_history | **verified** | 2026-11-25T14:26:27.953323Z | ok (hourly records; interest_8h is the 8-hour funding rate) | data.crypto (platform history) |
| DERIBIT | dated_futures | **verified** | 2026-11-25T14:26:28.196842Z | ok (open interest is in USD notional (inverse contracts); the index is estimated_delivery_price) | free_data.crypto.fetch_deribit_futures |
| DERIBIT | contract_specs | **not_implemented** | 2026-11-25T14:26:28.366500Z | ok (contract size, settlement currency, expiry, maximum leverage, commissions; the maintenance-margin schedule is NOT in the API) |  |
| DERIBIT | liquidations | **verified** | 2026-11-25T14:26:30.216586Z | ok (public trades; those with a `liquidation` flag are liquidations (partial window; the flag is absent when none occurred)) | free_data.crypto.fetch_deribit_liquidations |
| DERIBIT | order_book | **verified** | 2026-11-25T14:26:30.711749Z | ok (carries change_id, usable for sequence checks on incremental feeds) | free_data.crypto.deribit_book |
| COINBASE | order_book | **verified** | 2026-11-25T14:26:30.893665Z | ok (spot only; carries a sequence number) | free_data.crypto.coinbase_book |
| KRAKEN | order_book | **verified** | 2026-11-25T14:26:31.056956Z | ok (spot only) | free_data.crypto.kraken_book |
| KRAKEN | futures_tickers | **not_implemented** | 2026-11-25T14:26:31.238834Z | ok (perpetuals and dated futures with open interest and funding; reachable, adapter not implemented) |  |
| BITSTAMP | order_book | **verified** | 2026-11-25T14:26:31.480782Z | ok (spot only) | free_data.crypto.bitstamp_book |
| BYBIT | open_interest | **blocked** | 2026-11-25T14:26:32.068325Z | api.bybit.com refused the request (HTTP 403) (daily open interest in coins) | free_data.crypto.bybit_open_interest |
| BYBIT | margin_tiers | **blocked** | 2026-11-25T14:26:32.140765Z | api.bybit.com refused the request (HTTP 403) (not asked again: it refused a moment ago) (public risk-limit tiers) |  |
| BINANCE | open_interest | **blocked** | 2026-11-25T14:26:32.140844Z | fapi.binance.com does not serve this location (HTTP 451) (about 30 days of daily history) | free_data.crypto.binance_open_interest |
| COINBASE | derivatives | **unsupported** |  | the Coinbase Exchange public API in use is spot only (the Coinbase Exchange public API in use is spot only) |  |
| BITSTAMP | derivatives | **unsupported** |  | Bitstamp lists no derivatives (Bitstamp lists no derivatives) |  |
| KRAKEN | spot_derivatives_data | **unsupported** |  | the spot API carries no derivatives data (Kraken Futures is a separate API) (the spot API carries no derivatives data (Kraken Futures is a separate API)) |  |
| BINANCE | margin_tiers | **unsupported** |  | Binance's leverage brackets need a signed (authenticated) request; no public endpoint exists, so none is used (Binance's leverage brackets need a signed (authenticated) request; no public endpoint exists, so none is used) |  |

Legend: `verified` answered live with the fields the adapter reads; `blocked` refused from the network this was run on (403/451): unverified; `failed` answered with an error or an unexpected schema; `unsupported` no public endpoint; `not_implemented` reachable, no adapter yet
