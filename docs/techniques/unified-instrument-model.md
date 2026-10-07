---
title: "The unified instrument model: FX, futures, crypto, options and swaps as one kind of thing"
slug: unified-instrument-model
difficulty: 2
chapter: Platform
prerequisites: [futures-and-commodity-curves]
stages: []
files: [src/instruments/base.py, src/instruments/calendars.py, src/instruments/futures.py, src/instruments/crypto.py, src/instruments/options.py, src/instruments/rates.py, src/instruments/registry.py]
figures: []
tests: [tests/test_instruments.py]
models: []
---

# The unified instrument model: FX, futures, crypto, options and swaps as one kind of thing

## In one sentence

Every tradable thing in the platform is an immutable `Instrument` with the same core fields (currency, tick and lot size, contract multiplier, calendar, underlying, expiry, settlement and margin type), so one engine and one ledger can hold spot FX, futures, perpetuals, options and swaps in the same portfolio.

## The idea

A strategy should not need to know whether it is trading a share, a forward or a swap, and an engine should not need a branch per asset class. What differs between instruments is a short list of facts: the currency P&L is paid in, how small a price move or a quantity can be (`tick_size`, `lot_size`), how many currency units one price point is worth (`contract_multiplier`), when the contract exists (`expiry`, `calendar`), what it is written on (`underlying_id`) and how collateral is required (`margin_type`). These are the base-class fields; the subclasses (`FXSpot`, `FXForward`, `FXSwap`, `Future` and `FutureChain`, `CryptoSpot`, `CryptoPerp`, `CryptoFuture`, `Option`, `InterestRateSwap`, `BasisSwap`, `CrossCurrencyBasisSwap`) add their own terms such as a perpetual's funding interval, an option's strike and exercise style, a swap's index, tenor, day counts and curves.

The one idea that makes a single ledger possible is the **cash style**. It says what a position is worth and when cash moves: `full_payment` (pay the price, hold an asset: shares), `premium` (options), `variation_margin` (no cash at entry, every mark-to-market change settled in cash: futures and perpetuals), `otc_mtm` (entered near zero value, carried at present value, cash moves at coupons: swaps and forwards) and `currency_exchange` (the balances are the position: spot FX and spot crypto). The ledger's arithmetic follows the cash style and nothing else.

Calendars are part of the model: business days, holiday rules generated for any year (US, TARGET, London, 24x7 for crypto, joint calendars), the ISDA adjustment conventions and trading sessions with time zones.

## Why it matters

Most multi-asset backtests are a pile of special cases: a futures module that knows about rolls, an options module that knows about expiry, each with its own P&L code. The special cases are where cash goes missing. Pushing the differences into data (the specification) and one small enumeration (the cash style) means the same tests of cash conservation apply to every instrument.

## How this repo uses it

`InstrumentRegistry` holds instruments and future chains, rejects a conflicting redefinition, validates that every derivative's underlying exists and round-trips to JSON. `build_chain` creates a futures chain from an expiry rule with margins and a `RollSpec`; `make_option`, `crypto_perp`, `fx_forward`, `make_irs` build the rest with market conventions. Every instrument serialises with `to_dict` and rebuilds with `instrument_from_dict`. The engine, the ledger, the risk model and the strategies all read the same objects.

## What we found

The tests check the calendars against known holidays (Easter, Thanksgiving, observance rules), that the business-day conventions never cross a month end when they should not, that every instrument kind survives a round trip through a dictionary, that quantities round toward zero to whole lots, that perpetual funding has the right sign and cap and inverse contracts pay in the coin, and that option exercise settlement is physical, cash or a futures position as the contract says.

## Pitfalls

- Holiday rules are generated, not a vendor feed: ad-hoc closures are missing unless you pass `extra_holidays`.
- A chain id (`ES`) and a contract id (`ESH24`) are different things; strategies may target a chain and let the roll rule choose the contract.
- Inverse contracts margin and settle in the coin: their P&L is not quote-currency P&L.
- The model describes contracts, not their data; prices come from the market-data layer.

## Try it

```python
from src.instruments import InstrumentRegistry, build_chain, crypto_perp, fx_spot, make_option

reg = InstrumentRegistry([fx_spot("EUR", "USD"), crypto_perp("BINANCE", "BTC", "USDT")])
reg.add_chain(build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), multiplier=50.0, tick_size=0.25))
reg.add(make_option("ES", "2024-06-21", "call", 5000.0, underlying_kind="future"))
print(reg.to_frame()[["asset_class", "type", "currency", "multiplier", "cash_style"]].head(8))
```
