---
title: "Contract lifecycle: rolls, expiry, funding, exercise and assignment"
slug: contract-lifecycle
difficulty: 3
chapter: Platform
prerequisites: [event-driven-engine-and-ledger, futures-and-commodity-curves, option-pricing-and-greeks]
stages: []
files: [src/engine/lifecycle.py, src/instruments/futures.py, src/instruments/crypto.py, src/instruments/options.py, src/ledger/margin.py]
figures: []
tests: [tests/test_lifecycle.py, tests/test_engine.py]
models: []
---

# Contract lifecycle: rolls, expiry, funding, exercise and assignment

## In one sentence

Contracts are born, rolled, funded, margined, exercised and expire, and each of those events is handled once, in the engine, by posting its cash and P&L to the same ledger rather than by strategy code.

## The idea

**Futures.** A chain holds contracts with first and last trading dates; the `RollSpec` says when the designated contract changes (calendar days before expiry, or when the next contract's volume or open interest overtakes). The engine rolls positions in managed chains by trading out of the old contract and into the new one, tagging both trades `roll` so the cost of rolling is measurable. A strategy that targets the chain id never names a contract. A contract not rolled expires: cash settlement at the final price (or an explicit refusal for physical delivery), and it can no longer be held.

**Perpetuals.** Every funding interval (typically 8 hours) a position pays or receives `-quantity x multiplier x mark x rate` (rate capped by the venue); longs pay when the rate is positive. Isolated margin liquidates when the mark reaches the liquidation price; cross margin draws on the whole account; inverse contracts margin and settle in the coin.

**Options.** At expiry a long option in the money by more than the auto-exercise threshold is exercised: a physical call buys the underlying at the strike, a put sells it, an index option pays cash, an option on a future creates a futures position at the strike. The short side is assigned symmetrically. American options can be exercised earlier on request, and an optional rule assigns short options early when their time value is gone. The gain of exercising is booked as lifecycle settlement so the premium, the gain and the delivered position can be told apart.

**Swaps and forwards.** Coupons and settlements are cashflows on dates from the schedule; the position is carried at the present value of the remaining flows (see [interest rate swaps](interest-rate-swaps.md)).

## Why it matters

These events are where mixed-asset books lose track: an expired contract still "held" at its last price, a roll that books a gain, an assignment that delivers shares at the wrong price. The ledger's quantity book and expired-contract check turn each of those into a failed reconciliation.

## How this repo uses it

`src/engine/lifecycle.py` schedules events when the engine starts (and when a swap is traded mid-run), dispatches them in the engine's event order and notifies strategies through `on_instrument_event` with kinds `roll`, `expiry`, `exercise`, `assignment`, `funding`, `margin_call`, `liquidation` and `rejected`.

## What we found

The tests compare exercise and assignment with analytic P&L: a long in-the-money call ends long 100 shares at the strike with the premium sunk, a short put assigned loses its intrinsic value less the premium, a cash-settled index option pays exactly the intrinsic value, an option on a future delivers a position at the strike, and a European option cannot be exercised early while an American one can. Funding paid equals the formula event by event; an isolated short squeezed past its liquidation price is liquidated; an inverse perpetual earns in the coin.

## Pitfalls

- Early assignment is a modelling choice (off by default); real assignment is not predictable.
- Partial exercise of a position is not modelled: an exercise request applies to the whole holding.
- Physical delivery of commodity futures is not simulated; unrolled physical futures are cash-settled (and counted) or refused by configuration.
- Funding needs a published rate series; a missing interval is reported as a diagnostic, never filled in.

## Try it

```python
from src.instruments import build_chain, crypto_perp, make_option

chain = build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), multiplier=50.0)
print(chain.roll_schedule().head(3))
print(crypto_perp("BINANCE", "BTC", "USDT").funding_payment(2.0, 30000.0, 0.0001))
print(make_option("SPY", "2024-06-21", "call", 100.0).exercise_settlement(1.0, 110.0))
```
