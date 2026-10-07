---
title: "The event-driven engine and the one ledger"
slug: event-driven-engine-and-ledger
difficulty: 3
chapter: Platform
prerequisites: [unified-instrument-model, point-in-time-market-data, backtest-engine-and-costs]
stages: []
files: [src/engine/engine.py, src/engine/events.py, src/engine/execution.py, src/engine/orders.py, src/ledger/ledger.py, src/ledger/reconcile.py, src/ledger/margin.py]
figures: []
tests: [tests/test_engine.py, tests/test_ledger.py]
models: []
---

# The event-driven engine and the one ledger

## In one sentence

One simulation loop replays market events and the engine's own events in a fixed order, turns orders into fills with the cost models, and books everything in one multi-currency journal whose entries obey an accounting identity that the reconciliation checks.

## The idea

**Event loop.** Market events (delivered at their `available_at`) are interleaved with engine events (strategy decisions, order arrivals, daily marks and accruals, rolls, expiries, funding payments, coupons, margin checks) in a total order `(time, priority, sequence)`, so a replay is deterministic: the same inputs and configuration give the same SHA-256 digest of the event log. An order decided at 16:30 fills at the next tradeable event (`fill_policy="next_event"`) or after a fixed latency, never at the price that triggered it. Market orders pay half the spread (observed from a quote, or an assumed width on a bar), impact and slippage plus a fee; limit orders rest until the market reaches them; stop orders trigger on a trade-through; a participation cap splits large orders over several events.

**Ledger.** Positions and cash in any number of currencies. A fill is booked as principal at the mid plus separate cost entries, so the journal shows what crossing the market cost. Each journal entry satisfies `cash + change in position value + translation = P&L + transfer`, checked as it is written. Marks settle variation margin in cash for futures and perpetuals, carry options and swaps at value, translate foreign balances at the observed pair prices and fail loudly if no rate links a currency to the base.

**Reconciliation.** Cash equals starting cash plus the journal; equity equals starting equity plus P&L plus transfers; every entry satisfies the identity at final rates; each position equals an independent quantity book built from fills and lifecycle events; no expired contract is held.

## Why it matters

Mixed-asset backtests fail by losing money from nowhere or from nowhere's opposite: a roll that creates P&L, a currency that is valued at the wrong rate, a funding payment booked twice. Because the loop is one loop and the books are one book, a single invariant suite covers every asset class.

## How this repo uses it

`Engine(registry, events, strategies, config, costs, financing, margin, liquidity, constraints, risk_model).run()` returns a `BacktestResult` with the equity curve, journal, fills, orders, diagnostics, digest and reconciliation. Attribution cuts the journal by category, instrument, asset class, strategy or tag and always adds up to the change in equity.

## What we found

The tests recompute a futures buy-and-hold through rolls independently from the fills and prices and match the ledger to 1e-6; show that trading at constant prices with no costs cannot change equity; that the costs account for the entire equity decline when prices are flat; that changing future data leaves the past equity curve and fills identical; that a paper session fed the same events reproduces the digest; and that tampering with a balance or a position is caught by the reconciliation.

## Pitfalls

- Attribution to a category is only as informative as the categories: fees, spread, impact and slippage are separate; interest and funding are separate from trading P&L.
- The isolated-margin liquidation check runs at the daily snapshot, not intraday.
- Constraints apply per strategy, not to the combined portfolio of several strategies.
- Daily bars give no spread: the cost of crossing one is an explicit assumption, not an observation.

## Try it

```python
import pandas as pd
from src.engine import Engine, EngineConfig, Schedule, Strategy, Target, CostSchedule
from src.instruments import Instrument, InstrumentRegistry
from src.marketdata import events_from_prices

idx = pd.bdate_range("2024-01-02", periods=30)
px = pd.DataFrame({"AAA": 100.0 + pd.Series(range(30), index=idx) * 0.1})
reg = InstrumentRegistry([Instrument(instrument_id="AAA", asset_class="equity", instrument_type="equity", currency="USD", calendar="WEEKDAY")])


class Buy(Strategy):
    name = "buy"
    schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

    def on_schedule(self, ctx):
        ctx.set_targets([Target("AAA", weight=0.5)])


res = Engine(reg, events_from_prices(px, "bar", "16:00", spread_bps=2.0), [Buy()], EngineConfig(start=idx[0], end=idx[-1]), CostSchedule()).run()
print(res.reconciliation.ok, round(res.summary()["total_pnl"], 2), res.digest[:12])
```
