---
title: "Paper trading, deterministic replay and the broker interface"
slug: paper-trading-and-replay
difficulty: 3
chapter: Platform
prerequisites: [event-driven-engine-and-ledger, point-in-time-market-data]
stages: []
files: [src/engine/broker.py, src/engine/engine.py, src/marketdata/store.py, src/marketdata/loaders.py, src/engine/demo.py, src/engine/synthetic.py]
figures: []
tests: [tests/test_engine.py, tests/test_engine_risk.py, tests/test_marketdata.py]
models: []
---

# Paper trading, deterministic replay and the broker interface

## In one sentence

A paper-trading session is the backtest engine fed a stream instead of a finished data set, so research code runs unchanged and a session given the same events as a backtest produces the same digest.

## The idea

A backtest hands the engine all its data at once. A live or paper session hands it data as it arrives, and the engine may only act on a clock that its data have reached. The engine is therefore split into `begin()`, `advance(until)` and `finish()`; `PaperSession` wraps it, adds `feed(events)` (appending to the point-in-time store, which rejects anything that would become available earlier than data already consumed) and exposes a `PaperBroker`. The contract is: add every event available up to `until`, then advance to `until`.

Feeds produce `(until, events)` batches: `ReplayFeed` for stored data in delivery order, `PollingFeed` over a REST polling adapter, `StreamFeed` over a WebSocket or any stream adapter. `run_session` pumps a feed through a session. The `Broker` interface (submit, cancel, open orders, positions, balances, fills) is implemented by `PaperBroker` on the simulated venue; `LiveBroker` is the abstract adapter a real venue needs and refuses to run, because a real adapter needs credentials and a sandbox that cannot be exercised offline. `reconcile_positions` compares the internal ledger with a venue's positions, the check a live deployment runs every cycle.

**Determinism.** Engine events are ordered by `(time, priority, sequence)`, the data are delivered at their availability times, the fill simulator is a pure function, and the `EventLog` hashes everything it did. Two runs of the same inputs have the same digest; a changed price or a changed rule changes it.

## Why it matters

Between a good backtest and a live strategy sit a rewrite and a dozen new bugs. Running the same engine and strategy code against a stream removes the rewrite and lets you compare paper results with the backtest of the same period.

## How this repo uses it

`run_mixed_asset_demo` builds the synthetic multi-asset market, a regime ensemble and a swap strategy and runs them through one engine; its report shows P&L by asset class and category, costs, diagnostics, risk and scenarios. The synthetic market (`synthetic_multi_asset_market`) has a known generating process for futures, FX, crypto, options and rates, so every component can be tested without data licences.

## What we found

The tests show a paper session fed day by day reproduces the backtest's digest and final equity exactly, that a manual order submitted through the paper broker fills at the next event and shows in positions and fills, that the position reconciliation reports exactly the instruments that disagree, and that the demo is deterministic and reconciles with positions in FX, futures, crypto, options and swaps.

## Pitfalls

- Streaming sources stamp arrival time; if the vendor's own timestamp is later than your arrival it is clock skew, not data from the future.
- A paper fill is a simulation: no queue position, no rejections, no outages.
- The session needs explicit session days (default: weekdays) because tomorrow's data days are not known today.
- There are no real broker adapters in this repository: connecting one needs credentials, sandbox accounts and its own error handling.

## Try it

```python
from src.engine import EngineConfig, PaperSession, ReplayFeed, run_session
from src.engine.strategies import TrendStrategy
from src.engine.synthetic import synthetic_multi_asset_market

m = synthetic_multi_asset_market(n_days=90, with_options=False)
cfg = EngineConfig(**m.config_kwargs())
session = PaperSession(m.registry, [TrendStrategy(["ES", "EURUSD"])], cfg)
run_session(session, ReplayFeed(m.events, cfg.start, cfg.end, "1D"))
res = session.result()
print(res.reconciliation.ok, session.broker.positions())
```
