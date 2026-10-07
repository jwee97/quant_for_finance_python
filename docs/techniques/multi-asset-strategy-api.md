---
title: "One strategy API for every signal type: trend, carry, basis, relative value, volatility, ensembles and ML"
slug: multi-asset-strategy-api
difficulty: 3
chapter: Platform
prerequisites: [event-driven-engine-and-ledger, cross-asset-carry, option-strategies-and-vol-premium, regime-detection]
stages: []
files: [src/engine/strategy.py, src/engine/strategies/trend.py, src/engine/strategies/carry.py, src/engine/strategies/basis.py, src/engine/strategies/relative_value.py, src/engine/strategies/volatility.py, src/engine/strategies/ensemble.py, src/engine/strategies/ml.py]
figures: []
tests: [tests/test_engine_strategies.py]
models: []
---

# One strategy API for every signal type: trend, carry, basis, relative value, volatility, ensembles and ML

## In one sentence

A strategy emits Signals, Targets or Orders through one interface and sees only point-in-time data, so a trend follower, a cash-and-carry harvester, a variance seller and a regime ensemble differ in their logic and nothing else.

## The idea

The interface has hooks (`on_start`, `on_market`, `on_instrument_event`, `on_portfolio_update`, `on_risk_update`, `on_schedule`, `on_end`) and three layers of output. `generate_signals` returns `Signal(instrument, value, confidence)`; `map_to_targets` turns signals into `Target`s (a quantity, a signed notional or a weight of the strategy's capital); `ctx.set_targets` sends them to the engine, which resolves chains to contracts, rounds to lots, applies the constraints, nets against the current holding and trades the difference. The context exposes only what is known now: prices and histories, universes, curves, funding, option chains, portfolio and risk state, cost estimates and instrument metadata.

The families shipped on this interface: **trend** (momentum, breakout and moving-average signals blended with a confidence equal to their agreement); **carry** (futures roll yield, FX rate differential, perpetual funding, ranked within asset class); **basis** (cash-and-carry, reverse cash-and-carry, premium mean reversion); **relative value** (calendar spreads, butterflies, FX triangles); **volatility** (variance premium, gamma, skew and term structure with vega-budget sizing and delta hedging); **ensembles** (members run unchanged and are blended by a regime detector, a risk overlay, a volatility target and a drawdown brake); **machine learning** (walk-forward ridge or trees pooled across instruments with optional regime-conditioned models, online forecast combination, logistic meta-labelling).

## Why it matters

When strategy code can place orders directly, risk limits and costs are applied by convention, and a mixed book cannot be netted or constrained. When strategies only declare what they want, the engine can net across them, apply one constraint set and one cost model, and report attribution by strategy.

## How this repo uses it

`src/engine/strategies/` holds the families; swap strategies live in `src/swaps/strategies.py`. The ensemble hands each member a context that records targets and refuses direct orders, so no member can bypass the overlay. Learners train on features computed from point-in-time history and labels that have been realised (the last `horizon` observations are never in a training set).

## What we found

The tests construct data where each family has a known answer: trend follows a persistent drift in either direction, carry goes long the high yielder and short the low one, cash-and-carry is delta neutral and earns the funding, stays out when the basis does not pay, the variance seller sells and hedges (median residual delta under a third of gross, and no hedge trades when `hedge=False`), the ensemble tilts weights toward trend in stress, and the forecast combiner moves weight to the forecaster that works. Features are causal, ML signals do not change when the future changes, and strategy runs are deterministic. On the synthetic market the strategies reconcile and trade; their P&L demonstrates the machinery, not a market premium.

## Pitfalls

- Targets are in terms of the strategy's own capital (`capital_share` of the account), not the account's.
- A strategy that omits an instrument from its targets leaves the position unchanged; to exit, target zero.
- Machine-learning strategies re-fit on a schedule: a model fit once on the whole sample is a leak.
- Shadow returns inside the ensemble ignore costs and carry; they size risk, they are not performance.

## Try it

```python
from src.engine import Engine, EngineConfig, CostSchedule
from src.engine.strategies import TrendStrategy
from src.engine.synthetic import synthetic_multi_asset_market

m = synthetic_multi_asset_market(n_days=140, with_options=False)
res = Engine(m.registry, m.events, [TrendStrategy(["ES", "EURUSD", "GBPUSD"])], EngineConfig(**m.config_kwargs()), CostSchedule()).run()
print(res.reconciliation.ok, len(res.fills), round(res.summary(ex_interest=True)["sharpe"], 2))
```
