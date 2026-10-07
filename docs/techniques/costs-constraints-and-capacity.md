---
title: "Costs, constraints, cost-aware optimisation and capacity for mixed portfolios"
slug: costs-constraints-and-capacity
difficulty: 3
chapter: Platform
prerequisites: [event-driven-engine-and-ledger, execution-and-impact, risk-managed-exposure]
stages: []
files: [src/engine/costs.py, src/engine/constraints.py, src/engine/optimise.py, src/engine/risk.py, src/ledger/margin.py]
figures: []
tests: [tests/test_engine_risk.py]
models: []
---

# Costs, constraints, cost-aware optimisation and capacity for mixed portfolios

## In one sentence

Costs, financing, limits and risk measures are configurable models of the engine, not afterthoughts in a strategy: a trade's cost is a fixed fee plus the spread plus market impact, the portfolio is held inside gross, net, margin, liquidity and concentration limits, and the optimiser and capacity estimate use the same cost models.

## The idea

**Costs.** `cost = fixed fee + spread cost + impact cost (+ slippage)`. Commission is per order, per contract and in basis points with a minimum (exchange fees are another commission model); the spread is the observed half spread when the data has a quote and an explicit assumption when it does not; impact follows the square-root law `Y sigma sqrt(Q / ADV)`; models are selected per instrument, type or asset class, so FX, futures, crypto and options each carry their own. Carrying costs are separate: interest on cash, borrow fees on shorts, perpetual funding, margin interest, futures roll costs (the fees and spread on trades tagged `roll`) and FX conversion on physical settlement. A participation cap makes a large order fill over several events.

**Constraints.** `ConstraintSet` limits per-instrument weight, asset-class gross, net currency exposure, factor bands, gross and net leverage, concentration, liquidity (share of average daily traded notional), a volatility target, drawdown scaling, margin utilisation and turnover. Cutting a net limit scales only the side that breaches it, so a hedge is never removed to satisfy it.

**Cost-aware optimisation.** Maximise `alpha'w - (lambda/2) w'Sigma w - c|w - w0| - k|w - w0|^1.5`. The proportional cost creates a no-trade region (do not move a weight unless the marginal alpha net of risk beats the cost), the power-1.5 term (impact) makes large trades disproportionately costly. `calibrate_sqrt_impact` fits `Y` to executed trades. `capacity_curve` scales a run's trading: linear costs grow with size, impact with size to the power 1.5, and `capacity_estimate` reports the size at which half the gross edge is gone.

**Risk.** `PortfolioRisk` reports gross and net exposure, Greeks, PV01, volatility, VaR and CVaR by historical simulation (options through delta and gamma, swaps through key-rate PV01), Euler tail contributions, factor exposure, scenarios by full repricing, funding and rollover risk.

## Why it matters

Gross performance of a multi-asset book is mostly a statement about which costs were left out. A futures roll, a perpetual's funding and an option's bid-ask spread are each a large share of a carry or volatility strategy's edge.

## How this repo uses it

`CostSchedule`, `FinancingModel`, `LiquidityModel` and `ConstraintSet` are arguments of the engine; `PortfolioRisk` is its risk model and feeds `on_risk_update`; `BacktestResult.cost_summary()` reports spread, impact, slippage, fees, roll costs and financing and funding; `capacity_curve` and `cost_aware_weights` take the same schedule.

## What we found

The tests check the cost identity term by term, that observed quotes beat assumed spreads and the most specific model wins, each constraint on constructed inputs (and that an infeasible concentration limit is left alone rather than driving everything to zero), that costs above any marginal gain freeze the portfolio, that impact makes the optimiser trade less than proportional costs alone, that the impact fit recovers a known coefficient, that capacity falls with size and that scenario P&L of a linear future equals delta times shock. One of these tests found a real bug: the USD-strength scenario had the wrong sign for pairs quoted in dollars; it is fixed.

## Pitfalls

- Constraints are applied per strategy, so two strategies each within limits can together exceed them.
- Capacity needs real ADV and volatility; the synthetic data have no volume, so the inputs are assumptions.
- A calibrated impact model is only as good as the trade data and extrapolates badly beyond the sizes it saw.
- VaR from a short history understates tail risk; the scenarios are the check.

## Try it

```python
import numpy as np
import pandas as pd
from src.engine import cost_aware_weights

ids = list("ABC")
alpha = pd.Series([0.002, -0.001, 0.003], ids)
cov = pd.DataFrame(np.diag([0.0004, 0.0004, 0.0009]), ids, ids)
now = pd.Series([0.3, 0.0, 0.0], ids)
for c in (0.0, 0.002, 0.02):
    print(c, cost_aware_weights(alpha, cov, now, cost=c)["trade"].round(3).tolist())
```
