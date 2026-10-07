---
title: "Option strategies, the variance premium and an options backtester"
slug: option-strategies-and-vol-premium
difficulty: 3
chapter: Ch. 18
prerequisites: [volatility-surfaces, backtest-engine-and-costs]
stages: []
files: [src/derivatives/backtest.py, src/derivatives/strategies.py, src/derivatives/volatility.py, src/derivatives/synthetic.py]
figures: []
tests: [tests/test_derivatives.py]
models: []
---

# Option strategies, the variance premium and an options backtester

## In one sentence

Selling options earns a premium most of the time and loses it in crashes; the options backtester trades real or synthetic option chains at the bid and ask with a one-day lag, settles expiries, hedges delta, and attributes the P&L to the Greeks.

## The idea

The **variance risk premium** is the gap between implied variance (what option sellers charge, measured by a VIX-style index) and the variance that is later realised; it is positive on average in equity indices, and `variance_risk_premium` reports it with both trailing and forward realised variance (the forward one is look-ahead and used only to measure, never to trade). Strategies that harvest or time it, all in `src.derivatives.strategies`:

- `short_put`, `covered_call`, `short_strangle`, `iron_condor`: premium selling with different tail protection, strikes chosen by delta;
- `delta_hedged_straddle`: buy or sell an at-the-money straddle and hedge delta daily, so the P&L is approximately `0.5 * gamma * S^2 * (realised variance - implied variance) dt`; `side=-1` sells the variance premium;
- `risk_reversal` (skew), `calendar_spread` (term structure);
- `vrp_timed`: trade only when implied volatility is rich against trailing realised volatility.

`OptionBacktester` enforces the accounting: orders placed on day `t` fill on `t + execution_lag` at that day's quotes, crossing the spread (a `fill_fraction` below one models fills inside it), commissions are charged per contract, hedges trade the underlying at a cost in basis points, expiry settles at intrinsic value, and `pnl_explain` attributes each day's P&L to delta, gamma, vega, theta and residual. **Dispersion** (`simulate_dispersion_world`, `dispersion_backtest`) sells index variance against single-name variance and is profitable when implied correlation exceeds what is realised.

## Why it matters

Most published option backtests use mid prices and ignore settlement, assignment and margin, which is where the premium disappears. The cost of crossing a wide option spread several times a month can exceed the premium of a short-dated strategy.

## How this repo uses it

Strategies are small classes that look at the chain and the current positions and return orders, so a new one is a few lines and runs through the same accounting. The institutional findings run each of them on three simulated years.

## What we found

The tests check that orders fill on the next date at that date's quotes and cross the spread, that expiries settle at intrinsic value, that hedge orders pay their cost, that every strategy runs without skipped orders and satisfies the accounting identity (the sum of daily P&L equals the change in equity), that the Greek attribution explains most of a hedged straddle's P&L, and that long and short straddles are mirror images up to costs. In the **synthetic** market the short variance strategies earn the premium that was built in and lose in the jump periods; that confirms the engine, and is not evidence about real markets, for which you would load a vendor chain with `load_option_csv`.

## Pitfalls

- Selling volatility has a short-gamma payoff: the Sharpe ratio flatters it, and the drawdown tells the truth.
- Mid-price backtests overstate results; this engine fills at the bid and ask.
- Expiry settlement is cash at intrinsic value, European style: early exercise and assignment are not modelled, which matters for short in-the-money American options near dividends.
- Margin and funding are not modelled and cash earns nothing: returns are P&L over a fixed `capital`, so set it to what the strategy would really tie up (the strike, for a cash-secured put).

## Try it

```python
from src.derivatives.backtest import OptionBacktester
from src.derivatives.strategies import make_strategy
from src.derivatives.synthetic import SyntheticMarketSpec, generate_market

m = generate_market(SyntheticMarketSpec(), n_days=120, seed=1, warm=60)
bt = OptionBacktester({d: m.chain(d) for d in m.dates()}, m.underlying["spot"], capital=10000.0)
res = bt.run(make_strategy("iron_condor"))
print({k: round(float(v), 3) for k, v in res.summary().items() if k in ("total_return", "max_drawdown", "costs_total")})
```
