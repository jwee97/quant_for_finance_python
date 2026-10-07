---
title: "Market microstructure: order flow, market making and optimal execution"
slug: market-making-and-order-flow
difficulty: 3
chapter: Ch. 20
prerequisites: [execution-and-impact]
stages: []
files: [src/microstructure/order_flow.py, src/microstructure/lob.py, src/microstructure/market_making.py, src/microstructure/execution.py]
figures: []
tests: [tests/test_microstructure.py]
models: []
---

# Market microstructure: order flow, market making and optimal execution

## In one sentence

At the finest scale, prices move because orders arrive; `src.microstructure` measures that flow, simulates a limit order book and a market maker's quoting, and solves the trader's problem of how fast to execute.

## The idea

**Order-flow measures** work from trades and quotes. `tick_rule` and `lee_ready` sign trades as buyer- or seller-initiated; `ofi` computes the **order-flow imbalance** of Cont, Kukanov and Stoikov (2014) from changes in best bid and ask and their sizes, and `ofi_price_impact` regresses mid-price changes on it (linear, with a slope inversely related to depth); `kyle_lambda` and `amihud` measure price impact per unit of signed volume and per dollar traded; `roll_spread` and `corwin_schultz` estimate the effective spread from prices and from high-low ranges when no quotes exist; `effective_spread` and `realized_spread` split the trading cost into what the liquidity provider keeps and what is lost to adverse selection; `vpin` is the volume-synchronised probability of informed trading.

`simulate_lob` is a Poisson **limit order book** (depth by level, limit orders, cancellations and market orders), used as test data for these measures: price impact here has a known mechanism.

**Market making** is the Avellaneda-Stoikov problem (2008). With inventory `q`, risk aversion `g`, volatility `s` and order-arrival decay `k`, the optimal quotes are centred on the **reservation price** `r = S - q g s^2 (T - t)` and have total spread `g s^2 (T - t) + (2/g) ln(1 + g/k)`: a maker long inventory shades both quotes down to sell. `as_quotes` computes them and `simulate_market_making` runs the strategy against a symmetric one with the same average spread, optionally with informed traders (adverse selection) and an inventory limit; `compare_strategies` returns the table.

**Optimal execution** (Almgren and Chriss 2001): liquidating `X` shares by `T` trades off temporary impact (cost grows with the trading rate) against price risk (waiting is risky). `almgren_chriss_schedule` gives the closed-form trajectory `x_j = X sinh(k (T - t_j)) / sinh(k T)` for risk aversion `lambda`; `efficient_frontier` traces expected cost against its standard deviation; `twap_schedule` and `vwap_schedule` are the benchmarks; `implementation_shortfall` evaluates any schedule by simulation.

## Why it matters

The costs in a daily backtest are an assumption about this level: spread, impact and adverse selection. Knowing how they behave, and what a market maker earns for bearing inventory, shows when a cost assumption is generous.

## How this repo uses it

The execution stage's square-root impact model is the daily-frequency summary of what is modelled here at the event level. This package needs tick or order-book data to say anything about a real market, which no free source provides, so it ships a simulated book and checks the code against known answers.

## What we found

The tests check the trade-signing rules, the OFI formula on hand-computed cases, that the OFI regression recovers a planted price-impact slope and has a positive, significant slope in the simulated book, that the Kyle, Amihud, Roll and Corwin-Schultz estimators recover planted values, that the effective and realised spreads separate adverse selection, the Avellaneda-Stoikov formulas, that inventory shading cuts the standard deviation of terminal wealth by more than 30% for a similar mean, that wealth decomposes exactly into spread capture, adverse selection and inventory P&L (and informed flow reduces wealth after the same fills), and that the Almgren-Chriss schedule is TWAP at zero risk aversion, front-loads as risk aversion rises and has the closed-form cost.

## Pitfalls

- Trade signing errors (the tick rule misclassifies a meaningful share of real trades) bias every order-flow statistic.
- A simulated book has no hidden liquidity, latency or queue-position effects; a market-making result here is a mechanism, not a profit forecast.
- Optimal execution uses a stylised, linear impact model; calibrate it to your own fills.
- VPIN's predictive claims are contested.

## Try it

```python
from src.microstructure import lob, order_flow as of

book = lob.simulate_lob(20000, seed=1)
o = of.ofi(book["bid"], book["bid_size"], book["ask"], book["ask_size"])
res = of.ofi_price_impact(o, book["mid"], 200)
print(round(res["beta"], 4), round(res["r2"], 2))
```
