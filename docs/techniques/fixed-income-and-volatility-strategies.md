---
title: "Fixed income and volatility strategy families"
slug: fixed-income-and-volatility-strategies
difficulty: 3
chapter: Ch. 22
prerequisites: [momentum, regime-detection]
stages: [30]
files: [src/strategies/fixed_income.py, src/strategies/volatility.py, experiments/stage30_library.py]
figures: [59, 60]
tests: [tests/test_strategies.py]
models: [curve_steepener, butterfly, carry_rolldown, duration_timing, vrp_timing, variance_carry, implied_vs_realized]
---

# Fixed income and volatility strategy families

## In one sentence

Bond strategies trade the shape of the yield curve with DV01-neutral legs, and volatility strategies trade the gap between implied and realised volatility.

## The idea

A curve steepener goes long a short-maturity bond and short a long-maturity one, scaled so a parallel move in rates nets to zero (DV01 neutral). A butterfly adds a belly against two wings. Carry and roll-down
favour the bonds with the highest yield and the steepest curve. Volatility strategies look at the volatility risk premium: implied variance usually exceeds realised, so being long equities when the gap is wide may be paid.

## Why it matters

These strategies are where multi-asset funds differ from equity funds. They also show how much a signal's apparent return comes from a hidden exposure to rates or equity.

## How this repo uses it

The DV01-neutral construction uses rolling data-estimated durations because ETF durations are not in the data. Value and quality in the cross-sectional family are price-based proxies and are labelled so. Dispersion trading is conceptual and not
built; cross-exchange spreads are not built either.

## What we found

In the library search these families were mixed: carry and roll-down earned a small positive net Sharpe, the steepener and butterfly did not (the butterfly net Sharpe was negative), and volatility-risk-premium timing was close to zero with a large drawdown.

## Going deeper

```
DV01-neutral steepener: long w_s of SHY, short w_l of TLT with  w_s * D_s = w_l * D_l   (D = rolling data-estimated duration)
butterfly: long belly, short wings with 50% of belly DV01 in each wing
VRP timing: premium_t = implied variance (VIX^2) - realised variance;  long equities when premium_t is above its trailing median
```
Durations are estimated by regressing each ETF's return on the change in the 10-year yield over a rolling window, because the data holds prices, not bond analytics.

## Pitfalls

- Duration estimates change; a neutral hedge today may not be neutral next month.
- ETF proxies for implied volatility are crude.
- Carry earns until it crashes.

## Try it

```bash
quant backtest --model curve_steepener --tearsheet
```
