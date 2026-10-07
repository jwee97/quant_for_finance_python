---
title: "Risk-managed exposure: volatility scaling, fear and credit"
slug: risk-managed-exposure
difficulty: 2
chapter: Ch. 21
prerequisites: [volatility-forecasting, plugin-framework]
stages: []
files: [src/strategies/risk_timing.py]
figures: []
tests: [tests/test_strategy_library_v2.py]
models: [vol_managed_long, vix_spike_reversion, credit_spread_timing]
---

# Risk-managed exposure: volatility scaling, fear and credit

## In one sentence

Instead of forecasting returns, these rules adjust how much risk to hold using measures of risk itself: recent volatility, a spike in implied volatility, and credit conditions.

## The idea

**Volatility-managed portfolios** (`vol_managed_long`; Moreira and Muir 2017) hold each asset in inverse proportion to its last month's realised variance: volatility is persistent, but the return earned per unit of risk is not higher when volatility is high, so shrinking exposure in turbulence raises the Sharpe ratio. **Buying fear** (`vix_spike_reversion`) holds equities after the VIX jumps more than two standard deviations above its 63-day mean and exits when it falls back, on the idea that implied volatility overshoots. **Credit risk appetite** (`credit_spread_timing`) compares credit funds with government bonds over three months against their own history: credit underperforming rates warns that risk appetite is fading.

## Why it matters

Risk timing is the only market timing with a solid theoretical footing: you are not guessing the direction, you are relying on the fact that risk clusters. It also works as an overlay: the pipeline's own `risk.mode: regime` and drawdown limits do the same job for any strategy.

## How this repo uses it

`vol_managed_long` sets each asset's exposure to its history's average variance over its last-month variance, capped at three times and long only; it uses no information beyond the asset's own returns. The VIX and credit rules use the macro panel and the asset-class labels (credit funds against the `rates` and `fixed_income` classes), and say what is missing when the bundle lacks them.

## What we found

All numbers below come from one run of [the strategy survey](../strategy_survey.md): the platform's 15 ETFs, default parameters, net of 10 bps costs, 79 strategies counted as trials in the deflated Sharpe ratio. Each rule starts on its own first day, so it is compared with equal weight **over the same dates** (the survey's comparison column), which ranges from 0.64 to 0.94 depending on the start. The numbers are exploratory, not tested hypotheses. Volatility management was among the better results, but it did not beat equal weight: `vol_managed_long` earned a net Sharpe of +0.81 (deflated probability 0.82) against +0.94 for equal weight over the same dates, with about 18% volatility and a 31% drawdown, because it is long-only and levers calm assets up to three times (bond funds in quiet years), so read it as a risk-sizing rule on a long-only book, not as a free lunch. The VIX rule (+0.24) was weak and credit timing (-0.17) lost money after costs. Compare `vol_managed_long` with equal weight scaled to the same volatility before calling the difference skill.

## Going deeper

```
vol_managed_long:   score_(i,t) = min( mean_(s<=t) V_(i,s) / V_(i,t), 3 ),   V = sum of squared daily returns over the last 21 days
vix_spike:          z = (VIX - mean_63) / std_63;  long equities when z > 2, until VIX < mean_63
credit:             ratio_t = mean log P(credit) - mean log P(rates);  signal = tanh( z_504( ratio_t - ratio_(t-63) ) / 2 )
```

## Pitfalls

- Volatility targeting lowers risk after the fact: it does not avoid the first day of a crash.
- The benefit depends on volatility being more persistent than returns are predictable; it is weaker for assets with mean-reverting volatility.
- Scaling by realised variance changes exposure every day; costs rise with the rebalance frequency.
- The VIX rule trades rarely, so its few trades dominate the result.

## Try it

```bash
quant backtest --model vol_managed_long --allocator sleeves --tearsheet
quant backtest --model vix_spike_reversion --allocator sleeves
```
