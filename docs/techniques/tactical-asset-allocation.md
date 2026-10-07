---
title: "Tactical asset allocation: Faber, Keller and the classic model portfolios"
slug: tactical-asset-allocation
difficulty: 2
chapter: Ch. 19
prerequisites: [risk-parity, plugin-framework]
stages: []
files: [src/strategies/taa.py, src/framework/allocators_portfolio.py, src/portfolio/diversification.py]
figures: []
tests: [tests/test_strategy_library_v2.py]
models: [faber_gtaa, paa, vaa, daa, adaptive_asset_allocation, model_portfolio]
---

# Tactical asset allocation: Faber, Keller and the classic model portfolios

## In one sentence

Tactical asset allocation moves between asset classes once a month on simple trend and breadth rules; the model portfolios (60/40, Permanent, All Weather) are what to compare it with.

## The idea

**Faber's GTAA** (`faber_gtaa`, 2007) holds each asset class with an equal weight only while its price is above its 10-month average, otherwise cash. **Protective Asset Allocation** (`paa`; Keller and Keuning 2016) measures the breadth of trend across many markets: with `N` risky assets of which `n` are trending up and a protection factor `a`, the bond fraction is `(N - n) / (N - a N/4)` (capped at one), and the rest is spread over the top assets by momentum. **Vigilant Asset Allocation** (`vaa`) goes all in on the best offensive asset while all four offensive assets have positive 13612W momentum, otherwise all in on the best defensive one. **Defensive Asset Allocation** (`daa`) lets two canary assets (emerging equities and aggregate bonds) decide how much of the portfolio goes to the best defensive asset. **Adaptive asset allocation** (`adaptive_asset_allocation`, ReSolve) holds the top assets by six-month momentum weighted by inverse volatility (the original minimises variance). `model_portfolio` holds the classic static mixes: 60/40, Permanent, All Weather, a Bogleheads three-fund mix, or equal weight across classes.

## Why it matters

Most investors hold something like a model portfolio, so any active rule should beat it, not only cash. And the trend and breadth rules are a cheap, transparent way to cut equity exposure before the worst of a bear market, at the price of whipsaw in sideways years.

## How this repo uses it

These are structured models: their weights are the strategy, decided on the last trading day of each month from data through that day. They use the platform's ETF tickers where they exist and fall back to asset classes, so they run on your own tickers once labelled. Two new allocators belong here too: `min_variance` and `max_diversification` (Choueifaty and Coignard 2008) build monthly books from a trailing shrinkage covariance, and can be combined with any strategy's selection.

## What we found

All numbers below come from one run of [the strategy survey](../strategy_survey.md): the platform's 15 ETFs, default parameters, net of 10 bps costs, 71 strategies counted as trials in the deflated Sharpe ratio, equal weight at about 0.67. They are exploratory, not tested hypotheses. The static mix did as well as the active rules: `model_portfolio` (60/40 by default) earned a net Sharpe of +0.83 with 1x of turnover a year, above equal weight (0.67). Among the tactical rules `daa` (+0.76), `faber_gtaa` (+0.75) and `paa` (+0.73) came close to it, while `adaptive_asset_allocation` (+0.55) and `vaa` (+0.39, 17x turnover a year) did worse. None has a deflated probability above 0.95 once all 71 strategies are counted. The honest summary is that trend-based allocation matched a 60/40 mix over this sample rather than beating it; its argument is protection in bad years, which a twenty-year sample with three of them can only illustrate.

## Going deeper

```
PAA:   MOM = P / SMA_13(month-end P) - 1;   BF = min(1, (N - n) / (N - a N / 4));   risky = top T by MOM, (1 - BF) / T each;   safe = best safe asset by MOM (cash if it is negative)
VAA:   13612W = (12 r1 + 4 r3 + 2 r6 + r12) / 19;   all offensive > 0 ? best offensive : best defensive
DAA:   CF = min(1, bad canaries / breadth);   (1 - CF) over the top offensive assets, CF in the best defensive asset
```
Where the published universes use tickers not present in your bundle, the defaults select by asset class instead; pass `risky`, `safe`, `offensive`, `defensive` or `canary` explicitly to override.

## Pitfalls

- Monthly rebalancing means trading on one day of the month: a crash that starts the day after is fully taken.
- Published results are often shown on the dates the authors designed the rule over; out-of-sample numbers are lower.
- These rules hold few assets at a time; concentration risk is real.
- With 15 ETFs the choice of what counts as "risky" or "safe" is a decision that changes the answer.

## Try it

```bash
quant backtest --model faber_gtaa --tearsheet
quant backtest --model model_portfolio --param preset=all_weather
quant backtest --model momentum --allocator max_diversification
```
