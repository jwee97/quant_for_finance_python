---
title: "Short-term mean reversion: pullbacks, bands and reversals"
slug: short-term-mean-reversion
difficulty: 2
chapter: Ch. 20
prerequisites: [mean-reversion, plugin-framework]
stages: []
files: [src/strategies/reversion.py]
figures: []
tests: [tests/test_strategy_library_v2.py]
models: [rsi2, ibs_reversion, bollinger_reversion, stochastic_reversion, consecutive_down, short_term_reversal, ou_reversion, range_reversion]
---

# Short-term mean reversion: pullbacks, bands and reversals

## In one sentence

After a sharp short-term fall, prices tend to bounce; these rules buy the pullback and sell the recovery.

## The idea

Liquidity providers are paid for absorbing selling pressure, and over a few days prices that fell on little news partly recover. Larry Connors popularised the **RSI(2) pullback** (`rsi2`): buy when the two-day RSI is below 10 while the price is above its 200-day average, exit when it closes above its five-day average. **Internal bar strength** (`ibs_reversion`) buys a close in the bottom fifth of the day's range. **Bollinger reversion** (`bollinger_reversion`) buys a close two deviations under its 20-day mean, and the **stochastic** version (`stochastic_reversion`) buys a low smoothed %K. `consecutive_down` buys after three straight lower closes. The cross-sectional **short-term reversal** (`short_term_reversal`; Jegadeesh 1990, Lehmann 1990) buys last week's losers and sells its winners. Two filters keep reversion where it belongs: `ou_reversion` fits an Ornstein-Uhlenbeck (AR(1)) model and acts only when a Dickey-Fuller test rejects a random walk and the half-life is short, and `range_reversion` trades only while the ADX says there is no trend.

## Why it matters

Reversion is the natural complement of trend: it earns when markets chop and loses when they trend, so it diversifies a trend book. It is also the family where backtests mislead most, because the edge per trade is small and costs are not: a rule that trades every other day pays the spread and the commission each time.

## How this repo uses it

The time-series rules run as independent sleeves, long-only by default (`shorts=False`) because shorting strength in a rising market is how reversion strategies blow up, and they rebalance daily because their signal changes daily. The up-trend filter (price above its 200-day average) is part of the published rules. Where a high and a low are missing, IBS uses a 10-day range of closes.

## What we found

All numbers below come from one run of [the strategy survey](../strategy_survey.md): the platform's 15 ETFs, default parameters, net of 10 bps costs, 71 strategies counted as trials in the deflated Sharpe ratio, equal weight at about 0.67. They are exploratory, not tested hypotheses. Two pullback rules stood out: `ibs_reversion` (net Sharpe +0.76, deflated probability 0.79) and `stochastic_reversion` (+0.76, 0.78), trading 6x and 3x a year, both above equal weight (0.67). The classic RSI(2) rule (+0.35) and Bollinger bands (+0.27) were positive but weaker, `consecutive_down` (-0.26) was negative, and the unfiltered versions lost badly after costs: `short_term_reversal` (-0.31, 99x turnover a year) and `range_reversion` (-1.36). `ou_reversion` almost never traded: on liquid ETFs the unit-root test rarely rejects a random walk, which is what the test is for. None of the winners reaches a deflated probability of 0.95 once all 71 strategies count as trials, and the two best are close cousins (both buy a close near the bottom of a range inside an up-trend), so they are one idea, not two.

## Going deeper

```
RSI(2) pullback:   long if RSI_2 < 10 and P > SMA_200;  exit when P > SMA_5          (state is held between the two events)
IBS:               IBS = (C - L) / (H - L);  long if IBS < 0.2 and P > SMA_200;  exit when IBS > 0.7
OU fit:            dx_t = a + g x_(t-1) + e;  half-life = ln 2 / -ln(1 + g);  DF t-statistic = g / se(g);  act if t < -2.86 and half-life <= 30
```
On a 120-day window a random walk looks mean-reverting by chance: the AR coefficient is biased downward by roughly 5 / T, giving an apparent half-life near 17 days. That is why `ou_reversion` demands the Dickey-Fuller test as well.

## Pitfalls

- The edge per trade is a few basis points to tens of basis points; assume 10 bps each way and check the net number.
- Reversion earns steadily and loses suddenly: a position that falls further is bought again, and in a crash that compounds.
- Daily rules are the most sensitive to execution (closing-auction fills, spreads on the day you need them).
- Single-stock reversal profits are largest in small, illiquid names; on liquid ETFs they are much smaller.

## Try it

```bash
quant backtest --model rsi2 --allocator sleeves --tearsheet
quant backtest --model ibs_reversion --allocator sleeves --alloc-param vol_scale=true
quant backtest --model short_term_reversal --allocator score_stack
```
