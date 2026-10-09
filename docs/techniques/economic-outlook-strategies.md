---
title: "Economic-outlook strategies: the yield curve and the credit cycle, with learned consequences"
slug: economic-outlook-strategies
difficulty: 3
chapter: Ch. 20
prerequisites: [macro-and-alternative-data, fixed-income-and-volatility-strategies, multiple-testing]
stages: []
files: [src/strategies/economic_outlook.py]
figures: []
tests: [tests/test_economic_outlook.py]
models: [curve_quadrant, credit_cycle_rotation]
---

# Economic-outlook strategies: the yield curve and the credit cycle, with learned consequences

## In one sentence

Turn the state of the yield curve or of credit spreads into a handful of named states, then let the data say, using only earlier months, which asset has paid in which state.

## The idea

The textbook table ("bear flattener: sell long bonds, favour cash") is a hypothesis, and the sign of the stock-bond relationship has changed with the inflation regime. These two strategies keep the useful part of the table, the states, and replace the consequences with measurements.

**Yield-curve strategy.** `curve_quadrant` looks at the 63-day change in the level of the curve (the average of the 2- and 10-year yields) and in its slope (10-year minus 2-year). The four combinations are the bull flattener (yields fall, the curve flattens), bull steepener (fall, steepen), bear flattener (rise, flatten) and bear steepener (rise, steepen), and they mean different things for growth, inflation and policy: a bull steepener is typically an easing cycle, a bear flattener a tightening one.

**Credit strategy.** `credit_cycle_rotation` takes the credit spread (the BAA-Treasury spread when the data has it, otherwise the relative performance of credit funds against Treasury funds, the spread widening when credit lags) and asks two questions: is it high or low against its own three-year median, and is it widening or tightening over 63 days? The four answers are recovery (high, tightening), expansion (low, tightening), contraction (high, widening) and late cycle (low, widening).

**What it does with a state.** For every (state, asset) pair the strategy computes the average return over the next `horizon` days (21) in that state, using only pairs whose forward window had finished by the decision date, and its t-statistic with the standard error corrected for the overlap of the windows (n/horizon independent observations). When the state today has at least `min_obs` matured observations and the pair's |t| is at least `tstat` (1.5) it takes a position in the direction of the sign, sized by tanh(t/2). A pair with too little evidence is flat. The same learning step, `conditional_tstat`, can be used with any state series you build.

## Why it matters

Hard-coded macro tables carry the regime of the author's sample. A rule that is allowed to learn only from matured history adapts when the relationship flips, and tells you when it does not know: no state with enough evidence, no position. The price is that it starts late (a state needs hundreds of matured days) and that with four states and fifteen assets there are sixty pairs, so some will be significant by chance.

## How this repo uses it

`curve_quadrant` needs `DGS2` and `DGS10` in the bundle's macro frame; the default bundle has them, and with your own tickers it says `needs macro series`. `credit_cycle_rotation` needs `BAA10Y` or an asset of class credit and one of class rates. Both are per-asset rules that run as sleeves with a weekly rebalance, and both expose `named_states(data)` to see the state on every date. The tests plant an effect (bonds that rise after falling yields, a risky asset that pays when spreads tighten) and check that it is found, that an unrelated asset is left alone, that nothing is used before its window finished (changing the future after a cutoff leaves every earlier statistic unchanged) and that the overlap correction makes one-day and 21-day windows give the same t-statistic for the same information.

## What we found

On the platform's 15 ETFs ([survey part two](../strategy_survey_2.md): 92 strategies counted as trials, 10 bps costs) `curve_quadrant` started in February 2015 (it needed that long to see each state often enough) and earned a net Sharpe of +0.70 against +0.78 for equal weight over the same dates, with 3% volatility and an 8.5% maximum drawdown; `credit_cycle_rotation` started in March 2016 and earned +0.43 (equal weight +0.91) at 1.9% volatility. Neither has a deflated probability near 0.95 (0.45 and 0.13). They are small, defensive books that trade a few significant pairs. That is the intended behaviour of a rule that says "I do not know" most of the time, and it is a result on one short post-2015 window.

## Going deeper

```
state_t      in {0, 1, 2, 3}                                 from the 63-day change of level and slope (curve) or of spread level and direction (credit)
fwd_s        = log return over days s+1 .. s+h               known at s+h
for date t:  pairs = { s <= t - h : state_s = state_t };   n = |pairs|
             t_stat = mean(fwd) / ( sd(fwd) * sqrt(h / n) )            (n / h independent windows)
score_t      = tanh( t_stat / 2 )  if  n >= min_obs and |t_stat| >= tstat,   else 0 (NaN while n < min_obs)
```

## Pitfalls

- Sixty (state, asset) pairs at |t| of 1.5 will produce false positives. Raise `tstat`, restrict `assets`, and count the model as one trial per parameter set you try.
- States defined from a 63-day change flip back and forth near zero. The strategy needs both changes to be non-zero and does not smooth them.
- A relationship learned over 2015-2025 is a statement about that decade's rates regime. Do not carry it to 1975.
- Macro series enter with their publication lags where the bundle applies them; the price-based credit proxy has none to worry about.

## Try it

```bash
quant backtest --model curve_quadrant --allocator sleeves --tearsheet
quant backtest --model credit_cycle_rotation --allocator sleeves --param tstat=2.0
```
