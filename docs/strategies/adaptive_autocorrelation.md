# adaptive_autocorrelation

*Family: time-series*

## What it bets on

Adaptive autocorrelation: continue yesterday's move where an asset's own returns have significant positive autocorrelation, fade it where it is significantly negative

Short-horizon returns are weakly predictable from their own past, with a sign that differs by asset and period (momentum in some, bid-ask bounce and overreaction in others). Measuring the
first-order autocorrelation over the last year and acting only when its t-statistic says it is not noise lets each asset choose between continuing and fading, and stay out when there is nothing to find.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 250, `scale` = 2.0

## Run it

```bash
quant backtest --model adaptive_autocorrelation --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.84, CAGR -6.7%, volatility 7.9%, max drawdown -78.7%, turnover 116.2 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Learn more

[Alpha-generating styles: long-term, short-term, news, outlook and corporate actions](../techniques/alpha-generating-styles.md)
