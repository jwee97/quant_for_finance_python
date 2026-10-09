# squeeze_breakout

*Family: time-series*

## What it bets on

Volatility squeeze breakout: when Bollinger bandwidth has been in the lowest fifth of its range, go with the first close outside the band and hold until the close crosses the middle band

Volatility clusters in time, so a spell of unusually narrow ranges is followed by a wider one; the direction of the first decisive close out of the band is the best available guess for the direction
of the expansion. The squeeze must have been present the day BEFORE the breakout, so the band that is broken is not itself widened by the breakout bar.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 20, `k` = 2.0, `lookback` = 126, `quantile` = 0.2

## Run it

```bash
quant backtest --model squeeze_breakout --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.41, CAGR -1.8%, volatility 4.2%, max drawdown -32.5%, turnover 12.5 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Learn more

[Alpha-generating styles: long-term, short-term, news, outlook and corporate actions](../techniques/alpha-generating-styles.md)
