# double_seven

*Family: time-series*

## What it bets on

Double 7s (Connors-Alvarez): buy a 7-day closing low while the price is above its 200-day average, sell a 7-day closing high

In an up-trend a close at a one-week low is a pullback that has usually reversed within days; the exit at a one-week high takes the bounce.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 7, `trend_len` = 200, `allow_short` = False

## Run it

```bash
quant backtest --model double_seven --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.49, CAGR 2.3%, volatility 4.9%, max drawdown -12.6%, turnover 18.5 times a year, deflated Sharpe probability 0.31 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- Connors and Alvarez (2009), Short Term Trading Strategies That Work, TradingMarkets Publishing

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
