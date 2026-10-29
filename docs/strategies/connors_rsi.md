# connors_rsi

*Family: time-series*

## What it bets on

ConnorsRSI pullback (Connors-Alvarez): buy when the average of RSI(3), the RSI of the up/down streak and the percent rank of the day's return falls below 10 in an up-trend, sell above 70

Three short-term measures of how stretched a fall is (the price's own RSI, the length of the losing streak, how large the day's loss is against the last 100) agree more often at real extremes than any alone.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `rsi_len` = 3, `streak_len` = 2, `rank_len` = 100, `buy_below` = 10.0, `exit_above` = 70.0, `trend_len` = 200, `allow_short` = False, `sell_above` = 90.0, `cover_below` = 30.0

## Run it

```bash
quant backtest --model connors_rsi --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.25, CAGR 0.4%, volatility 1.5%, max drawdown -4.5%, turnover 3.7 times a year, deflated Sharpe probability 0.07 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- Connors and Alvarez (2009), Short Term Trading Strategies That Work, TradingMarkets Publishing

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
