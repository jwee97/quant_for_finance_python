# carver_ewmac

*Family: time-series*

## What it bets on

Carver's EWMAC forecast: six exponentially weighted crossovers (2/8 to 64/256 days), each scaled to an average absolute forecast of 10 from the data's own history, capped at 20 and averaged with a diversification multiplier

Trend rules of different speeds earn different amounts and cost different turnover; scaling every rule to the same average size, capping the extremes and averaging them with a multiplier for
their imperfect correlation gives one smooth forecast (+20 is the strongest buy, 0 no view, 10 an average-sized position).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `fast_spans` = (2, 4, 8, 16, 32, 64), `slow_multiple` = 4.0, `vol_span` = 35, `target_abs` = 10.0, `cap` = 20.0, `min_history` = 500, `diversification` = True, `fdm_cap` = 2.5

## Run it

```bash
quant backtest --model carver_ewmac --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.05, CAGR -0.9%, volatility 9.3%, max drawdown -33.2%, turnover 22.4 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- Carver (2015), Systematic Trading, Harriman House
- Carver (2019), Leveraged Trading, Harriman House
- Carver (2023), Advanced Futures Trading Strategies, Harriman House

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
