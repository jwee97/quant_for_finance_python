# sell_in_may

*Family: time-series*

## What it bets on

Halloween indicator (sell in May): hold equities from November through April and stay out from May through October

Equity returns have historically been concentrated in the winter half of the year (Bouman and Jacobsen 2002), with no agreed risk explanation: a famous, fragile rule.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `start_month` = 11, `end_month` = 4, `classes` = ('equity',)

## Run it

```bash
quant backtest --model sell_in_may --allocator sleeves --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.45, CAGR 2.1%, volatility 5.0%, max drawdown -12.2%, turnover 0.7 times a year, deflated Sharpe probability 0.30 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Bouman and Jacobsen (2002), The Halloween indicator, 'Sell in May and go away': another puzzle, American Economic Review 92(5)](https://doi.org/10.1257/000282802762024683)

## Learn more

[Seasonality and calendar effects](../techniques/seasonality-and-calendar-effects.md)
