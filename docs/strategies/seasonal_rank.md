# seasonal_rank

*Family: cross-sectional*

## What it bets on

Same-month seasonality (Keloharju-Linnainmaa-Nyberg): rank assets by their average return in this calendar month over earlier years

Assets with a recurring pattern in a given month (heating-oil demand, tax-loss selling, earnings cycles) tend to repeat it; only earlier years are used.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `min_years` = 5

## Run it

```bash
quant backtest --model seasonal_rank --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.42, CAGR -3.0%, volatility 6.7%, max drawdown -42.3%, turnover 24.1 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Seasonality and calendar effects](../techniques/seasonality-and-calendar-effects.md)
