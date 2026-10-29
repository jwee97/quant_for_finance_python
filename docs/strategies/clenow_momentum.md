# clenow_momentum

*Family: cross-sectional*

## What it bets on

Clenow's momentum rank (Stocks on the Move): annualised slope of the exponential regression of the last 90 days times its R-squared, held only above the 100-day average and with no daily gap above 15%

A steep, straight climb scores high; a steep but jagged one is discounted by the R-squared. Names below their 100-day average or with a large recent gap are not eligible.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 90, `trend_window` = 100, `gap` = 0.15, `annual_days` = 250, `use_r2` = True

## Run it

```bash
quant backtest --model clenow_momentum --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.19, CAGR -1.7%, volatility 7.7%, max drawdown -30.3%, turnover 31.5 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- Clenow (2015), Stocks on the Move, Equilateral Publishing

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
