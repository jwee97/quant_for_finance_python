# momentum_13612w

*Family: cross-sectional*

## What it bets on

Keller's 13612W momentum: the weighted average of the 1-, 3-, 6- and 12-month returns (weights 12, 4, 2, 1)

Weighting recent months heavily makes the signal react faster to turning points than a plain 12-month return, which is why the Keller allocations use it.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `month` = 21

## Run it

```bash
quant backtest --model momentum_13612w --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.18, CAGR 1.1%, volatility 7.8%, max drawdown -21.7%, turnover 15.4 times a year, deflated Sharpe probability 0.05 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
