# value_momentum

*Family: cross-sectional*

## What it bets on

Value and momentum together (Asness-Moskowitz-Pedersen): the average rank of five-year reversal and 12-1 month momentum

Value and momentum are negatively correlated, so combining them gives a steadier premium than either one; a five-year loss stands in for cheapness.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `value_lookback` = 1260, `momentum_lookback` = 252, `skip` = 21, `value_weight` = 0.5

## Run it

```bash
quant backtest --model value_momentum --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.03, CAGR -0.0%, volatility 6.5%, max drawdown -27.4%, turnover 9.1 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Low risk, lottery demand, illiquidity and value with momentum](../techniques/low-risk-and-multi-factor.md)
