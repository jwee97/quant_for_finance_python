# residual_momentum

*Family: cross-sectional*

## What it bets on

Residual momentum (Blitz-Huij-Martens): 12-1 month momentum of the part of each return that the market does not explain, scaled by its volatility

Stripping out market beta leaves the asset-specific trend, which is steadier and less exposed to momentum crashes than raw momentum.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `beta_window` = 252, `lookback` = 252, `skip` = 21

## Run it

```bash
quant backtest --model residual_momentum --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.33, CAGR 2.0%, volatility 6.6%, max drawdown -16.4%, turnover 5.7 times a year, deflated Sharpe probability 0.15 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
