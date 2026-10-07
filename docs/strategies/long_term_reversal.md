# long_term_reversal

*Family: cross-sectional*

## What it bets on

Long-term reversal (value proxy): within each asset class, buy what has fallen over five years and sell what has risen

Prices revert toward fundamental value over multi-year horizons, so a five-year loser is cheap against its own history.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 1260, `skip` = 252

## Run it

```bash
quant backtest --model long_term_reversal --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.37, CAGR -1.4%, volatility 3.5%, max drawdown -20.5%, turnover 3.6 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Cross-asset carry, basis momentum and long-term reversal](../techniques/cross-asset-carry.md)
