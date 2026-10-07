# mean_reversion

*Family: cross-sectional*

## What it bets on

Generation 1 mean reversion: fade the 21-day price z-score

Prices that have stretched away from their recent average snap back (liquidity provision, overreaction).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 21, `basis` = 'log'

## Run it

```bash
quant backtest --model mean_reversion --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.13, CAGR -1.1%, volatility 6.9%, max drawdown -22.5%, turnover 22.7 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.04 (-0.08 over its own, longer live window), CAGR -0.3%, volatility 4.1%, max drawdown -18.6%, turnover 5.6 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Mean reversion and z-scores](../techniques/mean-reversion.md)
