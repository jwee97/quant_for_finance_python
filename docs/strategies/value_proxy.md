# value_proxy

*Family: cross-sectional*

## What it bets on

A price-based stand-in for value: long-horizon reversal, favouring assets that have fallen most over five years

Assets that are cheap relative to their own long history mean-revert (De Bondt and Thaler). Not fundamental value: no earnings, no book value.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 1260

## Run it

```bash
quant backtest --model value_proxy --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.48, CAGR -3.4%, volatility 6.8%, max drawdown -40.8%, turnover 2.7 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.16 (0.16 over its own, longer live window), CAGR 0.1%, volatility 0.8%, max drawdown -2.8%, turnover 0.5 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Cross-sectional factors: low volatility, value, quality, carry, defensive beta](../techniques/cross-sectional-factors.md)
