# donchian

*Family: time-series*

## What it bets on

Donchian channel trend following (turtle rules): enter on a 55-day breakout, exit on the 20-day opposite channel

Trends are long enough that buying new highs and selling new lows earns more than it costs.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `entry` = 55, `exit` = 20

## Run it

```bash
quant backtest --model donchian --allocator sleeves --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.52, CAGR -5.5%, volatility 9.8%, max drawdown -73.9%, turnover 1.4 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe nan (-0.39 over its own, longer live window), CAGR 0.0%, volatility 0.0%, max drawdown 0.0%, turnover 0.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Learn more

[Momentum and trend following](../techniques/momentum.md)
