# daa

*Family: allocation*

## What it bets on

Defensive Asset Allocation (Keller-Keuning): canary assets with negative momentum move the portfolio from the top offensive assets into the best defensive one

Two 'canary' markets (emerging equities and aggregate bonds) are the first to signal stress; each bad canary moves a share of the portfolio to safety.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `canary` = ('EEM', 'AGG'), `offensive` = ('SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'GLD', 'DBC', 'HYG'), `defensive` = ('IEF', 'SHY', 'LQD', 'TLT'), `top` = 6, `breadth` = 2

## Run it

```bash
quant backtest --model daa --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.76, CAGR 8.1%, volatility 11.0%, max drawdown -26.0%, turnover 13.0 times a year, deflated Sharpe probability 0.78 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Tactical asset allocation: Faber, Keller and the classic model portfolios](../techniques/tactical-asset-allocation.md)
