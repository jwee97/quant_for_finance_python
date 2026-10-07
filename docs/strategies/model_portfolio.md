# model_portfolio

*Family: allocation*

## What it bets on

Static model portfolios as benchmarks: 60/40, Permanent, All Weather, Bogleheads three-fund, or equal weight across asset classes, rebalanced monthly

The mixes most investors compare themselves with: any active strategy should be judged against them, not only against cash.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `preset` = '60_40'

## Run it

```bash
quant backtest --model model_portfolio --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.83, CAGR 9.1%, volatility 11.4%, max drawdown -26.4%, turnover 1.4 times a year, deflated Sharpe probability 0.86 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Tactical asset allocation: Faber, Keller and the classic model portfolios](../techniques/tactical-asset-allocation.md)
