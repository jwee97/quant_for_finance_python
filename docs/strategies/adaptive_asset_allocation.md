# adaptive_asset_allocation

*Family: allocation*

## What it bets on

Adaptive asset allocation (ReSolve): the top-k assets by 6-month momentum, weighted by inverse volatility (the original minimises variance)

Momentum picks what to own and risk weighting sizes it: strong assets in proportion to how calm they are. Inverse volatility stands in for the original's minimum variance.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `top_k` = 5, `months` = 6, `vol_window` = 63

## Run it

```bash
quant backtest --model adaptive_asset_allocation --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.55, CAGR 6.0%, volatility 12.0%, max drawdown -30.5%, turnover 9.3 times a year, deflated Sharpe probability 0.47 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Tactical asset allocation: Faber, Keller and the classic model portfolios](../techniques/tactical-asset-allocation.md)
