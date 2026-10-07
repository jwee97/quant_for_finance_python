# bab

*Family: cross-sectional*

## What it bets on

Betting against beta (Frazzini-Pedersen): long low-beta assets and short high-beta assets, each leg scaled to beta one, rebalanced monthly

Leverage-constrained investors bid up high-beta assets, so low-beta assets earn more per unit of market risk; the trade is leveraged long and beta-neutral.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252, `shrink` = 0.6, `min_assets` = 4

## Run it

```bash
quant backtest --model bab --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.06, CAGR 0.1%, volatility 10.7%, max drawdown -44.4%, turnover 3.7 times a year, deflated Sharpe probability 0.02 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Low risk, lottery demand, illiquidity and value with momentum](../techniques/low-risk-and-multi-factor.md)
