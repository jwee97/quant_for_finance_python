# max_effect

*Family: cross-sectional*

## What it bets on

MAX effect (Bali-Cakici-Whitelaw): avoid assets with an extreme recent daily gain, favour those without a lottery-like spike

Investors overpay for assets that recently had a huge one-day gain (a lottery preference), so those assets earn less afterwards.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 21

## Run it

```bash
quant backtest --model max_effect --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.03, CAGR -0.0%, volatility 6.0%, max drawdown -26.8%, turnover 5.9 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Low risk, lottery demand, illiquidity and value with momentum](../techniques/low-risk-and-multi-factor.md)
