# amihud_illiquidity

*Family: cross-sectional*

## What it bets on

Illiquidity premium (Amihud): favour assets whose price moves most per dollar traded, averaged over 63 days (needs volume)

Assets that are costly to trade must pay a higher expected return to attract holders; the price impact per dollar traded is the measure.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 63

## Run it

```bash
quant backtest --model amihud_illiquidity --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.39, CAGR -2.8%, volatility 6.6%, max drawdown -45.3%, turnover 2.5 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Low risk, lottery demand, illiquidity and value with momentum](../techniques/low-risk-and-multi-factor.md)
