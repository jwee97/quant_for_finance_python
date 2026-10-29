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

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Amihud (2002), Illiquidity and stock returns: cross-section and time-series effects, Journal of Financial Markets 5(1)](https://doi.org/10.1016/s1386-4181(01)00024-6)

## Learn more

[Low risk, lottery demand, illiquidity and value with momentum](../techniques/low-risk-and-multi-factor.md)
