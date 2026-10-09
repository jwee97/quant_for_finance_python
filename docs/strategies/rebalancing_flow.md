# rebalancing_flow

*Family: seasonal*

## What it bets on

Month-end rebalancing flow: when risky assets beat safe ones over the month, balanced funds must sell risky assets at the month-end, so lean the other way in its last days

A fund that targets a mix (60/40 is the archetype) finds its risky share too high after risky assets outperform and sells them near the month-end, and the reverse after they underperform:
the size of the flow is the drift in the mix times the fund's assets, and the drift is visible to everyone from prices. Harvey, Mazzoleni and Melone (NBER 33554, 2025) find that when pension funds are overweight
stocks, equity returns fall by about 17 basis points over the next day, and that the pressure fades within about two weeks; funds that rebalance on a calendar do so at month- and quarter-ends, which is why this strategy looks at the last days of the month. It measures
the relative return of the equal-weight risky basket over the safe basket since the last month-end, in units of its own history of months, and in the last days of the month holds the opposite
of the expected flow: short the risky assets and long the safe ones after a strong month, the reverse after a weak one. It is flat the rest of the month. A position stamped on day s earns the
return of day s+2 under the engine's one-day signal lag, so ``lead`` (default 2) places the stamps so that the book earns the last ``days`` trading days of the month. Whether the flow is large
enough to matter after costs is an empirical question: run it before believing it, and count it as a trial.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `days` = 3, `lead` = 2, `scale` = 1.0, `min_months` = 36, `safe_leg` = True

## Run it

```bash
quant backtest --model rebalancing_flow --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.02, CAGR 0.0%, volatility 1.7%, max drawdown -5.6%, turnover 7.7 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Learn more

[Portfolio rebalancing styles: policy bands, month-end flows, flight to quality, market outlook](../techniques/portfolio-rebalancing-styles.md)
