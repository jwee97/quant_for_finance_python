# jensen_alpha

*Family: cross-sectional*

## What it bets on

Persistent alpha (Jensen): favour assets whose market-adjusted return over the last three years was large relative to its noise, skipping the last month

The part of an asset's return that its exposure to the market does not explain, if it has been steady, tends to persist: managers and assets with a record of alpha keep a little of it, and
ranking on the appraisal ratio (alpha over residual volatility) rather than on alpha alone demotes the lucky streaks. The market is the equal-weight average of the assets here; the last month is skipped as
in 12-1 momentum, to keep short-term reversal out of a long-term signal.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 756, `skip` = 21

## Run it

```bash
quant backtest --model jensen_alpha --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.13, CAGR 0.6%, volatility 6.6%, max drawdown -19.4%, turnover 3.0 times a year, deflated Sharpe probability 0.02 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Alpha-generating styles: long-term, short-term, news, outlook and corporate actions](../techniques/alpha-generating-styles.md)
