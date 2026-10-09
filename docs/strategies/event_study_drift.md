# event_study_drift

*Family: event-driven*

## What it bets on

Walk-forward event study: learn the abnormal-return path that follows each type of event from completed past events only, and trade it when its t-statistic is large (events from price shocks, or from your file)

Corporate actions, index changes, announcements and price shocks all have the same structure: a date, an asset and a type, followed by a stretch of returns that is, on average, not what the market did.
The strategy measures that average path by event TYPE, using only events whose whole window has finished by the decision date, and holds the expected remaining drift of every event in its window
when (and only when) the average cumulative abnormal return of the type is at least ``tstat`` standard errors from zero on at least ``min_events`` events. It therefore learns whether a type
continues or reverts, and stays flat on noise. Without a file the events are moves beyond ``z`` standard deviations, split by direction and by heavy or ordinary volume. The signal is the expected
remaining abnormal return over its own standard deviation (squashed by tanh).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `path` = '', `z` = 2.0, `volume_ratio` = 2.0, `window` = 10, `min_events` = 30, `tstat` = 2.0

## Run it

```bash
quant backtest --model event_study_drift --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.78, CAGR -0.2%, volatility 0.3%, max drawdown -4.4%, turnover 3.9 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Learn more

[Alpha-generating styles: long-term, short-term, news, outlook and corporate actions](../techniques/alpha-generating-styles.md)
