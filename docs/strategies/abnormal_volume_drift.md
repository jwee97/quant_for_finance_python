# abnormal_volume_drift

*Family: time-series*

## What it bets on

News proxy: a move of more than two standard deviations on at least twice the usual volume drifts on for days; the same move on ordinary volume is faded

News shows in a price series as a large move that other traders confirm by trading: that kind of move tends to continue as the information spreads (under-reaction), while a large move nobody trades
on tends to be noise that reverts (Chan 2003; Campbell, Grossman and Wang 1993 on volume and reversals). The signal is the sign of the move, decaying linearly over ``horizon`` days. A proxy: it
cannot tell good news from bad news in a headline, only that something happened.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `k` = 2.0, `volume_ratio` = 2.0, `hold` = 10, `reversal` = 0.5, `vol_window` = 60

## Run it

```bash
quant backtest --model abnormal_volume_drift --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.55, CAGR -1.9%, volatility 3.4%, max drawdown -31.4%, turnover 15.3 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Learn more

[Alpha-generating styles: long-term, short-term, news, outlook and corporate actions](../techniques/alpha-generating-styles.md)
