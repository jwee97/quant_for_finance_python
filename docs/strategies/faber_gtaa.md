# faber_gtaa

*Family: allocation*

## What it bets on

Faber's GTAA / Ivy portfolio: hold each asset class with an equal weight only while its price is above its 10-month average, otherwise hold cash

A simple trend filter on each asset class keeps most of the return of buy-and-hold with far shallower drawdowns (Faber 2007).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `months` = 10

## Run it

```bash
quant backtest --model faber_gtaa --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.75, CAGR 6.9%, volatility 9.6%, max drawdown -27.8%, turnover 3.1 times a year, deflated Sharpe probability 0.75 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Faber (2007), A quantitative approach to tactical asset allocation, Journal of Wealth Management 9(4)](https://doi.org/10.3905/jwm.2007.674809)

## Learn more

[Tactical asset allocation: Faber, Keller and the classic model portfolios](../techniques/tactical-asset-allocation.md)
