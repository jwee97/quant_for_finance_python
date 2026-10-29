# paa

*Family: allocation*

## What it bets on

Protective Asset Allocation (Keller-Keuning): hold the top risky assets by momentum, moving a growing fraction into the best safe asset as fewer risky assets trend up

Breadth of trend across many markets is an early warning of a crash: when most risky assets are falling, step aside into bonds (a canary built from the whole risky set).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 12, `protection` = 1, `top` = 6, `risky` = ('SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'GLD', 'DBC', 'HYG'), `safe` = ('IEF', 'SHY', 'AGG', 'TLT')

## Run it

```bash
quant backtest --model paa --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.73, CAGR 7.9%, volatility 11.3%, max drawdown -31.2%, turnover 6.9 times a year, deflated Sharpe probability 0.73 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Keller and Keuning (2016), Protective asset allocation (PAA): a simple momentum-based alternative for term deposits, SSRN 2759734](https://doi.org/10.2139/ssrn.2759734)

## Learn more

[Tactical asset allocation: Faber, Keller and the classic model portfolios](../techniques/tactical-asset-allocation.md)
