# vaa

*Family: allocation*

## What it bets on

Vigilant Asset Allocation (Keller-Keuning): all-in on the best offensive asset while every offensive asset has positive 13612W momentum, else all-in on the best defensive one

Requiring ALL offensive assets to trend up makes the switch to safety fast, at the cost of being aggressive and concentrated when it is on.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `offensive` = ('SPY', 'EFA', 'EEM', 'AGG'), `defensive` = ('LQD', 'IEF', 'SHY')

## Run it

```bash
quant backtest --model vaa --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.39, CAGR 3.9%, volatility 11.3%, max drawdown -21.4%, turnover 16.5 times a year, deflated Sharpe probability 0.21 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Tactical asset allocation: Faber, Keller and the classic model portfolios](../techniques/tactical-asset-allocation.md)
