# trend_atr

*Family: time-series*

## What it bets on

Trend following with an ATR filter: follow the 50/200 trend only when the gap exceeds k ATRs (avoid whipsaw)

A moving-average cross inside the noise band is noise; require the trend to be large relative to daily range.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `fast` = 50, `slow` = 200, `k` = 0.5, `atr_window` = 14

## Run it

```bash
quant backtest --model trend_atr --allocator sleeves --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.24, CAGR 1.2%, volatility 5.8%, max drawdown -14.6%, turnover 1.1 times a year, deflated Sharpe probability 0.08 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.31 (0.29 over its own, longer live window), CAGR 2.9%, volatility 11.6%, max drawdown -32.1%, turnover 4.9 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Learn more

[Momentum and trend following](../techniques/momentum.md)
