# funding_carry

*Family: crypto*

## What it bets on

Funding-rate carry: hold long-spot / short-perpetual when trailing funding is high, reverse when it is negative

Funding is paid by whichever side is crowded; it persists, so trailing funding forecasts the carry earned next month.

## Inputs

- Macro or alternative series required: FUNDING_BTC, FUNDING_ETH
- Parameters: none

## Run it

```bash
quant backtest --model funding_carry --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 2.98, CAGR 3.7%, volatility 1.2%, max drawdown -1.5%, turnover 19.9 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

One exchange (Deribit), four assets, about five years of data; the inverse perpetual is treated as linear.

## Learn more

[Crypto carry: funding, basis and stablecoin flows](../techniques/crypto-carry.md)
