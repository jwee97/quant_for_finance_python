# basis_reversion

*Family: crypto*

## What it bets on

Basis trading: hold the carry position when the perpetual trades rich to spot relative to its own history

A wide perpetual-over-spot basis converges; the short-perpetual leg earns the convergence.

## Inputs

- Macro or alternative series required: BASIS_BTC, BASIS_ETH
- Parameters: none

## Run it

```bash
quant backtest --model basis_reversion --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -1.92, CAGR -2.6%, volatility 1.3%, max drawdown -13.4%, turnover 44.4 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

One exchange (Deribit), four assets, about five years of data; the inverse perpetual is treated as linear.

## Learn more

[Crypto carry: funding, basis and stablecoin flows](../techniques/crypto-carry.md)
