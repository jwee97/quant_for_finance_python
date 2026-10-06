# stablecoin_flow

*Family: crypto*

## What it bets on

Stablecoin flow factor: be long BTC and ETH when the total stablecoin supply has been growing quickly

Stablecoin issuance is dry powder entering the crypto market; supply growth should lead spot prices.

## Inputs

- Macro or alternative series required: STABLE_SUPPLY
- Parameters: `window` = 30

## Run it

```bash
quant backtest --model stablecoin_flow --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.41, CAGR -16.2%, volatility 30.4%, max drawdown -72.4%, turnover 1.3 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

One exchange (Deribit), four assets, about five years of data; the inverse perpetual is treated as linear.

## Learn more

[Crypto carry: funding, basis and stablecoin flows](../techniques/crypto-carry.md)
