# implied_vs_realized

*Family: volatility*

## What it bets on

Implied versus realised volatility by asset: long where the implied index exceeds trailing realised volatility by the most

Where fear (implied) runs ahead of experience (realised), the premium paid for protection is large and tends to be earned by the seller.

## Inputs

- Macro or alternative series required: VIX, VXN, GVZ, OVX
- Parameters: `realised_window` = 21

## Run it

```bash
quant backtest --model implied_vs_realized --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.48 (-0.47 over its own, longer live window), CAGR -5.7%, volatility 10.9%, max drawdown -62.4%, turnover 9.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Uses VIX-type indices as the implied-volatility input; no option-level data.

## Learn more

[Fixed income and volatility strategy families](../techniques/fixed-income-and-volatility-strategies.md)
