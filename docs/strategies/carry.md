# carry

*Family: cross-sectional*

## What it bets on

Carry: favour the bond ETFs whose yield exceeds the cash rate by the most

Assets that pay more than cash earn that extra yield if prices do not move; carry is a risk premium, not a free lunch.

## Inputs

- Macro or alternative series required: DGS2, DGS5, DGS10, DGS20, DFF
- Parameters: none

## Run it

```bash
quant backtest --model carry --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.25 (0.15 over its own, longer live window), CAGR 1.4%, volatility 6.7%, max drawdown -15.5%, turnover 3.7 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Cross-sectional factors: low volatility, value, quality, carry, defensive beta](../techniques/cross-sectional-factors.md)
