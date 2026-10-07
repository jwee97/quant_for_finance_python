# long_term_reversal

*Family: cross-sectional*

## What it bets on

Long-term reversal (value proxy): within each asset class, buy what has fallen over five years and sell what has risen

Prices revert toward fundamental value over multi-year horizons, so a five-year loser is cheap against its own history.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 1260, `skip` = 252

## Run it

```bash
quant backtest --model long_term_reversal --tearsheet
```

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.
