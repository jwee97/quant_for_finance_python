# macro_factor_timing

*Family: macro*

## What it bets on

Macro factor timing: forecast each price factor with the premium it earned before in the same macroeconomic state (Fed easing or tightening, M1, GDP, CPI or PPI growth, an up or down market)

Factors pay differently in different economies: momentum in rising and falling markets (Cooper, Gutierrez and Hameed 2004), value in expansions and contractions, low-risk stocks when policy
tightens. ``state`` is ``market`` (the default: needs no series), ``fed`` (the direction of the federal funds rate, series ``DFF``), ``m1`` (``M1SL``), ``gdp`` (``GDPC1``), ``inflation``
(``CPIAUCNS``) or ``ppi`` (``PPIACO``); the series come with their publication lags from the macro panel. The premium forecast in a state is what the factor earned in the same state in earlier
months, shrunk toward its overall mean; states are classified from data available at the date (growth against its own median so far, never against the full sample).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factor` = 'all', `state` = 'market', `shrink` = 12.0, `min_obs` = 36, `window` = 0, `market_months` = 12, `min_history` = 60, `flat_band` = 0.25

## Run it

```bash
quant backtest --model macro_factor_timing --tearsheet
```

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Factor timing: the calendar, the macroeconomy, the market's state and the earnings season](../techniques/factor-timing.md)
