# expression

*Family: custom*

## What it bets on

A strategy written as a one-line formula over prices (for example rank(mom(126, 21)) - rank(vol(63))); parsed safely, backward-looking only

A formula is a hypothesis in one line: ``mom(252, 21)`` bets that last year's winners keep winning. Higher score = expect a higher return.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `expr` = 'mom(252, 21)', `mode` = 'cross_sectional'

## Run it

```bash
quant backtest --model expression --tearsheet
```

## Learn more

[Your own strategies: formulas and Python files](../techniques/custom-strategies.md)
