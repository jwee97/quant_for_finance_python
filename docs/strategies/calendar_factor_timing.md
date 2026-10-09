# calendar_factor_timing

*Family: seasonal*

## What it bets on

Calendar factor timing: forecast each price factor with the premium it earned in the same calendar state before (January or not, the month, the month within the quarter, Nov-Apr or May-Oct)

The calendar is known in advance, so a rule that depends on it is not look-ahead; it is, however, among the most data-mined ideas in finance (the January effect, turn of the quarter, sell in May),
which is why the premium in each state is learned from earlier years only and shrunk toward the factor's overall premium until the state has been seen often. ``state`` is ``january``,
``month`` (twelve states, most to learn), ``quarter`` (the month within the quarter) or ``halloween``; ``factor`` is one price-based factor (``mom``, ``rev``, ``lowvol``, ``lowbeta``, ``nomax``,
``high``), a comma-separated list, or ``all``.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factor` = 'all', `state` = 'january', `shrink` = 12.0, `min_obs` = 24, `window` = 0

## Run it

```bash
quant backtest --model calendar_factor_timing --tearsheet
```

## Learn more

[Factor timing: the calendar, the macroeconomy, the market's state and the earnings season](../techniques/factor-timing.md)
