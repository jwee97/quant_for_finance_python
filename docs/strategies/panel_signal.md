# panel_signal

*Family: event-driven*

## What it bets on

Company outlook from a score you supply (analyst revisions, earnings surprises, guidance): a date-by-ticker file, usable after a publication lag and for a limited time

Any forward-looking company measure with a date (the change in analyst earnings estimates, the standardised earnings surprise, a guidance flag) is a signal once it is handled like a price: it may
not be used before it was published (``lag``), it goes stale (``expiry`` trading days), and it is compared across companies (``standardise``). ``change`` replaces the level by its change over that many
days, which is what a revision is. Needs ``data/user/signals.csv`` (wide: ``date`` and one column per ticker; or long: ``date, ticker, value``) or the path you give.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `path` = 'signals.csv', `lag` = 1, `expiry` = 63, `change` = 0, `standardise` = True, `direction` = 1.0

## Run it

```bash
quant backtest --model panel_signal --tearsheet
```

## Learn more

[Alpha-generating styles: long-term, short-term, news, outlook and corporate actions](../techniques/alpha-generating-styles.md)
