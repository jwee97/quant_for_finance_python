# news_sentiment

*Family: event-driven*

## What it bets on

Headline sentiment from a file you supply (date, ticker, headline): a finance word list scores each headline, scores decay with a half-life and act from the day after

Prices under-react to news for days, so the tone of recent headlines predicts the next few days' returns of the company they are about. Each headline gets a score in [-1, 1] from a word list (or
from a ``score`` column of your own), the scores for a ticker add up and decay with a half-life, and a headline dated D is used from the trading day ``lag`` after D (the default of one respects
news that arrives after the close). Needs ``data/user/headlines.csv`` or the path you give; without headlines it refuses to run rather than invent a signal.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `path` = 'headlines.csv', `half_life` = 3.0, `lag` = 1, `scale` = 1.5

## Run it

```bash
quant backtest --model news_sentiment --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## Learn more

[Alpha-generating styles: long-term, short-term, news, outlook and corporate actions](../techniques/alpha-generating-styles.md)
