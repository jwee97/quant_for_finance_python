# Using the dashboard

```bash
quant serve            # opens http://127.0.0.1:8765 ; --port, --no-browser, --host are available
```

Everything runs on your machine. The server listens on localhost only, and every request needs a token that exists only in the page it served.

## Backtest

1. **Tickers.** The platform's 15 ETFs are loaded. Remove any, or add Yahoo Finance symbols (`NVDA`, `BTC-USD`, `^GSPC`, `EURUSD=X`). Each is checked immediately: days of history and first date, or why it failed. Give new tickers an asset class if you use macro strategies. *Use data from* sets the earliest date.
2. **Strategy.** Pick any of the library's strategies and set its parameters, or choose `expression` and type a formula. Add up to four and choose how to combine their forecasts.
3. **Portfolio and costs.** Allocation (automatic, risk parity, hierarchical risk parity, Bayesian, DCC minimum variance, evolution-strategy policy, ...), a regime detector with an optional volatility target by regime, and assets under management for market-impact costs.
4. **Run.** Optionally run the look-ahead check (replaces the future with noise; slower). Slow strategies say so before you run them.

### Reading the result

- **Tiles:** net Sharpe ratio (the headline), CAGR, volatility, maximum drawdown, turnover and cost drag, and the deflated Sharpe ratio. Equal weight on the same tickers and dates sits under each tile for scale.
- **Chart window** (All, 10y, 5y, 3y, 1y) rebases the charts and the first four tiles. Costs, turnover and the deflated Sharpe stay full-sample; the tables always show the full backtest.
- **Growth of $1** against equal weight and risk parity (linear or log). **Drawdown. Rolling one-year Sharpe** shows whether the strategy only worked in some years. **Return by calendar year** (faded = partial year), **monthly heat map**, **gross and net exposure**, **average weight and return contribution by asset**.
- Every chart has a **Table** view with the numbers behind it, and a tooltip on hover.
- **Red flags** are fixed rules with their thresholds; a flag is a question, not a verdict. **Details** holds benchmark tests, sub-periods, regimes, attribution, costs, forecast quality, the look-ahead check and the exact specification (YAML), which you can copy or download and re-run with `quant run spec.yaml`.

### The trial counter

Every distinct idea you test on the same tickers counts as a trial, shown top right. The deflated Sharpe ratio is penalised for the number of trials, because the best of many tries looks good by luck. Repeating a run does not add a trial; changing the strategy, its parameters, the allocator or the regime does. Double-click the counter to reset it.

## Compare

Every run of the session, with up to six overlaid, rebased to 100 at the latest common start. A run keeps its colour whatever you tick.

## Strategy builder and guides

The builder has the formula editor (check against your tickers, function reference, examples) and the Python template; see [How to add a strategy](how_to_add_a_strategy.md). The Guides tab reads the technique guides, glossary and strategy cards.

## What it will not do

- It does not tune anything for you or choose a strategy.
- Platform ETFs end on the dataset snapshot date; other tickers run to today. A mixed universe stops at the shorter end and never fills a missing price.
- Weekend and holiday gaps: for mixed universes it keeps weekdays on which at least 60% of the tickers traded.
- Yahoo Finance data are free and unaudited: corporate actions and delistings are not curated. The platform's 15 ETFs are the versioned, cleaned dataset.
- Costs are the platform defaults (10 bps per unit traded, monthly rebalance); change them in `config/backtest.yaml` or the spec.
