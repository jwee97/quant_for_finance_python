---
title: "From a signal to a book: limits, costs and an honest benchmark"
slug: from-signal-to-constrained-book
difficulty: 3
chapter: Platform
prerequisites: [portfolio-theory-and-constrained-books, backtest-engine-and-costs]
stages: []
files: [src/fundamentals/bridge.py, src/equity/qp.py, src/backtest/engine.py, src/fundamentals/research.py]
figures: []
tests: [tests/test_fund_bridge.py]
models: []
---

# From a signal to a book: limits, costs and an honest benchmark

## In one sentence

A factor's information coefficient says the ranking has information; whether it makes money is a different question that depends on which companies are held, in what size, under what limits, at what cost, and against what alternative, so the signal is turned into a book decided from what was known on each date and run through the same backtest engine as every other strategy on the platform.

## The idea

**The signal** can be one library factor, a price-based signal, a library alpha, a formula, a combination by a rule, a model fitted walk-forward to several alphas, or a weighted composite of these. Higher means expect more.

**The book** is decided on every rebalance date from what was known on it. The companies that are investable (priced on the date, large enough, liquid enough) are ranked; the top *n* (or top percentile) are bought and, for a long-short book, the bottom *n* sold, each leg weighted equally, by how far the signal clears the cutoff, or by inverse volatility. Then the limits are applied: a position, a sector, an industry or a country limited as a share of the account; the net beta to the universe held inside a band; a name never held above a share of its own average daily dollar volume (so the book cannot outgrow the market's capacity to trade it); and a turnover limit.

**Meeting limits.** The nearest book that respects the limits is found in two steps: a linear programme (HiGHS) finds how much of each leg can be invested (the fill fraction *t*), then a quadratic programme finds the weights nearest the wanted ones inside the limits. If the limits cannot all be met, both legs are scaled down together so the book stays balanced, and the rest stays in cash; the report says how often. A date with too few eligible companies is held in cash for that date, never a smaller, riskier book.

**Turnover** is `max(Σ bought, Σ sold) / gross`, so building a whole book from nothing is 100%. A turnover limit trades only part of the way to the new book, which means the first book is built over several dates (and the cost of building it is charged on the first return day); hard limits (a position, a sector) are enforced again after the partial move.

**The run.** The weights, stamped on the decision date, go to the platform's own `BacktestEngine`. The book is struck `entry_lag` days after the decision (the same lag the factor tests use, so the two measure the same trade), drifts with prices between rebalances, and pays a linear cost on what it trades; a short pays a borrow fee on its notional; a company that stops trading is carried at its last price.

**The benchmark and the deflation.** The result is set against owning every eligible company equally with a daily rebalance, so *alpha* means return not explained by owning the universe. Its Sharpe ratio is deflated for the number of ideas tried to find the signal (taken from the library when the signal is a library alpha, or stated), and the book's performance is split by the dates the alpha was found on, so the locked block can be read apart from the training and validation blocks.

## Why it matters

A factor can have a real IC and be untradable: it may need positions bigger than a stock's volume allows, concentrate in one sector or country, turn over so fast that costs eat it, or sit entirely in the short leg of stocks that cannot be borrowed. A backtest that does not enforce limits reports a book nobody could hold. A benchmark of the wrong kind (cash, or an index with different weights) turns a market return into "alpha". And a Sharpe ratio chosen as the best of many tries is not the Sharpe ratio of the idea.

## How this repo uses it

`src/fundamentals/bridge.py` resolves the signal (`resolve`), decides the book (`build_book`, with `eligibility` and a limit projection that uses the quadratic solvers in `src/equity/qp.py`: an interior-point method up to a few hundred names and ADMM beyond), runs it (`engine_returns` through `src/backtest/engine.py`) and reports (`backtest`, `tear`). The **Alpha library → Backtest** view, the *Backtest this alpha* button on a study and an alpha's page, and `quant fundamentals backtest` all use it; the settings are the `PortfolioSpec` fields (long-only or long-short, names or percentile, weighting, leg exposure, the limits, account size, costs, borrow cost, dates). Every run is recorded as an experiment whose identifier includes the signal, the portfolio rules, the universe and the data version, so it can be shown and run again.

A backtest of a *Candidate* carries a notice that a good backtest is what a search is expected to produce; a *Rejected*, *Retired* or *Decaying* alpha carries its own. The notices are listed once each, naming the alpha, however many are in the blend.

## What we found

The tests give the book known truth: the names held are the top ones among those that may be bought; a long-short book has two separate legs of equal size; with too few eligible companies the date is cash and the report says why; **no limit is ever broken** and what cannot be invested stays in cash; the net beta stays in its band and a position never outgrows the name's liquidity; a turnover limit trades part of the way; a date is decided from what was known on it and not from prices that came later; **the engine earns what the factor tests say the same holdings earn**, period by period; costs and borrow are charged, including the cost of building the first book; a planted signal is found and noise is not; and the deflated Sharpe ratio falls with the number of ideas tried.

On the default demo world, `quant fundamentals backtest --demo --factor fcf_yield` (long only, the top 30, equal weights, 10 bps a side): 26.7% a year at a Sharpe ratio of 3.54, against 15.7% and 3.28 for owning everything; alpha 9.1% a year, beta 0.99, information ratio 1.8; turnover 2.6 times a year, a trading cost of 26 bps a year, and a breakeven one-way cost of 904 bps (the cost at which the net return would be zero). A net limit of 20% of the account in any one sector changed the book on 153 of the 167 rebalance dates and cost almost nothing here (26.4% a year, alpha 9.0%). A long-short book of the top and bottom 20 earned 20.5% a year net at a Sharpe ratio of 1.86 with a turnover of 6.0 times a year and trading costs of 60 bps a year. Read the *differences* from the benchmark and between books: simulated returns are far calmer than real ones (the benchmark's own Sharpe ratio is 3.3), so the levels mean nothing.

What has not been found: how any of this holds on real companies, with real borrow costs, real price impact and real index weights.

## Pitfalls

- **The benchmark is equal-weight, not cap-weight.** An alpha against it includes any tilt toward small companies, which is where an equal-weight universe earns its premium and where trading is dearest.
- **Costs are flat.** A number of basis points a side and a borrow rate, with no price impact except the limit on a position's share of daily trading value. Set that limit when positions are large against the volume.
- **Limits change the signal.** A sector cap that binds on most dates (153 of 167 here) means the book is as much the cap's choice as the signal's; the report says how often the limits bit, and a signal that only works unconstrained has not been shown to work.
- **The first book is expensive.** Building a whole portfolio from nothing is 100% turnover; the cost is charged, and a turnover limit spreads it over dates.
- **Survivorship and borrow.** Delisted companies are kept with their losses, but a short leg of delisting stocks assumes they could have been shorted at the stated fee.
- **A deflated Sharpe is only as honest as the count.** If the number of ideas tried is understated, so is the deflation; a library alpha brings the count of its own search, and a factor you chose after looking at others should carry a larger one.
- **A Candidate that backtests well is the expected outcome of a search**, not a result. Read the locked-block row, not the whole-sample line.

## Try it

```
quant fundamentals backtest --demo --factor fcf_yield
quant fundamentals backtest --demo --factor fcf_yield --max-sector 0.2
quant fundamentals backtest --demo --formula "div(fcf,market_cap)" --side long_short -n 20
```

```python
from src.fundamentals.service import FundamentalsService
from src.fundamentals.workbench import Workbench

wb = Workbench(service=FundamentalsService(".", db_path="data/user/fundamentals_demo.db"))      # made by `quant fundamentals simulate --demo`
ctx = wb.context({}, "M", 1)
result = wb.backtest(ctx, {"kind": "factor", "name": "fcf_yield"}, {"side": "long_only", "n": 30, "max_sector": 0.2})
print(result["summary"]["sharpe"], result["benchmark"]["sharpe"], result["limits"]["projected_dates"], result["costs"]["breakeven_bps"])
print(result["reading"])
```
