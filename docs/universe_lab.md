# Stock Universe Research Lab

Which stocks to hold, and whether a portfolio built from them survives honest out-of-sample testing. The **Stock Universe Explorer** tab, `POST /api/universe/explore` and `src/universelab/` answer one
question: *if I had followed this procedure (screening the names, choosing a construction method and a lookback on the data I had then), what would I have earned on the data that came next, after
costs?*

## What the universe is

A backtest that draws its stocks from **today's** list of index members has used the future: companies that failed, were acquired or left the index are not on the list. This platform does not ship a
point-in-time membership history and does not invent one. So there are two honest forms of eligibility (`src/universelab/membership.py`):

* **A listings table you supply** (`ticker,start,end,status`; `end` empty means still listed; status is active, delisted, failed, merged or suspended). Eligibility on a date is exactly what the table says.
  Coverage is `complete` only if you tick that the table includes the securities that left; otherwise it is `partial` and the survivorship limitation is shown beside every result.
* **No table.** Every ticker is eligible from its first price and the result says, in a warning above the numbers, that the universe was chosen with knowledge of who survived.

A security that leaves has no returns after its last day. The lab never invents a delisting return: a position held on that day earns zero from then on and is counted (`missing_return_days`).

## Screens (`screens.py`)

Computed only from the window of data that ends at the decision date: all names, liquid names (average daily dollar volume), the largest by dollar volume (a **size proxy**: shares outstanding are not in the
data), lowest volatility, highest 12-1 momentum, chosen sectors, chosen names. A volume screen on a universe with no volume is refused, not approximated.

## The walk (`walk.py`)

At each rebalance the decision reads only rows strictly before the last two days (data through day `d` is traded at the close of `d + 1` and earns from `d + 2`). Eligible names with a complete window are
screened, a construction method from the [portfolio lab](alphalab.md) (19 of them: equal weight, minimum variance, risk parity, HRP, HERC, NCO, CDaR, Black-Litterman, robust and others) builds weights,
weights are capped, drift with prices, and every change is charged at the configured cost. A method that fails falls back to equal weight and is counted.

## Discovery with nested validation (`discover.py`)

The screen, the method and the lookback are **one** selection process, not three:

1. every candidate (screen x method x lookback) is walked over the whole history once. Its return on a day depends only on data before that day, so a slice of it is what it would have earned then;
2. the out-of-sample period is cut into consecutive blocks. For each block the candidates are scored on the days before it (less an embargo), the best is chosen, and the block's return is that candidate's
   return on the block, which was not used to choose it;
3. the blocks are stitched into one out-of-sample series, compared with the benchmarks on the same days and **corrected for the number of candidates tried** (deflated Sharpe).

The whole-history table of candidates is shown as *in-sample context*. Its best row is a search result, never a finding. The grid is limited to 60 candidates: a larger search is a weaker claim.

## Robustness (`robust.py`, `study.py`)

Out-of-sample CAGR, Sharpe, maximum drawdown and expected shortfall; turnover and the annual cost drag; beta to the equal-weight benchmark; effective number of names and the largest group's share; constituent
stability between rebalances; the result without each of the three most-held names (stock-removal sensitivity); a block bootstrap of the Sharpe ratio (its spread, the chance it is positive, the chance it
beats equal weight); the result in rising, falling, calm and turbulent markets (the state on a day uses only earlier days); and a fixed equal-weight walk through the same engine with no screen and no search.

## Benchmarks

Equal weight of every eligible name with a return that day (daily rebalanced). Where volume exists, a dollar-volume-weighted benchmark, labelled a **proxy for cap-weight**. True cap-weight needs shares
outstanding, which the data does not have.

## Tests

On synthetic worlds with known answers: the walk uses only data known at each decision (the whole future is rewritten and nothing earlier changes); the last two days before a rebalance are never read; the
first trade is charged; a stock that is not yet a member cannot be held, and the same stock with no table is the survivorship trap that does pay; what is chosen for a block cannot depend on that block; with
no skill in the data the selected configuration does worse out of sample than it looked on the days it was chosen on; a planted low-volatility edge is found and pays out of sample.

## Limits

* The default universe is whatever tickers you give it. Without a membership table the hindsight warning applies to every result.
* Differences between configurations over a few years are usually inside the sampling error; the bootstrap and the deflated Sharpe show how much.
* Factor-constrained and Black-Litterman-with-views portfolios need inputs the page does not collect (views, factor exposures); they are available in the library (`src/portfolio`).
* This lab does not model liquidity capacity or market impact of the portfolio; the order-level execution model is separate (`execution.mode = realistic_cash_equity`).
