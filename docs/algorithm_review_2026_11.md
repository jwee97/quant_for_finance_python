# Review of the tabs' algorithms for personal use (November 2026)

What this is: a reading of how each tab turns data into a result, with one question in mind: *would a person trading their own account get what the page shows?* It says what was checked, what was changed, and what is still not modelled. It is not a proof that any strategy makes money.

How it was done: the code path of each tab's main run was read from the endpoint to the engine, the defaults were listed, and where a gap could be measured it was (a run with and without the missing cost). The tests named are the evidence for each change.

## What every backtest shares

The Backtest, Futures, Crypto, Machine learning and Fundamentals backtests all go through one engine (`src/backtest/engine.py`). So these hold for all five:

| Property | State |
|---|---|
| A signal computed at one close is traded at the next (`signal_lag` 1) and earns from the close after | In place, and tested by replacing the future with noise |
| Positions drift with prices between rebalances, so turnover is what a real book would trade | In place |
| A cost on every dollar traded | In place. The rate is per trade (one way), not per round trip: `config/backtest.yaml` calls it "round-trip" in a comment, which is wrong; the engine charges it on each change of position |
| An account that loses everything stops | In place |
| A cost of *holding* a position, whichever side | **Added in this review** (`holding_cost`), used for futures rolls |

## Tab by tab

### Backtest

Sound: the lag, the drift, per-ticker costs for the platform's ETFs (3 to 12 basis points), an optional market-impact cost when you give the assets under management, the count of trials that deflates the Sharpe ratio.

Changed: **After your broker** on the Earnings tab (fee per order, borrow fee on shorts, interest on leverage), added just before this review (`tests/test_account_costs.py`). Before it, a long-short or leveraged strategy paid nothing for its shorts or its borrowed money.

Still not modelled: whole shares, taxes, dividends withheld, a margin call, a ticker you add gets a flat 10 basis points whatever its real spread.

### Futures

Sound: strategies run on the excess return of holding the nearest contract and rolling it, which is the return a futures position earns (no interest on the notional is invented); real curves are used only on days the file shows the front of the market.

Changed: **the cost of rolling**. A futures position has to be rolled every contract month, and each roll pays the bid-ask of the calendar spread and two commissions. The backtest charged only for changes of position, so a book that held crude oil all year paid nothing for twelve rolls. It now charges, on the size held and on either side, each root's own cost: its spread in ticks times the tick value, plus two commissions, per contract, for every contract month, as a share of the contract's value. On the simulated roots that is about 0.02% a year for the S&P 500 contract, 0.4% for crude oil and 1.8% for natural gas. The result lists it per root (`roll_costs`) and the notes say it is included (`tests/test_holding_cost.py`).

Still not modelled: margin and the interest earned on it, limit moves, the capacity of a market. The built-in roots are simulations; only curves you download or supply are a market. Contract terms (tick, commission, margins) are illustrative: check your broker's.

### Crypto

Sound: fees plus slippage on every trade (7 basis points one way by default; a small account on a retail exchange can pay several times that, so set your own); the carry strategies trade an index that is long spot and short the perpetual and collects the funding actually paid.

Not changed, and worth knowing:

* A directional strategy that goes **short** is scored on the spot return alone. In practice a short is held through the perpetual and receives or pays funding. Funding has been positive most of the time, so this understates a short's result more often than it overstates it, but it is not what you would get.
* Marks are weekdays only: the weekend's move is folded into Monday. Sharpe ratios are annualised on 252 days to match the other tabs.
* Only BTC and ETH have data deep enough to backtest.

Set the fees to your own exchange's before reading a result.

### Options

Sound: orders fill a day after the signal, a buy pays the full half-spread above the mid and a sell receives the full half-spread below it, a commission is paid per contract (0.65 by default), positions are settled at expiry.

The limit that matters is the data, not the algorithm: the strategy backtests run on a **simulated option market** that pays the variance premium it was built with. They show that the machinery books costs, hedges and settlement correctly. They cannot show that selling options is profitable. A real chain is one day's snapshot, so nothing on this tab is a backtest on real option prices.

Not modelled: early assignment of short American options, margin on short options, pin risk at expiry.

### Machine learning

Sound: features are tested for look-ahead, training is walk-forward with a purge and an embargo, the backtest uses the shared engine with its lag, costs are 5 basis points commission plus 2 slippage one way, and the count of experiments feeds the deflated Sharpe ratio. A *realistic execution* re-run (orders, fills, participation limits) exists for any experiment.

Not changed: the realistic execution is opt-in; the default is the flat cost. For a strategy you mean to trade, run the realistic re-run and read that one. Borrow fees and margin interest are not in either.

### Forecast

Sound, and nothing to change for realism: every method is scored walk-forward on forecasts made with only the prices before each date, against the random walk, with a correction for the number of methods tried. A forecast is a distribution, not a trade, so there are no costs to model.

Limit: the future dates skip weekends but not exchange holidays.

### Fundamentals

Sound: figures enter a study only from the date they were public (filing date plus a lag, estimated and flagged when the date is unknown), restatements are kept as vintages, searches count their trials and hold back a locked test, and a backtest of a signal goes through the shared engine under position, sector, beta, volume and turnover limits.

Limits: the demo database is simulated; real data depends on your provider's coverage; survivorship is only as good as the list of companies you load (companies that were delisted and are not in your database bias every study upward).

## What to do before trading anything from this app

1. Set the costs to your own: the fee per order and the borrow and margin rates on the Backtest tab, the fees on the Crypto tab, the commission on the Options tab.
2. Prefer results on real data to results on simulated data; every page says which it is.
3. Follow the strategy on the Signals tab and let its record and paper portfolio build before committing money. A backtest is the past; the ledger is the only out-of-sample evidence the app can give you.
