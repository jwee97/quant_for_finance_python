# Signals: what your rules say today, what changed, and a record you can trust

A backtest tells you what a rule would have done on prices it has already seen. The **Signals** tab does the other half of the job: it reads the rules on the last day of the data, every day if you like, writes down what each one said, and later scores those written-down signals against the prices that came after. That record starts empty, and that is the point: it can only ever contain signals that existed before their outcome did.

Nothing on this page is advice or an order. The app does not trade.

## Five minutes

1. In the **Backtest** tab choose your tickers, data source and start date.
2. Open **Signals**. Under *2. Follow something new* choose **Stocks and ETFs**, give it a name and press **Follow**. The follow keeps the ticker list it was made with.
3. Add the other markets you care about: **Futures** (the simulated roots, or the curves you downloaded in the Futures tab), **Crypto** (BTC and ETH on the platform's Deribit data), **Options** (underlyings as the Options tab names them).
4. Tick **Download new prices first** and press **Read the signals now**.
5. Read each card: the data line and any warning first, then what the rules agree on, every rule, what changed, and the record.

From a terminal the same book is read and written:

```
quant signals follow --market equity --name "My ETFs" --assets SPY QQQ TLT GLD
quant signals run --refresh
quant signals list
quant signals history --id <id>
```

## Reading a card

**The data line and the warnings.** Every card says the day its data ends and where it came from. Two warnings matter more than anything below them:

* *The data ends on …, N trading days ago: these are that day's signals, not today's.* Stocks, futures and options are called stale after three trading days without a new bar, crypto after two days.
* *These signals are read on simulated data.* The Futures tab's built-in roots and the Options tab's model chains are simulations. They show how the board reads; they are not a market.

The platform's own 15 ETF files are a fixed dataset that ends when it was built. With **Download new prices first** ticked, the tab reads those tickers from your data source (Yahoo Finance or iTick) like any other ticker, so the signals reach the last close. If the download fails the run still happens on the old files and says so.

**What the rules agree on.** For each asset: how many rules are long, short or without a view, and the view in words. *Long* or *short* means at least half the rules net out that way; *leaning* means a fifth; otherwise *mixed*.

**Every rule.** One row per rule, one column per asset: ▲ long, ▼ short, – no view, with the rule's *strength*. Strength is the rule's score divided by the average size of its own score over the last two years, so 1 is a typical position, the sign is the side, and anything within ±0.25 counts as no view. A rule that cannot be computed (too little history, one asset for a cross-sectional rule) is listed underneath with the reason.

**What changed.** Every rule and asset whose side differs from the snapshot of the previous data day. Running twice on the same day does not hide the day's changes.

**How past signals did.** The record, from the ledger only:

* A signal read at the close of day *d* is entered at the close of *d + 1* and held 1, 5 or 21 trading days. That is the same one-day lag the backtests use.
* Each cell shows the number of signals scored, the share that were right, the average result, and in brackets how far that is above simply holding the asset over the same days. A rule earns its keep only by beating that.
* A signal whose horizon has not passed is not counted. One snapshot counts per data day (the first one issued).
* A snapshot read on data that was already old, so that the close it would have been entered at had passed before it was written down, is left out. Those prices existed when the signal was issued, and a record that counted them would be a backtest.
* Signals a day apart overlap when held longer than a day, so the count overstates the number of independent bets. Weeks of record are an anecdote; it takes many months before the numbers mean much.
* Option signals are not scored: a volatility position's result needs a history of option prices, and a chain is one day's snapshot.

## The rules

| Market | Rules |
|---|---|
| Stocks and ETFs | Six trend rules (one year, Carver's six speeds, t-statistic, three horizons, moving averages, adaptive), momentum against the other tickers, momentum net of the market, one-month reversal, mean reversion, low volatility, volatility-managed hold |
| Futures | The Futures tab's board: the trend rules, carry (through time and within the class), basis momentum, trend and carry together, skewness, a variance-ratio switch |
| Crypto | The Crypto tab's board: the trend rules, directional movement, the Mayer multiple, volatility-managed hold, funding carry, basis, stablecoin flow. Carry rules are scored on the coin's long-spot, short-perpetual index |
| Options | Three volatility rules of thumb: implied volatility in its one-year range (above 80% short volatility, below 20% long), implied against realised (below 0.9 times long, above 1.3 times short), an inverted term structure (long). The side is the side of *volatility* |

The rules are the platform's registered strategies with their default settings. Whether one is worth trading is what the Backtest tab, the Tune tab and this page's record are for.

## Every day

Under *3. Every day* switch on **Read the signals once a day**, set the time (this computer's clock) and save. While the app is running it starts one run a day at that time (weekdays only unless you untick it), downloading new prices first. A run that was due while the app was closed starts when the app is next opened that day.

**Alerts.** Give an address that accepts a JSON message (Slack, Discord and most chat tools provide a "webhook" address) and each daily run sends what changed and any data warning there. The address must start with `https://`, is stored on this computer only (`data/user/signals/settings.json`, which is not in the repository), and is never sent back to the page. Without an address nothing is sent anywhere; the alerts are kept in the book and listed on the page. **Send a test alert** checks the address.

On a computer that is not always on, use its own scheduler instead (cron, Task Scheduler) with `quant signals run --refresh`. It writes to the same ledger.

## Where it is kept

Everything is in `data/user/signals/` on the app's computer: `follows.json`, `ledger.jsonl` (every snapshot, one a line, only ever appended to), `latest.json`, `settings.json` and `alerts.jsonl`. Stopping a follow keeps its snapshots. Back the folder up if the record matters to you; on a host that resets its disk (a Replit workspace that is rebuilt) the ledger goes with it.

## What is not here

* No orders, no broker, no position sizes. The page says what rules say, not how much to hold.
* No costs in the record: a result is the asset's raw return over the holding period.
* No custom rules or settings yet: a follow uses each rule's defaults.
* No intraday data: signals are read on daily closes.
