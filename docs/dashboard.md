# Using the dashboard

```bash
quant serve            # opens http://127.0.0.1:8765 ; --port, --no-browser, --host are available
```

Everything runs on your machine. The server listens on localhost only, and every request needs a token that exists only in the page it served.

The server loads its code when it starts. After you update the files (a `git pull`), stop it and start it again: the page is read from disk on every load, so it can be newer than the server behind it. When it is, a banner says so (*The app is older than this page* needs a restart before anything works; *The app was updated after it started* means some of its code changed since it started). The Execution and Cash flows tabs need a newer app than the Backtest tab does: with an older one the Backtest keeps working and those two tabs say they need a restart.

## Backtest

1. **Tickers.** The platform's 15 ETFs are loaded. Remove any, or add symbols from the **data source** you choose under *Download other tickers from*: Yahoo Finance (`NVDA`, `BTC-USD`, `^GSPC`, `EURUSD=X`) or iTick (see [Data sources](#data-sources-yahoo-finance-or-itick)). Each is checked immediately: days of history and first date, or why it failed. Give new tickers an asset class if you use macro strategies. *Use data from* sets the earliest date. **One ticker is enough** for a strategy that trades each ticker on its own signal (see [One ticker](#one-ticker)).
2. **Strategy.** Pick any of the library's strategies and set its parameters, or choose `expression` and type a formula. Add up to four and choose how to combine their forecasts. A strategy that compares tickers with each other says in the list how many it needs (`momentum  (needs 2+ tickers)`).
3. **Portfolio and costs.** Allocation (automatic, risk parity, hierarchical risk parity, Bayesian, DCC minimum variance, evolution-strategy policy, ...), a regime detector with an optional volatility target by regime, assets under management for market-impact costs, and the **starting capital** the earnings are shown on (blank means the assets under management if you gave them, else $100,000).
4. **Run.** Optionally run the look-ahead check (replaces the future with noise; slower). Slow strategies say so before you run them. If the chosen strategy needs more tickers than you have, Run is disabled and the reason is shown above it.

### Reading the result

- **Tiles:** net Sharpe ratio (the headline), CAGR, volatility, maximum drawdown, turnover and cost drag, and the deflated Sharpe ratio. Equal weight on the same tickers and dates sits under each tile for scale (for a single ticker the comparison is buy and hold).
- **Earnings tiles** (a second row, over the full backtest): net profit in dollars on your starting capital, ending value, average profit per month, profitable months, profit factor and the deepest fall from a peak in dollars.
- **Chart window** (All, 10y, 5y, 3y, 1y) rebases the charts and the first four tiles. Costs, turnover and the deflated Sharpe stay full-sample; the tables always show the full backtest.
- **Account value in dollars** from your starting capital, against equal weight and risk parity (buy and hold for one ticker), linear or log. **Drawdown. Rolling one-year Sharpe** shows whether the strategy only worked in some years. **Return by calendar year** (faded = partial year), **monthly heat map**, **gross and net exposure**, **average weight and return contribution by asset**.
- Every chart has a **Table** view with the numbers behind it, and a tooltip on hover.
- **Red flags** are fixed rules with their thresholds; a flag is a question, not a verdict. **Details** opens on **Earnings** and **Trades** (below), then holds benchmark tests, sub-periods, regimes, attribution, costs, forecast quality, the look-ahead check and the exact specification (YAML), which you can copy or download and re-run with `quant run spec.yaml`.

### Earnings

The Earnings tab turns the daily net returns into dollars on your starting capital, compounded day by day after trading costs:

- **Profit:** starting capital, ending value, net profit and its percentage, the average per year, month and trading day, and what the benchmark would have made.
- **Costs and leverage:** profit before costs, the trading costs paid (and their share of that profit), and the average and peak exposure as a multiple of your capital. A banner appears when the strategy held more than 1x; only trading costs are charged, not borrowing or short-selling fees.
- **Best and worst** day, month and year in dollars.
- **How often it made money:** profitable months, years and days, the average winning and losing month, the **profit factor** (what the winning periods made divided by what the losing ones lost, by month and by day) and the longest winning and losing runs.
- **Deepest fall from a peak** in dollars and percent, the peak and trough dates, when (if ever) the peak was regained and the longest time spent below a peak.
- A **profit by calendar year** chart and table (start and end value, profit, return and worst fall within the year; part years are marked) and a **profit by month** table.

These are what the historical rules would have earned, not a forecast, and a strategy chosen from many tries will look better than it is: the deflated Sharpe ratio is the check on that.

### Trades

The Trades tab shows the buying and selling behind the returns. A **round trip** is one stretch in which a ticker is held on the same side: it opens when the position appears (a purchase, or a short sale) and closes when the position goes to zero or flips. For each one: side, opening and closing dates, days held, the adjusted closing prices on those days, the price move (reversed for a short) and the profit in dollars on the account value of each day, before trading costs. Positions still open are listed first. The statistics cover all of them: round trips, win rate, average winner and loser, payoff ratio, profit factor, expected profit per trip, days held, the share of days with a position, long and short trips, and the count of buys and sells (changes of at least 0.25% of the account; smaller ones are drift). The latest 200 round trips and 100 orders are listed; the profits of all round trips add up to the profit before costs on the Earnings tab.

**Strategies that buy and sell, not just hold.** Most of the time-series rules in the library already enter and leave positions on their own signal, so a backtest of one shows buys, sells and round trips on the Trades tab, and each needs only one ticker. They come in three kinds. *Always in the market, long or short:* `ma_crossover`, `macd_trend`, `supertrend`, `kama_trend` and `tsmom` are long while the trend is up and short while it is down, so a reversal is a sale followed by a short. *Long, short or flat:* the breakout and filtered-trend rules `donchian`, `keltner_breakout`, `adx_trend`, `ichimoku_trend`, `trend_atr` and `vol_breakout`, and the range rules `range_reversion` and `ou_reversion`, sit in cash when they have no signal. *Long or flat (cash):* the pullback rules `rsi2`, `ibs_reversion`, `bollinger_reversion`, `stochastic_reversion`, `consecutive_down` and `vix_spike_reversion`, the calendar rules `turn_of_month` and `sell_in_may`, and the asset-allocation filters `faber_gtaa`, `dual_momentum` and `accelerating_dual_momentum`; `rsi2`, `ibs_reversion` and `bollinger_reversion` also have a *shorts* setting, off by default, that lets them sell short. Choose the *Independent sleeves* allocation so that each ticker follows its own signal. What the library does not have is exits that depend on the position itself: stop-losses, take-profits, trailing stops and a maximum holding time. Positions follow the signal and the rebalance schedule, filled at the close with the one-day lag.

### The trial counter

Every distinct idea you test on the same tickers counts as a trial, shown top right. The deflated Sharpe ratio is penalised for the number of trials, because the best of many tries looks good by luck. Repeating a run does not add a trial; changing the strategy, its parameters, the allocator or the regime does. Double-click the counter to reset it.

## Data sources: Yahoo Finance or iTick

*Download other tickers from* chooses where tickers that are not among the platform's 15 ETFs come from. The ETFs always come from the platform dataset. Each source keeps its own cache for a day (`data/user/prices` for Yahoo Finance, `data/user/prices_itick` for iTick), so the two never mix.

**Yahoo Finance** needs nothing: prices are adjusted for splits and dividends, and unaudited.

**iTick** needs an account and an API key. Its free plan allows 5 REST calls a minute (iTick's pricing page describes the free plan as time-limited; paid plans allow 120, 600 or 1,200). It serves daily bars for US, Hong Kong and China A-shares, other stock markets, forex, indices and crypto.

### Keeping the iTick key safe

Until a key exists, iTick appears in the menu as *iTick (not set up)* and the page shows a short set-up guide with the steps below, a button that copies the secret name, and a button that reloads the page once you have restarted the app. There is no box to type the key into: it never goes through the page.

The key lives in the environment of the machine that runs the server, and nowhere else:

- **On Replit:** Tools, then **Secrets** (under *Setup*), **New Secret**, key `ITICK_API_KEY`, your key as the value, **Add Secret**; then stop and **Run** again so the app starts with it. Replit stores secrets encrypted and hands them to the app as environment variables; visitors to the Repl's cover page see neither their names nor their values. Anyone you add to the Repl as a collaborator can see the values, so add only people you trust.
- **On your own computer:** `export ITICK_API_KEY=...`, or put `ITICK_API_KEY=...` in a file called `.env` in the project folder (git ignores it; `.env.example` shows the format).
- The server reads the key when a download starts and sends it in one request header to iTick (over HTTPS only; it never follows a redirect). The page can learn only whether a key exists. The key is never put in a response, a log line, an error message or a cache file, and a request from the page cannot supply or replace it. Every message that leaves the iTick client is scrubbed of the key.
- Do **not** paste the key into a chat, an issue or a commit, and do not put it in `.replit` (its `[env]` section is part of the repository), `replit.nix` or `config/*.yaml`. If a key ever leaks, create a new one in your iTick account and replace the secret.
- A Replit link that is not published is still a link: anyone who has it can use the dashboard, which spends your iTick calls (they cannot see the key). The 5-a-minute limit below caps that, but do not share the link.

### The rate limit

Every iTick call goes through one limiter shared by everything in the app:

- At most 5 calls in any minute (set `ITICK_CALLS_PER_MINUTE` if your plan allows more), counted as a sliding window with a second to spare. The times of recent calls are kept in `data/user/itick_calls.json`, so restarting the app does not hand out a fresh allowance.
- A call that would break the limit waits, and the run's progress line says so (`iTick NVDA (2 of 3): waiting 41 s for a free call (the plan allows 5 calls a minute)`). If iTick answers HTTP 429 the limiter pauses for as long as `Retry-After` says (a minute if it does not) and tries again, at most three tries a request.
- Checking a ticker in the page does **not** call iTick (a ticker not yet saved shows `↓`, with the number of calls it will take); the run downloads it. After that the data are cached for a day, and a later top-up asks only for the newest bars (one call a ticker).
- A first download costs about one call per 1,000 daily bars (roughly four years) for each ticker, plus one more call that finds the end of a short history. A run needing more than 60 calls is refused up front with its cost (`ITICK_MAX_CALLS_PER_RUN`); a ticker stops at 12 pages (`ITICK_MAX_PAGES`) and says so; `ITICK_PAGE_SIZE` changes the bars asked for per call. Without a *Use data from* date iTick downloads from January 1 ten years back (about three or four calls a ticker) and Yahoo Finance from 2005; an earlier date costs about one more call for every four years.
- **Test the iTick connection** (or `quant itick-test` on the command line) spends one call on ten daily bars of AAPL and reports what came back: whether the key and host work and whether the bars are daily. It never prints the key.

### Writing symbols for iTick

You type the Yahoo-style symbol and the app translates it:

| You type | iTick instrument |
|---|---|
| `AAPL`, `BRK.B` | US stock |
| `0700.HK` | Hong Kong stock 700 |
| `600519.SS`, `000001.SZ` | Shanghai, Shenzhen A-shares |
| `7203.T`, `VOD.L`, `SAP.DE`, `D05.SI`, `2330.TW`, `RELIANCE.NS` | Tokyo, London, Germany, Singapore, Taiwan, India |
| `BTC-USD` (or `BTC-USDT`) | crypto, as `BTCUSDT` |
| `EURUSD=X` | forex |
| `^SPX` | index `SPX` (use the code from iTick's symbol list) |

### What to know before trusting iTick data

- **Not verified against the live service here.** The client follows iTick's published documentation, but its author had no account to test with. It therefore checks what comes back: the bars must be about a day apart (the documentation and community code disagree about which interval code is daily; the default is 8, and `ITICK_DAILY_KTYPE` overrides it), the answers must have the expected fields, and a history that is shorter than asked is reported. Press **Test the iTick connection** before the first real run.
- **Adjustments are not documented.** iTick's candle documentation does not say whether prices are adjusted for splits or dividends, and it offers split ratios separately. Prices are used as delivered, so a stock split can appear as a one-day crash and dividends are missing from the returns. A one-day price move that matches a common split ratio is flagged with a warning on the ticker and in the result; nothing is changed. Mixing iTick tickers with the platform's adjusted ETFs mixes bases, and for dividend payers iTick returns are price returns. If that matters, use Yahoo Finance for those tickers.
- Today's bar is never used: it is still being drawn. If a refresh fails, the saved copy is used and the ticker carries a note saying so.
- The depth of history the free plan returns is not documented; a short history is reported, and the strategies need at least 300 days.

## One ticker

A strategy needs only as many tickers as its idea needs:

- **One ticker is enough** for rules that look at each ticker on its own: the trend, breakout and moving-average rules, RSI(2) and the other pullback rules, the calendar rules, `dual_momentum`, `faber_gtaa` and the model portfolios. In the strategy list they carry no note. For the trend and pullback rules choose *Independent sleeves* under Portfolio (the page does so for you) so each ticker trades on its own signal.
- **Two or more** for rules that rank tickers against each other: cross-sectional momentum, value, low volatility, short-term reversal, relative strength, pairs and the machine-learning rankers. With one ticker every rank is the same and the book would be empty, so Run is disabled and the reason is shown; the server refuses the same request.
- **More than two** for a few: `bab` four, `sparse_basket` and `bvar_lead_lag` three, `pca_residual` its components plus one. Some allocations need several tickers to diversify between too: the static books (risk parity, hierarchical risk parity, mean-CVaR, mean-variance) and the Bayesian and evolution-strategy allocators need two; minimum variance, maximum diversification, Kelly, Black-Litterman and the DCC minimum variance need three.
- With one ticker the benchmark is **buy and hold**; with fewer than five the risk-parity benchmark (which needs five assets) is left out.
- One ticker concentrates everything in it: the *Independent sleeves* allocation gives it the whole account times its signal, and a signal larger than 1 means leverage. The Earnings tab shows the average and peak exposure.

## Compare

Every run of the session, with up to six overlaid, rebased to 100 at the latest common start. A run keeps its colour whatever you tick.

## Strategy builder

The builder has the formula editor (check against your tickers, function reference, examples) and the Python template; see [How to add a strategy](how_to_add_a_strategy.md).

## Execution

The Execution tab runs the simulators behind the [execution algorithms](techniques/execution-algorithms.md) from a form, with no code. Nothing is downloaded and nothing is sent to a broker: the market is made from the numbers you type (shares, average daily volume, price, daily volatility, spread, slices in the day). Each mode keeps its form and its last result while you use other tabs, and each result has a *Read the guide* button.

- **Compare algorithms.** Tick up to eight of TWAP, VWAP, POV, arrival price, implementation shortfall, AIM or PIM on VWAP, target cost and liquidity seeking, or type others by name (`pov:rate=0.15`, `exp_trade:kappa=3`; a tactic goes first: `aim+vwap`). Choose the trading style (aggressive, working order or passive), the market (normal, trending, mean-reverting or a crisis) and how many days to simulate. Every algorithm trades the same simulated days. The result is a table of the **shortfall** (the cost of the whole order against the arrival price in basis points of its value: positive is a cost, negative a gain) with its standard error, its variation between days and its 5th and 95th percentiles; a split of the cost into spread, temporary and permanent impact, timing and fees; bars of the average cost and of the variation by algorithm; how each one spreads the order through the day against the market's own volume (per slice or cumulative); and a plot of cost against variation, where a point that is higher and further right than another is beaten by it.
- **Cost-risk frontier.** For every price of risk, the cheapest schedule for the order: the curve between paying market impact by trading fast and carrying price risk by trading slowly, with VWAP and TWAP marked. Hover for the risk aversion and how much of the order the schedule trades in its first slice.
- **Basket.** A demonstration list of 2 to 16 buys and sells (random sizes, volumes and volatilities, one common factor; the command line goes up to 30, which takes about 40 seconds) traded together: the joint schedule against stock by stock (the objective is cost plus risk aversion times risk squared; lower is better), the risk of the unexecuted list through the day, the **minimum trading risk quantity** (which share of each order to execute when only part of the list's value can go, so that what is left is as hedged as possible), the **maximum trading opportunity** (how much can go if only the buys are on offer before the rest is riskier than the original list) and the **program-block** split with the names that can be entered in dark pools without raising risk.
- **High-frequency.** Pair trading on a mean-reverting spread, an ETF against its basket by the delay of the order, rebate and liquidity trading with and without a view of the order flow, and an inventory-shaded market maker against symmetric quotes. They size how much edge a given delay, cost or half-life leaves; they do not show that such an edge exists.

These are research simulators on a stylised market: the impact parameters are illustrative values from the equity literature, there is no order book or queue, and the passive and dark fill rates are round numbers. Use them to compare methods and see which way a setting pushes, not to quote a cost. [Algorithmic trading coverage](algorithmic_trading.md) lists every item of the execution taxonomy and where it lives.

## Cash flows

The Cash flows tab follows a portfolio of real tickers through money coming in and going out, from the data source and start date you chose on the Backtest tab (tickers are downloaded as for a backtest, within the iTick limit if you use it). The *Holdings and weights* field takes `SPY=60, IEF=40` (any scale, `AAA=60%` also works, no weights means equal weight; it starts as `SPY=60, IEF=40` when both are in your list, else the first three tickers of the list). It is the portfolio's mix for every mode but the last, where you name the bond funds to hedge with and the funds that seek return.

- **Deposits and withdrawals.** A starting value, a monthly deposit or withdrawal (optionally growing every year), a scheduled rebalance, a dividend yield and what to do with dividends, a trading cost, and the number of months to spread the starting value and each deposit over (dollar-cost averaging). Tick up to five **policies** for how each flow is traded: pro rata, fix drift (buy what is below target, sell what is above), a full rebalance with every flow, hold cash until the next rebalance, or most liquid first. You get the money each policy ends with, the **money-weighted** return (what the investor earned given when the money arrived) next to the **time-weighted** return (what the portfolio did), turnover, costs and the distance from the target mix, and weekly charts of the account value and of the drift.
- **Spending rules.** A starting portfolio, a first-year spending rate and inflation, and the rules to compare: a fixed real amount (the 4% rule), a share of the current value, the endowment rule and guardrails. Thousands of futures are made by resampling the portfolio's own daily history in blocks, and you get the chance of running out, the chance that spending had to be cut, the spending and the wealth left, the highest starting rate whose chance of ruin stays under your limit, and a **fan chart** of the middle half and the central nine in ten of futures for any rule, for the wealth left and for the spending paid.
- **Redemption.** A fund of a given size, the share redeemed, a cash buffer and a limit on the share of each asset's daily dollar volume. Selling in proportion, selling the most liquid asset first and using the cash buffer are priced with the execution cost model: cost as a share of the fund (the measure that compares policies selling different amounts) and of what was sold, days to finish, and the distance from the mix afterwards.
- **Liabilities (LDI).** A plan that owes 100 a year for some years, discounted at the 10-year Treasury yield, with a starting funding ratio: no hedge, a glide path that hedges more as the plan gets funded, and a full hedge, held in the bond funds you name against the funds you name for return. The funding ratio and the hedge share are drawn weekly.

A bootstrap of the past cannot produce a future worse than the history it samples, the liability model uses one flat yield, and every result is one path of history: read the tables and charts as a way to see what a policy does, not as a forecast. The [cash-flow strategies guide](techniques/cash-flow-strategies.md) explains the arithmetic.

## Guides

The Guides tab reads the technique guides, the glossary, the strategy cards and the [algorithmic trading coverage](algorithmic_trading.md) page.

## What it will not do

- It does not tune anything for you or choose a strategy.
- Platform ETFs end on the dataset snapshot date; other tickers run to today. A mixed universe stops at the shorter end and never fills a missing price.
- Weekend and holiday gaps: for mixed universes it keeps weekdays on which at least 60% of the tickers traded.
- Yahoo Finance data are free and unaudited: corporate actions and delistings are not curated. iTick's adjustments are undocumented (see above). The platform's 15 ETFs are the versioned, cleaned dataset.
- There are no stop-loss, take-profit or trailing-stop exits and no borrowing or short-selling fees: positions follow the signal and the rebalance schedule, and only trading costs are charged.
- The Execution tab is a simulator: it places no orders and measures nothing about your own executions, and its costs come from illustrative impact parameters. The Cash flows tab follows historical prices, so it shows what a policy would have done, not what it will.
- Costs are the platform defaults (10 bps per unit traded, monthly rebalance); change them in `config/backtest.yaml` or the spec.
