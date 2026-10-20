# Company fundamentals: studying factors, searching for alphas, and counting honestly

A company's accounts say something about what its shares may earn next, but only some of that is real, and most of what looks real in a backtest is not. The **Fundamentals** tab keeps the accounts of listed companies from around the world in a local database, tests ideas built from them (one factor at a time, or by searching thousands of formulas), keeps every idea it tried (the failures too), and puts each survivor through the tests that are meant to catch the ones that only look good: dates the search never saw, a count of everything tried, costs, and a check that no figure was used before it was public.

The one rule behind all of it: **an interesting factor is not a validated alpha.** A result found in the sample it is measured on is a lead. The page says so in every verdict, and the status of an alpha in the library moves from *Candidate* to *Validated* only when the evidence survives the correction for the number of ideas tried (or a person approves it, with the reason on the record).

What this chapter covers: five minutes with the demo world; setting up real data (and what it costs); how the tab decides what was known when; what the factors are; how a study is read; how the search works and what it keeps; putting alphas together; testing an alpha as a portfolio; what was measured; and what is not known.

## Five minutes, with the demo world

No account is needed for this. The demo is a *simulated* world of companies with a few alphas planted in it, so the machinery has something it should find and the page can show you what finding it looks like. It is clearly marked as simulated on every view, it lives in its own file, and nothing found in it says anything about real markets.

1. Open the **Fundamentals** tab. At the top of the left panel, *Data* says **Real data** (empty for now) and **Demo (simulated)**. Choose **Demo**.
2. Open **Data manager** and press **Make the simulated world**. A hundred and twenty companies in six countries over fourteen years take about 35 seconds on a four-core machine with nothing else running (forty companies take about ten). From a terminal: `quant fundamentals simulate --demo`.
3. **Overview** now shows what the database holds (companies, statements, prices, how clean they are, what the library holds).
4. **Factor lab** → *Study this factor*. The first factor is free-cash-flow yield. The result is a verdict in words, six numbers, a dozen charts (each with a table behind it) and breakdowns by country, sector and market regime. Open **Look-ahead audit** to see the same study run twice, once as it should be and once as if every figure were known at its period end. From a terminal: `quant fundamentals evaluate fcf_yield --demo --lookahead`.
5. **Alpha discovery** → *Start the search*. Eighty formulas take about fifteen seconds. Each is counted; most fall at the first tests. The leaderboard shows the best, none of them validated.
6. **Alpha library** shows the survivors (as *Candidates*) and, under **Graveyard**, every idea that failed and why. Tick two or three and open **Correlation**, **Combine**, **Model** or **Backtest**.
7. **Alpha monitor** asks whether each alpha still works on the data now in the database.

## Setting up real data

The real database is filled from [OpenFilings](https://openfilings.org), a hosted service that returns each company's annual statements *as filed*, with the day each was filed, for about twenty-five markets. It is a deliberate choice: figures that were later restated are not what an investor could have seen, and the filing date is what a point-in-time study needs.

1. **Get a key.** Sign in at openfilings.org and create an API key in the dashboard (the examples in their documentation show keys that begin `of_`; the application does not depend on that).
2. **Put it in a secret, never in the page.** On Replit: *Tools → Secrets → New Secret*, name `OPENFILINGS_API_KEY_1`, paste the key as the value, press *Add Secret*, then **Stop** and **Run** (an app only sees the secrets that existed when it started). Outside Replit, set the environment variable the same way. Nothing in the page or the command line accepts a key, the key is sent in a header and never in an address, it is never printed or logged, and the database holds only a salted one-way hash of it (to tell keys apart in the usage counts).
3. **Test the connection.** *Data manager → Test the connection* spends one credit and tells the app which plan the account is on, which sets the allowance the quota manager enforces.
4. **Add companies** (free): `us:AAPL jp:7203 uk:HSBA`, or `quant fundamentals add us:AAPL jp:7203`. Only the ticker and market are stored.
5. **Look at the plan, then fetch within a budget.** *Show the plan* says what is missing, what is already fresh, what it would cost and how many companies fit in today's allowance; `quant fundamentals plan`. *Fetch within the budget* will not start without a number of credits it may spend (`quant fundamentals sync --budget 100`), stops by itself when the budget or the allowance is reached, and can be stopped; what was fetched is kept.
6. **Load prices.** Yahoo Finance supplies daily prices (unadjusted and adjusted) and exchange rates; a market's usual suffix is tried first (`7203.T`, `HSBA.L`), London prices quoted in pence are divided by 100, and the symbol of a company can be set by hand. Companies that have delisted usually have no history there, which leaves them out of a study; the research context says how many.

### What it costs

Every call is metered in **credits** against a daily and a monthly allowance. A lookup or search costs 1, a company's statements 10, so a new company costs about **11 credits**. The free plan is 100 credits a day and 1,000 a month: about nine companies a day, about ninety a month. A universe of 500 companies is about 5,500 credits, which on the free plan is more than five months. The application therefore keeps everything it downloads (the raw answer too, so a rule can change without asking again), asks again only for what is missing or has had a new filing since, and never refetches a company the provider said it did not know for two months. A failed call is refunded by the provider and costs nothing here.

### Several keys, and what they cannot do

The page lists five key slots (`OPENFILINGS_API_KEY_1` to `_5`; a plain `OPENFILINGS_API_KEY`, a list in `OPENFILINGS_API_KEYS` and numbers up to 20 are read too), because several accounts might seem a way to five times the allowance. It is not, and the application does not pretend otherwise. OpenFilings' usage policy does not allow rotating accounts or API keys to get past its quotas, and several keys of one account share one allowance. So the **first working key does all the work**; a later key is a standby that takes over only if the first is rejected as invalid (revoked, say); when the allowance is spent the run stops, says when it resets, and waits. There is no switch that rotates keys. More credits come from a larger plan or prepaid top-ups (which never reset). The Data manager shows each slot, its role (*active* or *standby*), its state and what it has used.

### What is not supported yet

- **Annual statements only.** The provider's documentation does not list the fiscal-period values for quarters or half-years, and a guess costs ten credits a try. A request for quarters is refused before a credit is spent. The trailing-twelve-month machinery exists and is tested on simulated quarterly data.
- **The sign of cash outflows is not documented**, so capital expenditure, dividends, buybacks, acquisitions and interest are read as magnitudes.
- Some filings in Japan and Korea are collected on demand: the first request may answer "still backfilling", which is treated as *ask again tomorrow*, not as a failure.
- **The adapter was written from the published documentation and tested against stand-in answers of the documented shape. It has not been run against a live key.** The first run with a key is the real test: fetch three or four companies and read the Data quality card.

## Two databases, never mixed

`data/user/fundamentals.db` holds the real data; `data/user/fundamentals_demo.db` holds the simulated world. The page and every command choose one (**Real data** or **Demo**, `--demo`), the demo never calls a provider, a simulated world is refused in a database that holds real companies, and an alpha found in one never reaches the library of the other. Both files are ignored by git.

## When a figure counts as known

This is the part most studies get wrong, so the application enforces it in one place (`src/fundamentals/pit.py`) and every panel is built through it.

- A figure is available on **the day it became public plus a lag you choose** (default one day, because a filing may come after the close), never on the last day of the period it describes: `available = filing date + information lag`. A restatement is a new version that is public from the day it was first seen, not from the original filing date; the original is kept beside it.
- When no filing date is known the application assumes a conservative lag after the period end for that kind of period, **flags the period as estimated**, and says how many periods needed that. A study can leave such periods out altogether.
- A study runs on a grid of rebalance dates (weekly, monthly or quarterly, the last trading day of the period). A figure appears on the first date on or after it became available and holds until the next filing; one that is too old expires instead of being carried for ever.
- The trade is made **a day after the signal** (default), never at the close the signal was computed from. Forward returns carry a company that delists before the horizon ends to its last price, so the losers stay in the sample.
- Anything learned from realised returns (a combination weight, a model fit, the sign of a formula) is used on a date only if the return window it learned from had ended by then: the *settled lag*, read off the grid, not assumed.
- **The look-ahead audit** runs a factor study twice, once as above and once as if every figure were known at its period end, and says whether the naive version flatters the result. On a simulated world in which net income is known a quarter early, the figure "predicts" returns when it is known early (IC above 0.04) and does not when it is known on time (below 0.01), and the audit says so.

## Missing data, bad data, and survivorship

- **A missing figure is never a zero.** A ratio is computed only when every figure it uses is present; a denominator that cannot meaningfully divide (zero, negative, missing) makes the ratio missing, not huge; growth needs a positive base a year earlier found by date, not by row. A company with no value for a factor has no view on it, and says so.
- Every stored figure carries a **quality flag** (`ok`, `suspicious_jump`, `identity_error`, ...) and every period says whether its filing date is real or estimated. Flagged figures are treated as missing in research, never repaired silently; the Overview and Data manager count them, and a company's page lists the flagged figures.
- **Survivorship.** Companies that delisted are kept with their losses, and every study reports how many it has. The application never claims a sample is free of survivorship bias unless the securities were declared to be the historical constituents, which a database built from today's tickers is not.
- Derived metrics are persisted with a **calculation version**; a change to a formula bumps it, so old and new numbers are never mixed.

## The factors

Eighty-four factors in thirteen families, each defined in one place with its direction (higher or lower is better, from theory), the figures it reads and a sentence on what it is: **value** (earnings, free-cash-flow, sales, book, EBIT and EBITDA yields, dividend and buyback yield, shareholder yield, a composite), **quality**, **profitability** (margins, ROE, ROA, ROIC), **growth** (revenue, profit, EPS, cash flow, three-year revenue growth, acceleration), **investment** (asset growth, capex, share issuance), **leverage**, **liquidity**, **efficiency** (turnovers, the cash conversion cycle), **earnings quality** (accruals, cash conversion, the accounting anomaly score), **fundamental momentum** (the change and acceleration of margins, returns and leverage), **price momentum**, **risk**, and **size**. The composites are the direction-signed average of their standardised parts; a missing part is not a zero.

A factor passes through a **pipeline** of steps, written as data so a study can be stored, shown and run again: winsorise each date's tails (default 1%), neutralise (remove what the factor shares with a sector, industry, country, the market, size, beta or volatility by a regression on each date), then put the companies on a common scale (z-score, rank or percentile). Every step uses only the cross-section of its own date or the past of its own company.

**On a company's page** (Company explorer → Open): every figure with the day it became public, the ratios built from them, the five-step **DuPont** split of the change in return on equity (tax burden, interest burden, operating margin, asset turnover, equity multiplier) whose contributions add up exactly to the change, the **Piotroski F-score** (nine yes-or-no signs of improving health) and an **accounting anomaly score** of the Beneish kind. The two scores are screens. The anomaly score was fitted on United States manufacturers of the 1980s and early 1990s; a high value means *look closer*, and the application never calls anything fraud on its strength. See [accounting analytics](techniques/accounting-analytics-dupont-piotroski-accruals.md).

## Studying one factor

The **Factor lab** takes a library factor or a formula of your own (`div(fcf,market_cap)`, `delta(roic,12)`, `sector_rank(div(operating_income,total_assets))`), a universe, a way of rebalancing and a way of testing, and returns:

| What | The question it answers |
|---|---|
| **Mean rank IC and its t-statistic** | Across companies, does the factor order the next period's returns? The t-statistic allows for the overlap of the return windows (Newey-West). |
| **Return by bucket** and **monotonicity** | Do the buckets rise in order, or does the whole effect sit in one tail? |
| **Long-short portfolio**, gross and net | Does top-minus-bottom pay after trading costs (default 10 bps one way) and the cost of borrowing the short side (default 50 bps a year)? Sharpe, return, worst fall, turnover. |
| **Decay** | Is the signal still there at three or six months, or gone in a week? It sets how slowly it can be traded. |
| **By year, half, country, sector, region, regime** | Is it steady or the work of a few dates, and does it hold everywhere? |

The verdict is a sentence with a symbol (✓, ! or ✗), never a colour alone: *Strong in this sample* needs |t| of at least 3 and a net Sharpe above 0.5; *Some evidence* needs |t| of at least 2; below that, *No clear evidence*. None of them says validated, and each says so. **Scan every factor** sets all eighty-four side by side under one set of conditions, with a warning that with that many a few show |t| above 2 by luck alone: it is a map of where to look, not a list of alphas. Two factors from the same family are often one idea (see below). See [factor studies](techniques/factor-studies-ic-quantiles-and-decay.md) and [point-in-time fundamentals](techniques/point-in-time-fundamentals.md).

## Searching for alphas

**Alpha discovery** builds formulas as expression trees (a statement figure, a derived ratio, `add sub mul div`, `neg abs log`, `delta growth accel rolling_mean rolling_std` over a number of periods, `rank sector_rank industry_rank zscore`), from the classic formulas of the literature, your own, and random and bred ones, and tests each in stages, cheapest first:

| Stage | A formula must |
|---|---|
| 0 sanity | compute, cover at least 40% of investable companies, take more than five distinct values on a typical date, and not have been tried before under the same conditions |
| 1 fast IC | have a rank IC with next month's return, on the **training** dates, with |t| of at least 1.65 |
| 2 stability | be positive on 55% of dates and in both halves, and keep its t-statistic above 1 without its best year |
| 3 quantiles | have buckets that rise in order |
| 4 long-short | earn gross |
| 5 costs | still earn after costs and borrowing, at a turnover that can be traded (at most 70% a rebalance) |
| 6 validation | keep its sign on a second block of dates that stages 1 to 5 never used |
| 7 locked test | (the finalists only, chosen from training and validation) be tested **once** on the last block, which nothing before has looked at |

The dates are cut into three consecutive blocks (default 55%, 20%, 25%) with a gap as wide as the forecast horizon between them, so no return window sits in two blocks. The sign of a formula (whether high or low should earn) is learned on the training block and held fixed afterwards, which makes the test two-sided; the IC figures shown are in that direction, and the library says *higher is better* or *lower is better*.

What keeps the count honest:

- **Every formula that reaches stage 1 is a test and is counted.** The training p-values of all of them go through Benjamini-Hochberg to give each finalist a false-discovery **q-value**; its Sharpe ratio is deflated for the number tried (Bailey and López de Prado); and, with enough survivors, the **probability of backtest overfitting** is estimated by combinatorially symmetric cross-validation. A finalist must also have a q-value of at most 0.25 over everything tried in the run.
- **The same idea is never paid for twice.** A rank, z-score or sign on the outside of a formula does not change which companies it prefers, so `neg(rank(x))` and `x` are one idea (a *signal key*); a formula more than 90% correlated with one already accepted in the run is the same idea too.
- **The locked dates are counted.** Every look at them is recorded (`oos_looks`), and an alpha says how many looks had been made before its own.
- **Nothing is deleted.** A rejected formula goes to the **graveyard** with the stage it failed at and the reason, under a unique key of formula and conditions; only the metadata of a survivor goes to the library. Ideas tried = alphas in the library + ideas in the graveyard, always.
- **A run can be stopped and resumed exactly.** Every candidate is written the moment it is judged and the stream of formulas is a function of the seed and the position. A finished run cannot be reopened: its locked test has been looked at, and looking again would spend it.
- A complexity penalty in the fitness (information coefficient, stability, Sharpe, monotonicity and unlikeness to the others, less turnover and complexity) prefers the simpler formula when the evidence is equal.

### What the library calls an alpha

| Status | Meaning |
|---|---|
| **Candidate** | Passed every screen, including dates it was not chosen on. Not trusted. This is the best a search can give. |
| **Validated** | A candidate whose locked-dates t-statistic is at least 1.0 *and* whose evidence survives the correction for the number tried (q-value at most 0.10, or deflated-Sharpe probability at least 0.90). Or one a person approved, with the reason recorded and marked *forced*. |
| **Production** | A validated alpha someone decided to use. |
| **Decaying** | A validated or production alpha whose recent behaviour has weakened (set by the monitor). |
| **Retired**, **Rejected** | No longer used; turned down. Neither can be moved again. |

Moves follow a fixed table (a Candidate can become Validated, Rejected or Retired; Production goes to Decaying, Retired or back to Validated), each needs a reason and is logged. The bars for *Validated* are constants in `src/fundamentals/library.py`; the locked-dates bar in particular is low, because a single look at a short block cannot support more, so treat *Validated* as "not refuted yet", never as "will work". See [automated alpha discovery](techniques/automated-alpha-discovery.md).

### The monitor

An alpha that worked can stop working. The monitor recomputes every alpha that is not retired or rejected on the data now in the database (in the direction it was found) and compares its recent information coefficient (the last twelve months, by default) with the one it earned when it was found. **Healthy**: recent IC at least half the original. **Weakening**: below half. **Decaying**: zero or negative over a year, or under a quarter of the original over two. **Broken**: reversed with evidence (a t-statistic of −2 against the original direction), or covering fewer than 30% of the companies it once covered. A validated or production alpha judged Decaying or Broken is moved to Decaying automatically; nothing is deleted. Fetch new statements and prices first if you want it to see recent months.

## Putting alphas together

Two alphas that rank the companies the same way are one idea, and five copies of one idea should not outvote one different idea. **Alpha library → Correlation** shows the average, over dates, of the rank correlation between each pair (on the companies both cover), the heatmap in the order that puts alike alphas side by side, the clusters (average linkage on 1 − |correlation|), the pairs that are 80% or more alike, and the **effective number of independent ideas** (from the eigenvalues of the correlation matrix: 1 for identical alphas, the number of alphas for unrelated ones).

**Combine** weights the alphas by a rule that uses only what was known on each date: equal weights, their recent IC, their recent IC information ratio, the maximum information ratio (allowing for how their ICs move together), or equal weight to each cluster of alike alphas. **Model** fits one of ridge, lasso, elastic net, random forest, extra trees, gradient boosting, XGBoost or LightGBM *walk-forward* (no training row whose return was still unfolding, with an extra gap if you ask), reports what it relied on (coefficients, the model's own importance, permutation importance and, for the two boosting libraries, SHAP values), and compares the result with the plain rules over the same dates with a paired test. A model that cannot beat a plain average is not earning its complexity; fed pure noise, it is not credited with having beaten the rules. A missing alpha is *no view* (zero), never a bad value. See [combining alphas](techniques/combining-alphas-and-model-combination.md).

## Backtest this alpha

**Backtest** turns a signal (a library alpha, a factor, a formula, a blend, a model) into a portfolio under limits and runs it through the platform's own backtest engine, net of costs, against owning every eligible company equally with a daily rebalance. You choose long-only or long-short, how many names or what percentile each leg holds, the weights (equal, by signal, by inverse volatility), limits (largest position, most net exposure to one sector, industry or country, net beta band, smallest market value and traded value, the largest share of a company's daily trading value one position may be, the most traded at one rebalance), costs and dates.

The limits are met by the nearest book that respects them (a linear programme for how far the book may move toward the wanted one, then a quadratic projection). When they cannot all be met, both legs are scaled down together and the rest stays in cash, and the report says how often. A turnover limit trades part of the way and builds the first book over several dates (the cost of building it is charged). A date with too few eligible names is held in cash and the report says why. The engine earns what the factor tests say the same holdings earn (a test compares the two over every period, to five millionths). The report gives the deflated Sharpe ratio for the number of ideas tried to find the alpha (from the library when it is one), the net-return-against-cost table and its breakeven cost, the result on the dates the alpha was found on and the locked dates it was tested on, how the limits bit, the latest book, and an experiment id that records the signal, the portfolio rules, the universe and the data it ran on. A backtest of a *Candidate* carries a notice that a good backtest is what a search is expected to produce. See [from signal to book](techniques/from-signal-to-constrained-book.md).

## What was measured, and what was not

Everything below is on the **simulated** world (`quant fundamentals simulate --demo`: 120 companies, 2009 to 2022, seed 7), so the commands reproduce it. Simulated returns are far calmer than real ones (the equal-weight universe alone has a Sharpe ratio of 3.3), so read the *difference* from the benchmark, never the level.

| Command | What it printed |
|---|---|
| `evaluate fcf_yield --demo --lookahead` | Mean rank IC +0.057 (t = 8.0), positive in 14 of 14 years and on 67% of dates; buckets 0.5%, 0.9%, 1.4%, 1.5%, 2.0% a month, monotonic; long-short net Sharpe 1.95 (gross 2.06), 21.4% a year, worst fall 10.2%, turnover 24%. The look-ahead audit: +0.057 honest, +0.056 naive (it does not lean on early knowledge). |
| `evaluate d_roic --demo` | Mean rank IC +0.019 (t = 3.0): *Some evidence*, net long-short Sharpe 0.49, positive in 10 of 14 years, strong in Germany (t = 2.7) and Health Care (2.7), nothing in Australia (0.1). A smaller planted effect, and the page reads it as smaller. |
| `factors --scan --demo` | 84 factors; the top ten by t-statistic are all value ratios (sales to price 11.8, earnings yield 10.8, ... free-cash-flow yield 8.0), which is what one expects when they all divide by the same market value: **one idea seen many times**, which is why the correlation view exists. |
| `discover --demo --n 80 --finalists 6` | 80 formulas tried, 72 tested in full; 20 passed the first IC test, 19 stability, 17 quantiles, 16 long-short and costs, 8 validation, 6 the locked test. Six Candidates, one look at the locked dates, probability of backtest overfitting 4%. The best: `div(revenue,market_cap)`, IC +0.072 on training, +0.092 on validation, +0.080 on the locked dates (t = 4.5). 66 failed ideas are in the graveyard. |
| `backtest --demo --factor fcf_yield` | A long-only book of the top 30: 26.7% a year, Sharpe 3.54, against 15.7% and 3.28 for owning everything; alpha 9.1% a year, beta 0.99, information ratio 1.8; turnover 2.6 times a year, trading cost 26 bps a year, breakeven one-way cost 904 bps. |

And what the tests prove, with known truth: **a placebo world** (each date's returns dealt out to other companies at random, which destroys any relation between a formula and what happened next) yields **no candidates** in two runs of 200 formulas, where the real world yields at least five in one; a world where nothing predicts anything is not made to look otherwise by the number of tries (one seed once let a lucky formula through every screen until the q-value gate was added). The combining rules and the models give **identical earlier weights and predictions when the future returns are altered**. The portfolio is **decided from what was known on each date**. A model fed noise is not credited with beating the rules. The DuPont contributions add up to the change in ROE to rounding. The deflated Sharpe ratio falls with the number of ideas tried.

**Found by using the tab in a browser, then fixed with a test for each:** the monitor flipped an alpha's signal a second time, so an alpha used upside down and still working was judged *Broken*; the library showed the training IC unflipped beside flipped validation and locked figures; the warnings of a blend of several alphas were repeated and did not say which alpha they were about; a count of years under a chart was a share printed as a count; one busy state for the whole tab made a long search block every other view.

**Not measured:** anything on real companies. No real key has been used. Whether the OpenFilings adapter, the Yahoo symbol mapping and the real filing dates behave as documented is unknown until a first run; the research layer has been tested only on data whose truth is known.

## What is not known

- **Annual only**, so a figure can be up to a year old by the time the next one arrives, and factors that need quarterly figures (earnings surprise, short-term fundamental momentum) cannot be built from real data yet.
- **Restatements** are kept when the provider sends them, but OpenFilings sends the figures as filed; whether a later filing for the same year arrives as a new vintage depends on the provider.
- **Sectors and industries are as of the day they were fetched**, not point in time. A company that changed sector is in its latest sector for the whole history, which can flatter a sector-neutral factor.
- **Yahoo symbols are a best-effort mapping**, delisted companies usually have no history, and prices are in the trading currency (converted with the exchange rate of the date where an amount is set against a price).
- **Survivorship bias remains** in any database built from today's tickers, however many delisted companies it keeps.
- **A backtest is a measurement on a sample.** Costs are a flat number of basis points a side plus a borrow rate, with no market impact beyond the limit on a position's share of daily volume.
- The probability of backtest overfitting needs enough survivors to estimate; with few it is reported as not available.

## The command line, the routes and the code

`quant fundamentals ACTION` has `status`, `simulate`, `add`, `plan`, `sync`, `prices`, `factors`, `evaluate`, `discover`, `library`, `monitor` and `backtest`; `--demo` works in the simulated database and `--json FILE` saves the whole result. `quant fundamentals library --validate ID` and `--set-status ID --to Rejected --reason "why"` change a status. A first Ctrl-C asks a running job to stop after its step (what is done is kept); a second ends the command.

The routes behind the tab are under `/api/fund/` (`status`, `overview`, `catalogue`, `companies`, `company`, `quality`, `library`, `alpha`, `graveyard`, `runs`, `run`, `experiments` to read; `companies/add`, `sync/plan`, `provider/test`, `price-symbol`, `formula/check`, `library/status`, `library/validate` to write; `sync`, `simulate`, `prices`, `scan`, `evaluate`, `evaluate/formula`, `lookahead`, `discover`, `monitor`, `correlation`, `combine`, `ml`, `backtest` as jobs on the app's worker, and `stop`). Slow work runs as a job, so closing the page does not stop it, and each view of the tab has its own running job, so a long search does not stop you reading the library or a company.

The code is `src/fundamentals` (`schema`, `store`, `providers`, `keys`, `usage`, `service` for the data; `pit`, `metrics`, `analytics`, `research` for what was known when; `factors`, `transforms`, `evaluate` for studies; `expr`, `discovery`, `library` for the search and its memory; `combine`, `bridge` for combinations and the portfolio; `workbench`, `cli`, `report` for what the page and the terminal share), `src/webapp/fundamentals_api.py`, and `src/webapp/static/fundamentals.js` with `fundamentals_lab.js`. The technique guides: [point-in-time fundamentals](techniques/point-in-time-fundamentals.md), [accounting analytics](techniques/accounting-analytics-dupont-piotroski-accruals.md), [factor studies](techniques/factor-studies-ic-quantiles-and-decay.md), [automated alpha discovery](techniques/automated-alpha-discovery.md), [combining alphas](techniques/combining-alphas-and-model-combination.md) and [from signal to book](techniques/from-signal-to-constrained-book.md).
