---
title: "Point-in-time fundamentals: a figure counts from the day it was public"
slug: point-in-time-fundamentals
difficulty: 2
chapter: Platform
prerequisites: [point-in-time-market-data, walk-forward-and-leakage]
stages: []
files: [src/fundamentals/pit.py, src/fundamentals/research.py, src/fundamentals/store.py, src/fundamentals/evaluate.py]
figures: []
tests: [tests/test_fund_pit.py, tests/test_fund_research.py, tests/test_fund_evaluate.py]
models: []
---

# Point-in-time fundamentals: a figure counts from the day it was public

## In one sentence

A company's accounts describe a period that ended weeks or months before anyone could read them, so a study may use a figure only from the day it was filed (plus a lag for the reader), and a study that uses it from the end of the period it describes has given every factor months of foresight.

## The idea

A December year-end is usually filed in February or March. A table keyed by the period it describes puts the December figure next to the December prices, and a factor built from it appears to know, in January, what the market learns in March. Nothing in the data looks wrong: the figure is right, the date is just the wrong one. The repair is a second date on every figure, the day it became public, and one rule: **a research date may use a figure only if it was available on or before that date.**

The rule the application enforces, in one place (`src/fundamentals/pit.py`), for every stored period:

- **Availability** is the earliest filing, acceptance or publication date a provider gave, plus the *information lag* you choose (one day by default: a filing may arrive after the close). A restated figure is a new vintage that is public from the day it was *first seen*, not from the original filing, and the original is kept beside it.
- **When no date is known**, the period end plus a conservative lag for that kind of period is used and the period is flagged *estimated*, never silently given the period end. The page counts how many periods needed that, and a study can leave them out.
- A figure appears on the first rebalance date on or after it was available and holds until the next filing; a figure that is too old expires instead of being carried for ever.
- A company's year earlier is found **by period end, not by row position**, so a gap in a company's history gives a missing growth rate, not a wrong one.
- **Trailing twelve months** at each quarter end adds the flows of the last four quarters (or two half-years) and takes the balance sheet as at the latest quarter; quarterly figures that do not add up to the annual ones (year-to-date numbers mistaken for quarters) are not used. A year-end row stands in where no quarterly history exists.
- Where two providers report the same period the first in the priority list leads and the other fills only the gaps, and only with values that were public no later than the leading row.

Two more clocks matter, and the application sets both: the trade is struck **a day after the signal** (`entry_lag`), never at the close the signal was computed from; and anything *learned from realised returns* (a combining weight, a model fit, the sign of a formula) is used on a date only if the return window it learned from had ended by then. That distance (`settled_lag`) is read off the grid of dates, not assumed.

## Why it matters

Look-ahead bias is the most common error in fundamental factor research and the hardest to see, because the flattered factor looks like a good one: high information coefficients, a smooth equity line, a story that makes sense. It is not a small effect. A figure that is known three months early does not need to be a good predictor to look like one, since the market has not yet moved on it. A strategy built on dates like that cannot be traded, and the discovery search, which tries thousands of formulas, will find the ones that benefit most from the error and rank them first.

## How this repo uses it

Every panel a study reads (statement figures, derived metrics, market capitalisation, enterprise value, returns) is built from the local database through `src/fundamentals/pit.py` and `src/fundamentals/research.py`; no factor has to remember the rule. The settings are in `PitSettings` (`info_lag_days`, `use_estimated`, `naive_availability`, the quality flags to exclude, the maximum age of an annual and a trailing figure, the provider priority). The page shows the day each figure became public next to it on a company's page, says which are estimated, and the study's notes say how many periods of the sample were.

The **look-ahead audit** (`look_ahead_audit` in `src/fundamentals/evaluate.py`, the *Look-ahead audit* button, `--lookahead`) runs one study twice, once as above and once with `naive_availability=True` (every figure known at its period end), and reports both ICs, their difference and a verdict that says which to believe. It exists to be run on any factor before trusting it, and to show how much a naive join would have flattered it.

## What we found

On a simulated world built to test it, in which net income is known a quarter early, the figure *predicts* returns when it is used as known (mean IC above 0.04) and does not when it is used from the day it was public (below 0.01), and the audit flags the study as flattered. On the default demo world, whose filings carry real dates, the audit finds nothing to flag: free-cash-flow yield has a mean IC of +0.057 either way (+0.057 honest, +0.056 naive), and 2.6% of its periods needed an estimated filing date (3% of filing dates were removed from the simulated data on purpose). The tests also pin the small rules: a figure is available the day after the earliest date a provider gave; a late restatement of an old year never replaces a newer year; a value holds until the next filing and then expires; securities are independent; and a figure missing from the latest period stays missing instead of borrowing an older one.

What has *not* been found: how real filing dates behave. The OpenFilings adapter has not been run against a live key, so the share of periods with an estimated date, the lag between a period's end and its filing in each market, and whether restatements arrive as new vintages are unknown until a first run.

## Pitfalls

- **An information lag of zero** assumes every filing arrives before the close. One day is the least that is defensible; markets that file late deserve more.
- **Estimated dates are guesses.** A conservative lag protects against look-ahead but not against a database whose "filing date" is really the date it was loaded.
- **Annual statements only** means a figure can be up to a year old when the next arrives, which is honest but slow; factors that need quarterly figures cannot be built from real data yet.
- **Sectors and industries are as of the day they were fetched**, not point in time, so a company that changed sector sits in its latest sector for the whole history.
- **Prices matter too.** Market capitalisation uses the unadjusted close and the share count known on the date, and a yield sets an amount against a price in the trading currency; the exchange rate used is the one of the date.
- The audit tests the figures' dates. It cannot find look-ahead in a formula you wrote that uses a future value some other way (a rolling window that looks forward, say); the operators available in a formula only look back.

## Try it

```
quant fundamentals simulate --demo
quant fundamentals evaluate fcf_yield --demo --lookahead
```

```python
from src.fundamentals import evaluate as E
from src.fundamentals.research import UniverseSpec
from src.fundamentals.store import FundamentalsStore

store = FundamentalsStore("data/user/fundamentals_demo.db")        # made by the command above
audit = E.look_ahead_audit(store, E.EvalSpec(factor="fcf_yield", breakdowns=()), UniverseSpec())
print(audit["verdict"])
print(audit["honest"]["mean_ic"], audit["naive"]["mean_ic"], audit["flattered"])
```
