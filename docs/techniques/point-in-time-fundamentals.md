---
title: "Point-in-time fundamentals: a figure counts from the day it was public"
slug: point-in-time-fundamentals
difficulty: 2
chapter: Platform
prerequisites: [point-in-time-market-data, walk-forward-and-leakage]
stages: []
files: [src/fundamentals/pit.py, src/fundamentals/research.py, src/fundamentals/store.py, src/fundamentals/evaluate.py]
figures: []
tests: [tests/test_fund_pit.py, tests/test_fund_research.py, tests/test_fund_evaluate.py, tests/test_fund_alpha_testing.py, tests/test_fund_exact_values.py]
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

The **look-ahead audit** (`look_ahead_audit` in `src/fundamentals/evaluate.py`, the *Look-ahead audit* button, `--lookahead`) runs one study twice, once as above and once with `naive_availability=True` (every figure known at its period end), and reports both ICs, their difference and a verdict that says which to believe. It exists to be run on any factor before trusting it, and to show how much a naive join would have flattered it. It also **waits**: the same study is repeated with each figure used only 1, 30, 60 and 90 days after its filing (`by_lag`), and the audit says whether the IC falls to under half within a month (`fades_when_waiting`: the signal lives in the days right after a filing and is only as good as how fast it is acted on) or is still there after three months (it is not about being first to read the filing). Where filing dates are only estimated, a long wait is also the safe reading of how stale a figure may be.

## What we found

On a simulated world built to test it, in which net income is known a quarter early, the figure *predicts* returns when it is used as known (mean IC above 0.04) and does not when it is used from the day it was public (below 0.01), and the audit flags the study as flattered. On the default demo world, whose filings carry real dates, the audit finds nothing to flag: free-cash-flow yield has a mean IC of +0.059 either way (+0.059 honest, +0.058 naive), and 2.6% of its periods needed an estimated filing date (3% of filing dates were removed from the simulated data on purpose). Waiting leaves it where it was (+0.060, +0.061 and +0.061 at 30, 60 and 90 days after the filing), so it is not a signal about being first. On a world in which a figure's information reaches the market in the three months before it is filed, the same wait takes the IC to under half within a month, and the audit says so. The tests also pin the small rules, with exact values: the assumed lags (120 days for a year, 90 for a half, 60 for a quarter) and the age limits (640 days for an annual figure, 540 for a trailing one) are the documented ones and a figure is kept on the last day of its limit and dropped the next; four consecutive quarters or two halves add up to a trailing year exactly, a missing quarter breaks the run, and quarters more than a tenth away from the annual figure are not used; a figure is available the day after the earliest date a provider gave; a late restatement of an old year never replaces a newer year; a value holds until the next filing and then expires; securities are independent; and a figure missing from the latest period stays missing instead of borrowing an older one.

What has been found on real data, for US companies: reading the SEC's XBRL facts of twelve large companies gave 226 annual statements, every one with a real filing date (no estimated date at all), and each dated by the report in which the year first appeared, which is why a figure's date is never earlier than the day it was public; the lag between a year's end and the filing of its own report was 24 to 66 days (median 52, 90th percentile 58) over the 202 years that were not first seen as comparatives, against the 120 days the engine assumes when it has no date. The other 24 (the first two years of each company's history) appear only as comparatives in a later report and are dated by it, which is late and never early. What has *not* been found: how filing dates behave in other markets (the OpenFilings adapter has met a live key only once, a company fetched for 11 credits), and whether restatements arrive as new vintages there.

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
