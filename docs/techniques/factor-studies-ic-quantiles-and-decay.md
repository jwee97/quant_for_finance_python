---
title: "Factor studies: IC, buckets, a costed long-short and how fast it fades"
slug: factor-studies-ic-quantiles-and-decay
difficulty: 2
chapter: Platform
prerequisites: [information-coefficient, cross-sectional-factors]
stages: []
files: [src/fundamentals/evaluate.py, src/fundamentals/factors.py, src/fundamentals/transforms.py, src/fundamentals/research.py]
figures: []
tests: [tests/test_fund_evaluate.py, tests/test_fund_factors.py, tests/test_fund_transforms.py]
models: []
---

# Factor studies: IC, buckets, a costed long-short and how fast it fades

## In one sentence

A factor study asks, on every date, whether a company's rank on the factor says anything about its rank in the returns that follow, and then asks the follow-up questions that decide whether that is worth anything: are the buckets in order, does a long-short portfolio still pay after costs and borrowing, how fast does the signal fade, and does it hold in every year, country, sector and market.

## The idea

**The information coefficient (IC)** on a date is the rank correlation, across companies, between the factor and the return over the next period. It is computed on every date and then averaged; a factor that works earns a modestly positive IC most of the time, not a large one on a few dates. Three numbers describe the series: its mean, the share of dates on which it is positive, and the *information ratio of the IC* (mean over standard deviation, scaled to a year). The t-statistic is a Newey-West one, because the returns of successive dates overlap (a one-month return measured every month is not independent of last month's), and an ordinary t-statistic would be too generous.

**Buckets.** On each date the companies are sorted into five (or up to ten) groups by the factor, and the average next-period return of each is recorded. What matters is whether the bars rise in order (the Spearman correlation of the bucket number with its return, the *monotonicity*) and whether the spread between top and bottom is large next to the noise. A factor whose whole effect sits in one tail, or whose middle buckets are out of order, is a different thing from a steady slope.

**A long-short portfolio** buys the top bucket and sells the bottom one, held one rebalance period, and reports its return, volatility, Sharpe, Sortino, worst fall, hit rate and turnover, **gross and net**: net takes off the trading cost (default 10 basis points one way) on what changed, and the cost of borrowing the short side (default 50 basis points a year). The gap between the gross and net Sharpe is the cost drag; a factor with a high turnover can lose most of its edge to it.

**Decay** is the IC against returns over one week, one, three, six and twelve months. A slow factor (value, say) is still there at six months and can be traded cheaply; a fast one has gone in a week and must be traded fast.

**Stability and breakdowns** repeat the study by calendar year and in two halves of the sample, inside each country, sector and region, and by market regime (bull or bear, high or low volatility, each known at the start of the period it describes).

**The direction.** Every factor carries a direction from theory (higher is better, or lower). A study may flip it, but choosing the sign that happened to work is a choice made on the data, so the option is called *auto* and the report counts it as a second trial.

## Why it matters

A backtest of one chosen factor is an anecdote. The same factor measured this way says whether the effect is steady or was made by three good years, whether it comes from one country or sector, whether it survives its own trading costs, and how quickly it must be traded, and those are the questions on which most published factors fail when they are looked at again. The study also fixes what "better than luck" means: a t-statistic that allows for overlap, and a verdict that is worded to match.

## How this repo uses it

`src/fundamentals/evaluate.py` holds the arithmetic (`rank_ic`, `ic_stats` with the Newey-West error, `quantile_returns`, `monotonicity`, `long_short`, the decay and breakdown functions, `evaluate`), `src/fundamentals/factors.py` the eighty-four named factors, and `src/fundamentals/transforms.py` the pipeline a factor passes through (winsorise each date's tails, neutralise against sector, industry, country, the market, size, beta or volatility by a regression on each date, then z-score, rank or percentile). The **Factor lab** and `quant fundamentals evaluate` run it; **Scan every factor** runs every factor under one set of conditions.

The verdict is a sentence with a symbol and never says validated: *Strong in this sample* needs |t| of at least 3 and a net long-short Sharpe above 0.5, *Some evidence* needs |t| of at least 2, and below that there is *No clear evidence*. Every verdict adds that an interesting factor is not a validated alpha: a result in the sample it is measured on is a lead, and the discovery search is where the number of ideas tried is counted.

## What we found

The tests give the arithmetic known truth: a perfect signal has an IC of one and rising buckets; the Newey-West error matches a reference and widens with autocorrelation; costs come off in proportion to what was traded, and a constant signal costs almost nothing; and a study of a window depends on nothing after it. Against a simulated world with alphas planted in it, the planted ones are found (the change in ROIC has t above 3 and a net Sharpe above 0.5, free-cash-flow yield t above 4 and monotonicity of at least 0.9) and three ordinary factors (current ratio, debt to equity, operating margin) have |t| below 2.5.

On the default demo world (`quant fundamentals evaluate fcf_yield --demo`): mean rank IC +0.057 (t = 8.0), positive in 14 of 14 years and on 67% of dates; buckets 0.5%, 0.9%, 1.4%, 1.5%, 2.0% a month; long-short net Sharpe 1.95 against 2.06 gross (a cost drag of 0.11), turnover 24% a rebalance; the IC rising with the horizon from +0.035 at a week to +0.206 at a year. The smaller planted effect, the change in ROIC, reads as *Some evidence* (+0.019, t = 3.0, net Sharpe 0.49), strong in Germany (t = 2.7) and Health Care (2.7) and nothing in Australia. Scanning all eighty-four factors, the top ten by t-statistic are all value ratios, which one expects when they all divide by the same market value: **one idea seen many times**, which is why the library has a correlation view.

What has not been found: any result on real companies.

## Pitfalls

- **Many factors, one best.** The scan sets eighty-four factors side by side and says so: about one in twenty shows |t| above 2 by luck alone. It is a map of where to look, not a list of alphas; the discovery search is what counts the tries.
- **Overlap.** A one-year return measured monthly overlaps eleven twelfths with the last one. The Newey-West error allows for it; a plain correlation of the IC series does not.
- **A high IC from a few companies.** A factor that covers few companies says something about them only; the study reports the share of investable companies it covers on each date.
- **Neutralising changes the question.** Removing sector and size asks whether the factor works *within* them; a factor can be real and still be a sector bet.
- **The sign.** Taking the direction that worked is a second trial.
- **Costs are flat.** A number of basis points a side and a borrow rate, with no market impact; a small-cap factor can cost more than that when traded in size (the backtest's limit on a position's share of daily volume is the guard).
- **Survivorship.** Delisted companies are kept with their losses, but a database built from today's tickers is not a complete history of who was listed; the study says how many are inactive and never claims the sample is unbiased.

## Try it

```
quant fundamentals simulate --demo
quant fundamentals evaluate fcf_yield --demo --horizon 3m --cost-bps 25
quant fundamentals factors --scan --demo --top 15
```

```python
from src.fundamentals import evaluate as E
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.store import FundamentalsStore

ctx = ResearchContext(FundamentalsStore("data/user/fundamentals_demo.db"), UniverseSpec())
study = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=()))
print(study["verdict"])
print(study["ic"]["mean_ic"], study["ic"]["t_stat"], study["portfolio"]["net"]["sharpe"])
```
