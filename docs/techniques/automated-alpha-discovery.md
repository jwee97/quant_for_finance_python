---
title: "Automated alpha discovery: search widely, count everything, test once"
slug: automated-alpha-discovery
difficulty: 3
chapter: Platform
prerequisites: [multiple-testing, reality-check-spa-pbo, walk-forward-and-leakage]
stages: []
files: [src/fundamentals/expr.py, src/fundamentals/discovery.py, src/fundamentals/library.py, src/fundamentals/evaluate.py]
figures: []
tests: [tests/test_fund_expr.py, tests/test_fund_discovery.py]
models: []
---

# Automated alpha discovery: search widely, count everything, test once

## In one sentence

A computer can try thousands of formulas built from a company's accounts far faster than a person can, and the best of thousands of tries looks good by luck alone, so the search is only worth running if it counts every formula it tried, throws most of them out cheaply, corrects the survivors for how many there were, and tests the last few once on dates it never used to choose them.

## The idea

**Formulas as trees.** A formula is a small typed expression tree: a statement figure, a derived ratio or a market quantity; `add sub mul div`, `neg abs log`; `delta growth accel rolling_mean rolling_std` over a number of periods; `rank sector_rank industry_rank zscore`. It is JSON and nothing else (a formula typed as text is parsed, never run as code), so it can be stored, shown, hashed and run again. The types are checked: an amount divided by an amount is a ratio, a reporting-currency amount set against a price is first converted at the exchange rate of the date, and adding an amount to a ratio is refused, so the search does not spend tests on formulas that mean nothing. `div` gives a missing value unless the denominator is positive.

**One idea, one identity.** The canonical form puts commutative arguments in order and removes double negations and repeated ranks. Beyond that, a rank, a z-score or a sign on the *outside* of a formula does not change which companies it prefers, so `rank(x)`, `zscore(x)` and `neg(x)` are the same idea as `x`: the *signal key* finds duplicates the hash would miss, and a formula more than 90% correlated with one already accepted in the run is counted as the same idea.

**Stages, cheapest first.** Each formula is judged on the *training* dates only until the last two stages. Stage 0: it computes, covers at least 40% of the investable companies, takes more than five distinct values on a typical date and has not been tried before under the same conditions. Stage 1: a rank IC with next month's return of |t| at least 1.65 (Newey-West). Stage 2: positive on 55% of dates and in both halves, and still |t| above 1 without its best year. Stage 3: buckets that rise in order. Stage 4: a long-short portfolio that earns gross. Stage 5: still earns after costs and borrowing at a turnover of at most 70% a rebalance. Stage 6: the same sign on a second block of dates (validation) that stages 1 to 5 never used. Stage 7: the finalists, chosen from training and validation only, are tested **once** on the last block.

**Blocks with a gap.** The dates are cut into three consecutive blocks (55%, 20%, 25% by default), with a gap as wide as the forecast horizon between them so that no return window sits in two blocks. The sign of a formula (whether high or low should earn) is learned on the training block and held fixed, which makes the test two-sided.

**Multiple testing.** Every formula that reaches stage 1 is a test and is counted. The training p-values of all of them go through the Benjamini-Hochberg procedure to give each finalist a false-discovery *q-value*; the training-and-validation Sharpe ratio is deflated for the number of formulas tried (Bailey and López de Prado); with enough survivors the probability of backtest overfitting is estimated by combinatorially symmetric cross-validation. A finalist needs a q-value of at most 0.25 over everything tried in the run.

**Memory.** Every candidate is written to the database the moment it is judged. A rejected formula goes to the *graveyard* with the stage it failed at and why, under a unique key of formula and conditions, and is never deleted; only the metadata of a survivor (the formula, the conditions, a few dozen numbers) goes to the *library*, and the panels an experiment makes are recomputed when needed, never kept. The stream of formulas is a function of the seed and the position, so a stopped or crashed run continues exactly where it was.

## Why it matters

Every significance statement about the winner depends on how many were tried, and a record of only the survivors loses that number. A search that remembers its failures can say *this was the best of 72 tested formulas, with a q-value of x*; one that does not can only say *this one worked*. The locked dates have the opposite job: they are valuable because they are looked at rarely, so each look is counted and an alpha says how many looks had been made before its own.

## How this repo uses it

`src/fundamentals/expr.py` is the formula language and generator (well-typed random trees, mutation, crossover, a built-in list of the classics of the literature); `src/fundamentals/discovery.py` is the staged screen, the fitness (information coefficient, stability, Sharpe, monotonicity and unlikeness to the others, less turnover and a penalty for the number of nodes, so the simpler formula wins when the evidence is equal) and the resumable run; `src/fundamentals/library.py` is the library, the graveyard, the status table and the monitor. The **Alpha discovery** view and `quant fundamentals discover` start a run; each can be stopped (what is done is kept) and resumed. A *finished* run cannot be reopened: its locked test has been looked at.

The best status a search can give is **Candidate**. *Validated* needs a locked-dates t-statistic of at least 1.0 and either a q-value of at most 0.10 or a deflated-Sharpe probability of at least 0.90; a person can approve an alpha anyway, with a written reason that is recorded and marked forced. Moves between statuses follow a fixed table and are logged with their reasons. The **monitor** recomputes each alpha that is still in use on the data now in the database and compares its recent IC with the one it earned when it was found.

## What we found

On the default demo world, `quant fundamentals discover --demo --n 80 --finalists 6` tried 80 formulas and tested 72 in full. The funnel: 20 passed the first IC test, 19 stability, 17 quantiles, 16 long-short and costs, 8 validation and 6 the locked test. Six Candidates, one look at the locked dates, a probability of backtest overfitting of 4%. The best, `div(revenue,market_cap)`, had an IC of +0.072 on training, +0.092 on validation and +0.080 on the locked dates (t = 4.5). Five of the six are value ratios, and they are largely one idea: three of them put through the correlation view had pairwise rank correlations of 0.64 to 0.75 and made about 1.5 independent ideas out of 3. Sixty-six failed ideas are in the graveyard, 52 of them at the first IC test.

The tests give the search known truth. **In a world where nothing predicts anything** (each date's returns dealt out to other companies at random) two runs of 200 formulas produce no Candidate at all and at most five survivors of the screens between them, none with a q-value of 0.25 or below; a seed that once let a lucky formula through every screen is why the q-value gate exists; the same search on the real simulated world produces at least five Candidates. A stopped run continues exactly where it was and the count of tests agrees; the graveyard is never overwritten and finds the same idea in another spelling; and the locked dates are counted every time they are looked at (a second candidate says its locked dates had already been looked at once).

Found by using the tab, then fixed with a test: the library stored the training IC unflipped next to flipped validation and locked figures, so a formula used upside down showed a negative IC beside positive ones; and the monitor flipped the signal a second time, so such an alpha, still working, was judged Broken. Both are fixed (the figures are all in the direction chosen, and the direction is shown), and a test keeps the monitor honest for an alpha used upside down.

What has not been found: anything on real companies, or how many true alphas a real universe holds. The bars for *Validated* are modest by design (one look at a short block cannot support more); for a decision about money, raise them.

## Pitfalls

- **Counting only survivors.** The number tried must include every formula that reached the first test, the failures included. The graveyard is what makes that number available later.
- **Looking at the locked dates more than once.** Each look is a test. A second look that changes what you do with the result has turned the locked block into training data. The counter exists to make that visible.
- **Choosing the sign after the fact.** The sign is learned on the training block only; flipping an alpha because it did badly on the locked block is a second trial.
- **Raising the thresholds after seeing the funnel** is selecting on the answer. Change them before a run, say why, and keep the run.
- **Many copies of one idea.** Five value ratios are one finding, not five; look at the correlation view before counting survivors as independent.
- **Complexity is a cost.** A formula of fifteen parts has had fifteen chances to fit noise; the fitness penalises it, and an alpha you cannot explain deserves less trust than one you can.
- **Candidate is not an alpha.** It is a formula that passed screens, one of which used data it was not chosen on. Validation needs the evidence to hold after the correction for the number tried, and time to pass.

## Try it

```
quant fundamentals simulate --demo
quant fundamentals discover --demo --n 80 --finalists 6
quant fundamentals library --demo
quant fundamentals library --demo --graveyard --top 10
```

```python
from src.fundamentals import discovery as D
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.store import FundamentalsStore

store = FundamentalsStore("data/user/fundamentals_demo.db")
disc = D.Discovery(store, ResearchContext(store, UniverseSpec()))
run = disc.create(D.DiscoverySpec(n_candidates=40, seed=3, seeds=["div(fcf,market_cap)", "delta(roic,12)"], finalists=4))
summary = disc.run(run)
print(summary["counts"]["tested"], summary["counts"]["candidates"], summary["counts"]["oos_looks"])
print([(r["expression"], r["status"]) for r in summary["leaderboard"][:3]])
```
