---
title: "Distress scores, the Magic Formula, owner earnings, Graham's tests and what a share might be worth"
slug: distress-scores-graham-tests-and-valuation-yardsticks
difficulty: 2
chapter: Platform
prerequisites: [fundamental-factors, accounting-analytics-dupont-piotroski-accruals]
stages: []
files: [src/fundamentals/scores.py, src/fundamentals/valuation.py, src/fundamentals/factors.py, src/fundamentals/workbench.py, src/references.py]
figures: []
tests: [tests/test_fund_scores_exact.py, tests/test_fund_factors_exact.py, tests/test_fund_valuation.py, tests/test_references.py]
models: []
---

# Distress scores, the Magic Formula, owner earnings, Graham's tests and what a share might be worth

## In one sentence

Rules from the distress-prediction papers and the value-investing books are written out once, run on the same point-in-time statements as every other metric, shown on a company's page with their zones, their tests and a row of bars set against the last price, and offered as factors; none is advice, and each one says what it assumes.

## The idea

**Altman's Z-scores** (Altman 1968; for non-manufacturers and emerging markets, Altman, Hartzell and Peck 1995 and Altman 2005) weigh a few ratios of the balance sheet and the income statement into one number and say which side of a fitted line a company stands on. The ratios are working capital, retained earnings and EBIT, each over total assets; equity over total liabilities; and, in two of the three versions, sales over total assets. The original (1968) score uses the *market* value of the equity and was fitted on public manufacturers (1.2, 1.4, 3.3, 0.6 and 1.0 on the five ratios; distress below 1.81, safe above 2.99). **Z′** replaces it with the *book* value for private firms (0.717, 0.847, 3.107, 0.420, 0.998; below 1.23 and above 2.90). **Z″** drops sales over assets so that the industry matters less (6.56, 3.26, 6.72 and 1.05 on the first four ratios; below 1.10 and above 2.60). The application computes Z″ and Z′ for every company-year and the original score, which needs the price, as a factor.

**Distance to default** (Merton 1974, in the naive form of Bharath and Shumway 2008) reads the equity of a company as a call option on its assets. With equity E, debt F (short-term debt plus half the long-term debt), the equity volatility of the last year σ_E, a debt volatility taken as 0.05 + 0.25 σ_E, an asset volatility σ_V = E/(E+F) σ_E + F/(E+F) (0.05 + 0.25 σ_E) and the last year's return μ as the drift, the distance is `(ln((E+F)/F) + μ − σ_V²/2) / σ_V`, in standard deviations of asset value, over one year.

**The Magic Formula** (Greenblatt 2005) ranks companies on two things and buys the best on both: operating income over enterprise value, and **return on capital**, operating income over (working capital without cash and without borrowings, plus net fixed assets). The factor is the average of the two percentile ranks among the companies of the date.

**Owner earnings** (Buffett 1986) are reported earnings plus depreciation and the other non-cash charges, less the capital spending the business needs to stay where it is (and any extra working capital). The application uses net income plus depreciation less *all* capital spending and leaves the working capital out, which is the letter's idea made computable and no more: Buffett says the last term is a guess.

**Sustainable and internal growth** (Higgins 1977) say how fast a company can grow on the profit it keeps. With the share of profit kept b = one less the payout, and the return on equity ROE, the sustainable rate is `b·ROE / (1 − b·ROE)` (equity grows by the profit kept; sales grow with it at a constant margin and leverage, so no new shares are sold). With the return on assets in place of ROE it is the internal rate: no new shares and no new borrowing either. A company growing faster than the rate is financing the difference with shares, with debt, or with better use of its assets.

**Graham's tests and numbers** (Graham, *The Intelligent Investor*, chapter 14; Graham and Dodd, *Security Analysis*). The **Graham number** is `√(22.5 × earnings per share × book value per share)`, the price at which a share is at 15 times its average earnings of the last three years and 1.5 times its book value at once. **Net current asset value** is current assets less every liability and the preferred stock, per share; Graham bought at two-thirds of it or less, and Oppenheimer (1984, 1986) tested his rules on a later sample. The five **tests of the defensive investor** that the statements can answer are: current assets at least twice current liabilities; long-term debt no larger than working capital; a profit in every year; a dividend in every year; and earnings per share up by a third over ten years (the mean of the last three against the mean of the three that ended seven years before). Graham asked for ten years of profits and twenty of dividends. The application looks at the years the database holds, up to ten, and does not judge a test on fewer than five; an unjudged test is empty, never a failure. The two price tests (no more than 15 times the three-year average earnings, no more than 1.5 times book) are set against the last close.

**The PEG ratio** (Lynch 1989) is the price-to-earnings ratio over the growth rate of earnings in per cent; a fairly priced company has a PEG of 1. The factor is its inverse, so that more is better and a loss does not blow it up: the three-year growth in earnings per share, in per cent, times the earnings yield.

**Net operating assets** (Hirshleifer, Hou, Teoh and Zhang 2004) are debt plus equity plus minority interest less cash, which is operating assets less operating liabilities. Over last year's assets, companies whose balance sheets carry a lot of operations that cash flow has not paid for have tended to earn less afterwards; the change in the amount is the balance-sheet counterpart of Sloan's accruals.

**Three composites** use the library's own parts: a quality-minus-junk style score (Asness, Frazzini and Pedersen 2019: profitable, safe and conservatively financed), Lakonishok, Shleifer and Vishny's (1994) contrarian value (cheap, and with low past growth: the growth leg is subtracted), and Piotroski's (2000) idea of buying cheap companies that are getting healthier. They are the ideas of the papers built from the parts the library has, not the papers' exact sorts.

## Why it matters

These are the questions the books teach an investor to ask of a statement: is the company in trouble, what does the business earn on the capital it needs, what is left for the owner, how fast can it grow without selling shares, does it pass the old tests, and what would the share be worth if each rule's assumption held. Written as metrics they are tested and reproducible; written as factors they can be studied against what happened next like any other (the factor lab names the paper or book under each); written as bars beside the price they show how far apart the rules are, which is the honest answer to "what is it worth".

The evidence on distress is a warning against reading a zone as a trade: Dichev (1998) and Campbell, Hilscher and Szilagyi (2008) found that stocks with a high risk of failure earned *lower* returns, not higher, so the prior of the distress factors is that a safer company earns more, and a Z-score is a screen for risk, not a return forecast.

## How this repo uses it

`src/fundamentals/scores.py` holds every score as a registered metric (27 of them, from the Altman ratios to the Graham tests and net operating assets), with its constants and zones in one place and an `altman_zone` function; they join the 91 older metrics, so the formula language and the discovery search can use them. `src/fundamentals/factors.py` registers twenty factors built on them (a score read as it is, or set against the price: the Altman original, the distance to default, owner earnings yield, the inverse PEG, the Graham number and net current asset value over the price, the Magic Formula) with a prior direction each, and `src/references.py` names the book or paper behind every one (the factor lab shows it under the description; a book has no link, a journal article has a DOI that was checked against Crossref).

On a company's page (Company explorer → Open): two tiles (the Altman Z″ score and its zone; the Graham tests passed of those that could be judged); Z″ and Z′ by year with the three zones behind the line and what each ratio added to the latest Z″; the DuPont split of the change in return on equity as stacked columns, year by year; the growth the profit kept can pay for against actual revenue growth; owner earnings next to net income and free cash flow; the Graham tests by year; and **What might the share be worth?**, a football field of seven bars (the Graham number, net current asset value, Greenwald's earnings power value, owner earnings and free cash flow capitalised, the Lynch line, book value per share) against the last close, with two inputs to change, the discount rate and the growth for ever, and the Graham price tests under it. The page asks the server for the bars at `/api/fund/valuation?id=&rate=&growth=`. Every chart has a table.

The bars are per share in the statements' own currency and are set against the last close only when it is quoted in that currency (otherwise the page says so and shows no price). Each bar moves the discount rate one point either way. The rate must exceed the growth by two points, or a perpetuity has no finite value, and the server refuses a pair that does not.

## What we found

The arithmetic is pinned by hand-worked tests. A company with working capital 400, retained earnings 600 and EBIT 250 on assets of 2,000, equity 1,000 against liabilities 1,000 and sales 1,800 has a Z″ of exactly 4.18 (1.312 + 0.978 + 0.84 + 1.05) and a Z′ of 2.104075; the same balance sheet with negative equity keeps the negative ratio and loses points. Operating income 250 over (900 − 200) − (500 − 100) + 700 = 1,000 of capital is a return of 25%; net income 150 plus depreciation 50 less capital spending 80 is owner earnings of 120; a payout of 30% and a return on equity of 15% give a sustainable growth of 0.105 / 0.895 = 11.7%; average earnings per share of 2.0 and book value per share of 10 give a Graham number of √450 = 21.2. The distance to default is checked against the closed form on the volatility and drift the research context itself produced, a company with no debt has none (not an infinite one), and the three composites equal the mean of their standardised parts, the contrarian one with its growth leg subtracted.

On the first company of a simulated demo world (40 companies, 2010 to 2018, seed 5; `S0001`) the page reads: Z″ of 2.65 in the safe zone just above the 2.60 line, Z′ of 1.31 in the grey zone, two of five Graham tests passed (profits and dividends in every year held; the current ratio, the debt test and the ten-year growth test fail), a Graham number of 3.94, no net current asset value (current assets do not cover the liabilities), an earnings power value of 2.31 (1.95 to 2.75 over a rate of 8% to 10%), owner earnings capitalised at 5.74 and free cash flow at 5.80, book value per share of 2.54, no Lynch line (earnings were not positive three years earlier), against a last close of 14.69: every bar sits well under the price and the Graham price tests read 54 times earnings and 5.8 times book. That is a property of how the demo world sets prices (from revenue, not from earnings), not a finding about markets.

A scan of all 104 factors on the demo world the chapter documents (120 companies, 2009 to 2022, seed 7) puts the book factors where that world's design predicts. Owner earnings yield (mean rank IC +0.067, t = 8.6) and the Graham number over the price (+0.069, t = 8.6) join the top ten with the other value ratios, one idea seen many times: the world builds its prices from revenue and plants a value effect. The inverse PEG (t = 6.4, on 60% of the company-dates: it needs two positive earnings figures), the Magic Formula (6.1) and cheap-and-improving (4.5) follow. The scores that rise with the share price point the wrong way in that world, as they should: the original Altman Z (t = −3.7) and the distance to default (t = −5.6) both carry the price, and twelve-month momentum has a mean IC of −0.032 (t = −4.0) there: in that world a share that has risen is, on average, one that has since given some of it back. The book-value scores Z″ and Z′ have no relation to the next month's return (t = 0.9 and 0.6), net operating assets and their change are negative as their prior says (t = −2.1 and −2.5), and net current asset value over the price, which is positive on 27% of the company-dates and never reaches Graham's 1.5 (a price of two-thirds of it), has nothing (t = −0.7). That is a demonstration that the machinery runs and counts, not evidence about markets: 45 of the 104 factors are beyond |t| of 2 in that world, which is why the scan says what that means before it prints the table.

What has not been found: how any of this behaves on real companies. Nor were the constants read in the papers themselves: Altman's 1968 paper and his own review of the models could not be opened (his page redirects in a loop), so the coefficients and zones of the three forms are those that several secondary sources, found by search, agree on, and the emerging-market constant of Altman (2005) is not in the application. The rules are the published ones, applied to figures whose mapping from the SEC's tags (retained earnings is `us-gaap:RetainedEarningsAccumulatedDeficit` or `ifrs-full:RetainedEarnings`) and from OpenFilings' keys is tested on written-out values, not on a real filing: the four real filings in the test fixtures were trimmed before retained earnings was read. The search draws its formulas from every ratio the library has, so adding 27 metrics changed which formulas a seed draws; the measurements of the discovery and the placebo in [the chapter](../fundamentals.md) were run again and replaced (the factor study, the backtest and the preflight did not move).

## Pitfalls

- **A zone is a screen, not a forecast.** Altman fitted his lines on manufacturers of the 1940s to the 1960s (the original) and later samples; a company near a line is near it. The evidence that distressed stocks have not been paid for the risk (Dichev; Campbell, Hilscher and Szilagyi) is why the prior is "safer earns more".
- **Banks and insurers do not fit.** They have no current assets or working capital, so the scores are missing for them, as they should be.
- **EBIT is operating income** where the statements give no other figure, and the scores need retained earnings: statements fetched before this version do not have it, and the page says the score is not available for them. A fetch that brings the years again fills it in; the application cannot ask for a held year just to add one figure.
- **Owner earnings treat all capital spending as the spending needed to stand still**, so a growing company looks poorer than Buffett would call it, and a company in decline richer.
- **The growth rates assume constant margin and leverage**, use the closing equity, count dividends and not buybacks, and treat a cash-flow statement with no dividend line as no dividend. They are not forecasts.
- **A Graham number needs positive average earnings and positive book value**, uses the earnings as reported (not normalised), and the net-net rule rarely finds anything in a modern market of large companies; where it does the shares can be illiquid. The tests are judged on the years held.
- **The earnings power value** assumes capital spending equals depreciation, uses the average margin of up to five years on this year's revenue and the company's own tax rate when it is plausible (else 25%), and depends entirely on the discount rate you choose.
- **A perpetuity is dominated by the gap between the rate and the growth.** Two points is the smallest gap the page allows and the bar for the lowest rate in a range is already wider than the bar for the highest; do not read the middle of a bar as a price target.
- **The PEG ratio uses past growth** (three years of earnings per share), where Lynch meant growth to come; it is missing unless both ends are positive; the Lynch line is capped at 40 times earnings.
- **The distance to default needs prices** and uses the book value of debt as the face value; it is a screen for how much volatility the equity can take before the debt is at risk, not a probability of failure.
- **The Magic Formula leaves out financial companies and utilities** in the book. The factor does not; restrict the universe by sector for the faithful version.
- **The three composites are not the papers' sorts.** They are built from the library's parts, equal weighted, and say so in their descriptions.

## Try it

```python
import tempfile

from src.fundamentals.service import FundamentalsService
from src.fundamentals.workbench import Workbench

svc = FundamentalsService(tempfile.mkdtemp())
svc.load_simulated(n_companies=40, start_year=2010, end_year=2018, seed=5)        # a simulated world: the numbers say nothing about real markets
wb = Workbench(service=svc)
sid = int(wb.companies(limit=1)["companies"][0]["security_id"])

company = wb.company(sid)
for model in company["distress"]["models"]:
    last = model["series"][-1]
    print(model["label"], round(last["score"], 2), last["zone"], round(sum(last["contributions"].values()), 2))      # the ratios' points add up to the score
graham = company["graham"]["rows"][-1]
print(int(graham["met"]), "of", int(graham["judged"]), "Graham tests")

bars = wb.valuation(sid, rate=0.10, growth=0.03)                                   # a higher rate and growth than the defaults (9% and 2%)
print(bars["price"])
for row in bars["rows"]:
    print(row["label"], row["low"], row["point"], row["high"], row["missing"])
```
