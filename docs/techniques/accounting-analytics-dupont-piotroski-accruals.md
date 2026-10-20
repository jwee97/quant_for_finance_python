---
title: "Accounting analytics: DuPont, the Piotroski score, accruals and an anomaly screen"
slug: accounting-analytics-dupont-piotroski-accruals
difficulty: 2
chapter: Platform
prerequisites: [fundamental-factors]
stages: []
files: [src/fundamentals/analytics.py, src/fundamentals/metrics.py, src/fundamentals/factors.py]
figures: []
tests: [tests/test_fund_metrics.py, tests/test_fund_factors.py]
models: []
---

# Accounting analytics: DuPont, the Piotroski score, accruals and an anomaly screen

## In one sentence

A company's return on equity can be taken apart into five steps that multiply back to it exactly and whose contributions to a year's change add up exactly, nine yes-or-no signs can say whether its health is improving, and the gap between profit and cash can say how much of the profit is paper, but each of these is a screen that tells you where to look and none of them is a verdict.

## The idea

**DuPont.** Return on equity is the product of five steps, all on closing balances so the identity is exact: `roe = tax burden × interest burden × operating margin × asset turnover × equity multiplier`, where the tax burden is net income over pre-tax income, the interest burden is pre-tax income over operating income, the operating margin is operating income over revenue, the asset turnover is revenue over assets and the equity multiplier is assets over equity. The *change* from one year to the next is then split between the steps by the logarithmic-mean Divisia method: `contribution_i = L(roe, roe_prior) × ln(step_i / step_i_prior)` with `L(a, b) = (a − b) / ln(a / b)`. The contributions add up to the change exactly and do not depend on the order of the steps, which the usual "change one step at a time" attribution cannot claim. It needs every step positive in both years; otherwise the contributions are missing and the row says why.

**Piotroski's F-score** (2000) counts nine signs of improving financial health that hold: return on assets is positive, operating cash flow is positive, return on assets improved, operating cash flow exceeds net income, leverage fell, the current ratio rose, no new shares were issued, gross margin rose, asset turnover rose. A sign is missing, not "no", when an input is missing, and the score is missing if any sign is.

**Accruals and cash conversion.** Profit that is not backed by cash is a warning. The *accruals ratio* is net income less operating cash flow over closing assets, *Sloan accruals* use the average of this and last year's assets (Sloan 1996), and *cash conversion* is operating cash flow (or free cash flow) over net income, missing when net income is not positive.

**An accounting anomaly score of the Beneish kind** (1999) combines eight indices of how this year's accounts differ from last year's (receivables against sales, gross margin, asset quality, sales growth, depreciation, selling and administrative expense, leverage and total accruals) into one number; a high value means the year's changes resemble those of companies that were later found to have manipulated earnings. The weights are Beneish's, fitted on United States manufacturers of the 1980s and early 1990s; the cut-off (−1.78) is the one he gave at a 20:1 cost of missing a manipulator against flagging an honest company. It is the model's own sample, not a law.

## Why it matters

These are the questions a fundamental analyst asks before a number is used: *why* did return on equity rise (margin, efficiency, or only more debt?), is the improvement broad, is the profit real. They also make factors: a change in ROIC, accruals and the Piotroski score are all in the factor library, and the discovery search can combine them.

## How this repo uses it

`src/fundamentals/metrics.py` computes the ratios once from the point-in-time tables, with a version stamp, and `src/fundamentals/analytics.py` builds the DuPont table, the nine signals and the anomaly score on top of them. A company's page (Company explorer → Open) shows each: the ratios, the DuPont table with the contribution of each step to the change in ROE, a year-by-year matrix of the nine signals, and the eight Beneish indices with a reading. The rules that keep a ratio honest apply throughout: a denominator that cannot meaningfully divide (zero, negative, missing) makes the ratio missing, not huge; a return on equity needs positive equity; a missing input is missing, never zero; averages need both ends, so they are missing in a company's first year.

## What we found

The tests pin the arithmetic: the DuPont steps multiply to ROE, the contributions add up to the change in ROE, the split is refused when a step is not positive, and the first year has no prior and says so. On a company of the simulated demo world the last year's return on equity was 6.3%, up 0.15 points on the year before, split exactly between five steps (the sum of the five contributions equals the change to six decimals); its Piotroski score was 7 and its anomaly score was within the range of the model's ordinary companies. On the demo world as a whole, in the direction theory gives each, the Piotroski score has a t-statistic of 3.3 for its mean rank IC and the change in operating margin 3.6, against 8.0 for free-cash-flow yield, which is the effect that was planted: the quality-style factors carry some information in this world, and far less than the one built to carry it. The accruals ratio (−1.6) and the anomaly score (−2.7) came out the wrong way round for the theory, in a world where nothing was planted for them; with eighty-four factors looked at, one or two like that are what luck gives, which is why the scan says so.

What has not been found: how these screens behave on real companies. The Beneish model was fitted to United States manufacturers of the 1980s and early 1990s, and neither it nor the F-score was built for a global universe.

## Pitfalls

- **The anomaly score is not a fraud detector, and the application never says it is.** Beneish's cut-off treats a missed manipulator as twenty times worse than a false alarm, so it flags many honest companies by design, and the model was fitted on one country, one kind of company and one era.
- **A financial company's accounts do not fit.** Revenue, gross margin, receivables and assets mean different things for a bank or an insurer; the ratios are computed but a screen built for manufacturers says little about them.
- **Equity multipliers hide leverage.** A rise in ROE from a rise in the equity multiplier is more debt, not a better business; the DuPont split shows it, and a single ROE factor does not.
- **Negative or tiny equity** makes ROE undefined or huge, so it is missing, and so are the steps that need it.
- **Year-end balances, not averages**, give the exact identity; Sloan accruals use averages and are missing in a company's first year.
- **Reporting differences between countries** (what counts as operating income, how leases are treated) move these ratios; compare a company with its own past before comparing it with a foreign one.

## Try it

```python
from src.fundamentals.service import FundamentalsService
from src.fundamentals.workbench import Workbench

wb = Workbench(service=FundamentalsService(".", db_path="data/user/fundamentals_demo.db"))      # made by `quant fundamentals simulate --demo`
sid = wb.companies(limit=1)["companies"][0]["security_id"]
company = wb.company(sid)
last = company["dupont"][-1]
steps = [s["key"] for s in company["dupont_steps"]]
print(last["roe"], last["d_roe"], sum(last["c_" + k] for k in steps), last["method"])   # the contributions add up to the change
print(company["piotroski"][-1]["piotroski_f"], company["anomaly_reading"])
```
