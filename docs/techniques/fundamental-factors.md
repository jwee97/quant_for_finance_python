---
title: "Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects"
slug: fundamental-factors
difficulty: 2
chapter: Platform
prerequisites: [information-coefficient, cross-sectional-factors, alpha-model-construction]
stages: []
files: [src/equity/fundamentals.py, src/equity/factors.py, src/equity/dcf.py, src/equity/contextual.py, src/strategies/fundamental_factors.py, experiments/equity_world.py]
figures: []
tests: [tests/test_equity_fundamentals.py, tests/test_equity_dcf.py, tests/test_equity_contextual.py, tests/test_fundamental_models.py, tests/test_equity_world_experiment.py]
models: [fundamental_value, fundamental_quality, fundamental_momentum, fundamental_dcf, fundamental_alpha, fundamental_nonlinear]
---

# Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects

## In one sentence

Twenty-seven stock characteristics built from company statements and analyst estimates (cheapness, quality of earnings and of the balance sheet, momentum and revisions, discounted cash flow), each computed only from what was public on the date, then combined by the [alpha model](alpha-model-construction.md), with squares, products and bucket-by-bucket weights for effects that are not a straight line.

## The idea

**Value** asks what a dollar of market value buys. Operating measures are divided by enterprise value (market value plus debt and preferred less cash): cash flow from operations (`cfo2ev`), EBITDA (`ebitda2ev`), sales (`s2ev`). Shareholder measures are divided by market value: trailing earnings (`earnings_yield`), consensus earnings for the year (`forward_earnings_yield`), book equity (`b2p`), and dividends plus buybacks less issuance (`payout_yield`). `nxf2ev` is net external financing over enterprise value: a company that raises money is the opposite of one that returns it.

**Quality** asks how well the business is run and how it funds itself. `rnoa` is after-tax operating income over average net operating assets; `cfroi` is the internal rate of return of the asset base (the return that equates gross investment with the gross cash flow earned over the life of the plant plus the non-depreciating assets recovered at the end); `operating_leverage` is costs over assets, and its change. The rest are growth measures the literature finds to be bad news: the increase in working capital (accruals) and in net non-current operating assets, capital expenditure over its three-year average (`icapx`) and its growth (`capxg`), external financing over assets (`xf`) and the growth in the share count (`share_inc`).

**Momentum and revisions.** `ret1` is last month's return (it reverses, so the model buys the losers), `ret9` the nine-month return ending a month ago (it continues), `adj_ret9` the same per unit of its own volatility. With analyst columns: `earn_rev9` is the nine-month change in consensus earnings per share over price, `earn_diff9` the share of analysts revising up less those revising down, averaged over nine months, `ltg_rev9` the change in the long-term growth forecast.

**Valuation.** A discounted cash flow takes free cash flow, grows it at a first-stage rate that fades to a terminal rate, discounts it at a rate built from the stock's own beta, subtracts net debt and divides by shares. The multipath version (MDCF) draws the growth rate, the starting cash flow, the discount rate and the terminal rate from distributions around those inputs and values every draw: the factor is the median over the price, or the share of draws above the price. Value is convex in the discount rate, so uncertainty adds value to a long-lived claim, which a point estimate cannot show.

**Contextual and nonlinear.** A factor does not work the same everywhere (value may pay among slow-growth stocks and not among fast-growth ones), so the alpha model can run separately in each tercile of value, growth, earnings variability or size. And returns are not always a straight line in a factor: capital expenditure above and below the firm's norm can both be bad, which a linear IC cannot see. Squares, products and conditional terms (`quadratic:icapx`, `interaction:b2p:rnoa`, `conditional:ret9:b2p:high`) enter a Fama-MacBeth forecast.

## Why it matters

Fundamentals are where equity quant research differs most from trading prices: the data is late, restated and revised, and the easiest way to produce a spectacular backtest is to use a figure before the market had it. Every figure here is used from its filing date and no earlier, and it goes stale after a limit. A dozen related ratios are not a dozen ideas, so the same data has to be able to tell which factor is carrying the information.

## How this repo uses it

This repository carries prices, not statements, so the models read a file you supply: `data/user/fundamentals.csv`, one row per company and filing, with `ticker`, `period_end` (and `available`, the first date the figures were public; if absent it is `period_end` plus 60 days) and any of the columns listed in `src/equity/fundamentals.py` (sales, cogs, ebitda, net_income, cfo, capex, total_assets, book_equity, shares_outstanding, eps_fy1, and so on; flows are trailing twelve months, balance-sheet items as of the period end). Analyst columns can be rows of their own. A factor whose columns the file lacks is skipped in a composite and refused by name when asked for alone; without the file the models refuse to run, except the price momentum factors. The data/user folder is not part of the repository.

`fundamental_value`, `fundamental_quality` and `fundamental_momentum` take one factor or the equal-weighted composite of a style (each factor standardised across stocks and turned round where the literature expects the high end to be bad); `fundamental_dcf` takes `dcf_upside`, `mdcf_upside` or `mdcf_prob`; `fundamental_alpha` runs the [alpha model](alpha-model-construction.md) on factors and styles you list, optionally orthogonalised and optionally per bucket; `fundamental_nonlinear` adds the terms you list. All are monthly decisions held between dates, causal (checked by the generic test), and combine with the `optimal_ic` rule like any other models.

To test the tools without data, `src/equity/synthetic.py` simulates a world: companies with internally consistent quarterly statements (the balance sheet balances, earnings follow from operating income, interest and tax), monthly analyst fields, and daily prices whose expected returns depend on six drivers with known premia (log book to price, return on operating assets, an inverted U in abnormal capital expenditure, momentum, reversal and estimate revisions). The library's factors, computed independently from the statements, are rank-correlated 1.000 with the drivers the simulation used.

## What we found

`python -m experiments.equity_world` (80 companies, 14 years, seed 0, nothing tuned): the monthly rank IC and its t-statistic, from 36 months in, for the factors that carry the planted drivers: book to price 0.056 (t 5.3), nine-month momentum 0.042 (4.4), the one-month reversal -0.044 (-4.6, the sign that was planted), the earnings revision 0.061 (5.8), return on operating assets 0.034 (3.1). The value yields that share the market value in the denominator also look good (0.062 to 0.073), because cheapness is in all of them: stand-alone ICs say little about which one is the driver (see the [alpha model guide](alpha-model-construction.md)). A linear IC on capital expenditure is -0.018 (t -1.8); with its square in a Fama-MacBeth forecast, the square's slope is -0.003 a month per standard deviation (t -4.6), the planted size, and the forecast's IC rises from 0.075 to 0.083. The DCF factors score 0.07 in this world because free cash flow over price is a value measure too; neither says what a DCF would do on a real market.

What this does not show: that any of these earn their premium in real markets after costs. The simulated world pays what it was told to pay; it is a test of the plumbing (the arithmetic of each factor, the point-in-time rule, the walk-forward weights), and it is only as good as the definitions, which are stated in the code and the table above.

## Going deeper

```
EV           = price * shares + debt + preferred + minority interest - cash          (not meaningful if <= 0)
rnoa         = operating_income (1 - tax) / mean( NOA_t, NOA_(t-1y) ),   NOA = (assets - cash - LT investments) - (liabilities - debt)
cfroi        : solve  GI = GCF (1 - (1+r)^-N) / r + NDA / (1+r)^N   for r;  GCF = NI + depreciation + interest,  GI = assets + accumulated depreciation,  N = gross plant / depreciation
icapx        = capex_t / mean( capex_(t-1y), capex_(t-2y), capex_(t-3y) ) - 1
xf           = ( issuance - buybacks - dividends + debt issued - debt repaid ) / mean( assets_t, assets_(t-1y) )
adj_ret9     = ( P_(t-21) / P_(t-210) - 1 ) / ( daily vol over the same 189 days * sqrt(189) )
DCF          : FCF_t = FCF_(t-1)(1+g_t),  g_t from g_1 to g_T linearly over N years;  EV = sum FCF_t/(1+r)^t + FCF_N (1+g_T)/(r-g_T)/(1+r)^N
               with g_1 = g_T = g this is  FCF_0 (1+g)/(r-g)
nonlinear    : slopes_s = OLS of r_(s+1) on [1, z, z^2, z_a z_b ...] across stocks;   forecast_t = mean(last slopes with s <= t - 1) . terms_t
```

## Pitfalls

- A statement file is only as good as its dates. If `available` is really the fiscal period end, the strategy trades on figures the market did not have; leave the column out and let the 60-day default apply unless you know the filing dates.
- Restated figures: a database that shows the latest version of a past year is not point-in-time. Use a source that keeps the as-first-reported value, or accept that the backtest flatters.
- The definitions differ between papers and vendors (operating leverage, CFROI, abnormal investment). These are stated here so you can change them, not so you can compare numbers with a published table.
- Shares must be on the same split-adjusted basis as the prices, in the same units as the dollar columns; a stock split that is applied to one and not the other moves a stock across the cross-section.
- With 10 to 20 stocks a tercile has three to six names. Contextual models need a wide cross-section; `min_assets` in the code exists to stop a bucket from learning from a handful.
- DCF values are dominated by the terminal term. A discount rate close to the terminal growth rate makes the value explode; the code keeps the two apart and caps the first-stage growth, and neither is a substitute for judgement.

## Try it

```bash
python -m experiments.equity_world
# with your own statements in data/user/fundamentals.csv:
quant backtest --model fundamental_value --param factor=cfo2ev --tearsheet
quant backtest --model fundamental_alpha --param factors=value,quality,momentum --param context=growth --tearsheet
quant backtest --model fundamental_dcf --param factor=mdcf_prob
quant backtest --model fundamental_nonlinear --param terms=quadratic:icapx
```
