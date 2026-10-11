---
title: "Reading an option chain: smiles, skew, Greeks and payoffs"
slug: option-chains-and-greeks
difficulty: 2
chapter: Ch. 18
prerequisites: [option-pricing-and-greeks]
stages: []
files: [src/assets/options_analytics.py, src/webapp/asset_api.py, src/webapp/static/options.js, src/derivatives/synthetic.py, src/assets/free_data/options.py, src/webapp/data_api.py]
figures: []
tests: [tests/test_assets_workspaces.py, tests/test_free_data_options.py, tests/test_webapp_data.py]
models: []
---

# Reading an option chain: smiles, skew, Greeks and payoffs

## In one sentence

An option chain is a table of quotes that you read through what it implies: the volatility each strike trades at, how that volatility changes across strikes and expiries, and what a position built from it is exposed to.

## The idea

Every row of a chain is a call or a put at one strike and one expiry, with a bid, an ask, a volume and an open interest. The price on its own says little; the **implied volatility** (the volatility that makes the Black-Scholes price equal the quote) puts every row on one scale. Plotted against strike, implied volatility is not flat: it is usually higher for low strikes (the *skew*, the price of crash insurance) and curves up at both ends (the *smile*). Plotted against expiry it gives the *term structure*; plotted against both it is the *surface*. A position is a handful of rows with quantities, and its **Greeks** are the sensitivities of its value: delta to the price of the underlying, gamma to the change in delta, vega to a volatility point, theta to a day passing, rho to rates. A **payoff diagram** is the position's profit at one date as a function of the underlying's price, and its break-evens are where it crosses zero.

## Why it matters

A chain is easy to misread. A wide bid-ask spread, a quote that has not updated, or an open interest of zero makes a price that looks cheap untradeable. A position that looks hedged on delta can still lose money on a volatility move. Seeing the smile, the Greeks and the stress table together is what separates "I own an option" from "I know what I own".

## How this repo uses it

The **Options** tab of the dashboard (`src/assets/options_analytics.py`, served by `src/webapp/asset_api.py`) is read-only first: an overview (at-the-money implied volatility, volatility rank and percentile against the underlying's own history, put/call ratio, term-structure and skew state), the chain itself (calls and puts side by side with the at-the-money row marked, filters by expiry and moneyness, spread, volume, open interest and a stale-quote warning), the smile and term structure, the surface, and a risk view with the position Greeks in dollars and a stress table of price and volatility shocks. A strategy builder prices a single trade (long call, covered call, vertical spreads, straddle, strangle, butterfly, condor, calendar, ratio spread and others) and reports the payoff at expiry and today, the maximum profit and loss, break-evens, the Greeks, an illustrative margin estimate, the probability of profit and the expected value under a lognormal view you can change. A last view backtests volatility strategies through the platform's options engine.

**Provenance is part of the data.** The repository holds no option quotes. For an ETF in the platform's universe the chain is *generated*: the platform's arbitrage-free SSVI surface anchored to the underlying's real last close and real realised volatility, with an assumed rate and yield and simulated volume, open interest and quote ages. Every payload says `simulated` and the page prints "Simulated: a model" above every table. A vendor CSV in `data/user/options/` is analysed exactly as it is and labelled "Your file".

**Real chains without sourcing them.** The **Real option chains (free)** card of the tab downloads a chain from Cboe's delayed quotes (about 15 minutes late; Yahoo Finance when Cboe cannot be reached) for any US stock, ETF or index with listed options, or Bitcoin and Ether options from Deribit, and saves it under `data/user/downloads/options/`. The chain appears in the Underlying menu as "SPY (real chain, …)", is analysed like any other and is badged "Downloaded: free data". A vendor chain gives prices but not the rate and the dividend yield they were struck with, so the page assumes the rate (4% a year, stored with the download) and reads the dividend yield of **each expiry from put-call parity at the money** (the forward the market implies); Deribit publishes its forwards, which are used as they are. The vendor's own implied volatilities are ignored and every row is inverted from its mid, so a downloaded chain and a model chain are treated the same. [What is downloaded and what it is not](../free_data.md).

## What we found

The tests pin the machinery, not a market. The generated chain is reproducible and priced off the model surface; its term structure has no calendar arbitrage (total variance rises with expiry) and a sane slope; its skew has the equity sign (low-strike puts above calls) and the delta-based strikes are consistent; the surface view reports the static-arbitrage checks (butterfly and calendar) and finds none larger than one price tick; the surface grid is never extrapolated beyond the quoted strikes. A position's Greeks equal the sum of its legs', and a full revaluation matches the Taylor expansion for small shocks. A long straddle's break-evens sit one premium either side of the strike, a short straddle is reported as unlimited loss with a margin estimate, and the value of a strategy today starts at the cost of trading it. None of this says anything about a market, because the quotes are a model's. For downloaded chains the tests pin the plumbing the same way: Cboe's contract names become expiry, strike and right, and the quote date is the session's, not today's; unusable rows (no offer, a crossed market, an expiry out of range) are dropped and counted; the dividend yield implied by put-call parity recovers the one that priced a generated chain, and the implied volatilities that come back out of the file recover the volatility that priced it; Deribit's coin-unit prices are converted to dollars and its forward is read directly; Yahoo takes over, and the chain says so, when Cboe cannot be reached.

## Going deeper

```
implied vol:     solve  BSM(S, K, T, r, q, sigma) = mid          (a root find per row)
total variance:  w(k, T) = sigma(k, T)^2 * T,   k = ln(K / F)      (the SSVI surface is parametrised in w)
skew (25 delta): sigma(25-delta put) - sigma(25-delta call)
position vega:   sum_i  q_i * mult * vega_i                         (dollars per volatility point)
probability of profit under a lognormal view:  P( payoff(S_T) > 0 ),  S_T ~ lognormal(F, view vol)
```
A butterfly with a negative price, or total variance that falls as expiry grows, would be a static arbitrage: the chain checks both.

## Pitfalls

- A simulated chain is a way to check that the analytics and the page behave; its implied volatilities are not a signal and its spreads are not what you would pay.
- Mid-price implied volatility ignores the spread: at wide spreads the volatility could be several points either way.
- Black-Scholes Greeks assume one volatility per option, so a stress that moves the spot without moving the smile understates what a real surface does.
- The margin estimate is an illustrative rule of thumb, not an exchange or broker requirement.
- Probability of profit is a statement about a distribution you chose, not about the market.
- A downloaded chain is one delayed snapshot, not a history: the volatility rank, which needs a year of implied volatility, reads "n/a" for it. Its quotes can be stale (the page counts them), and the bid and ask of a far out-of-the-money option are often a few cents wide whatever the option is worth.
- Stock and ETF options can be exercised early and are priced as European, so deep in-the-money puts look slightly cheap against parity and the arbitrage view checks only live quotes within 5% of the forward. An implied dividend that moves a lot from one expiry to the next is a sign of thin quotes, not of a dividend policy.

## Try it

```bash
quant serve            # the Options tab: pick an underlying, an expiry, then open Strategy
python -m pytest tests/test_assets_workspaces.py -q -k options
```
To analyse a real chain without any file, open the Options tab, tick SPY (or type any ticker) in the **Real option chains (free)** card and press *Download real data*. To bring your own, save a vendor CSV as `data/user/options/<NAME>.csv` (column spellings as accepted by `derivatives.synthetic.load_option_csv`) and choose `file:<NAME>` as the underlying: the badge changes to "Your file" and implied volatilities missing from the file are inverted from the mid prices.
