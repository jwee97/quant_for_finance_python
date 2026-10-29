---
title: "Option structures, the distribution the quotes imply, and where the open interest sits"
slug: option-structures-distribution-and-positioning
difficulty: 3
chapter: Platform
prerequisites: [option-chains-and-greeks, option-strategies-and-vol-premium, volatility-surfaces]
stages: []
files: [src/assets/options_analytics.py, src/derivatives/strategies.py, src/derivatives/backtest.py, src/derivatives/surface.py, src/webapp/asset_api.py, src/webapp/static/options.js, src/webapp/static/charts.js, src/webapp/static/workspace.js, src/references.py]
figures: []
tests: [tests/test_options_chain_exact.py, tests/test_options_strategies_exact.py, tests/test_webapp_assets_ml.py, tests/test_webapp_workspaces_browser.py]
models: []
---

# Option structures, the distribution the quotes imply, and where the open interest sits

## In one sentence

The Options tab now draws twenty-five option structures (eleven new) with the odds, the profit by price and day and the Greeks by price beside each, reads the distribution of the price that a chain implies and the volatility the VIX formula finds in it, maps where the open interest and the gamma exposure sit, and backtests eleven rolling strategies (four new, from the Cboe benchmark indices) with a filter on the variance premium, a sweep of one setting and a comparison of all of them.

## The idea

**Structures.** A structure is a set of legs whose payoff at expiry is the sum of their intrinsic values less what they cost. The eleven new ones, with where each comes from: the **collar** (own the shares, buy a put below and sell a call above: the Cboe CLL and CLLZ indices; Israelov and Klein 2016), the **cash-secured put** (the Cboe PutWrite index), the **iron butterfly** (the Cboe BFLY index), the **long condor**, the **risk reversal**, the **strap** and the **strip** (Hull's textbook), the **call and put backspreads** (Natenberg), the **diagonal spread**, and the **tail hedge** (a put far below the market, the insurance Taleb 1997 argues for and Bondarenko 2014 shows to be dear). The structures are standard and catalogued by McMillan (2012), Natenberg (2014), Hull (2018) and Cohen (2005); the tab lists the sources of each under its description.

**The distribution the quotes imply.** Breeden and Litzenberger (1978) showed that the price of a call, differentiated twice with respect to the strike, is the market's probability density for the price at expiry (times a discount factor): a butterfly of very narrow width pays one dollar if the price ends in its band. The quotes are too noisy to difference directly, so the tab first fits a smooth curve to the expiry's out-of-the-money implied volatilities (the SVI curve of Gatheral and Jacquier 2014, which `derivatives.surface` already fits) and reads the density from it (Figlewski 2010 and Jackwerth and Rubinstein 1996 discuss the choices and the tails). It is drawn against the lognormal with the at-the-money volatility, with quantiles, the chance of a fall or a rise of 5, 10, 15 and 20% against the lognormal's, and the move the at-the-money straddle prices in (a straddle is worth about 0.8 of a standard deviation of a normal move).

**Model-free volatility.** The price of a variance swap does not depend on a model: Britten-Jones and Neuberger (2000), Demeterfi, Derman, Kamal and Zou (1999) and Neuberger's log contract (1994) show that the out-of-the-money options of every strike, weighted by `1 / K^2`, replicate it. The Cboe's VIX is this formula on the S&P 500's options, interpolated to 30 days. The tab applies the Cboe rules (a put below the strike at or just under the forward, a call above, the average of the two at it, each at the mid, stopping after two strikes in a row without a bid) to every expiry and sets the result beside the at-the-money volatility: the difference is what the skew adds. Jiang and Tian (2005) find that this measure carries more information about future volatility than the at-the-money one.

**The cone.** Burghardt and Lane (1990) ask whether options are cheap by setting their volatility against the range of the volatility the underlying has really had over windows of the same length: the volatility cone. The tab draws it from the underlying's own history, with the chain's at-the-money volatility at each length on top.

**Open interest and gamma.** For each strike, how many contracts are outstanding, the strikes with the most of them (the walls), and the *max pain* strike of an expiry, the settlement at which the open interest would pay its holders least. The *gamma exposure* is the change in the dollar delta of all the open contracts for a 1% move, `Gamma x OI x 100 x S^2 / 100`, calls positive and puts negative, and the *flip* is the price at which its sign changes. Open interest has no sign, so this convention is an assumption: it follows if the dealers hold the other side of customers who write calls and buy puts. The evidence that option positions move prices is real but different from the chart: prices cluster at strikes with large open interest near expiry (Ni, Pearson and Poteshman 2005; Ni, Pearson, Poteshman and White 2021), the demand for options moves their prices (Garleanu, Pedersen and Poteshman 2009; Bollen and Whaley 2004), and a gamma imbalance built from signed trades predicts returns (Barbon and Buraschi 2020).

**Four more rolling strategies, and three ways to tune them.** The `put_spread` (a credit put spread), `iron_butterfly` (the Cboe BFLY index: the at-the-money put and call sold, wings 5% out of the money bought), `collar` (a put at 95% and a call at 105% of the price, on top of the shares) and `protective_put` (the Cboe PPUT index) join the seven that were there. A variance-premium filter trades any of them only while implied volatility exceeds the trailing realised volatility by a margin you set (Bollerslev, Tauchen and Zhou 2009; Carr and Wu 2009). A **sweep** runs the same backtest at two to nine values of one setting, so that the dependence of the result on it is drawn, and a **comparison** runs every strategy at its defaults on one market.

## Why it matters

A payoff diagram says what a structure pays at expiry and nothing about whether it is a good bet. The odds, the profit by price and day and the structures beside each other answer that. The implied distribution says how much of a fall the market is charging for, which a single at-the-money volatility hides; the model-free volatility is the number the VIX reports, so it is what a variance trade is priced on. A sweep says whether a setting was found or fitted.

## How this repo uses it

`src/assets/options_analytics.py` has the structures (`strategy_legs`, `STRATEGY_GROUPS`, `named_structure`, `compare_structures`, `strategy_analysis` with its `density`, `grid` and `greeks_curve`), the chain analytics (`implied_distribution`, `expected_moves`, `model_free_variance`, `model_free_term`, `volatility_cone`, `positioning`, `max_pain`) and the backtest (`run_strategy_backtest` with `timing`, `run_strategy_sweep`, `run_strategy_comparison`). `src/derivatives/strategies.py` has the four new rolling strategies. The routes are `GET /api/options/distribution` and `GET /api/options/positioning`, `POST /api/options/strategy/payoff` (with `compare`), `/strategy/backtest`, `/strategy/sweep` and `/strategy/compare`. The backtest page draws the drawdown, the volatility through time, the years, months and distribution of days and every risk measure with the same code as the other tabs (`risk.performance`, `WS.riskCharts`).

## What we found

**The machinery recovers what it should.** A chain priced at one volatility gives back the lognormal density to a fifth of a per cent, its quantiles and tail chances to the same, and a model-free variance of exactly the volatility squared; the straddle is worth 0.8 of a standard deviation; the Cboe formula is worked strike by strike by hand in the tests, including the forward between strikes and the two-missing-bids cut-off; the payoff of every structure is the sum of its legs, and each new one has its maximum, minimum and break-evens checked against its strikes and its price.

**On a real chain it reproduces the VIX.** The tests carry the SPX options of 29 October 2026 (Cboe's delayed quotes, as the Free data card downloads them) on the two expiries around 30 days, 29 and 36 days out. The model-free volatility of the two, interpolated to 30 days by the Cboe's own rule, is **17.14%, against the 17.20 the exchange published for the VIX** at 16:15 ET (15 minutes after the chain's quotes). The at-the-money volatility interpolated to the same 30 days is 13.38%: most of what the VIX says is the skew. For the 27 November expiry the implied distribution has a skewness of -1.02 and an excess kurtosis above 4; it puts **2.3% on a fall of 10%, seven times the lognormal's 0.33%** and several times its own chance of a rise of that size (0.4%); the straddle prices in a move of 3.0% against one standard deviation of 3.8%. It is one snapshot of one index: the test pins what the code gives for it and its agreement with the published figure, not a study of the VIX.

**The four new strategies on the simulated option market.** Run on the same three simulated years as the others ([the table](../institutional_findings.md), section 8): the put spread had a Sharpe ratio of +0.35 against the short put's +0.72, at twice the costs: the put it buys gives back almost all of the short put's return (0.2% against 2.6% over the 330 days) and caps the worst fall at -3.9% against -5.5%. The iron butterfly lost 4.3% (Sharpe -0.86): the at-the-money premium is the dearest to sell against a market with jumps, and the wings and $81 of costs take the rest. The collar earned a Sharpe of +0.03 and the protective put +0.18, against +0.31 for the covered call: in this market the insurance, and the call that pays for part of it, both cost. Three simulated years are too few to rank strategies (the put spread's Sharpe was -0.15, +1.49 and -0.28 in the three), and the market pays the variance premium it was built with: the table says the books balance, not which structure to use.

## Going deeper

```
density:          q(K) = e^{rT} d2C/dK2                       (Breeden-Litzenberger)
SVI total var:    w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + s^2)],   k = ln(K / F)
density of k:     q(k) = g(k) / sqrt(2 pi w) exp(-d2^2 / 2),   g = (1 - k w' / 2w)^2 - (w'^2 / 4)(1/w + 1/4) + w'' / 2
model-free var:   sigma^2 = (2 / T) e^{rT} sum_i (dK_i / K_i^2) Q(K_i) - (1 / T)(F / K0 - 1)^2        (Cboe; Demeterfi et al.)
30-day figure:    100 sqrt( [T1 s1^2 (T2 - t) + T2 s2^2 (t - T1)] / (T2 - T1) / t ),   t = 30 / 365
straddle move:    (C + P) at the forward = S [2 N(sigma sqrt(T) / 2) - 1]  ~  0.8 sigma sqrt(T) S
gamma exposure:   GEX = sum_calls Gamma OI 100 S^2 / 100  -  sum_puts Gamma OI 100 S^2 / 100
max pain:         argmin_S  sum_K OI_call(K) max(S - K, 0) + OI_put(K) max(K - S, 0)
```

## Pitfalls

- The distribution is **risk-neutral**: it has the market's price of risk inside it, so it puts more weight on bad outcomes than they have had in practice. It is not a forecast of the price.
- The SVI curve is fitted to the quotes that exist and extended beyond them: the far tails of the density are the curve's, not the market's. A fit that misses the quotes by points (the tab prints how far) gives a shape not to trust.
- The model-free volatility is cut off at the lowest and highest strikes quoted, and is only as good as the quotes: it is slightly too low where the wings are thin. A delayed or stale quote moves it.
- Open interest has no sign. The gamma exposure is a map of where the option sensitivity sits under one convention about who holds what, not a statement about dealers' books. Max pain is a folk rule for a single expiry.
- A sweep over nine values is nine looks at one market: the best of them is optimistic, and a Sharpe ratio that peaks at one value and falls either side is a fit. The simulated market pays the premium it was built with, so the sellers of premium lead the comparison by construction.
- Strategies that hold shares are measured on the same fixed capital as those that do not; margin and funding are not modelled.

## Try it

```python
from src.assets import options_analytics as oa

chain = oa.prepare_chain(oa.build_model_chain(500.0, "2026-10-29", 0.15, 0.18, 0.04, 0.013, 0.15, "SPY"))
dist = oa.implied_distribution(chain, oa.nearest_expiry_to(chain, 30))
print({k: round(v, 3) for k, v in dist["moments"].items()}, round(oa.model_free_term(chain)["iv_30d"], 4))      # skewness, kurtosis and the VIX-style 30-day volatility
print(oa.run_strategy_sweep("put_spread", "short_delta", [0.15, 0.25, 0.35], n_days=252, seed=1)["rows"][1]["sharpe"])
```

To use the real chain, tick SPY or SPX in the **Real option chains (free)** card of the Options tab and open *Implied distribution*.

Sources: [Breeden and Litzenberger (1978)](https://doi.org/10.1086/296025), [Gatheral and Jacquier (2014)](https://doi.org/10.1080/14697688.2013.819986), [Figlewski (2010)](https://doi.org/10.1093/acprof:oso/9780199549498.003.0015), [Jackwerth and Rubinstein (1996)](https://doi.org/10.1111/j.1540-6261.1996.tb05219.x), [Britten-Jones and Neuberger (2000)](https://doi.org/10.1111/0022-1082.00228), [Demeterfi, Derman, Kamal and Zou (1999)](https://doi.org/10.3905/jod.1999.319129), [Neuberger (1994)](https://doi.org/10.3905/jpm.1994.409478), [Jiang and Tian (2005)](https://doi.org/10.1093/rfs/hhi027), [Cboe (2025), Volatility Index mathematics methodology](https://cdn.cboe.com/resources/indices/Cboe_Volatility_Index_Mathematics_Methodology.pdf), [Burghardt and Lane (1990)](https://doi.org/10.3905/jpm.1990.409259), [Ni, Pearson and Poteshman (2005)](https://doi.org/10.1016/j.jfineco.2004.08.005), [Ni, Pearson, Poteshman and White (2021)](https://doi.org/10.1093/rfs/hhaa082), [Gârleanu, Pedersen and Poteshman (2009)](https://doi.org/10.1093/rfs/hhp005), [Bollen and Whaley (2004)](https://doi.org/10.1111/j.1540-6261.2004.00647.x), [Barbon and Buraschi (2020)](https://doi.org/10.2139/ssrn.3725454), [Israelov and Klein (2016)](https://doi.org/10.3905/jai.2016.19.1.041), [Bondarenko (2014)](https://doi.org/10.1142/s2010139214500153), [Bollerslev, Tauchen and Zhou (2009)](https://doi.org/10.1093/rfs/hhp008), [Carr and Wu (2009)](https://doi.org/10.1093/rfs/hhn038), [Cboe (2015), strategy benchmark indexes](https://ir.cboe.com/news/news-details/2015/CBOE-Introduces-10-Options-Based-Strategy-Performance-Benchmark-Indexes-07-29-2015/default.aspx); and the books McMillan (2012), *Options as a Strategic Investment*, Natenberg (2014), *Option Volatility and Pricing*, Hull (2018), *Options, Futures, and Other Derivatives*, Cohen (2005), *The Bible of Options Strategies*, Taleb (1997), *Dynamic Hedging*, Gatheral (2006), *The Volatility Surface*.
