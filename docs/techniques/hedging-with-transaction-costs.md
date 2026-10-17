---
title: "Hedging an option when every trade costs money"
slug: hedging-with-transaction-costs
difficulty: 3
chapter: Ch. 18
prerequisites: [option-pricing-and-greeks, option-chains-and-greeks]
stages: []
files: [src/derivatives/costs.py, src/assets/option_models.py, src/webapp/models_api.py, src/webapp/static/options.js]
figures: []
tests: [tests/test_costs.py, tests/test_option_models.py, tests/test_models_api.py, tests/test_webapp_models_browser.py]
models: []
---

# Hedging an option when every trade costs money

## In one sentence

Black-Scholes prices an option as the cost of a hedge that is rebalanced continuously and costs nothing; with a cost on every trade that hedge costs infinity, so the real question is how often to trade, how far from the Black-Scholes delta to stray, and what the option must therefore cost.

## The idea

Selling an option and hedging it with the underlying is risk-free only if the hedge tracks the option's delta at every instant. Each trade pays half the bid-ask spread, so tracking continuously is ruinous and tracking rarely leaves risk. Three answers, from the simplest to the most faithful:

**Leland (1985).** Rebalance every `dt`. A price move of one standard deviation, `sigma S sqrt(dt)`, calls for a trade of `Gamma` times that many shares, so the expected cost per unit of time is proportional to the option's gamma, which is what the volatility term `sigma^2 S^2 Gamma / 2` multiplies. The effect is Black-Scholes at a changed volatility, `sigma^2 (1 + Le)` for the seller and `sigma^2 (1 - Le)` for the buyer, with the **Leland number** `Le = 2 sqrt(2/pi) cost / (sigma sqrt(dt))`. The cost grows like `cost / sqrt(dt)`: halving the cost of a trade is worth as much as trading a quarter as often. When `Le >= 1` a buyer cannot replicate the option at a profit.

**Whalley and Wilmott (1997).** With exponential utility and trading allowed at any instant, the best hedge is a **no-trade band** around the Black-Scholes delta of half-width `(3 e^{-r tau} cost S Gamma^2 / (2 gamma))^{1/3}`, with trades only at its edge. The cost of hedging then grows like `cost^{2/3}`, much more slowly than the cost of hedging on a clock.

**Hodges-Neuberger (1989) and Davis-Panas-Zariphopoulou (1993).** In a binomial market an investor with exponential utility has a *price at which they are indifferent to writing* the option (the ask) and another *at which they are indifferent to buying* it (the bid). Dynamic programming over the holding gives both; with no cost they collapse to the Cox-Ross-Rubinstein price, with cost the option cannot be hedged at a profit between them, and the writer's optimal holding has a no-trade region whose edges are the Whalley-Wilmott band.

## Why it matters

The spread you pay is not a rounding error on a delta hedge: it is the main cost of running one. Knowing the Leland price tells you what to charge, the band tells you when *not* to trade, and the simulation shows that the rules differ in risk as well as in cost: trading less costs less and leaves more variance.

## How this repo uses it

`derivatives/costs.py` has `leland_number`, `leland_sigma`, `leland_price` and `leland_pde` (a grid for payoffs whose gamma changes sign, where the adjustment switches sign and the equation is non-linear), `whalley_wilmott_band`, `simulate_hedging` (sell at the Black-Scholes price, hedge by a rule on simulated paths, pay the cost on every trade including the first purchase and the final liquidation) and `dpz_prices` (the binomial tree, with the writer's no-trade band).

The **Options tab → Hedging cost** view takes an option on a chain (strike, expiry, call or put), the volatility (default the at-the-money one), the cost of one trade in basis points, the rebalancing dates and a risk aversion, and shows Leland's price against the number of rebalancing dates, the band around the delta, the distribution of profit and loss of three rules (delta on every date, Leland's adjusted delta, the band) and the utility-based bid and ask. `quant models hedge SPY --cost-bp 5` prints the same.

**Risk aversion is in units of the option's price:** a loss of the whole premium is worth that many units of utility. It makes the band's width independent of the price level of the underlying, which a risk aversion per dollar is not.

## What we found

- **The formulas hold where they should.** The Leland price brackets Black-Scholes and its grid reproduces it for a vanilla; the band follows its formula and scales as cost to the one-third; with no cost, discrete delta hedging is fair and its error falls with the frequency; the indifference prices equal the Cox-Ross-Rubinstein price at zero cost whatever the risk aversion and drift, widen into a bid and an ask with the cost, and the tree's no-trade band agrees with Whalley-Wilmott's for small costs and scales as risk aversion to the power minus one-third.
- **An example, at the money, 88 days, volatility 13.1%, 5 basis points a trade, 61 daily dates, 10,000 paths.** The Black-Scholes price was 22.39. Leland's price for the seller was 23.3 (4% higher, the cost of hedging as 0.6 volatility points, Leland number 0.10). Selling at 22.39 and hedging on every date lost 1.38 on average and paid 1.36 in costs. The band traded 39% as often and paid 51% of the cost, and its profit and loss was twice as spread out (standard deviation 4.2 against 2.25): what the band saves in costs it pays in risk. The utility-based tree gave a writer's price of 23.3 and a buyer's of 21.5 around a frictionless 22.39.
- **Selling at Leland's price pays for the hedge,** to first order: the Leland rule's mean loss was about a third of the plain delta hedge's. It is an approximation: it does not remove the risk.

## Going deeper

```
Le = 2 sqrt(2/pi) cost / (sigma sqrt(dt))         sigma_seller^2 = sigma^2 (1 + Le),  sigma_buyer^2 = sigma^2 (1 - Le)
cost of delta hedging on a clock   ~ cost / sqrt(dt)               (grows without bound as dt -> 0)
cost with a Whalley-Wilmott band   ~ cost^(2/3)                    H = (3 e^{-r tau} cost S Gamma^2 / (2 gamma))^(1/3)
DPZ:  value  -exp(-gamma x e^{r(T-t)}) phi(t, y, S);   ln phi recursion:  min over the new holding y' of
      [ (y' - y) S + cost S |y' - y| ] + ln E[ phi(t+1, y', S') ]
      writer's price  p_w = (ln phi_short - ln phi_none) / (gamma e^{rT}),   buyer's  p_b = (ln phi_none - ln phi_long) / (gamma e^{rT})
```
The tree has no dividend; the page runs it with the prepaid forward `S e^{-qT}`, which pays none, so its frictionless price is the Cox-Ross-Rubinstein price of the same option.

## Pitfalls

- **The cost here is proportional and the same for every trade.** Real spreads widen in a storm, exactly when a hedge must trade; the results are optimistic for that reason.
- Leland's adjustment is first-order and for a payoff of one sign of gamma; a spread of options needs the non-linear grid.
- A simulation hedges on paths of one model with one volatility: it says nothing about volatility that is mis-specified, which is usually the larger error.
- The utility-based prices depend on the risk aversion; the page expresses it in units of the option's price so that 1 means moderate and 0.1 very tolerant, but there is no market value of it.
- The binomial tree holds the stock in a limited range (about plus or minus 1.25 shares) and a finite grid of holdings; it is a teaching tool for the shape of the answer, not a quoting engine.

## Try it

```bash
quant models hedge SPY --cost-bp 5 --dte 90
quant models hedge SPY --cost-bp 25 --steps 21 --risk-aversion 0.5 --right put
python -m pytest tests/test_costs.py -q
```
In the dashboard: **Options → Hedging cost**, set the cost of a trade, press *Cost the hedge*.
