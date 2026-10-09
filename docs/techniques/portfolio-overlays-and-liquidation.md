---
title: "Portfolio overlays and liquidation costs: beta-neutral hedges, liquidity caps, optimisation with trading costs"
slug: portfolio-overlays-and-liquidation
difficulty: 3
chapter: Ch. 22
prerequisites: [costs-constraints-and-capacity, execution-and-impact, mean-variance-and-shrinkage]
stages: []
files: [src/framework/allocators_overlay.py, src/algo/liquidation.py, src/algo/impact.py]
figures: []
tests: [tests/test_allocators_overlay.py]
models: []
---

# Portfolio overlays and liquidation costs: beta-neutral hedges, liquidity caps, optimisation with trading costs

## In one sentence

Three allocators that take a portfolio and change it for a reason that is not the forecast (remove its market exposure, make it sellable in a few days, make it pay for the trades it asks for), and the arithmetic of what it costs to get out.

## The idea

**Hedging and market neutrality: `beta_neutral`.** A book that is long what it likes is also long the market, and its profit and loss mixes the stock-picking it wants to measure with the market move it does not. The overlay estimates each asset's beta to the equal-weight average of the investable assets over a trailing window and removes the book's net beta (down to `target_beta`). `hedge=""` subtracts the multiple of the beta vector that makes the net beta zero with the smallest change to the weights; `hedge="SPY"` instead sells exactly that much of one instrument.

**Liquidation risk: `liquidity_cap`.** With the fund worth `aum` dollars, an asset trading `ADV` dollars a day and a `participation` share of volume you are willing to be, the weight that can be sold in `days` days is `participation * days * ADV / aum`. The overlay clips every weight to that limit; what is clipped stays in cash. An asset whose volume is unknown gets no position, because there is no evidence it can be sold.

**Liquidation cost: `liquidation_profile`.** For a position of `X` dollars in an asset with dollar volume `ADV` sold at a participation rate `p` it takes `X / (p ADV)` days. The profile prices a constant-rate liquidation with the cost model of the execution algorithms: half the spread, power-law temporary impact `eta sigma (q / V)^beta` on each day's slice, the permanent impact `gamma sigma X / (2 ADV)`, and the timing risk of the shares still held at the end of each day. `liquidation_horizon` gives the days to turn a share of the whole book into cash. [`quant cashflow redeem`](cash-flow-strategies.md) compares ways of meeting a redemption with it.

**Portfolio optimisation with transaction-cost analysis: `tca_mvo`.** Plain mean-variance chases forecast noise with expensive turnover. Here each month-end solves `maximise mu'w - (risk_aversion / 2) w'Sigma w - periods * sum_i cost_i(|w_i - h_i|)` where `h` is what the book holds after drifting and `cost_i(x) = a x + c x^(1 + beta) + d x^2` is the spread, the temporary impact and the permanent impact of trading `x` of the fund in asset `i`, from its volatility and the fund's size against its dollar volume. The cost is convex and its slope at zero is the half-spread, so an asset whose expected gain over the risk it adds is smaller than that is left alone (a no-trade region), and a large trade is spread over several months where a small one is not. This is the portfolio-level counterpart of the trade-schedule optimisation in [execution algorithms](execution-algorithms.md); Garleanu and Pedersen (2013) solve a dynamic version with quadratic costs in closed form.

## Why it matters

Hedged, capped and cost-aware books answer different questions from the unadjusted one: what would this have earned without the market, how much of it could have been sold when it mattered, and what is left after paying for the turnover it asks for. The first is the usual defence against mistaking beta for skill; the second is the usual cause of losses in a crowded exit.

## How this repo uses it

The overlays wrap another allocator through `inner` (`{"allocator": "forecast_stack"}` or `{"book": "risk_parity"}`; by default `sleeves` for a single per-asset strategy and `forecast_stack` otherwise) and use only data through each date they stamp. Choose them in the dashboard's portfolio list. `tca_mvo` needs forecasts and trading volume and takes `aum`, `spread_bps` and the impact parameters of the execution model (`eta`, `beta`, `gamma`; illustrative defaults from the equity literature). The tests check the hedged book's beta is zero to machine precision and its realised correlation with the market near zero out of sample; that the cap holds and is exactly binding on an illiquid asset; that the optimiser agrees with brute force on a two-asset problem, reduces to plain mean-variance with costs off, leaves the book alone when the edge is smaller than the spread, and trades far less than the cost-free optimiser on a forecast that flips every month; and that the liquidation numbers equal their closed forms.

## What we found

On the 15 ETFs with `xs_momentum` as the forecast (one run each, 10 bps costs, not a study): the long-only forecast book earned a net Sharpe of +0.50 with 7.7% volatility and a market beta of +0.40. `beta_neutral` by projection took the beta to +0.03, halved the volatility to 3.8% and kept the Sharpe (+0.48): the signal was not just the market. Hedging with one instrument (SPY) took the beta to zero as well but the Sharpe to 0.00, because shorting the best-performing fund over these years cost what the book had earned; the choice of hedge is a bet. `liquidity_cap` at $5bn of assets (10% of volume for 5 days) cut turnover from 4.2x to 1.7x and the Sharpe to +0.38. With the same forecasts, `tca_mvo` at $100m traded 0.6x a year for 4 bps of costs and a net Sharpe of +0.88, against +0.79 and 1.6x turnover with its costs switched off; a Black-Litterman book on the same forecasts earned +0.95. These are single runs on twenty years of fifteen funds: the ordering is an illustration of what the overlays do, not evidence that one is better.

## Going deeper

```
beta_neutral:     w' = w - ((beta . w - target) / (beta . beta)) * beta          (beta from a trailing window against the equal-weight market)
liquidity_cap:    |w_i| <= participation * days * median_20d( price_i * volume_i ) / aum
liquidation:      days = X / (p ADV);  cost/X = spread/2 + eta sigma p^beta + gamma sigma X / (2 ADV);   risk = sigma sqrt( sum_d (x_d / X)^2 )  with x_d the dollars left after day d
tca_mvo cost:     a = spread / 2,   c = eta sigma (aum / ADV)^beta,   d = gamma sigma (aum / ADV) / 2,   trades solved as b - s with b, s >= 0 so the optimum has exact zeros
```

## Pitfalls

- A beta estimated on a short window is noisy, and a hedge built on it adds noise. The default is a year.
- Equal-weight "market" betas describe the assets in your universe, not the index you may benchmark against; name a hedge instrument for that.
- The impact parameters are illustrative. Calibrate `eta`, `beta` and `gamma` to your own executions before trusting a cost-aware optimiser on a real book.
- A liquidity cap sized on average volume says nothing about a crisis, when volume and your participation limit both change. `liquidity_seeking` ([basket and liquidity algorithms](basket-and-liquidity-algorithms.md)) is the execution-side answer.
- `tca_mvo` drops holdings in assets that leave the investable set without charging for the sale.

## Try it

```bash
quant backtest --model xs_momentum --allocator beta_neutral --alloc-param window=252
quant backtest --model xs_momentum --allocator tca_mvo --alloc-param aum=100000000 --alloc-param spread_bps=3
quant cashflow redeem --mix SPY=0.5,IEF=0.3,GLD=0.2 --aum 1e9 --redemption 0.1
```
