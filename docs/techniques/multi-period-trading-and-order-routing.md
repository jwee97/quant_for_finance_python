---
title: "Trading with costs: multi-period portfolios, the mix of limit and market orders, and smart order routing"
slug: multi-period-trading-and-order-routing
difficulty: 3
chapter: Platform
prerequisites: [execution-algorithms, costs-constraints-and-capacity, portfolio-theory-and-constrained-books]
stages: []
files: [src/equity/multiperiod.py, src/framework/allocators_trading.py, src/algo/limit_orders.py, src/algo/routing.py, src/equity/qp.py, experiments/trading_world.py]
figures: []
tests: [tests/test_equity_multiperiod.py, tests/test_allocators_trading.py, tests/test_limit_orders.py, tests/test_routing.py]
models: []
---

# Trading with costs: multi-period portfolios, the mix of limit and market orders, and smart order routing

## In one sentence

Trading is not free, so the portfolio you hold now should depend on where you are and on what you expect to want later (multi-period optimisation), a single order should be worked partly with patient limit orders and partly with market orders that take what is on offer (a dynamic programme), and an order sent to several venues should go where it is most likely to fill, which has to be learned from fills that reveal only a lower bound on the liquidity there (smart order routing).

## The idea

**The portfolio over several periods.** Take the objective of a mean-variance investor who pays for trading: in each period, the expected return of the holdings, less a charge for their variance, less the cost of the trade that got there. With a cost that is quadratic in the trade (market impact), the best policy is known exactly (Garleanu and Pedersen 2013; Mei, DeMiguel and Nogales 2016 for many risky assets and general costs) and has two parts. The *aim portfolio* is not the portfolio you would hold if trading were free: it is a weighted average of that portfolio now and of the portfolios you expect to want in the following periods, because a position bought today keeps earning while the forecast fades. And you do not trade all the way to it: you close a fraction of the gap each period, the *trade rate*, which falls as trading gets dearer and rises with risk aversion. `lq_solve` computes it by a backward recursion (a Riccati equation), exactly, for any path of expected returns.

**Constraints and proportional costs.** With limits on positions, leverage or exposures, or with a proportional cost (a spread or a commission), no closed form exists, but each period's problem over the next few periods is still a quadratic programme. Skaf and Boyd (2009) show how to use that: re-solve every period, trade only the first step (*model-predictive control*), and let a quadratic value function from the unconstrained problem stand for everything beyond the horizon (*approximate dynamic programming*). A proportional cost makes the best policy a *no-trade region*: while your holdings are within a band around the aim (the band is where the marginal gain of trading is smaller than the cost), do nothing; outside it, trade to the band's edge, not to the aim. Because constraints can only lower what is achievable, the unconstrained optimum is an upper bound on any constrained policy.

**One order: limit orders and market orders.** A market order is certain and costs the spread plus impact. A limit order at the bid earns the half-spread if it fills, costs nothing if it does not, may fill only in part, and often fills just before the price moves against you. While a share waits, the price may run away. The problem is a stopping problem: in each interval, with some of the order left, choose how much to buy now by market order and how much to post. `limit_orders.solve` solves it backward for every (interval, shares left) state, and `simulate` plays the policy against random fills.

**Many venues: smart order routing.** A dark pool fills you only as much as the other side happens to hold there, and you never see that quantity: send 500 shares and get 300 and you know the pool held exactly 300; get 500 and you know only that it held at least 500 (the observation is *censored*). Each venue is described by `P(V >= j)`, the chance it can fill `j` shares. The first `s` shares sent to a venue are worth `sum over j <= s of P(V >= j)`, which has diminishing increments, so the best split of an order gives the next share to whichever venue is most likely to fill it *at the margin*. The probabilities are learned from the censored fills with the Kaplan-Meier estimator, made optimistic about what the data say little about so that the router keeps trying venues and sizes until they have been shown to be poor (Ganchev, Kearns, Nevmyvaka and Vaughan 2010; Almgren and Harts 2008 for a related dynamic router).

## Why it matters

The cost of a trade is paid once and the benefit of a position is earned over its life, so a rule that looks only at this month's forecast (a one-period optimiser, even one that charges the cost) trades too little when the forecast is persistent and too much when it is not. And the cost of an order depends as much on how it is worked and where it is sent as on what the portfolio decided: the platform's portfolio books and its execution algorithms are two halves of one cost, and these are the pieces that join them.

## How this repo uses it

`src/equity/multiperiod.py` has the closed form (`lq_solve`, `stationary_trade_rate`, `cost_matrix`, `utility`), the constrained plan (`mpc_plan`, `mpc_step` with the approximate-dynamic-programming terminal reward, solved by the dense interior-point method of `src/equity/qp.py` for plans of a few hundred variables) and the proportional-cost no-trade region (`one_period_trade`, `no_trade_region`). The allocator `multi_period` puts it on the platform: each month-end it takes the combined forecast of the models in the spec, lets it fade by `persistence` a month (`auto`: the correlation of the forecast with last month's), plans the next `plan_horizon` of `horizon` months with `cost_bps` on every unit traded and an optional quadratic `impact`, under the named `book`'s limits, and trades the first step. `src/algo/limit_orders.py` and `src/algo/routing.py` are the order-level and venue-level models; `quant algo limit` and `quant algo route` run them from the command line, and `experiments/trading_world.py` produces the numbers below. All three are simulations on stylised markets: there is no order book, queue or latency model, and the venues' liquidity is drawn from the distributions the model assumes.

## What we found

All simulated (`python -m experiments.trading_world`, about three minutes), so these are checks of what each method does and how the methods compare, not evidence about any market.

*Planning ahead against trading one month at a time.* Ten worlds of 240 months and six assets whose expected returns follow a persistent process (monthly autocorrelation 0.9), quadratic costs, the investor seeing the expected returns. Utility is the return of the holdings less the risk charge and the cost, in basis points a month:

| policy | utility | minus the one-period policy | s.e. | turnover (x NAV a month) |
|---|---|---|---|---|
| ignore costs, hold the Markowitz portfolio | 53 | -98 | 7 | 1.40 |
| one period at a time, with costs | 151 | 0 | | 0.39 |
| multi-period, planning three months with the rest summarised | 166 | +15 | 3 | 0.55 |
| multi-period, the full closed form | 166 | +15 | 3 | 0.55 |

Ignoring costs gives most of the return away. Accounting for the cost one month at a time recovers it. Looking ahead adds another 10% of utility, and it does so by trading *more* than the myopic cost-aware rule, not less: that rule under-trades, because it credits a position with one month's return when the forecast lasts several. Planning three months and summarising the rest gives exactly the closed-form answer, as it should when nothing binds.

*With limits.* The same worlds with each position within 30%, gross exposure at most 100% and net within 30%, solved by model-predictive control (six worlds of 120 months): one-month plans 80.6 basis points a month; three-month plans 81.5 and six-month plans 81.5, a gain of 0.9 with a standard error of 3.4; the unconstrained optimum, which the limits keep out of reach, 144.4. With limits this tight the book spends its time against them and there is little to plan; what the limits cost (64) dwarfs what planning could add.

*The trade rate and the no-trade region.* For one asset the long-run trade rate is the share of the gap to the aim closed each month. At risk aversion 5 it is 0.98 when the impact cost is 0.0005, 0.92 at 0.002, 0.73 at 0.01, 0.46 at 0.05 and 0.24 at 0.25; at a given cost a more risk-averse investor closes the gap faster. With a proportional cost the no-trade band around the aim is as wide as the cost permits: against a forecast of 1% a month, a cost of 5, 10, 25 and 50 basis points leaves a band of width 10%, 20%, 50% and 100% of the aim.

*One order, limit against market.* Buying 100,000 shares in ten intervals with a half-spread of 3 basis points, an impact of 8 basis points if all of it is taken at the market, and a first unit that fills with probability 0.5. Expected cost in basis points of the whole order, and the share bought by limit order:

| price drift per interval | optimal mix | all at the market now | limit orders, market at the end | bought by limit order |
|---|---|---|---|---|
| 0 bp | -1.5 | 11.0 | -1.5 | 97% |
| 2 bp | 2.8 | 11.0 | 3.3 | 75% |
| 5 bp | 6.0 | 11.0 | 10.6 | 38% |
| 12 bp | 8.7 | 11.0 | 27.6 | 16% |

Patient when the price stays put (the order earns the spread), a market order when it is running away, and a mix in between that beats both pure rules; a fixed rule of "post limit orders, take the rest at the end" is better than the market when the price is quiet and far worse when it is not. A limit order is also unattractive when its fills are toxic: with an adverse move of 12 basis points after each fill, the optimal mix uses none.

*Routing across five venues.* A lit exchange that nearly always fills a little, three pools of increasing depth and decreasing probability, and one that almost never fills; 1,000 lots a round for 500 rounds. Expected fill rate in the last third of the rounds: even split 0.290; send everything to the venue most likely to fill anything 0.063; the learning router 0.343 (standard error 0.005); an oracle that knows the true distributions 0.364. The learning router reaches 94% of the oracle after 500 rounds. Without the optimism, it gets stuck: a venue that disappoints in its first few visits is never tried again.

*In the pipeline.* The same alpha (a regression on momentum and reversal, planted) through a 130/30 book that trades one month at a time and through `multi_period`, both told the cost the engine charges (10 basis points), in four worlds of 30 stocks for 3.8 years: net Sharpe 0.895 against 0.850 (difference -0.045 with a standard error of 0.030), annual turnover 11.5 against 12.7. Planning did not help here. The proportional cost is small next to the forecast, so the no-trade band is narrow, the forecast is noisy rather than a clean fading process, and with only proportional costs the terminal reward carries no information; the planning gain above needs a cost that matters and a forecast whose fading you know.

## Going deeper

```
objective      J = sum_t beta^t [ alpha_t'x_t - (gamma/2) x_t'Sigma x_t - (1/2)(x_t - x_(t-1))'Lambda(x_t - x_(t-1)) - kappa'|x_t - x_(t-1)| ]
recursion      V_t(x) = -1/2 x'A_t x + b_t'x + c_t        M_t = gamma Sigma + Lambda + beta A_(t+1)        g_t = alpha_t + beta b_(t+1)
               A_t = Lambda - Lambda M_t^-1 Lambda         b_t = Lambda M_t^-1 g_t                           x_t = M_t^-1 (g_t + Lambda x_(t-1))
partial        x_t - x_(t-1) = Q_t (a_t - x_(t-1))         aim a_t = (M_t - Lambda)^-1 g_t = (gamma Sigma + beta A_(t+1))^-1 (alpha_t + beta b_(t+1))        trade rate Q_t = I - M_t^-1 Lambda
no-trade       one period, cost kappa:   no trade if  | alpha - gamma Sigma x | <= kappa  in every asset;  otherwise trade until the gradient sits on the boundary
limit orders   V_t(q) = min over (m, s) of  m (h + k m/Q) + E_f [ f (a - h) + V_(t+1)(q - m - f) + drift (q - m - f) + risk (q - m - f)^2 ],     P(f >= j) = z d^(j-1) for j <= s
routing        value of s lots at venue i = sum_(j<=s) P_i(V >= j);   give the next lot to argmax_i value_i P_i(V >= s_i + 1)
Kaplan-Meier   h_k = (# exact V = k) / (# exact V >= k + # censored with sent > k);    P(V >= j+1) = P(V >= j)(1 - h_j);   optimism adds z standard errors of a proportion on the n_j observations at risk
```

## Pitfalls

- The closed form is exact for quadratic costs and a known fading of the forecast. The gain from planning depends on how well the fading is known: if the forecast is a noisy ranking whose persistence is guessed, planning ahead buys positions for a future that does not arrive, as the pipeline run above shows. Estimate the persistence from the forecasts (the `auto` setting) and compare with the one-month book on your own data before trusting it.
- Proportional costs make the plan a quadratic programme with an auxiliary variable per name and period: a plan of three months over 30 names is a few hundred variables and takes about a tenth of a second, so a nine-year monthly backtest of the book took about nine seconds; a universe of hundreds of names would need the first-order solver and much longer.
- Limits that bind most of the time leave little for planning to do, and the tight limit costs far more than the planning gains: read the upper bound as what the limits cost.
- The limit-order model has one price level, fills that are independent across intervals and a fixed adverse-selection cost. Real queues, cancellations and fees change the numbers; the directions (more patience when the price is quiet, less when fills are toxic) are what carries over.
- A router that learns from censored fills needs variety in what it sends: an even split never learns the deep pools' depth, and a router that is not optimistic abandons a venue after a few bad draws. Venue liquidity changes through the day; this one treats it as stationary.
- None of this is evidence about costs in a real market. Calibrate the impact, the fill probabilities and the venue distributions to your own executions.

## Try it

```bash
python -m experiments.trading_world
quant backtest --model characteristic_regression --allocator multi_period --alloc-param book=130_30 --alloc-param cost_bps=20 --alloc-param persistence=0.7 --alloc-param plan_horizon=3
quant algo limit --drift-bps 2 --shares 100000
quant algo limit --drift-bps 12 --adverse-bps 1
quant algo route --rounds 500 --lots 400
```
