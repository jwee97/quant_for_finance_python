---
title: "Execution algorithms: VWAP, TWAP, POV, arrival price, implementation shortfall, goals and tactics"
slug: execution-algorithms
difficulty: 3
chapter: Ch. 22
prerequisites: [execution-and-impact, mean-variance-and-shrinkage]
stages: []
files: [src/algo/market.py, src/algo/impact.py, src/algo/optimize.py, src/algo/algos.py, src/algo/simulate.py, src/algo/tactics.py, src/algo/factory.py, src/algo/catalog.py]
figures: []
tests: [tests/test_algo.py, tests/test_algo_catalog.py]
models: []
---

# Execution algorithms: VWAP, TWAP, POV, arrival price, implementation shortfall, goals and tactics

## In one sentence

A parent order is cut into slices over the day, and the shape of the cut is a choice between paying market impact by trading fast and paying price risk by trading slowly; the named algorithms are points on that trade-off, a simulator shows what each does, and an optimiser finds the best one for a stated goal.

## The idea

**The cost model.** Selling or buying `q` shares in an interval in which `V` shares trade costs a temporary impact `eta sigma (q / V)^beta` of the price, paid on that slice only (Almgren, Thum, Hauptmann and Li 2005 measured `beta` near 0.6), a permanent impact `gamma sigma q / ADV` that moves the price for everything after, and half the spread on every marketable share. Shares left unexecuted are exposed to the price: their variance is `sigma^2` times the sum, over intervals, of the shares still held squared. A schedule's objective is `cost + lambda * variance` (Almgren and Chriss 2000).

**The named algorithms.** `twap` slices equally; `vwap` follows the day's expected volume, which at zero risk aversion with linear impact is also the cheapest schedule; `pov` trades a fixed share of the volume as it prints; `arrival_price` front-loads to stay near the price when the order arrived (an `urgency` setting); `is` (implementation shortfall) is the Almgren-Chriss optimum for a risk aversion, optionally with a view on the price drift. Kissell's I-Star model (`istar_estimate`) gives the pre-trade cost estimate a desk would quote.

**Best-execution goals.** Five ways of saying what "best" means, each a point on the cost-risk frontier: `min_cost`; `min_cost_risk` (the cheapest schedule whose timing risk is under a limit); `min_risk_cost` (the least risky schedule whose cost is under a limit); `balanced` (a risk aversion); `price_improvement` (the schedule most likely to beat a target cost, `Phi((target - E) / std)`).

**Schedule optimisation.** The problem is a convex quadratic program when impact is linear (or linearised at the order's average participation), solved exactly by an active-set method (`solve_qp_eq`) and polished against the true power law with SLSQP; at zero risk aversion it returns VWAP, and as risk aversion grows the schedule moves to the front, the Almgren-Chriss family. One-parameter families describe a strategy in a number: `exp_trade` (the trade rate decays like `exp(-kappa t)`), `exp_residual` (the shares remaining do) and `trade_rate` (a fixed participation).

**Trading styles.** The same schedule can be worked aggressively (every share crosses the spread), as a working order (a mix of limit, dark and market orders, topped up with market orders when behind) or passively (mostly limit orders and dark pools: earns the spread and a rebate, leaks little, may not fill, and fills least when the price runs away).

**Adaptation tactics.** A schedule is a plan made before the day; a tactic changes the pace as the day unfolds. Aggressive in the money (`aim+vwap`) speeds up when the price is in your favour, for markets that overshoot and come back; passive in the money (`pim+vwap`) slows down when the price is in your favour and speeds up to limit a loss, for markets that trend; target cost (`target_cost+is`) re-chooses, at every interval, the least risky schedule that fits the temporary-impact budget that is left.

## Why it matters

Execution cost is the difference between a strategy's backtest and its P&L, and it is not a constant: it depends on size against volume, on how fast the order is worked and on what the price does meanwhile. An algorithm cannot beat the market; it can choose which risk to take, and the simulator makes that visible.

## How this repo uses it

`src/algo/` is the package; `quant algo list` maps every item of the execution taxonomy to its code (and `docs/algorithmic_trading.md` shows the same table), `quant algo run --algos vwap is:risk_aversion=0.003 aim+vwap` compares algorithms on a simulated day, `quant algo frontier` traces the cost-risk frontier, and the dashboard's Execution tab does both with a form. The simulator draws many days (U-shaped volume with noise, a price walk with optional drift and autocorrelation, a stress window) and reports the implementation shortfall against the arrival price and its parts. With noiseless volumes and a plain schedule the simulated cost and risk equal the closed forms; the tests check that, the sinh solution of Almgren-Chriss as the limit of the discrete program, the exact solver against a general one, and each tactic in the market it is built for (and that it loses in the other). Text specifications such as `target_cost+is:risk_aversion=0.003` build any algorithm.

## What we found

A $10m buy (200,000 shares, 10% of the day's volume) in a stock with 2% daily volatility and a 4 bps spread, 1,500 simulated days, aggressive orders: shortfall against the arrival price was 15.7 bps for VWAP, 16.5 for TWAP, 16.1 for POV, with a standard deviation across days of about 110 bps; implementation shortfall with a risk aversion of 0.003 cost 20.7 bps and cut the standard deviation to 45 bps, and arrival price cost 24.0 bps for 37 bps of deviation: paying 5 to 8 bps more removes more than half of the dispersion. The closed-form frontier says the same: VWAP 12.1 bps of cost at 110 bps of risk; a risk aversion of 0.004 costs 20.1 bps at 36 bps; 0.03 costs 31.7 at 11. Worked passively, VWAP's spread cost fell from 2.0 bps to 0.9 but its timing cost rose from 3.5 to 7.2 and the dispersion from 111 to 132 bps: passive orders trade a certain cost for an uncertain one. The tactics follow the memory of the market: with mean-reverting prices AIM saved 0.7 bps against plain VWAP and PIM cost 1.2; in a rising market for a buyer AIM cost 2.9 bps and PIM saved 2.0. Neither knows which market it is in.

## Going deeper

```
cost(q)       = sum_i  ref * q_i * eta sigma (q_i / V_i)^beta   +   (1/2) ref gamma sigma X^2 / ADV   +   (1/2) ref * spread * X
variance(q)   = ref^2 sigma^2 sum_i w_i x_i^2          x_i = shares left after slice i,  w_i = interval share of a day's variance
J(q)          = cost + lambda * variance                sum q = X,  q >= 0         (lambda = 0: VWAP;  larger: front-loaded)
AIM / PIM:    pace_i = planned_i * exp( +/- strength * z_i ),   z_i = price move since arrival in standard deviations (positive when in your favour)
```

## Pitfalls

- The impact parameters (`eta = 0.142`, `beta = 0.6`, `gamma = 0.3`) are illustrative values from the equity literature, not your executions. Calibrate before trusting a number to the basis point.
- The simulator has no order book, queue, latency or other participants reacting to you, and its passive and dark fill rates are round numbers. Use it to compare methods and to see which way a parameter pushes.
- A tactic is a bet on the market's memory. Averaged over markets of both kinds it has no edge; measure it in the one you trade.
- Lower average cost with higher dispersion is not better: say which you are minimising.

## Try it

```bash
quant algo list
quant algo run --algos twap vwap pov is:risk_aversion=0.003 arrival_price aim+vwap pim+vwap target_cost+is:risk_aversion=0.003 --paths 1500
quant algo run --algos vwap --style passive --scenario mean_reverting --target-bps 15
quant algo frontier --shares 400000
```
