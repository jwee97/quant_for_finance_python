---
title: "Cash-flow strategies: deposits, redemptions, dividends, liabilities and payments"
slug: cash-flow-strategies
difficulty: 2
chapter: Ch. 22
prerequisites: [backtest-engine-and-costs, drawdown-ruin-and-kelly]
stages: []
files: [src/cashflow/policies.py, src/cashflow/simulate.py, src/cashflow/redemption.py, src/cashflow/liabilities.py, src/cashflow/spending.py, src/cashflow/flows.py]
figures: []
tests: [tests/test_cashflow.py]
models: []
---

# Cash-flow strategies: deposits, redemptions, dividends, liabilities and payments

## In one sentence

Money moves in and out of a portfolio for reasons that have nothing to do with the strategy (contributions, redemptions, dividends, promised payments), and which assets are traded to handle each flow decides how far the portfolio drifts, what it costs and whether it lasts.

## The idea

A backtest assumes one pot of money that is never added to or drawn on. Real money is. Five kinds of flow, five questions.

**Cash deposits.** A contribution can be spread over the target weights (`pro_rata`), used to buy only what is below target (`correct_drift`, which fixes drift for the price of the contribution alone, with no selling), put through a full rebalance (`rebalance`, which corrects everything and trades the most), or left in cash until the next scheduled rebalance (`cash`). Dollar-cost averaging (`dca_months`) invests a lump sum in equal parts over months and is a bet on the path: in a rising market it loses to investing at once, in a falling one it wins.

**Redemptions.** Cash for a redemption comes from selling pro rata (the mix is kept, illiquid positions are sold as hard as liquid ones), from selling what has drifted above target (`correct_drift`), from the most liquid assets first (`liquid`: cheapest to sell, but the book that is left is less liquid and further from target) or from a cash buffer. `redemption_cost` prices each with the liquidation model of the execution algorithms and reports cost, days to finish and the drift left behind.

**Cash dividends.** A dividend can be reinvested in the asset that paid it (which keeps adding to the winner), put through the deposit policy (so the dividend from the overweight asset buys the underweight one, a free rebalancing flow), or held as cash (a drag). On the same day a dividend and a withdrawal net: the dividend funds the payment.

**Liabilities.** A plan that owes dated payments has a liability, the present value of those payments, which rises when interest rates fall. The funding ratio is assets over liabilities. Liability-driven investing (LDI) holds a bucket of long bonds whose dollar duration offsets a share of the liabilities' (the hedge ratio) and the rest in return-seeking assets, and a glide path raises the hedge ratio as the funding ratio improves, locking in the gain.

**Payments.** An investor who spends from a portfolio needs a rule. `fixed_real` takes the first year's rate of the starting value and raises it with inflation whatever happens (the "4% rule" of Bengen 1994); `percent_of_nav` takes a share of the current value (never ruined, but spending follows the market); `endowment` smooths between the two (the Yale and Tobin rule: a share of last year's spending raised with inflation plus a share of the rate times the current value); `guardrails` raises or cuts spending by 10% when the withdrawal rate drifts outside a band (Guyton and Klinger 2006). `simulate_spending` runs a rule over many futures made by resampling the portfolio's own daily history in blocks, so volatility clusters survive, and reports ruin, the spread of final wealth and the chance that spending had to be cut.

## Why it matters

A time-weighted return, which is what a backtest reports, says what the portfolio did; the money-weighted return says what the investor got, and the two diverge by exactly the timing of the flows. The way flows are traded decides drift, turnover and cost, and for a fund the way redemptions are met is often the largest cost it ever pays. For a retiree the spending rule matters at least as much as the asset mix.

## How this repo uses it

`src/cashflow/` is a small package: `policies.allocate_flow` turns one flow into dollar trades; `simulate.simulate_cashflows` follows a portfolio day by day (target weights as a mix or a strategy's own book, flows, dividends, scheduled rebalances, costs) and reports `irr`, `twr_annual`, turnover, cost and drift; `redemption`, `liabilities` and `spending` do the rest. The command line is `quant cashflow simulate | spending | redeem | ldi`, and the dashboard's Cash flows tab runs the same four from a form on the tickers you chose, with the charts. The tests prove the accounting (a buy-and-hold with no flows equals the growth of the mix; deposits grow like the asset they bought; the money-weighted and time-weighted returns agree for constant growth and separate when money arrives at a top or a bottom), the policy properties (a drift-correcting deposit never sells and never increases drift; a liquid-first redemption costs less and drifts more), dollar-cost averaging in rising and falling markets, closed forms for each spending rule, the guardrails after a planted crash, the sustainable rate against the annuity formula, and that the LDI-hedged plan's funding ratio barely moves with interest rates while the unhedged one does.

## What we found

On the 15 ETFs, a 60/40 SPY/IEF mix over 20 years with $1,000 added each month and no scheduled rebalance (a single run, 5 bps of cost): pro-rata deposits ended with the most money ($1.46m, money-weighted return 9.8%) because equities outran bonds and the portfolio drifted toward equities (mean distance from target 10 points); drift-correcting deposits ended with $1.35m (9.3%) and 6 points of drift while trading the same dollars (2.5% of the portfolio a year, as for pro rata); a full rebalance with every deposit ended with $1.23m (8.7%), 0.7 points of drift and nine times the turnover (22% a year), and holding the deposits in cash $0.86m (6.3%). Drift was not a cost in this sample, it was a reward; the policy buys control, not return. The same holds for withdrawals: taking $3,300 a month from $1m with 2% dividends, selling the overweight asset left the book 4.6 points from target instead of 7.2 and ended with $2.99m instead of $3.06m. Spending 4% of $100,000 a year (inflation 2%, 30 years, 600 resampled futures from the 60/40's own history): a fixed real withdrawal ran out in 1.0% of futures, cut spending below 80% of its first-year level in 0.8%, and the highest starting rate whose chance of ruin stayed under 5% was 4.8%. Spending a share of the value never ran out but cut spending in 26% of futures; the endowment rule 14%, guardrails 10%. Meeting a 10% redemption of a $1bn fund (50% SPY, 30% IEF, 20% GLD) pro rata cost 2.7 bps of the amount raised and took 0.6 days; selling only the most liquid asset cost 2.4 bps, finished in under a day and left the book 6 points off target. For an LDI plan funded at 85%, owing 100 a year for 25 years, hedging with TLT and seeking with SPY and EFA, the funding ratio's annualised volatility fell from 21.7% unhedged to 18.0% with a glide path and its worst fall from 66% to 17%.

## Going deeper

```
correct_drift:   desired = target * (V + F);   gap = desired - values;   deposit F > 0: trades = gap+ * F / sum(gap+)       withdrawal: trades = -gap- * |F| / sum(gap-)
money-weighted:  initial + sum_k flow_k / (1 + r)^t_k = final / (1 + r)^T            time-weighted: product of (NAV_t - flow_t) / NAV_(t-1)
liability:       PV(y) = sum_k c_k (1 + y)^-t_k;   duration = sum t_k PV_k / (PV (1 + y));   hedge weight = ratio * D_L / (funding_ratio * D_hedge)
fixed real:      spend_y = rate * N_0 * (1 + inflation)^y;   N_(y+1) = (N_y - spend_y)(1 + R_y)        endowment: spend_y = s * spend_(y-1)(1 + inflation) + (1 - s) * rate * N_y
```

## Pitfalls

- A history-based bootstrap cannot produce a future worse than the past it samples. Twenty-one years of a 60/40 contains one financial crisis and one inflation shock; widen the sample or the stress before reading "1% ruin" as a promise.
- Prices are treated as total-return series; a `dividend_yield` splits each day's return into a price part and a dividend paid at quarter-end. It changes what is done with the cash, not the total return.
- Withdrawals in a falling market are the most damaging flows (sequence-of-returns risk); the order of returns matters, not only their average.
- LDI here uses a flat discount yield (the 10-year Treasury) and equal-weight buckets; a real plan discounts every payment on its own curve and hedges key-rate durations, not one number.

## Try it

```bash
quant cashflow simulate --mix SPY=0.6,IEF=0.4 --deposit 1000 --policies pro_rata correct_drift rebalance cash
quant cashflow simulate --mix SPY=0.6,IEF=0.4 --initial 1000000 --withdraw 3300 --dividend-yield 0.02
quant cashflow spending --mix SPY=0.6,IEF=0.4 --rate 0.04 --years 30
quant cashflow redeem --mix SPY=0.5,IEF=0.3,GLD=0.2 --aum 1e9 --redemption 0.1
quant cashflow ldi --hedge TLT --seeking SPY EFA --funding-ratio 0.85
```
