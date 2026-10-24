# Option models: what a chain implies, what an exotic is worth, what a hedge costs

Black-Scholes gives every strike the same volatility. A real option chain does not, and the way it does not is information: how likely the market thinks a crash is, how fast the skew fades with the maturity, how much a barrier or an American option should cost when no one quotes it. The **Model fit**, **Model prices** and **Hedging cost** views of the Options tab (and `quant models` in the terminal) fit five pricing models to a chain, price products under all of them and cost the hedging of an option when every trade costs money.

The short version of what the record says: stochastic volatility (Heston) reproduced the platform's simulated SPY smile far better than the jump models, which still beat a flat volatility by a wide margin; on a vanilla the models differ by how well each fits it (10% across the five at the money, mostly Kou and Merton fitting worse), and on an exotic they differ by that and by what the quotes do not determine, which showed clearly on a knock-in call above the price (26% across the models against 10% on the vanilla) and not on a barrier near the money; the *model risk* of a product is its spread set beside the vanilla's; early exercise adds a few per cent to a put and the jump models differ on it; and hedging a sold option on a daily clock costs roughly what the spread costs, which a no-trade band cuts by half at the price of a wider spread of outcomes. Every number below is on a **simulated** chain (the platform holds no option quotes), so they show how the machinery behaves and how much the answer depends on the model, and nothing about what a market believes.

## Five minutes, with SPY

1. Open **Options** and choose **Model fit**. The chain for SPY is a model (an arbitrage-free surface at SPY's real price and realised volatility), and the page says so in a banner; for a real chain, download one first with the *Real option chains (free)* card.
2. Under *2. Models to fit* all five models are ticked. Under *3. Which quotes* the defaults use up to six expiries from a week out, within 30% of the forward.
3. Press **Fit the models**. It is a background job (about half a minute for all five on a laptop); you can leave and come back.
4. Read the result (below). Then open **Model prices**, choose a product (try the American put at 105% of the price, then the barrier) and press **Price the product**: the fits are kept for the session, so this takes seconds.
5. Open **Hedging cost**, leave the cost at 5 basis points and press **Cost the hedge**.

## Reading Model fit

**The best model and the benchmark.** The first tiles say which model reproduced the quotes best, its error in *volatility points* (one point is a volatility of 20% against 21%) and what share of the smile's variance it explained against the benchmark every model must beat: **one flat Black-Scholes volatility per expiry**. A model that does not beat it has learned nothing about the smile.

**How far each model is from the quotes** is that error for each model as a bar, with the flat benchmark in grey. **The fitted parameters** table gives each model's parameters with their standard errors, the largest error, the share explained, and the *conditioning* of the fit. A parameter on the edge of its allowed range is marked *not a measurement*; one known to within many times its own size is marked *not pinned down by the quotes*. Both are warnings printed above the tiles as well.

**The smile** draws the market's out-of-the-money quotes as dots and each model's implied volatility as a line, for the expiry you choose with the buttons (the first is the one nearest 30 days). A line that follows the dots reproduces the smile; one that goes straight does not.

**What the quotes imply for the price** is the risk-neutral density of the return to the expiry nearest 30 days under each model, against the lognormal of Black-Scholes at the at-the-money volatility; **the chance of a fall at least this big** is the left tail of the same densities on a log scale, and the last table reads off the odds of a 10% and a 20% fall and of a 10% rise, the skewness and the excess kurtosis. These are *risk-neutral* odds: they include what the market charges for crash protection and are not forecasts.

## Reading Model prices

Choose a product, a strike (as a per cent of the price) and an expiry on the chain. Each model you tick prices it; Black-Scholes at the at-the-money volatility is always there for comparison, and for a vanilla so is the market's own smile.

- **The tiles** give the price under the best-fitting model that can price the product, Black-Scholes, and the **range across the models** as a share of the Black-Scholes price, with the range of the same models on the *vanilla at the same strike* beside it. A vanilla's range is only how well each model fits the quotes (the fit error is in the table); for another product the part of its range beyond the vanilla's is its *model risk*, and a note above the tiles says which it is.
- **How far each price is from Black-Scholes** shows each model's price as a per cent above or below it, with the price and the standard error of a simulated one beside the bar.
- **By strike** (vanilla and digital): the difference from Black-Scholes across strikes from 80% to 120%, which is the smile's effect. **The early-exercise boundary** (American): the price at which exercising stops being optimal, by days left, for the best jump model and Black-Scholes.
- **Every price** lists the price, the standard error, the gap to Black-Scholes, the fit error of the model, and how each was computed. The Heston model has no American price here (its variance is a second state variable) and the table says so.

## Reading Hedging cost

Choose an option, the cost of one trade in basis points of the value traded, the rebalancing dates and a risk aversion (in units of the option's price). The tiles give the Black-Scholes price, the seller's price with costs (Leland), the cost in volatility points, the utility-based writer's and buyer's prices and the half-width of the no-trade band. **What rebalancing more often costs** plots Leland's price against the number of dates; **the band** is the holding to keep against the price of the underlying; **profit and loss** is the distribution of the result of selling the option and hedging it by each of three rules on simulated paths, after every cost; the table beneath gives the mean, spread, tails, cost paid and number of trades of each.

## The models

| Model | What it adds | What it cannot do |
|---|---|---|
| Merton | Rare jumps of normal size on top of a diffusion. | One typical jump size: it cannot give a steep short-dated skew and a flat long-dated one. |
| Kou | Double-exponential jump sizes, a different tail up and down. | Its tail parameters are poorly pinned down by a chain: it often ends on the edge of its range. |
| Variance gamma | Infinite activity, a free skew and kurtosis, no diffusion. | The skew fades with maturity as for a random walk, faster than most markets'. |
| NIG | The same with somewhat heavier tails. | The same flattening. |
| Heston | Stochastic variance, correlated with the price; the smile does not flatten as fast. | No jumps; no American price here. |

## The products

| Product | How it is priced | What to watch |
|---|---|---|
| European | Fourier integral of the characteristic function (exact to about 1e-9). | The models were fitted to these: they should agree with the market. |
| Digital | The strike-derivative of the call, `-dC/dK`. | It is the slope of the smile in disguise; Black-Scholes cannot price it. |
| Barrier | Monte Carlo on the trading days, with the models on the same random numbers. | A barrier watched daily is weaker than one watched continuously; the page shows both for Black-Scholes. |
| Average-price | Monte Carlo with a control variate. | Averaging damps the tails, so the models disagree less. |
| American | The integro-differential equation on a grid with the early-exercise constraint (jump models); finite differences for Black-Scholes. | Heston is not offered. |

## In the terminal

```bash
quant models list                                          # the models, their parameters, the products
quant models fit SPY                                       # fit all five to SPY's chain (a model chain, and it says so)
quant models fit --chain my_chain.csv --models merton heston --json fit.json --notes
quant models price SPY --product barrier --barrier-kind down-out --barrier 90 --dte 90
quant models price SPY --product american --right put --strike 105
quant models hedge SPY --cost-bp 5 --dte 90 --steps 61
```
`--chain` takes a vendor CSV (date, expiry, strike, right, bid, ask, underlying, and rate and dividend if you have them); without it the chain is the model chain of one of the platform's ETFs. The command prints the same tables as the page and `--json` writes everything it computed.

## What was found

- **The numerical machinery is exact where it can be checked** (guide: [option pricing models and calibration](techniques/option-pricing-models-and-calibration.md)): Fourier prices match the Black-Scholes and Merton closed forms, the Heston quadrature and exact simulation; the finite-difference and integro-differential solvers converge at second order and agree with the Fourier price and with a Longstaff-Schwartz simulation; every identity an exotic must obey holds under every model. Checking the Fourier price against the older Merton series found that series double-counting the jump compensation (about 4% too high at two jumps a year); it was fixed.
- **Stochastic volatility fits this smile best.** On SPY's simulated chain at the September 2026 prices Heston was 0.35 points from the quotes, Kou and Merton about 1.5, NIG 2.2 and variance gamma 2.6, against 3.8 for a flat volatility; Kou's tail parameters sat on the edge of their ranges. It is a model fitted to a model: with a real chain the order may differ, and on real chains it partly did (next bullet).
- **On four real chains the order was mostly the same, and the one-week smile of SPY and QQQ was the exception.** Downloaded from Cboe's delayed quotes (the 23 October 2026 close; the application's own `fetch_chain`, the code behind the Data manager's Options download) and fitted with `quant models fit --chain` to the out-of-the-money quotes of six expiries from 7 to 357 days, the error in volatility points (Heston, Kou, Merton, NIG, variance gamma; the flat volatility to beat) was: SPY 2.33, 2.33, 2.43, 2.73, 3.23 (flat 8.41); QQQ 2.11, 2.95, 3.37, 2.83, 3.05 (7.40); IWM 0.87, 1.60, 1.62, 2.25, 2.47 (4.86); TLT 0.45, 0.38, 0.40, 0.81, 1.00 (1.42). Every model beat the flat volatility, by a wide margin on the equity ETFs. Heston was best on the three equity ETFs (level with Kou on SPY) and the jump models on the bond ETF. The average hides where they differ: Heston matched the smiles from five weeks to a year out to within 0.3 to 1.0 points on SPY, QQQ and IWM, and missed the one-week smile of SPY by 5.4 points (the other four models were 2.9 to 3.3 off) and of QQQ by 5.0 (the others 3.9 to 5.1), where its volatility of volatility sat on the upper end of its range on SPY; on IWM the one-week smile was fitted to 1.0 point, the best of the five. So the simulated chain's verdict (Heston far ahead, 0.35 points) holds for the longer expiries of a real chain, and the one-week smile of SPY and QQQ is where a stochastic-volatility model alone is not enough. One day's delayed snapshot of four ETFs, with the rate assumed (4%) and the early-exercise premium inside the quotes: it is the first look at a real chain, not a study.
- **Model risk shows on some exotics and not on others, and the vanilla is the yardstick.** On the simulated SPY chain (88 days, five models) the at-the-money call ranged from 21.15 to 23.39 (10% of the Black-Scholes price: that is how well each model fits, 0.35 points for Heston, 1.5 to 2.6 for the rest), and a 90% down-and-out call, an average-price call, a digital put and an American put had ranges of 10%, 11%, 11% and 14%, which is the same fit again. A knock-in call at 110% of the price is different: 3.59 under Heston, 4.94 under NIG, 6.1 to 6.5 under the others and 11.14 under Black-Scholes, a range of 26%, well beyond the vanilla's 10%: what the quotes cannot tell you is how likely the price is to touch a level above it, and the models disagree.
- **Early exercise is worth a few per cent on a put,** between 0.32 and 0.39 on a one-year at-the-money put of 6.6 to 7.7 under four jump models, and the PIDE price agreed with an independent simulation to within 1.5 standard errors.
- **Hedging on a clock costs about what the spread costs.** Daily hedging of an 88-day at-the-money call at 5 basis points a trade lost 1.38 on average with 1.36 paid in costs; Leland's price was 4% above Black-Scholes; a Whalley-Wilmott band traded 39% as often and paid 51% of the cost, with twice the spread of outcomes.

## What the models cannot tell you

- **A calibration is not a forecast.** It describes the risk-neutral measure, with the market's price of risk inside it. The Forecast tab, not this one, forecasts (and its stochastic-volatility and Lévy methods did not beat GARCH on the ETFs: see [forecasting](forecasting.md)).
- **On a simulated chain the models describe a model.** Download a real chain for anything about a market, and treat one delayed snapshot as one snapshot (the four real chains above are one snapshot each, from one close).
- **A good fit to the vanillas does not choose between models on an exotic.** That is why the page fits five and shows the range, and why it sets the range beside the vanilla's.
- **The hedging results use a constant proportional cost, one volatility and one model.** Real spreads widen in a storm and volatility is never known.
- **Listed ETF options are American and the chain's quotes are priced as European.** The page prices the American option separately; the quotes in a chain still carry the early-exercise premium of deep in-the-money puts.

## Where the ideas come from

The topics (Lévy and Heston models by Fourier methods, finite-difference and integro-differential solvers for European, barrier and American options, jump models, transaction costs, Ornstein-Uhlenbeck trading thresholds, Kalman filters for reversion and volatility, closed-form mean-variance) follow the notebooks of Cantaro's *Financial Models Numerical Methods* repository, which is a good place to see them worked in a different style. That repository is licensed AGPL-3.0 and this one MIT, so **no code was copied or translated**: everything here is written from the published mathematics (Lewis 2001; Carr and Madan 1999; Cont and Voltchkova 2005; Leland 1985; Whalley and Wilmott 1997; Hodges and Neuberger 1989; Bertram 2010; Andersen 2008 and others, cited in the module docstrings) and tested against closed forms, independent methods and simulations with a known truth.

Guides: [option pricing models and calibration](techniques/option-pricing-models-and-calibration.md), [exotic and American options](techniques/exotic-and-american-options.md), [hedging with transaction costs](techniques/hedging-with-transaction-costs.md), [finite-difference solvers](techniques/finite-difference-solvers-and-linear-systems.md), [SDE simulation and estimation](techniques/sde-simulation-and-estimation.md), [short-rate models and Kalman trackers](techniques/short-rate-models-and-kalman-trackers.md), [stochastic volatility and Lévy shocks in forecasting](techniques/stochastic-volatility-and-levy-forecasting.md).
