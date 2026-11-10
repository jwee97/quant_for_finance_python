# Scenarios: thousands of possible futures, and how far each way of making them can be trusted

A risk number, a stress test or a Monte Carlo of a portfolio all need the same thing first: a set of possible futures for the assets, each a path of daily returns. The **Scenarios** tab makes them in eight ways, from replaying the past to training a GAN, a variational autoencoder or a diffusion model, and then does the part that matters: it scores every set against the history it was made from.

A scenario set that matches the past says nothing about what comes next. It is a model of what has happened, and the table shows how faithful the model is. A neural network is not better for being a neural network: on a few years of daily returns a small GAN can miss the spread and the correlations that a plain bootstrap gets right, and the table shows where.

## Five minutes

1. In the **Backtest** tab choose the tickers (up to twelve) and the data source and start date. The Scenarios tab uses exactly those, so a start date three or more years back gives the learned generators something to learn from.
2. Open **Scenarios**. *Recommended* ticks block bootstrap, AR(1)-GARCH and the factor VAE; *All of them* ticks the eight.
3. Choose **1 month (21 trading days)** and 500 scenarios, and press **Generate the scenarios**. Resampling and statistical generators take under two seconds each; a GAN takes about ten and a diffusion model about five on four assets.
4. Read the quality table, the chart and the risk table (below). **Stop and show what it has so far** ends a run after the generator it is on.

The same thing from a terminal, on the platform's ETFs or on a CSV of prices:

```
quant scenarios --list
quant scenarios SPY QQQ TLT GLD --generators bootstrap arima_garch factor_vae --horizon 21 --n 500
quant scenarios --prices prices.csv --generators all --json scenarios.json
```

## The eight generators

| Generator | Kind | What it does | Where it breaks |
|---|---|---|---|
| Historical windows | resampling | Random stretches of consecutive days from the actual history | Nothing worse than the worst stretch in the data can occur |
| Block bootstrap | resampling | Whole days drawn in linked blocks (stationary bootstrap, 20 days on average), so the assets keep moving together | Cannot make a day more extreme than one that has happened |
| Copula | statistical | Each asset's own distribution joined by a Student-t copula, drawn independently day by day | No volatility clustering: a calm month and a crisis month look alike |
| Risk factors | statistical | The first three principal components bootstrapped in blocks plus each asset's own leftover noise | Correlations among the leftovers are dropped |
| AR(1)-GARCH | statistical | Per asset an AR(1) mean and a GJR-GARCH volatility, driven by shocks resampled jointly across assets | A model of the past: a regime it has not seen is not in the scenarios |
| GAN (WGAN-GP) | learned | A Wasserstein GAN with gradient penalty (Arjovsky et al. 2017; Gulrajani et al. 2017) trained on the days of the history | Adversarial training is unstable: it can miss the tails and the spread |
| Factor VAE | learned | A variational autoencoder (Kingma and Welling 2014) with a factor structure and Student-t noise | No clustering; a regime with different factors is not in the scenarios |
| Diffusion | learned | A denoising diffusion model (Ho et al. 2020) trained on the days of the history | Slow to fit; a short history gives it little to learn |

The learned generators are small and are trained for seconds, not hours: the app uses a 64-unit, two-layer GAN for 1,200 steps, the library's factor VAE as it is, and a 96-unit, two-layer diffusion model for 250 epochs. They are fitted on the whole history in the table, so their scores say how well they reproduce the past they saw, not how they would do on a new one. The learned generators draw one day at a time, so they have no volatility clustering however good the rest of the table looks; the clustering column shows it.

## Reading the result

**The quality table** has one row per generator and one column per property of the history. A ✓ means within a rule-of-thumb distance of a perfect copy and a **!** means further than that (never colour alone); the target of a perfect copy is under the table and in each column's tooltip.

| Column | What it measures | Perfect |
|---|---|---|
| Return distributions | The largest gap between a day's generated and real return distribution, averaged over the assets (Kolmogorov-Smirnov) | 0 |
| Volatility | The worst asset's relative error in the size of a day's moves | 0 |
| Correlations | The largest gap in any pair's correlation (two or more assets) | 0 |
| Crashing together | The gap in how often the assets are all in their worst 5% of days together | 0 |
| Fat tails | Generated kurtosis over real kurtosis | 1 |
| Volatility clustering | The gap in the autocorrelation of absolute returns, the footprint of calm and stormy spells (horizon over one day) | 0 |
| Worst-5% loss | The portfolio's average loss on its worst 5% of days, generated over real | 1 |
| Horizon spread | The spread of the portfolio's return over the whole horizon, generated over real | 1 |

**The chart** shows the equal-weighted portfolio's return over the horizon at each percentile of the scenarios, beside the same percentiles of overlapping stretches of the history (dashed). A generator that follows the dashed line has the right spread and tails; one whose left end lies above it is too gentle on losses. The history's line comes from overlapping stretches, which are far fewer independent observations than their count: three years of days hold about three dozen separate months, so the history's own 1% and 99% percentiles are very rough.

**The risk table** gives each set's average return, its spread, the 95% value at risk (the loss that 95% of scenarios stay inside) and the 95% expected shortfall (the average loss on the worst 5%) for the portfolio, with the history's own beside them.

## What is not here

* The weights are equal and fixed, and the portfolio is rebalanced every day. The tab compares generators; it does not say how to invest.
* The scenarios are not fed to the backtest or the optimisers. The functions behind the tab (`src.scenarios.generate`, `quality_report`) are the ones to call from your own code.
* A generator is scored on the history it was fitted on, with no held-out stretch. A generator that memorises the history would score perfectly and be useless; the learned ones here are too small to memorise but the table cannot tell you that.
* A start date too recent, or tickers that did not all exist, leaves fewer than 500 complete days, and the tab says so rather than make scenarios from too little.
