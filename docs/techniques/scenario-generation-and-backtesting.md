---
title: "Scenario generation and backtesting: bootstraps, copulas, GARCH, learned generators and combinatorial cross-validation"
slug: scenario-generation-and-backtesting
difficulty: 3
chapter: Platform
prerequisites: [resampling-and-monte-carlo, tail-risk-evt-and-copulas, walk-forward-and-leakage]
stages: []
files: [src/scenarios/generators.py, src/scenarios/quality.py, src/scenarios/backtest.py, src/validation/cpcv.py, src/models/generative.py, experiments/deep_world.py]
figures: []
tests: [tests/test_scenarios.py, tests/test_generative_pinn.py]
models: []
---

# Scenario generation and backtesting: bootstraps, copulas, GARCH, learned generators and combinatorial cross-validation

## In one sentence

History happened once, so risk and strategy questions about the future are asked of *generated* histories: windows of the past, bootstrapped rows or blocks, a copula with each asset's own marginal, principal-component factors, an ARIMA–GARCH model with resampled shocks, or a trained generator (GAN, VAE, diffusion), judged by a common yardstick of what each preserves, and a strategy's results are looked at over many generated paths and, for a trained rule, over the many paths of combinatorial purged cross-validation.

## The idea

**What each generator preserves.** *Historical* windows keep everything that happened and nothing else. *Bootstrap* of whole rows keeps the cross-section; with blocks (moving, circular, stationary) it also keeps serial dependence such as volatility clustering. A *copula* keeps each asset's marginal distribution (empirical or extreme-value) and a fitted dependence (Gaussian, Student, ...) with its tail dependence, but draws days independently. *Risk-factor* scenarios bootstrap the first principal components in blocks (dependence and clustering) and each asset's residual on its own. *ARIMA–GARCH* gives each asset an AR(1) mean and a GJR-GARCH volatility, driven by jointly resampled standardised residuals (clustering, leverage, cross-asset dependence, volatility that can exceed anything observed). *Learned* generators (WGAN-GP, factor VAE, diffusion) fit the joint distribution of windows of `window` days and sample from it.

**The yardstick.** `quality_report` compares scenarios with the history on: the Kolmogorov–Smirnov distance of the marginals, the error of the volatilities, of the correlations and of the lower-tail dependence, the ratio of kurtosis (1 is right, below 1 means tails too thin), the error of the first autocorrelation of absolute returns within a path (volatility clustering), the ratio of the portfolio's 5% expected shortfall, and the ratio of the standard deviation of the compounded horizon return. Each generator passes some and fails others.

**Scenario backtests.** `scenario_backtest(strategy, returns, generator, ...)` runs a rule over many generated paths and returns the distribution of Sharpe ratio, drawdown and terminal wealth: the question is how a rule behaves over the range of futures a model of the past thinks possible. It can say nothing the generator cannot produce.

**Combinatorial purged cross-validation (CPCV).** Cut the sample into `G` groups and take every choice of `k` as the test set: `C(G, k)` splits. Training uses the rest, *purged* of observations whose label window overlaps a test group and *embargoed* after it. Each group is tested in `C(G-1, k-1)` splits, so out-of-sample predictions can be assembled into that many complete paths through history. The spread of a statistic over the paths shows how much of a trained rule's result is the order of events, which one walk-forward path cannot say.

## Why it matters

A VaR model, a stress test, a drawdown limit and a position-sizing rule are all statements about a distribution of futures; the bootstrap and the copula are good at some of it and blind to the rest, and a learned generator can look convincing in a scatter plot while having tails half as thick as the data's. And a single backtest path is one draw: CPCV and scenario backtests show how wide the draw can be.

## How this repo uses it

`src/scenarios/generators.py` has the eight generators behind `generate(name, returns, n, horizon, seed, **params)` returning `(n, horizon, assets)` daily returns; the classical ones are numpy, the learned ones (`wgan_gp`, `factor_vae`, `diffusion`) need torch and take `window=` to generate paths. `src/scenarios/quality.py` has `quality_report` and `compare`. `src/scenarios/backtest.py` has `scenario_backtest` (a strategy is a function from a DataFrame of daily returns to weights held at each close, earning the next day's return; costs on turnover). `src/validation/cpcv.py` has `cpcv_splits` (disjoint, purged, embargoed), `n_paths` and `assemble_paths`; walk-forward and purged k-fold were already in `src/validation/walk_forward.py`. The generators use only the history they are handed; to feed a backtest, hand them what was known at the time.

## What we found

Five ETFs (SPY, TLT, GLD, EEM, IEF), the last 2500 days, 500 scenarios of 21 days each (`python -m experiments.deep_world`):

| generator | marginals (KS) | volatility error | correlation error | kurtosis ratio | clustering error | ES ratio |
|---|---|---|---|---|---|---|
| historical windows | 0.010 | 0.07 | 0.05 | 0.84 | 0.04 | 0.98 |
| bootstrap, iid | 0.008 | 0.02 | 0.02 | 1.05 | 0.26 | 1.03 |
| bootstrap, stationary blocks | 0.012 | 0.04 | 0.03 | 0.93 | 0.02 | 0.96 |
| copula (Student) | 0.009 | 0.03 | 0.05 | 1.02 | 0.25 | 0.95 |
| risk factors | 0.022 | 0.04 | 0.11 | 0.88 | 0.03 | 0.97 |
| ARIMA–GARCH | 0.023 | 0.09 | 0.06 | 0.56 | 0.11 | 0.95 |
| factor VAE | 0.032 | 0.17 | 0.12 | 0.47 | 0.26 | 0.81 |
| WGAN-GP | 0.041 | 0.05 | 0.15 | 0.50 | 0.26 | 1.02 |
| diffusion | 0.036 | 0.05 | 0.04 | 0.76 | 0.26 | 1.00 |

The iid bootstrap and the copula get marginals and correlations right but have none of the data's volatility clustering (error 0.26); the block bootstrap keeps it (0.02) and the risk-factor model nearly (0.03). The learned generators are no better than the bootstrap on any measure, have tails about half as thick (kurtosis ratio 0.47 to 0.76) and, drawing days independently, no clustering. None of this says learned generators are bad: it says that for five liquid ETFs with 2500 days there is little for them to add to a bootstrap.

*A VaR test.* Fit on the first 1500 days of a 3000-day sample, ask each generator for the 99% one-day VaR of the equal-weight portfolio, and count exceedances in the other 1500 days (which include 2020 and 2022): the bootstrap (iid) gets 1.3% (Kupiec p-value 0.22, not rejected); historical windows and stationary blocks 1.8% (p 0.005); the copula, ARIMA–GARCH and the WGAN-GP 2.3% (p about 2e-5); the risk-factor model 2.1%; diffusion 3.7%; the factor VAE 5.3%. Every generator under-predicted the risk of a later period that contained larger moves than its training years, and the ones with the thinnest tails did worst.

*Scenario backtests* (100 paths of 756 days, 5 bp cost): the equal-weight book's median Sharpe ratio is 0.85, 0.86 and 0.78 under the iid bootstrap, the block bootstrap and ARIMA–GARCH, and its 5th percentile -0.05, 0.22 and -0.37; a six-month trend rule has medians of 0.51, 0.72 and 0.47 and 5th percentiles of -0.57, -0.23 and -0.26, with a shallower median drawdown (-8% to -9% against -11% to -12%). The block bootstrap, which keeps the trending and clustering of the data, is the kindest to the trend rule; the iid bootstrap, which destroys serial dependence, is the harshest.

*CPCV.* A ridge regression of next-day return of the equal-weighted ETF book on its last five days, traded by the sign of its forecast, with 8 groups and 2 tested (28 splits, 7 paths): the Sharpe ratio over the paths averages -0.01 with a standard deviation of 0.11 (from -0.15 to 0.18), against 0.17 for one walk-forward split on the second half and 0.69 for holding the book. The rule has nothing; a single walk-forward number would have suggested a little.

## Pitfalls

- A generator reproduces the sample it was fitted on, including its omissions: a period without a crisis gives scenarios without one (the VaR test above). Stress scenarios have to be added deliberately.
- A good-looking joint plot is not a good tail. Read `kurtosis_ratio`, `tail_dep_error` and `es_ratio`, not only the correlation matrix.
- Scenario backtests with an i.i.d. generator understate how trend and volatility-managed rules behave in the real world, which trends and clusters; compare generators.
- CPCV is expensive (one fit per split) and its paths share data, so their spread understates the uncertainty of the future; it measures sensitivity to the order of events, not out-of-sample performance in advance.
- A learned generator trained on the whole history leaks the future into any backtest that uses it for the past: refit it at each origin, or use it only for forward-looking risk.

## Try it

```bash
python -m experiments.deep_world
python - <<'PY'
from src.framework.data import load_default_bundle
from src.utils.config import load_config
from src.scenarios import generate, quality_report
r = load_default_bundle(load_config()).returns[["SPY", "TLT", "GLD"]].dropna()
print(quality_report(r, generate("arima_garch", r, 500, 21, seed=0)))
PY
```
