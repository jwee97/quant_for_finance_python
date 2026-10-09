---
title: "Building an alpha model: standardise, orthogonalise, weigh by the IC covariance, attribute"
slug: alpha-model-construction
difficulty: 3
chapter: Platform
prerequisites: [information-coefficient, forecast-combination, regression-and-panel-statistics]
stages: []
files: [src/equity/alpha_model.py, src/equity/qp.py, src/equity/synthetic.py, src/framework/ic_combination.py, experiments/equity_world.py]
figures: []
tests: [tests/test_equity_alpha_model.py, tests/test_equity_qp.py, tests/test_ic_combination.py, tests/test_equity_world_experiment.py]
models: []
---

# Building an alpha model: standardise, orthogonalise, weigh by the IC covariance, attribute

## In one sentence

A handful of factor tables becomes one alpha in four steps: put every factor on one scale, remove what the factors say in common, weigh them by the means and covariance of their information coefficients so that the combination has the highest information ratio the data support, and run a multivariate cross-sectional regression to see what each one adds to the others.

## The idea

**Standardise.** On every date, clip each factor at its 1st and 99th percentile, subtract the cross-sectional mean and divide by the cross-sectional standard deviation. After that a weight of 0.3 means the same thing for every factor; before it, the factor with the widest spread decides the answer.

**Orthogonalise.** Book to price and earnings yield, momentum and earnings revisions say much the same thing, and a weighted sum of near-copies counts one idea several times. Gram-Schmidt keeps the first factor, replaces the second with the residual of a cross-sectional regression on the first, the third with its residual on the first two, and so on, date by date. Each factor then carries only what the earlier ones did not. The order is a decision, because the factor listed first keeps everything it shares. Symmetric (Loewdin) orthogonalisation removes the privilege: it returns the orthonormal set closest to the originals, whatever their order.

**Weigh.** The information coefficient (IC) of a factor on a date is the correlation across stocks of the factor with the *next* period's returns. If the factors are orthonormal on every date, the IC of a weighted sum is exactly `w'IC / ||w||` and its information ratio (mean over standard deviation of the composite's IC over time) is `w'mu / sqrt(w' Sigma w)`, where `mu` is the vector of mean ICs and `Sigma` their covariance across dates. That ratio is largest at `w = Sigma^-1 mu`, and the best ratio the data allow is `sqrt(mu' Sigma^-1 mu)` (Qian, Hua and Sorensen 2007). The covariance matters as much as the means: a factor whose IC rises when another's falls is worth more than its own IC says, and a copy of a factor already in the mix is worth less.

**Attribute.** A Fama-MacBeth regression of the next period's returns on all the factors at once, run on every date, gives each factor's return with the others held fixed, with a Newey-West t-statistic. A factor with a good stand-alone IC and a small multivariate slope is a proxy for another.

## Why it matters

Equal weights are the usual blend and a good one, because estimated weights are noisy. Weights from each factor's own IC are the next step and they fail in a specific way: they cannot see that two factors are the same bet. Weights from the IC covariance can, and with a few years of monthly ICs they can also be badly estimated, so the interesting questions are how much to trust them (shrink the covariance toward its diagonal; keep weights non-negative), and whether they beat the plain blend out of sample.

## How this repo uses it

`src/equity/alpha_model.py` holds the four steps. `standardize_all`, `gram_schmidt` and `symmetric_orthogonalize` work on dates by assets tables and treat a missing value as neutral (or drop the stock, your choice). `information_coefficients`, `ic_moments` (with the shrinkage), `max_ir_weights`, `composite_ir` and `best_possible_ir` are the weighing; the non-negative version solves a small quadratic programme with `src/equity/qp.py`, an OSQP-style ADMM solver written for this repository (Ruiz scaling, adaptive step, active-set polish) that the constrained portfolios use too. `optimal_alpha` runs the whole thing walk-forward on a grid of rebalance dates: the weights on date `t` come from the ICs of dates up to `t - horizon`, whose forward windows had ended, and equal weights until `min_obs` of them exist. `marginal_contributions` is the attribution.

Two combination rules put the same idea in the pipeline that combines *models*: `optimal_ic` weighs the models' forecasts by `Sigma^-1 mu` of their matured daily ICs (non-negative, monthly, divided by each model's dispersion so that the weights apply to forecasts in return units), and `orthogonal_ic` makes the forecasts orthogonal first and returns one combined forecast. Both appear in the dashboard's combination menu and in the tables as `ic_combination`. `fundamental_alpha` is the model that runs the alpha model on the fundamental factors (see [the fundamental factors guide](fundamental-factors.md)).

The tests plant the truth. A world of factors with known premia and known correlations checks the algebra exactly: orthogonality of every output date, the span of the originals preserved, the composite IC equal to `w'IC/||w||` to 1e-12, the weights equal to `Sigma^-1 mu` and better than 2,000 random weight vectors, the walk-forward weights recovering the planted signs, and a causality check (change the returns after a date and no earlier alpha moves).

## What we found

On a world with two noisy copies of one source of return, one independent source and one pure-noise model (4 models, 20 stocks, 1,800 days, forecast horizon one day), the covariance-aware weights gave the independent model 1.6 to 2.0 times the weight of each copy and the noise model almost nothing (shares of 0.27, 0.24, 0.48 and 0.00 for the two copies, the independent model and the noise model, on the first of the seeds tried). The existing IC-weighted rule gave it 1.2 to 1.4 times, because it sees only each model's own IC. The information ratio the data promised at each date (0.52 to 0.60) was the ratio the combination delivered out of sample (0.54 to 0.59); an equal-weight blend delivered 0.49 to 0.51.

On the simulated company world of the [fundamental factors guide](fundamental-factors.md) (80 companies, 14 years, seed 0, `python -m experiments.equity_world`), five style composites had mean monthly rank ICs between 0.034 (quality) and 0.079 (value). Combined with equal weights they reached 0.098 (t 9.7); with learned weights 0.102, with Gram-Schmidt 0.102 and with symmetric orthogonalisation 0.101. Nearly all the gain is from combining, not from learning the weights: when the factors are about equally good, the weights have little to find.

The attribution shows why stand-alone ICs mislead. Four value yields that share the market value in their denominator had stand-alone ICs of 0.062 (book to price), 0.076 (earnings yield), 0.077 (cash flow to enterprise value) and 0.068 (sales to enterprise value), and the one that was planted is the *lowest*. Held fixed against each other, the t-statistics of the slopes were 2.3 for book to price and 0.7, 1.5 and 0.3 for the others. All of this is a simulated world with a known truth; it says the tools recover what was planted, not that a real market pays the same.

## Going deeper

```
standardise:   z_i = (clip(x_i) - mean(clip(x))) / sd(clip(x))                     per date, across stocks
Gram-Schmidt:  q_1 = z_1;  q_k = z_k - sum_{j<k} <z_k, q_j> / <q_j, q_j> q_j          per date, then rescaled to unit variance
Loewdin:       Q = Z (Z'Z)^(-1/2) sqrt(n-1)                                         the orthonormal set nearest to Z in Frobenius norm
composite:     IC(w'q) = w'IC / ||w||         IR = w'mu / sqrt(w' Sigma w)         exact for Pearson IC and orthonormal q
optimum:       w* = Sigma^-1 mu,   IR* = sqrt(mu' Sigma^-1 mu)                      with Sigma shrunk: (1-s) Sigma + s diag(Sigma)
no shorts:     minimise  1/2 w'Sigma w - mu'w  subject to  w >= 0                   same direction as the maximum-IR problem
Fama-MacBeth:  r_(i,t+1) = g_0t + sum_k g_kt z_(k,i,t) + e;   report mean(g_k) and its Newey-West t
```

## Pitfalls

- With K factors and 36 monthly ICs the sample covariance is noisy and its inverse is worse. Shrink it (`shrink=0.3` here), keep K small by combining factors within a style first, and prefer `nonnegative=True` when you cannot defend a short weight.
- An IC measured for date `s` needs the returns after `s`. Weights that use an IC whose window has not ended look ahead. `optimal_alpha` and the pipeline rules use only matured ICs; any code you add must too.
- Gram-Schmidt order is a modelling decision. Put the factor you trust most (or the one with the best economic story) first, or use the symmetric version.
- The identity `IC(w'q) = w'IC / ||w||` is exact for Pearson ICs. Rank ICs make it approximate, which is fine for weights and wrong for proofs.
- Overlapping forward windows (daily ICs of a 21-day return) have strong serial correlation: the mean is unbiased but the standard error is far smaller than the sample size suggests. Do not read a t-statistic from them without a Newey-West correction.

## Try it

```bash
python -m experiments.equity_world                      # the planted-truth world: factor ICs, attribution, the combined alpha
quant backtest --model fundamental_alpha --param path=fundamentals.csv --tearsheet
```

```python
from src.equity.alpha_model import optimal_alpha, marginal_contributions
model = optimal_alpha(factors, forward, horizon=1, window=36, shrink=0.3, orthogonalize="gram_schmidt")
model.alpha          # the composite, dates x assets
model.weights        # the weights used on each date
marginal_contributions(factors, realized)   # what each factor adds with the others held fixed
```
