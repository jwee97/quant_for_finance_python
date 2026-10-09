---
title: "Hierarchical Bayes, Bayesian decisions and Gaussian processes"
slug: hierarchical-bayes-decisions-and-gaussian-processes
difficulty: 3
chapter: Platform
prerequisites: [bayesian-portfolio-construction, mean-variance-and-shrinkage, factor-models-and-machine-learning]
stages: []
files: [src/portfolio/hierarchical_bayes.py, src/portfolio/decision_theory.py, src/framework/allocators_bayes.py, src/models/gaussian_process.py, src/strategies/gp_models.py, experiments/bayes_world.py]
figures: []
tests: [tests/test_hierarchical_bayes.py, tests/test_gaussian_process.py]
models: [gp_factor_model]
---

# Hierarchical Bayes, Bayesian decisions and Gaussian processes

## In one sentence

Three ways of being honest about what the data do not say: shrink each asset's expected return toward its group's and each group's toward the whole by amounts the data choose (a hierarchical prior), choose the portfolio that maximises expected utility over what the returns might be once parameter uncertainty is included (Bayesian decision theory), and forecast with a Gaussian process, a nonparametric regression that reports how sure it is at every point.

## The idea

**Hierarchical Bayes.** The sample mean of a few years of returns is noise of the same size as the differences between assets, so an optimiser fed sample means builds on error. Shrinking all means toward the grand mean helps (the Bayes-Stein estimator, already in the [Bayesian portfolio guide](bayesian-portfolio-construction.md)) but treats a bond and a stock as alike. A hierarchical model has two scales: `tau_a`, how far an asset's expected return lies from its group's, and `tau_g`, how far a group's lies from the common mean. The sample mean is `xbar ~ N(mu, Sigma/T)`; given the scales the posterior of `mu` is Gaussian in closed form, and the scales themselves have a marginal likelihood that can be evaluated on a grid, so the whole posterior comes without a Markov chain. When the groups differ and their members are alike it pools within groups; when the groups are alike it collapses to ordinary shrinkage; when nothing differs it pools everything.

**Bayesian decision theory.** A plug-in optimiser estimates parameters, treats them as true and optimises. A Bayesian one asks what return distribution to expect with the parameter uncertainty included (draw parameters from the posterior, then a return from each) and picks the action with the highest expected utility over those draws. The utility can be power (`crra`; gamma = 1 is log), exponential (`cara`), the mean-variance quadratic, loss-averse (`shortfall`: losses weigh `kappa` times a gain) or a tail measure (`cvar`). Two quantities say what the uncertainty costs: the *regret* of a decision (the expected utility of the best weights for the true parameters, less that of the decision) and the *expected value of perfect information*, the most that any extra data could add.

**Gaussian process regression.** A prior over functions in which any finite set of values is multivariate normal with covariance `k(x, x')`; observing noisy values gives a normal posterior at every point, with a mean and a variance, in closed form. The kernel is the model: a squared-exponential kernel with a length scale per input learns smooth nonlinear effects and shows which inputs matter (an irrelevant one gets a very long length scale); a Matern kernel allows rougher functions; a linear kernel is Bayesian ridge; kernels add. The hyperparameters are set by the marginal likelihood, whose gradient is analytic here, so there is no validation set to spend.

## Why it matters

The expected return is the input every optimiser trusts most and knows least. Pooling, a decision that respects what is unknown, and a regression that says where it rests on little data are three answers to the same problem, and each can be checked against a world whose truth is known.

## How this repo uses it

`src/portfolio/hierarchical_bayes.py` has `hierarchical_posterior` (the grid posterior, its mean, covariance, draws, and the inferred scales) and `hierarchical_means`; the allocator `hierarchical_bayes` takes the groups from the bundle's groups or asset classes and builds the capped long-only mean-variance portfolio on the posterior mean. `src/portfolio/decision_theory.py` has the utilities, `bayes_weights` (the sample-average maximiser under bounds and a budget), `predictive_draws`, `regret` and `value_of_information`; the allocator `bayes_expected_utility` draws from the normal-inverse-Wishart posterior of the trailing window and maximises the chosen utility of next month's returns. `src/models/gaussian_process.py` is an exact GP (Cholesky, analytic gradients, ARD, sums of kernels, posterior samples), checked against scikit-learn to rounding; the model `gp_factor_model` fits it each month to the price characteristics and next month's relative return, with at most `max_points` rows and hyperparameters re-optimised every `optimize_every` months, and `model.uncertainty(bundle)` returns the posterior standard deviation behind each forecast.

## What we found

All simulated (`python -m experiments.bayes_world`, about a minute and a half), so these check the tools against a known truth; they are not evidence about markets.

*Hierarchical means.* Twenty assets in four groups, 60 periods, 60 simulated worlds each. Mean squared error of the estimated expected returns (x 1e5), and the true expected utility of the long-only mean-variance portfolio built on them (basis points a year):

| world | sample mean | one common mean | hierarchical |
|---|---|---|---|
| groups differ | 5.41 (utility 1389.1) | 5.24 (1389.2) | 3.68 (1394.8) |
| groups alike | 5.41 (908.7) | 3.54 (913.8) | 3.54 (912.2) |
| everything alike | 5.41 (797.7) | 0.25 (799.0) | 0.27 (798.7) |

Where the groups differ, pooling within them cuts the error by 30% against both the sample and a single common mean; where they do not, it costs nothing against the common mean. The portfolio gain is small, a few basis points on 1,389, because a diversified mean-variance portfolio is less sensitive to the means than the means are to their noise.

*Expected utility and the Merton fraction.* For one risky asset with a 6% expected return and 16% volatility, maximising expected power utility over simulated returns gives 1.169, 0.469 and 0.234 of wealth at relative risk aversion 2, 5 and 10; the closed form `mu / (gamma sigma^2)` is 1.172, 0.469 and 0.234.

*What uncertainty changes.* Two assets, the first with an expected return of 1% a month known to within a standard deviation of 1%. Taking the uncertainty into account moves the weight on the first asset from 0.458 to 0.437 under power utility, 0.320 to 0.308 under loss aversion, and not at all under CVaR (0.285), and changes expected utility by less than 0.0001 in each case. Perfect information would be worth 0.0021, 0.0017 and 0.0010 of utility. The decision is not very sensitive to the uncertainty here; it matters more when the estimates are further from the truth than their stated spread or when positions are concentrated.

*The Gaussian process.* With three inputs of which one matters, the length scales found are 1.3 for it and 100 (the upper bound) for the other two, the noise variance 0.0104 against 0.01 planted, and the out-of-sample error 0.034. 94.5% of new observations fall inside the 95% intervals. A square and a product of two inputs: out-of-sample R-squared 0.998 for the process with a squared-exponential plus linear kernel, -0.02 for the linear one.

*As a return forecaster.* 60 stocks, ridge, a random forest and the process on the rank IC: a world where momentum and reversal pay in a straight line gave ridge 0.130 (t 6.8), the forest 0.125 and the process 0.102 (t 5.1); a world with a U-shaped momentum effect plus a straight reversal gave ridge 0.085, the forest 0.110 and the process 0.045 (t 3.3). The process is exact only up to a few hundred points here (500 rows against several thousand for ridge), and a signal this weak next to the noise gives a flexible kernel little to learn from: it did not beat the forest on either world. It is a good regression tool and a modest return forecaster at this data size; a sparse approximation would let it use more rows.

## Going deeper

```
hierarchy      xbar | mu ~ N(mu, Sigma/T);   mu = Gm theta + delta,  delta ~ N(0, tau_a^2 I);   theta = m + eta,  eta ~ N(0, tau_g^2 I);   m flat
marginal       xbar ~ N(m 1, Omega + V),   Omega = tau_a^2 I + tau_g^2 G G' + s0^2 11',   V = Sigma/T         (evaluated on a grid of (tau_a, tau_g))
posterior      mean = m 1 + Omega (Omega + V)^-1 (xbar - m 1)        cov = Omega - Omega (Omega + V)^-1 Omega          mixed over the grid by the marginal likelihood
decision       w* = argmax_w  (1/S) sum_s u(w; R_s),   R_s ~ N(mu_s, Sigma_s),  (mu_s, Sigma_s) ~ posterior          regret(w) = E_theta[ U(w*(theta); theta) - U(w; theta) ]
GP             mean(x*) = k(x*, X)(K + s^2 I)^-1 y     var(x*) = k(x*, x*) - k(x*, X)(K + s^2 I)^-1 k(X, x*)     log ML = -y'(K + s^2 I)^-1 y / 2 - log|K + s^2 I| / 2 - n log(2 pi) / 2
```

## Pitfalls

- The groups are an assumption. A grouping that does not match how returns really cluster does not hurt much (the scale for groups shrinks to zero) but does not help; name the groups from the economics, not from the data you are about to shrink.
- The model takes the covariance as known and uses a shrunk sample estimate. Uncertainty about the covariance is what the normal-inverse-Wishart posterior in the decision allocator carries.
- Expected utility over a posterior is as good as the posterior. With a flat or badly centred prior the Bayes decision inherits its mistakes; it is not a way of knowing more than the data say.
- Utilities that depend on a tail (CVaR, shortfall) need many draws to be stable; the optimiser here uses a few hundred and long-only bounds, which is enough to see the direction and not to resolve small differences.
- A Gaussian process costs a Cholesky factorisation of an n by n matrix. At a few thousand rows it is slow, and the hyperparameters of a flexible kernel on a very noisy target are poorly determined; check the length scales and the noise variance, which tell you whether the process found anything.
- The posterior standard deviation is the model's own opinion, calibrated only if the kernel is a good description of the function. The coverage check above is on data drawn from the prior itself.

## Try it

```bash
python -m experiments.bayes_world
quant backtest --model gp_factor_model --param kernel=rbf+linear --param max_points=400
quant backtest --model characteristic_regression --allocator hierarchical_bayes --alloc-param lookback=756
quant backtest --model characteristic_regression --allocator bayes_expected_utility --alloc-param utility=crra --alloc-param risk=5
```
