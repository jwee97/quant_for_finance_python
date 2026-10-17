---
title: "Stochastic differential equations: schemes, exact samplers and fitting"
slug: sde-simulation-and-estimation
difficulty: 3
chapter: Ch. 18
prerequisites: [resampling-and-monte-carlo, option-pricing-models-and-calibration]
stages: []
files: [src/probability/sde.py, src/probability/ou.py, src/derivatives/levy.py]
figures: []
tests: [tests/test_sde.py, tests/test_ou.py, tests/test_levy.py]
models: []
---

# Stochastic differential equations: schemes, exact samplers and fitting

## In one sentence

A path of `dX = a(X) dt + b(X) dW` is built step by step, how fast the error falls as the step shrinks depends on the scheme and on what you want from the path, and for the few equations that matter most (Brownian motion with drift, Ornstein-Uhlenbeck, the square-root process, the Lévy models) there are exact samplers that need no step at all.

## The idea

Two errors matter. The **strong** error `E|X_T - X_T^h|` measures how close the simulated path is to the true one driven by the same noise: it matters for hedging and for path-dependent payoffs. The **weak** error `|E f(X_T) - E f(X_T^h)|` measures how close the *distribution* is, which is all a European price needs. **Euler-Maruyama** has strong order 1/2 and weak order 1; **Milstein** adds the term `b b' (dW^2 - dt) / 2` and has strong order 1.

The square-root (Cox-Ingersoll-Ross) process `dv = kappa (theta - v) dt + xi sqrt(v) dW` is the variance of the Heston model and a short rate. Its transition law is a scaled **non-central chi-square**, so it can be sampled exactly. When the exact sampler is not available (inside a Heston step with a correlated price), Euler needs a rule for negative variance (**full truncation**) and very many substeps; **Andersen's (2008) quadratic-exponential (QE)** scheme matches the first two moments of the variance step by a quadratic of a normal (when the variance is high) or by a mixture of a point mass and an exponential (when it is low), and corrects the log-price step so that the discounted price stays a martingale for any step, with the step being a day or a week.

The **Ornstein-Uhlenbeck** process is exact too: `X_{t+dt} = mu + (X_t - mu) e^{-theta dt} + sigma sqrt((1 - e^{-2 theta dt}) / (2 theta)) Z`, an AR(1) with coefficient `e^{-theta dt}`. The **Lévy** models of the pricing guide have exact increments over any interval (a gamma or inverse-Gaussian clock for the variance gamma and NIG, a Poisson number of jumps for Merton and Kou).

**Fitting** uses the exact transition densities, so there is no discretisation bias: ordinary regression followed by the map `theta = -ln(phi) / dt` for the Ornstein-Uhlenbeck process (the Euler shortcut `theta = (1 - phi) / dt` is biased when `theta dt` is not small), the non-central chi-square likelihood for CIR, closed forms for geometric Brownian motion, all with standard errors (by the delta method or the information matrix).

## Why it matters

Monte Carlo is the fallback for every product without a formula, and a forecaster is a simulator fitted to data; both inherit the scheme's error. Using an exact sampler where one exists removes the error; using QE where one does not removes the failure that makes a naive Heston simulation misprice at realistic step sizes (a daily step with a high volatility of variance).

## How this repo uses it

`probability/sde.py` has `euler_maruyama` and `milstein` (with the Brownian increments returnable so that a run can be compared with the exact solution), `strong_error` and `weak_error`, `simulate_cir` (exact, or full-truncation Euler), `simulate_heston` (QE or full truncation) and `fit_gbm` and `fit_cir`. `probability/ou.py` has the exact Ornstein-Uhlenbeck simulation, fit, half-life and Dickey-Fuller statistic (see the guide on pairs). `derivatives/levy.py` has the exact samplers, the densities (closed form where one exists, Fourier inversion otherwise) and `fit_returns`, a maximum-likelihood fit of a Lévy model's shape and drift to a series of returns, ranked by `compare_fits` with the information criteria. The Forecast tab's `garch_vg`, `garch_nig` and `heston` methods use them.

## What we found

- **The orders are what the theory says.** Euler's strong order measured near one half and Milstein's close to one; the weak order of both is about one when the noise is shared with the exact solution.
- **The exact CIR sampler has the exact conditional mean and variance and never goes negative;** full-truncation Euler approaches it as the step shrinks.
- **QE keeps the discounted price a martingale and the variance's mean for any step,** agrees with a fine full-truncation Euler on an option price, and with the volatility of variance set to zero the log-return variance is the integrated variance. It refuses a step that would break the correction rather than returning a wrong number.
- **The fits recover what generated the data, and their errors are honest:** the standard errors describe the spread of the estimates over repeated samples for geometric Brownian motion, Ornstein-Uhlenbeck and CIR. The Euler map and the exact map differ visibly when the step is not small.
- **A Lévy fit identifies the family only roughly.** It recovers the variance, the skew and the tails of the law that made the data, prefers the normal for normal data and a fat-tailed law for fat-tailed data, and a NIG fitted to variance-gamma data matches the moments but not the family: with a few thousand returns the likelihood cannot tell them apart, and on daily ETF returns the standardised residuals of a GARCH model are only mildly non-normal.

## Going deeper

```
Euler:      X_{n+1} = X_n + a(X_n) h + b(X_n) dW_n                            strong 1/2, weak 1
Milstein:   ... + (1/2) b(X_n) b'(X_n) (dW_n^2 - h)                           strong 1
CIR exact:  v_{t+h} = c * chi2'(d, lambda),  c = xi^2 (1 - e^{-kappa h}) / (4 kappa),  d = 4 kappa theta / xi^2,  lambda = v_t e^{-kappa h} / c
QE:         psi = s^2 / m^2  (variance / mean^2 of the exact step);   psi <= 1.5:  v' = a (b + Z)^2;   psi > 1.5:  v' = 0 w.p. p, else exponential(beta)
OU fit:     X_{t+1} = c + phi X_t + e   ->   theta = -ln(phi)/dt,  mu = c/(1 - phi),  sigma^2 = 2 theta s^2/(1 - phi^2)
```

## Pitfalls

- **A strong-order result is about paths, a weak-order result about distributions.** A scheme with a good weak order can still misprice a barrier, which depends on the path.
- Euler-Maruyama on a square-root process needs a rule for negative values; the choice (truncation, reflection, absorption) changes the result when the Feller condition `2 kappa theta >= xi^2` fails.
- The QE correction needs the step to be small enough that the martingale-correction constant exists; the simulator raises when it does not.
- Maximum likelihood of a mean-reversion speed is poorly determined by a short sample: a half-life of months needs years of data, and the standard error says so.
- A fit is only as good as the model: a random walk fitted as an Ornstein-Uhlenbeck process comes out with a small positive speed and a large standard error; test for the unit root first.

## Try it

```bash
python -m pytest tests/test_sde.py tests/test_ou.py tests/test_levy.py -q
python - <<'EOF'
from src.probability import ou
x = ou.simulate_ou(0.0, 8.0, 0.0, 0.2, horizon=2000 / 252, steps=2000, seed=1)[0]       # eight years of a spread with a half-life of a month
print(ou.fit_ou(x, 1 / 252).estimates)
EOF
```
