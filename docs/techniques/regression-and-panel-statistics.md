---
title: "Regression, HAC inference and panel methods: OLS to Fama-MacBeth"
slug: regression-and-panel-statistics
difficulty: 2
chapter: Ch. 11
prerequisites: [performance-metrics]
stages: []
files: [src/stats/regression.py, src/stats/panel.py, src/stats/inference.py]
figures: []
tests: [tests/test_stats.py]
models: []
---

# Regression, HAC inference and panel methods: OLS to Fama-MacBeth

## In one sentence

Most of what quant research calls evidence is a regression with standard errors, and the standard errors are the part that is usually wrong; `src.stats` provides OLS, WLS, GLS, HAC and clustered errors, rolling regressions, panels and Fama-MacBeth with the corrections that financial data needs.

## The idea

For `y = X b + e` the OLS estimate is `b = (X'X)^-1 X'y`. Its textbook covariance `s^2 (X'X)^-1` assumes the errors are uncorrelated and have constant variance, neither of which holds for returns. The sandwich form `(X'X)^-1 S (X'X)^-1` replaces `s^2 X'X` by an estimate `S` of the covariance of the scores `x_t e_t`:

- **White** uses `sum e_t^2 x_t x_t'` (heteroskedasticity only).
- **Newey-West HAC** adds Bartlett-weighted autocovariances of the scores up to a lag `L`, so the errors may also be serially correlated; `newey_west_lag` gives the usual `floor(4 (n/100)^(2/9))`, and the `cov` argument takes `HC0` to `HC3`, `HAC`, `cluster` or `cluster2`.
- **Cluster-robust** (`cov="cluster"`) sums the scores within a group (a firm, a date) before squaring; `cluster2` is the Cameron-Gelbach-Miller two-way version that allows correlation by asset and by date. In a panel, **Driscoll-Kraay** (`panel_ols(cov="driscoll_kraay")`) sums the scores across assets for each date and applies a HAC to the result, so the errors may be correlated across assets and through time.

`wls` and `gls` use a known error covariance; `feasible_gls_ar1` estimates an AR(1) error by Cochrane-Orcutt style iteration. `rolling_ols` re-estimates on a moving window. The diagnostics (`durbin_watson`, `jarque_bera`, `breusch_pagan`, `white_test`, `breusch_godfrey`, `ramsey_reset`, `variance_inflation`) test the assumptions you are about to rely on.

For panels `panel_ols` gives pooled, entity and time fixed effects with clustered errors; `random_effects` is the Swamy-Arora style estimator; `hausman_test` compares them. **Fama-MacBeth** (`fama_macbeth`) runs one cross-sectional regression per date and reports the time-series mean and standard error of each slope, with a Newey-West option, so the standard error reflects variation across time rather than across stocks. `fama_macbeth_two_pass` adds the Shanken correction for estimating betas first, and `grs_test` is the Gibbons-Ross-Shanken test of whether all alphas are jointly zero.

## Why it matters

A t-statistic of 3 from white-noise standard errors on overlapping 21-day returns is closer to 1: overlapping windows make the errors autocorrelated by construction. Choosing the wrong covariance estimator is the most common way a spurious factor gets published, so the library makes the honest estimator one argument away.

## How this repo uses it

The other modules lean on it: `src.stats.inference` has the Sharpe-ratio inference (probabilistic, deflated and haircut Sharpe, minimum track-record length, Romano-Wolf and Storey q-values) used by the validation stage; `src.stats.event_study` and `src.derivatives.volatility` use the same regressions. Every estimator is a plain function of numpy arrays or pandas frames returning a small result object with coefficients, standard errors, t-statistics, p-values and the covariance matrix.

## What we found

The test suite checks the OLS, HAC and cluster estimators against `statsmodels`, the fixed-effects estimators against a least-squares dummy-variable regression, the clustered panel errors against the closed-form formula, and Fama-MacBeth on a simulation with a planted premium that it recovers; the two-pass test confirms that the Shanken correction widens the standard errors and that the GRS test accepts a true model. The real-data sections of [the institutional findings](../institutional_findings.md) use these tools on the platform's 15 ETFs.

## Going deeper

```
OLS:           b = (X'X)^-1 X'y
Sandwich:      V = (X'X)^-1 S (X'X)^-1,   S = G_0 + sum_{l=1..L} w_l (G_l + G_l'),   w_l = 1 - l/(L+1)
Fama-MacBeth:  b_hat = mean_t(b_t),   se = sd(b_t)/sqrt(T)
Shanken:       V_corrected = (1 + lambda' Sigma_f^-1 lambda) V + Sigma_f
GRS:           (T - N - K)/N * a' S^-1 a / (1 + mu' Omega^-1 mu)  ~  F(N, T - N - K)
```

## Pitfalls

- A HAC lag that is too short leaves the size distortion in place; too long inflates variance. Report the lag.
- Fama-MacBeth assumes the cross-sectional regressions are independent through time; it does not fix errors that are persistent in the slopes.
- With few clusters (under about 30) clustered standard errors are too small.
- Fixed effects remove the variation you may care about; look at the Hausman statistic rather than assuming.

## Try it

```python
import numpy as np
from src.stats import ols

rng = np.random.default_rng(0)
x = rng.standard_normal((500, 2))
y = x @ [0.5, 0.0] + rng.standard_normal(500)
fit = ols(y, x, cov="HAC", lags=5)
print(fit.params.round(3).to_dict(), fit.tvalues.round(1).to_dict())
```
