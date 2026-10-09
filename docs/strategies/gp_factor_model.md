# gp_factor_model

*Family: machine learning*

## What it bets on

Gaussian process regression on price characteristics: a nonlinear forecast with a length scale per characteristic and a posterior standard deviation, refitted monthly on returns that have already happened

A Gaussian process fitted each month-end to the characteristics (standardised across assets) and the following month's return in excess of the cross-section's average, over the last ``train_months``
months whose returns are known, subsampled to ``max_points`` rows. ``kernel`` is ``rbf``, ``matern32``, ``matern52``, ``linear`` or a sum (``rbf+linear``); the hyperparameters (a length scale per
characteristic, the signal variance, the noise) are re-optimised on the marginal likelihood every ``optimize_every`` months and held fixed in between, when only the data are refreshed. A characteristic that
does not help ends with a long length scale, and a signal that is not there ends with the noise variance near the target's, so the forecast shrinks toward zero by itself.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `kernel` = 'rbf', `characteristics` = 'mom,rev,lowvol,lowbeta,nomax,high', `train_months` = 48, `min_months` = 24, `max_points` = 500, `optimize_every` = 12, `restarts` = 0, `seed` = 0

## Run it

```bash
quant backtest --model gp_factor_model --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Hierarchical Bayes, Bayesian decisions and Gaussian processes](../techniques/hierarchical-bayes-decisions-and-gaussian-processes.md)
