# ml_factor_model

*Family: machine learning*

## What it bets on

Machine learning on price characteristics: OLS, ridge, lasso, elastic net, PCR, PLS, a decision tree, a random forest or a small neural network, refitted monthly on returns that have already happened

The characteristics of ``characteristic_regression`` fed to a learner chosen by ``kind``, to predict the next month's return in excess of the cross-section's average. Each month-end the learner is fitted
on every earlier month-end whose following month has ended (the last ``train_months``), with characteristics standardised across assets and the target demeaned across assets, then applied to today's
characteristics. ``kind`` is ``ols``, ``ridge``, ``lasso`` (sparse: weak characteristics get exactly zero), ``enet`` (between the two), ``pcr`` (regress on the first ``components`` principal
components of the characteristics) or ``pls`` (the components most correlated with the target), ``cart`` (one decision tree of depth ``depth``), ``forest`` (an average of ``trees`` of them) or ``mlp`` (a small
neural network). Trees and the network can find nonlinear and interaction effects the linear learners cannot; the signal-to-noise ratio of monthly returns is low, so expect them to need a lot of data.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `kind` = 'lasso', `characteristics` = 'mom,rev,lowvol,lowbeta,nomax,high', `alpha` = 0.05, `components` = 3, `depth` = 3, `trees` = 100, `train_months` = 60, `min_months` = 24, `refit_every` = 0

## Run it

```bash
quant backtest --model ml_factor_model --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Factor models and machine learning: statistical APT, macroeconomic factors, cross-sectional characteristics, LASSO to neural networks](../techniques/factor-models-and-machine-learning.md)
