# ml_trees

*Family: machine learning*

## What it bets on

Walk-forward tree ensemble (gradient boosting, random forest, extra trees, LightGBM, XGBoost or CatBoost) on the same price features as the ridge

Trees can capture interactions and non-linearities a ridge cannot (momentum matters more when volatility is low); with a signal-to-noise ratio this small they usually overfit unless the
leaves are large and the learning rate small, which is what the defaults enforce.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `learner` = 'hgb', `n_estimators` = 150, `max_depth` = 3, `learning_rate` = 0.05, `min_samples_leaf` = 100, `seed` = 0, `features` = 'price', `min_train` = 1260, `refit_every` = 252, `embargo` = 21

## Run it

```bash
quant backtest --model ml_trees --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Gradient boosting and representation learning on returns](../techniques/gradient-boosting-and-representation-learning.md)
