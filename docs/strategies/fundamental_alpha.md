# fundamental_alpha

*Family: fundamental*

## What it bets on

The optimal alpha model on fundamental factors: weights from the covariance of their information coefficients, factors made orthogonal first, optionally per bucket of value, growth or earnings variability

Qian, Hua and Sorensen's alpha model: every month, standardise each factor across stocks, make them independent of each other (Gram-Schmidt in the order listed, or symmetrically), measure each factor's
information coefficient against the next month's return, and weigh the factors by ``Sigma^-1 mu`` of those ICs over the last ``window`` months (the weights that maximise the information ratio of the
composite). Only ICs whose month has ended are used. ``factors`` lists factors and styles: a style (``value``, ``quality``, ``momentum``, ``estimates``, ``valuation`` or ``all``) is the composite of its
factors, so the default weighs three composites. With ``context`` set the whole model runs separately in each tercile of that variable (``value``, ``growth``, ``earnings_variability`` or ``size``), so
the factors can matter differently for cheap and dear stocks. ``weights='equal'`` skips the learning. Needs ``data/user/fundamentals.csv``.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factors` = 'value,quality,momentum', `weights` = 'optimal', `orthogonalize` = 'gram_schmidt', `context` = '', `window` = 36, `min_obs` = 12, `shrink` = 0.3, `nonnegative` = False, `path` = 'fundamentals.csv', `lag_days` = 60, `expiry` = 400

## Run it

```bash
quant backtest --model fundamental_alpha --tearsheet
```

## Caveats

Reads company statements from a file you supply (data/user/fundamentals.csv); without it the model refuses to run, except the price-only momentum factors. Needs a wide cross-section of stocks, not 15 ETFs, and statements that are point-in-time.

## Learn more

[Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects](../techniques/fundamental-factors.md)
