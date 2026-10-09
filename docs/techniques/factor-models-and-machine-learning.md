---
title: "Factor models and machine learning: statistical APT, macroeconomic factors, cross-sectional characteristics, LASSO to neural networks"
slug: factor-models-and-machine-learning
difficulty: 3
chapter: Platform
prerequisites: [cross-sectional-factors, machine-learning-for-returns, information-coefficient, portfolio-theory-and-constrained-books]
stages: []
files: [src/strategies/factor_models.py, src/equity/theory.py, src/equity/contextual.py, experiments/portfolio_world.py]
figures: []
tests: [tests/test_factor_models.py, tests/test_equity_theory.py]
models: [apt_alpha, macro_factor_model, characteristic_regression, ml_factor_model]
---

# Factor models and machine learning: statistical APT, macroeconomic factors, cross-sectional characteristics, LASSO to neural networks

## In one sentence

Four ways to forecast which assets will do better next month from what is known today: what an asset earned beyond a few statistical factors, how exposed it is to macroeconomic risks the market has been paying for, this month's returns regressed on last month's characteristics, and the same characteristics fed to a learner (OLS, ridge, LASSO, elastic net, principal-components or partial-least-squares regression, a decision tree, a random forest or a small neural network).

## The idea

**Statistical APT (`apt_alpha`).** The arbitrage pricing theory says only exposure to a few common factors is paid; anything beyond it is a mispricing, or luck. Without named factors the common ones are taken to be the first K principal components of the trailing two years of returns. The factor returns are the *raw* returns of the portfolios the components define, not the demeaned scores, because the means of the factors are the premia that price the assets and a factor with zero mean by construction prices nothing. Each asset is regressed on them with an intercept; the score is the intercept over the residual volatility (an appraisal ratio), skipping the latest month so the short-term reversal does not leak into a long-term signal. It is `jensen_alpha` with K factors instead of the market alone.

**Macroeconomic factors (`macro_factor_model`).** Chen, Roll and Ross: returns move with unexpected changes in a few economic variables, and an asset more exposed to a risk that is priced earns more. The factors are daily changes of the 10-year yield, the term spread, the credit spread, the VIX, breakeven inflation and the dollar (or the series you name), made unexpected by an AR(1) filter and scaled by their own volatility. Each asset's loadings come from a rolling multivariate regression. At each month-end the premium per unit of loading is the average, over the last five years of months whose returns are already known, of the cross-sectional slope of the next month's return on the loadings (Fama and MacBeth); the score is loadings times premia.

**Cross-sectional characteristics (`characteristic_regression`).** Fama and MacBeth's design as a forecaster. Every month-end, regress the following month's returns of all assets on their characteristics (momentum, reversal, low volatility, low beta, no extreme day, closeness to the one-year high, each oriented so more is better); the forecast is the average slope of the last 60 months whose returns are known, times today's characteristics. A characteristic that has not paid gets a slope near zero and so little weight.

**Machine learning on the same characteristics (`ml_factor_model`).** Instead of the average slope, a learner chosen by `kind` is fitted each month on every earlier month-end whose following month has ended (the last 60), with characteristics standardised across assets and returns demeaned across assets:

- `ols`, and `ridge` (shrinks all slopes towards zero), `lasso` (sets weak ones to exactly zero), `enet` (between the two);
- `pcr` regresses on the first few principal components of the characteristics, which are the directions that vary most; `pls` takes the directions most correlated with the return, which are the ones that matter, so with few components it usually keeps more of the signal;
- `cart` is one decision tree, `forest` an average of a hundred of them, `mlp` a small neural network. Trees and networks can find a U shape or an interaction that no straight line can.

## Why it matters

A forecast is an input to everything else in the platform (the [alpha model](alpha-model-construction.md), the books in [portfolio theory](portfolio-theory-and-constrained-books.md)), and how it is built decides what it can find. A linear model cannot see that both very high and very low capital expenditure are bad; a tree can, and at the price of needing a lot more data to find a simple thing. The statistical APT and the macro model answer a different question from the characteristics: not which stocks look good, but what is paid for taking which risk.

## How this repo uses it

All four are monthly decisions on data through the decision date, computed from prices (the macro model also needs macro series in the bundle: it says which ones it wants and refuses to run on fewer than two, and its defaults are daily market series, so no release lags arise), and checked by the generic causality test and by tests that change the data after a date and require every earlier score to be identical. The learners are scikit-learn; the slow ones (trees, forest, network) are refitted quarterly by default and the linear ones monthly (`refit_every` overrides). Nothing is tuned: the regularisation strength, the number of components and the tree depth are fixed defaults. `src/equity/theory.py` has the same ideas as closed-form tools (`apt_two_pass`, `statistical_factors`, `apt_arbitrage_portfolio`) described in the [portfolio theory guide](portfolio-theory-and-constrained-books.md).

## What we found

All simulated (`python -m experiments.portfolio_world`), so these are checks that each model finds what was planted and how they compare, not evidence that anything works in a market.

*Characteristics and learners.* A world of 60 stocks for 12 years in which momentum and the one-month reversal carry the premia (0.8% and 0.6% a month per standard deviation) and nothing else does. Mean monthly rank IC and its t-statistic after 36 months of warm-up:

| model | mean IC | t | time |
|---|---|---|---|
| `characteristic_regression` | 0.126 | 7.6 | |
| `ml_factor_model` OLS | 0.130 | 7.6 | 0.8 s |
| ridge | 0.130 | 7.6 | 0.4 s |
| LASSO | 0.131 | 7.3 | 0.5 s |
| elastic net | 0.131 | 7.5 | 0.4 s |
| PCR (3 components) | 0.123 | 7.1 | 0.6 s |
| PLS (3 components) | 0.134 | 7.8 | 0.5 s |
| decision tree (depth 3) | 0.100 | 5.6 | 0.5 s |
| random forest | 0.124 | 6.8 | 11.8 s |
| neural network | 0.114 | 6.9 | 5.4 s |
| `apt_alpha` (3 components) | 0.082 | 5.6 | |

The truth in this world is a straight line, so every linear learner is at the ceiling (0.12 to 0.13) and the nonlinear ones are slower and a little worse (0.10 to 0.12): they have to learn from data what the linear models are told. The statistical APT scores 0.08 here because an appraisal ratio of the unexplained mean return picks up persistent winners, which is a momentum effect; it was not built for this world.

*When the truth is not a line.* Out-of-sample R-squared when the return depends on one feature through a U shape (its square), and through a U shape plus an interaction of two features: OLS, LASSO and PLS 0.00 and 0.00 (a straight line sees nothing in a square); decision tree 0.93 and 0.78; random forest 0.92 and 0.81; neural network 1.00 and 0.99. The cost is data: a month of 60 stocks is 60 rows, and a model fitted to five years of them is 3,600 rows at a signal-to-noise ratio of a few percent, which is why the trees did not beat the line above.

*The number of factors in the statistical APT.* Six worlds with three planted common factors, each with a positive premium, and a planted mispricing in every asset. Removing 1, 2, 3 or 5 components: mean IC 0.184, 0.153, 0.130 and 0.123, t-statistics 6.7, 7.6, 10.8 and 10.0. The mean IC *falls* as more factors are removed and the t-statistic *rises* until the true number of three: with too few components, the score still contains the premia of the factors left in, which do predict returns (free-riding on the factor premia) but are not alpha and make the IC swing with the factors' own returns. Removing the right number gives the steadiest signal, not the highest mean. Five components is a little worse than three, as the extra two regressions fit noise.

*Macro factors.* A world in which the exposure to two macroeconomic changes is priced (a premium of 0.07% a day per unit of loading, far larger than any real one): the model's mean IC is 0.32 (t 8.6). The same world with nothing priced: -0.02 (t -0.7). It claims skill where there is a priced exposure and none where there is not.

## Going deeper

```
characteristic_regression : r(i,s+1) = a_s + sum_k g(k,s) z(k,i,s) + e          at each month-end s;   z standardised across assets
                            forecast(i,t) = mean( g(.,s) : the last W months with s + 1 <= t ) . z(i,t)
ml_factor_model           : y(i,s) = r(i,s+1) - mean_i r(i,s+1);  fit f on {(z(i,s), y(i,s)) : s = t-60 .. t-1};  forecast(i,t) = f(z(i,t))
                            ridge  (X'X + aI)^-1 X'y        lasso  min |y - Xb|^2 / 2n + a |b|_1        PCR  OLS on the first L principal components of X        PLS  components that maximise cov(Xw, y)
apt_alpha                 : V = principal directions of the demeaned returns;  F = R V_K  (raw returns);   R_i = a_i + b_i' F + e_i;   score_i = a_i / sd(e_i)
macro_factor_model        : x_k = (change in series k, AR(1)-filtered) / trailing sd;   b_i = rolling OLS of r_i on x;   g_s = OLS of r(.,s+1) on b(.,s);   score = mean(g over matured months) . b(.,t)
```

## Pitfalls

- With six characteristics, nine learners and a handful of settings, a good IC is easy to find by trying. The platform counts the models as trials in its multiple-testing tools; read a result of this kind with that in mind.
- The planted premia are large and the cross-section has no regime changes. A tree that does well in a simulated world with a U shape will not necessarily find one in a market, where five years of monthly data hold few observations of any extreme.
- A mean IC that goes up when you remove a hedge is a warning, not a success: the APT section above shows a score that scores better because it carries factor premia, and carries their risk too. Look at the IC's steadiness and at what the score is exposed to.
- Standardising characteristics across assets each month, and demeaning returns, makes these relative forecasts: the models say which assets beat the others, not whether the market will rise.
- LASSO's selection is unstable when characteristics are correlated: two nearly identical ones will trade places from month to month, which turns over the book. Ridge or elastic net are steadier.
- The macro model estimates loadings over two years and premia over five, so it has a long warm-up and reacts slowly; macro premia are the least stable thing in empirical finance, and a model that finds them in simulated data says nothing about whether they exist today.

## Try it

```bash
python -m experiments.portfolio_world
quant backtest --model characteristic_regression --tearsheet
quant backtest --model ml_factor_model --param kind=lasso --param alpha=0.05
quant backtest --model ml_factor_model --param kind=pls --param components=2
quant backtest --model ml_factor_model --param kind=forest --param depth=4 --param trees=200
quant backtest --model apt_alpha --param factors=3 --param window=504
quant backtest --model macro_factor_model --param series=DGS10,T10Y3M,VIX
```
