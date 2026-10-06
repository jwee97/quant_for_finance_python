# Glossary

Plain-language definitions of every term the guides and reports use. Each entry says where in the guides to read more. `quant explain <term>` prints an entry from the terminal.

### ADV

*Also: average daily volume*

Average daily volume: the typical amount of an asset traded per day, which sets how much you can trade before moving the price.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Allocator

In this repo, a component that turns forecasts and regimes into portfolio weights (static books, forecast stack, confidence, regime switch).

See: [plugin-framework](techniques/plugin-framework.md), [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md)

### Alpha

The part of a return that is not explained by exposure to known risks or factors; in practice, the return a strategy earns beyond what its benchmark or factor exposures predict.

See: [attribution](techniques/attribution.md)

### Annualised volatility

*Also: ann vol*

Daily return standard deviation multiplied by the square root of 252 trading days.

See: [performance-metrics](techniques/performance-metrics.md)

### Antithetic sampling

Drawing random perturbations in plus/minus pairs so their sampling noise cancels; used in evolution strategies.

See: [reinforcement-learning](techniques/reinforcement-learning.md)

### Attention

A neural-network operation that lets each element of a sequence or graph take a learned weighted average of the others.

See: [deep-time-series-models](techniques/deep-time-series-models.md), [graph-neural-networks](techniques/graph-neural-networks.md)

### AUC

Area under the ROC curve: the probability that a randomly chosen positive case receives a higher score than a randomly chosen negative case; 0.5 is chance.

See: [machine-learning-for-returns](techniques/machine-learning-for-returns.md)

### Backcast

In N-BEATS, the part of the input window a block explains; it is subtracted before the next block sees the window.

See: [deep-time-series-models](techniques/deep-time-series-models.md)

### Backtest

A simulation of how a strategy would have performed on past data, under stated rules for timing and costs.

See: [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md)

### Base rate

How often an event happens regardless of any signal, for example equities rise in about 60% of months; a forecast must beat it to have skill.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Basis

The difference between a futures or perpetual price and the spot price.

See: [crypto-carry](techniques/crypto-carry.md)

### Bayes-Stein shrinkage

Pulling noisy estimated mean returns toward a common value, with a weight that depends on how noisy they are.

See: [bayesian-portfolio-construction](techniques/bayesian-portfolio-construction.md)

### Benjamini-Hochberg

*Also: BH, false discovery rate, FDR*

A procedure that controls the expected share of false discoveries among the hypotheses you declare significant when testing many at once.

See: [multiple-testing](techniques/multiple-testing.md)

### Beta

The sensitivity of an asset's return to a market return: a beta of 0.5 means it tends to move half as much.

See: [attribution](techniques/attribution.md), [cross-sectional-factors](techniques/cross-sectional-factors.md)

### Black-Litterman

A way to combine market-implied expected returns with an investor's views, weighted by their confidence, to get steadier optimiser inputs.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Block bootstrap

*Also: stationary bootstrap*

Resampling blocks of consecutive observations rather than single days, so autocorrelation and volatility clustering are preserved.

See: [paired-sharpe-bootstrap](techniques/paired-sharpe-bootstrap.md)

### BOCPD

*Also: change point*

Bayesian online change-point detection: tracks the probability that the time since the last distribution change is any given length.

See: [regime-detection](techniques/regime-detection.md)

### Breadth

The number of independent bets per year; in the fundamental law, information ratio grows with the square root of it.

See: [information-coefficient](techniques/information-coefficient.md)

### Brier score

The mean squared error between predicted probabilities and 0/1 outcomes; lower is better.

See: [calibration](techniques/calibration.md)

### Brinson-Fachler

An attribution method that splits active return versus a benchmark into allocation, selection and interaction effects.

See: [attribution](techniques/attribution.md)

### Calibration

The match between stated probabilities and observed frequencies: events forecast at 70% should happen about 70% of the time.

See: [calibration](techniques/calibration.md)

### Calmar ratio

Compound annual growth divided by the size of the worst drawdown.

See: [performance-metrics](techniques/performance-metrics.md)

### Capacity

The amount of money a strategy can run before its trading costs erase its edge; here, the assets at which net Sharpe halves.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Carry

The return you earn from holding an asset if prices do not change, such as a bond's yield above cash or a perpetual's funding rate.

See: [cross-sectional-factors](techniques/cross-sectional-factors.md), [crypto-carry](techniques/crypto-carry.md)

### Causality test

*Also: look-ahead test*

A check that replaces all data after a cutoff with noise and requires every output on or before the cutoff to be identical, proving no future information leaks in.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### CDS spread

The price of insurance against a borrower's default; an alternative-data indicator of credit stress.

See: [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Christoffersen test

A test that value-at-risk breaches are independent through time rather than clustered.

See: [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### Clark-West test

A test of whether a larger forecasting model improves on a smaller model nested inside it, corrected for the noise cost of the extra parameters.

See: [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Cointegration

A long-run relationship between non-stationary prices such that some combination of them is stationary and tends to revert.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md)

### Condition number

The ratio of the largest to the smallest eigenvalue of a matrix; a large one means inverting it magnifies noise.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Confidence

In this repo, |2 Phi(mean/std) - 1|: how many forecast standard deviations the mean is from zero, scaled to 0-1.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Configuration fingerprint

A short hash of the declared configuration; it changes if any declared value changes and is stored with every result.

See: [experiment-database-and-reproducibility](techniques/experiment-database-and-reproducibility.md)

### Contamination

When a model's training data includes the period or series you test it on, so a good result may be memory rather than forecasting.

See: [foundation-models](techniques/foundation-models.md)

### Covariance matrix

A table of how every pair of assets' returns vary together; the foundation of portfolio risk.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### CRPS

Continuous ranked probability score: a proper scoring rule for a whole forecast distribution that generalises absolute error; lower is better.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### CSCV

Combinatorially symmetric cross-validation: the splitting scheme behind the probability of backtest overfitting.

See: [reality-check-spa-pbo](techniques/reality-check-spa-pbo.md)

### CVaR

*Also: expected shortfall*

Conditional value at risk, or expected shortfall: the average loss on the days worse than the VaR quantile.

See: [mean-cvar](techniques/mean-cvar.md), [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### DCC-GARCH

Dynamic conditional correlation: GARCH volatilities combined with a time-varying correlation matrix.

See: [dynamic-covariance](techniques/dynamic-covariance.md)

### Deflated Sharpe ratio

*Also: DSR*

The probability that a strategy's Sharpe exceeds what the best of N random trials would show, given sample length, skew and kurtosis.

See: [multiple-testing](techniques/multiple-testing.md)

### Diebold-Mariano test

*Also: DM test*

A test of whether two forecasts have different average loss, with a robust standard error.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Difference in differences

*Also: DiD*

A causal estimator comparing the change over time in a treated group with the change in a control group.

See: [causal-inference](techniques/causal-inference.md)

### Diffusion model

*Also: DDPM*

A generative model trained to reverse a gradual noising process, so it can turn noise into realistic samples.

See: [diffusion-scenarios](techniques/diffusion-scenarios.md)

### Double machine learning

*Also: DML*

A causal estimator that removes the effect of controls from treatment and outcome using flexible models, cross-fitted, then regresses the residuals.

See: [causal-inference](techniques/causal-inference.md)

### Drawdown

The fall from a running peak to a later trough, as a share of the peak.

See: [performance-metrics](techniques/performance-metrics.md)

### Dropout

*Also: MC dropout*

Randomly zeroing parts of a network during training to reduce overfitting; kept on at prediction time it gives Monte Carlo uncertainty.

See: [bayesian-deep-learning](techniques/bayesian-deep-learning.md)

### DV01

The change in a bond's value for a one-basis-point move in yield; used to make bond trades neutral to parallel rate moves.

See: [fixed-income-and-volatility-strategies](techniques/fixed-income-and-volatility-strategies.md)

### Effective rank

The exponential of the entropy of normalised eigenvalues: the number of independent risk dimensions a covariance matrix really contains.

See: [pca-effective-rank](techniques/pca-effective-rank.md)

### Embargo

A gap between the end of training labels and the start of testing that stops overlapping outcomes from leaking.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Ensemble

Several models trained separately whose predictions are averaged; disagreement between them estimates uncertainty.

See: [bayesian-deep-learning](techniques/bayesian-deep-learning.md)

### Epistemic uncertainty

Uncertainty from not knowing the right model or parameters, as opposed to irreducible noise in the data.

See: [bayesian-deep-learning](techniques/bayesian-deep-learning.md)

### Evolution strategies

*Also: ES*

A way to optimise a policy by perturbing its parameters randomly and moving toward the perturbations that scored best.

See: [reinforcement-learning](techniques/reinforcement-learning.md)

### EWMA

Exponentially weighted moving average: an average where recent observations weigh more, controlled by a half-life.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### Expected calibration error

*Also: ECE*

The weighted average gap between forecast probability and observed frequency across probability bins.

See: [calibration](techniques/calibration.md)

### Factor

A systematic source of return shared across many assets, such as market, size, value, momentum.

See: [attribution](techniques/attribution.md), [cross-sectional-factors](techniques/cross-sectional-factors.md)

### Forecast

In this repo, an object with a mean, a standard deviation and a confidence, rather than a single number.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md), [plugin-framework](techniques/plugin-framework.md)

### Foundation model

A large model pre-trained on a vast range of data that can be applied to new tasks with little or no further training.

See: [foundation-models](techniques/foundation-models.md)

### Fundamental law of active management

Information ratio is approximately the information coefficient times the square root of breadth.

See: [information-coefficient](techniques/information-coefficient.md)

### Funding rate

The periodic payment between longs and shorts of a perpetual futures contract that keeps it near spot.

See: [crypto-carry](techniques/crypto-carry.md)

### GARCH

A volatility model in which today's variance depends on yesterday's squared return and yesterday's variance, with reversion to a long-run level.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### Gaussian mixture

A distribution made of several weighted normal distributions; used to combine forecasts while keeping their disagreement.

See: [forecast-combination](techniques/forecast-combination.md)

### Graph attention network

*Also: GAT*

A neural network that lets each node of a graph weight its neighbours' features by learned attention.

See: [graph-neural-networks](techniques/graph-neural-networks.md)

### Half-life

The age at which an observation's weight in an exponential average has fallen to one half.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### Hedge (algorithm)

An online expert-aggregation rule that multiplicatively down-weights experts with recent losses.

See: [online-learning](techniques/online-learning.md)

### Hidden Markov model

*Also: HMM*

A model where an unobserved state follows a Markov chain and each state generates returns from its own distribution.

See: [regime-detection](techniques/regime-detection.md)

### HRP

*Also: HERC*

Hierarchical risk parity: allocates by recursively splitting a correlation-based tree of assets by inverse cluster variance.

See: [hierarchical-risk-parity](techniques/hierarchical-risk-parity.md)

### Impact

*Also: market impact*

The price movement caused by your own trading, approximated here as proportional to volatility times the square root of size over volume.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Information coefficient

*Also: IC*

The rank correlation between a signal and subsequent returns.

See: [information-coefficient](techniques/information-coefficient.md)

### Information ratio

*Also: IR*

Excess return over a benchmark divided by the volatility of that excess (tracking error).

See: [information-coefficient](techniques/information-coefficient.md)

### Instrumental variable

*Also: IV, 2SLS*

A variable that moves the treatment but affects the outcome only through it, used to estimate causal effects when confounding is present.

See: [causal-inference](techniques/causal-inference.md)

### Integrated gradients

A neural-network attribution method that integrates the gradient along a path from a baseline input to the actual input.

See: [explainability](techniques/explainability.md)

### Isotonic regression

Fitting a non-decreasing step function; used to recalibrate probabilities without assuming a shape.

See: [calibration](techniques/calibration.md)

### Kalman filter

A recursive estimator of a hidden state that updates as each observation arrives; used here for time-varying hedge ratios and coefficients.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md), [online-learning](techniques/online-learning.md)

### Kelly criterion

The bet size that maximises long-run growth; fractional Kelly uses a safer fraction of it.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Kupiec test

A test that the proportion of value-at-risk breaches equals the promised rate.

See: [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### Leakage

*Also: look-ahead bias*

Any path by which future information reaches a decision made in the past.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Ledoit-Wolf shrinkage

A formula that blends the sample covariance matrix with a structured target to reduce estimation error.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Log loss

The negative average log probability assigned to what actually happened; a proper score for probability forecasts.

See: [calibration](techniques/calibration.md)

### Marchenko-Pastur

The distribution of eigenvalues of a random covariance matrix; its upper edge separates signal from noise eigenvalues.

See: [pca-effective-rank](techniques/pca-effective-rank.md)

### Market regime

*Also: regime*

A persistent state of the market (such as high volatility or inflation) in which returns, risks or correlations behave differently.

See: [regime-detection](techniques/regime-detection.md)

### Mean-variance optimisation

*Also: MVO*

Choosing weights to maximise expected return for a given variance.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Minimum detectable effect

*Also: MDE*

The smallest true effect a test detects with a chosen probability (usually 80%) at a chosen significance level.

See: [power-analysis](techniques/power-analysis.md)

### Momentum

The tendency of recent winners to keep winning over the following weeks to months.

See: [momentum](techniques/momentum.md)

### N-BEATS

*Also: N-HiTS*

A neural forecaster of stacked fully connected blocks that each model part of a series and subtract it before the next.

See: [deep-time-series-models](techniques/deep-time-series-models.md)

### Net return

Return after trading costs.

See: [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md)

### Newey-West

*Also: HAC*

A standard-error estimator that is robust to autocorrelation and heteroskedasticity up to a chosen lag.

See: [attribution](techniques/attribution.md)

### No-trade band

A rule not to trade unless a position is further than a set distance from its target, saving costs at the price of tracking error.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Online learning

Updating a model after each new observation instead of training once.

See: [online-learning](techniques/online-learning.md)

### Overfitting

Fitting noise in the sample so well that performance on new data is much worse.

See: [multiple-testing](techniques/multiple-testing.md), [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Pairs trading

Taking opposite positions in two related assets to profit when the gap between them closes.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md)

### PBO

Probability of backtest overfitting: how often the best in-sample rule ranks below the median out of sample.

See: [reality-check-spa-pbo](techniques/reality-check-spa-pbo.md)

### PCA

*Also: principal component*

Principal component analysis: rotating correlated variables into uncorrelated components ordered by variance.

See: [pca-effective-rank](techniques/pca-effective-rank.md)

### Permutation importance

How much error rises when one input is shuffled; a model-agnostic measure of use.

See: [explainability](techniques/explainability.md)

### Perpetual future

*Also: perp*

A futures contract without expiry that stays near spot through funding payments.

See: [crypto-carry](techniques/crypto-carry.md)

### PIT

Probability integral transform: the forecast's cumulative distribution evaluated at the outcome; uniform if the forecast is calibrated.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Platt scaling

Calibrating probabilities by fitting a logistic curve to the raw scores.

See: [calibration](techniques/calibration.md)

### Point in time

Data arranged so that on each date only what was known then is visible, including publication lags and not revisions.

See: [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Posterior predictive

The distribution of a future observation averaged over the posterior uncertainty about parameters.

See: [bayesian-portfolio-construction](techniques/bayesian-portfolio-construction.md)

### Power

The probability a test rejects the null when a specified true effect exists.

See: [power-analysis](techniques/power-analysis.md)

### Pre-registration

Declaring hypotheses, rules and decision criteria before looking at results; here, committed configuration precedes the stage that uses it.

See: [experiment-database-and-reproducibility](techniques/experiment-database-and-reproducibility.md)

### Probabilistic Sharpe ratio

*Also: PSR*

The probability that the true Sharpe exceeds a threshold, given the estimate's uncertainty, skew and kurtosis.

See: [multiple-testing](techniques/multiple-testing.md)

### Reality Check

*Also: SPA*

White's bootstrap test of whether the best of many strategies beats a benchmark after accounting for the search.

See: [reality-check-spa-pbo](techniques/reality-check-spa-pbo.md)

### Regime-switching allocation

A portfolio rule that blends different allocators according to the probability of each regime.

See: [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md)

### Reliability diagram

A plot of observed frequency against forecast probability; the diagonal is perfect calibration.

See: [calibration](techniques/calibration.md)

### Ridge regression

Linear regression with a penalty on the size of coefficients, which stabilises estimates when predictors are noisy or correlated.

See: [machine-learning-for-returns](techniques/machine-learning-for-returns.md)

### Risk parity

Sizing assets so each contributes equally to portfolio risk.

See: [risk-parity](techniques/risk-parity.md)

### RLS

Recursive least squares: an exact online update of a linear regression with forgetting.

See: [online-learning](techniques/online-learning.md)

### Shapley value

*Also: SHAP*

A fair division of a prediction among inputs, averaging each input's marginal contribution over all orders.

See: [explainability](techniques/explainability.md)

### Sharpe ratio

Mean excess return divided by its standard deviation, annualised.

See: [performance-metrics](techniques/performance-metrics.md)

### Shrinkage

Pulling an estimate toward a simpler target to reduce its variance at the price of some bias.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Slippage

The difference between the price you expected and the price you got, beyond spread and impact.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Sortino ratio

Mean excess return divided by downside deviation.

See: [performance-metrics](techniques/performance-metrics.md)

### Spread (bid-ask)

The gap between the best price to buy and the best price to sell.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Stationary

A series whose statistical properties do not change over time; required for a spread to revert.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md)

### Survivorship bias

Testing only on assets that still exist, which overstates past performance.

See: [data-integrity](techniques/data-integrity.md)

### Tear sheet

A one-page summary of a strategy with a rule-based red-flag list.

See: [plugin-framework](techniques/plugin-framework.md)

### Tracking error

The volatility of the difference between a strategy's return and its benchmark's.

See: [paired-sharpe-bootstrap](techniques/paired-sharpe-bootstrap.md), [power-analysis](techniques/power-analysis.md)

### Transformer

*Also: PatchTST*

A neural network built on attention, now standard for sequences.

See: [deep-time-series-models](techniques/deep-time-series-models.md)

### Turnover

The amount traded per year as a multiple of the portfolio; it drives trading cost.

See: [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md)

### Value at risk

*Also: VaR*

A quantile of the loss distribution: the loss exceeded only (1 - confidence) of the time.

See: [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### Volatility targeting

*Also: vol target*

Scaling positions so the portfolio runs at a chosen level of volatility.

See: [risk-parity](techniques/risk-parity.md), [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md)

### Walk-forward

Testing by repeatedly fitting on the past and predicting the next block of the future.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Yield curve

Yields of bonds of different maturities; its slope and shape are classic macro signals and the basis of steepener and butterfly trades.

See: [fixed-income-and-volatility-strategies](techniques/fixed-income-and-volatility-strategies.md), [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Z-score

A value minus its mean, divided by its standard deviation: how many standard deviations from normal.

See: [mean-reversion](techniques/mean-reversion.md)

### Zero-shot

Applying a pre-trained model to a new task without any training on that task's data.

See: [foundation-models](techniques/foundation-models.md)
