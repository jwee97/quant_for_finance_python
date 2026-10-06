# Technique guides

Each guide answers: what is it in one sentence, the idea, why it matters, how this repository uses it, what was found, what goes wrong, and how to run it. Difficulty 1 needs no prior knowledge; 3 assumes the guides it lists as prerequisites.

## Difficulty 1: start here

- [Clean data you can trust](data-integrity.md): Before any model runs, every price is checked, repaired only by declared rules, and nothing is silently deleted.
- [How a backtest turns signals into returns](backtest-engine-and-costs.md): A backtest engine lags the decision, charges trading costs on what actually changed, lets weights drift between rebalances, and reports net returns.
- [Mean reversion and z-scores](mean-reversion.md): When a price is far from its recent average, bet that it moves back, with the distance measured in standard deviations.
- [Momentum and trend following](momentum.md): Assets that have gone up recently tend to keep going up for a while, and the strategies in this family differ only in how they measure 'recently' and 'up'.
- [Reading performance honestly](performance-metrics.md): A single Sharpe ratio hides drawdowns, tails and luck; a fair summary reports several measures and how uncertain each one is.

## Difficulty 2

- [Attribution: where did the return come from?](attribution.md): Attribution splits a return into pieces (allocation, selection, interaction, costs, factors) that add back to the total.
- [Calibration: do 70% forecasts come true 70% of the time?](calibration.md): Calibration repairs a forecaster whose stated probabilities do not match how often events actually happen.
- [Comparing two strategies fairly: the paired bootstrap](paired-sharpe-bootstrap.md): To ask whether strategy A really has a higher Sharpe than strategy B, resample the two return streams together so their correlation is preserved.
- [Cross-sectional factors: low volatility, value, quality, carry, defensive beta](cross-sectional-factors.md): A factor is a characteristic that ranks assets by expected return, and these strategies go long the top of a ranking and short (or underweight) the bottom.
- [Forecasting volatility: EWMA and GARCH](volatility-forecasting.md): Volatility clusters, so tomorrow's risk is best estimated by weighting recent squared returns more than old ones.
- [How many independent bets do I own? (PCA)](pca-effective-rank.md): Principal component analysis rotates correlated returns into uncorrelated components, and the number of components that matter is how many independent risks a portfolio really holds.
- [Risk parity and inverse-volatility weighting](risk-parity.md): Risk parity sizes each asset so that it contributes the same amount of risk, rather than the same amount of money.
- [Running thousands of experiments in parallel](distributed-experiments.md): Because each experiment is independent, a grid of them can run on many cores or machines and must give exactly the answer a serial run gives.
- [The information coefficient and the fundamental law](information-coefficient.md): The information coefficient is the correlation between a signal today and the return that follows, and it is the first, cheapest honest test of whether a signal knows anything.
- [The plugin framework: every strategy is a forecast model](plugin-framework.md): Market data flows through regime detection, forecast models, confidence, combination, allocation, risk and validation, and adding a strategy is writing one forecast model that plugs in.
- [The research database and reproducibility](experiment-database-and-reproducibility.md): Every experiment is stored with its hypothesis, data version, configuration fingerprint and decision, so results can be queried and reproduced.
- [Value at risk, expected shortfall and how to test them](var-cvar-and-backtests.md): VaR is a quantile of the loss distribution, and the only honest way to judge it is to count how often reality breaches it.
- [Walk-forward testing and look-ahead leakage](walk-forward-and-leakage.md): Walk-forward testing refits a model only on the past and scores it on the future in rolling steps, and a leakage test proves the code cannot see tomorrow.

## Difficulty 3

- [Adapting the portfolio to the regime](regime-adaptive-allocation.md): Choose a different allocator, tilt and risk target in each regime, weighted by the regime probabilities, and test it against a placebo.
- [Bayesian portfolio construction](bayesian-portfolio-construction.md): Treat expected returns and covariances as uncertain, and optimise against the whole posterior instead of a single estimate.
- [Causal inference: from correlation to cause](causal-inference.md): Correlation says two things move together; causal methods try to say what would happen to one if you changed the other.
- [Combining forecasts: stacks, trust rules and mixtures](forecast-combination.md): Several weak forecasts can be better than one, if the way they are combined does not itself overfit.
- [Crypto carry: funding, basis and stablecoin flows](crypto-carry.md): Perpetual futures have no expiry, so they pay a funding rate to stay near spot; holding spot and shorting the perpetual collects it with little price risk.
- [Deep learning for time series: transformers, mixers, N-BEATS, N-HiTS](deep-time-series-models.md): Modern neural forecasters read a window of past returns and output a forecast; on financial returns their main risk is learning noise.
- [Diffusion models for scenario generation](diffusion-scenarios.md): A diffusion model learns to turn noise into realistic return vectors, giving unlimited stress-test scenarios with the right fat tails and correlations.
- [Dynamic covariance: DCC-GARCH and orthogonal GARCH](dynamic-covariance.md): Let both volatilities and correlations change through time, with correlations that tend to rise in a crisis.
- [Execution: spread, impact and capacity](execution-and-impact.md): Trading costs are a spread, a market impact that grows with the square root of your size, slippage and commission, and they set how much money a strategy can run.
- [Explaining a model: permutation importance, Shapley values, integrated gradients](explainability.md): Attribution methods say which inputs drove a prediction, and each comes with an identity you can test.
- [Fixed income and volatility strategy families](fixed-income-and-volatility-strategies.md): Bond strategies trade the shape of the yield curve with DV01-neutral legs, and volatility strategies trade the gap between implied and realised volatility.
- [Forecasts as distributions: CRPS and the PIT](probabilistic-forecasting.md): A forecast should say how unsure it is, and the continuous ranked probability score rewards forecasts that are both accurate and honestly spread.
- [Foundation models for time series (zero-shot Chronos)](foundation-models.md): A foundation model is pre-trained on huge numbers of series and can forecast a new series with no training at all.
- [Graph attention across assets](graph-neural-networks.md): Treat each ETF as a node, connect it to its most correlated peers, and let attention decide how much each neighbour's features matter for its forecast.
- [Hierarchical risk parity (HRP) and HERC](hierarchical-risk-parity.md): Cluster the assets by correlation and split risk down the tree, so that similar assets share a budget and no matrix inversion is needed.
- [Machine learning on returns: ridge, forests and honesty](machine-learning-for-returns.md): Flexible models can find nonlinear patterns, and in financial returns the signal is so weak that most of what they find is noise.
- [Macro and alternative data without look-ahead](macro-and-alternative-data.md): Macro series are published late and revised, so using them correctly means using only what was known on each date.
- [Market regimes: HMM, volatility states, change points](regime-detection.md): A regime detector labels each day with the state the market is probably in, and outputs a probability rather than a verdict.
- [Mean-CVaR optimisation: optimising the tail](mean-cvar.md): Instead of minimising variance, choose weights that minimise the average loss in the worst few percent of scenarios.
- [Mean-variance optimisation and why it needs shrinkage](mean-variance-and-shrinkage.md): Mean-variance optimisation picks the portfolio with the best expected return per unit of risk, and amplifies every error in its inputs unless the inputs are shrunk.
- [Multiple testing, false discovery and the deflated Sharpe ratio](multiple-testing.md): If you try enough ideas, the best one will look good even if none works; the remedy is to count the ideas and demand more of the winner.
- [Online learning: updating as data arrives](online-learning.md): Instead of training once and freezing, an online learner nudges its parameters each time a new observation arrives.
- [Pairs, cointegration and statistical arbitrage](pairs-and-statistical-arbitrage.md): Trade the gap between related assets, betting it closes, with a hedge that removes the shared market move.
- [Power: what could the tests have found?](power-analysis.md): Statistical power is the chance a test notices an effect that is really there, and without it a 'not significant' result tells you very little.
- [Reality Check, SPA and probability of backtest overfitting](reality-check-spa-pbo.md): These three tools ask whether the best strategy found by a search is better than luck would produce from that same search.
- [Reinforcement learning for allocation (and why it comes last)](reinforcement-learning.md): Learn an allocation rule by trial and error against a reward; in finance the reward is a noisy backtest, so the main result is usually overfitting.
- [Text as data: FOMC statements and the research assistant](text-features-and-the-assistant.md): Turn documents into structured numbers (tone, change, action) with a method you can audit, and keep language models away from predicting prices.
- [The strategy library as a search, not a menu](strategy-library-search.md): Writing 31 strategies and reporting the best is a search, and the evidence for the best must be judged against the whole search.
- [Uncertainty from neural networks: ensembles and MC dropout](bayesian-deep-learning.md): Train several networks (an ensemble) and sample with dropout switched on; where they disagree, the model is telling you it is unsure.
