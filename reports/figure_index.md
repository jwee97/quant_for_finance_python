# Figure index

58 figures. Each one answers a stated research question; a figure without a question is not included.

| # | File | Research question |
|---|------|-------------------|
| 1 | `fig01_data_availability.png` | On which dates is each asset actually investable, and how much common history do we have? |
| 2 | `fig02_cumulative_returns.png` | How differently have the asset classes compounded, and is any single sleeve dominant enough to make diversification pointless? |
| 3 | `fig03_return_distributions.png` | Are daily returns normal? The answer determines whether parametric VaR (Ch. 21) is defensible on this universe. |
| 4 | `fig04_rolling_volatility.png` | How much does volatility move through time, and does it move together across asset classes? (Motivates vol targeting and time-varying covariance.) |
| 5 | `fig05_correlation_matrix.png` | Is the diversification visible in the full-sample correlation matrix still there when it is needed? |
| 6 | `fig06_rolling_correlation.png` | Does the equity-bond hedge hold through time, and does average correlation spike exactly when diversification is needed? |
| 7 | `fig07_pca_explained_variance.png` | If we own 15 ETFs, how many genuinely independent statistical risk dimensions do we actually possess, and is that number stable? |
| 8 | `fig08_momentum_signal.png` | Is the momentum signal well behaved, and are different lookbacks measuring different things or the same thing? |
| 9 | `fig09_momentum_ic.png` | Does momentum have a positive and statistically reliable information coefficient on this universe, and is it stable through time? |
| 10 | `fig10_ic_decay.png` | How quickly does momentum's predictive information decay, and does any combination of lookback and horizon show a genuine edge? |
| 11 | `fig11_mean_reversion_signal.png` | Do prices actually mean revert on this universe, and over what horizon, before any trading rule is built on top? |
| 12 | `fig12_mean_reversion_ic.png` | Is the relationship between the z-score and subsequent returns negative, as the mean-reversion hypothesis requires, and is it statistically reliable? |
| 13 | `fig13_signal_inference.png` | Once overlapping observations are corrected for, how much statistical evidence remains, and what information ratio does the Fundamental Law imply at each horizon? |
| 14 | `fig14_gross_vs_net.png` | How much of each strategy's gross performance survives realistic transaction costs, and how much turnover is it paying for? |
| 15 | `fig15_cost_sensitivity.png` | At what level of transaction costs does each strategy stop working, and is that level plausible for these instruments? |
| 16 | `fig16_portfolio_weights.png` | What does each allocation method actually hold, and is the book stable or does it churn? |
| 17 | `fig17_risk_contributions.png` | Does risk parity actually equalise risk contributions, and does holding 15 assets deliver 15 assets' worth of diversification? |
| 18 | `fig18_covariance_evaluation.png` | Which covariance method produces the most stable and useful out-of-sample risk forecasts, and what does shrinkage actually buy? |
| 19 | `fig19_var_backtest.png` | Does the VaR model produce the right number of breaches, and are those breaches independent through time? |
| 20 | `fig20_crisis_performance.png` | How does each book behave in the regimes that matter, and does diversification survive when correlations rise? |
| 21 | `fig21_strategy_combination.png` | Does combining weakly performing but uncorrelated strategies produce a better return stream than any of them alone? |
| 22 | `fig22_parameter_sensitivity.png` | Does performance hold across the whole parameter family, or does it depend on one narrow choice that would signal parameter mining? |
| 23 | `fig23_is_vs_oos.png` | How much performance is lost moving from in-sample to walk-forward out-of-sample, and is any of it distinguishable from zero? |
| 24 | `fig24_machine_learning.png` | Does nonlinear machine learning deliver genuine out-of-sample economic improvement over the simpler models, or only a better classification score? |
| 25 | `fig25_final_comparison.png` | Which model ladder step actually improved risk-adjusted returns after costs, and where did the improvement come from? |
| 26 | `fig26_pairs_trading.png` | Are any of these pairs genuinely cointegrated on training data, and does the spread revert fast enough to be tradable? |
| 27 | `fig27_pca_stat_arb.png` | Do returns unexplained by the first three principal components mean revert, and is a factor-neutral book built on them profitable net of costs? |
| 28 | `fig28_macro_panel.png` | Can macro series be used without look-ahead, how stale is the information on a typical trading day, and how many independent dimensions do the features really carry? |
| 29 | `fig29_macro_predictability.png` | Do macro features forecast asset-class returns beyond what price features already do, how much would ignoring publication lags have flattered the answer, and does it survive costs? |
| 30 | `fig30_regimes_timeline.png` | What do the regime models say through time, how different is the tradable (filtered) answer from the hindsight (smoothed) one, and how long do the regimes last? |
| 31 | `fig31_regime_conditional.png` | Do regimes differ in risk and diversification, does any regime state predict the NEXT day once persistence and variance are respected, and why does the test need to be built this way? |
| 32 | `fig32_changepoints.png` | Does Bayesian online change-point detection flag dated volatility shocks quickly, at what false-alarm cost, and does it beat a one-line volatility-jump rule? |
| 33 | `fig33_regime_strategies.png` | Do regime-aware rules beat static allocations after costs, and how much would a backtest on hindsight probabilities have overstated? |
| 34 | `fig34_dynamic_covariance.png` | How do the dynamic covariance forecasts differ from the static ones, what do the models learn, and when does the extra flexibility pay? |
| 35 | `fig35_dynamic_covariance_payoff.png` | Do the dynamic forecasts beat the static ones on loss, in crises, and as the input to a minimum-variance book? |
| 36 | `fig36_hierarchical_anatomy.png` | What structure does hierarchical clustering find in a 15-ETF universe, and how does that change what is held? |
| 37 | `fig37_hierarchical_results.png` | Do HRP and HERC produce more stable weights than minimum variance under estimation noise, and does that translate into a statistically better risk-adjusted return than risk parity? |
| 38 | `fig38_probabilistic_calibration.png` | Are the probability forecasts calibrated, do they beat benchmarks that carry no feature information, and is the Gaussian forecast's spread right? |
| 39 | `fig39_confidence_sizing.png` | Does sizing positions by calibrated confidence or fractional Kelly beat a direction-only book, and is a bigger edge worth a bigger bet? |
| 40 | `fig40_brinson_attribution.png` | Is the difference between risk-based allocators and equal weight explained by asset-class allocation or by selection within classes, and which bets paid? |
| 41 | `fig41_risk_cost_sleeves.png` | Which asset classes carry the risk, the return and the transaction cost of each allocator, and which sleeve supplied the combined strategy's return? |
| 42 | `fig42_bayesian_weights.png` | How uncertain is the optimal allocation once parameter uncertainty is acknowledged, and are posterior-averaged weights more stable than plug-in ones under estimation noise? |
| 43 | `fig43_bayesian_results.png` | Does treating mean and covariance as uncertain earn a better net Sharpe than treating them as known or than risk parity, and does the posterior predictive distribution cover realised returns at its nominal rates? |
| 44 | `fig44_online_forecasting.png` | Do online-updating models (recursive least squares, normalised LMS, a Kalman filter) forecast 21-day ETF returns better than the annually refitted ridge, with the same features and volatility? |
| 45 | `fig45_online_aggregation.png` | Does an online expert-aggregation rule over the strategy sleeves beat the inverse-volatility blend, and how large is its regret against the best sleeve in hindsight? |
| 46 | `fig46_nonprice_panel.png` | Can credit spreads, options-implied volatility, CFTC positioning and jobless claims be used without look-ahead, and how fresh is each on a typical trading day? |
| 47 | `fig47_nonprice_results.png` | Do non-price features add out-of-sample forecasting information to price and macro features, and does speculative positioning predict the matching ETF's next-month return? |
| 48 | `fig48_alpha_descriptors.png` | For each alpha, where does its information live (decay), what does its trading cost, and how correlated are the alphas as forecasts and as books? |
| 49 | `fig49_alpha_combination.png` | Does a cost-aware rule for how much to trust each alpha, applied to the combined forecast, beat equal weighting and the best Generation 1 alpha net of costs? |
| 50 | `fig50_capacity.png` | Once market impact is charged on the fund's own size, how fast does each book's net Sharpe decay with assets under management, and at what size does it lose half of its linear-cost Sharpe? |
| 51 | `fig51_bands_scheduling.png` | Does holding positions inside a no-trade band save more in impact and spread than it costs in tracking, and how much does spreading a trade over several days trade cost for timing risk? |
| 52 | `fig52_search.png` | Across a grid of 1,584 simple rules, how does net Sharpe distribute, how much of the best is what a search of that size produces from noise, and does any rule have a positive expected net return after White's Reality Check and Hansen's SPA? |
| 53 | `fig53_overfitting.png` | How often would the best-in-sample rule from this grid land in the bottom half out of sample, how much does its Sharpe degrade, and which parts of the grid are on average above or below zero? |
| 54 | `fig54_deep_scores.png` | Do a patch transformer and an MLP-mixer forecast 21-day ETF returns better than the annually refitted price-only ridge, on the same rows and with the same volatility forecast? |
| 55 | `fig55_deep_training.png` | How well do the deep models fit, how much of their skill is seed noise, and how big are they next to the 252-feature linear control? |
| 56 | `fig56_text_corpus.png` | What does the FOMC statement corpus look like through the offline lexicon: how many statements, what tone, and how much does each statement rewrite the last? |
| 57 | `fig57_text_results.png` | Does tone, change or the announced rate action in FOMC statements add out-of-sample forecasting information to price and macro features for the five sleeves, and does the answer depend on how the restricted model is trained? |
| 58 | `fig58_research_database.png` | What does the research database hold, how many hypotheses did each generation declare and with what outcome, and how many result tables does each stage contribute? |
