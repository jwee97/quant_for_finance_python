# Findings digest

Generated from `experiments/registry.jsonl`. A *retain* means the declared rule was met; a *reject* means it was not. Neither is proof.

| decision | count |
|---|---|
| reject | 48 |
| record | 22 |
| retain | 19 |
| investigate | 5 |

## Every decision-bearing hypothesis

| id | stage | decision | hypothesis |
|---|---|---|---|
| EXP-001 | stage01_data | retain | The ETF panel is fit for systematic research after validation and an explicit missing-data policy, with no observation deleted. |
| EXP-002 | stage02_eda | retain | A 15-ETF multi-asset universe contains far fewer independent risk dimensions than it contains assets, and that number falls further in crises. |
| EXP-003 | stage02_eda | retain | Daily ETF returns are not normally distributed, so parametric VaR will understate the tail. |
| EXP-005 | stage03_momentum | reject | vol scaled 126 momentum predicts subsequent 21-day cross-sectional ETF returns. |
| EXP-006 | stage04_mean_reversion | reject | The 21-day price z-score is NEGATIVELY related to subsequent 5-day returns (mean-reversion hypothesis, beta < 0). |
| EXP-008 | stage06_backtest | retain | M3_momentum generates positive risk-adjusted returns after realistic transaction costs. |
| EXP-009 | stage06_backtest | reject | M4_mean_reversion generates positive risk-adjusted returns after realistic transaction costs. |
| EXP-010 | stage06_backtest | reject | M5_momentum_plus_mr generates positive risk-adjusted returns after realistic transaction costs. |
| EXP-012 | stage08_covariance | retain | Which covariance and volatility estimators produce the most useful out-of-sample risk forecasts? (spec §33) |
| EXP-013 | stage09_risk | reject | The VaR model is validated out of sample: breaches arrive at the promised rate and are independent through time. |
| EXP-014 | stage09_risk | retain | Risk-based allocation survives the post-COVID inflation shock better than equity beta does (Ch. 19 closing case study). |
| EXP-015 | stage10_combination | reject | Combining strategy return streams improves risk-adjusted performance beyond the best individual strategy (Ch. 22 §22.5). |
| EXP-016 | stage11_validation | retain | The backtest engine and every production strategy are free of look-ahead bias. |
| EXP-018 | stage11_validation | reject | mean_reversion performance is robust across its parameter family, not the product of one fortunate choice. |
| EXP-019 | stage11_validation | retain | Out-of-sample walk-forward performance justifies the model ladder's added complexity. |
| EXP-020 | stage12_results | retain | Final model comparison: which step of the ladder actually improved risk-adjusted returns after costs? (spec §55-§56) |
| EXP-021 | stage13_ml | reject | Nonlinear machine learning provides genuine out-of-sample economic improvement over the simpler econometric and systematic models (spec §47). |
| EXP-023 | stage14_extensions | reject | Returns unexplained by the first three principal components mean revert, and a factor-neutral book built on them is profitable net of costs (Extension F, Ch. 22 |
| EXP-024 | stage15_macro | reject | Macro features improve out-of-sample forecasts of monthly asset-class excess returns beyond what price features already provide. |
| EXP-026 | stage15_macro | reject | A signal built from price and macro, f(price, macro), beats the price-only signal after transaction costs. |
| EXP-027 | stage16_regimes | retain | A filtered high-volatility HMM state at the close of day t predicts larger absolute equity-sleeve returns on day t+1. |
| EXP-028 | stage16_regimes | reject | A filtered high-volatility HMM state at the close of day t predicts a different mean equity-sleeve return on day t+1. |
| EXP-029 | stage16_regimes | reject | Named rule-based regimes (bear market, high volatility, inflation shock, liquidity crisis) are followed by different mean sleeve returns on the next day. |
| EXP-030 | stage16_regimes | reject | Bayesian online change-point detection flags dated volatility shocks within 60 trading days with few false alarms. |
| EXP-031 | stage16_regimes | reject | De-risking an equal-weight book toward cash as the filtered HMM high-volatility probability rises beats equal weight after costs. |
| EXP-032 | stage16_regimes | reject | Blending equal weight and risk parity by the filtered high-volatility probability beats BOTH constant allocations after costs. |
| EXP-033 | stage16_regimes | reject | Gating the momentum book off in regimes where it lost money on the training sample beats the ungated book after costs. |
| EXP-036 | stage17_dynamic_covariance | reject | DCC-GARCH covariance forecasts have a lower out-of-sample loss than each of the sample, EWMA and shrinkage estimators. |
| EXP-037 | stage17_dynamic_covariance | reject | DCC-GARCH covariance forecasts have a lower loss than each static estimator inside the GFC, COVID and 2022 windows. |
| EXP-038 | stage17_dynamic_covariance | retain | A long-only minimum-variance book built from DCC-GARCH forecasts realises lower volatility than the same book built from the shrinkage estimator. |
| EXP-039 | stage17_dynamic_covariance | reject | Orthogonal GARCH covariance forecasts have a lower out-of-sample loss than each of the sample, EWMA and shrinkage estimators. |
| EXP-040 | stage17_dynamic_covariance | reject | Orthogonal GARCH covariance forecasts have a lower loss than each static estimator inside the GFC, COVID and 2022 windows. |
| EXP-041 | stage17_dynamic_covariance | reject | A long-only minimum-variance book built from Orthogonal GARCH forecasts realises lower volatility than the same book built from the shrinkage estimator. |
| EXP-043 | stage18_hierarchical | reject | HRP and HERC produce more stable weights than minimum-variance optimisation under estimation noise (Lopez de Prado's central claim). |
| EXP-044 | stage18_hierarchical | reject | HRP / HERC deliver a statistically better net Sharpe ratio than risk parity out of sample. |
| EXP-045 | stage19_probabilistic | reject | Platt scaling lowers the expected calibration error of the raw direction classifier. |
| EXP-046 | stage19_probabilistic | reject | The calibrated price-only direction classifier has a lower log loss than the per-asset base rate. |
| EXP-047 | stage19_probabilistic | reject | The price-only Gaussian forecast has a lower CRPS than the same Gaussian centred on the asset's historical mean. |
| EXP-048 | stage19_probabilistic | reject | Adding macro and regime features improves the probabilistic forecasts (log loss or CRPS) over price features alone. |
| EXP-049 | stage19_probabilistic | reject | Sizing positions by the size of the calibrated edge beats direction-only sizing of the same signal, net of costs. |
| EXP-050 | stage19_probabilistic | retain | Fractional-Kelly sizing from the Gaussian forecast beats direction-only sizing of the classifier signal, net of costs. |
| EXP-059 | stage21_bayesian | reject | Bayesian portfolio construction improves the allocation: the posterior-predictive book beats the same inputs treated as certain AND the Generation 1 risk-parity |
| EXP-060 | stage21_bayesian | retain | Posterior-averaged weights are more stable than plug-in mean-variance weights under estimation noise. |
| EXP-062 | stage22_online | reject | Models that update every month forecast 21-day ETF returns better (lower CRPS) than the annually refitted ridge. |
| EXP-063 | stage22_online | reject | A Hedge aggregate of the strategy sleeves has a higher net Sharpe than the Stage 10 inverse-volatility blend. |
| EXP-064 | stage23_altdata | reject | Non-price data (credit spread, implied-volatility structure, CFTC positioning, jobless claims) add out-of-sample forecasting information to price and macro feat |
| EXP-065 | stage23_altdata | reject | Speculative positioning (CFTC) predicts the next month's excess return of the matching ETF. |
| EXP-066 | stage24_combination | reject | A cost-aware rule for how much to trust each alpha, applied to the combined forecast, beats BOTH equal weighting of the alphas and the Generation 1 momentum boo |
| EXP-068 | stage25_execution | retain | At $1bn of assets the impact-inclusive net Sharpe of every allocator (M0, M1, M2, M9, M11, M12) lies within 0.05 of its Generation 1 linear-cost net Sharpe. |
| EXP-069 | stage25_execution | reject | A 1% no-trade band improves the impact-inclusive net Sharpe at $1bn of EACH of the momentum, mean-reversion and combined books. |
| EXP-071 | stage26_distributed | reject | Across a grid of 1,584 simple rules, at least one has a positive expected net return after accounting for the search: both White's Reality Check and Hansen's SP |
| EXP-073 | stage27_deep | retain | A patch transformer or an MLP-mixer forecasts 21-day ETF returns with a significantly lower mean CRPS than the Stage 19 annually refitted price-only ridge (same |
| EXP-074 | stage28_text | reject | Tone, change and the announced rate action of FOMC statements (offline lexicon features) add out-of-sample forecasting information to price and macro features f |
| EXP-076 | stage30_library | retain | Across the 42 specifications of the strategy library, at least one has a positive expected net return after accounting for the search: both White's Reality Chec |
| EXP-077 | stage30_library | reject | Across the 42 specifications, at least one beats passive equal weight after accounting for the search (Reality Check and SPA on differences of daily net returns |
| EXP-078 | stage30_library | reject | The confidence-weighted combination of all 31 library models has a higher net Sharpe than both the equal-weight combination and the Generation 1 momentum specif |
| EXP-080 | stage31_adaptive | reject | A regime-switching allocator (Crisis: mean-CVaR, LowVol: MVO, Inflation: commodity tilt, otherwise risk parity) has a higher net Sharpe than both static risk pa |
| EXP-081 | stage31_adaptive | retain | The regime path adds value: the switching allocator's net Sharpe exceeds the 90th percentile of 200 circular-shift placebos. |
| EXP-082 | stage31_adaptive | reject | Regime-dependent volatility targets give a higher net Sharpe than a constant 10% target for both risk parity and equal weight. |
| EXP-083 | stage31_adaptive | retain | Combining twelve models with regime-conditional trust weights has a higher net Sharpe than their unconditional equal combination. |
| EXP-084 | stage32_crypto | reject | Timing the funding carry, trading the basis, or following stablecoin supply growth beats the unconditioned alternative (always-on carry, buy and hold) after Ben |
| EXP-085 | stage33_frontier | reject | At least one of N-BEATS, N-HiTS, a TimeMixer-style mixer or a graph attention network forecasts 21-day ETF returns with a significantly lower mean CRPS than the |
| EXP-086 | stage33_frontier | reject | For the best model (nbeats), widening the forecast standard deviation by the deep-ensemble and MC-dropout epistemic variance lowers the mean CRPS. |
| EXP-089 | stage35_explain | reject | Isotonic recalibration of the Stage 19 monthly up-probability has a lower mean log loss than Platt scaling (both refitted on matured outcomes, 2016 onward). |
| EXP-090 | stage36_diffusion | reject | The diffusion-model 5% value-at-risk of the equal-weight 21-day return has a lower mean pinball loss than both the Gaussian and the bootstrap quantile (Benjamin |
| EXP-091 | stage37_causal | retain | A more hawkish FOMC statement (a standardised increase in the Stage 28 tone score) changes the same-day standardised return of TLT or SPY (double machine learni |
| EXP-092 | stage38_rl | reject | A policy learned by evolution strategies (linear softmax over price features, mean-variance utility net of costs, annual refits) has a higher net Sharpe than eq |
