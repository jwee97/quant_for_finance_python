# Generation 5: A Research Platform, a Strategy Library, and the Frontier, with Honest Statistics Throughout

### A plugin framework and a newcomer layer around the Generation 1-4 research, thirteen stages of new evidence, each decision rule written down before its result

**Universe and sample:** unchanged (15 ETFs, 2006-01-03 to 2026-09-21); crypto branch Deribit 2021-08 to 2026-10 (four assets, one venue)
**Data version:** `f13af1fed1f7` · **FOMC corpus:** `41cf8d5c35c3` · **Crypto data:** `0d69950db703` · **Framework series:** `b463e499a0ef` · **Fama-French factors:** `data/metadata/factors_manifest.json`
**Config fingerprints:** core `12f9cfd14055`, Generations 1-2 `c25e66aa003a`, Generation 3 `4a4c779b0c8d`, Generation 4 `ea5f48a282e9` (all four unchanged and pinned by a test), Generation 5 scope ``8cad6ee7344a``
**Reproduce with:** `python -m experiments.run_all --fresh` (all 40 stages, about 66 minutes on 4 cores) or `--generation 5` (Stages 30-40)

---

## 1. Executive summary

### 1.1 What was asked, and what was built

The instruction was to handle a long roadmap, restated twice, covering the Generation 2, 3 and 4 items, a list of extra strategies in seven families, and a capability pipeline from market data to an experiment database in which "adding a new strategy is simply writing another forecast model that plugs into the pipeline". Generations 2-4 had already built most of the named capabilities in Gen 1-4, so this generation was organised as **the platform layer plus everything on the roadmap that was still missing**, with a newcomer layer because an earlier question had been what would make this useful as a research tool and as a way to learn the techniques.

| Roadmap item | What exists now | What does not |
|---|---|---|
| The pipeline as capabilities; a strategy is a forecast model | A plugin framework: `Forecast(mean, std, confidence)`, `Regime(name, probability)`, registries for models, regime detectors and allocators, a forecast-calibration step, six combination rules (including regime-conditional trust), allocators (static books, forecast stack, confidence, regime switch), a regime risk policy, a causality test run on every plugin, and one `Pipeline.run` (framework) | Nothing declared missing |
| Experiment manager, database, comparison, dashboard | `quant backtest / run / leaderboard / compare / tearsheet / sql / dashboard / explain / demo / docs`, a persistent experiment database that counts trials for the deflated Sharpe ratio, a tear sheet with fixed red-flag rules, bring-your-own-data checks, a self-contained HTML dashboard | A live dashboard server |
| Additional strategies, seven families | **34 registered models** (31 on the ETF universe, 3 crypto), each with a strategy card; Stage 30 treats them as a search | Dispersion trading (conceptual: no option-level data), cross-exchange crypto spreads (one reachable venue), accounting-based value and quality (only price proxies) |
| Regime-adaptive allocation, risk and signal trust | Stage 31, four declared decisions, with a placebo that shifts the regime path | |
| Transformers, N-BEATS, N-HiTS, TimeMixer, Chronos, GNN, Bayesian deep learning | Stages 33 and 34 (TimeMixer in a simplified multiscale form) | TimesFM, TimeGPT; fine-tuning of foundation models |
| Explainable AI and calibration | Stage 35: permutation importance, exact Shapley values and integrated gradients, with their identities tested on every row; Platt against isotonic | |
| Diffusion models, causal inference, reinforcement learning | Stages 36, 37 and 38 | Causal forests (an R-learner with a ridge effect model instead); IV is validated but not applied (no instrument) |
| Crypto branch (optional) | Stage 32: funding carry, basis, stablecoin flow, on Deribit and DefiLlama data | Binance and Bybit were unreachable; no cross-exchange spreads |
| Make it a useful research tool and let a newcomer understand the techniques | 47 technique guides, a 124-term glossary, a chapter map, 34 generated strategy cards, three executed notebooks, a docs site, `quant explain`, `quant demo`, `quant dashboard`, a worked "add a strategy" example whose code is tested | |
| Two statistical diagnostics the earlier generations lacked | Stage 39 (the power of the platform's own tests) and Stage 40 (factor attribution) | |

**No language model was called to produce any result**, as in Generation 4: there are no API credentials in the build environment. The Chronos model in Stage 34 is a time-series foundation model run locally, not a language model. Docker and the CI workflow were again not run.

Every stage from 30 onward carries a rule written in a committed configuration file before its result existed (Section 13 lists each commit and each deviation).

### 1.2 The answer

**Fourteen decision-bearing hypotheses were declared across Stages 30-38; four were retained, and for each of the four the qualification is the finding.** Stages 34, 35 (the explanation part), 39 and 40 are descriptive by declaration and carry no decision.

| Hypothesis | Decision | Numbers |
|---|---|---|
| Stage 30: after accounting for the search, some library specification beats cash | **Retained** | 42 specifications, best (dual momentum) net Sharpe 0.90; Reality Check p = 0.0015, SPA p = 0.0085; probability of backtest overfitting 0.05 |
| Stage 30: some specification beats passive equal weight after the search | Rejected | Equal weight Sharpe 0.74; Reality Check p = 0.61, SPA p = 0.28 |
| Stage 30: the confidence-weighted combination beats both equal weighting and Gen 1 momentum | Rejected | Confidence 0.34 against equal 0.61 (significantly **worse**, p = 0.014) |
| Stage 31: a regime-switching allocator beats both risk parity and equal weight | Rejected | Sharpe 0.84 against 0.77 and 0.65; p = 0.65 and 0.20 |
| Stage 31: the regime *path* adds value (versus 200 shifted placebo paths) | **Retained** | 0.84 against placebo 90th percentile 0.72 and maximum 0.79; one-sided p = 0.005 |
| Stage 31: regime-dependent volatility targets beat a constant 10% target | Rejected | Sharpe difference +0.0003, p = 0.99 |
| Stage 31: regime-conditional trust in twelve models beats equal trust | **Retained** (fragile) | 0.88 against 0.78, p = 0.0675; one of four declared decisions |
| Stage 32: timing the funding carry, basis or stablecoin supply beats the unconditioned alternative | Rejected | Always-on carry Sharpe 3.31 against timed 2.98; basis reversion significantly worse (p = 0.0005) |
| Stage 33: a deep model beats the **historical mean** (N-BEATS, N-HiTS, TimeMixer-style, graph attention) | Rejected | Best CRPS difference -0.00009 (p = 0.32) |
| Stage 33: ensemble and dropout uncertainty improve the forecast spread | Rejected | Widening +0.2%, CRPS slightly **worse** (p = 0.021, wrong direction) |
| Stage 35: isotonic recalibration beats Platt scaling | Rejected | Log loss +0.0012 worse (p = 0.38); raw probabilities scored best |
| Stage 36: diffusion scenarios give a better 5% VaR than Gaussian and bootstrap | Rejected | Pinball difference +0.00004 (p = 0.75) and -0.00014 (p = 0.49) |
| Stage 37: FOMC tone moves the same-day return of TLT or SPY | **Retained** (fragile) | DML effects -0.16 (TLT) and +0.21 (SPY) per standard deviation, p = 0.054 and 0.050; the declared DML validation **failed** at n = 500 |
| Stage 38: an evolution-strategies allocator beats equal weight | Rejected | Sharpe 0.21 against 0.75, difference -0.54 (significantly **worse**, p = 0.0045) |

### 1.3 Ten findings worth stating plainly

1. **Searching a library is not discovering a strategy.** Of 42 specifications, 38% had a positive net Sharpe and the best, dual momentum at 0.90, survives a search-aware test against cash. It does not beat passive equal weight (0.74), and the effective number of independent ideas is about 21 of 42 (Stage 30).
2. **Combination did not rescue it.** Confidence-weighting the 31 models gave a Sharpe of 0.34 against 0.61 for plain averaging, significantly worse. Weighting schemes are parameters too.
3. **Regime timing is real in the sense of a placebo test and unimpressive in the sense of a benchmark.** The regime path beats 99.5% of shifted placebo paths, yet the allocator it drives does not beat risk parity or equal weight, and regime-dependent volatility targets did nothing at all (Stage 31).
4. **Regime-conditional trust is the one adaptive result that survives, and only just.** Sharpe 0.88 against 0.78 at p = 0.0675, and 44% of a post-hoc sensitivity grid has p below 0.10.
5. **The deep-learning "win" of Generation 4 was the ridge's weakness.** Against the historical mean, none of four modern architectures has a significantly lower CRPS, though all four beat the ridge (Stage 33). Zero-shot Chronos is significantly worse than the historical mean (CRPS +0.0016, p = 0.00002) because it extrapolates a positive drift, while its rank information coefficient (0.08) matches the trained models.
6. **Ensemble disagreement carries no information about error.** Epistemic variance is tiny (the networks agree), the widening is 0.2%, and CRPS gets slightly worse (Stage 33).
7. **Attribution methods agree with themselves and disagree with each other.** Shapley efficiency and integrated-gradients completeness hold on all 202 rows, but the five method-model pairs rank features with Spearman correlations from -0.43 to 0.78, and out-of-sample permutation importance is near zero for most features (Stage 35).
8. **A causal estimator is only as good as its nuisance models, and the platform showed it on itself.** Double machine learning failed its declared validation (bias 0.58 on a true effect of 0.5); an oracle with the true nuisance functions has no bias, and a stronger learner at ten times the data cuts the bias to 0.08. The FOMC application is therefore marginal and fragile (Stage 37).
9. **A reinforcement-learned allocator overfits exactly as the literature warns.** A training-period utility gain of +0.0104 a month becomes -0.0031 out of sample; net Sharpe 0.21 against 0.75, with turnover of 13.6 times a year (Stage 38).
10. **Most of the platform's earlier "rejects" could not have been anything else.** With 15.7 years and a 6% tracking error the paired Sharpe test needs a true difference of about 0.35 for 80% power (0.18 at 3% tracking error); the CRPS test with 187 months needs a true R-squared of about 1%. The Stage 31 allocation comparisons (differences 0.07 and 0.19) were below detectability whatever the truth (Stage 39).

---

## 2. Scope, assumption and method

**Interpretation of the instruction.** The roadmap was received three times in the same words. Items already built in Generations 1-4 were mapped, not rebuilt; every missing capability was built or its absence is listed in Section 14. The numbering of stages follows build order (30 to 40), with the research database (Stage 29) running last because it ingests every stage.

**Method, unchanged:** a decision rule in committed configuration before each result; paired stationary-bootstrap Sharpe comparisons and Diebold-Mariano CRPS tests with Benjamini-Hochberg control inside each declared family; post-hoc analyses labelled and unable to overturn a declared decision; every deviation disclosed (Section 13).

**Multiplicity.** Fourteen decisions across eight stages were declared; no correction is applied across stages. Of the four retained, two are fragile on their face (p = 0.0675 and p about 0.05). Stage 31 declared four decisions and Stage 33 two; the family-wise control is within each stage.

---

## 3. The framework and the strategy library (Stage 30)

The framework (`src/framework/`) makes the pipeline of the roadmap executable: a forecast model turns a market bundle into a score; calibration against matured outcomes turns it into a `Forecast`; a combination rule merges forecasts; an allocator and a risk policy produce weights; the Generation 1 engine backtests them; validation runs the causality test on every plugin, paired benchmark tests and the deflated Sharpe ratio. Thirty-four models, five regime detectors and six allocators are registered and every model passes the causality test (`tests/test_strategies.py`).

**The library as a search.** Forty-two specifications (31 models and their declared variants) were backtested on a common window from 2013-02-05 (3,427 days, the first date every specification has a forecast; the definition was amended after the first run, Section 13).

| | |
|---|---|
| Specifications | 42; 38% with a positive net Sharpe; median -0.05 |
| Best | dual momentum, net Sharpe 0.898; expected best of 42 noise strategies 0.60; deflated Sharpe probability 0.86 |
| Against cash | Reality Check p = 0.0015; SPA p = 0.004 (lower) to 0.0085 (upper): retained |
| Against equal weight (0.736) | Reality Check p = 0.61; SPA p = 0.28: rejected |
| Overfitting | probability of backtest overfitting 0.055; mean in-sample best 0.96, out-of-sample 0.56 |
| Independence | effective number of specifications 20.7; mean pairwise correlation 0.03 |

By family: time-series trend models (median Sharpe 0.40, best 0.90) and macro rules (median 0.26, best 0.50) carry the library; stat-arb, volatility, fixed income and machine learning had median Sharpe at or below zero. Fixed income's best member (carry and roll-down) reached 0.17.

**Combination.** Equal 0.61, IC-weighted 0.64, cost-aware 0.65, confidence-weighted 0.34, against equal weight at 0.74 and Generation 1 momentum at -0.15. The confidence combination is significantly worse than equal (difference -0.27, p = 0.014). The declared hypothesis (confidence beats both equal and momentum) is rejected; the IC-weighted and cost-aware rows are reported, not judged.

Figures 59-61.

---

## 4. The adaptive pipeline (Stage 31)

A composite detector (volatility state, macro inflation/deflation, a crisis rule) feeds four declared decisions (`config/adaptive.yaml`): (1) a probability-weighted switch between allocators (Crisis to mean-CVaR, LowVol to mean-variance, HighVol to risk parity, Inflation and Deflation to risk parity with +10-point tilts), against both risk parity and equal weight; (2) a placebo that shifts the regime path circularly by at least 252 days, 200 draws; (3) probability-weighted volatility targets against a constant 10%; (4) regime-conditional trust in twelve models against equal trust.

| Decision | Result | Verdict |
|---|---|---|
| (1) switch vs risk parity / equal weight | Sharpe 0.84 vs 0.77 / 0.65; differences +0.07, +0.19; p = 0.65, 0.20 | rejected |
| (2) regime path vs placebo | 0.84 against placebo mean 0.62, 90th percentile 0.72, maximum 0.79; p = 0.005 | **retained** |
| (3) regime vs constant vol target | risk parity 0.732 vs 0.732 (p = 0.985); equal weight 0.586 vs 0.586 (p = 0.99) | rejected |
| (4) regime-conditional vs equal trust | 0.881 vs 0.780, p = 0.0675 | **retained** (fragile) |

Reading (1) and (2) together: the soft switch has Sharpe 0.84 and a hard switch 0.88, but a *static* blend of the same books (same average weights, no timing) has 0.69, so the timing is what lifts the books; yet the lift is not large enough to be distinguished from the benchmarks. By the power study (Section 7) the differences were below what the test could detect. For (4), a post-hoc 3 x 3 sensitivity to the two trust parameters has differences from 0.057 to 0.110 with 44% of cells below p = 0.10; it is labelled post-hoc and does not change the decision.

Figures 62-64.

---

## 5. The crypto branch (Stage 32)

Data: Deribit funding, perpetual and index daily series at the 08:00 UTC settlement and DefiLlama stablecoin supply (committed, 3.5 MB, manifest with hashes); Binance and Bybit were unreachable. Four assets; carry asset = funding + spot - perpetual, a linear approximation of an inverse contract.

| | Sharpe |
|---|---|
| Always-on carry | 3.31 (volatility 1.1%, 2021-08 to 2026-10) |
| Funding-carry timing | 2.98 (vs always-on: p = 0.71) |
| Basis reversion | -1.92 (vs always-on: -5.23, p = 0.0005, significantly worse) |
| Stablecoin-flow factor | -0.41 vs buy and hold 0.38 (p = 0.30) |

None of the three declared comparisons passed. A funding-carry Sharpe above 3 with 1% volatility is a statement about a small, venue-specific premium over five years, not about a free lunch: margin, venue risk and the linearisation are not in the number.

Figure 65.

---

## 6. The frontier models (Stages 33-38)

### 6.1 Deep-forecasting family, graph attention and Bayesian deep learning (Stage 33)

N-BEATS (137,007 parameters), N-HiTS (73,056), a TimeMixer-style multiscale mixer (18,681; a simplification, not the paper's architecture) and a graph attention network (5,689; each ETF linked to its four most correlated peers, recomputed at every origin), each trained on the Stage 27 schedule with three seeds, against the asset's **historical mean** (declared primary after Stage 27 showed the ridge was the weak benchmark), same volatility forecast, same 2,805 rows over 187 monthly origins.

| Model | CRPS difference vs historical mean | p | Rank IC | vs Stage 19 ridge |
|---|---:|---:|---:|---:|
| N-BEATS | -0.000088 | 0.32 | 0.083 | -0.0007 (p = 0.003) |
| N-HiTS | -0.000061 | 0.53 | 0.080 | -0.0007 (p = 0.004) |
| TimeMixer-style | -0.000082 | 0.35 | 0.103 | -0.0007 (p = 0.004) |
| Graph attention | +0.000097 | 0.34 | 0.046 | -0.0005 (p = 0.006) |

Nothing passes Benjamini-Hochberg control. All four have a significantly lower CRPS than the ridge, which says again that the ridge was the weak benchmark. Against the Stage 27 patch transformer three of four are significantly better (p = 0.005 to 0.016); against the MLP-mixer none are (p = 0.11 to 0.75).

**Bayesian deep learning.** Three seeds times 30 dropout passes widen the forecast standard deviation by sqrt(1 + epistemic variance of z): the mean ratio is 1.002 to 1.003, the effect on CRPS is +0.000003 (p = 0.021 for N-BEATS, in the wrong direction), and the correlation between epistemic variance and absolute error is -0.08 to +0.03. Interval coverage barely moves (90% interval: 0.889 to 0.890 for N-BEATS). The declared hypothesis is rejected.

Figures 66-67.

### 6.2 A zero-shot foundation model (Stage 34, descriptive)

Chronos-Bolt (small, hub revision pinned in `stage34_model.json`) on the same 252-day windows, no fitting. Declared before the result: the pre-training corpus overlaps the sample, so a good result would not have been clean evidence while a bad one is. It is a bad one: CRPS +0.0016 against the historical mean (p = 0.00002) and +0.0010 against the ridge (p = 0.034). It forecasts a positive drift (79% of forecasts positive, mean absolute forecast about three times the historical mean's), so its scale is wrong even where its ranking has some information (rank IC 0.080).

Figure 68.

### 6.3 Explainability and calibration (Stage 35)

**Explanation (descriptive).** A 16-unit MLP and a gradient-boosted model on the twelve price features, explained on 202 out-of-sample rows (13 per refit; the declared 200 rounded up per refit). Shapley efficiency holds to 1.2e-7 and integrated-gradients completeness to 0.26% of the output range on every row. The five global rankings (permutation and Shapley for each model, integrated gradients for the MLP) have Spearman correlations from -0.43 to 0.78: Shapley and integrated gradients agree for the MLP (0.78), permutation and Shapley agree for the boosted model (0.58), and little else does. Out-of-sample permutation importance is near zero or negative for most features, so the models lean on features that do not help.

**Calibration (declared hypothesis).** From 2016-01-31, monthly refits of Platt and isotonic on matured outcomes (126 origins, 1,890 rows):

| Arm | Log loss | Brier | ECE |
|---|---:|---:|---:|
| Raw | 0.6748 | 0.2408 | 0.024 |
| Platt | 0.6763 | 0.2416 | 0.034 |
| Isotonic | 0.6775 | 0.2423 | 0.034 |
| Stage 19's own calibration | 0.6985 | 0.2525 | 0.101 |

Isotonic minus Platt: +0.0012 (p = 0.38): rejected. Raw probabilities scored best on all three; Stage 19's calibration was the worst, consistent with the Stage 19 finding that its Platt step was fitted on too little data. The clipped-isotonic variant (post-hoc) is identical because no isotonic output reached 0 or 1.

Figures 69-70.

### 6.4 Diffusion-model scenarios (Stage 36)

A 100-step cosine-schedule DDPM (MLP denoiser) on 15-dimensional vectors of volatility-standardised 21-day returns, 2,000 scenarios per origin, equal-weight 5% and 1% value at risk against a multivariate normal and the empirical distribution, over 187 origins.

| 5% VaR, equal weight | Exceedances | Rate | Kupiec p | Mean pinball |
|---|---:|---:|---:|---:|
| Diffusion | 10 | 5.3% | 0.83 | 0.00306 |
| Gaussian | 13 | 7.0% | 0.25 | 0.00302 |
| Bootstrap | 13 | 7.0% | 0.25 | 0.00320 |

Diffusion minus Gaussian pinball +0.00004 (p = 0.75); minus bootstrap -0.00014 (p = 0.49): rejected. At 1% (two expected breaches) Gaussian breaches five times (Kupiec p = 0.057) and diffusion and bootstrap twice; the test has little power. Scenario realism: mean excess kurtosis 2.22 (training 2.15, Gaussian 0.03); correlation-matrix error 0.076 (bootstrap 0.016, Gaussian 0.009); 2.9% of generated vectors outside the training range (the bootstrap, by construction, 0%). The diffusion model reproduces the tails, is slightly worse on correlation, and invents some extremes.

Figures 71-72.

### 6.5 Causal inference (Stage 37)

**Validation on known truth (300 simulated worlds of n = 500, declared):** two-stage least squares recovers a true effect of 0.5 (mean 0.479, coverage 96%) where OLS is biased by 0.85; difference in differences recovers 1.0 under parallel trends (1.002, coverage 96%) and is biased by 0.80 when trends diverge (a designed failure); the R-learner recovers a heterogeneous effect (correlation with the truth 0.99+, slope 0.94 against 1.0). **Double machine learning failed:** bias +0.58 with gradient-boosted nuisance models and +1.10 with ridge (naive OLS: +1.18). Post-hoc diagnostics, labelled as such: with the true nuisance functions the final step has bias -0.002 (the code is right); at n = 5,000 the declared learner still has bias +0.43; a higher-capacity learner at n = 5,000 has +0.075. The failure is the learner's, as theory predicts, and it is reported rather than repaired.

**Application.** For 168 FOMC statements (2006 onward), the effect of a one-standard-deviation increase in tone on the statement day's standardised return:

| | SPY | TLT |
|---|---:|---:|
| Naive OLS | +0.105 (p = 0.32) | -0.145 (p = 0.075) |
| OLS with controls | +0.079 (p = 0.47) | -0.178 (p = 0.032) |
| DML, ridge | +0.082 (p = 0.44) | -0.199 (p = 0.020) |
| DML, gradient-boosted (declared) | +0.206 (p = 0.050) | -0.163 (p = 0.054) |

The declared rule (either asset significant at 10% after Benjamini-Hochberg) retains the hypothesis. The TLT sign (a more hawkish statement lowers bond prices) is stable across all four estimators; the SPY effect is only significant under the learner whose validation failed, and its ridge estimate has p = 0.44. The R-learner's effect by prior-volatility tercile for TLT is -0.13, -0.05 and -0.38. The announcement-day difference in differences (TLT and IEF against gold, absolute standardised return) is -0.096 (p = 0.29): no evidence announcement days differ. IV was not applied: no credible instrument exists in these data.

Figures 73-74.

### 6.6 Reinforcement learning (Stage 38)

A linear softmax policy over the twelve price features and an asset bias (zero parameters is equal weight), trained by evolution strategies on mean-variance utility net of 10 bp costs, annually refitted on matured months, three seeds averaged, through the Generation 1 engine.

| | Net Sharpe | Turnover |
|---|---:|---:|
| RL policy (3-seed average) | 0.213 | 13.6x |
| Seeds 1, 2, 3 | 0.103, 0.347, 0.158 | 13.8x to 14.4x |
| Equal weight | 0.754 | 0.3x |

Paired difference -0.54 (95% interval -0.86 to -0.25, p = 0.0045): significantly **worse** than equal weight, rejected. The training-period utility gain over equal weight is +0.0104 a month and the following year's is -0.0031. The policy tilts about 15 points toward each of QQQ and TLT. Random tilts drawn from the ES perturbation distribution have a mean Sharpe of 0.73 (sd 0.045), so the policy's shortfall is far outside what a random tilt does.

Figures 75-76.

---

## 7. What the platform's tests can detect (Stage 39)

The paired Sharpe test and the CRPS Diebold-Mariano test, run on simulated data with a known edge (the real equal-weight return series, resampled, as benchmark; 200 and 400 replications per cell).

| Paired Sharpe test, 80% power requires a true difference of | 4 years | 8 years | 15.7 years |
|---|---:|---:|---:|
| tracking error 3% | 0.32 | 0.25 | 0.18 |
| tracking error 6% | above 0.60 | 0.45 | 0.35 |

Its false-positive rate is close to nominal (0.10 to 0.125; mean 0.114). The CRPS test with 100, 187 and 400 monthly origins needs a true out-of-sample R-squared of 1.85%, 0.95% and 0.44%. A forecaster that is pure noise (mean forecasts of standard deviation 0.1 against unit-variance outcomes) is found significantly worse 86% of the time at 187 origins, so *harm* is far easier to detect than benefit. The Stage 31 allocation comparisons (observed differences 0.07 and 0.19 at roughly 18.6 years and 7% tracking error, minimum detectable 0.35) could not have been detected if real; the RL shortfall (-0.54) is well beyond it. The CRPS null cell (R-squared 0) is undefined by construction (identical forecasts) and is reported as such.

Figures 77-78.

---

## 8. Factor attribution (Stage 40, descriptive)

The Generation 1 books regressed on the Fama-French five factors and momentum (daily, Newey-West lag 5), declared before the result with the limitation that these are equity factors and the universe is half bonds, gold and commodities.

| Book | Intercept per year (t) | R-squared | Market beta | Momentum beta (t) |
|---|---:|---:|---:|---:|
| M0 equal weight | +0.9% (0.7) | 0.73 | 0.46 | -0.00 |
| M2 risk parity | +1.2% (1.2) | 0.60 | 0.25 | -0.01 |
| M3 momentum | +0.6% (0.5) | 0.11 | -0.03 | +0.12 (11.2) |
| M5 momentum + reversion | -2.3% (-1.7) | 0.10 | 0.08 | +0.12 (10.4) |
| M9 mean-CVaR | +1.2% (1.3) | 0.14 | 0.07 | -0.02 (-2.4) |

The momentum books load on the momentum factor as they should, which is a sanity check on the pipeline as much as a finding; no intercept is significant at 5%. The equal-weight and risk-parity books are largely equity-market exposure scaled down. Figure 79.

---

## 9. The newcomer layer

A research platform is only useful if someone else can learn it. The additions, all tested:

- **`quant explain`** answers terms, techniques, strategies, stages and experiment IDs from files in the repository (no model call), so explanations are checkable and identical everywhere. `quant explain EXP-080` explains a result and what its decision does and does not mean.
- **47 technique guides** (`docs/techniques/`), each with the same sections (in one sentence, the idea, why it matters, how this repository uses it, what was found, formulas, pitfalls, how to run it). A test checks that every file, figure, test and model a guide names exists, that every experiment ID it cites exists *and belongs to the stage the guide claims*, and that every registered model is covered.
- **A 124-term glossary**, **34 generated strategy cards** and a **chapter map** from book chapter to guide, stage, code, figures and tests; the generated files are compared with a fresh build in a test.
- **`quant demo`** (13 seconds): one strategy through the pipeline, its red-flag list, the regimes, and a five-lookback search that ends on the two questions every backtest must answer (is it better than luck; is it better than doing nothing clever). Dual momentum's best lookback has a Sharpe of 0.86 (median 0.61), a deflated Sharpe probability of 0.99, and still does not beat equal weight (difference -0.02, p = 0.91).
- **`quant dashboard`**: a self-contained offline HTML page with the decision ledger, strategy library with parameter sensitivity, your runs, IC decay, walk-forward, risk, figures and the power table; rendered and script-checked in a headless browser in a test.
- **Three executed notebooks** (a first honest backtest, regimes with a placebo, and a zero-skill simulation of how many ideas it takes to look brilliant), re-executed from scratch in a test; a worked "add a strategy" example whose code is extracted from the document and run in a test; a mkdocs site.

---

## 10. Reading the generation together

Three patterns recur across the generation. First, **the benchmark is the finding**: against cash the library wins, against equal weight nothing does; against a ridge the deep models win, against the historical mean none does; against random tilts a reinforcement-learned policy loses badly. Second, **placebos and oracles are more informative than p-values**: the regime-path placebo separates timing from average exposure; the oracle nuisance functions in Stage 37 separate estimator code from learner error. Third, **harm is easier to detect than benefit** (Stage 39), which is why so many comparisons end in a rejection that should be read as "no evidence" and not "no effect".

---

## 11. What Generation 5 establishes, and what it does not

**Established:** the framework and library run and pass the causality test; a search-aware test against cash is passed and against equal weight is not; regime timing beats a placebo; the identities of the attribution methods hold; two-stage least squares, difference in differences and the R-learner recover known truth; the platform's tests have the stated power.

**Not established:** that any strategy here earns money out of sample; that regime-conditional trust is real (p = 0.0675, one of fourteen declared decisions); that the FOMC tone effect is causal (the estimator's validation failed and the sample is 168 statements); anything about language models (none was called) or about live trading (none was done). The universe is 15 ETFs that survive to today; the final holdout was opened in Generation 1.

---

## 12. Cumulative ledger

| Generation | Retained | Rejected | Investigate | Descriptive (record) | Decision-bearing |
|---|---:|---:|---:|---:|---:|
| Generation 1 | 9 | 9 | 5 | 0 | 18 |
| Generation 2 | 3 | 20 | 0 | 12 | 23 |
| Generation 3 | 2 | 7 | 0 | 3 | 9 |
| Generation 4 | 1 | 2 | 0 | 2 | 3 |
| Generation 5 | 4 | 10 | 0 | 5 | 14 |
| **All** | 19 | 48 | 5 | 22 | 67 |

Generation 5 retained: the library beats cash after the search (EXP-075); the regime path beats its placebo (EXP-080); regime-conditional signal trust (EXP-082, p = 0.0675); the FOMC-tone effect (EXP-090, fragile). Generations 1-4 are unchanged from their own reports: a clean run of all 40 stages reproduced all 75 earlier registry decisions with no flip (the differences in the regenerated tables are floating-point noise of order 1e-13 and run-time columns).

---

## 13. Disclosed deviations and post-hoc analyses

Each commit hash is where the rule or the correction is on record; "before any result" means the commit precedes the first run that produced the stage's decision numbers.

| # | What | Commit / status |
|---|---|---|
| 1 | Stage 30 declared (42 specifications, rules) before any result | `d66193b` |
| 2 | Stage 30's common-window definition was amended from "first non-zero position" to "first forecast date" after a first run collapsed the window to 512 days; the amendment is a comment in the YAML and the first-run numbers are kept in the repository history | `439ca0b` |
| 3 | Stage 31 declared before any result | `648e6c1`; its post-hoc signal-trust sensitivity grid is labelled and does not feed the decision |
| 4 | Stage 32 declared; `execution.min_assets: 1` was added before any result because the engine default of 5 zeroed a four-asset book | `dc1ce80`, `d60a2f7` |
| 5 | Stage 33 declared; the key `mc_samples: 30` was added to the YAML as a machine-readable copy of "30 stochastic passes" (value unchanged); scoring is on the 2,805 rows with a realised outcome common to all four models (the last two Stage 19 origins have none) | `ddc002e`, `960f9c5` |
| 6 | Stages 34 and 35 declared before any result. Stage 34's first run scored 30 rows without a realised outcome; corrected before any number was recorded. Stage 35 explained 202 rows (rounding of 200 by refit) and the global-importance heatmap's labels were in the wrong order in the first figure (a display error, corrected; the tables were always right) | `08b2dfd`, `533c60a`, `44fc62d` |
| 7 | Stages 36-38 declared before any result. Stage 36's sampler uses the posterior-mean reverse step with the predicted x0 clipped at 6 standardised units: the plain epsilon form exploded in a unit test, before any stage result | `94daba1` |
| 8 | Stage 37: the declared DML validation failed; three post-hoc diagnostics (true-nuisance oracle, n = 5,000, a stronger learner) are labelled post-hoc; the application's decision follows the declared rule and is reported as fragile | `e157b8c` |
| 9 | Stages 39 and 40 declared (`config/diagnostics.yaml`) before any result; French factors cached with hashes. Stage 39's R-squared-zero CRPS cell is undefined and reported so; a pure-noise-forecaster "harm" cell was added post-hoc | `6704236`, `b9f494c` |
| 10 | Stage numbering: the stages are 30-40 contiguous; an early plan had a gap at 36 | no effect |
| 11 | The tear sheet's red-flag text now says when its "deflated Sharpe probability" is the one-trial probabilistic Sharpe ratio | `0ef49c5` |
| 12 | The clean run of all 40 stages re-timed the Stage 26 grid (207 s on four cores against 241 s before), so the two wall-clock sentences in the Generation 4 report were updated to the new timing; the same run reproduced all 75 earlier registry decisions, and the regenerated tables of Stages 1-29 differ only by floating-point noise of order 1e-13 | this commit |
| 13 | The research database (Stage 29) now runs last of 40, so the registry IDs of Stages 30-40 are one lower than in the build order (for example the library's first hypothesis is EXP-075); the guides' experiment citations were remapped and a test checks every cited ID against its stage | this commit |

---

## 14. Not built

- **Language-model results**: no model was called (no credentials). The Anthropic backend is tested against a fake client only.
- **TimesFM, TimeGPT, fine-tuning of foundation models, causal forests, factor stochastic volatility, dynamic covariance filters beyond DCC and O-GARCH.**
- **Dispersion trading, cross-exchange crypto spreads, accounting-based value and quality, ETF flows, analyst and earnings revisions, CDS spreads**: the data are not available here.
- **Ray backend run** (the adapter exists; Ray is not installed), **Docker and CI runs**.
- **A live dashboard server** (the dashboard is a static page).
- **Binance and Bybit**: unreachable from the build environment; the crypto branch is one venue.

---

## Appendix A: Reproducing this report

```bash
pip install -r requirements.txt
pip install chronos-forecasting einops        # Stage 34 only; the stage skips cleanly without them
python -m experiments.run_all --fresh         # all 40 stages
pytest -q                                     # 621 tests
python -m src docs build                      # regenerate the strategy cards, chapter map and findings digest
```

Tests that pin this report: `tests/test_generation5_report.py` (every number above against the generated tables and registry).

## Appendix B: Figures

Figures 59-79: library search (59), correlation (60), combination (61); regimes (62), adaptive allocation (63), adaptive risk and signal (64); crypto (65); deep family against the historical mean (66), Bayesian deep learning (67); zero-shot foundation model (68); explanation (69), calibration (70); diffusion VaR (71), scenario realism (72); causal validation (73), FOMC application (74); reinforcement learning (75, 76); power (77, 78); factor attribution (79). See `reports/figure_index.md`.
