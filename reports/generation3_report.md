# Generation 3: Bayesian Portfolios, Online Learning, Non-Price Data, an Alpha Engine, and an Execution Model

### Five extensions to the Generation 1-2 platform, each held to a rule written down before its result existed

**Universe and sample:** unchanged (15 ETFs, 2006-01-03 to 2026-09-21)
**Data version:** `f13af1fed1f7` · **Macro:** `0cb282ac19f2` · **Non-price series:** `d416cae26f99` · **CFTC positioning:** `b2ec476d7703`
**Config fingerprints:** core `12f9cfd14055` (unchanged since Generation 1), all Generation 1-2 namespaces `c25e66aa003a` (unchanged), Generation 3 scope `4a4c779b0c8d`
**Reproduce with:** `python -m experiments.run_all --fresh` (all 25 stages, about 60 minutes; Stage 21 takes about 21 and Stage 13 about 14 of them) or `--generation 3`

---

## 1. Executive summary

### 1.1 What was asked, and what was built

The instruction was "Continue generation 3". Generation 3 is the third tier of the advisory roadmap (twenty priorities, four generations)
and names five items: Bayesian portfolio optimisation (Priority 3), online learning (10), alternative data (6), an alpha-combination
engine (13) and an execution and transaction-cost model (7). All five were built as Stages 21-25. Generation 4 was **not** built (Section 12).

Each stage was run the way Generations 1 and 2 were: the question and the decision rule are in a YAML file committed before the stage's
results existed (Stage 21 `e3b5e04`, Stage 22 `d3d4261`, Stage 24 `6fcebcc`, Stage 25 `b80c63c`; Stage 23's rule is in `e51b041`, which
also holds the Stage 22 code and precedes the Stage 23 code and results). Tests are paired or nested-model-aware, families are
Benjamini-Hochberg controlled at 10%, and every causal question is answered by a truncation or identity test rather than by inspection.

### 1.2 The answer

**Almost nothing new survived, again.** Of **9** decision-bearing hypotheses declared in advance, **two** were retained and seven rejected. Both retained results
come with a qualification:

| Retained | What it is | Why it is not a discovery |
|---|---|---|
| Posterior-averaged weights are more stable than plug-in mean-variance weights (94% of windows; dispersion 0.57 against 0.87) | The Bayesian book moves less | Averaging 100 optimisation results is *mechanically* smoother than one of them. The Bayes-Stein book, which uses a shrunk mean but a single optimum, is **not** more stable (0.88). Stability did not buy Sharpe (Section 4) |
| At $1bn of assets every allocator's net Sharpe stays within 0.05 of its linear-cost value (largest change −0.042) | Impact is immaterial for the low-turnover allocators | A bound, not a significance test; the margin is thin (mean-CVaR −0.042 against a 0.05 limit) and at the doubled impact coefficient mean-CVaR **breaks** it (−0.086). It says nothing about the alpha books, which do not survive size at all (Section 8) |

### 1.3 Seven findings worth stating plainly

1. **Bayesian estimation did not improve the allocation.** The posterior-predictive book has a net Sharpe of 0.613 against 0.606 for the same inputs treated as certain (+0.006, p = 0.94) and
   0.811 for plain risk parity (−0.198, p = 0.065). The posterior's own calibration is decent (52.8% of realised monthly returns inside the 50% interval, 87.2% inside the 90%).

2. **Learning online made forecasts worse, not better.** Of three recursive learners (RLS with forgetting, NLMS, a Kalman filter) against the annually refitted ridge, RLS is indistinguishable
   (CRPS difference −0.0001, p = 0.30) and the other two are significantly **worse** (+0.0022 and +0.0013, p < 0.001): updating monthly on 187 origins mostly adds noise.

3. **Hedge aggregation of the strategy sleeves is not better than a simple blend.** Net Sharpe 0.857 against 0.769 for the Stage 10 inverse-volatility blend (+0.088, p = 0.30), with regret to the best
   single sleeve of 23.7 against a theoretical envelope of 129.3, so the algorithm behaved as its theory says; there was just nothing to learn.

4. **Non-price data add no out-of-sample information.** Credit spread, implied-volatility structure (VIX3M, VXN, GVZ, OVX), jobless claims and CFTC positioning added to price and macro features give a mean out-of-sample R² of −1.1% over 125 months
   and no sleeve is significant (smallest p = 0.137). CFTC positioning predicts none of six matching ETFs (smallest p = 0.36).

5. **Combining forecasts rather than books did not rescue the alphas.** The cost-aware combination (net Sharpe 0.181) is not better than equal weight (0.067; +0.115, p = 0.46) and is *below* the Generation 1 momentum book alone
   (0.398; −0.217, p = 0.25) on the common window from 2011. The alpha with the highest information coefficient (PCA residual, 0.022) has a net Sharpe of −0.32 and turns over 19.8 times a year.

6. **The execution model separates the two halves of the platform sharply.** The six allocators lose 0.006 to 0.042 of Sharpe at $1bn and have capacity beyond $100bn. The momentum book halves its Sharpe at $102m of capital and has none left by roughly $0.6bn; mean reversion is loss-making before any impact, the combined book turns negative by $10m, and no alpha combination has a capacity above about $1.2m (two are already loss-making or below the grid).

7. **A 1% no-trade band does not help.** Banded minus unbanded net Sharpe at $1bn: +0.011 (momentum, p = 0.12), −0.004 (mean reversion, p = 0.34), +0.009 (combined, p = 0.054). The band cuts annual turnover by only 0.4% to 2% (the median M3 trade is 2.8% of capital, well outside the band), so there is little to save.

---

## 2. Scope, assumption and method

**Assumption, stated once.** "Continue generation 3" was taken to mean the five items the roadmap lists for Generation 3 and nothing else. Where a roadmap item named data that could not be sourced for free it was left out and said so (Section 6), not replaced by a proxy under the same name.

**Pre-registration.** Design and decision rules are in `config/bayes.yaml`, `online.yaml`, `altdata.yaml`, `combination.yaml` and `execution.yaml`. Post-hoc diagnostics carry `posthoc` in the table name and the registry text and never overturn a declared decision. Every change made after a first look is in Section 11.

**Identity of a result.** `fingerprint("core")` and `fingerprint("all")` are pinned by a test and did not change; the new `"gen3"` scope covers all 21 configuration namespaces. The Bayesian book cache is keyed by the core configuration plus the `bayes` namespace only, so editing Stage 22-25 settings does not invalidate a 20-minute computation.

**Timing.** Unchanged: `R_t = w_{t-1}' r_t`, signals lagged one day, monthly rebalance, weights drift. New for Generation 3: non-price series enter only after their availability lag (the CFTC reports are known four calendar days after their Tuesday date, so usable from the following Monday); online learners are updated only with labels that have matured; trailing information coefficients are shifted by the forecast horizon so each is known when it is used; the impact model's volatility and volume are both measured through the previous close.

---

## 3. Bayesian portfolio construction (Stage 21)

**Method.** A conjugate normal-inverse-Wishart posterior over daily mean and covariance (prior strength 126 days) with a Jorion Bayes-Stein prior mean (the common mean is that of the minimum-variance portfolio, its strength set by empirical Bayes: 0.61 on the latest window) and a constant-correlation covariance target. The weights are a *distribution*: one constrained mean-variance optimum per posterior draw (100 draws), and the traded book is the posterior mean of those weights. Controls: the plug-in sample mean-variance book and a Bayes-Stein book that optimises once.

**Declared rules.** `h_bayes`: the posterior-predictive book beats **both** the plug-in book and Generation 1 risk parity on net Sharpe (paired stationary bootstrap, a family of four, BH at 10%). `h_stability`: retained if the posterior book's weights are more stable than the plug-in's in at least 75% of rolling windows (60 block-bootstrap resamples, 20 draws each).

| Net Sharpe, 2007-03 onward | Sharpe | Annual turnover |
|---|---:|---:|
| Bayes predictive (posterior-mean weights) | 0.613 | 2.9x |
| Bayes-Stein (single optimum) | 0.566 | 4.7x |
| Plug-in sample mean-variance | 0.606 | 4.6x |
| Risk parity (M2) | 0.811 | 0.4x |

**Outcome.** `h_bayes` **rejected**: +0.006 against plug-in (p = 0.94), −0.198 against risk parity (p = 0.065). The Bayes-Stein variant is −0.041 against plug-in (p = 0.35) and −0.245 against risk parity (p = 0.16). `h_stability` **retained**: 94% of windows, mean pairwise weight distance 0.57 against 0.87, effective number of positions 8.8 against 4.5, mean maximum weight 0.19 against 0.25 (the cap). As the table in Section 1.2 says, this is averaging, not information: the Bayes-Stein book is as unstable as the plug-in (0.88).

**Posterior-predictive check.** Over 235 months the 50% predictive interval for next month's book return holds 52.8% of realisations and the 90% interval 87.2%. Daily Gaussian returns scaled to 21 days understate fat tails and clustering, so the slight under-coverage at 90% is in the expected direction. Descriptive, not judged.

*Figures 42-43.*

---

## 4. Online learning (Stage 22)

**Method.** Recursive least squares with exponential forgetting (sufficient statistics, ridge floor), normalised LMS, and a random-walk Kalman filter, updated each month only on labels that have matured and with scaling fixed from the warm-start block, forecasting 21-day ETF returns from the same features, volatility forecast and origins as the Stage 19 annually refitted ridge. Separately, the Hedge algorithm (learning rate √(8 ln N / T)) aggregates the strategy sleeves' net returns, with regret to the best sleeve measured against its envelope.

**Declared rules.** `h_online_forecast`: retained if at least one online model has a significantly lower CRPS than the frozen ridge after BH control. `h_online_blend`: Hedge has a higher net Sharpe than the Stage 10 inverse-volatility blend (paired bootstrap).

**Outcome.** Both **rejected**.

| Model (187 monthly origins from 2011-01) | CRPS | Difference to frozen ridge | p |
|---|---:|---:|---:|
| Frozen ridge (Stage 19) | 0.0230 | | |
| RLS with forgetting | 0.0229 | −0.0001 | 0.30 |
| NLMS | 0.0252 | +0.0022 | < 0.001 (worse) |
| Kalman filter | 0.0243 | +0.0013 | 0.0006 (worse) |

The two significant results are in the wrong direction and do not count. Hedge: Sharpe 0.857, inverse-volatility blend 0.769, equal blend 0.783; difference +0.088, p = 0.30; final regret 23.7 inside the 129.3 envelope.

*Figures 44-45.*

---

## 5. Alternative and non-price data (Stage 23)

**What was sourced** (free, point-in-time by construction, manifests with hashes committed): the Baa-minus-10-year credit spread (FRED `BAA10Y`, one-day lag), initial jobless claims (`ICSA`, six-day lag), the implied-volatility structure (VIX3M, and VXN, GVZ, OVX for Nasdaq, gold and crude oil; Yahoo, same-day close), and CFTC positioning for six contracts (E-mini S&P and Nasdaq, 10-year Treasury, gold, silver, crude oil) from the Traders in Financial Futures (leveraged funds) and Disaggregated (managed money) reports, 2006 to 2026. Features are expanding z-scores with a 756-day burn-in, which is what shortens the usable sample to 125 monthly origins from 2010.

**Declared rules.** `h_nonprice_nested`: for each of the five sleeves a nested Clark-West test of price + macro + non-price against price + macro; retained if at least one sleeve passes at FDR 10% **and** the mean out-of-sample R² is positive. `h_positioning`: CFTC speculative positioning predicts the next month's excess return of the matching ETF (two-sided, six contracts, BH at 10%).

**Outcome.** Both **rejected**. No sleeve is significant (smallest p = 0.137, equities); mean out-of-sample R² −1.1%. Positioning: smallest p = 0.36 of six.

**Post-hoc controls** (labelled, not judged). A *matched-sample* control trains the restricted model on exactly the same rows as the unrestricted one; the mean R² turns **positive** (+1.4%) and still no sleeve is BH-significant, so the sign of the headline R² depends on how much history the restricted model is allowed. A control that ignores publication lags gives −1.7% and is also not significant. Of 84 feature-by-ETF rank correlations, 8 are significant at 5% raw (about 4 expected by chance) and none after BH.

**Power.** 125 monthly origins is a short sample for a many-feature nested test; this is weak evidence of absence, as Stage 15's was.

*Figures 46-47.*

---

## 6. What the data scope leaves out

The roadmap's alternative-data list also names ETF fund flows, earnings and analyst revisions, creation and redemption data, and options flow. None of these is available free at the history and point-in-time quality this platform requires, and none is **proxied** here: the stage says "non-price data" because that is what it holds, not "alternative data" in the sense the industry uses the phrase. The credit-spread series is `BAA10Y` rather than an ICE BofA option-adjusted spread, whose free history is limited to the most recent years.

---

## 7. The alpha-combination engine (Stage 24)

**Method.** Four alphas (momentum, mean reversion, PCA residual, the Stage 19 ridge forecast) are winsorised, cross-sectionally demeaned, z-scored and clipped, then combined **at the forecast level**, so that opposing trades net before costs. Each alpha carries the five descriptors the roadmap names: expected return (rank IC at 21 days), confidence (trailing information ratio of the matured IC), turnover, IC decay (1, 5, 10, 21, 63 days) and correlation (signals and books). Four trust rules: equal; IC-IR-weighted (positive part, trailing 504 days of *matured* ICs); cost-aware (positive part of trailing 504-day net Sharpe of each alpha's standalone book, shrunk 50% toward equal, equal if none is positive); Hedge.

**Declared rule.** `h_engine`: the cost-aware combination has a higher net Sharpe than **both** equal weighting and the Generation 1 momentum book (declared in advance as the best Generation 1 alpha), paired bootstrap, a family of two, BH at 10%.

| Net, 2011-01-31 onward | Sharpe | Annual turnover | Deflated Sharpe probability |
|---|---:|---:|---:|
| Momentum | 0.398 | 9.2x | 0.55 |
| Mean reversion | −0.144 | 22.4x | 0.02 |
| PCA residual | −0.322 | 19.8x | 0.003 |
| Ridge forecast | −0.313 | 14.7x | 0.004 |
| Combination: equal | 0.067 | 19.6x | 0.12 |
| Combination: IC-weighted | −0.086 | 18.6x | 0.04 |
| **Combination: cost-aware** | **0.181** | 15.7x | 0.23 |
| Combination: Hedge | 0.224 | 13.6x | 0.28 |

**Outcome.** **Rejected.** Cost-aware against equal: +0.115 (p = 0.46); against momentum: −0.217 (p = 0.25). The Hedge combination has the highest Sharpe of the four combinations, but it was not the declared candidate. The deflated Sharpe probabilities, with eight trials in the stage, say that only momentum is plausibly distinguishable from zero after selection. The descriptors show why: the alphas are weak (mean 21-day IC 0.013 to 0.022), only weakly correlated with one another (largest 0.36, reversion with the PCA residual; momentum is −0.29 against reversion), so there is diversification but little to diversify, and the alphas with the best ICs turn over the most.

*Figures 48-49.*

---

## 8. The execution model (Stage 25)

**Method.** The cost per dollar of capital traded in asset *i* on day *t* is the Generation 1 per-asset linear rate plus square-root market impact: `Y · σ_i,t · sqrt(|Δw_i,t| · AUM / ADV_i,t)`, with σ the 21-day daily volatility and ADV the 63-day average dollar volume, both through the previous close, and `Y = 1`. As AUM → 0 the model **is** Generation 1; this is asserted to 1e-12 on every book in the run and pinned by unit tests. Participation above 100% of ADV is capped in the formula and flagged. Capacity is the AUM at which a book's net Sharpe falls to half its linear-cost value. Y is an order-one coefficient from the square-root law, not an estimate from these ETFs, and Y = 0.5 and 2 are reported.

**Declared rules.** `h_capacity`: at $1bn the net Sharpe of every allocator (M0, M1, M2, M9, M11, M12) lies within 0.05 of its linear-cost value. `h_bands`: a 1% no-trade band improves the net Sharpe at $1bn of **each** of the three alpha books (paired bootstrap, BH across three).

**Net Sharpe by assets under management** (0 is Generation 1):

| Book | 0 | $10m | $100m | $1bn | $10bn | Capacity |
|---|---:|---:|---:|---:|---:|---|
| M0 equal weight | 0.686 | 0.685 | 0.684 | 0.679 | 0.670 | above $100bn |
| M1 inverse volatility | 0.851 | 0.850 | 0.846 | 0.837 | 0.821 | above $100bn |
| M2 risk parity | 0.789 | 0.788 | 0.784 | 0.777 | 0.764 | above $100bn |
| M9 mean-CVaR | 0.808 | 0.802 | 0.788 | 0.766 | 0.739 | above $100bn |
| M11 HRP | 0.770 | 0.768 | 0.765 | 0.757 | 0.743 | above $100bn |
| M12 HERC | 0.766 | 0.764 | 0.758 | 0.746 | 0.726 | above $100bn |
| M3 momentum | 0.312 | 0.261 | 0.157 | −0.043 | −0.293 | $102m |
| M4 mean reversion | −0.243 | −0.405 | −0.704 | −1.172 | −1.617 | n/a (loss-making) |
| M5 momentum + reversion | 0.069 | −0.064 | −0.305 | −0.733 | −1.178 | below $1m |
| Combination: equal | −0.034 | −0.184 | −0.459 | −0.936 | −1.411 | n/a |
| Combination: cost-aware | 0.040 | −0.067 | −0.264 | −0.617 | −1.013 | below $1m |
| Combination: Hedge | 0.080 | −0.033 | −0.233 | −0.556 | −0.892 | $1.2m |

**Outcome.** `h_capacity` **retained**: largest change −0.042 (mean-CVaR), then −0.020 (HERC), −0.015, −0.013, −0.012, −0.006. **Qualifications:** the rule is a bound; the margin is 0.008; at Y = 2 mean-CVaR changes by −0.086 and the bound fails, at Y = 0.5 the largest change is −0.021 (`stage25_sharpe_Y_sensitivity_1e9.csv`). `h_bands` **rejected**: +0.011 (momentum, p = 0.12), −0.004 (mean reversion, p = 0.34), +0.009 (combined, p = 0.054); none passes BH.

**Why the alpha books are so exposed.** At $1bn the mean trade in the alpha books is 0.5 to 1.2 times the asset's average daily dollar volume (95th percentile 2.4 to 5.6), against 0.02 to 0.11 for the allocators; the thinnest ETFs in the universe are DBC (median ADV $36m), SHY ($151m) and SLV ($221m), against $25bn for SPY. For 0.4% to 0.9% of traded asset-days participation exceeds 100% of ADV and is capped in the formula, which means the cost of those trades is **understated**, not overstated.

**Scheduling** (descriptive; `stage25_scheduling.csv`). For a median M3 trade (2.8% of capital, last year's median volatility 0.77% and ADV $2.3bn), spreading the trade over 1, 2, 5, 10 days cuts cost from 14.5 to 12.0, 9.8 and 8.7 bps and raises the one-standard-deviation timing risk from 0 to 39, 85 and 130 bps: spreading a trade trades a certain saving for a larger uncertain timing cost. Descriptive; no decision rests on it. An Almgren-Chriss *optimal trajectory* with a risk-aversion parameter was not built (Section 12).

*Figures 50-51.*

---

## 9. Reading the five stages together

The platform has two kinds of book. **Allocators** (equal weight, inverse volatility, risk parity, hierarchical) turn over 0.4 to 0.6 times a year, carry almost all of the Sharpe in this repository, and are insensitive to size: no estimation refinement tried in Generations 2 and 3 (regimes, dynamic covariance, Bayesian posteriors, calibrated probabilities, non-price data) improved a net allocation in a way that survived its own qualification. **Alpha books** turn over 9 to 22 times a year, earn a gross Sharpe of 0.4 at best (momentum), and are consumed by costs at any realistic size: at the linear Generation 1 rate they are marginal, with impact they are not viable above about $100m.

The Generation 3 results are the first in which the *size* dimension appears, and it reinforces the earlier conclusion: the signals the literature recommends are not the problem on this universe; the cost of trading them is.

---

## 10. What Generation 3 establishes, and what it does not

**Establishes.** (1) The Bayesian posterior averaging makes weights materially more stable and does not improve Sharpe. (2) Online learning on 187 monthly origins adds noise. (3) Credit, implied-volatility structure, jobless claims and positioning add nothing out of sample beyond price and macro, with the caveat that the matched-sample control moves the sign of the headline R². (4) Combining alphas at the forecast level gives books that turn over 13.6 to 19.6 times a year against 9.2 to 22.4 for the standalone alphas, and none beats the Generation 1 momentum book. (5) The impact-inclusive capacity of every alpha book is orders of magnitude below that of the allocators. (6) The execution layer reduces to Generation 1 exactly at zero size and is pinned by tests, and the banded executor reproduces the engine at band zero to 1e-12.

**Does not establish.** That these ideas are useless: only that they were not shown to help on 15 liquid ETFs over twenty years with the rules declared. The square-root impact coefficient is assumed, not estimated, and conclusions about *capacity* scale with it (Y = 0.5 moves M3's capacity to $631m, Y = 2 to $24m). The ETF volume series is consolidated daily volume, not executable depth; there is no intraday data, so the cost of crossing the spread in the closing auction and the actual impact of trading at the close are modelled, not observed.

**Limitations specific to Generation 3.** The same twenty years, now spent a third time; no multiple-testing correction *across* stages (9 decisions were declared in total, 2 retained, each with a qualification, so a chance retention would not be surprising); the usable non-price sample starts in 2010 and the combination window in 2011; Bayesian priors are the textbook conjugate and Jorion choices, not tuned.

---

## 11. Disclosed deviations and post-hoc analyses

Everything below happened after a first look at a result or after a first implementation. None overturned a declared decision.

| Where | What changed or was added | Why |
|---|---|---|
| Stage 21 | Jorion unit-test tolerance relaxed (0.2 to 0.3) | The empirical-Bayes weight 0.226 on a fixed seed is within the textbook range; tolerance was my guess |
| Stage 21 | Bootstrap stability uses 20 posterior draws per resample, the books use 100 | Compute; stated in the registry notes |
| Stage 21 | The stability rule's "inconclusive" band is recorded as reject | Rule written as retain / reject / inconclusive; only retain and reject exist in the registry |
| Stage 22 | Kalman observation variance is the warm-start residual variance and the prior coefficient variance is 1e-4 | Not in the declaration; implementation choices fixed before the result was seen, disclosed in the registry |
| Stage 22 | Forgetting-factor sweep (`posthoc_forgetting_sweep`) | Post-hoc, after the null; not judged |
| Stage 23 | Matched-sample control, lags-ignored control, and the 84-correlation BH count | Post-hoc; labelled in the file names |
| Stage 23 | The usable sample starts in 2010, later than Stage 15's | Feature burn-in (756 days) and the start dates of the implied-volatility series; declared as a limit in the registry |
| Stage 24 | Common window begins 2011-01-31, the declaration said 2011-01-03 | 2011-01-31 is the first Stage 19 origin, hence the first date on which every alpha exists |
| Stage 24 | After the result, the stage was changed to save the combination books for Stage 25 | Additive; the stage was re-run and every table reproduced to 1e-12 |
| Stage 25 | A first version of the banded executor did not reproduce the engine at band zero | Books beginning in cash (all-zero target rows) were treated as missing instead of held; found by the run's own assertion, fixed and pinned by a test **before** any `h_bands` result was read |
| Stage 25 | Missing volatility or volume is filled by the day's cross-sectional median | A gap must not make a trade free; counts are in `stage25_participation.csv` |

---

## 12. Not built

- **Generation 4:** transformers, an LLM research assistant, distributed experiments, a research database, cloud deployment.
- **Within Generation 3's own priorities:** ETF flows, earnings and analyst revisions, creation and redemption data, options flow (Section 6); hierarchical or model-averaged Bayesian priors beyond normal-inverse-Wishart plus Jorion; an Almgren-Chriss optimal trajectory (only the even-slicing table); intraday data and a spread model fitted to quotes; estimation of the impact coefficient.
- **Roadmap items outside Generation 3:** explainable machine learning (Priority 9), factor-model attribution (12), the software-engineering items of Priority 20, Wishart and factor stochastic-volatility covariance models (Priority 4).
- **The rename.** The project keeps its name.

---

## Appendix A: Reproducing this report

```bash
pip install -r requirements.txt
python -m experiments.run_all --fresh --download   # all 25 stages, ~60 minutes
python -m experiments.run_all --generation 3       # Generation 3 only
pytest -q
```

## Appendix B: Figures

Figures 42-51, each with the research question it answers in `reports/figures/*.txt` and `reports/figure_index.md`.

| Stage | Figures |
|---|---|
| 21 Bayesian portfolios | 42, 43 |
| 22 online learning | 44, 45 |
| 23 non-price data | 46, 47 |
| 24 alpha-combination engine | 48, 49 |
| 25 execution model | 50, 51 |
