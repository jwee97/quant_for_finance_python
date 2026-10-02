# Generation 2: Regimes, Dynamic Covariance, Probabilistic Forecasts, Attribution

### Six extensions to the Generation 1 platform, each held to a rule written down before its result existed

**Universe and sample:** unchanged from Generation 1 (15 ETFs, 2006-01-03 to 2026-09-21)
**Data version:** `f13af1fed1f7` · **Macro data version:** `0cb282ac19f2` · **Config fingerprint (all namespaces):** `c25e66aa003a` (core: `12f9cfd14055`, unchanged)
**Reproduce with:** `python -m experiments.run_all --fresh` (all 20 stages, about 33 minutes) or `--generation 2`

---

## 1. Executive summary

### 1.1 What was asked, and what was built

The request was a long advisory roadmap (twenty priorities, a four-generation plan, and a suggested
rename to "Atlas") with no explicit instruction in it. It was read as authorisation for **Generation 2
only**, and the six items the roadmap names for it were built: regime detection (Priority 1),
probabilistic forecasting (2), dynamic covariance (4), macro features (5), performance attribution (8)
and hierarchical risk parity (11). Generations 3 and 4 were **not** built (Section 12).

Each stage was run the way Generation 1 was: the question and the decision rule are written to a YAML file
and committed **before** any result exists (regimes `43522cb`, dynamic covariance `7e682fc`, probabilistic
forecasting `e4302ef`; the macro stage's rule is in `c4c6b33`, which precedes the commit of its results,
`30b72f2`). One exception, stated plainly: the hierarchical stage's rule was committed together with its first
results (also `c4c6b33`), so git cannot show that it came first; that declaration rests on the
commit message and the registry text alone. Every test is paired or nested-model-aware,
families of tests are Benjamini-Hochberg controlled, and every look-ahead question is answered by the
automated truncation test rather than by inspection.

### 1.2 The answer

**Almost nothing new survived.** Of 23 decision-bearing hypotheses
declared in advance, **three** were retained and the rest rejected; and each of the three comes with a qualification:

| Retained | What it is | Why it is not a discovery |
|---|---|---|
| A filtered high-volatility HMM state predicts larger *absolute* next-day equity returns | Volatility clusters | A sanity check: a model that could not do this would be broken. It is not a trading result |
| A DCC-GARCH minimum-variance book is 3.9% less volatile than the shrinkage one (p = 0.008) | Dynamic correlations help a minimum-variance book a little | 0.15 points of volatility, for 1.8x turnover against 0.7x; the Sharpe ratio is not better (+0.032, p = 0.58) |
| Fractional-Kelly sizing beats direction-only sizing (+0.84 Sharpe, p = 0.021) | A confidence-sized book | The comparison is confounded: Kelly is net long through a bull market, direction-only trades a classifier that loses. Kelly loses to passive equal weight (−0.55, p = 0.008) and does not beat sign-only sizing of its own signal (−0.07, p = 0.54) |

The honest summary is the Generation 1 conclusion again, now with six more places where it could have been
overturned: the information that is easy to detect (volatility, correlation, regime labels) does not turn
into a better allocation after costs, and the models that estimate the most do the least well.

### 1.3 Seven findings worth stating plainly

1. **An engine bug was found and fixed.** The weight-drift step was one day stale. It moved the Generation 1
   Sharpe ratios by between −0.016 and +0.015, changed no ranking and no decision, and every Generation 1 number was
   regenerated (Section 3). It was found because the attribution identities would not close.

2. **Macro features do not forecast monthly sleeve returns beyond price.** 0 of 5 sleeves pass Clark-West at
   FDR 10% (smallest p = 0.101); mean out-of-sample R² of adding macro to price is +0.42%, with only
   177 months per sleeve and about 5.3 independent feature dimensions. This is weak evidence of absence.

3. **Regimes found in daily returns are volatility clusters, not bull and bear markets.** The two-state HMM's
   high-volatility spells last 3.2 days on average out of sample. The state predicts the size of tomorrow's move
   (p = 0.0012) and not its sign (p = 0.76).

4. **A regime overlay does not beat what it overlays.** De-risking equal weight on the filtered probability gives
   a net Sharpe of 0.904 against 0.858 (p = 0.45); a one-line volatility-percentile rule does as well (0.916).
   Gating momentum by regime changes its Sharpe by +0.002 (p = 0.90).

5. **Dynamic covariance wins on average loss against every static estimator and on significance against none.**
   One origin (2020-02-28) is 89% of the gap to the sample estimator; in a typical month the static estimators are as good or better.

6. **Hierarchical risk parity is not better than risk parity, and "more stable" meant "sits in a corner".**
   HRP and HERC are more stable than minimum variance in 0% and 6% of windows by the
   declared rule, but minimum variance holds 3.6 assets with a 97% maximum weight. No pair of the six allocators differs
   significantly in Sharpe ratio.

7. **Calibration made the probabilities worse.** Platt scaling raised the calibration error from 0.034 to 0.081;
   its slope came out negative in 4 of 16 refits and pooled AUC fell from 0.540 to 0.458. Neither the classifier nor
   the Gaussian mean beats a benchmark that knows nothing but each asset's historical average.

---

## 2. Scope, assumption and method

**Assumption, stated once.** The roadmap contained no imperative. Generation 2 was taken to be what was meant to be built; Generation 3
(Bayesian portfolio optimisation, online learning, alternative data, an alpha-combination engine, an execution model) and
Generation 4 (transformers, an LLM assistant, distributed experiments, a research database, cloud deployment) were left out.

**Pre-registration.** For every stage the design and the decision rules are in a YAML file under `config/`, committed
before any result was computed for Stages 15, 16, 17 and 19; the Stage 18 rule was committed with its first results (Section 1.1). Post-hoc diagnostics are labelled as such in the registry, in the tables (`posthoc` in the name) and
here, and never overturn a declared decision. Every change made after a first look is listed in Section 11.

**Identity of a result.** The core configuration fingerprint is unchanged by Generation 2 (`12f9cfd14055`); the Generation 2 namespaces
extend the fingerprint of the full configuration to `c25e66aa003a`. Macro data has its own manifest (`0cb282ac19f2`) and was frozen
(`bd0c330`) before Stage 15 was run on it.

**Timing.** Unchanged: `R_t = w_{t-1}' r_t`, signals lagged one day, monthly rebalance, per-asset linear costs, weights drift. Macro series
enter only after their publication lag (consumer prices 45 days, unemployment 40, the Philadelphia Fed survey 22, daily series 1).

---

## 3. Erratum: the weight-drift step, and what fixing it changed

**The bug.** `drift_weights` grew the book by day *t*'s return when stepping to *t+1*. That applied the rebalance day's return to a book
that had only just been struck at that close, left every later row one day stale, and measured a rebalance's trade against the previous
row's book instead of the drifted book at the same close. It was found when the Stage 20 attribution (start-of-month weights times
monthly asset returns) did not reconcile with the engine's own monthly returns.

**The fix** (`4ae25fa`). `held[t]` is the book in force at the close of *t*: the target on a rebalance row (it has not earned that day),
otherwise the previous close's book grown by day *t*'s return and renormalised to constant gross; trades are measured against the same
close's drifted book. Three exact tests pin it: drifted weights against a hand-computed buy-and-hold path, the engine's compounded return
over a holding month against buy-and-hold to 1e-12, and a rebalance trade against the drifted book.

**The effect, measured** (`reports/errata/drift_fix_final_comparison.csv`):

| Book | Sharpe before | Sharpe after | Change | CAGR | Max drawdown |
|------|-----:|-----:|-----:|------|------|
| M0 equal weight | 0.70 | 0.69 | −0.015 | 7.3% → 7.1% | −33.0% → −33.4% |
| M1 inverse volatility | 0.87 | 0.85 | −0.014 | 5.5% → 5.4% | −17.6% → −17.7% |
| M2 risk parity | 0.80 | 0.79 | −0.016 | 5.2% → 5.0% | −18.4% → −18.5% |
| M3 momentum | 0.30 | 0.31 | +0.015 | 1.8% → 1.9% | −15.2% → −14.8% |
| M4 mean reversion | −0.24 | −0.24 | −0.004 | −1.8% → −1.9% | −34.4% → −34.7% |
| M5 momentum + reversion | 0.06 | 0.07 | +0.007 | 0.2% → 0.2% | −20.4% → −19.9% |
| M6 combined alpha + MVO | 0.51 | 0.50 | −0.008 | 3.6% → 3.5% | −21.2% → −21.2% |
| M7 + shrinkage covariance | 0.50 | 0.50 | −0.008 | 3.6% → 3.5% | −21.2% → −21.1% |
| M8 Black-Litterman | 0.43 | 0.42 | −0.008 | 4.3% → 4.2% | −29.3% → −29.5% |
| M9 mean-CVaR | 0.81 | 0.81 | −0.002 | 3.3% → 3.3% | −14.2% → −14.2% |
| SPY buy & hold | 0.65 | 0.65 | +0.000 | 11.1% → 11.1% | −55.2% → −55.2% |

Second order. Sharpe changes lie in [−0.016, +0.015]; the ranking of the eleven books is identical; breakeven costs move from
27.8 / −1.1 / 8.5 bps to 28.9 / −1.2 / 8.8 bps (momentum, mean reversion, combined). **No registry
decision changed (0 of 58).** The larger moves are in crisis windows (inverse volatility's GFC return goes from −10.6% to −11.4%) and in
walk-forward ordering (mean-CVaR now edges inverse volatility, 0.829 against 0.826, inside any confidence interval). Forty-six of 113
Generation 1 tables changed; the Generation 1 report was regenerated cell by cell against them.

**Other corrections made in the same pass** (not caused by the fix; found because every number was re-read against its table):

- the CVaR column of the Generation 1 headline table mixed values from different sources (for example M8 1.71% where its table said 1.68%);
  it now comes from one table;
- Section 11 of that report quoted a mean stream correlation of +0.31; the registry value was +0.24;
- "one-third to one-half of SPY's drawdown" was always "a quarter to a third" (0.26 to 0.34 of it);
- the Section 11 claim "lowest drawdown of any book" is now "joint-lowest, level with the alpha-only blend";
- a few cells of the stress, walk-forward and holdout tables differed from their CSVs in the last digit.

The Generation 1 consistency tests now parse the headline, walk-forward and holdout tables and compare every cell to its CSV.

---

## 4. Macro features and nested-model predictability (Stage 15)

**Data.** Nine series (FRED: ten-year minus two-year and minus three-month yield curves, the effective federal funds rate, the ten-year
breakeven, consumer prices, unemployment, the Philadelphia Fed survey; Yahoo: VIX and MOVE) become 13 features (levels, changes and the
Sahm gap), each standardised on an expanding window and available only after its release lag. The typical staleness of a monthly series on a
trading day is 15 days. Thirteen features carry about 5.3 independent dimensions (participation ratio of the eigenvalue spectrum).

**Question.** Does adding macro to price features improve out-of-sample monthly sleeve excess returns (five sleeves, expanding window,
177 to 188 months)? **Test.** Clark-West, because the models are nested and a plain Diebold-Mariano test is undersized
on nested models; BH at 10% across the 15 comparisons. **Rule (declared).** Retain if at least one sleeve passes for price+macro against price *and* the mean
out-of-sample R² of that comparison is positive.

**Result: rejected.** No comparison is significant after BH control. Mean out-of-sample R²: price against the historical mean −0.81%,
macro against the historical mean −0.36%, price+macro against price +0.42% (smallest p, bonds: 0.101). With 177 months and about five
independent dimensions there is little to find and little power to find it: a null here is weak evidence of absence.

**Publication-lag control (recorded, not a decision).** Ignoring release lags inflates the pre-declared comparison's R² by only
+0.04%; post-hoc, by +0.48% on price+macro against price, and direct tests of whether ignoring lags makes the same model forecast better find
0 of 10 pairs significant. The lags are two to six weeks and the features barely forecast, so there is little to inflate. The availability
machinery is kept because it is correct by construction. (The control's first draft, "retain if inflation > 0", could not tell an effect from noise and
was reworded before any decision rested on it; the original wording is in the git history.)

**Does it survive costs?** A price-only signal has a net Sharpe of −0.158, price+macro +0.004; the paired bootstrap difference is +0.162 (p = 0.35): not an improvement. Both books are far below the passive allocations of Generation 1.

*Figures 28-29.*

---

## 5. Regime detection (Stage 16)

**Models.** A two-state Gaussian HMM (and a three-state variant) fitted walk-forward with a refit every 126 days; the **filtered** probability at the
close of *t* is the only tradable quantity, the **smoothed** one uses the future and is kept only as a look-ahead control. A Gaussian mixture on the
same inputs, Adams-MacKay Bayesian online change-point detection (BOCPD, constant hazard, alarm when the short-run mass is high), and four
named rule-based regimes (bear market, high volatility, inflation shock, liquidity crisis). The forward filter is implemented directly and checked against
`hmmlearn`.

**A test built to be correctly sized.** Regime labels and next-day returns are both persistent, so a shuffle test is badly oversized. The test is a circular-shift
permutation test on a **studentised** statistic. Simulation: with a persistent label and persistent outcome, a naive shuffle rejects a true null
78% of the time at the 5% level and the circular shift 6.7%; for a rare high-variance regime with no mean effect the plain
difference of means rejects 36% of the time and the studentised statistic 3.3%.

**Results against the declared rules.**

| Hypothesis | Outcome |
|---|---|
| High-volatility state predicts larger absolute returns next day | **Retained**: 1.31% against 0.75% a day, p = 0.0012, t = 10.5 (a sanity check) |
| High-volatility state predicts a different mean return | Rejected: p = 0.76; the point estimate is of the opposite sign to what a de-risking rule assumes |
| Named rule-based regimes predict different sleeve returns | Rejected: none significant after BH (12 tests; 3 untestable for too few days) |
| BOCPD flags five dated volatility shocks within 60 days with few false alarms | Rejected: 2 of 5 under the rule (an alarm must start on or after the date), 1.01 false alarms a year |
| De-risking equal weight on P(high vol) beats equal weight | Rejected: Sharpe 0.904 against 0.858, p = 0.45 |
| Blending equal weight and risk parity by P(high vol) beats both | Rejected: 0.888 against 0.858 and 0.924 (p = 0.46, 0.55) |
| Gating momentum off in losing regimes beats ungated momentum | Rejected: 0.295 against 0.293, p = 0.90 |

**What the regimes are.** Out-of-sample, the two-state high-volatility spells last 3.2 days on average (1.4 for the mixture, which has no Markov chain; 4.0 for the three-state model);
the full-sample fit implies 7 days in the high state and 20 in the calm one. The named rules last far longer (bear market 94 days, high volatility 41).
A monthly overlay samples one day in twenty from a state that lasts a few days, which is part of why it adds nothing.

**Caveats that matter.** (1) The window starts in December 2008, the first out-of-sample HMM probability; every book's Sharpe is flattered by the
post-crisis bull market (equal weight 0.86 here against 0.69 in the full sample), which is why the stage's deflated-Sharpe probabilities are high for every book and mean little. (2) The one-line rule
"expanding volatility percentile above the 80th" reaches 0.916 with a −13.6% maximum drawdown against the HMM's −17.1%: a slower state estimate suits a monthly overlay better. (3) BOCPD
fired one to four days *before* three of the dated events, on the same sell-offs; crediting that (post-hoc) gives 5 of 5 at 0.86 false alarms a year, but the
volatility-jump rule detects all five as well and the Bayesian machinery is not shown to beat it.

**Look-ahead.** The automated truncation test passed all 16 honest rules and flagged all 20 deliberate look-ahead controls (smoothed probabilities, and filtered
probabilities with full-sample parameters). At the monthly overlay hindsight did **not** inflate the Sharpe ratio (honest 0.904, smoothed 0.888); at a weekly overlay (post-hoc)
it did (honest 0.871, smoothed 0.906, p = 0.41). A leak need not show in the headline number to be a leak, which is why the test exists.

*Figures 30-33.*

---

## 6. Dynamic covariance (Stage 17)

**Models.** DCC-GARCH (two-step Gaussian quasi-maximum likelihood; univariate GARCH(1,1) per asset, then the correlation recursion) and Orthogonal GARCH
(three principal factors with GARCH variances, constant diagonal for the rest), both fitted walk-forward with annual refits and forecasting the **average daily
covariance over the next 21 days**, against the sample, EWMA and shrinkage estimators. The loss is the Gaussian deviance (multivariate QLIKE). In simulation the
DCC estimator recovers (a, b) = (0.043, 0.935) for a truth of (0.04, 0.94). On the data a is 0.013 to 0.018 and b 0.950 to 0.982, persistence 0.966 to 0.995.
Every forecast of every model was positive definite.

**Results against the declared rules.**

| Hypothesis | Outcome |
|---|---|
| DCC has lower loss than sample, EWMA and shrinkage (196 monthly origins) | Rejected. Mean loss lower against all three (−6.2, −5.6, −4.2); none significant (p = 0.27, 0.33, 0.11) |
| The same inside the GFC, COVID and 2022 windows (27 origins, 14 assets) | Rejected: p = 0.22, 0.25, 0.19; little power by construction |
| A DCC minimum-variance book has lower realised volatility than the shrinkage book | **Retained**: 3.67% against 3.82%, p = 0.008 |
| Orthogonal GARCH has lower loss than each static estimator | Rejected: worse than all three (+0.7, +1.3, +2.6) |
| Orthogonal GARCH, crisis windows | Rejected |
| An Orthogonal GARCH minimum-variance book has lower volatility than shrinkage | Rejected: 4.1% *higher* (significantly, p = 0.013, in the wrong direction) |

**Post-hoc, labelled.** The mean favours DCC but a few stress months drive it: DCC has the lower loss on only
41% of origins against the sample estimator and 33% against EWMA, but 64% against shrinkage. A single origin, 2020-02-28, is
89% of the total gap to the sample estimator; against shrinkage the advantage is broad and survives dropping it. Inside the windows the picture differs:
DCC is ahead in the GFC, far ahead in COVID, and level in 2022. The retained minimum-variance result is real but small: the books' returns are 0.97
correlated, the volatility gain is 0.15 points, it costs 1.8x turnover against 0.7x (13 bp a year of cost against 5), and the Sharpe difference is not significant. For risk-parity books no model changes volatility.
One observation outside the rule, flagged for transparency: the Orthogonal GARCH minimum-variance book has a much higher Sharpe ratio (0.96 against 0.72, uncorrected p = 0.003) with *higher* volatility.
It is one of four Sharpe comparisons that were not declared (the rule was about volatility), it is a different asset mix rather than a better risk forecast, and it is not relied on.
Wishart and factor stochastic-volatility covariance models are not implemented.

*Figures 34-35.*

---

## 7. Hierarchical risk parity (Stage 18)

**Methods.** HRP (López de Prado 2016: single-linkage clustering of the correlation distance, quasi-diagonalisation, recursive bisection with capital split inversely to cluster variance) and HERC
(Raffinot 2017: Ward clustering, the dendrogram cut into *k* clusters, equal risk contribution between and within clusters). Two departures from the HERC paper, stated in the code: *k* is chosen by the silhouette
score rather than the gap statistic, and within-cluster allocation is exact equal risk contribution. Costs, timing and constraints are those of the Generation 1 allocators.

**Stability (rule declared in advance).** Under bootstrapped estimation noise (16 windows), retain if the hierarchical methods have the lower weight dispersion than
minimum variance in at least 75% of windows, reject if in at most 25%. HRP is more stable in 0% of windows and HERC in 6.2%: **rejected.** This does *not* mean
minimum variance is well behaved: it holds 3.6 assets with a 97% maximum weight and an effective N of 1.06, so its weights cannot move (the Generation 1 lesson that a book
can be stable by sitting in a corner). HRP is nearly as concentrated (effective N 1.39; 81% in the short-duration bond fund at the latest rebalance); HERC is the most diversified of the hierarchical pair (effective N 4.5).
Post-hoc: acting on a noisy estimate raises realised volatility above the baseline book's by 1.4% (HRP), 1.3% (HERC), 0.5% (risk parity) and 0.8% (minimum variance).

**Performance (rule declared in advance).** Net Sharpe, full sample: inverse volatility 0.851, mean-CVaR 0.808, risk parity 0.789, HRP 0.770, HERC 0.766, equal weight 0.686.
Paired bootstrap on all 15 pairs, BH controlled: **no pair is significant**, so neither HRP nor HERC is "statistically better than risk parity" (rejected).
The same machinery re-checks a Generation 1 sentence: the three leading allocators are indistinguishable (all p above 0.95 on the pairs among them), but so are *equal weight and
any of them* (p = 0.11 to 0.67 out of sample): the Generation 1 statement that risk-based allocation "helps" is a point estimate, not a significant difference.

*Figures 36-37.*

---

## 8. Probabilistic forecasting and confidence-aware sizing (Stage 19)

**Forecasts.** At each month-end, from January 2011 (origins after a 1,260-day minimum training window, annual refits, a 21-day embargo), for each of the 15 ETFs: (a) the
probability that the next 21-day return is positive, from an L2 logistic classifier on 12 price features (+ the filtered regime probability and macro features, interacted with asset-class
dummies, in the second variant), Platt-calibrated on the last 20% of each training block after an embargo; (b) a Gaussian forecast for the return, with a ridge mean and an EWMA volatility. They are scored with
proper rules (log loss, Brier, CRPS, negative log score, PIT, reliability) against benchmarks that carry no feature information: each asset's base rate, and
each asset's training mean with the *same* volatility.

**Results against the declared rules.**

| Hypothesis | Outcome |
|---|---|
| Platt scaling lowers calibration error | Rejected: ECE 0.034 raw, 0.081 calibrated (p of "not improved" = 0.99); calibration slope 0.35 raw, −0.47 calibrated |
| Calibrated classifier beats the base rate (log loss) | Rejected: 0.7001 against 0.6807 (p = 0.00002, significantly *worse*); AUC 0.458 |
| Gaussian mean beats the training mean (CRPS) | Rejected: 0.0230 against 0.0224 (p = 0.003) |
| Macro and regime features improve either score | Rejected: log loss p = 0.57; CRPS *worse* by 0.0016 (p = 0.001) |
| Probability-sized books beat direction-only | Rejected: −0.082 Sharpe (p = 0.29) |
| Fractional Kelly beats direction-only | **Retained by the letter of the rule**: +0.84 (p = 0.021, BH-significant); confounded, below |

**Why calibration failed.** The Platt slope is estimated on the last 20% of each training block, a few dozen months that share a market, so it is very noisy: it was negative in
4 of 16 refits, reversing the classifier's ordering there. Calibration makes probabilities honest about the information they have; fitted on too little data it manufactures a confident, wrong signal.
Post-hoc: the **raw** classifier's log loss is 0.6847 against 0.6807 (p = 0.51), so most of the damage is the calibration step, and the raw classifier still carries no skill once each asset's base rate is removed (AUC 0.520).
Realised signed return by |edge| quintile *falls* from +0.17% (smallest edge) to −0.71% (largest): a bigger claimed edge was a worse bet.
The Gaussian forecasts' spread is off in the tails (14% of outcomes below the 10th percentile, 9% above the 90th), so the PIT histogram is skewed.

**The retained Kelly result, with the post-hoc controls.** Direction-only trades the classifier's signal long and short and loses (net Sharpe −0.66, mean net exposure
−0.15); the Kelly book trades a different signal (the Gaussian mean) and is net long (0.67) through a bull market. Against passive equal weight (net Sharpe 0.73) Kelly's difference is
−0.55 (p = 0.008); against sign-only sizing of the same Gaussian signal it is −0.07 (p = 0.54). The decision is not overturned; its interpretation is: **there is no evidence here that sizing by
confidence adds value**. The price+macro versions are reported but not judged.

*Figures 38-39.*

---

## 9. Performance attribution (Stage 20)

Accounting, not hypothesis testing: every identity is checked to machine precision and the stage raises if one fails (28 checks, worst error 8.5e-14).
All books are attributed over the **same** 233 months (2007-05 to 2026-09, those in which every book is fully invested); a first version measured each book over its own life
and compared cumulative returns over different months, which the cross-check between the two analyses (now part of the identity checks) was added to prevent.

**Where the active return against equal weight came from** (Brinson-Fachler by asset class, Carino-linked so the effects sum *exactly* to the cumulative active return):

| Book | Cumulative active return | Allocation | Selection | Interaction | Volatility (book / equal weight) |
|---|--:|--:|--:|--:|--:|
| Inverse volatility | −94.6% | −81.1% | −9.0% | −4.5% | 6.5% / 10.2% |
| Risk parity | −91.4% | −78.4% | −7.2% | −5.8% | 6.6% / 10.2% |
| HRP | −59.8% | −47.6% | −7.6% | −4.6% | 7.3% / 10.2% |
| HERC | −104.4% | −88.2% | −10.7% | −5.6% | 6.5% / 10.2% |

Every risk-based allocator trails equal weight in cumulative *gross* return (percentage points of cumulative return), and almost all of the shortfall is **allocation**, not
selection within classes: inverse volatility holds on average 18 percentage points more of the book in rates and 14 points less in equity than equal weight, through a long equity bull market. The comparison is **not risk-adjusted**: these books run at roughly
two thirds of equal weight's volatility, and their Sharpe ratios are higher (Generation 1 table). Attribution says where the return difference came from, not whether it was worth taking.

**Where the risk sits.** Equal weight puts 55% of forecast volatility in equities; inverse volatility and risk parity 47% and HERC 40%. Risk parity is meant to equalise risk contributions across *assets*;
the dispersion of asset risk shares across month-ends is 0.034 for risk parity against 0.054 for equal weight (0.046 HRP, 0.035 HERC), not zero, because the covariance is estimated over a trailing window and
long-only caps bind.

**What trading costs.** Basis points a year over the window: equal weight 2.2, the four risk-based books 3.1-3.8, momentum 58, mean reversion 140.

**Which sleeve supplied the combined return.** The Stage 10 inverse-volatility blend returned 78.1% cumulatively: mean-CVaR +27.1% (30% of the weight), inverse volatility +26.0%,
risk parity +23.5%, momentum +6.4%, mean reversion −4.9%.

Not built: factor-model attribution (Priority 12).

*Figures 40-41.*

---

## 10. What Generation 2 establishes, and what it does not

**Establishes.** (1) The engine's drift timing was wrong by a day, the effect is second order, and the attribution reconciliation is what exposed it. (2) On this universe and
sample none of the six extensions improves a net-of-cost allocation by an amount distinguishable from noise. (3) The failures have identifiable causes: too few
independent observations (macro), labels that last days against a monthly rebalance (regimes), a mean-based test dominated by a few crisis months (dynamic covariance), "stable" meaning "concentrated" (HRP),
and an estimator fitted on a few dozen months (calibration). (4) The methodological pieces are reusable and tested: a correctly sized permutation test, a nested-model test, an
automated look-ahead test with failing controls for filtered and smoothed probabilities, proper scoring rules, and exact attribution identities.

**Does not establish.** That regimes, dynamic covariance or probabilistic forecasts are useless: only that they were not shown to help *here*, with 177 to 196 independent monthly
origins and a sample that contains one deflationary crisis and one inflationary one. Several nulls (macro, crisis-window covariance) have little power and are stated as weak evidence of absence.

**Limitations specific to Generation 2.** The same twenty years, now spent a second time; no multiple-testing correction *across* stages (within each family the control is BH at 10%; 23 decisions
were declared in total and three were retained, each with a qualification, so a chance retention would not be surprising); the HMM sample starts in December 2008;
the macro data are today's vintages with modelled release lags rather than real-time vintages, so revisions are not captured.

---

## 11. Disclosed deviations and post-hoc analyses

Everything below happened after a first look at a result. None overturned a declared decision.

| Where | What changed or was added | Why |
|---|---|---|
| Stage 16 | The circular-shift statistic was **studentised** | The plain statistic put three rare high-variance flags across the line; the size simulation (Section 5) showed 36% rejection of a true null against 3%. Both p-values are in `stage16_permutation_tests.csv` |
| Stage 16 | BOCPD scored by the pre-declared rule (alarm must start on/after the date); "credited early" version post-hoc | Three alarms fired one to four days before the date |
| Stage 16 | Weekly overlay, full-sample-parameter and smoothed variants (post-hoc and controls) | To show the look-ahead test catches a leak that the headline Sharpe ratio hides |
| Stage 15 | The lag-control entry's decision rule reworded; the original is in git history | It could not distinguish an effect from noise |
| Stage 17 | Concentration, sign tests, win rates, O-GARCH factor-count sensitivity | Post-hoc diagnostics of the mean-based loss test |
| Stage 18 | Rule committed with the first results, not before | Section 1.1 |
| Stage 18 | Risk-inflation columns | Post-hoc, after the dispersion result |
| Stage 19 | Raw-classifier controls; passive and sign-only controls for Kelly | Post-hoc, after the calibration failure and the Kelly result |
| Stage 19 | A YAML pre-declaration committed with a syntax error (implicit string concatenation) was amended before any result was computed | Hygiene |
| Stage 20 | Common window for all books; window-agreement checks | A first version used each book's own life (Section 9) |
| Engine | Weight-drift fix | Section 3 |

---

## 12. Not built

- **Generation 3:** Bayesian portfolio optimisation, online learning, alternative data, the alpha-combination engine, the execution model.
- **Generation 4:** transformers, an LLM research assistant, distributed experiments, a research database, cloud deployment.
- **Within Generation 2's own priorities:** Wishart and factor stochastic-volatility covariance models (Priority 4); factor-model performance attribution (Priority 12); the software-engineering items of Priority 20.
- **The rename.** The project keeps its name; "Atlas" was an aspiration in the roadmap, not an instruction.

---

## Appendix A: Reproducing this report

```bash
pip install -r requirements.txt
python -m experiments.run_all --fresh --download   # all 20 stages, ~33 minutes
python -m experiments.run_all --generation 2       # Generation 2 only
pytest -q
```

## Appendix B: Figures

Figures 28-41, each with the research question it answers in `reports/figures/*.txt` and `reports/figure_index.md`.

| Stage | Figures |
|---|---|
| 15 macro | 28, 29 |
| 16 regimes | 30, 31, 32, 33 |
| 17 dynamic covariance | 34, 35 |
| 18 hierarchical risk parity | 36, 37 |
| 19 probabilistic forecasting | 38, 39 |
| 20 attribution | 40, 41 |
