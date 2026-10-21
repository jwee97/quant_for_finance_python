---
title: "Combining alphas: how alike they are, simple rules, and a model fitted without a glance ahead"
slug: combining-alphas-and-model-combination
difficulty: 3
chapter: Platform
prerequisites: [information-coefficient, factor-models-and-machine-learning, walk-forward-and-leakage]
stages: []
files: [src/fundamentals/combine.py, src/fundamentals/workbench.py, src/fundamentals/research.py]
figures: []
tests: [tests/test_fund_combine.py, tests/test_fund_workbench.py]
models: []
---

# Combining alphas: how alike they are, simple rules, and a model fitted without a glance ahead

## In one sentence

Two alphas that rank the companies the same way are one idea, so the first thing to know about a set of alphas is how many independent ideas it holds; the second is that a weighting learned from the past has to be learned only from returns that had already been realised, and that a model with more freedom than a plain average has to prove, out of sample, that the freedom earned something.

## The idea

**How alike.** The correlation of two signals is the average, over dates, of the rank correlation between them across the companies both cover (so signals with different coverage still compare on what they share). The distance is `1 − |correlation|` (an opposite is as much a copy as a twin), the alphas are grouped by average-linkage hierarchical clustering, and the heatmap is drawn in the order that puts alike alphas side by side. The **effective number of independent alphas** is `(Σλ)² / Σλ²` from the eigenvalues of the correlation matrix: 1 for identical alphas, the number of alphas for unrelated ones.

**Rules.** Each rule standardises every alpha on every date (cut the tails at 1%, then z-score among the investable companies) and then weighs them using only what was known on that date. *Equal*: the same weight to each. *IC*: by each alpha's trailing mean IC. *ICIR*: by its trailing information ratio. *Diversified*: the weights that maximise the information ratio of the combination given how the alphas' ICs move together (Qian, Hua and Sorensen), after orthogonalising the alphas and shrinking the covariance toward its diagonal. *Cluster*: equal weight to each cluster, then within it, so five copies of one idea do not outvote one different idea. The weights on a date use only ICs whose forward windows had ended by then, and until enough are known they are equal.

**A model.** A model learns to predict the cross-section of next-period returns from the alphas, trained on an expanding or rolling window and refitted every so often (ridge, lasso, elastic net, random forest, extra trees, gradient boosting, LightGBM, XGBoost). Each prediction comes from a model that saw only dates at least `purge` before the prediction date, with `purge` as long as the forecast horizon so that no training target overlaps the prediction window, plus an `embargo` you can widen. The predictions are therefore out of sample throughout and go through the same evaluation as any signal. A missing alpha is *no view* (zero after standardising: neither liked nor disliked), never a bad value.

**What it relied on.** Linear models report their coefficients; trees their own importance; every model its *permutation importance* (how much the IC falls when one alpha is shuffled across companies within each date); LightGBM and XGBoost their built-in TreeSHAP contributions. These answer different questions and can disagree.

**The comparison that matters.** The model's IC and the rules' ICs are laid side by side over the same dates, with a paired Newey-West test of the model's IC minus each rule's, and the best single alpha for scale (marked as chosen after looking at the same dates, so a generous yardstick).

## Why it matters

Combining is where most of the apparent gain in a multi-factor study comes from, and where much of the self-deception lives. Five value ratios look like five confirmations and are one; a model that sees the whole history to set its weights has been told the answer; and a gradient-boosted model will find *something* in noise and report an in-sample IC to prove it. The only combination worth having is one that is built from information available at the time and that beats the plain average it replaced, on dates it did not train on.

## How this repo uses it

`src/fundamentals/combine.py` has `correlation`, `cluster`, `effective_number`, `analyse` (the page's Correlation view), `combine` with its five rules, and `ml_combine` with `compare_with_rules`; `src/fundamentals/workbench.py` resolves each part (a library alpha, a factor, a price-based signal or a formula) to a signal panel with its warnings. The research context supplies the *settled lag*: the least number of periods after which every forward window has ended, read off the grid of dates and used for both the rules and the model's purge. The **Alpha library → Correlation, Combine and Model** views run it, `quant fundamentals` has no separate command for it (use the page or the Python below), and a combination or a model can be backtested as a signal.

## What we found

The tests pin the properties that matter: copies, opposites and strangers are found and grouped; the cluster rule gives each idea the same total weight; **altering the returns after a date leaves every earlier weight and prediction unchanged** (for every rule and model); every fold trains on dates far enough before it; tuning chooses from its grid on a validation slice inside each training window; and a model fed noise is not credited with beating the rules.

On the demo world, a trio built to be honest about the point (free-cash-flow yield and the change in ROIC, which have planted effects, and 12-1 month momentum, which has none and came out the wrong way round: mean IC +0.059, +0.023 and −0.032): the three are nearly unrelated (about 2.9 independent ideas out of 3). Equal weights give a mean IC of +0.003 (t = 0.4), because the wrong-way alpha cancels the other two; weighting by recent IC gives +0.038 (t = 4.9), by the IC information ratio +0.037 (t = 4.9) and the maximum-information-ratio rule +0.037 (t = 4.7); the best single alpha alone gives +0.059, a yardstick chosen after the fact. **None of the combinations beat the best part**: learning weights costs something in noise. A ridge model fitted walk-forward on the same three gave +0.037 over 126 dates, better than equal weights by +0.029 (t = 2.2) and worse than the IC-ratio rule by −0.009 (t = −1.0), and the page's verdict is that it *does not clearly beat the simple rules*. On three value ratios (which are alike: about 1.6 independent ideas) equal weights earned +0.076 and the same model, again, did not clearly beat the simple rules.

What has not been found: whether combining helps on real companies, where the alphas that matter are fewer and the noise is larger.

## Pitfalls

- **Counting alphas, not ideas.** Look at the effective number before you believe a set is diversified.
- **Using the best part as a yardstick.** It is chosen after seeing the same dates, so a combination that loses to it has lost to the lucky one.
- **Weights from the whole history.** Anything learned from realised returns must come only from windows that had ended; the settled lag is that, not "last month".
- **Orthogonalising or shrinking is not free.** The diversified rule needs enough history to estimate how the ICs move together; with little, its weights are noisy, and until enough is known it falls back to equal.
- **A model's importance depends on the method.** A coefficient, a split count, a permutation and a SHAP value answer different questions and can name different alphas.
- **A model that wins by a hair is not a win.** The test is the paired t-statistic of the gap; below about 2 the extra complexity is not shown to earn anything.
- **Standardising hides scale.** Each alpha is put on the same footing before weighing, so a weight is a statement about information, not about the size of the raw number.

## Try it

```python
from src.fundamentals.service import FundamentalsService
from src.fundamentals.workbench import Workbench

wb = Workbench(service=FundamentalsService(".", db_path="data/user/fundamentals_demo.db"))      # made by `quant fundamentals simulate --demo`
ctx = wb.context({}, "M", 1)
parts = [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}, {"kind": "technical", "name": "mom_12_1"}]
print(wb.correlation(ctx, parts)["reading"])
for rule in ("equal", "icir", "diversified"):
    out = wb.combination(ctx, parts, rule)
    print(rule, round(out["combined"]["mean_ic"], 3), round(out["combined"]["t_stat"], 1))
print(wb.machine_learning(ctx, parts, "ridge")["comparison"]["verdict"])
```
