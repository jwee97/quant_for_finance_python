---
title: "Research operations: artifact store, model registry, honest tuning and CPCV"
slug: research-operations
difficulty: 3
chapter: Ch. 21
prerequisites: [experiment-database-and-reproducibility, multiple-testing]
stages: []
files: [src/ops/store.py, src/ops/hpo.py, src/ops/cv.py, src/ops/profiling.py]
figures: []
tests: [tests/test_ops.py]
models: []
---

# Research operations: artifact store, model registry, honest tuning and CPCV

## In one sentence

A result you cannot reproduce, or that came from a search whose size you did not record, is not a result; `src.ops` stores artifacts by their content hash, versions models with lineage and stages, tunes hyperparameters while counting every trial, cross-validates without leakage, and measures what things cost.

## The idea

**ArtifactStore** saves objects under the SHA-256 of their bytes, so identical content is stored once, a reference proves what was saved, and `get` refuses an object whose bytes no longer match its hash. **ModelRegistry** is a SQLite table of versioned models: parameters, data key, git commit, config fingerprint, metrics, artifact, parent and a lifecycle stage (`staging`, `production`, `archived`). Promoting a version to production archives the previous one, so exactly one is live, and `lineage` walks the parents. `register_run` registers a recorded pipeline experiment as a model version.

**Hyperparameter optimisation** (`Study.optimize`) offers random search, grid, a Tree-structured Parzen Estimator (Bergstra et al. 2011) and successive halving (Jamieson and Talwalkar 2016). What matters for finance is what comes after: `selection_report` computes the **deflated Sharpe ratio** of the winner with the trial variance estimated from all trials, the **probability of backtest overfitting** by combinatorially symmetric cross-validation, and the **Romano-Wolf** family-wise p-values. `tune_pipeline` runs this around a research pipeline, so the objective is a real net-of-cost Sharpe ratio and every trial is a separate run; failed trials still count.

**Purged and combinatorial purged cross-validation** (`purged_kfold`, `cpcv_splits`, `cpcv_evaluate`; Lopez de Prado 2018): a label looking `h` days ahead overlaps the neighbouring fold, so ordinary k-fold leaks. Purging removes training rows whose label windows overlap the test fold, and an embargo skips rows after it. CPCV splits the sample into `N` groups, tests on every combination of `k`, and recombines the test pieces into `k/N * C(N, k)` complete paths: a distribution of out-of-sample performance instead of one path.

**Profiling** (`profile_call`, `benchmark_models`, `scaling_exponent`) measures time, peak memory and the log-log scaling of any function; a rise in the scaling exponent after a change is a performance bug even when absolute times look fine.

## Why it matters

The most common way for a backtest to lie is a hidden search: the reported strategy is the best of many that were tried. Counting the trials and deflating the winner turns "Sharpe 1.2" into "Sharpe 1.2 out of 50 trials, deflated probability 0.4". Content addressing and lineage answer "which data and code produced this number?" months later.

## How this repo uses it

The registry sits next to the experiment database: any recorded experiment can be registered, promoted and traced. The dashboard and CLI use the same pipeline runner as `tune_pipeline`, so a tuned strategy is an ordinary experiment. The findings script tunes the lookback of time-series momentum on the ETFs and reports the selection statistics beside the winner.

## What we found

The tests check that the store round-trips every kind, deduplicates, detects corruption and collects garbage; that the registry versions models, keeps one production version, records lineage and artifacts, and registers an experiment run; that purged k-fold never leaks labels and covers the sample, that CPCV yields the right number of splits and paths with purging, and that CPCV evaluation separates signal from noise; that the search methods find the optimum on a known objective and count every trial, failed ones included; that successive halving spends its budget on the survivors; that the selection report deflates a winner picked from pure noise and keeps a real edge; and that the scaling exponent tells linear from quadratic work. (The TPE only beat random search after its Parzen estimator got a prior component; the early version did worse, and the test guards that.) See the tuning section of [the findings](../institutional_findings.md) for an actual search.

## Pitfalls

- Reusing a test set for tuning is the same as searching on it: count it.
- Embargo and purging lengths must cover the label horizon and the feature look-back.
- A TPE or successive-halving search is still a search; its winner is deflated like any other.
- The `frame` artifact kind is pickled: open only stores you trust.

## Try it

```python
import tempfile
from src.ops import ArtifactStore, Study, Float, cv

store = ArtifactStore(tempfile.mkdtemp())
ref = store.put({"model": "momentum", "lookback": 126})
print(store.get(ref))

study = Study({"x": Float(-3, 3)}, direction="maximize", seed=0)
study.optimize(lambda p: -(p["x"] - 1.0) ** 2, n_trials=40, method="random")
print(round(study.best.params["x"], 2), study.n_trials)
print(len(cv.cpcv_splits(1000, n_groups=6, n_test_groups=2, horizon=21, embargo=5)), cv.n_cpcv_paths(6, 2))
```
