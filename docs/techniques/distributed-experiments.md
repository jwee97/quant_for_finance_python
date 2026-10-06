---
title: "Running thousands of experiments in parallel"
slug: distributed-experiments
difficulty: 2
chapter: Ch. 7
prerequisites: [multiple-testing]
stages: [26]
files: [src/distributed/executor.py, experiments/stage26_distributed.py]
figures: []
tests: [tests/test_stage26_grid.py]
models: []
---

# Running thousands of experiments in parallel

## In one sentence

Because each experiment is independent, a grid of them can run on many cores or machines and must give exactly the answer a serial run gives.

## The idea

An executor takes a function and a list of tasks and runs them with a chosen backend (serial, joblib, dask). Tasks must be pure (same inputs, same outputs, with all randomness seeded) so that the backend does not matter.

## Why it matters

Search-aware research needs many experiments, and hundreds of thousands of backtests are impractical on one core. But parallelism that changes results is worse than none.

## How this repo uses it

Stage 26's 1,584-rule grid runs in the executor, and the run asserts equality between serial and parallel results to 1e-12. The deep models and the diffusion model use the same executor for their annual refits.

## What we found

Parallel and serial results were identical (EXP-072); the speedup depends on the hardware.

## Going deeper

```python
run_tasks(function, tasks, backend="joblib", n_jobs=-1, batch_size=1)    # same results for "serial", "joblib" and "dask"
```
A task function must be a pure function of its arguments: build everything it needs from the task and shared read-only data, and seed every random generator from values in the task.

## Pitfalls

- Unseeded randomness makes parallel runs irreproducible.
- Sending large shared data to every task can cost more than the work.
- A Ray adapter exists in the executor but was never run (Ray is not installed here); serial, joblib and dask were run and agree.

## Try it

```bash
python -m experiments.stage26_distributed --backend dask
```
