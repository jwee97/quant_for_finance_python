"""Evaluation over predeclared seeds (spec v3, Release 5, P1-23).

A model that depends on a random seed (a reinforcement-learning agent, a neural network, a bagged ensemble) has two sources of uncertainty that are different in kind and must not be merged or hidden:

* **policy (seed) variability** - what retraining the same design with another seed would have given on the same market windows;
* **time-series uncertainty** - how much one seed's result would change on another sample of the market, given its dependence over time.

``run_seeds`` trains the same configuration under each of a list of seeds declared *before* the first run, on the identical data and splits, and reports every seed: the per-seed value of the declared
metric on the declared segment, the dependence-aware Sharpe interval of each seed's own return stream (time-series uncertainty), and the spread across seeds (policy variability), side by side. It does
not choose a seed: asking for the best one is refused, the point estimate is the median over seeds, and a spread that is as large as a seed's own sampling error is flagged. Each seed is a look at the
study (it is recorded in the lineage and counted), which is the conservative reading of "several tries of the same design".

Training seed and evaluation seed are both ``config.seed`` for the platform's models; where a model evaluates its policy deterministically (the RL agents' roll-forward does), the record says so.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from . import experiments as xp
from . import inference as inf

VERSION = "multiseed-v1"
MIN_SEEDS = 3


def declare(seeds) -> dict:
    s = [int(x) for x in seeds]
    if len(set(s)) != len(s):
        raise ValueError("the seeds must be distinct")
    if len(s) < MIN_SEEDS:
        raise ValueError(f"declare at least {MIN_SEEDS} seeds: a spread across fewer cannot be read")
    return {"seeds": s, "hash": hashlib.sha256(json.dumps(sorted(s)).encode()).hexdigest()[:16]}


def run_seeds(overrides: dict, seeds, *, segment: str = "validation", metric: str = "sharpe", select: str = "median", frequency: str = "daily", n_boot: int = 300, **kw) -> dict:
    """Train ``overrides`` once per seed and summarise. ``kw`` goes to ``run_experiment`` (``bundle``, ``config``, ``builder``, ``registry``, ``study``, ``stop_after`` ...)."""
    if select != "median":
        raise ValueError("a seed is never selected by its result: the point estimate is the median over the predeclared seeds (select='median')")
    decl = declare(seeds)
    rows = []
    for seed in decl["seeds"]:
        exp = xp.run_experiment({**overrides, "seed": seed, "name": f"{overrides.get('name', 'seeded')}-seed{seed}"}, explain=False, final_model=False, robustness=False, **kw)
        stats = exp.report["segments"]["stats"].get(segment) or {}
        value = stats.get(metric)
        net = exp.net[exp.labels.reindex(exp.net.index) == segment]
        sh = inf.sharpe_inference(net, frequency=frequency, label=f"seed {seed} {segment}", n_boot=n_boot, seed=seed)
        rows.append({"seed": seed, "experiment_id": exp.id, "value": None if value is None or not np.isfinite(value) else float(value), "n_days": stats.get("n_days"),
                     "sharpe_hac_interval": list(sh["hac"].interval) if sh["hac"].interval else None, "time_series_se_per_period": sh["hac"].extras.get("se_per_period"),
                     "eligibility": sh["hac"].eligibility})
    vals = np.array([r["value"] for r in rows if r["value"] is not None], dtype=float)
    out = {"version": VERSION, "kind": "multiseed", "segment": segment, "metric": metric, "declaration": decl, "rows": rows, "selection": "median over the predeclared seeds; no seed is chosen by its result",
           "training_seed": "config.seed", "evaluation_seed": "config.seed (deterministic policy evaluation where the model supports it)",
           "same_market_windows": True, "notes": ["each seed is recorded as a look at the study and counted", "policy variability and time-series uncertainty are reported separately and are not combined"]}
    if len(vals) < MIN_SEEDS:
        out.update(eligibility="insufficient", reason=f"{len(vals)} seeds produced a value on {segment}: fewer than {MIN_SEEDS}", summary=None)
        return out
    se_ts = [r["time_series_se_per_period"] for r in rows if r["time_series_se_per_period"]]
    ann = float(np.sqrt(inf.periods_per_year(frequency))) if metric == "sharpe" else 1.0
    within = float(np.mean(se_ts)) * ann if se_ts else None
    across = float(vals.std(ddof=1))
    out["eligibility"] = "applied"
    out["summary"] = {"median": float(np.median(vals)), "mean": float(vals.mean()), "std_across_seeds": across, "min": float(vals.min()), "max": float(vals.max()), "range": float(vals.max() - vals.min()),
                      "mean_time_series_se": within, "seed_spread_to_time_series_se": None if not within else across / within}
    out["flag"] = ("seed-to-seed spread is at least as large as a single seed's own sampling error: a result from one seed would have been mostly seed luck"
                   if within and across >= within else "seed-to-seed spread is smaller than a single seed's sampling error" if within else "")
    return out
