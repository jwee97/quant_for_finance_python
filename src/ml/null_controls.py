"""Null-control studies: how often does the whole adaptive pipeline promote a strategy when there is nothing to find? (spec v3, P0-26; protocol in 6.3)

A deflated Sharpe ratio or a Reality Check p-value is a promise about the *procedure*: if the pipeline is run on data with no signal, it will promote a strategy at most this often. The only way to check the promise
is to run the pipeline on such data, many times, with everything the real pipeline does (the model and setting search, the selection by the validation score, the portfolio construction) and ask how often the frozen
validation gate says yes. Permuting the selected strategy's returns afterwards would test the last step only, so each replication here runs the whole pipeline afresh on its own placebo data.

**Null generator.** ``placebo_dataset`` shifts the returns (and with them the prices, the targets and the investable mask) of all assets *together* by a random number of days, circularly, against an unchanged feature
panel (the targets are recomputed from the shifted prices). The features keep their own history and cross-asset structure, the returns keep theirs (volatility clustering, cross-asset correlation, fat tails), and any relation between the two is destroyed: the
hypothesis tested is "the features carry no information about later returns". Limits: it also severs the contemporaneous link between a feature and the size of the next move (so it is not a null for models that
forecast volatility), the shift has one seam where the return series wraps (its single return is set to zero, so a label spanning it contains one flat day), and the real market may have real predictability that the
placebo removes (that is the point of a null).

**Protocol.** The specification (budget, candidate kinds, generator, statistic, gate, tolerance, precision, minimum replications, seed) is hashed and stored BEFORE any replication runs; a replication whose
spec differs is refused, and changing anything after seeing results is a new study. Each replication records its seed, the offset, the candidates tried, the selected one, the gate's verdicts and whether the
pipeline promoted. ``summarize`` reports the replications, the false promotions, the empirical rate and exact (Clopper-Pearson) and Wilson intervals, the Monte Carlo resolution, and ``sufficient``: a run with too
few replications, or an interval wider than the predeclared precision, validates the plumbing only and is labelled so. It does not require any individual null Sharpe to be negative (some will be positive by
chance) and it does not assume lagged returns are unpredictable.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import research_controls as rc
from .provenance import canon, digest

NULL_VERSION = 1


@dataclass(frozen=True)
class NullSpec:
    """Everything fixed before the first replication."""

    name: str = "search pipeline on placebo returns"
    study_id: str = "u-etf"
    base: dict = field(default_factory=dict)                                      # the experiment configuration (overrides) the pipeline starts from
    kinds: tuple = ("ridge", "lasso")
    n_random: int = 4                                                             # the search budget of each replication
    n_bayes: int = 0
    replications: int = 20
    seed: int = 0
    generator: str = "circular_shift_returns"
    min_shift: int = 200                                                          # days: well beyond every feature window, so no feature can reach across the shift
    statistic: str = "one-sided test that the selected strategy's mean daily net return on the placebo test segment is positive"
    gate: dict = field(default_factory=lambda: {"selection": ["dsr", "spa"], "holdout_alpha": 0.05, "dsr_min": 0.95, "alpha": 0.05})
    tolerance: float = 0.10                                                       # the largest false-promotion rate that is acceptable
    precision: float = 0.10                                                       # the widest acceptable half-width of the confidence interval for the rate
    min_replications: int = 100                                                   # fewer is a smoke run
    confidence: float = 0.95

    def frozen(self) -> dict:
        d = asdict(self)
        d["kinds"] = list(d["kinds"])
        return {"version": NULL_VERSION, **d}


def spec_hash(spec: NullSpec) -> str:
    return digest(spec.frozen(), 24)


# ------------------------------------------------------------------------------------------------------------------ the null generator
def placebo_dataset(ds, offset: int):
    """A copy of ``ds`` whose returns, prices and investable mask are shifted by ``offset`` days (circularly, all assets together) against the unchanged features, with the targets recomputed from the shifted
    prices. The seam, where the end of the return series meets its start, is one day of zero return, so prices stay continuous and a label that spans it contains one flat day. The features, their dates
    and their dependence structure are untouched."""
    from .datasets import Dataset

    n = len(ds.dates)
    k = int(offset) % n
    if k == 0:
        raise ValueError("a placebo shift of zero days is the real data")
    ret0 = ds.returns.to_numpy(dtype=float)
    rolled = np.roll(ret0, k, axis=0).copy()
    rolled[k] = 0.0                                                                               # the wrap-around: the day the end of the series meets its start
    rolled[0] = ret0[0]
    returns = pd.DataFrame(rolled, index=ds.returns.index, columns=ds.returns.columns)
    step = np.nan_to_num(rolled, nan=0.0)
    step[0] = 0.0
    p0 = ds.prices.iloc[0].to_numpy(dtype=float)
    prices = pd.DataFrame(p0 * np.cumprod(1.0 + step, axis=0), index=ds.prices.index, columns=ds.prices.columns)
    prices = prices.where(returns.notna() | (np.arange(n) == 0)[:, None])
    inv = pd.DataFrame(np.roll(ds.investable.to_numpy(), k, axis=0), index=ds.investable.index, columns=ds.investable.columns)
    from .targets import make_targets

    names = list(ds.Y.columns)
    targets = make_targets(names, prices, returns, int(ds.lag), str(ds.meta.get("benchmark", "equal_weight")), groups=dict(ds.group or {}))         # the labels are recomputed from the placebo prices, row for row where the real dataset had rows
    dpos = pd.Series(np.arange(n), index=ds.dates).reindex(ds.Y.index.get_level_values(0)).to_numpy()
    apos = pd.Index(ds.assets).get_indexer(ds.Y.index.get_level_values(1))
    Y = pd.DataFrame({t: targets[t].reindex(index=ds.dates, columns=ds.assets).to_numpy(dtype=float)[dpos, apos] for t in names}, index=ds.Y.index)
    meta = dict(ds.meta)
    meta["placebo"] = {"generator": "circular_shift_returns", "offset": k}
    return Dataset(ds.name + "-placebo", ds.X, Y, prices, returns, inv, meta, dict(ds.asset_class), dict(ds.group))


def offset_for(seed: int, n_dates: int, min_shift: int) -> int:
    """A deterministic shift in ``[min_shift, n_dates - min_shift]`` for a replication seed."""
    if n_dates < 2 * min_shift + 10:
        raise ValueError(f"the data has {n_dates} dates; a placebo shift needs at least {2 * min_shift + 10} (min_shift={min_shift} on each side)")
    return int(np.random.default_rng(seed).integers(min_shift, n_dates - min_shift + 1))


# ------------------------------------------------------------------------------------------------------------------ the pipeline under test
def search_pipeline(spec: NullSpec, bundle=None, config=None, builder=None, info=None):
    """The default pipeline under test: the platform's staged search over ``spec.kinds`` and their settings (validation-only), the best trial by the validation score, the selection tests (DSR and SPA) on the
    candidates' validation returns, and the frozen gate: promote only if both pass AND the selected strategy's mean daily net return on the placebo's test segment is positive at ``holdout_alpha``. Returns a function
    ``replicate(seed) -> record``."""
    from .config import build_key, experiment_config
    from .experiments import prepare, run_experiment
    from .optimization import apply_params
    from .selection import deflated, superior_predictive_ability
    from .strategy_search import SearchRunner, _available_kinds, _with_kind, kind_space

    base = experiment_config(spec.base)
    ds0, info = prepare(base, bundle, config, builder, None, info)
    kinds = _available_kinds(spec.kinds, base["dataset"]["target"])
    if not kinds:
        raise ValueError("none of the requested model kinds can run on this target")
    spaces = {k: kind_space(k) for k in kinds}

    def replicate(seed: int) -> dict:
        offset = offset_for(seed, len(ds0.dates), spec.min_shift)
        placebo = placebo_dataset(ds0, offset)
        rng = np.random.default_rng(seed)
        runner = SearchRunner(spec.base, bundle, config, builder, None, f"null-{seed}", None, info)
        runner._datasets[build_key(runner.base)] = placebo                                                           # noqa: SLF001 - the dataset every trial of this replication shares
        rows, columns = [], {}
        total = spec.n_random + spec.n_bayes
        for i in range(total):
            kind = kinds[i % len(kinds)]
            params = _with_kind({p: dist.sample(rng) for p, dist in spaces[kind].items() if not p.startswith(("dataset.", "model.kind"))}, kind)
            score, net, detail = runner.evaluate(params, "random")
            rows.append({"number": i, "kind": kind, "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}, "score": None if score != score else float(score)})
            if net is not None and np.isfinite(score):
                columns[i] = net
        ok = [r for r in rows if r["score"] is not None]
        record = {"seed": int(seed), "offset": int(offset), "candidates": rows, "n_candidates": len(rows), "selected": None, "promoted": False, "reason": "no candidate could be evaluated"}
        if not ok:
            return record
        best = max(ok, key=lambda r: r["score"])
        record["selected"] = {"number": best["number"], "params": best["params"], "score": best["score"]}
        M = pd.DataFrame(columns).dropna() if columns else pd.DataFrame()
        crit = {"alpha": spec.gate["alpha"], "dsr_min": spec.gate["dsr_min"], "pbo_max": 0.5, "min_obs": 60}
        verdicts = {}
        if "dsr" in spec.gate["selection"]:
            verdicts["dsr"] = deflated(M, total, crit) if not M.empty else None
        if "spa" in spec.gate["selection"]:
            verdicts["spa"] = superior_predictive_ability(M, total, crit, None, int(seed), 200) if not M.empty else None
        record["selection"] = {k: (None if v is None else {"status": v.status, "passes": v.statistic.get("passes"), "reason": v.reason}) for k, v in verdicts.items()}
        sel_ok = all(v is not None and v.status == "applied" and v.statistic.get("passes") is True for v in verdicts.values())
        cfg = apply_params(runner.base, {**{k: v for k, v in best["params"].items() if k != "model.kind"}, "model.kind": best["params"]["model.kind"]})
        exp = run_experiment(cfg, dataset=placebo, info=info, registry=None, save=False, explain=False, final_model=False, robustness=False)
        test = exp.net[exp.labels == "test"].dropna()
        t = float(test.mean() / (test.std(ddof=1) / np.sqrt(len(test)))) if len(test) > 5 and test.std(ddof=1) > 0 else float("nan")
        p = float(sps.norm.sf(t)) if t == t else float("nan")
        hold_ok = bool(p == p and p <= spec.gate["holdout_alpha"])
        record.update({"test_mean": float(test.mean()) if len(test) else None, "test_p": None if p != p else p, "selection_passed": bool(sel_ok), "holdout_passed": hold_ok,
                       "promoted": bool(sel_ok and hold_ok), "reason": "promoted by the gate" if sel_ok and hold_ok else ("selection tests not met" if not sel_ok else "holdout test not met")})
        return record

    return replicate


# ------------------------------------------------------------------------------------------------------------------ summary
def binomial_intervals(k: int, n: int, confidence: float = 0.95) -> dict:
    """The exact (Clopper-Pearson) and Wilson intervals for ``k`` successes in ``n`` trials, and the Monte Carlo resolution (the smallest rate one replication can move)."""
    if n <= 0:
        return {"rate": None, "exact": [None, None], "wilson": [None, None], "resolution": None}
    a = 1.0 - confidence
    lo = 0.0 if k == 0 else float(sps.beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    z = float(sps.norm.ppf(1.0 - a / 2.0))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": float(p), "exact": [lo, hi], "wilson": [float(centre - half), float(centre + half)], "resolution": 1.0 / n}


def summarize(spec: NullSpec, records: list[dict]) -> dict:
    """The study's result from its replication records: counts, rate, intervals, and the verdict under the predeclared tolerance and precision."""
    n = len(records)
    k = sum(1 for r in records if r.get("promoted"))
    ci = binomial_intervals(k, n, spec.confidence)
    half = None if ci["rate"] is None else (ci["exact"][1] - ci["exact"][0]) / 2.0
    enough = n >= spec.min_replications and half is not None and half <= spec.precision
    if not enough:
        verdict = (f"{n} replications of the {spec.min_replications} planned as the minimum (half-width {half if half is None else round(half, 3)} against the predeclared precision {spec.precision}): "
                   "this run validates the plumbing only and says nothing about the error rate")
        controlled = None
    elif ci["exact"][1] <= spec.tolerance:
        verdict, controlled = f"the false-promotion rate is {ci['rate']:.3f} (exact {spec.confidence:.0%} interval {ci['exact'][0]:.3f} to {ci['exact'][1]:.3f}), within the predeclared tolerance {spec.tolerance}", True
    elif ci["exact"][0] > spec.tolerance:
        verdict, controlled = f"the false-promotion rate is {ci['rate']:.3f} (interval {ci['exact'][0]:.3f} to {ci['exact'][1]:.3f}), ABOVE the predeclared tolerance {spec.tolerance}", False
    else:
        verdict, controlled = f"the false-promotion rate {ci['rate']:.3f} (interval {ci['exact'][0]:.3f} to {ci['exact'][1]:.3f}) neither meets nor clearly violates the tolerance {spec.tolerance}: inconclusive", None
    return {"replications": n, "planned": spec.replications, "false_promotions": k, "rate": ci["rate"], "exact_interval": ci["exact"], "wilson_interval": ci["wilson"], "confidence": spec.confidence,
            "resolution": ci["resolution"], "tolerance": spec.tolerance, "precision": spec.precision, "min_replications": spec.min_replications, "sufficient": bool(enough), "controlled": controlled, "verdict": verdict,
            "limitations": ["the placebo severs every link between the features and later returns; it is not a null for volatility forecasts", "one circular shift per replication: replications share the same features and are not independent samples of the market",
                            "the interval treats replications as independent draws of the procedure given the data, which understates the uncertainty about the real market", "a smoke run (few replications) validates plumbing only"]}


# ------------------------------------------------------------------------------------------------------------------ persistence and the runner
def register(registry, spec: NullSpec) -> dict:
    """Freeze the specification: store it with its hash before any replication runs. Returns ``{"null_id", "spec_hash"}``."""
    h = spec_hash(spec)
    null_id = f"null-{h[:12]}-{uuid.uuid4().hex[:6]}"
    rc.ensure_study(registry, spec.study_id)
    with registry._lock, registry._conn() as c:                                                    # noqa: SLF001
        c.execute("INSERT INTO null_studies (null_id, study_id, spec, spec_hash, state, planned, done, result, created, finished) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (null_id, spec.study_id, canon(spec.frozen()), h, "frozen", int(spec.replications), 0, None, time.strftime("%Y-%m-%d %H:%M:%S"), None))
    return {"null_id": null_id, "spec_hash": h}


def load(registry, null_id: str) -> dict:
    with registry._conn() as c:                                                                    # noqa: SLF001
        row = c.execute("SELECT * FROM null_studies WHERE null_id=?", (null_id,)).fetchone()
        if row is None:
            raise rc.ControlError(f"unknown null study '{null_id}'")
        reps = [dict(r) for r in c.execute("SELECT * FROM null_replications WHERE null_id=? ORDER BY replication", (null_id,))]
    out = dict(row)
    out["spec"] = json.loads(row["spec"])
    out["result"] = json.loads(row["result"]) if row["result"] else None
    out["replication_records"] = [dict(json.loads(r["detail"] or "{}"), replication=r["replication"], seed=r["seed"], promoted=bool(r["promoted"]), selected=json.loads(r["selected"] or "null"), statistic=r["statistic"])
                                  for r in reps]
    return out


def spec_from(frozen: dict) -> NullSpec:
    d = {k: v for k, v in frozen.items() if k != "version"}
    d["kinds"] = tuple(d["kinds"])
    return NullSpec(**d)


def run(registry, null_id: str, replicate, *, progress=None, stop=None) -> dict:
    """Run the frozen study's missing replications (a resumed study continues where it stopped) with ``replicate(seed) -> record``, persist each as it completes, and the summary at the end. ``stop()`` returning
    True ends the run after the replication it is on; the study is then ``stopped`` and can be resumed."""
    study = load(registry, null_id)
    spec = spec_from(study["spec"])
    if spec_hash(spec) != study["spec_hash"]:
        raise rc.ControlError("the stored specification no longer matches its hash: a null study's design cannot change after it was frozen; register a new one")
    have = {r["replication"] for r in study["replication_records"]}
    todo = [r for r in range(spec.replications) if r not in have]
    with registry._lock, registry._conn() as c:                                                    # noqa: SLF001
        c.execute("UPDATE null_studies SET state='running' WHERE null_id=?", (null_id,))
    stopped = False
    for r in todo:
        if stop is not None and stop():
            stopped = True
            break
        seed = spec.seed + r * 7919
        if progress:
            progress(f"null replication {r + 1} of {spec.replications}")
        with rc.begin(registry, "null_replication", f"null-{null_id}"[:80], op_key=f"{null_id}-{r}", counts=False, seed=seed, detail={"null_id": null_id, "replication": r}) as op:
            rec = replicate(seed)
            op.finish_detail = {"promoted": bool(rec.get("promoted"))}
        with registry._lock, registry._conn() as c:                                                # noqa: SLF001
            c.execute("INSERT OR IGNORE INTO null_replications (null_id, replication, seed, promoted, selected, statistic, detail, created) VALUES (?,?,?,?,?,?,?,?)",
                      (null_id, r, seed, int(bool(rec.get("promoted"))), json.dumps(rec.get("selected"), default=str), rec.get("test_p"), json.dumps(rec, default=str), time.strftime("%Y-%m-%d %H:%M:%S")))
            c.execute("UPDATE null_studies SET done=(SELECT COUNT(*) FROM null_replications WHERE null_id=?) WHERE null_id=?", (null_id, null_id))
    study = load(registry, null_id)
    result = summarize(spec, study["replication_records"])
    result["state"] = "stopped" if stopped else "done"
    with registry._lock, registry._conn() as c:                                                    # noqa: SLF001
        c.execute("UPDATE null_studies SET state=?, result=?, finished=? WHERE null_id=?", ("stopped" if stopped else "done", json.dumps(result, default=str), time.strftime("%Y-%m-%d %H:%M:%S"), null_id))
    return {"null_id": null_id, "spec_hash": study["spec_hash"], **result}


__all__ = ["NullSpec", "binomial_intervals", "load", "offset_for", "placebo_dataset", "register", "replace", "run", "search_pipeline", "spec_from", "spec_hash", "summarize"]
