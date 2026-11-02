"""Constrained multi-objective selection of models and strategies on validation data only (spec v3, Release 5 and 8 follow-up, P1-15).

A search produces many candidates and a leaderboard rewards the lucky ones. This module chooses among them in four steps that are declared before the numbers are seen and recorded with them:

1. **Evidence.** Every candidate is reduced to its **chronologically pooled validation net returns**: the validation days of all its walk-forward folds laid end to end in date order, one return per day. The
   Sharpe ratio, drawdown and the other figures are computed on that one series. They are never the average of per-fold Sharpe ratios, which weights a 20-day fold like a 200-day one, discards the
   dependence between adjacent folds and lets one noisy fold's ratio dominate (``pool_chronological`` refuses folds that disagree about the same day, and ``aggregation`` reports the naive mean beside
   the pooled figure so the difference is visible). Only the validation segment is read: a candidate that carries test or forward dates is refused, and ``scope`` says that nothing protected was read.
2. **Eligibility.** Before anything is ranked, a candidate must pass the declared requirements for **data** (days, coverage, assets), **risk** (drawdown, volatility), **exposure** (gross and net),
   **liquidity** (turnover, measured capacity) and **statistical** evidence (effective observations, a positive net Sharpe, optionally a lower confidence bound). Each gate is reported with its threshold,
   the observed value and a status: ``pass``, ``fail``, ``unavailable`` (the evidence was not produced: for an enforced gate that is not a pass) or ``not_enforced``. A candidate that failed to run is
   ``failed``, one with too little data is ``insufficient``, one that broke a gate is ``ineligible``: four different states. If nothing is eligible nothing is selected.
3. **Ranking and Pareto.** Eligible candidates are ranked by the primary metric (validation net Sharpe unless the policy says otherwise), ties broken by drawdown, turnover and complexity. Beside the
   ranking, a Pareto analysis over the primary metric and the secondary criteria (drawdown, turnover, stability across validation sub-periods, complexity) says which candidates no other candidate beats on
   every criterion, which dominates which, and how the selected one compares with each other (``compare``).
4. **Simplicity.** The simplest eligible candidate that is not meaningfully worse than the highest-ranked one is selected (``simplicity.choose_simplest``): a paired, dependence-aware rule that is a
   heuristic unless the policy declares a margin and asks for the formal non-inferiority test. The report says which of the two it is.

Every threshold is a field of ``SelectionPolicy`` with a documented default; the policy and its hash are stored with the result. The selection never reads the test or forward segments and cannot be
tuned on them. It selects; it does not promote.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import inference as inf
from . import nested as nst
from . import simplicity as sm

VERSION = "model-selection-v1"
SEGMENT = "validation"
PROTECTED = ("test", "forward")
PRIMARY = ("net_sharpe", "net_sortino", "net_calmar", "net_ann_return", "ic")
SECONDARY = ("max_drawdown", "turnover", "instability", "complexity")
EXTRA_OBJECTIVES = ("n_features", "model_complexity", "dependencies", "training_cost", "inference_latency")
DIRECTION = {"net_sharpe": "max", "net_sortino": "max", "net_calmar": "max", "net_ann_return": "max", "ic": "max", "max_drawdown": "min", "turnover": "min", "instability": "min", "complexity": "min",
             **{k: "min" for k in EXTRA_OBJECTIVES}}
STATES = ("eligible", "ineligible", "insufficient", "failed")
ANN = ev.ANN
OPS = {">=": lambda x, t: x >= t, "<=": lambda x, t: x <= t, ">": lambda x, t: x > t}


class SelectionError(ValueError):
    """A selection that cannot be made as declared."""


class SelectionLeakError(SelectionError):
    """A candidate carried dates outside the validation segment: selection reads validation data only."""


@dataclass(frozen=True)
class Eligibility:
    """The requirements a candidate must meet before it is ranked. ``None`` switches a gate off (it is then reported as ``not_enforced``). The defaults are conventions, not tuned to any data set."""

    min_validation_days: int = 126             # data: pooled validation days with a return (about half a trading year)
    min_coverage: float | None = 0.80          # data: share of the validation days on which at least two assets were scored
    min_assets: int | None = 1                 # data: assets that were scored
    max_drawdown: float | None = 0.40          # risk: the largest peak-to-trough fall of the validation equity, as a positive fraction
    max_ann_vol: float | None = None           # risk: annualised volatility of the validation net returns
    max_gross_exposure: float | None = 2.0     # exposure: the largest sum of absolute weights on any validation day
    max_abs_net_exposure: float | None = 1.0   # exposure: the largest absolute sum of weights on any validation day
    max_ann_turnover: float | None = None      # liquidity: annualised one-way turnover on the validation days
    min_capacity_share: float | None = None    # liquidity: the share of the book that one session's participation limit lets through at ``capacity_capital`` (measured, never assumed)
    capacity_capital: float | None = None      # liquidity: the capital level at which ``min_capacity_share`` is read
    min_effective_obs: float | None = 60.0     # statistical: n (1 - r) / (1 + r) for the lag-1 autocorrelation r of the returns (clipped to [0, 0.95])
    min_net_sharpe: float | None = 0.0         # statistical: the validation net Sharpe must be above this
    min_sharpe_lower_bound: float | None = None  # statistical: the lower end of the HAC interval of the Sharpe ratio must be above this (off by default: it is strict on short samples)
    max_label_overlap: int | None = 0          # statistical: the folds whose training labels reached past the date the fit may inform (read from the recorded per-fold lineage; any is a leak)
    sharpe_level: float = 0.95                 # the confidence level of that interval
    missing_evidence: str = "ineligible"       # what an enforced gate does when its evidence was not produced: ineligible | skip (the gate is then reported as unavailable and ignored)

    def __post_init__(self):
        if self.missing_evidence not in ("ineligible", "skip"):
            raise SelectionError("missing_evidence is ineligible or skip")
        if int(self.min_validation_days) < 1 or not 0.0 < float(self.sharpe_level) < 1.0:
            raise SelectionError("min_validation_days >= 1 and sharpe_level in (0, 1)")
        for name in ("min_coverage", "max_drawdown", "max_gross_exposure", "max_abs_net_exposure", "min_capacity_share"):
            v = getattr(self, name)
            if v is not None and not float(v) >= 0.0:
                raise SelectionError(f"{name} must be non-negative")
        if self.min_capacity_share is not None and self.capacity_capital is None:
            raise SelectionError("min_capacity_share is read at a capital level: give capacity_capital")


@dataclass(frozen=True)
class NestedPolicy:
    """The nested estimate of the selection rule: at each cutoff the rule (rank by the primary metric on the inner window, ties to the simplest) chooses among the eligible candidates using only the days before
    the cutoff, and the choice is carried unchanged over the days after it. What the chained days earned is the out-of-sample record of the *rule*."""

    enabled: bool = True
    cutoff_fractions: tuple = (0.4, 0.55, 0.7, 0.85)     # where, as a share of the pooled validation days, the cutoffs fall
    inner_days: int = 126                                # the most recent days before a cutoff the rule looks at
    min_inner: int = 60                                  # fewer inner days than this and the cutoff makes no choice

    def __post_init__(self):
        f = list(self.cutoff_fractions)
        if not f or f != sorted(set(f)) or not 0.0 < f[0] or not f[-1] < 1.0 or int(self.inner_days) < int(self.min_inner) or int(self.min_inner) < 20:
            raise SelectionError("cutoff_fractions are strictly increasing shares in (0, 1), min_inner >= 20 and inner_days >= min_inner")


@dataclass(frozen=True)
class SelectionPolicy:
    primary_metric: str = "net_sharpe"
    secondary: tuple = SECONDARY
    pareto: tuple | None = None                 # the objectives of the Pareto analysis; default: the primary metric and the secondary criteria
    eligibility: Eligibility = field(default_factory=Eligibility)
    simplicity: sm.SimplicityPolicy = field(default_factory=sm.SimplicityPolicy)
    nested: NestedPolicy = field(default_factory=NestedPolicy)
    blocks: int = 4                             # chronological sub-periods of the validation segment used for stability
    min_block_days: int = 20

    def __post_init__(self):
        if self.primary_metric not in PRIMARY:
            raise SelectionError(f"the primary metric must be one of {list(PRIMARY)}")
        bad = [s for s in self.secondary if s not in DIRECTION or s in PRIMARY]
        if bad:
            raise SelectionError(f"secondary criteria are drawn from {[k for k in DIRECTION if k not in PRIMARY]}, not {bad}")
        if self.pareto is not None and [p for p in self.pareto if p not in DIRECTION]:
            raise SelectionError(f"Pareto objectives are drawn from {list(DIRECTION)}")
        if int(self.blocks) < 2 or int(self.min_block_days) < 5:
            raise SelectionError("blocks >= 2 and min_block_days >= 5")

    def objectives(self) -> list[str]:
        return list(self.pareto) if self.pareto is not None else [self.primary_metric, *self.secondary]

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:16]


def policy_from(raw: dict | None) -> SelectionPolicy:
    """A policy from a plain mapping (the API and the command line): unknown keys are refused, nested sections are built from their own mappings."""
    raw = dict(raw or {})
    allowed = set(SelectionPolicy.__dataclass_fields__)
    extra = sorted(set(raw) - allowed)
    if extra:
        raise SelectionError(f"a selection policy has {sorted(allowed)}, not {extra}")
    if "eligibility" in raw:
        e = dict(raw["eligibility"] or {})
        bad = sorted(set(e) - set(Eligibility.__dataclass_fields__))
        if bad:
            raise SelectionError(f"eligibility has {sorted(Eligibility.__dataclass_fields__)}, not {bad}")
        raw["eligibility"] = Eligibility(**e)
    if "simplicity" in raw:
        try:
            raw["simplicity"] = sm.policy_from(raw["simplicity"] or {}) if not isinstance(raw["simplicity"], sm.SimplicityPolicy) else raw["simplicity"]
        except sm.SimplicityError as exc:
            raise SelectionError(str(exc)) from None
    if "nested" in raw and not isinstance(raw["nested"], NestedPolicy):
        n = dict(raw["nested"] or {})
        bad = sorted(set(n) - set(NestedPolicy.__dataclass_fields__))
        if bad:
            raise SelectionError(f"nested has {sorted(NestedPolicy.__dataclass_fields__)}, not {bad}")
        if "cutoff_fractions" in n:
            n["cutoff_fractions"] = tuple(n["cutoff_fractions"])
        raw["nested"] = NestedPolicy(**n)
    for k in ("secondary", "pareto"):
        if raw.get(k) is not None:
            raw[k] = tuple(raw[k])
    return SelectionPolicy(**raw)


# ------------------------------------------------------------------------------------------------------------------------------ the evidence: pooled chronological validation returns
def pool_chronological(folds) -> pd.Series:
    """The returns of several folds laid end to end in date order. A date reported by two folds must carry the same return; if it does not, the folds disagree about the same observation and the pooling is
    refused. This is the aggregate that Sharpe, drawdown and the rest are computed on: one series, not a ratio per fold."""
    parts = [pd.Series(f).astype(float) for f in folds]
    if not parts:
        raise SelectionError("no fold returns to pool")
    s = pd.concat(parts)
    if s.index.has_duplicates:
        spread = s.groupby(level=0).agg(lambda v: float(np.nanmax(v) - np.nanmin(v)))
        if (spread > 1e-12).any():
            raise SelectionError(f"two folds report different returns for the same day (first: {spread[spread > 1e-12].index[0]}): the folds overlap and disagree")
        s = s[~s.index.duplicated(keep="first")]
    return s.sort_index()


def naive_fold_sharpe(folds) -> float | None:
    """The average of the per-fold Sharpe ratios: reported only to show how far it is from the pooled figure; it is never used to select."""
    vals = [ev.sharpe_of(pd.Series(f).dropna()) for f in folds if len(pd.Series(f).dropna()) > 2]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def max_drawdown(r: pd.Series) -> float:
    """The largest peak-to-trough fall of the compounded equity, as a positive fraction."""
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float(-(eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")


def block_sharpes(r: pd.Series, blocks: int = 4, min_days: int = 20) -> list[float]:
    """The Sharpe ratio of each of ``blocks`` consecutive chronological blocks of the pooled series (blocks shorter than ``min_days`` are left out)."""
    vals = [ev.sharpe_of(pd.Series(b)) for b in np.array_split(r.dropna().to_numpy(), int(blocks)) if len(b) >= int(min_days)]
    return [v for v in vals if math.isfinite(v)]


def instability(r: pd.Series, blocks: int = 4, min_days: int = 20) -> float | None:
    """The spread (ddof=1) of the block Sharpe ratios: how much the result depends on which part of the validation segment is looked at. The definition of the Overfitting Center's ``instability``."""
    b = block_sharpes(r, blocks, min_days)
    return float(np.std(b, ddof=1)) if len(b) >= 2 else None


def effective_n(r: pd.Series) -> float:
    """``n (1 - r) / (1 + r)`` with ``r`` the lag-1 autocorrelation clipped to [0, 0.95]: the number of independent observations a serially dependent series is worth (negative dependence is not credited)."""
    v = r.dropna().to_numpy(dtype=float)
    if len(v) < 3 or np.std(v) == 0:
        return float(len(v))
    rho = float(np.clip(np.corrcoef(v[:-1], v[1:])[0, 1], 0.0, 0.95)) if len(v) > 3 else 0.0
    return float(len(v) * (1.0 - rho) / (1.0 + rho))


def metrics_of(r: pd.Series, policy: SelectionPolicy | None = None, *, ic_mean: float | None = None) -> dict:
    """Every figure of one pooled validation series."""
    policy = policy or SelectionPolicy()
    x = r.dropna()
    n = int(len(x))
    out = {"n_obs": n, "first": str(x.index[0]) if n else None, "last": str(x.index[-1]) if n else None}
    if n < 3:
        return {**out, "net_sharpe": None, "net_sortino": None, "net_calmar": None, "net_ann_return": None, "ann_vol": None, "max_drawdown": None, "effective_n": float(n), "instability": None, "hit_rate": None, "ic": ic_mean}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    down = float(np.sqrt(np.mean(np.minimum(x.to_numpy(dtype=float), 0.0) ** 2)))
    dd = max_drawdown(x)
    sharpe = ev.sharpe_of(x)
    out.update(net_sharpe=sharpe if math.isfinite(sharpe) else None, net_ann_return=mean * ANN, ann_vol=sd * math.sqrt(ANN), max_drawdown=dd, hit_rate=float((x > 0).mean()),
               net_sortino=mean / down * math.sqrt(ANN) if down > 0 else None, net_calmar=mean * ANN / dd if dd > 0 else None, effective_n=effective_n(x),
               instability=instability(x, policy.blocks, policy.min_block_days), ic=ic_mean)
    return out


def series_hash(r: pd.Series) -> str:
    """A fingerprint of the dates and values a candidate was judged on."""
    x = r.dropna()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x.index.values.astype("datetime64[ns]").astype("int64")).tobytes() if isinstance(x.index, pd.DatetimeIndex) else str(list(x.index)).encode())
    h.update(np.ascontiguousarray(x.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------------------------ the outer windows of a candidate, and the independent assertion on them
def fold_lineage(fold_list: list[dict]) -> dict:
    """The outer windows a candidate was produced by: for each fitted walk-forward fold, its training window, how many rows were fitted (and their hash), the date the fit may inform decisions from (the fit
    cutoff), the last date any training label reached and the block it predicted. The assertion is made here, from those recorded facts, and not taken from the run: a fold whose training labels reached
    past its fit cutoff is a violation, because the model then learned from outcomes that its own first decision could not have known."""
    rows, bad = [], []
    for f in fold_list or []:
        if f.get("skipped") or not f.get("fit_cutoff") or not f.get("label_end_max"):
            continue
        rows.append({k: f.get(k) for k in ("fold", "train_start", "train_end", "n_train", "train_rows_hash", "train_row_ranges", "fit_cutoff", "label_end_max", "test_start", "test_end", "n_test", "test_row_ranges")})
        if pd.Timestamp(f["label_end_max"]) > pd.Timestamp(f["fit_cutoff"]):
            bad.append({"fold": f.get("fold"), "label_end_max": f["label_end_max"], "fit_cutoff": f["fit_cutoff"]})
    return {"checked": bool(rows), "n_folds": len(rows), "folds": rows, "violations": bad,
            "rule": "label_end_max <= fit_cutoff for every fitted fold", "note": None if rows else "the experiment recorded no fold lineage (it was saved before that was kept)"}


def audit_folds(ds, fold_list: list[dict], horizon: int) -> dict:
    """Re-derive the facts of each fold from the rows it recorded, without trusting the numbers recorded beside them: the rows must match their hash and count; the last day any training label reaches
    (``position + lag + horizon``) must not be after the fit cutoff; no training row may be a test row; every test row must be at or after the fit cutoff and after every training row. A fold with no
    recorded rows (saved before they were kept, or too scattered to store as runs) is counted as not audited, never as clean."""
    from . import intervals as iv
    from .training import expand_ranges, row_hash

    pos_all = ds.positions()
    folds, bad, audited = [], [], 0
    for f in fold_list or []:
        if f.get("skipped") or not f.get("fit_cutoff"):
            continue
        if f.get("train_row_ranges") is None or f.get("test_row_ranges") is None:
            folds.append({"fold": f.get("fold"), "audited": False, "reason": "no row ranges were recorded for this fold"})
            continue
        tr, te = expand_ranges(f["train_row_ranges"]), expand_ranges(f["test_row_ranges"])
        cut = int(ds.dates.get_indexer([pd.Timestamp(f["fit_cutoff"])])[0])
        problems = []
        if f.get("train_rows_hash") and row_hash(tr) != f["train_rows_hash"]:
            problems.append("the recorded training rows do not match their hash")
        if f.get("n_train") is not None and len(tr) != int(f["n_train"]):
            problems.append(f"{len(tr)} training rows recorded against a count of {f['n_train']}")
        end_pos = int(iv.label_ends(pos_all[tr], ds.lag, horizon).max()) if len(tr) else -1
        if cut < 0 or end_pos > cut:
            problems.append(f"a training label reaches position {end_pos}, after the fit cutoff at {cut}")
        if len(te) and len(tr) and (np.intersect1d(tr, te).size or pos_all[tr].max() >= pos_all[te].min()):
            problems.append("a training row is a test row, or is dated at or after the first test row")
        if len(te) and cut >= 0 and pos_all[te].min() < cut:
            problems.append("a test row is dated before the fit cutoff")
        audited += 1
        label_end = None if end_pos < 0 else str(ds.dates[min(end_pos, len(ds.dates) - 1)].date())
        folds.append({"fold": f.get("fold"), "audited": True, "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)), "label_end_recomputed": label_end, "fit_cutoff": f["fit_cutoff"], "ok": not problems})
        bad += [{"fold": f.get("fold"), "problem": p} for p in problems]
    return {"audited": audited, "folds": folds, "violations": bad, "rule": "re-derived from the recorded rows: hash, count, last label day <= fit cutoff, train and test rows disjoint and ordered"}


def audit_lineage(registry, exp_id: str) -> dict:
    """``audit_folds`` for a saved experiment, against its own saved panel."""
    from . import experiments as xp
    from .targets import parse_target

    _, ds, train, _ = xp.load_run(registry, exp_id)
    return audit_folds(ds, train.folds, int(parse_target(train.target)[1]))


# ------------------------------------------------------------------------------------------------------------------------------ eligibility
def _gate(out: list, name: str, family: str, observed, threshold, op: str, missing: str, note: str = "") -> None:
    if threshold is None:
        status = "not_enforced"
    elif observed is None or (isinstance(observed, float) and not math.isfinite(observed)):
        status = "unavailable"
    else:
        status = "pass" if OPS[op](observed, threshold) else "fail"
    blocks = status == "fail" or (status == "unavailable" and missing == "ineligible")
    out.append({"gate": name, "family": family, "operator": op, "threshold": threshold, "observed": observed, "status": status, "blocks": bool(blocks), "note": note})


def check_eligibility(c: dict, m: dict, pol: Eligibility) -> list[dict]:
    """The gates of one candidate. ``c`` carries the evidence a series cannot give (coverage, assets, exposure, turnover, liquidity); ``m`` is ``metrics_of`` its pooled returns."""
    g: list = []
    miss = pol.missing_evidence
    _gate(g, "min_validation_days", "data", m["n_obs"], pol.min_validation_days, ">=", miss)
    _gate(g, "min_coverage", "data", c.get("coverage"), pol.min_coverage, ">=", miss, "the share of validation days on which at least two assets were scored")
    _gate(g, "min_assets", "data", c.get("n_assets"), pol.min_assets, ">=", miss)
    _gate(g, "max_drawdown", "risk", m["max_drawdown"], pol.max_drawdown, "<=", miss)
    _gate(g, "max_ann_vol", "risk", m["ann_vol"], pol.max_ann_vol, "<=", miss)
    ex = c.get("exposure") or {}
    _gate(g, "max_gross_exposure", "exposure", ex.get("gross_max"), pol.max_gross_exposure, "<=", miss)
    _gate(g, "max_abs_net_exposure", "exposure", ex.get("net_abs_max"), pol.max_abs_net_exposure, "<=", miss)
    _gate(g, "max_ann_turnover", "liquidity", c.get("turnover"), pol.max_ann_turnover, "<=", miss)
    liq = c.get("liquidity") or {}
    share = None
    if pol.min_capacity_share is not None and liq.get("status") == "measured":
        share = next((float(lv["fill_share"]) for lv in liq.get("levels", []) if abs(float(lv["capital"]) - float(pol.capacity_capital)) <= 1e-9 * max(1.0, abs(float(pol.capacity_capital))) and lv.get("fill_share") is not None), None)
    _gate(g, "min_capacity_share", "liquidity", share, pol.min_capacity_share, ">=", miss,
          "capacity is measured or it is unavailable: no volume is not unlimited liquidity" + ("" if liq.get("status") == "measured" or pol.min_capacity_share is None else f" (status: {liq.get('status', 'not recorded')})"))
    _gate(g, "min_effective_obs", "statistical", m["effective_n"], pol.min_effective_obs, ">=", miss)
    _gate(g, "min_net_sharpe", "statistical", m["net_sharpe"], pol.min_net_sharpe, ">", miss, "the validation net Sharpe must be above this" if pol.min_net_sharpe is not None else "")
    lower = None
    if pol.min_sharpe_lower_bound is not None and c.get("returns") is not None:
        res = inf.sharpe_inference(c["returns"], level=float(pol.sharpe_level), n_boot=200)["hac"]
        lower = None if res.eligibility != "applied" or res.interval is None else float(res.interval[0])
    _gate(g, "min_sharpe_lower_bound", "statistical", lower, pol.min_sharpe_lower_bound, ">=", miss, f"the lower end of the {pol.sharpe_level:.0%} HAC interval of the Sharpe ratio")
    lin = c.get("lineage") or {}
    _gate(g, "max_label_overlap", "statistical", len(lin["violations"]) if lin.get("checked") else None, pol.max_label_overlap, "<=", miss,
          "folds whose training labels ended after the date their fit may inform; read from the lineage the experiment recorded, so one saved before that was recorded is unavailable")
    return g


# ------------------------------------------------------------------------------------------------------------------------------ Pareto
def dominates(a: dict, b: dict, objectives: list[str]) -> bool:
    """``a`` is at least as good as ``b`` on every objective and strictly better on one (direction per objective from ``DIRECTION``)."""
    strictly = False
    for o in objectives:
        x, y = a[o], b[o]
        better = x > y if DIRECTION[o] == "max" else x < y
        worse = x < y if DIRECTION[o] == "max" else x > y
        if worse:
            return False
        strictly = strictly or better
    return strictly


def pareto_analysis(values: dict, objectives: list[str]) -> dict:
    """The non-dominated sorting of ``values`` (``id -> {objective: number or None}``). An objective that any candidate lacks is dropped from the comparison and listed with the reason: a missing value is
    neither the best nor the worst. ``ranks`` is the non-dominated level (1 is the Pareto front); ``dominated_by`` names who beats each candidate on every kept objective."""
    ids = list(values)
    dropped = [{"objective": o, "reason": "not available for " + ", ".join(i for i in ids if values[i].get(o) is None)} for o in objectives if any(values[i].get(o) is None for i in ids)]
    kept = [o for o in objectives if all(values[i].get(o) is not None for i in ids)]
    if not ids or not kept:
        return {"objectives": kept, "dropped": dropped, "front": [], "ranks": {}, "dominated_by": {}, "eligibility": "insufficient", "reason": "no objective is available for every candidate"}
    v = {i: {o: float(values[i][o]) for o in kept} for i in ids}
    dom_by = {i: [j for j in ids if j != i and dominates(v[j], v[i], kept)] for i in ids}
    ranks, remaining, level = {}, set(ids), 1
    while remaining:
        front = sorted(i for i in remaining if not any(j in remaining for j in dom_by[i]))
        for i in front:
            ranks[i] = level
        remaining -= set(front)
        level += 1
    return {"objectives": kept, "directions": {o: DIRECTION[o] for o in kept}, "dropped": dropped, "front": sorted(i for i in ids if ranks[i] == 1), "ranks": ranks, "dominated_by": dom_by,
            "eligibility": "applied"}


def compare(a_id: str, b_id: str, values: dict, objectives: list[str]) -> dict:
    """The objective-by-objective comparison of two candidates and what it means for dominance."""
    rows, a_better, b_better = [], [], []
    for o in objectives:
        x, y = values[a_id].get(o), values[b_id].get(o)
        if x is None or y is None:
            rows.append({"objective": o, "direction": DIRECTION[o], a_id: x, b_id: y, "better": None, "reason": "not available for both"})
            continue
        win = None if x == y else (a_id if (x > y) == (DIRECTION[o] == "max") else b_id)
        rows.append({"objective": o, "direction": DIRECTION[o], a_id: x, b_id: y, "better": win})
        if win == a_id:
            a_better.append(o)
        elif win == b_id:
            b_better.append(o)
    verdict = ("tied on every available objective" if not a_better and not b_better else f"{a_id} dominates {b_id}" if a_better and not b_better else f"{b_id} dominates {a_id}" if b_better and not a_better else
               "neither dominates: each is better on some objective")
    return {"a": a_id, "b": b_id, "objectives": rows, "better_a": a_better, "better_b": b_better, "verdict": verdict}


# ------------------------------------------------------------------------------------------------------------------------------ the selection
def _validation_only(c: dict) -> None:
    labels = c.get("labels")
    if labels is None or c.get("returns") is None:
        return
    seen = pd.Series(labels).reindex(pd.Series(c["returns"]).dropna().index)
    bad = seen[seen.isin(PROTECTED)]
    if len(bad):
        raise SelectionLeakError(f"candidate '{c['id']}' carries {len(bad)} return dates labelled {sorted(set(bad))}: selection reads the {SEGMENT} segment only. Give it the validation rows")


def evaluate_candidate(c: dict, policy: SelectionPolicy) -> dict:
    """One candidate's card: its metrics on the pooled validation series, its gates and its state."""
    cid = str(c["id"])
    card = {"id": cid, "label": c.get("label") or cid, "kind": c.get("kind"), "params": c.get("params") or {}}
    if c.get("error"):
        return {**card, "state": "failed", "reason": str(c["error"]), "metrics": None, "gates": [], "complexity": c.get("complexity")}
    _validation_only(c)
    folds = c.get("folds")
    r = pool_chronological(folds) if folds else (None if c.get("returns") is None else pd.Series(c["returns"]).astype(float).sort_index())
    if r is None or len(r.dropna()) < 3:
        return {**card, "state": "insufficient", "reason": "no usable validation returns", "metrics": None, "gates": [], "complexity": c.get("complexity")}
    c = {**c, "returns": r}
    ic = c.get("ic")
    m = metrics_of(r, policy, ic_mean=None if ic is None and c.get("ic_mean") is None else (float(pd.Series(ic).mean()) if ic is not None else float(c["ic_mean"])))
    if c.get("turnover") is not None:
        m["turnover"] = float(c["turnover"])
    else:
        m["turnover"] = None
    cx = dict(c.get("complexity") or {})
    if cx and cx.get("stability") is None:
        cx["stability"] = m["instability"]
    gates = check_eligibility(c, m, policy.eligibility)
    primary = m.get(policy.primary_metric)
    reasons = []
    if m["n_obs"] < policy.eligibility.min_validation_days:
        state = "insufficient"
        reasons.append(f"{m['n_obs']} validation days; at least {policy.eligibility.min_validation_days} are needed")
    else:
        blocked = [x for x in gates if x["blocks"]]
        state = "ineligible" if blocked else "eligible"
        reasons += [f"{x['gate']}: " + (f"{x['observed']:.4g} against {x['threshold']:g}" if x["status"] == "fail" else "the evidence was not produced") for x in blocked]
    if state == "eligible" and (primary is None or not math.isfinite(float(primary))):
        state, reasons = "ineligible", [f"the primary metric {policy.primary_metric} is undefined for this candidate"]
    agg = {"method": "pooled chronological validation returns", "folds": None if not folds else len(folds), "naive_mean_of_fold_sharpes": naive_fold_sharpe(folds) if folds else None}
    idx = r.dropna().index
    rows = {"index_hash": hashlib.sha256((np.ascontiguousarray(idx.values.astype("datetime64[ns]").astype("int64")).tobytes() if isinstance(idx, pd.DatetimeIndex) else str(list(idx)).encode())).hexdigest()[:16],
            "n": int(len(idx)), "first": str(idx[0]) if len(idx) else None, "last": str(idx[-1]) if len(idx) else None}
    return {**card, "state": state, "reason": "; ".join(reasons) if reasons else None, "metrics": m, "gates": gates, "primary": primary, "series_hash": series_hash(r), "aggregation": agg, "complexity": cx or None,
            "unevaluated_gates": [x["gate"] for x in gates if x["status"] == "unavailable"], "rows": rows, "lineage": c.get("lineage"), "_returns": r, "_ic": c.get("ic"), "ignored_rows": c.get("ignored_rows")}


def _sharpe_of_array(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    sd = v.std(ddof=1) if len(v) > 2 else float("nan")
    return float(v.mean() / sd * math.sqrt(ANN)) if np.isfinite(sd) and sd > 1e-12 else float("-inf")


def nested_estimate(ranked: list[dict], policy: SelectionPolicy, order_keys: dict) -> dict:
    """Run the selection rule through the nested procedure on the eligible candidates' pooled validation returns. ``ranked`` are the eligible candidate cards (they carry ``_returns``); ``order_keys`` gives
    each id's simplicity order key, so that an exact tie goes to the simplest, as in the selection itself. The result is a statement about the RULE: what choosing by the primary metric on the days known at
    each cutoff earned on the days after it, set beside what the candidates earned on those days and what the finally selected one would have earned (it was chosen with all the days, so it is optimistic)."""
    pol = policy.nested
    base = {"version": nst.VERSION, "policy": asdict(pol), "rule": f"rank by {policy.primary_metric} on the inner window; ties to the simplest"}
    if not pol.enabled:
        return {**base, "eligibility": "not_applicable", "reason": "the nested estimate is switched off by the policy"}
    if len(ranked) < 2:
        return {**base, "eligibility": "not_applicable", "reason": "a rule that chooses among fewer than two candidates has nothing to choose"}
    if policy.primary_metric != "net_sharpe":
        return {**base, "eligibility": "not_applicable", "reason": f"the nested estimate is run for the net Sharpe ranking; the primary metric here is {policy.primary_metric}"}
    names = [c["id"] for c in sorted(ranked, key=lambda c: (order_keys[c["id"]], c["id"]))]          # declared order: the simplest first, which is where a tie goes
    series = {c["id"]: c["_returns"].dropna() for c in ranked}
    dates = sorted(set.intersection(*[set(s.index) for s in series.values()]))
    if len(dates) < pol.min_inner + 20:
        return {**base, "eligibility": "insufficient", "reason": f"{len(dates)} days that every eligible candidate has; the rule needs at least {pol.min_inner + 20}"}
    idx = pd.DatetimeIndex(dates)
    cands = {n: pd.DataFrame({"r": series[n].reindex(idx)}) for n in names}
    cut = []
    for f in pol.cutoff_fractions:
        d = idx[max(int(len(idx) * f) - 1, 0)]
        if not cut or d > cut[-1]:
            cut.append(d)
    res = nst.select_as_of(cands, pd.DataFrame({"r": np.zeros(len(idx))}, index=idx), pd.Series(idx, index=idx), cut, inner_days=int(pol.inner_days), min_inner=int(pol.min_inner), until=idx[-1],
                           score=lambda s, t: s.iloc[:, 0], aggregate=_sharpe_of_array)
    audit = nst.audit(res, pd.Series(idx, index=idx))
    chain = res["chain"]["r"].dropna()
    if res["eligibility"] != "applied" or len(chain) < 20:
        return {**base, "eligibility": "insufficient", "reason": res.get("reason") or f"{len(chain)} chained days: fewer than 20", "steps": res["steps"], "audit": audit}
    outer = chain.index
    on_outer = {n: _sharpe_of_array(series[n].reindex(outer).to_numpy()) for n in names}
    finite = {n: v for n, v in on_outer.items() if math.isfinite(v)}
    return {**base, "eligibility": "applied", "cutoffs": [str(c.date()) for c in cut], "steps": res["steps"], "choices": res["choices"], "n_selected": res["n_selected"], "chain_days": int(len(chain)),
            "chain_sharpe": _sharpe_of_array(chain.to_numpy()), "candidates_on_outer_days": on_outer, "median_candidate": float(np.median(list(finite.values()))), "best_in_hindsight": float(max(finite.values())),
            "audit": audit, "outer_first": str(outer[0].date()), "outer_last": str(outer[-1].date()),
            "reading": "the chained Sharpe is what choosing by this rule earned on days after each choice; the best candidate in hindsight and the finally selected one were chosen with those days in view",
            "notes": ["each inner window holds only days up to the cutoff", "the outer windows are disjoint and follow their cutoffs", "this is the record of the ranking rule; the simplicity step is not part of it"]}


def predeclared(policy: SelectionPolicy, plan: dict | None) -> dict:
    """Was this policy written into a frozen evaluation plan before the results existed? ``plan`` is ``research_controls.plan_state`` of the plan. A policy that was only supplied with the data is not evidence
    that it was chosen beforehand, and the result says so."""
    if plan is None:
        return {"predeclared": None, "note": "no evaluation plan was given, so nothing shows the policy was fixed before the results were seen"}
    declared = ((plan.get("content") or {}).get("decision_policy") or {}).get("selection")
    try:
        same = declared is not None and policy_from(declared).hash() == policy.hash()
    except SelectionError:
        same = False
    ok = bool(same and plan.get("state") in ("frozen", "consumed"))
    return {"predeclared": ok, "plan_id": plan.get("plan_id"), "plan_state": plan.get("state"),
            "note": "the policy is in the frozen plan" if ok else "the policy was supplied with the results: it is not evidence that it was chosen before them"}


def select(candidates: list[dict], policy: SelectionPolicy | None = None, plan: dict | None = None) -> dict:
    """Choose among candidates. Each is ``{"id", "label", "kind", "params", "returns" (the validation net returns) or "folds" (their per-fold pieces), "labels" (optional: the segment of each date),
    "ic" (optional daily rank IC), "turnover" (annual), "coverage", "n_assets", "exposure": {"gross_max", "net_abs_max"}, "liquidity": {"status", "levels"}, "complexity": simplicity.complexity_vector(...),
    "error" (the candidate could not be run)}``. Returns every candidate's card, the ranking, the Pareto analysis, the simplicity decision and the selection. Nothing is promoted."""
    policy = policy or SelectionPolicy()
    ids = [str(c["id"]) for c in candidates]
    if len(set(ids)) != len(ids):
        raise SelectionError("candidate ids must be distinct")
    cards = [evaluate_candidate(c, policy) for c in candidates]
    eligible = [c for c in cards if c["state"] == "eligible"]
    skey = lambda c: sm.order_key(c["complexity"], policy.simplicity) if c.get("complexity") else (math.inf,)          # noqa: E731
    ranked = sorted(eligible, key=lambda c: (-float(c["primary"]), c["metrics"]["max_drawdown"], math.inf if c["metrics"]["turnover"] is None else c["metrics"]["turnover"], skey(c), c["id"]))
    ranking = [{"rank": i + 1, "id": c["id"], "label": c["label"], "primary": c["primary"], "max_drawdown": c["metrics"]["max_drawdown"], "turnover": c["metrics"]["turnover"], "instability": c["metrics"]["instability"]}
               for i, c in enumerate(ranked)]
    order = {tuple(k): i + 1 for i, k in enumerate(sorted({skey(c) for c in eligible}))}
    values = {}
    for c in eligible:
        cx = c.get("complexity") or {}
        values[c["id"]] = {policy.primary_metric: c["primary"], "net_sharpe": c["metrics"]["net_sharpe"], "net_sortino": c["metrics"]["net_sortino"], "net_calmar": c["metrics"]["net_calmar"],
                           "net_ann_return": c["metrics"]["net_ann_return"], "ic": c["metrics"]["ic"], "max_drawdown": c["metrics"]["max_drawdown"], "turnover": c["metrics"]["turnover"],
                           "instability": c["metrics"]["instability"], "complexity": float(order[tuple(skey(c))]) if cx else None, **{k: cx.get(k) for k in EXTRA_OBJECTIVES if k != "inference_latency"},
                           "inference_latency": cx.get("inference_latency")}
    objectives = policy.objectives()
    pareto = pareto_analysis({i: {o: v.get(o) for o in objectives} for i, v in values.items()}, objectives) if eligible else {"objectives": [], "dropped": [], "front": [], "ranks": {}, "dominated_by": {},
                                                                                                                                     "eligibility": "insufficient", "reason": "no eligible candidate"}
    simplicity = None
    selected = ranked[0]["id"] if ranked else None
    if ranked and policy.simplicity.enabled and len(ranked) >= 2:
        simplicity = sm.choose_simplest([{"id": c["id"], "score": c["primary"], "complexity": c.get("complexity") or sm.complexity_vector("?"), "returns": c["_returns"], "ic": c.get("_ic")} for c in ranked],
                                        policy.simplicity, metric=policy.primary_metric, horizon=1)
        if simplicity["eligibility"] == "applied":
            selected = simplicity["selected"]
    nested_out = nested_estimate(ranked, policy, {c["id"]: skey(c) for c in ranked}) if ranked else {"eligibility": "not_applicable", "reason": "no eligible candidate"}
    if nested_out.get("eligibility") == "applied" and selected is not None:
        nested_out["selected_on_outer_days"] = nested_out["candidates_on_outer_days"].get(selected)
    versus = []
    if selected is not None and pareto["eligibility"] == "applied":
        versus = [compare(selected, c["id"], values, pareto["objectives"]) for c in ranked if c["id"] != selected]
    counts = {s: sum(1 for c in cards if c["state"] == s) for s in STATES}
    evidence_rows = {c["rows"]["index_hash"]: [str(d.date()) if hasattr(d, "date") else str(d) for d in c["_returns"].dropna().index] for c in cards if c.get("rows")}
    out_cards = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cards]
    return {"version": VERSION, "policy": asdict(policy), "policy_hash": policy.hash(), "primary_metric": policy.primary_metric,
            "predeclaration": predeclared(policy, plan),
            "scope": {"segments": [SEGMENT], "protected_segments_read": False, "aggregation": "pooled chronological validation returns: one series per candidate, never an average of per-fold Sharpe ratios",
                      "note": "a candidate that carries test or forward dates is refused; nothing here was tuned on them"},
            "n_candidates": len(cards), "counts": counts, "candidates": out_cards, "evidence_rows": evidence_rows, "nested": nested_out, "ranking": ranking, "highest_ranked": ranked[0]["id"] if ranked else None, "pareto": pareto, "against_selected": versus,
            "simplicity": simplicity, "selected": selected, "selection_status": "selected" if selected is not None else "no_eligible_candidate",
            "changed_by_simplicity": bool(selected is not None and ranked and selected != ranked[0]["id"]),
            "claims": {"selection": "a heuristic for choosing among candidates on validation data" if (simplicity or {}).get("claim", {}).get("kind") != "formal_non_inferiority" else
                       "a formal non-inferiority test against the highest-ranked candidate within a declared margin, on validation data",
                       "statistical_equivalence": "not claimed" if not (simplicity or {}).get("equivalence_claimed") else "shown within the declared margin for the selected candidate (see simplicity.table)",
                       "out_of_sample": "nothing here is a statement about test, forward or live performance"},
            "reason": None if selected is not None else "no candidate passed the eligibility requirements; nothing is selected rather than the least bad"}


# ------------------------------------------------------------------------------------------------------------------------------ candidates from experiments and search trials
def _inference_ms(fold_list: list) -> float | None:
    """Milliseconds to score 1000 rows, from the time each walk-forward fold's model took to predict its block (``None`` for an experiment saved before that was timed: not measured is not free)."""
    done = [f for f in fold_list if not f.get("skipped") and f.get("predict_seconds") is not None and f.get("n_test")]
    rows = sum(int(f["n_test"]) for f in done)
    return None if not done or rows <= 0 else float(sum(float(f["predict_seconds"]) for f in done) / rows * 1e6)


def _liquidity(report: dict) -> dict:
    ex = report.get("execution") or {}
    cap = ex.get("capacity") or {}
    return {"status": "measured" if str(cap.get("status", "")).startswith("measured") else "not_established", "levels": cap.get("levels") or [], "mode": ex.get("mode", "vectorized_legacy")}


def _complexity(kind: str, params: dict, report: dict) -> dict:
    res = (report.get("model_extras") or {}).get("resources") or {}
    return sm.complexity_vector(kind, params, n_features=(report.get("features") or {}).get("n"), n_params=res.get("n_params"), train_seconds=res.get("train_seconds"),
                                inference_ms=_inference_ms((report.get("folds") or {}).get("list") or []))


def evidence_from_run(report: dict, net: pd.Series, labels: pd.Series, turnover: pd.Series, weights: pd.DataFrame | None, scores: pd.DataFrame | None, *, n_features=None, train_seconds=None,
                      fold_list=None) -> dict:
    """What a selection needs from one run besides its returns, restricted to the validation days: turnover, coverage, assets, exposure, liquidity and the complexity inputs."""
    on = (labels.reindex(net.index) == SEGMENT).to_numpy()
    days = net.index[on]
    out: dict = {"turnover": float(turnover.reindex(days).mean() * ANN) if len(days) else None}
    if scores is not None:
        s = scores.reindex(days)
        scored = s.notna().sum(axis=1)
        out.update(coverage=float((scored >= min(2, max(1, scores.shape[1]))).mean()) if len(days) else None, n_assets=int(s.notna().any(axis=0).sum()))
    if weights is not None and len(days):
        w = weights.reindex(days).dropna(how="all")
        if len(w):
            gross, net_w = w.abs().sum(axis=1), w.sum(axis=1)
            out["exposure"] = {"gross_mean": float(gross.mean()), "gross_max": float(gross.max()), "net_mean": float(net_w.mean()), "net_abs_max": float(net_w.abs().max())}
    out["liquidity"] = _liquidity(report)
    res = (report.get("model_extras") or {}).get("resources") or {}
    out["resources"] = {"n_features": n_features if n_features is not None else (report.get("features") or {}).get("n"), "n_params": res.get("n_params"),
                        "train_seconds": train_seconds if train_seconds is not None else res.get("train_seconds"), "inference_ms": _inference_ms(fold_list if fold_list is not None else (report.get("folds") or {}).get("list") or [])}
    ic = ((report.get("ic") or {}).get("by_segment") or {}).get(SEGMENT) or {}
    out["ic_mean"] = ic.get("mean")
    out["lineage"] = fold_lineage(fold_list if fold_list is not None else (report.get("folds") or {}).get("list") or [])
    return out


def candidate_from_experiment(registry, exp_id: str, *, until=None, with_ic: bool = False, audit: bool = False) -> dict:
    """A saved experiment as a candidate: its validation rows only. Test and forward rows of a finalised experiment are counted in ``ignored_rows`` and never enter any figure; ``until`` ends the evidence
    at a date (everything later is ignored too)."""
    rep = registry.report(exp_id)
    d = registry.root / "experiments" / exp_id
    ret = pd.read_parquet(d / "returns.parquet")
    seg = ret["segment"]
    on = seg == SEGMENT
    if until is not None:
        on &= ret.index <= pd.Timestamp(until)
    val = ret[on]
    ignored = {str(k): int(v) for k, v in seg[~on].value_counts().items()}
    preds = pd.read_parquet(d / "predictions.parquet")
    pv = preds[preds["segment"] == SEGMENT]
    scores = pv["score"].unstack(level=1) if len(pv) else None
    pos = d / "positions.parquet"
    weights = pd.read_parquet(pos) if pos.exists() else None
    cfg = rep.get("config") or {}
    kind, params = cfg["model"]["kind"], cfg["model"].get("params") or {}
    ev_ = evidence_from_run(rep, val["net"], seg[on], val["turnover"], weights, scores)
    cand = {"id": exp_id, "label": rep.get("name") or exp_id, "kind": kind, "params": params, "returns": val["net"].dropna(), "labels": seg[on], "turnover": ev_["turnover"], "coverage": ev_.get("coverage"),
            "n_assets": ev_.get("n_assets"), "exposure": ev_.get("exposure"), "liquidity": ev_["liquidity"], "ic_mean": ev_.get("ic_mean"), "ignored_rows": ignored, "lineage": ev_.get("lineage"),
            "complexity": sm.complexity_vector(kind, params, n_features=ev_["resources"]["n_features"], n_params=ev_["resources"]["n_params"], train_seconds=ev_["resources"]["train_seconds"],
                                               inference_ms=ev_["resources"]["inference_ms"])}
    if audit and cand["lineage"] is not None:                                                    # re-derive the folds' facts from the rows they recorded (loads the saved panel)
        a = audit_lineage(registry, exp_id)
        cand["lineage"] = {**cand["lineage"], "audit": a, "violations": cand["lineage"]["violations"] + a["violations"], "checked": bool(cand["lineage"]["checked"] and a["audited"])}
    if with_ic:
        from . import comparison as cmp_

        series, _, _ = cmp_._series_saved(registry, exp_id)                                              # noqa: SLF001  (the validation rank-IC series of the saved run)
        cand["ic"] = series["ic"]
    return cand


def candidate_from_trial(record: dict, net: pd.Series | None, base_kind: str = "?") -> dict:
    """A search trial as a candidate: its validation net returns (kept by the trial runner) and the evidence it recorded."""
    params = record.get("params") or {}
    kind = params.get("model.kind", base_kind)
    cid = f"trial-{record['number']}"
    if record.get("error") or net is None:
        return {"id": cid, "label": cid, "kind": kind, "params": params, "error": record.get("error") or "the trial produced no validation returns"}
    e = record.get("evidence") or {}
    res = e.get("resources") or {}
    return {"id": cid, "label": f"trial {record['number']} ({kind})", "kind": kind, "params": params, "returns": net, "turnover": e.get("turnover"), "coverage": e.get("coverage"), "n_assets": e.get("n_assets"),
            "exposure": e.get("exposure"), "liquidity": e.get("liquidity"), "ic_mean": e.get("ic_mean"), "lineage": e.get("lineage"),
            "complexity": sm.complexity_vector(kind, {k.split(".", 2)[2]: v for k, v in params.items() if k.startswith("model.params.")}, n_features=res.get("n_features"), n_params=res.get("n_params"),
                                               train_seconds=res.get("train_seconds"), inference_ms=res.get("inference_ms"))}


def select_trials(records: list[dict], series: dict, policy: SelectionPolicy | None = None, base_kind: str = "?") -> dict:
    """Select among the trials of a search or a tuning, from the validation returns the trial runner kept. The trials count: how many were tried is in the result."""
    cands = [candidate_from_trial(r, series.get(r["number"]), base_kind) for r in records]
    out = select(cands, policy)
    out["selected_trial"] = None if out["selected"] is None else int(str(out["selected"]).split("-", 1)[1])
    out["highest_ranked_trial"] = None if out["highest_ranked"] is None else int(str(out["highest_ranked"]).split("-", 1)[1])
    out["origin"] = {"kind": "search trials", "n_trials": len(records), "note": "every trial of the search is a candidate, failed ones included; selecting the best of N trials on the same validation data is optimistic, which the deflated Sharpe ratio of the search addresses"}
    return out


def selection_id(result: dict) -> str:
    key = json.dumps({"policy": result["policy_hash"], "candidates": sorted((c["id"], c.get("series_hash")) for c in result["candidates"])}, sort_keys=True, default=str)
    return "msel-" + hashlib.sha256(key.encode()).hexdigest()[:14]


def select_saved(registry, exp_ids: list[str], policy: SelectionPolicy | None = None, *, until=None, study_id: str | None = None, actor: str = "api", store: bool = True, plan_id: str | None = None,
                 audit: bool = False) -> dict:
    """Select among saved experiments, on their validation rows. The result is stored (append-only, kind ``model_selection``) with the policy hash, so the same call can be audited later; selecting does not
    promote anything and changes no experiment."""
    policy = policy or SelectionPolicy()
    if len(exp_ids) < 1 or len(set(exp_ids)) != len(exp_ids):
        raise SelectionError("give one or more distinct experiment ids")
    cands = []
    for eid in exp_ids:
        try:
            cands.append(candidate_from_experiment(registry, eid, until=until, with_ic=policy.primary_metric == "ic", audit=audit))
        except Exception as exc:                                                                         # an experiment that cannot be read is a failed candidate, not a failed selection
            cands.append({"id": eid, "label": eid, "kind": None, "params": {}, "error": f"{type(exc).__name__}: {exc}"})
    plan = None
    if plan_id:
        from . import research_controls as rc

        plan = rc.plan_state(registry, plan_id)
    out = select(cands, policy, plan)
    out["origin"] = {"kind": "saved experiments", "experiments": list(exp_ids), "until": None if until is None else str(pd.Timestamp(until).date()),
                     "ignored_rows": {c["id"]: c.get("ignored_rows") for c in out["candidates"] if c.get("ignored_rows")},
                     "note": "only the validation rows of each experiment are read; test and forward rows of a finalised experiment are counted here and never used"}
    return store_result(registry, out, policy, study_id=study_id, actor=actor, store=store)


def store_result(registry, out: dict, policy: SelectionPolicy, *, study_id: str | None = None, actor: str = "api", store: bool = True) -> dict:
    """Give a selection its id and, with a registry, write it (append-only, kind ``model_selection``, with the policy hash). Returns the JSON-safe result."""
    from . import diagnostics as dx

    out["selection_id"] = selection_id(out)
    if store and registry is not None:
        out["record_seq"] = dx.store(registry, "model_selection", out["selection_id"], dx._jsonable(out), policy_hash=policy.hash(), study_id=study_id, actor=actor)               # noqa: SLF001
    return dx._jsonable(out)                                                                                 # noqa: SLF001


def stored_selection(registry, selection_id_: str) -> dict | None:
    from . import diagnostics as dx

    rec = dx.latest(registry, selection_id_, "model_selection")
    return None if rec is None else {**rec["content"], "record_seq": rec["seq"], "stored_at": rec["created"]}
