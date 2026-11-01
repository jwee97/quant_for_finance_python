"""Predeclared regime and stress evaluation (spec v3, Release 5, P1-19).

A strategy's average hides when it earns and when it does not. The windows here are *declared before the evaluation* (``declare`` fixes their definitions and returns a hash that the results cite), so
no one can pick the flattering ones afterwards. They come in two kinds, and every result says which:

* **causal** regimes use only information known before each day (a trailing trend, volatility tercile or volume drop measured on data that ends the day before). A rule that trades on them could have
  been run live; the slice can be used to *condition* a position.
* **retrospective** slices are named date ranges chosen with hindsight (a crisis calendar). They describe what happened; they must not be used to tune the strategy, and a result built on them says so.

A regime with fewer than ``min_days`` observations has *insufficient evidence*; it is never dropped silently and never read as zero or as a pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import inference as inf

VERSION = "regimes-v1"
KINDS = ("trend", "volatility", "liquidity", "sideways", "crisis")


@dataclass(frozen=True)
class RegimeSpec:
    """One declared way of cutting time. ``params`` are fixed at declaration; ``causal`` is a fact about the definition, not a choice at evaluation."""

    name: str
    kind: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")

    @property
    def causal(self) -> bool:
        return self.kind != "crisis"

    def as_dict(self) -> dict:
        return {**asdict(self), "causal": self.causal}


def declare(specs: list[RegimeSpec]) -> dict:
    """Freeze the list of definitions and return it with its hash. Evaluate against this declaration; a result that cites a different hash was not evaluated against it."""
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("regime names must be unique")
    body = [s.as_dict() for s in specs]
    return {"version": VERSION, "specs": body, "hash": hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]}


def default_specs(crisis_windows: dict[str, tuple[str, str]] | None = None) -> list[RegimeSpec]:
    """A conventional declaration: trend (63 days, +-5%), volatility terciles (63-day volatility against its own history), liquidity stress (20-day dollar volume under 70% of its 252-day median) and the
    crisis windows given. The numbers are illustrative defaults, to be declared by the person who evaluates; they are not tuned."""
    specs = [RegimeSpec("trend_63", "trend", {"window": 63, "threshold": 0.05}), RegimeSpec("volatility_63", "volatility", {"window": 63, "min_history": 252}),
             RegimeSpec("liquidity_20", "liquidity", {"short": 20, "long": 252, "ratio": 0.7}), RegimeSpec("sideways_63", "sideways", {"window": 63, "threshold": 0.05})]
    for name, (a, b) in (crisis_windows or {}).items():
        specs.append(RegimeSpec(f"crisis:{name}", "crisis", {"start": a, "end": b}))
    return specs


def _lag(s: pd.Series) -> pd.Series:
    return s.shift(1)                                                                         # known at the close of the previous day


def label(spec: RegimeSpec, market: pd.Series, volume: pd.Series | None = None) -> pd.Series:
    """The regime of each day: a string, or ``NaN`` where the definition cannot be evaluated (too little history, no volume). ``market`` is a daily return series, ``volume`` the universe's daily dollar volume."""
    p = spec.params
    idx = market.index
    if spec.kind in ("trend", "sideways"):
        trailing = _lag((1.0 + market).rolling(p["window"], min_periods=p["window"]).apply(np.prod, raw=True) - 1.0)
        thr = float(p["threshold"])
        if spec.kind == "trend":
            return pd.Series(np.where(trailing.isna(), None, np.where(trailing > thr, "up", np.where(trailing < -thr, "down", "flat"))), index=idx, dtype=object)
        return pd.Series(np.where(trailing.isna(), None, np.where(trailing.abs() <= thr, "sideways", "trending")), index=idx, dtype=object)
    if spec.kind == "volatility":
        vol = _lag(market.rolling(p["window"], min_periods=p["window"]).std(ddof=1))
        lo = vol.expanding(min_periods=p["min_history"]).quantile(1.0 / 3.0)               # terciles of the volatility seen SO FAR, never of the whole sample
        hi = vol.expanding(min_periods=p["min_history"]).quantile(2.0 / 3.0)
        return pd.Series(np.where(lo.isna(), None, np.where(vol <= lo, "low", np.where(vol >= hi, "high", "mid"))), index=idx, dtype=object)
    if spec.kind == "liquidity":
        if volume is None:
            return pd.Series([None] * len(idx), index=idx, dtype=object)
        v = volume.reindex(idx)
        short = _lag(v.rolling(p["short"], min_periods=p["short"]).mean())
        base = _lag(v.rolling(p["long"], min_periods=p["short"]).median())
        return pd.Series(np.where(base.isna() | short.isna(), None, np.where(short < p["ratio"] * base, "stressed", "normal")), index=idx, dtype=object)
    a, b = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
    return pd.Series(np.where((idx >= a) & (idx <= b), spec.name.split(":", 1)[-1], None), index=idx, dtype=object)


def evaluate(strategy: pd.Series, market: pd.Series, declaration: dict, *, volume: pd.Series | None = None, frequency: str = "daily", min_days: int = 60, label_prefix: str = "") -> dict:
    """The strategy's return in every declared regime: the days, the mean with its HAC uncertainty, the dependence-aware Sharpe interval, the annualised naive Sharpe beside it, and the eligibility. A regime
    with fewer than ``min_days`` has insufficient evidence and no number. The result cites the declaration's hash and says which regimes are causal and which retrospective."""
    s = strategy.dropna()
    rows, notes = [], []
    for spec_d in declaration["specs"]:
        spec = RegimeSpec(spec_d["name"], spec_d["kind"], spec_d["params"])
        lab = label(spec, market, volume).reindex(s.index)                                  # defined on the market's whole history, then read on the strategy's days
        states = sorted({x for x in lab.dropna().unique()})
        if not states:
            rows.append({"regime": spec.name, "state": None, "kind": spec.kind, "causal": spec.causal, "eligibility": "insufficient", "n_days": 0,
                         "reason": "the definition could not be evaluated on this sample (no days fall in it, too little history, or no volume)"})
            continue
        for st in states:
            part = s[lab == st]
            row = {"regime": spec.name, "state": st, "kind": spec.kind, "causal": spec.causal, "n_days": int(len(part))}
            if len(part) < min_days:
                row.update(eligibility="insufficient", reason=f"{len(part)} days in this regime: fewer than the {min_days} declared as the minimum for a reading")
            else:
                mean = inf.hac_mean(part, frequency=frequency, label=f"{label_prefix}{spec.name}={st}")
                sharpe = inf.sharpe_inference(part, frequency=frequency, label=f"{label_prefix}{spec.name}={st}", n_boot=300)
                row.update(eligibility="applied", mean=mean.value, mean_interval=list(mean.interval) if mean.interval else None, mean_p=mean.p_value, sharpe_naive=sharpe["naive"].value,
                           sharpe_hac_interval=list(sharpe["hac"].interval) if sharpe["hac"].interval else None, sharpe_bootstrap_interval=list(sharpe["bootstrap"].interval) if sharpe["bootstrap"].interval else None)
            if not spec.causal:
                row["note"] = "retrospective slice chosen with hindsight: descriptive only, not evidence for a rule and not to be tuned on"
            rows.append(row)
    if any(not r["causal"] for r in rows):
        notes.append("retrospective slices describe what happened; a causal regime rule would need a causal definition")
    n_ok = sum(1 for r in rows if r["eligibility"] == "applied")
    return {"version": VERSION, "declaration_hash": declaration["hash"], "min_days": min_days, "frequency": frequency, "rows": rows, "n_applied": n_ok, "n_insufficient": len(rows) - n_ok,
            "coverage_note": "a regime with too few days is listed as insufficient evidence, never dropped" + ("" if n_ok else "; no regime had enough days for a reading"), "notes": notes}
