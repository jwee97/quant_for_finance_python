"""Calibration of a search: what it finds when there is nothing to find.

A search that tries hundreds of formulas will always find some that look good, and how many it finds from luck alone depends on the data (how many companies, how many dates, how alike the
formulas are) and on the bars it sets. The only way to know is to measure it, on the data in hand, in a world where the answer is known: the **placebo**.

In a placebo world the returns of each date are dealt out among the companies at random: each company gets the return of another company, the same relabelling for every window that starts on
that date (the next period, the next quarter, the next year, and the period the long-short portfolio holds). The market, the cross-section of returns on each date, the number of companies with a
return and all the statements are untouched, but no signal can predict anything. A search run there should find nothing, and what it lets through is luck. Three things are read off it:

* **how much of a real run is luck**: the formulas that get through every screen before the locked test, per formula tried, in a world where none can be real; set beside the number a real run lets
  through, it estimates the share of the real run's survivors that are luck (the permutation estimate of a false-discovery proportion, ``E[V] / R``);
* **whether the t-statistic can be believed on this data**: on shuffled returns the t-statistics of the formulas should have a spread of 1 (about one formula in ten reaching ``|t| >= 1.65``). A
  spread above 1 means the standard errors are too small for this data (few dates, formulas that take few different values), and the bars should be raised by that factor;
* **whether the correction for the number tried holds**: the lowest false-discovery q-value any shuffled formula reached. A bar looser than that lets luck through.

What a shuffle cannot show: it destroys the tendency of companies in one country or sector to move together, so it says nothing about the extra overconfidence this causes for a formula that is the
same for a whole sector or country (such a formula has far fewer independent observations than companies). The Factor lab's neutralised studies and the Fama-MacBeth premium are the checks for that.

Nothing is written to the database being studied: the placebo search runs on a throwaway one, so the library, the graveyard, the counts of ideas tried and the look at the locked dates are not
touched.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .discovery import STAGES, Discovery, DiscoverySpec, bh_qvalues
from .evaluate import holding_returns
from .jsonutil import clean
from .research import ResearchContext, horizon_days
from .schema import now_iso
from .store import FundamentalsStore

MAX_REPEATS = 5
MIN_FORMULAS_FOR_T = 150                                            # shuffled formulas needed before the spread of their t-statistics is read (fewer, and it is mostly noise)
TAILS = (1.65, 2.0, 3.0)                                            # |t| levels the shuffled t-statistics are counted at
EXPECTED_TAIL = {t: float(2.0 * stats.norm.sf(t)) for t in TAILS}    # what a correct t-statistic gives on shuffled returns: 9.9%, 4.6%, 0.27%


class PlaceboContext(ResearchContext):
    """A research context in which no signal predicts anything: on every date the returns that exist are dealt out at random among the companies that have one, the same way for every window that
    starts on that date. Everything known on a date (statements, prices up to then, size, liquidity) is as it was, and so is how many companies have a return on the date."""

    def __init__(self, source: ResearchContext, seed: int):                     # not a call of the parent: it shares what the source has built and changes what comes after the date
        self.__dict__.update(source.__dict__)
        self._cache = {k: v for k, v in source._cache.items() if not (isinstance(k, tuple) and k and k[0] in ("forward", "holding"))}
        self._source = source
        self._keys = np.random.default_rng(seed).random((len(self.dates), self.n))         # one random key per company and date: the order in which a date's returns are dealt out
        self._cache[("holding",)] = self._deal(holding_returns(source))

    def _deal(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.to_numpy(float)
        out = np.full_like(values, np.nan)
        for t in range(len(values)):
            have = np.flatnonzero(np.isfinite(values[t]))
            out[t, have] = values[t, have[np.argsort(self._keys[t, have])]]
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)

    def forward(self, horizon="1m") -> pd.DataFrame:
        key = ("forward", horizon_days(horizon))
        if key not in self._cache:
            self._cache[key] = self._deal(self._source.forward(horizon))
        return self._cache[key]


def placebo_context(ctx: ResearchContext, seed: int = 101) -> PlaceboContext:
    return PlaceboContext(ctx, seed)


def _one_run(ctx: ResearchContext, spec: DiscoverySpec, seed: int, say, stop) -> dict:
    placebo = PlaceboContext(ctx, seed)
    with tempfile.TemporaryDirectory() as folder:
        store = FundamentalsStore(Path(folder) / "placebo.db")
        search = Discovery(store, placebo)
        run_id = search.create(spec)
        summary = search.run(run_id, say, stop)
        candidates = search.candidates(run_id, limit=1_000_000)
    tested = [c for c in candidates if c["metrics"].get("p_train") is not None]
    t = np.array([c["metrics"]["t_train"] for c in tested if c["metrics"].get("t_train") is not None and np.isfinite(c["metrics"]["t_train"])], dtype=float)
    p = np.array([c["metrics"]["p_train"] if np.isfinite(c["metrics"]["p_train"]) else 1.0 for c in tested], dtype=float)
    counts = summary["counts"]
    return {"seed": seed, "status": summary["status"], "formulas": int(counts.get("generated", 0)), "tested": int(counts.get("tested", 0)), "funnel": {s: int(counts.get(s, 0)) for s in STAGES},
            "survivors": int(counts.get("survivors", 0) or 0), "finalists": int(counts.get("finalists", 0) or 0), "candidates": int(counts.get("candidates", 0) or 0),
            "failed_locked_test": int(counts.get("failed_locked_test", 0) or 0), "min_q": float(bh_qvalues(p).min()) if len(p) else None, "t": t.tolist()}


def calibrate_search(ctx: ResearchContext, spec: DiscoverySpec, repeats: int = 3, seed: int = 101, progress=None, stop=None, real: dict | None = None) -> dict:
    """Run ``spec`` ``repeats`` times in placebo worlds and report what it found. ``real`` (the counts of the run being checked: ``tested``, ``survivors`` and ``candidates``) is set beside the result."""
    repeats = max(1, min(int(repeats), MAX_REPEATS))
    say = progress or (lambda text: None)
    runs = []
    for r in range(repeats):
        if stop is not None and stop():
            break
        run = _one_run(ctx, spec, seed + 7919 * r, lambda done, total, text, r=r: say(f"shuffled returns {r + 1} of {repeats}: {done} of {total} formulas, {text}"), stop)
        if run["status"] != "finished":                                              # a search stopped half way has not reached the stages that count the survivors
            break
        runs.append(run)
    out = summarise(runs, spec, real)
    out["complete"] = len(runs) == repeats
    return out


def poisson_interval(count: float, exposure: float = 1.0, level: float = 0.95) -> list[float]:
    """The exact interval for a rate when ``count`` events were seen in ``exposure`` units of exposure."""
    a = (1.0 - level) / 2.0
    k = int(round(count))
    low = 0.0 if k == 0 else float(stats.chi2.ppf(a, 2 * k) / 2.0)
    high = float(stats.chi2.ppf(1.0 - a, 2 * (k + 1)) / 2.0)
    return [low / exposure, high / exposure]


def summarise(runs: list[dict], spec: DiscoverySpec, real: dict | None = None) -> dict:
    n = len(runs)
    t = np.concatenate([np.array(run["t"], dtype=float) for run in runs]) if runs else np.array([])
    tested = sum(run["tested"] for run in runs)
    survivors = sum(run["survivors"] for run in runs)
    candidates = [run["candidates"] for run in runs]
    abs_t = np.abs(t)
    t_sd = float(t.std()) if len(t) > 5 else float("nan")
    t_sd_error = float(t_sd / np.sqrt(2.0 * max(len(t) / 3.0, 2.0))) if len(t) > 5 else float("nan")        # the formulas of a search are alike: about a third as many are independent
    shape = "unknown"
    if len(t) >= MIN_FORMULAS_FOR_T and np.isfinite(t_sd):
        shape = "generous" if t_sd >= 1.25 else ("careful" if t_sd <= 0.8 else "close")
    luck = survivors / n if n else float("nan")                                           # formulas let through every screen before the locked test, per run
    summary = {"runs": n, "tested": tested, "tested_per_run": tested / n if n else float("nan"), "t_sd": t_sd, "t_sd_error": t_sd_error, "t_mean": float(t.mean()) if len(t) else float("nan"), "t_calibration": shape,
               "tail": {f"{level}": {"share": float((abs_t >= level).mean()) if len(t) else float("nan"), "expected": EXPECTED_TAIL[level]} for level in TAILS},
               "t_bar_5pct": float(np.percentile(abs_t, 95)) if len(t) >= MIN_FORMULAS_FOR_T else float("nan"),
               "survivors_mean": float(luck), "survivors_interval": [x / n for x in poisson_interval(survivors)] if n else [float("nan")] * 2,
               "survivors_per_100": float(100.0 * survivors / tested) if tested else float("nan"), "finalists_mean": float(np.mean([run["finalists"] for run in runs])) if runs else float("nan"),
               "candidates_mean": float(np.mean(candidates)) if candidates else float("nan"),
               "candidates_max": int(max(candidates)) if candidates else 0, "runs_with_a_candidate": int(sum(c > 0 for c in candidates)),
               "min_q": float(min(run["min_q"] for run in runs if run["min_q"] is not None)) if any(run["min_q"] is not None for run in runs) else None,
               "funnel_mean": {s: float(np.mean([run["funnel"].get(s, 0) for run in runs])) for s in STAGES} if runs else {}}
    if real and real.get("survivors"):                                                     # E[V] / R: what luck is expected to contribute to what the real run let through
        r = float(real["survivors"])
        scale = (float(real["tested"]) / summary["tested_per_run"]) if real.get("tested") and summary["tested_per_run"] else 1.0
        summary["luck_share"] = float(min(1.0, luck * scale / r))
        summary["luck_share_interval"] = [float(min(1.0, x * scale / r)) for x in summary["survivors_interval"]]
    return clean({"spec": spec.describe(), "strictness": spec.strictness, "runs": [{k: v for k, v in run.items() if k != "t"} for run in runs], "summary": summary,
                  "real": real, "reading": reading(summary, real), "t_histogram": _hist(t)})


def _hist(t: np.ndarray, bins: int = 24) -> dict:
    if len(t) < 5:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(np.clip(t, -6, 6), bins=bins, range=(-6, 6))
    return {"edges": [round(float(e), 3) for e in edges], "counts": [int(c) for c in counts]}


def reading(s: dict, real: dict | None = None) -> str:
    """The sentences under the numbers."""
    if not s.get("runs"):
        return "No placebo run finished."
    n, parts = s["runs"], []
    per_run = s["tested_per_run"]
    parts.append(f"On shuffled returns, where nothing can be found, this search let {s['survivors_mean']:.1f} of the {per_run:.0f} formulas it tried through every screen before the locked test "
                 f"(between {s['survivors_interval'][0]:.1f} and {s['survivors_interval'][1]:.1f} allowing for chance, from {n} shuffle{'s' if n != 1 else ''}); {s['finalists_mean']:.1f} of them were past the "
                 f"correction for the number tried, and a Candidate came out of {s['runs_with_a_candidate']} of the {n}.")
    if real and real.get("survivors") is not None:
        got = real["survivors"]
        if got and "luck_share" in s:
            lo, hi = s["luck_share_interval"]
            parts.append(f"The real search let {got} through, so about {s['luck_share']:.0%} of them ({lo:.0%} to {hi:.0%}) are expected to be luck." if s["luck_share"] < 0.9
                         else f"The real search let {got} through, which is no more than luck lets through on this data: there is no sign of anything real in it.")
        elif not got:
            parts.append("The real search let nothing through.")
    sd, bar, err = s["t_sd"], s["t_bar_5pct"], s["t_sd_error"]
    if s["t_calibration"] == "unknown":
        if np.isfinite(sd):
            parts.append(f"Too few formulas were tried to judge the t-statistics ({s['tested']} where {MIN_FORMULAS_FOR_T} or more are needed; their spread was {sd:.2f}, give or take {err:.2f}, where 1.00 is "
                         "correct): search more formulas, or shuffle more times.")
    elif s["t_calibration"] == "generous":
        parts.append(f"The t-statistics are too generous on this data: on shuffled returns their spread is {sd:.2f} (give or take {err:.2f}) where it should be 1.00, so the standard errors are about "
                     f"{sd - 1:.0%} too small and a t of {bar:.1f}, not 1.96, is the 5% bar here. Use the strict setting, or add companies or history.")
    elif s["t_calibration"] == "careful":
        parts.append(f"The t-statistics are careful on this data: on shuffled returns their spread is {sd:.2f} (give or take {err:.2f}) where it should be 1.00, so the first screen is stricter than it looks.")
    elif sd - 2.0 * err > 1.0:
        parts.append(f"The t-statistics are a little generous on this data: their spread on shuffled returns is {sd:.2f} (give or take {err:.2f}; 1.00 is correct), so a t of {bar:.1f} is the 5% bar here, "
                     "not 1.96.")
    else:
        parts.append(f"The t-statistics are about right on this data: their spread on shuffled returns is {sd:.2f} (give or take {err:.2f}; 1.00 is correct).")
    if n < 3:
        parts.append("With fewer than three shuffles these figures are rough.")
    return " ".join(parts)


def save(store: FundamentalsStore, result: dict) -> str:
    """Keep a calibration with the other experiments (never in the library or the graveyard)."""
    experiment_id = "P" + uuid.uuid4().hex[:10]
    counts = store.counts()
    with store.db.transaction() as conn:
        conn.execute("INSERT OR REPLACE INTO experiments (experiment_id, kind, created_at, spec, data_version, code_version, summary, status) VALUES (?,?,?,?,?,?,?,?)",
                     (experiment_id, "placebo", now_iso(), json.dumps(clean(result["spec"])), f"{counts['companies']}c/{counts['observations']}o/{counts['price_rows']}p@{now_iso()[:10]}", "fundamentals-1",
                      json.dumps(clean({"summary": result["summary"], "reading": result["reading"], "runs": result["runs"], "strictness": result["strictness"], "real": result.get("real")})), "finished"))
    return experiment_id


__all__ = ["MIN_FORMULAS_FOR_T", "PlaceboContext", "placebo_context", "calibrate_search", "summarise", "reading", "poisson_interval", "save", "MAX_REPEATS", "TAILS", "EXPECTED_TAIL"]
