"""Portfolio discovery with nested chronological validation.

The question is not "which screen and which optimiser had the best backtest?": with enough combinations one of them always does. The question is "if I had followed this *procedure*, choosing the screen, the
construction method and the lookback on the data I had at the time, what would I have earned on the data that came next?". So the three choices are ONE selection process:

1. every candidate (screen x method x lookback) is walked forward over the whole history once; its return on a day depends only on data before that day, so a slice of it is what it would have earned then;
2. the out-of-sample period is cut into consecutive blocks. For each block, the candidates are scored on the days before it (less an embargo), the best is chosen, and the block's return is that candidate's
   return on the block, which was not used to choose it;
3. the blocks are stitched into one out-of-sample series, judged against the benchmarks on the same days and corrected for the number of candidates tried.

The full-history table of candidates is returned for context and labelled as in-sample: its best row is a search result, never a finding. The number of candidates is reported with every result.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..stats import inference
from . import robust, screens as sc, walk

MAX_CANDIDATES = 60


def _id(screen: dict, method: str, lookback: int) -> str:
    return f"{sc.label(screen)} | {method} | {lookback}d"


def candidates(screen_specs: list[dict], methods: list[str], lookbacks: list[int]) -> list[dict]:
    specs = [sc.validate(s) for s in screen_specs]
    out = [{"id": _id(s, m, lb), "screen": s, "method": m, "lookback": int(lb)} for s, m, lb in itertools.product(specs, methods, lookbacks)]
    if not out:
        raise ValueError("no candidates: choose at least one screen, one method and one lookback")
    if len(out) > MAX_CANDIDATES:
        raise ValueError(f"{len(out)} candidates is more than the {MAX_CANDIDATES} this lab will run: a larger search is a different, and weaker, claim; narrow the grid")
    return out


def benchmarks(returns: pd.DataFrame, eligible: pd.DataFrame | None, dollar_volume: pd.DataFrame | None, index) -> dict[str, pd.Series]:
    """Equal weight of every eligible name with a return that day (daily rebalanced: the platform's default benchmark), and, where there is volume, a weight proportional to the previous day's dollar volume.
    The second is a proxy for cap-weight (shares outstanding are not in the data) and is labelled so."""
    R = returns.where(eligible.reindex_like(returns).fillna(False)) if eligible is not None else returns
    out = {"equal_weight": R.mean(axis=1, skipna=True).fillna(0.0).reindex(index)}
    if dollar_volume is not None:
        w = dollar_volume.reindex_like(R).shift(1).where(R.notna())
        out["dollar_volume_weighted (a cap-weight proxy)"] = ((w.div(w.sum(axis=1), axis=0)) * R).sum(axis=1).reindex(index)
    return out


def discover(returns: pd.DataFrame, screen_specs: list[dict], methods: list[str], lookbacks: list[int], *, dollar_volume: pd.DataFrame | None = None, eligible: pd.DataFrame | None = None,
             groups: dict | None = None, folds: int = 4, min_train_days: int = 504, embargo: int = 5, step: int = 21, cost_bps: float = 10.0, max_weight: float = 1.0, metric: str = "sharpe",
             progress=None) -> dict:
    cands = candidates(screen_specs, methods, lookbacks)
    if metric not in ("sharpe", "calmar_free_return"):
        raise ValueError("metric is sharpe")
    progress = progress or (lambda msg: None)
    runs, failed = {}, []
    for i, c in enumerate(cands):
        progress(f"walking candidate {i + 1} of {len(cands)}")
        try:
            runs[c["id"]] = walk.run(returns, c["screen"], c["method"], dollar_volume=dollar_volume, eligible=eligible, groups=groups, lookback=c["lookback"], step=step, cost_bps=cost_bps, max_weight=max_weight)
        except (ValueError, sc.ScreenError) as exc:
            failed.append({"id": c["id"], "reason": str(exc)})
    if not runs:
        raise ValueError("no candidate could be run: " + "; ".join(f["reason"] for f in failed[:3]))
    nets = pd.DataFrame({k: r.net for k, r in runs.items()})
    dates = nets.dropna(how="all").index
    if len(dates) < min_train_days + folds * 63:
        raise ValueError(f"the history is too short: {len(dates)} usable days, need at least {min_train_days + folds * 63} for {folds} out-of-sample blocks after {min_train_days} days of selection history")
    begin = min_train_days
    edges = np.linspace(begin, len(dates), folds + 1).astype(int)
    stitched, table, picks = [], [], []
    for j in range(folds):
        a, b = edges[j], edges[j + 1]
        block = dates[a:b]
        known = nets.loc[dates[:max(0, a - embargo)]]
        scores = {k: robust.sharpe(known[k]) for k in nets.columns if known[k].notna().sum() >= 126}
        scores = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not scores:
            table.append({"fold": j + 1, "from": str(block[0].date()), "to": str(block[-1].date()), "chosen": None, "note": "no candidate had enough history to be scored"})
            continue
        best = max(scores, key=scores.get)
        oos = nets.loc[block, best]
        stitched.append(oos)
        picks.append(best)
        table.append({"fold": j + 1, "from": str(block[0].date()), "to": str(block[-1].date()), "chosen": best, "selection_days": int(known[best].notna().sum()), "selection_sharpe": scores[best],
                      "oos_sharpe": robust.sharpe(oos), "oos_return": float((1 + oos.fillna(0)).prod() - 1), "candidates_scored": len(scores), "runner_up_gap": float(sorted(scores.values())[-1] - sorted(scores.values())[-2]) if len(scores) > 1 else None})
    if not stitched:
        raise ValueError("no out-of-sample block could be scored")
    oos = pd.concat(stitched)
    bench = benchmarks(returns, eligible, dollar_volume, oos.index)
    fixed = walk.run(returns, {"kind": "all"}, "equal_weight", dollar_volume=dollar_volume, eligible=eligible, groups=groups, lookback=min(c["lookback"] for c in cands), step=step, cost_bps=cost_bps,
                     max_weight=max_weight).net.reindex(oos.index)
    freq = pd.Series(picks).value_counts()
    modal = freq.index[0]
    trial_sh = np.array([robust.sharpe(nets[k]) / np.sqrt(ANN) for k in nets.columns if np.isfinite(robust.sharpe(nets[k]))])
    dsr = None
    if len(trial_sh) > 1:
        try:
            dsr = inference.deflated_sharpe_ratio(oos.dropna().to_numpy(), trial_sharpes=trial_sh, n_trials=len(cands))
        except Exception:
            dsr = None
    sel_run = runs[modal]
    in_sample = sorted(({"id": k, "sharpe": robust.sharpe(nets[k]), "turnover": float(runs[k].turnover.mean() * 252 / 12 if len(runs[k].turnover) else np.nan), "fallbacks": runs[k].fallbacks,
                         "skipped": runs[k].skipped} for k in nets.columns), key=lambda r: -(r["sharpe"] if np.isfinite(r["sharpe"]) else -9))
    return {"folds": table, "picks": {k: int(v) for k, v in freq.items()}, "modal_choice": modal, "pick_stability": float(freq.iloc[0] / len(picks)), "n_candidates": len(cands), "failed": failed,
            "oos": oos, "benchmarks": bench, "fixed_baseline": fixed, "modal_run": sel_run, "runs": runs, "dsr": dsr, "in_sample": in_sample,
            "status": "selected_on_prior_data_only", "design": {"folds": folds, "min_train_days": min_train_days, "embargo": embargo, "step": step, "cost_bps": cost_bps, "max_weight": max_weight, "metric": metric,
                                                                  "n_candidates": len(cands)}}


ANN = 252.0
