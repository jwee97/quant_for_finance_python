"""One call from a panel of prices to the whole study: eligibility, nested discovery, and every robustness diagnostic, as a JSON-ready dictionary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio import lab
from . import discover as dc, membership as mb, robust, walk

NOTES = ("The out-of-sample series is stitched from blocks, each judged with a configuration chosen only on the days before it (less an embargo).",
         "The table of every candidate over the whole history is in-sample context. Its best row is a search result, not a finding.",
         "A benchmark is shown, not matched: a long-only portfolio is comparable with equal weight; a screen that concentrates in a few names or a sector is a different bet, and the beta says by how much.",
         "Costs are the linear cost on traded weight (one way). Delisted names earn zero from their last day; no delisting return is invented.")


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def run_study(prices: pd.DataFrame, screen_specs: list[dict], methods: list[str], lookbacks: list[int], *, volume: pd.DataFrame | None = None, membership=None, groups: dict | None = None,
              folds: int = 4, min_train_days: int = 504, embargo: int = 5, step: int = 21, cost_bps: float = 10.0, max_weight: float = 1.0, bootstrap: int = 300, progress=None) -> dict:
    R = prices.pct_change(fill_method=None).iloc[1:]
    dv = None if volume is None else (volume.reindex_like(prices) * prices).iloc[1:]
    elig, einfo = mb.eligibility(R.index, list(R.columns), membership)
    d = dc.discover(R, screen_specs, methods, lookbacks, dollar_volume=dv, eligible=elig, groups=groups, folds=folds, min_train_days=min_train_days, embargo=embargo, step=step, cost_bps=cost_bps,
                    max_weight=max_weight, progress=progress)
    oos, bench = d["oos"], d["benchmarks"]
    bname = "equal_weight"
    b = bench[bname]
    modal = d["modal_run"]
    w_oos = modal.weights.loc[(modal.weights.index >= oos.index[0]) & (modal.weights.index <= oos.index[-1])]
    # stock-removal sensitivity: the modal configuration without each of its three most-held names, over the same days
    held = modal.weights.mean().sort_values(ascending=False)
    spec = modal.spec
    removal = []
    base_sh = robust.sharpe(modal.net.reindex(oos.index))
    for name in list(held.index[:3]):
        try:
            r = walk.run(R, spec["screen"], spec["method"], dollar_volume=dv, eligible=elig, groups=groups, lookback=spec["lookback"], step=step, cost_bps=cost_bps, max_weight=max_weight, exclude=(name,))
            removal.append({"removed": name, "average_weight": float(held[name]), "sharpe": robust.sharpe(r.net.reindex(oos.index)), "change": robust.sharpe(r.net.reindex(oos.index)) - base_sh})
        except ValueError as exc:
            removal.append({"removed": name, "average_weight": float(held[name]), "note": str(exc)})
    oos_sum = robust.summary(oos)
    costs = float(d["modal_run"].turnover.mean() * 2 * cost_bps * 1e-4 * (252 / step)) if len(modal.turnover) else float("nan")
    out = {"design": d["design"], "eligibility": einfo, "n_names": int(R.shape[1]), "period": {"from": str(R.index[0].date()), "to": str(R.index[-1].date())},
           "folds": d["folds"], "picks": d["picks"], "modal_choice": d["modal_choice"], "pick_stability": d["pick_stability"], "n_candidates": d["n_candidates"], "failed": d["failed"],
           "status": d["status"], "oos": oos_sum, "benchmarks": {k: robust.summary(v) for k, v in bench.items()}, "fixed_baseline": robust.summary(d["fixed_baseline"]),
           "deflated_sharpe": d["dsr"], "beta": robust.beta(oos, b), "concentration": robust.concentration(w_oos, groups), "stability": robust.stability(w_oos),
           "turnover_per_year": float(modal.turnover.mean() * 2 * (252 / step)) if len(modal.turnover) else None, "annual_cost_drag": costs,
           "removal": removal, "bootstrap": robust.block_bootstrap(oos, b, n=bootstrap), "regimes": robust.regimes(oos, b), "in_sample": d["in_sample"],
           "equity": {"dates": [str(x.date()) for x in oos.index], "oos": [float(x) for x in (1 + oos.fillna(0)).cumprod()], **{k: [float(x) for x in (1 + v.fillna(0)).cumprod()] for k, v in bench.items()},
                      "fixed_equal_weight_walk": [float(x) for x in (1 + d["fixed_baseline"].fillna(0)).cumprod()]},
           "methods": {k: lab.METHODS[k].label for k in methods}, "notes": list(NOTES)}
    if einfo["hindsight"]:
        out["warnings"] = [mb.HINDSIGHT]
    elif einfo.get("limitation"):
        out["warnings"] = [einfo["limitation"]]
    return _clean(out)
