"""Real-market walk-forward study of the ML portfolio allocation on NVDA and MU.

Protocol, fixed before any result was read:

* Data: Yahoo Finance daily adjusted closes through the platform's own downloader (provenance recorded by the ticker store), 2008-01-02 to the last session available.
* Model: one ridge regression on price, momentum and volatility features of a 5-day forward return (``fwd_ret_5``), fitted walk-forward (purged, expanding); the same predictions feed every allocator.
* Chronology: warm-up and validation first, then the test and forward segments, which are reserved. Everything chosen (the allocator settings, the list of allocators, the limits, the costs) is declared below
  and is NOT tuned. The validation segment is read once to describe the allocators; the test and forward segments are read once, through a frozen evaluation plan, after the declaration.
* Allocators on identical forecasts, dates, limits (long only, fully invested, no weight above 80%), costs (5 bps commission + 2 bps slippage) and execution lag: equal weight, inverse volatility, equal risk
  contribution, mean-variance, robust mean-variance, Black-Litterman. Equal weight is the benchmark.

Stages (each re-runnable): ``data``, ``develop`` (train to the end of validation, compare allocators and calibrations on validation), ``final`` (frozen plan, one evaluation event), ``report``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config                                           # noqa: E402
from src.webapp.api import App                                                      # noqa: E402
from src.webapp.universe import TickerStore, UniverseBuilder, yahoo_fetch            # noqa: E402

WORK = Path("/tmp/study")
ALLOCATORS = ("equal_weight", "inverse_vol", "erc", "mean_variance", "robust_mv", "black_litterman")
LIMITS = {"long_only": True, "cash_allowed": False, "max_weight": 0.8}
ALLOCATION = {"calibrate": "slope", "calib_inference": "hac", "calib_min_t": 1.0, "cov_method": "lw_constant_correlation", "cov_window": 252, "risk_aversion": 5.0, "limits": LIMITS, "opt_every": 5, "on_failure": "risk_only"}
BASE = {"dataset": {"universe": ["NVDA", "MU"], "feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5", "horizon": 5, "min_history": 126, "start": "2008-01-02"},
        "split": {"min_train_days": 756, "refit_every": 63, "validation_fraction": 0.2, "test_fraction": 0.15, "forward_fraction": 0.15},
        "backtest": {"rebalance": "weekly", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": None}, "model": {"kind": "ridge"}, "explain": False}


def make_app() -> App:
    config = load_config()
    root = WORK / "root"
    if not root.exists():
        from tests.app_roots import isolated_root

        isolated_root(root, config)
    store = TickerStore(WORK / "prices", fetch=yahoo_fetch)
    return App(config, builder=UniverseBuilder(config, store), root=root)


def wait(app: App, resp: dict, timeout: float = 3600.0) -> dict:
    t0 = time.time()
    while True:
        job = app.job(resp["job"])
        if job["status"] in ("done", "error"):
            if job["status"] == "error":
                raise RuntimeError(job["error"])
            return job["result"]
        if time.time() - t0 > timeout:
            raise TimeoutError(str(job))
        time.sleep(0.5)


def candidate(allocator: str, name: str | None = None) -> dict:
    al = {**ALLOCATION, "allocator": allocator}
    return {**BASE, "name": name or f"nvda-mu-{allocator}", "signal": {"method": "portfolio_allocation", "holding_period": 1, "allocation": al}}


def stage_develop(exp_id: str | None = None) -> dict:
    """Train to the end of validation, then describe the allocators and the calibrations on the validation segment only."""
    from src.ml.experiments import calibration_saved, compare_saved

    app = make_app()
    if exp_id is None:
        out = wait(app, app.extra_post["/api/ml/train"]({**BASE, "name": "nvda-mu-dev", "mode": "development", "signal": {"method": "proportional", "holding_period": 1}}))
        exp_id = out["id"]
    registry = app.extra_get["/api/ml/lineage"].__self__.registry
    alloc = {k: v for k, v in ALLOCATION.items()}
    cmp_ = compare_saved(registry, exp_id, ALLOCATORS, select_on="validation", allocation=alloc, include_existing_rule=False)
    from src.ml import allocator_compare as ac

    cmp_pub = ac.public(cmp_)
    variants = {"pooled_sqrt_horizon": {"calib_inference": "sqrt_horizon"}, "pooled_hac": {"calib_inference": "hac"}, "regularized_hac": {"calib_method": "regularized", "calib_inference": "hac"},
                "asset_hac": {"calib_method": "asset", "calib_inference": "hac"}, "partial_pooling_hac": {"calib_method": "partial_pooling", "calib_inference": "hac"}}
    cal = calibration_saved(registry, exp_id, variants, select_on="validation", allocation=alloc)
    res = {"experiment": exp_id, "compare_validation": cmp_pub, "calibration_validation": cal}
    (WORK / "develop.json").write_text(json.dumps(res, default=str, indent=1))
    return res





def stage_plan(study: str = "nvda-mu-allocation", notes: str = "NVDA/MU real-data allocation study: six allocators, one frozen evaluation") -> dict:
    """Freeze an evaluation plan whose candidates are the six allocators (settings above, declared, untuned) and whose baseline is equal weight. Reserving the plan builds no result: it only records which
    outcomes the evaluation will reveal."""
    app = make_app()
    cands = [candidate(a) for a in ALLOCATORS]
    resp = app.extra_post["/api/ml/evaluation-plan"]({"study": study, "candidates": cands, "selection_adjustment": "holm",
                                                      "uncertainty_method": "stationary block bootstrap and HAC", "notes": notes, "freeze": True})
    state = wait(app, resp)
    (WORK / "plan.json").write_text(json.dumps(state, default=str, indent=1))
    return state


def stage_final(plan_id: str, key: str = "nvda-mu-final-1", allow_reuse: bool = False) -> dict:
    app = make_app()
    out = wait(app, app.extra_post["/api/ml/evaluate"]({"plan_id": plan_id, "idempotency_key": key, "allow_reuse": allow_reuse}))
    (WORK / "final.json").write_text(json.dumps(out, default=str, indent=1))
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "develop"
    if stage == "develop":
        stage_develop(sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage == "plan":
        stage_plan(notes=sys.argv[2] if len(sys.argv) > 2 else "NVDA/MU real-data allocation study: six allocators, one frozen evaluation")
    elif stage == "final":
        stage_final(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "nvda-mu-final-1", len(sys.argv) > 4 and sys.argv[4] == "reuse")
