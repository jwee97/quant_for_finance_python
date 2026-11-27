"""Optimiser robustness on a larger real universe (Yahoo daily adjusted closes through the platform's downloader): cluster stability of HERC and NCO, the Wasserstein ambiguity radius, and pooled versus partial-pooling
calibration, all on the development part of the history (warm-up and validation). The test and forward segments are not read: the development mode stops predicting at the end of validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.nvda_mu_allocation_study import WORK, make_app, wait                       # noqa: E402

TICKERS = ["AAPL", "MSFT", "NVDA", "MU", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "KO", "CAT", "PG"]
BASE = {"dataset": {"universe": TICKERS, "feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5", "horizon": 5, "min_history": 126, "start": "2008-01-02"},
        "split": {"min_train_days": 756, "refit_every": 126, "validation_fraction": 0.2, "test_fraction": 0.15, "forward_fraction": 0.15},
        "backtest": {"rebalance": "weekly", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": None}, "model": {"kind": "ridge"}, "explain": False,
        "signal": {"method": "proportional", "holding_period": 1}}
ALLOC = {"calibrate": "slope", "calib_inference": "hac", "calib_min_t": 1.0, "cov_method": "lw_constant_correlation", "cov_window": 252, "risk_aversion": 5.0, "limits": {"long_only": True, "cash_allowed": False, "max_weight": 0.25},
         "opt_every": 5, "on_failure": "risk_only"}


def main():
    from src.ml import alloc_adapters as ad
    from src.ml.experiments import calibration_saved, compare_saved
    from src.ml import allocator_compare as ac
    from src.ml.registry import Registry  # noqa: F401

    app = make_app()
    out = wait(app, app.extra_post["/api/ml/train"]({**BASE, "name": "robustness-12", "mode": "development"}))
    exp_id = out["id"]
    registry = app.extra_get["/api/ml/lineage"].__self__.registry
    from src.ml.experiments import load_run

    cfg, ds, train, _ = load_run(registry, exp_id)
    labels = train.plan.labels()
    dev_end = labels[labels == "validation"].index[-1]
    rets = ds.returns.loc[:dev_end]                                                   # nothing after the validation segment is read
    res = {"experiment": exp_id, "universe": TICKERS, "development_end": str(dev_end.date()), "rows": int(len(rets))}
    res["cluster_stability"] = {k: ad.cluster_stability(rets, k, window=252, step=21, n_boot=60) for k in ("herc", "nco")}
    scen = ad.__dict__  # noqa: F841
    from src.ml import portfolio_alloc as pa

    scenarios = pa.scenario_matrix(rets, len(rets) - 1, 252, 5, list(rets.columns))
    res["wasserstein_radius"] = ad.wasserstein_radius_sensitivity(scenarios, list(rets.columns), lb=0.0, ub=0.25)
    variants = {"pooled_hac": {"calib_inference": "hac"}, "regularized_hac": {"calib_method": "regularized", "calib_inference": "hac"}, "asset_hac": {"calib_method": "asset", "calib_inference": "hac"},
                "partial_pooling_hac": {"calib_method": "partial_pooling", "calib_inference": "hac"}, "partial_pooling_block": {"calib_method": "partial_pooling", "calib_inference": "block_bootstrap"}}
    res["calibration"] = calibration_saved(registry, exp_id, {"pooled_sqrt_horizon": {"calib_inference": "sqrt_horizon"}, **variants}, select_on="validation", allocation=ALLOC)
    cmp_ = compare_saved(registry, exp_id, ("equal_weight", "inverse_vol", "erc", "mean_variance", "robust_mv", "black_litterman", "herc", "nco", "near_optimal", "wasserstein_cvar"), select_on="validation",
                         allocation=ALLOC, include_existing_rule=False)
    res["allocators_validation"] = ac.public(cmp_)
    (WORK / "robustness.json").write_text(json.dumps(res, default=str, indent=1))
    print("done", exp_id)


if __name__ == "__main__":
    main()
