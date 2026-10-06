"""``quant causal``: the four estimators against simulated worlds with a known effect, beside the naive estimate that ignores the problem."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import did_2x2, dml_plr, iv_2sls, naive_ols
from .simulate import confounded_plr, did_panel, endogenous_iv


def recovery_table(n: int = 3000, seed: int = 7, theta: float = 0.5) -> pd.DataFrame:
    """One row per (world, estimator): the true effect, the estimate, its standard error and whether the 95% interval covers the truth.

    A good estimator covers the truth in the world it was built for; the naive regression should miss it wherever there is confounding. DML is only as good
    as its nuisance learner: the small trees under-fit the nonlinear controls and leave bias (Stage 37 found the same), the larger ones remove most of it.
    """
    if n < 200:
        raise ValueError("n must be at least 200")
    rng = np.random.default_rng(seed)
    rows = []

    def add(world: str, method: str, truth: float, result: dict) -> None:
        low, high = result["theta"] - 1.96 * result["se"], result["theta"] + 1.96 * result["se"]
        rows.append({"world": world, "method": method, "true_effect": truth, "estimate": result["theta"], "se": result["se"], "covers_truth": bool(low <= truth <= high)})

    w = confounded_plr(n, rng, theta)
    add("confounded (nonlinear controls)", "naive OLS", theta, naive_ols(w["y"], w["d"]))
    add("confounded (nonlinear controls)", "DML, small boosted trees", theta, dml_plr(w["y"], w["d"], w["X"], learner="gbm", seed=seed))
    add("confounded (nonlinear controls)", "DML, larger boosted trees", theta, dml_plr(w["y"], w["d"], w["X"], learner="gbm_strong", seed=seed))
    w = endogenous_iv(n, rng, theta)
    add("unobserved confounder + instrument", "naive OLS", theta, naive_ols(w["y"], w["d"]))
    add("unobserved confounder + instrument", "2SLS", theta, iv_2sls(w["y"], w["d"], w["z"]))
    w = did_panel(max(n // 2, 100), rng, effect=1.0)
    add("two groups, two periods (parallel trends)", "difference in differences", 1.0, did_2x2(w["y"], w["treated"], w["post"], w["unit"]))
    w = did_panel(max(n // 2, 100), rng, effect=1.0, trend_gap=0.5)
    add("trends NOT parallel (assumption violated)", "difference in differences", 1.0, did_2x2(w["y"], w["treated"], w["post"], w["unit"]))
    return pd.DataFrame(rows)
