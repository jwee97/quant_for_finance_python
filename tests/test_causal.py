"""Causal estimators recover planted truth where they should, and fail where the design says they must."""

from __future__ import annotations

import numpy as np

from src.causal import did_2x2, dml_from_residuals, dml_plr, iv_2sls, naive_ols, r_learner
from src.causal.simulate import confounded_plr, did_panel, endogenous_iv, heterogeneous_effect


def _mean(fn, reps=60):
    return float(np.mean([fn(np.random.default_rng(i)) for i in range(reps)]))


def test_oracle_dml_is_unbiased_while_naive_ols_is_not():
    oracle = _mean(lambda r: (lambda s: dml_from_residuals(s["y"] - s["m_y"], s["d"] - s["m_d"])["theta"])(confounded_plr(500, r)))
    naive = _mean(lambda r: (lambda s: naive_ols(s["y"], s["d"])["theta"])(confounded_plr(500, r)))
    assert abs(oracle - 0.5) < 0.03 and naive > 1.2


def test_dml_with_a_learner_that_can_fit_the_nuisance_is_unbiased():
    """On a linear-nuisance world a ridge learner can fit both functions, so DML recovers the effect where the unadjusted regression cannot."""
    def world(rng, n=600):
        X = rng.normal(size=(n, 5))
        d = X @ np.array([1.0, -0.5, 0.0, 0.3, 0.0]) + rng.normal(size=n)
        y = 0.5 * d + X @ np.array([2.0, 1.0, 0.0, -1.0, 0.5]) + rng.normal(size=n)
        return y, d, X
    dml = _mean(lambda r: dml_plr(*world(r), learner="ridge", seed=1)["theta"], reps=40)
    naive = _mean(lambda r: naive_ols(*world(r)[:2])["theta"], reps=40)
    assert abs(dml - 0.5) < 0.05 and abs(naive - 0.5) > 0.3


def test_2sls_removes_endogeneity_bias_and_reports_a_strong_first_stage():
    ols = _mean(lambda r: (lambda s: naive_ols(s["y"], s["d"])["theta"])(endogenous_iv(500, r)))
    iv = [iv_2sls(**{k: v for k, v in endogenous_iv(500, np.random.default_rng(i)).items() if k in ("y", "d", "z")}) for i in range(60)]
    assert ols > 1.0 and abs(np.mean([e["theta"] for e in iv]) - 0.5) < 0.06 and np.mean([e["first_stage_F"] for e in iv]) > 20


def test_did_is_unbiased_under_parallel_trends_and_biased_by_the_trend_gap_otherwise():
    f = lambda gap: _mean(lambda r: (lambda s: did_2x2(s["y"], s["treated"], s["post"], s["unit"])["theta"])(did_panel(250, r, trend_gap=gap)))
    assert abs(f(0.0) - 1.0) < 0.06 and abs(f(0.8) - 1.8) < 0.06


def test_r_learner_recovers_a_heterogeneous_effect():
    s = heterogeneous_effect(1500, np.random.default_rng(3))
    fit = r_learner(s["y"], s["d"], s["X"], np.column_stack([np.ones(1500), s["X"][:, 0]]), "gbm", seed=3)
    assert np.corrcoef(fit["tau"], s["tau"])[0, 1] > 0.99 and abs(fit["gamma"][1] - 1.0) < 0.2


def test_stage37_tables_and_decision_consistent():
    import json
    from pathlib import Path

    import pandas as pd
    import pytest

    root = Path(__file__).resolve().parents[1]
    path = root / "reports" / "tables" / "stage37_fomc_effects.csv"
    if not path.exists():
        pytest.skip("stage 37 not run")
    t = pd.read_csv(path, index_col=0)
    entry = next(json.loads(l) for l in (root / "experiments" / "registry.jsonl").read_text().splitlines() if '"stage37_causal"' in l)
    assert entry["decision"] == ("retain" if bool(t["bh_significant"].any()) else "reject")
    v = pd.read_csv(root / "reports" / "tables" / "stage37_validation.csv").set_index(["world", "estimator"])
    assert abs(v.loc[("endogenous_iv", "2SLS"), "bias"]) < 0.06 < v.loc[("endogenous_iv", "naive OLS (y on d)"), "bias"]
    assert abs(v.loc[("plr_oracle_posthoc", "DML final step, true nuisance"), "bias"]) < 0.03
    assert v.loc[("confounded_plr", "DML, GBM nuisance"), "bias"] > 0.3, "the declared DML validation failed; the report says so"
