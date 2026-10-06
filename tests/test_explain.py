"""The explanation methods must satisfy their own identities and recover planted structure."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.explain import integrated_gradients, permutation_importance, shapley_exact


def test_shapley_of_a_linear_model_is_coefficient_times_deviation():
    rng = np.random.default_rng(0)
    coef = np.array([2.0, -1.0, 0.0, 0.5])
    f = lambda X: X @ coef + 3.0
    bg = rng.normal(size=(30, 4))
    x = rng.normal(size=4)
    phi, base = shapley_exact(f, x, bg)
    assert np.allclose(phi, coef * (x - bg.mean(axis=0)), atol=1e-10)
    assert abs(phi.sum() - (f(x[None])[0] - base)) < 1e-10


def test_shapley_efficiency_holds_for_an_interaction_model_and_splits_the_interaction_equally():
    f = lambda X: X[:, 0] * X[:, 1]
    bg = np.zeros((1, 3))
    x = np.array([2.0, 3.0, 5.0])
    phi, base = shapley_exact(f, x, bg)
    assert base == 0.0 and abs(phi.sum() - 6.0) < 1e-12
    assert np.allclose(phi, [3.0, 3.0, 0.0])                    # the product 6 shared equally; the unused feature gets nothing


def test_integrated_gradients_completeness_and_zero_for_unused_feature():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    with torch.no_grad():
        net[0].weight[:, 3] = 0.0                                # feature 3 cannot influence the output
    wrapped = lambda z: net(z).squeeze(-1)
    class Net(torch.nn.Module):
        def forward(self, z):
            return wrapped(z)
    attribution, gap = integrated_gradients(Net(), np.array([0.5, -1.0, 2.0, 7.0]), np.zeros(4), steps=200)
    assert gap < 1e-3 and attribution[3] == 0.0


def test_permutation_importance_ranks_the_planted_feature_first_and_ignores_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5))
    y = 2.0 * X[:, 2] + 0.5 * X[:, 0] + rng.normal(size=2000) * 0.1
    f = lambda Z: 2.0 * Z[:, 2] + 0.5 * Z[:, 0]
    imp = permutation_importance(f, X, y, n_repeats=5, seed=3)
    assert imp.argmax() == 2 and imp[2] > imp[0] > 10 * max(abs(imp[1]), abs(imp[3]), abs(imp[4]))


def test_stage35_tables_agree_with_their_declared_rules():
    from pathlib import Path
    import pandas as pd

    tables = Path(__file__).resolve().parents[1] / "reports" / "tables"
    if not (tables / "stage35_calibration_tests.csv").exists():
        pytest.skip("stage 35 not run")
    checks = pd.read_csv(tables / "stage35_identity_checks.csv")
    assert (checks.loc[checks["kind"] == "shapley_efficiency", "max_gap"] < 1e-6).all()
    assert (checks.loc[checks["kind"] == "ig_completeness", "max_relative_gap"] < 1e-2).all()
    corr = pd.read_csv(tables / "stage35_importance_rank_correlation.csv", index_col=0)
    assert list(corr.index) == list(corr.columns) and (corr.to_numpy().diagonal() == 1).all()
    tests = pd.read_csv(tables / "stage35_calibration_tests.csv")
    declared = tests[tests["status"] == "declared"].iloc[0]
    registry = (tables.parents[1] / "experiments" / "registry.jsonl").read_text().splitlines()
    import json
    entry = next(json.loads(l) for l in registry if '"stage35_explain"' in l and "Isotonic recalibration" in l)
    assert entry["decision"] == ("retain" if (declared["p_value"] <= 0.10 and declared["mean_log_loss_difference"] < 0) else "reject")
