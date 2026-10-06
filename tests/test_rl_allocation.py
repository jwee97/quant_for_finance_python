import numpy as np

from src.models.rl_allocation import evolution_strategies, softmax_weights, utility


def test_zero_parameters_are_equal_weight_and_masked_assets_get_nothing():
    X = np.random.default_rng(0).normal(size=(5, 4, 3))
    w = softmax_weights(X, np.zeros(3), np.zeros(4))
    assert np.allclose(w, 0.25)
    mask = np.ones((5, 4), dtype=bool)
    mask[:, 3] = False
    wm = softmax_weights(X, np.ones(3), np.zeros(4), mask)
    assert np.allclose(wm.sum(axis=1), 1) and (wm[:, 3] == 0).all()


def test_es_learns_a_planted_signal_and_the_learned_tilt_has_higher_utility():
    rng = np.random.default_rng(1)
    T, A, F = 240, 6, 3
    X = rng.normal(size=(T, A, F))
    R = 0.02 * X[:, :, 0] + rng.normal(0, 0.03, size=(T, A))               # feature 0 predicts the next return
    fit = lambda p: utility(p, X[:160], R[:160], F, 5.0, 0.001)
    params, history = evolution_strategies(fit, F + A, seed=3, iterations=60)
    assert params[0] > 0.2 and abs(params[1]) < params[0] and abs(params[2]) < params[0]
    assert utility(params, X[160:], R[160:], F, 5.0, 0.001) > utility(np.zeros(F + A), X[160:], R[160:], F, 5.0, 0.001)
    assert history[-1] > history[0]


def test_costs_lower_utility_of_a_flipping_policy():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4, 2))
    R = rng.normal(0, 0.02, size=(60, 4))
    p = np.concatenate([[3.0, 3.0], np.zeros(4)])
    assert utility(p, X, R, 2, 5.0, 0.01) < utility(p, X, R, 2, 5.0, 0.0)


def test_stage38_decision_follows_the_declared_rule():
    import json
    from pathlib import Path

    import pandas as pd
    import pytest

    root = Path(__file__).resolve().parents[1]
    path = root / "reports" / "tables" / "stage38_tests.csv"
    if not path.exists():
        pytest.skip("stage 38 not run")
    t = pd.read_csv(path).iloc[0]
    entry = next(json.loads(l) for l in (root / "experiments" / "registry.jsonl").read_text().splitlines() if '"stage38_rl"' in l)
    assert entry["decision"] == ("retain" if (t["p_value"] <= 0.10 and t["difference"] > 0) else "reject")
    gap = pd.read_csv(root / "reports" / "tables" / "stage38_overfitting_gap.csv")
    assert len(gap) > 0 and gap["train_gain"].mean() > gap["oos_gain"].mean(), "the report says the training gain does not survive"
